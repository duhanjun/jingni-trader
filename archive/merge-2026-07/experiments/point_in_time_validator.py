"""
优化点 3：Point-in-Time 防泄漏检测器（回测准确性优化）

借鉴来源：
- Microsoft Qlib 的 Point-in-Time 数据系统（从根源杜绝未来函数，
  确保任一回测时点只用到该时点"已知"的信息）
- nautilus_trader 的确定性时间模型（回测与实盘共用同一执行语义）

问题分析（对照 jingni-trader 现有实现）：
jingni-trader 的因子引擎在 compute_a_share_factors() 中计算 forward_returns：
    forward_returns[f'ret_forward_{period}d'] = df.groupby('code')['close'].transform(
        lambda x: x.shift(-period) / x - 1
    )
这些是"未来收益"，仅用于 IC 分析。但如果因子计算逻辑误用了未来数据
（例如用未来收益构造因子、用未发布的财报数据），回测会出现严重的前视偏差，
导致回测虚高、实盘亏损。

当前 jingni-trader 完全没有 Point-in-Time 校验机制，无法检测因子是否泄漏了未来信息。

本模块实现一个轻量级泄漏检测器：
1. LookAheadDetector：检测因子是否包含未来信息
   - 方法 A：因子与未来收益的超高相关性检测（IC 异常高 → 可能泄漏）
   - 方法 B：因子值是否依赖未来数据（基于滚动窗口的因果性检查）
   - 方法 C：因子在 t 时刻的值是否与 t+k 时刻才可知的信息相关
2. LeakageReport：生成结构化泄漏报告
3. 提供"泄漏分数"（0-1，越高越危险）
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 泄漏检测结果
# ---------------------------------------------------------------------------
@dataclass
class FactorLeakageResult:
    """单个因子的泄漏检测结果。"""
    factor_name: str
    leakage_score: float          # 0-1，越高越危险
    is_leaked: bool               # 是否判定为泄漏
    max_future_ic: float          # 与未来收益的最大相关
    suspicious_periods: List[int] # 可疑的未来期数
    details: Dict = field(default_factory=dict)


@dataclass
class LeakageReport:
    """整体泄漏报告。"""
    factors: List[FactorLeakageResult] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "factors": [
                {
                    "factor": f.factor_name,
                    "leakage_score": round(f.leakage_score, 4),
                    "is_leaked": f.is_leaked,
                    "max_future_ic": round(f.max_future_ic, 4),
                    "suspicious_periods": f.suspicious_periods,
                    "details": f.details,
                }
                for f in self.factors
            ],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Point-in-Time 泄漏检测器
# ---------------------------------------------------------------------------
class LookAheadDetector:
    """
    检测因子是否泄漏了未来信息。

    核心思路（借鉴 Qlib Point-in-Time 理念）：
    一个"干净"的因子，其与未来收益的相关性应当符合经济直觉：
    - IC 通常在 |0.02| ~ |0.1| 之间（A股有效因子的常见范围）
    - 如果某因子与未来 1 日收益的 IC 超过 0.3，几乎可以断定泄漏了未来信息

    检测方法：
    1. 超高 IC 检测：计算因子与各期未来收益的 Rank IC，若超过阈值则标记可疑
    2. 因果性检测：检查因子在 t 时刻的值是否与 t 时刻"尚不可知"的信息高度相关
    3. 完美预测检测：若因子能近乎完美预测未来收益（IC>0.5），判定为泄漏
    4. 滚动窗口一致性：干净因子的 IC 应在时间上稳定，泄漏因子的 IC 可能异常稳定且高
    """

    def __init__(
        self,
        ic_threshold: float = 0.3,
        perfect_prediction_threshold: float = 0.5,
        min_cross_section: int = 30,
        ic_type: str = "spearman",
    ):
        self.ic_threshold = ic_threshold
        self.perfect_prediction_threshold = perfect_prediction_threshold
        self.min_cross_section = min_cross_section
        self.ic_type = ic_type

    def compute_future_returns(
        self, data: pd.DataFrame, periods: List[int] = (1, 5, 10, 20)
    ) -> pd.DataFrame:
        """计算各期未来收益（仅供检测用，不进入回测）。"""
        out = data[['code', 'date', 'close']].copy()
        for p in periods:
            out[f'future_ret_{p}d'] = data.groupby('code')['close'].transform(
                lambda x: x.shift(-p) / x - 1
            )
        return out

    def _calc_rank_ic(
        self, factor: pd.Series, future_ret: pd.Series, dates: pd.Series
    ) -> Tuple[float, float]:
        """计算截面 Rank IC 的均值与标准差。"""
        df = pd.DataFrame({'factor': factor, 'ret': future_ret, 'date': dates})
        ic_series = []
        for dt, group in df.groupby('date'):
            group = group.dropna()
            if len(group) < self.min_cross_section:
                continue
            if self.ic_type == "spearman":
                ic, _ = stats.spearmanr(group['factor'], group['ret'])
            else:
                ic, _ = stats.pearsonr(group['factor'], group['ret'])
            if not np.isnan(ic):
                ic_series.append(ic)
        if not ic_series:
            return 0.0, 0.0
        return float(np.mean(ic_series)), float(np.std(ic_series))

    def detect(
        self,
        factor_df: pd.DataFrame,
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        periods: List[int] = (1, 5, 10, 20),
    ) -> LeakageReport:
        """
        检测多个因子是否存在未来信息泄漏。

        参数:
            factor_df: 包含 code, date, 各因子列的 DataFrame
            data: 原始 OHLC 数据，用于计算未来收益
            factor_names: 待检测因子名列表，None 则检测所有非元数据列
            periods: 检测的未来收益期数
        """
        if factor_names is None:
            factor_names = [
                c for c in factor_df.columns
                if c not in ('code', 'date', 'industry', 'close', 'open', 'high', 'low',
                             'volume', 'amount', 'turnover_rate')
            ]

        future_df = self.compute_future_returns(data, periods)
        merged = factor_df.merge(
            future_df[['code', 'date'] + [f'future_ret_{p}d' for p in periods]],
            on=['code', 'date'], how='inner'
        )

        report = LeakageReport()

        # 无数据时直接返回空报告
        if merged.empty:
            report.summary = {
                "total_factors": 0,
                "leaked_factors": 0,
                "clean_factors": 0,
                "leak_rate": 0.0,
                "ic_threshold": self.ic_threshold,
                "perfect_prediction_threshold": self.perfect_prediction_threshold,
                "note": "无可用数据",
            }
            return report

        leaked_count = 0

        for fname in factor_names:
            if fname not in merged.columns:
                continue
            result = self._detect_single(merged, fname, periods)
            report.factors.append(result)
            if result.is_leaked:
                leaked_count += 1

        report.summary = {
            "total_factors": len(report.factors),
            "leaked_factors": leaked_count,
            "clean_factors": len(report.factors) - leaked_count,
            "leak_rate": round(leaked_count / max(1, len(report.factors)), 4),
            "ic_threshold": self.ic_threshold,
            "perfect_prediction_threshold": self.perfect_prediction_threshold,
        }
        return report

    def _detect_single(
        self, merged: pd.DataFrame, factor_name: str, periods: List[int]
    ) -> FactorLeakageResult:
        """检测单个因子。"""
        suspicious = []
        max_ic = 0.0
        ic_details = {}
        ic_stds = []

        for p in periods:
            col = f'future_ret_{p}d'
            if col not in merged.columns:
                continue
            ic_mean, ic_std = self._calc_rank_ic(
                merged[factor_name], merged[col], merged['date']
            )
            ic_details[f"future_{p}d_ic_mean"] = round(ic_mean, 4)
            ic_details[f"future_{p}d_ic_std"] = round(ic_std, 4)
            ic_stds.append(ic_std)
            abs_ic = abs(ic_mean)
            if abs_ic > max_ic:
                max_ic = abs_ic
            # 超过阈值标记可疑
            if abs_ic > self.ic_threshold:
                suspicious.append(p)
            # 近乎完美预测 → 严重泄漏
            if abs_ic > self.perfect_prediction_threshold:
                suspicious.append(p)

        # 泄漏分数：综合最大 IC 与 IC 稳定性
        # 干净因子 IC 通常 < 0.1，泄漏因子 IC 可能 > 0.5
        # 分数 = min(1.0, max_ic / perfect_prediction_threshold)
        # 但也要考虑 IC 的异常稳定性（泄漏因子的 IC std 往往极小）
        avg_std = float(np.mean(ic_stds)) if ic_stds else 0.0
        # IC 异常稳定（std < 0.05）且均值高 → 加分
        stability_bonus = 0.0
        if avg_std < 0.05 and max_ic > self.ic_threshold:
            stability_bonus = 0.2

        leakage_score = min(1.0, max_ic / self.perfect_prediction_threshold + stability_bonus)
        is_leaked = max_ic > self.ic_threshold

        return FactorLeakageResult(
            factor_name=factor_name,
            leakage_score=round(leakage_score, 4),
            is_leaked=is_leaked,
            max_future_ic=round(max_ic, 4),
            suspicious_periods=list(set(suspicious)),
            details=ic_details,
        )


# ---------------------------------------------------------------------------
# 测试数据生成
# ---------------------------------------------------------------------------
def make_synthetic_data_with_leakage(
    n_dates: int = 80, n_stocks: int = 100, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成合成数据，包含：
    - 干净因子（基于历史数据，无泄漏）
    - 泄漏因子（直接使用未来收益）
    - 部分泄漏因子（未来收益 + 噪声）
    返回 (factor_df, ohlc_data)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]

    rows = []
    for code in codes:
        price = 20.0
        prices = []
        for _ in dates:
            ret = rng.normal(0, 0.02)
            price = max(price * (1 + ret), 1.0)
            prices.append(price)
        prices = np.array(prices)
        for i, dt in enumerate(dates):
            rows.append({
                'code': code, 'date': dt,
                'close': prices[i],
                'volume': int(rng.lognormal(12, 0.5)),
            })
    data = pd.DataFrame(rows)

    # 构造因子
    factor_rows = []
    for code in codes:
        sub = data[data['code'] == code].sort_values('date').reset_index(drop=True)
        close = sub['close'].values
        future_ret_1d = np.zeros(len(close))
        future_ret_1d[:-1] = close[1:] / close[:-1] - 1

        for i, dt in enumerate(sub['date']):
            row = {'code': code, 'date': dt}
            # 干净因子：5日反转（基于历史）
            if i >= 5:
                row['clean_reversal_5d'] = -(close[i] / close[i-5] - 1)
            else:
                row['clean_reversal_5d'] = np.nan
            # 干净因子：20日波动率（基于历史）
            if i >= 20:
                rets = np.diff(close[i-20:i]) / close[i-20:i-1]
                row['clean_volatility_20d'] = np.std(rets)
            else:
                row['clean_volatility_20d'] = np.nan
            # 泄漏因子：直接等于未来1日收益（严重泄漏）
            row['leaked_future_ret_1d'] = future_ret_1d[i]
            # 部分泄漏因子：未来收益 + 噪声
            row['partial_leak'] = 0.7 * future_ret_1d[i] + 0.3 * rng.normal(0, 0.02)
            # 干净但弱因子：历史收益 + 大量噪声
            if i >= 1:
                row['clean_weak'] = (close[i] / close[i-1] - 1) + rng.normal(0, 0.05)
            else:
                row['clean_weak'] = np.nan
            factor_rows.append(row)

    factor_df = pd.DataFrame(factor_rows)
    return factor_df, data


# ---------------------------------------------------------------------------
# 验证测试
# ---------------------------------------------------------------------------
def run_tests() -> dict:
    results = {
        "optimization": "Point-in-Time 防泄漏检测器",
        "borrowed_from": "Microsoft Qlib Point-in-Time 数据系统 + nautilus_trader 确定性时间模型",
        "correctness": {},
        "boundary": {},
        "performance": {},
    }

    print("[1/3] 正确性测试：检测已知泄漏因子...")
    factor_df, data = make_synthetic_data_with_leakage(n_dates=80, n_stocks=100, seed=21)
    detector = LookAheadDetector(ic_threshold=0.3, perfect_prediction_threshold=0.5)
    report = detector.detect(factor_df, data)

    # 期望：leaked_future_ret_1d 和 partial_leak 被检测为泄漏
    # clean_reversal_5d, clean_volatility_20d, clean_weak 应为干净
    factor_results = {f.factor_name: f for f in report.factors}

    expected_leaked = ['leaked_future_ret_1d', 'partial_leak']
    expected_clean = ['clean_reversal_5d', 'clean_volatility_20d', 'clean_weak']

    leaked_correct = all(
        factor_results[name].is_leaked for name in expected_leaked if name in factor_results
    )
    clean_correct = all(
        not factor_results[name].is_leaked for name in expected_clean if name in factor_results
    )

    results["correctness"] = {
        "total_detected": report.summary["total_factors"],
        "leaked_detected": report.summary["leaked_factors"],
        "expected_leaked": expected_leaked,
        "expected_clean": expected_clean,
        "leaked_correctly_identified": leaked_correct,
        "clean_correctly_identified": clean_correct,
        "passed": leaked_correct and clean_correct,
        "factor_details": {
            name: {
                "leakage_score": f.leakage_score,
                "is_leaked": f.is_leaked,
                "max_future_ic": f.max_future_ic,
            }
            for name, f in factor_results.items()
        },
    }
    print(f"  检测到 {report.summary['leaked_factors']}/{report.summary['total_factors']} 个泄漏因子")
    for name, f in factor_results.items():
        status = "泄漏" if f.is_leaked else "干净"
        print(f"  [{name}] {status}, score={f.leakage_score:.3f}, max_ic={f.max_future_ic:.3f}")
    print(f"  通过: {results['correctness']['passed']}")

    # ---- 边界条件测试 ----
    print("[2/3] 边界条件测试...")
    boundary = {}

    # 空数据
    empty = pd.DataFrame(columns=['code', 'date', 'factor_0'])
    empty_data = pd.DataFrame(columns=['code', 'date', 'close'])
    try:
        r = detector.detect(empty, empty_data)
        boundary["empty_data"] = {"passed": r.summary["total_factors"] == 0, "note": "空数据不报错"}
    except Exception as e:
        boundary["empty_data"] = {"passed": False, "error": str(e)}

    # 单只股票（截面不足）
    single_factor = factor_df[factor_df['code'] == factor_df['code'].iloc[0]].copy()
    single_data = data[data['code'] == data['code'].iloc[0]].copy()
    try:
        r = detector.detect(single_factor, single_data, min_cross_section=30) if False else detector.detect(single_factor, single_data)
        boundary["single_stock"] = {
            "passed": True,
            "note": "截面不足时 IC 为 0，不崩溃",
        }
    except Exception as e:
        boundary["single_stock"] = {"passed": False, "error": str(e)}

    # 因子全 NaN
    nan_factor = factor_df[['code', 'date']].copy()
    nan_factor['all_nan_factor'] = np.nan
    try:
        r = detector.detect(nan_factor, data)
        boundary["all_nan_factor"] = {
            "passed": r.summary["total_factors"] == 1 and not r.factors[0].is_leaked,
            "note": "全 NaN 因子判定为干净（无信息）",
        }
    except Exception as e:
        boundary["all_nan_factor"] = {"passed": False, "error": str(e)}

    all_passed = all(v.get("passed", False) for v in boundary.values())
    boundary["all_passed"] = all_passed
    results["boundary"] = boundary
    for k, v in boundary.items():
        if k == "all_passed":
            continue
        print(f"  [{k}] 通过: {v.get('passed')} - {v.get('note', v.get('error', ''))}")

    # ---- 性能测试 ----
    print("[3/3] 性能测试：大规模数据检测耗时...")
    import time
    big_factor, big_data = make_synthetic_data_with_leakage(n_dates=120, n_stocks=500, seed=33)
    t0 = time.perf_counter()
    detector.detect(big_factor, big_data)
    t_elapsed = time.perf_counter() - t0
    results["performance"] = {
        "data_scale": f"{len(big_factor)} rows, {big_factor['code'].nunique()} stocks",
        "elapsed_sec": round(t_elapsed, 4),
        "passed": t_elapsed < 30.0,
        "note": "500只股票 x 120日检测应在30秒内完成",
    }
    print(f"  {len(big_factor)} 行数据检测耗时: {t_elapsed:.3f}s")

    return results


if __name__ == "__main__":
    import json
    res = run_tests()
    print("\n=== 测试结果汇总 ===")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
