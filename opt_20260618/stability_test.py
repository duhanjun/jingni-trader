"""
多源稳健性测试

【借鉴来源】
- Moon Dev AI (moondevonyt/moon-dev-ai-agents): RBI Agent
  * 同一策略在多个数据源/时间窗/参数下并行回测
  * 用稳定性筛选候选策略
- Qlib (microsoft/qlib): benchmark workflows 在不同市场 (CSI300/CSI500/CSI100)
- Zipline / QuantConnect: 多市场、多时间框架测试

【问题背景】
原 jingni-trader 的回测流程：
1. 只在单一数据集、单一时间窗上回测
2. 没有稳健性测试：参数微调、噪声注入、子样本测试
3. 容易过拟合到特定时间段或股票集合

【设计目标】
1. 提供 StabilityTester 类，可在多个数据切片/参数下并行回测同一策略
2. 输出稳定性指标（mean IC、IC std、最大回撤分布等）
3. 借鉴 Moon Dev AI 的"多源验证"思想：策略要稳定通过 5+ 个测试才算可靠

【关键功能】
- 时间窗口切片测试
- 参数敏感性扫描
- 股票池抽样稳健性
- 多数据源（A 股 vs 港股 vs ETF）回测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StabilityResult:
    """单次回测的结果"""
    name: str
    metrics: Dict[str, float]
    extra: Dict[str, Any] = field(default_factory=dict)


class StabilityTester:
    """
    多源稳健性测试器

    核心思想：在多种数据切片/参数下运行同一策略，看结果是否稳定。

    使用:
        tester = StabilityTester(strategy=my_strategy, backtest_engine=engine)
        tester.add_time_window_test(start='2020-01-01', end='2024-01-01', window_months=12)
        tester.add_param_sweep_test({'topk': [10, 20, 30, 50]})
        tester.add_universe_sample_test(n_samples=5, frac=0.5)
        results = tester.run(data, factors)
        print(tester.summarize(results))
    """

    def __init__(
        self,
        backtest_fn: Callable,
    ):
        """
        参数:
            backtest_fn: 回测函数签名 backtest_fn(data, factors) -> dict[metrics]
        """
        self.backtest_fn = backtest_fn
        self.tests: List[Dict[str, Any]] = []

    def add_time_window_test(
        self,
        start: str,
        end: str,
        window_months: int = 12,
        step_months: int = 3,
    ) -> "StabilityTester":
        """添加时间窗口切片测试"""
        self.tests.append({
            "type": "time_window",
            "start": start,
            "end": end,
            "window_months": window_months,
            "step_months": step_months,
        })
        return self

    def add_param_sweep_test(
        self,
        param_grid: Dict[str, List[Any]],
    ) -> "StabilityTester":
        """添加参数扫描测试"""
        self.tests.append({
            "type": "param_sweep",
            "param_grid": param_grid,
        })
        return self

    def add_universe_sample_test(
        self,
        n_samples: int = 5,
        sample_frac: float = 0.7,
        random_state: int = 42,
    ) -> "StabilityTester":
        """添加股票池抽样测试"""
        self.tests.append({
            "type": "universe_sample",
            "n_samples": n_samples,
            "sample_frac": sample_frac,
            "random_state": random_state,
        })
        return self

    def add_bootstrap_test(
        self,
        n_bootstrap: int = 20,
        block_size: int = 20,
        random_state: int = 42,
    ) -> "StabilityTester":
        """添加 Bootstrap 稳健性测试（块状 bootstrap）"""
        self.tests.append({
            "type": "bootstrap",
            "n_bootstrap": n_bootstrap,
            "block_size": block_size,
            "random_state": random_state,
        })
        return self

    def run(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
    ) -> List[StabilityResult]:
        """运行所有测试"""
        results: List[StabilityResult] = []
        for test in self.tests:
            if test["type"] == "time_window":
                results.extend(self._run_time_window(data, factors, test))
            elif test["type"] == "param_sweep":
                results.extend(self._run_param_sweep(data, factors, test))
            elif test["type"] == "universe_sample":
                results.extend(self._run_universe_sample(data, factors, test))
            elif test["type"] == "bootstrap":
                results.extend(self._run_bootstrap(data, factors, test))
        return results

    # ── 各测试实现 ─────────────────────────────────────
    def _run_time_window(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame],
        test: Dict[str, Any],
    ) -> List[StabilityResult]:
        results = []
        dates = pd.to_datetime(data["date"])
        start = pd.to_datetime(test["start"])
        end = pd.to_datetime(test["end"])
        window_days = int(test["window_months"] * 30)
        step_days = int(test["step_months"] * 30)

        cur = start
        idx = 0
        while cur + pd.Timedelta(days=window_days) <= end:
            window_end = cur + pd.Timedelta(days=window_days)
            mask = (dates >= cur) & (dates <= window_end)
            sub_data = data[mask].copy()
            sub_factors = factors[mask] if factors is not None else None
            if len(sub_data) < 60:
                cur += pd.Timedelta(days=step_days)
                idx += 1
                continue
            try:
                r = self.backtest_fn(sub_data, sub_factors)
                results.append(StabilityResult(
                    name=f"time_window_{idx}_{cur.date()}_{window_end.date()}",
                    metrics=r.get("metrics", {}),
                    extra={"window": f"{cur.date()}~{window_end.date()}"},
                ))
            except Exception as e:
                results.append(StabilityResult(
                    name=f"time_window_{idx}_ERROR",
                    metrics={},
                    extra={"error": str(e)},
                ))
            cur += pd.Timedelta(days=step_days)
            idx += 1
        return results

    def _run_param_sweep(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame],
        test: Dict[str, Any],
    ) -> List[StabilityResult]:
        """参数扫描：穷举 param_grid 的笛卡尔积"""
        from itertools import product
        keys = list(test["param_grid"].keys())
        values = [test["param_grid"][k] for k in keys]
        results = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            try:
                # 注入参数：假设 backtest_fn 接受第三个参数 params
                r = self.backtest_fn(data, factors, param_dict)
                name = "param_" + "_".join(f"{k}={v}" for k, v in param_dict.items())
                results.append(StabilityResult(
                    name=name,
                    metrics=r.get("metrics", {}),
                    extra={"params": param_dict},
                ))
            except Exception as e:
                results.append(StabilityResult(
                    name=f"param_{combo}_ERROR",
                    metrics={},
                    extra={"error": str(e), "params": dict(zip(keys, combo))},
                ))
        return results

    def _run_universe_sample(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame],
        test: Dict[str, Any],
    ) -> List[StabilityResult]:
        """股票池抽样"""
        rng = np.random.default_rng(test["random_state"])
        all_codes = data["code"].unique()
        n_codes = max(int(len(all_codes) * test["sample_frac"]), 10)
        results = []
        for i in range(test["n_samples"]):
            sampled = rng.choice(all_codes, size=n_codes, replace=False)
            sub_data = data[data["code"].isin(sampled)].copy()
            sub_factors = factors[factors["code"].isin(sampled)].copy() if factors is not None else None
            try:
                r = self.backtest_fn(sub_data, sub_factors)
                results.append(StabilityResult(
                    name=f"universe_sample_{i}",
                    metrics=r.get("metrics", {}),
                    extra={"n_codes": n_codes},
                ))
            except Exception as e:
                results.append(StabilityResult(
                    name=f"universe_sample_{i}_ERROR",
                    metrics={},
                    extra={"error": str(e)},
                ))
        return results

    def _run_bootstrap(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame],
        test: Dict[str, Any],
    ) -> List[StabilityResult]:
        """
        块状 Bootstrap：对时间序列用 block_size 长度的连续块拼接
        借鉴 Politis & Romano (1994) 块状 bootstrap 思想
        """
        rng = np.random.default_rng(test["random_state"])
        unique_dates = pd.to_datetime(data["date"]).sort_values().unique()
        n = len(unique_dates)
        block_size = test["block_size"]
        n_blocks = n // block_size

        results = []
        for i in range(test["n_bootstrap"]):
            # 随机抽取 n_blocks 个块
            block_starts = rng.integers(0, max(n - block_size, 1), size=n_blocks)
            sampled_dates = np.concatenate([
                unique_dates[s: s + block_size] for s in block_starts
            ])
            sampled_dates = pd.to_datetime(sampled_dates).unique()
            sampled_dates = pd.Series(sampled_dates)
            mask = pd.to_datetime(data["date"]).isin(sampled_dates)
            sub_data = data[mask].copy()
            sub_factors = factors[mask].copy() if factors is not None else None
            try:
                r = self.backtest_fn(sub_data, sub_factors)
                results.append(StabilityResult(
                    name=f"bootstrap_{i}",
                    metrics=r.get("metrics", {}),
                    extra={"n_days": len(sampled_dates)},
                ))
            except Exception as e:
                results.append(StabilityResult(
                    name=f"bootstrap_{i}_ERROR",
                    metrics={},
                    extra={"error": str(e)},
                ))
        return results

    # ── 汇总 ──────────────────────────────────────────
    def summarize(self, results: List[StabilityResult]) -> pd.DataFrame:
        """
        汇总所有测试结果，输出稳定性指标

        返回 DataFrame，列为: name, annual_return, sharpe, max_drawdown, ...
        并附带 mean/std/median 等统计
        """
        rows = []
        for r in results:
            row = {"name": r.name}
            row.update(r.metrics)
            if "error" in r.extra:
                row["error"] = r.extra["error"]
            rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def stability_score(self, results: List[StabilityResult]) -> Dict[str, float]:
        """
        计算稳定性评分。

        指标：
        - sharpe_consistency: sharpe 在多次测试中保持正值的比例
        - drawdown_range: 最大回撤的极差（越小越稳定）
        - return_dispersion: 年化收益的标准差 / 均值（变异系数）
        - 总体稳定性 = sharpe_consistency * 0.5 + (1 - drawdown_range/0.5) * 0.3 + (1 - return_dispersion) * 0.2
        """
        valid = [r for r in results if r.metrics.get("sharpe_ratio") is not None]
        if not valid:
            return {"stability_score": 0.0, "note": "无有效结果"}

        sharpes = [r.metrics["sharpe_ratio"] for r in valid]
        drawdowns = [r.metrics.get("max_drawdown", 0) for r in valid]
        returns = [r.metrics.get("annual_return", 0) for r in valid]

        sharpe_consistency = sum(1 for s in sharpes if s > 0) / len(sharpes)
        drawdown_range = max(drawdowns) - min(drawdowns)
        return_dispersion = (
            float(np.std(returns) / abs(np.mean(returns)))
            if np.mean(returns) != 0 else float("inf")
        )

        score = (
            0.5 * sharpe_consistency
            + 0.3 * max(0, 1 - drawdown_range / 0.5)
            + 0.2 * max(0, 1 - min(return_dispersion, 1.0))
        )

        return {
            "stability_score": float(score),
            "sharpe_consistency": float(sharpe_consistency),
            "drawdown_range": float(drawdown_range),
            "return_dispersion": float(return_dispersion),
            "n_valid_tests": len(valid),
            "n_total_tests": len(results),
        }
