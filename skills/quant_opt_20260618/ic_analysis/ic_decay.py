"""
IC Decay Analysis 工具集
========================

借鉴来源
--------
- Microsoft Qlib: qlib/contrib/evaluate.py 中的 risk_analysis 与
  qlib/contrib/report/analysis_position 中的 IC Decay 概念
- RD-Agent: 多周期 IC 联合评估因子有效性

核心思想
--------
Qlib 的因子评估不仅计算单期 IC，还会沿着多个 forward period
(1d/3d/5d/10d/20d/40d) 计算 IC 序列，得到 IC Decay 曲线，
从而回答两个核心问题：

  1. 因子的预测能力在哪一期最强？(peak lag)
  2. 因子的"半衰期"是多久？ (IC 衰减到峰值 50% 的 lag)

jingni-trader 现状
------------------
`skills/factor-engine/engine.py` 的 `ic_analysis` 只评估
1d/5d/20d 三个固定周期，且没有：
  - 任意 lag 扫描
  - 衰减曲线拟合
  - 最优持有期识别
  - 与基准的相对衰减

本模块提供：
  - 任意 lag 区间的 IC 扫描
  - IC 衰减曲线与最优 lag 识别
  - 半衰期 (half-life) 估计
  - 与"常数零假设"的 t 检验
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class ICLagResult:
    """单个 lag 的 IC 统计"""
    lag: int
    ic_mean: float
    ic_std: float
    ic_ir: float            # ic_mean / ic_std
    ic_t_stat: float        # t = mean / (std / sqrt(n))
    ic_p_value: float       # 双侧 t 检验
    ic_pos_ratio: float     # IC > 0 的比例
    n_obs: int              # 有效截面数

    def to_dict(self) -> dict:
        return asdict(self)


class ICDecayAnalyzer:
    """
    IC Decay 分析器
    ----------------
    在 [min_lag, max_lag] 范围内，对每个整数 lag 计算横截面 Spearman IC，
    拼接成 IC Decay 曲线，并识别最优 lag 与半衰期。

    借鉴自 Qlib 的 `qlib.contrib.evaluate` 与
    vnpy.alpha 中 `AlphaLab.ic_decay_analysis`。
    """

    def __init__(
        self,
        min_lag: int = 1,
        max_lag: int = 20,
        min_cross_size: int = 30,
    ):
        if min_lag < 1:
            raise ValueError("min_lag must be >= 1")
        if max_lag < min_lag:
            raise ValueError("max_lag must be >= min_lag")
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.min_cross_size = min_cross_size

    def _forward_returns(self, data: pd.DataFrame, lag: int) -> pd.DataFrame:
        """计算 lag 期 forward return:
            ret_{t->t+lag} = close.shift(-lag) / close - 1
        """
        out = data[["code", "date", "close"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values(["code", "date"])
        out[f"fwd_{lag}"] = out.groupby("code")["close"].transform(
            lambda x: x.shift(-lag) / x - 1.0
        )
        return out[["code", "date", f"fwd_{lag}"]]

    def _cross_section_ic(
        self,
        factor: pd.Series,
        fwd_ret: pd.Series,
    ) -> Optional[float]:
        """横截面 Spearman IC"""
        df = pd.concat([factor, fwd_ret], axis=1).dropna()
        if len(df) < self.min_cross_size:
            return None
        if df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
            return None
        ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
        return float(ic) if not np.isnan(ic) else None

    def calc_ic_decay(
        self,
        data: pd.DataFrame,
        factor_col: str,
    ) -> List[ICLagResult]:
        """
        扫描 [min_lag, max_lag] 区间内每个 lag 的 IC 统计量

        参数
        ----
        data: 包含 code/date/close/factor_col 的 DataFrame
        factor_col: 因子列名

        返回
        ----
        按 lag 升序的 ICLagResult 列表
        """
        if factor_col not in data.columns:
            raise ValueError(f"factor_col {factor_col} not in data")
        df = data[["code", "date", "close", factor_col]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=[factor_col, "close"])
        if df.empty:
            return []

        results: List[ICLagResult] = []
        for lag in range(self.min_lag, self.max_lag + 1):
            fwd = self._forward_returns(df, lag)
            merged = df.merge(fwd, on=["code", "date"], how="inner")
            merged = merged.dropna(subset=[factor_col, f"fwd_{lag}"])

            # 逐日计算横截面 IC
            ic_series: List[float] = []
            for _, grp in merged.groupby("date"):
                ic = self._cross_section_ic(
                    grp[factor_col], grp[f"fwd_{lag}"]
                )
                if ic is not None:
                    ic_series.append(ic)

            if len(ic_series) < 5:
                # 太少有效截面，丢弃
                continue

            arr = np.asarray(ic_series, dtype=float)
            n = len(arr)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            ir = mean / std if std > 0 else 0.0
            t_stat = (
                mean / (std / np.sqrt(n)) if std > 0 and n > 1 else 0.0
            )
            p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))) if n > 1 else 1.0
            pos_ratio = float((arr > 0).mean())

            results.append(
                ICLagResult(
                    lag=lag,
                    ic_mean=mean,
                    ic_std=std,
                    ic_ir=ir,
                    ic_t_stat=float(t_stat),
                    ic_p_value=p_val,
                    ic_pos_ratio=pos_ratio,
                    n_obs=n,
                )
            )

        return results

    def find_optimal_lag(self, results: List[ICLagResult]) -> Optional[int]:
        """IC 绝对值最大的 lag 即为最优持有期 (peak lag)"""
        if not results:
            return None
        return max(results, key=lambda r: abs(r.ic_mean)).lag

    def estimate_half_life(
        self,
        results: List[ICLagResult],
    ) -> Optional[int]:
        """
        估算 IC 衰减半衰期 (half-life)
        定义为 IC 绝对值从峰值衰减到 50% 所需的 lag 增量。
        若始终未衰减到 50% 以下则返回 None。
        """
        if not results:
            return None
        sorted_res = sorted(results, key=lambda r: r.lag)
        peak_abs = max(abs(r.ic_mean) for r in sorted_res)
        if peak_abs == 0:
            return None
        peak_lag = max(sorted_res, key=lambda r: abs(r.ic_mean)).lag
        for r in sorted_res:
            if r.lag <= peak_lag:
                continue
            if abs(r.ic_mean) <= 0.5 * peak_abs:
                return r.lag
        return None

    def summarize(
        self,
        data: pd.DataFrame,
        factor_col: str,
    ) -> Dict:
        """一次性返回 IC Decay 完整报告"""
        decay = self.calc_ic_decay(data, factor_col)
        optimal_lag = self.find_optimal_lag(decay)
        half_life = self.estimate_half_life(decay)
        return {
            "factor": factor_col,
            "n_lags": len(decay),
            "results": [r.to_dict() for r in decay],
            "optimal_lag": optimal_lag,
            "half_life_lag": half_life,
            "peak_abs_ic": (
                max(abs(r.ic_mean) for r in decay) if decay else None
            ),
        }
