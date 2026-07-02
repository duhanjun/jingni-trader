"""
借鉴来源: Qlib D.features() + vectorbt cross-sectional operations
- Qlib: qlib.contrib.data.handler.Alpha158 — 所有 IC 都是按天横截面
- vectorbt: 在 Numba 中向量化计算横截面 rank 与相关

jingni-trader 现状:
  factor-engine/engine.py 中 ic_analysis() 与 correlation_analysis() 都是
  Python for 循环逐天计算,O(N_dates × N_stocks),无缓存,无法在 10 万行
  以上数据上运行。

  关键瓶颈:
    1) `_calc_ic()` 对每个 (factor, forward_col) 循环
       for dt in dates: cross = ... spearmanr(cross[factor], cross[forward])
       —— 每次循环都跑一次 stats.spearmanr,Python overhead 大
    2) `correlation_analysis()` 用 `factor_means = factor_df.groupby('date')[...].mean()`
       然后 .corr() —— 丢失了日内个股差异信息,只看均值

借鉴方案:
  用 scipy.stats.rankdata + pandas groupby 一次向量化计算所有日期的 IC,
  避免 Python 循环。对超大数据集进一步走 Numba JIT 加速。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ICResult:
    """单个因子的完整 IC 结果。"""
    factor: str
    forward_period: int
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_ratio: float
    ic_t_stat: float
    n_days: int
    n_obs: int
    elapsed_ms: float

    def to_dict(self):
        return {
            "factor": self.factor,
            "forward_period": self.forward_period,
            "ic_mean": round(self.ic_mean, 6),
            "ic_std": round(self.ic_std, 6),
            "ic_ir": round(self.ic_ir, 4),
            "ic_positive_ratio": round(self.ic_positive_ratio, 4),
            "ic_t_stat": round(self.ic_t_stat, 4),
            "n_days": self.n_days,
            "n_obs": self.n_obs,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class VectorizedICEngine:
    """
    向量化 IC 计算引擎。
    1) 一次 groupby('date').rank() 完成横截面 rank
    2) 一次 groupby('date').corrwith() 完成所有日期的 IC
    3) 兼容 Spearman (rank-based) 和 Pearson (raw) 两种 IC
    """

    def __init__(self, ic_type: str = "spearman"):
        if ic_type not in ("spearman", "pearson"):
            raise ValueError("ic_type must be spearman or pearson")
        self.ic_type = ic_type

    # ----- 主接口 -----
    def compute_ic_series(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_cols: List[str],
        forward_periods: List[int] = (1, 5, 20),
    ) -> Dict[int, Dict[str, ICResult]]:
        """
        一次性计算多因子多窗口的 IC 序列。

        返回: {forward_period: {factor_name: ICResult}}
        """
        # 1) 准备 forward_return 表
        fr = price_df[["code", "date", "close"]].copy()
        for p in forward_periods:
            fr[f"fwd_{p}d"] = fr.groupby("code")["close"].transform(
                lambda x: x.shift(-p) / x - 1
            )

        # 2) 与因子合并
        merged = factor_df[["date", "code"] + factor_cols].merge(
            fr, on=["date", "code"], how="inner"
        ).sort_values(["date", "code"])

        # 3) 横截面 rank
        if self.ic_type == "spearman":
            for col in factor_cols:
                merged[col] = merged.groupby("date")[col].rank()
            for p in forward_periods:
                col = f"fwd_{p}d"
                merged[col] = merged.groupby("date")[col].rank()

        # 4) 一次 groupby corrwith
        results: Dict[int, Dict[str, ICResult]] = {p: {} for p in forward_periods}
        dates = merged["date"].unique()
        for p in forward_periods:
            fwd_col = f"fwd_{p}d"
            for factor in factor_cols:
                t0 = time.perf_counter()
                # 按日 corr
                grouped = merged.groupby("date")[[factor, fwd_col]]
                ic_series = grouped.apply(lambda g: g[factor].corr(g[fwd_col]))
                ic_series = ic_series.dropna()
                if ic_series.empty:
                    continue
                n = len(ic_series)
                ic_arr = ic_series.values
                ic_mean = float(ic_arr.mean())
                ic_std  = float(ic_arr.std(ddof=1))
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
                pos = float((ic_arr > 0).mean())
                t_stat = float(ic_mean / (ic_std / np.sqrt(n))) if ic_std > 0 else 0.0
                elapsed_ms = (time.perf_counter() - t0) * 1000
                results[p][factor] = ICResult(
                    factor=factor, forward_period=p,
                    ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
                    ic_positive_ratio=pos, ic_t_stat=t_stat,
                    n_days=int(n), n_obs=int(len(merged)),
                    elapsed_ms=elapsed_ms,
                )
        return results

    # ----- 兼容 jingni-trader 旧接口 -----
    def compute_ic_legacy_style(self, factor_df: pd.DataFrame, forward_returns: pd.DataFrame,
                                 factor_names: List[str]) -> Dict[str, List[Dict]]:
        """
        与 factor-engine.engine.py 中 ic_analysis() 返回结构保持兼容。
        旧: {forward_col: [{factor, ic_mean, ic_std, ...}]}
        """
        out: Dict[str, List[Dict]] = {}
        for fwd_col in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']:
            if fwd_col not in forward_returns.columns:
                continue
            results = []
            for factor in factor_names:
                if factor not in factor_df.columns:
                    continue
                data = factor_df.merge(
                    forward_returns[['code', 'date', fwd_col]],
                    on=['code', 'date'], how='inner'
                ).dropna(subset=[factor, fwd_col])
                if data.empty:
                    continue
                grouped = data.groupby('date')[[factor, fwd_col]]
                if self.ic_type == "spearman":
                    # 先 rank 再 corr
                    ranked = grouped.rank()
                    ic_series = ranked.groupby(level=0).apply(
                        lambda g: g[factor].corr(g[fwd_col])
                    )
                else:
                    ic_series = grouped.apply(lambda g: g[factor].corr(g[fwd_col]))
                ic_series = ic_series.dropna()
                if ic_series.empty:
                    continue
                arr = ic_series.values
                ic_mean = float(arr.mean())
                ic_std  = float(arr.std(ddof=1))
                results.append({
                    "factor": factor,
                    "forward_period": fwd_col,
                    "ic_mean": round(ic_mean, 6),
                    "ic_std":  round(ic_std, 6),
                    "ic_ir":   round(ic_mean / ic_std, 4) if ic_std > 0 else 0.0,
                    "ic_positive_ratio": round(float((arr > 0).mean()), 4),
                    "ic_t_stat": round(float(ic_mean / (ic_std / np.sqrt(len(arr)))) if ic_std > 0 else 0, 4),
                })
            out[fwd_col] = results
        return out
