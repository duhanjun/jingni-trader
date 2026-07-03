"""
Quantile Return Analysis
========================

借鉴来源
--------
- Microsoft Qlib: qlib.contrib.evaluate 中的 `long_short` 与
  `quantile` 分析模块
- vnpy.alpha.dataset.alpha_dataset: 分层回测
- 学术文献: 因子投资中 quintile portfolio 是评估因子单调性的标准方法

核心思想
--------
将每日股票按因子值排序分成 N 个分位 (quantile)，
然后计算每个分位的"等权日均收益"，再画出分位收益曲线：

    Q1 (low)  Q2  Q3  Q4  Q5 (high)
     |        |   |   |   |
     v        v   v   v   v
    ret_1   ret_2 ret_3 ret_4 ret_5

判读规则
--------
1. 单调性 (Monotonicity): 收益是否随分位单调变化
2. 多空收益 (Long-Short Spread): Q5 - Q1 的日均收益
3. 多空夏普: Q5-Q1 的日收益序列的夏普比率
4. 换手率 (Turnover): 跨日分位变化比例

jingni-trader 现状
------------------
- factor-engine 没有分位评估
- backtest-engine 直接以 alpha_score 排名前 20% 作为多头，
  没有评估因子的分位单调性
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class QuantileStats:
    """单分位统计"""
    quantile: int
    mean_daily_return: float
    std_daily_return: float
    sharpe: float
    cumulative_return: float

    def to_dict(self) -> dict:
        return asdict(self)


class QuantileAnalyzer:
    """
    Quantile Return Analyzer
    -------------------------
    在每个交易日将股票按因子值分成 N 个分位，
    计算每个分位的等权组合日收益。
    """

    def __init__(
        self,
        n_quantiles: int = 5,
        min_stocks_per_quantile: int = 5,
        forward_lag: int = 1,
    ):
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        self.n_quantiles = n_quantiles
        self.min_stocks_per_quantile = min_stocks_per_quantile
        self.forward_lag = forward_lag

    def _assign_quantile(self, factor: pd.Series) -> pd.Series:
        """
        使用 qcut 分桶，duplicates='drop' 防止重复边界
        """
        ranked = pd.qcut(
            factor.rank(method="first"),
            q=self.n_quantiles,
            labels=False,
            duplicates="drop",
        ) + 1  # 1-indexed
        return ranked.astype("Int64")

    def compute_quantile_returns(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
        factor_col: str,
    ) -> pd.DataFrame:
        """
        返回 DataFrame: index=date, columns=[q1, q2, ..., qN, long_short]
        """
        fwd_col = f"fwd_ret_{self.forward_lag}d"
        if fwd_col not in forward_returns.columns:
            raise ValueError(f"{fwd_col} not in forward_returns")

        merged = factor_df[["code", "date", factor_col]].merge(
            forward_returns[["code", "date", fwd_col]],
            on=["code", "date"],
            how="inner",
        ).dropna(subset=[factor_col, fwd_col])

        if merged.empty:
            return pd.DataFrame()

        merged["quantile"] = (
            merged.groupby("date")[factor_col]
            .transform(self._assign_quantile)
        )
        merged = merged.dropna(subset=["quantile"])
        merged["quantile"] = merged["quantile"].astype(int)

        # 过滤掉分位内股票数太少的日期
        counts = merged.groupby(["date", "quantile"]).size().reset_index(name="n")
        valid_pairs = counts[counts["n"] >= self.min_stocks_per_quantile]
        merged = merged.merge(valid_pairs[["date", "quantile"]], on=["date", "quantile"])

        pivot = (
            merged.groupby(["date", "quantile"])[fwd_col]
            .mean()
            .unstack("quantile")
        )
        pivot.columns = [f"q{int(c)}" for c in pivot.columns]
        if self.n_quantiles >= 2 and f"q{self.n_quantiles}" in pivot.columns and "q1" in pivot.columns:
            pivot["long_short"] = pivot[f"q{self.n_quantiles}"] - pivot["q1"]
        return pivot.sort_index()

    def summarize(
        self,
        quantile_returns: pd.DataFrame,
        trading_days: int = 252,
    ) -> Dict:
        """计算每个分位的统计量与单调性评估"""
        if quantile_returns.empty:
            return {"stats": [], "monotonicity": None, "long_short_sharpe": None}

        stats_list: List[QuantileStats] = []
        for col in quantile_returns.columns:
            if not col.startswith("q") or col == "long_short":
                continue
            r = quantile_returns[col].dropna()
            if len(r) < 2:
                continue
            mean = float(r.mean())
            std = float(r.std(ddof=1))
            sharpe = float(mean / std * np.sqrt(trading_days)) if std > 0 else 0.0
            cum = float((1 + r).prod() - 1)
            q = int(col[1:])
            stats_list.append(QuantileStats(
                quantile=q,
                mean_daily_return=mean,
                std_daily_return=std,
                sharpe=sharpe,
                cumulative_return=cum,
            ))

        # 评估单调性: q1..qN 的 mean_daily_return 序列与 rank 的 Spearman 相关
        if len(stats_list) >= 2:
            x = [s.quantile for s in stats_list]
            y = [s.mean_daily_return for s in stats_list]
            from scipy.stats import spearmanr
            corr, pval = spearmanr(x, y)
            monotonicity = {
                "spearman_corr": float(corr),
                "p_value": float(pval),
                "is_monotonic": abs(corr) > 0.6 and pval < 0.05,
            }
        else:
            monotonicity = None

        # 多空夏普
        if "long_short" in quantile_returns.columns:
            ls = quantile_returns["long_short"].dropna()
            if len(ls) > 1:
                ls_mean = float(ls.mean())
                ls_std = float(ls.std(ddof=1))
                ls_sharpe = (
                    float(ls_mean / ls_std * np.sqrt(trading_days))
                    if ls_std > 0
                    else 0.0
                )
            else:
                ls_sharpe = None
        else:
            ls_sharpe = None

        return {
            "stats": [s.to_dict() for s in stats_list],
            "monotonicity": monotonicity,
            "long_short_sharpe": ls_sharpe,
        }