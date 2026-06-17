"""
增强 IC 分析（Enhanced IC Analysis）
=====================================

借鉴来源
--------
- microsoft/qlib.qlib.contrib.analysis 的 IC 分析（rank_ic, ic_ir, ic_decay）
- alphalens 的 factor_returns / quantile_returns / mean_return_by_quantile
- jingni-trader 原 factor-engine/engine.py 的 ic_analysis

设计目标
--------
扩展原 factor-engine 的 IC 分析能力：
1. 增加 IC 衰减分析（IC Decay）：考察不同 forward period（1/5/10/20/40/60d）的 IC
2. 增加分位数收益分析：把因子值分 5/10 组，看各组的平均收益
3. 增加换手率分析（Turnover Analysis）：相邻两期因子排名的变化程度
4. 增加因子衰减半衰期估计（IC Half-life）：用 AR(1) 估计
5. 输出标准化报告 dict，便于落盘 + 集成到 reports-engine

API
---
>>> from quant_opt_20260617.ic_analysis import ICAnalyzer
>>> analyzer = ICAnalyzer(forward_periods=[1, 5, 10, 20, 40, 60], n_quantiles=5)
>>> report = analyzer.run(factor_df, price_df)
>>> report['ic_decay']      # DataFrame
>>> report['quantile_returns']  # DataFrame
>>> report['turnover']      # DataFrame
>>> report['half_life']     # DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class ICAnalyzer:
    """IC 分析器"""

    def __init__(
        self,
        forward_periods: List[int] = (1, 5, 10, 20, 40, 60),
        n_quantiles: int = 5,
        min_cross_size: int = 30,
        rank_method: str = "spearman",
    ):
        self.forward_periods = list(forward_periods)
        self.n_quantiles = n_quantiles
        self.min_cross_size = min_cross_size
        self.rank_method = rank_method

    # --------------------------------------------------------
    # IC 衰减
    # --------------------------------------------------------

    def calc_ic_decay(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_cols: List[str],
    ) -> pd.DataFrame:
        """
        IC Decay：计算每个因子在每个 forward_period 的 IC
        """
        # 准备 forward returns
        price_df = price_df.sort_values(['code', 'date']).copy()
        fwd_dict = {'code': price_df['code'], 'date': price_df['date']}
        for p in self.forward_periods:
            fwd_dict[f'fwd_{p}d'] = (
                price_df.groupby('code')['close'].transform(
                    lambda x: x.shift(-p) / x - 1
                )
            )
        fwd_df = pd.DataFrame(fwd_dict)

        merged = factor_df.merge(fwd_df, on=['code', 'date'], how='inner')

        rows = []
        for factor in factor_cols:
            if factor not in merged.columns:
                continue
            for p in self.forward_periods:
                col = f'fwd_{p}d'
                if col not in merged.columns:
                    continue
                daily_ics = []
                for dt, group in merged.groupby('date'):
                    sub = group[[factor, col]].dropna()
                    if len(sub) < self.min_cross_size:
                        continue
                    if self.rank_method == "spearman":
                        ic, _ = spearmanr(sub[factor], sub[col], nan_policy='omit')
                    else:
                        ic = sub[factor].corr(sub[col])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                if daily_ics:
                    rows.append({
                        "factor": factor,
                        "forward_period": p,
                        "ic_mean": float(np.mean(daily_ics)),
                        "ic_std": float(np.std(daily_ics)),
                        "ic_ir": float(np.mean(daily_ics) / (np.std(daily_ics) + 1e-9)),
                        "ic_pos_ratio": float(np.mean([1 if x > 0 else 0 for x in daily_ics])),
                        "ic_t_stat": float(np.mean(daily_ics) / (np.std(daily_ics) / np.sqrt(len(daily_ics)) + 1e-9)),
                        "n_days": len(daily_ics),
                    })
        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # 分位数收益
    # --------------------------------------------------------

    def calc_quantile_returns(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_cols: List[str],
        forward_period: int = 5,
    ) -> pd.DataFrame:
        """
        Quantile Returns：每天按因子值分 Q 组，计算每组的下期平均收益
        输出长表: factor, quantile, mean_return, n_days
        """
        price_df = price_df.sort_values(['code', 'date']).copy()
        fwd = price_df.groupby('code')['close'].transform(
            lambda x: x.shift(-forward_period) / x - 1
        )
        price_df[f'fwd_{forward_period}d'] = fwd

        merged = factor_df.merge(
            price_df[['code', 'date', f'fwd_{forward_period}d']],
            on=['code', 'date'], how='inner'
        ).dropna(subset=[f'fwd_{forward_period}d'])

        rows = []
        for factor in factor_cols:
            if factor not in merged.columns:
                continue
            # 每天按因子分位数分组
            merged['_q'] = merged.groupby('date')[factor].transform(
                lambda x: pd.qcut(x.rank(method='first'), self.n_quantiles, labels=False, duplicates='drop')
            )
            grouped = merged.groupby('_q')[f'fwd_{forward_period}d'].agg(['mean', 'std', 'count'])
            for q, row in grouped.iterrows():
                rows.append({
                    "factor": factor,
                    "quantile": int(q) + 1,  # 1-indexed
                    "mean_return": float(row['mean']),
                    "std_return": float(row['std']) if not np.isnan(row['std']) else 0.0,
                    "n_obs": int(row['count']),
                })
            # 多空收益（最大组 - 最小组）
            qs = sorted(set(r['quantile'] for r in rows if r['factor'] == factor))
            if len(qs) >= 2:
                top = next(r['mean_return'] for r in rows if r['factor'] == factor and r['quantile'] == qs[-1])
                bot = next(r['mean_return'] for r in rows if r['factor'] == factor and r['quantile'] == qs[0])
                rows.append({
                    "factor": factor,
                    "quantile": "long_short",
                    "mean_return": top - bot,
                    "std_return": 0.0,
                    "n_obs": 0,
                })
        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # 换手率
    # --------------------------------------------------------

    def calc_turnover(
        self,
        factor_df: pd.DataFrame,
        factor_cols: List[str],
    ) -> pd.DataFrame:
        """
        Turnover：相邻两期因子截面排名的平均变化程度
        取值范围 [0, 1]，越高表示因子越不稳定
        """
        df = factor_df.sort_values(['date', 'code']).copy()
        rows = []
        for factor in factor_cols:
            if factor not in df.columns:
                continue
            df['_rank'] = df.groupby('date')[factor].rank(pct=True, na_option='keep')
            # 按 code 对齐：每只股票观察其排名在相邻两期的变化
            pivot = df.pivot(index='date', columns='code', values='_rank')
            delta = pivot.diff().abs()
            # 每天的换手率 ≈ 平均 rank 变化
            daily_to = delta.mean(axis=1)
            for dt, v in daily_to.items():
                if pd.isna(v):
                    continue
                rows.append({
                    "factor": factor,
                    "date": pd.Timestamp(dt),
                    "turnover": float(v),
                })
        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # Half-life 估计（AR(1)）
    # --------------------------------------------------------

    @staticmethod
    def estimate_half_life(series: pd.Series) -> float:
        """
        用 AR(1): x_t = c + phi * x_{t-1} + e_t
        half_life = -log(2) / log(|phi|)
        """
        s = series.dropna()
        if len(s) < 10:
            return float('nan')
        # 用 OLS 拟合 AR(1)
        x = s.values[1:]
        x_lag = s.values[:-1]
        if np.std(x_lag) < 1e-12:
            return float('nan')
        # phi = cov(x, x_lag) / var(x_lag)
        phi = np.cov(x, x_lag, ddof=1)[0, 1] / (np.var(x_lag, ddof=1) + 1e-12)
        if abs(phi) >= 1.0 or phi <= 0:
            return float('inf')
        return float(-np.log(2) / np.log(abs(phi)))

    def calc_half_life(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_cols: List[str],
        forward_period: int = 5,
    ) -> pd.DataFrame:
        """
        对每个因子估计其 IC 半衰期（half-life）
        数值越小说明因子衰减越快
        """
        price_df = price_df.sort_values(['code', 'date']).copy()
        fwd = price_df.groupby('code')['close'].transform(
            lambda x: x.shift(-forward_period) / x - 1
        )
        price_df[f'fwd_{forward_period}d'] = fwd

        merged = factor_df.merge(
            price_df[['code', 'date', f'fwd_{forward_period}d']],
            on=['code', 'date'], how='inner'
        )

        rows = []
        for factor in factor_cols:
            if factor not in merged.columns:
                continue
            # 逐日 IC
            daily_ic = []
            for dt, group in merged.groupby('date'):
                sub = group[[factor, f'fwd_{forward_period}d']].dropna()
                if len(sub) < self.min_cross_size:
                    continue
                ic, _ = spearmanr(sub[factor], sub[f'fwd_{forward_period}d'], nan_policy='omit')
                if not np.isnan(ic):
                    daily_ic.append((pd.Timestamp(dt), float(ic)))
            if not daily_ic:
                continue
            ic_series = pd.Series([v for _, v in daily_ic], index=[d for d, _ in daily_ic])
            hl = self.estimate_half_life(ic_series)
            rows.append({
                "factor": factor,
                "ic_half_life": hl,
                "n_days": len(ic_series),
                "ic_mean": float(ic_series.mean()),
            })
        return pd.DataFrame(rows)

    # --------------------------------------------------------
    # 统一入口
    # --------------------------------------------------------

    def run(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_cols: Optional[List[str]] = None,
        forward_period_quantile: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """
        一站式分析，返回所有结果
        """
        if factor_cols is None:
            factor_cols = [
                c for c in factor_df.columns
                if c not in ('code', 'date', 'industry')
            ]

        report: Dict[str, pd.DataFrame] = {}
        report["ic_decay"] = self.calc_ic_decay(factor_df, price_df, factor_cols)
        report["quantile_returns"] = self.calc_quantile_returns(
            factor_df, price_df, factor_cols, forward_period=forward_period_quantile
        )
        report["turnover"] = self.calc_turnover(factor_df, factor_cols)
        report["half_life"] = self.calc_half_life(
            factor_df, price_df, factor_cols, forward_period=forward_period_quantile
        )
        return report
