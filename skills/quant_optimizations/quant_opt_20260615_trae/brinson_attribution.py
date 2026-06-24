"""
Brinson-Fachler 三因素归因
==========================

现有 jingni-trader reports-engine 中的 `calc_brinson_attribution` 实现了
简化的 Brinson 模型，但缺少 (1) 交互项归因数值校验、(2) 与业绩基准可加性
的一致性检查、(3) 行业层面分解。

本模块实现学术标准 Brinson-Fachler (1985) 三因素模型：
    - 配置效应 (Allocation):  (w_p - w_b) * r_b
    - 选择效应 (Selection):   w_b * (r_p - r_b)
    - 交互效应 (Interaction): (w_p - w_b) * (r_p - r_b)
    - 总超额收益 = 配置 + 选择 + 交互

References:
    - Brinson, Hood, Beebower (1986)
    - Brinson, Fachler (1985) — 使用基准收益而非组合收益的 Allocation 项
    - 业界实现: pyfolio, quantstats
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def brinson_fachler(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """
    Brinson-Fachler 三因素归因。

    参数:
        portfolio_weights: 组合在每个行业 / 资产上的权重 (sum≈1)
        benchmark_weights: 基准在每个行业 / 资产上的权重 (sum≈1)
        portfolio_returns: 组合在每个行业 / 资产上的收益
        benchmark_returns: 基准在每个行业 / 资产上的收益

    返回:
        dict with allocation_effect, selection_effect, interaction_effect,
        total_excess_return
    """
    if portfolio_weights.empty:
        return {}

    # 对齐索引
    common = (
        portfolio_weights.index
        .intersection(benchmark_weights.index)
        .intersection(portfolio_returns.index)
        .intersection(benchmark_returns.index)
    )
    if len(common) == 0:
        return {}

    w_p = portfolio_weights.reindex(common).fillna(0.0)
    w_b = benchmark_weights.reindex(common).fillna(0.0)
    r_p = portfolio_returns.reindex(common).fillna(0.0)
    r_b = benchmark_returns.reindex(common).fillna(0.0)

    # Brinson-Fachler 三因素
    allocation = float(((w_p - w_b) * r_b).sum())
    selection = float((w_b * (r_p - r_b)).sum())
    interaction = float(((w_p - w_b) * (r_p - r_b)).sum())
    total_excess = allocation + selection + interaction

    # 一致性校验
    direct = float((w_p * r_p - w_b * r_b).sum())
    diff = abs(total_excess - direct)
    if diff > 1e-6:
        raise ValueError(
            f"Brinson-Fachler 分解不闭合: 残差={diff:.6e}, "
            f"应等于直接计算 {direct:.6f}"
        )

    return {
        "allocation_effect": allocation,
        "selection_effect": selection,
        "interaction_effect": interaction,
        "total_excess_return": total_excess,
        "direct_excess_return": direct,
        "residual": diff,
    }


def brinson_by_industry(
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    按截面 (行业 × 日期) 进行 Brinson-Fachler 归因。

    参数:
        *_weights / *_returns: DataFrame[date x industry]
    """
    if portfolio_weights.empty:
        return pd.DataFrame()

    # 对齐
    pw, bw = portfolio_weights.align(benchmark_weights, join="inner")
    pr, br = portfolio_returns.align(benchmark_returns, join="inner")

    rows = []
    for dt in pw.index:
        result = brinson_fachler(
            pw.loc[dt], bw.loc[dt], pr.loc[dt], br.loc[dt]
        )
        if result:
            result["date"] = dt
            rows.append(result)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def brinson_attribution_summary(
    portfolio_weights_by_period: pd.DataFrame,
    benchmark_weights_by_period: pd.DataFrame,
    portfolio_returns_by_period: pd.DataFrame,
    benchmark_returns_by_period: pd.DataFrame,
) -> Dict[str, float]:
    """
    跨期聚合 Brinson-Fachler 归因 (对每日 / 每周效应简单累加)。
    """
    if portfolio_weights_by_period.empty:
        return {}

    df = brinson_by_industry(
        portfolio_weights_by_period,
        benchmark_weights_by_period,
        portfolio_returns_by_period,
        benchmark_returns_by_period,
    )
    if df.empty:
        return {}

    return {
        "allocation_cumulative": float(df["allocation_effect"].sum()),
        "selection_cumulative": float(df["selection_effect"].sum()),
        "interaction_cumulative": float(df["interaction_effect"].sum()),
        "total_excess_cumulative": float(df["total_excess_return"].sum()),
        "allocation_daily_mean": float(df["allocation_effect"].mean()),
        "selection_daily_mean": float(df["selection_effect"].mean()),
        "interaction_daily_mean": float(df["interaction_effect"].mean()),
        "n_periods": int(len(df)),
    }