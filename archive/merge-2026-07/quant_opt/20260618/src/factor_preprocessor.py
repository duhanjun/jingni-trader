"""
因子预处理器 (factor_preprocessor)
====================================

借鉴来源:
  - AlphaPurify (alphapurify): 40+ 标准化方法
  - Alphalens (quantopian): clean_factor_and_forward_returns
  - jingni-trader/skills/factor-engine/engine.py 的 neutralize 函数

设计目标:
  1. 提供 winsorize / standardize / neutralize 三类基础工具
  2. 输出可被 jingni-trader 直接使用的标准格式
  3. 与 jingni-trader 的因子分析 IC 分析无缝衔接
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. Winsorization (缩尾)
# ============================================================

def winsorize_zscore(
    factor: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    """
    Z-Score Winsorization: 把绝对值 > threshold * std 的样本截尾

    公式: x_clipped = clip((x - mean) / std, -threshold, threshold) * std + mean
    """
    if factor.empty:
        return factor
    mean = factor.mean()
    std = factor.std()
    if std == 0 or pd.isna(std):
        return factor
    z = (factor - mean) / std
    z_clipped = z.clip(-threshold, threshold)
    return z_clipped * std + mean


def winsorize_quantile(
    factor: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """
    Quantile Winsorization: 把低于 lower 分位 / 高于 upper 分位的样本截尾
    """
    if factor.empty:
        return factor
    lo = factor.quantile(lower)
    hi = factor.quantile(upper)
    return factor.clip(lo, hi)


def winsorize_mad(
    factor: pd.Series,
    n_mad: float = 5.0,
) -> pd.Series:
    """
    MAD Winsorization: 中位数绝对偏差法（对极端值更稳健）
    """
    if factor.empty:
        return factor
    med = factor.median()
    mad = (factor - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return factor
    lo = med - n_mad * 1.4826 * mad
    hi = med + n_mad * 1.4826 * mad
    return factor.clip(lo, hi)


# ============================================================
# 2. Standardization (标准化)
# ============================================================

def standardize_zscore(factor: pd.Series) -> pd.Series:
    """横截面 Z-Score 标准化: (x - mean) / std"""
    if factor.empty:
        return factor
    std = factor.std()
    if std == 0 or pd.isna(std):
        return factor - factor.mean()
    return (factor - factor.mean()) / std


def standardize_rank(factor: pd.Series) -> pd.Series:
    """横截面 Rank 标准化（百分位秩），对离群值稳健"""
    if factor.empty:
        return factor
    return factor.rank(pct=True) - 0.5


# ============================================================
# 3. Neutralization (中性化)
# ============================================================

def neutralize_industry_mcap(
    factor_df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "industry",
    mcap_col: str = "lncap",
    neutralize_industry: bool = True,
    neutralize_mcap: bool = True,
) -> pd.Series:
    """
    行业 + 市值中性化：对每天的横截面，把因子值对 (industry_dummy, lncap) 做回归，取残差

    Args:
        factor_df: 包含 factor_col、industry_col、mcap_col、date 的 DataFrame
        factor_col: 待中性化的因子列名
        industry_col: 行业列
        mcap_col: 市值对数列
        neutralize_industry: 是否做行业中性化
        neutralize_mcap: 是否做市值中性化
    """
    if factor_df.empty:
        return pd.Series(dtype=float)

    result = pd.Series(index=factor_df.index, dtype=float)
    for dt, grp in factor_df.groupby("date"):
        if len(grp) < 30:
            result.loc[grp.index] = grp[factor_col]
            continue

        y = grp[factor_col].fillna(grp[factor_col].median()).values
        x_parts = []
        if neutralize_mcap and mcap_col in grp.columns:
            x_parts.append(grp[mcap_col].fillna(grp[mcap_col].median()).values.reshape(-1, 1))
        if neutralize_industry and industry_col in grp.columns:
            dummies = pd.get_dummies(grp[industry_col], dummy_na=True)
            x_parts.append(dummies.values)

        if not x_parts:
            result.loc[grp.index] = y
            continue

        X = np.hstack(x_parts)
        try:
            model = LinearRegression()
            model.fit(X, y)
            resid = y - model.predict(X)
        except Exception:
            resid = y
        result.loc[grp.index] = resid
    return result


# ============================================================
# 4. 因子清洗 Pipeline (借鉴 Alphalens clean_factor)
# ============================================================

def clean_factor(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    winsorize: str = "zscore",     # None / "zscore" / "quantile" / "mad"
    standardize: str = "zscore",    # None / "zscore" / "rank"
    neutralize: bool = False,
    industry_col: str = "industry",
    mcap_col: str = "lncap",
    factor_col: str = "factor",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    一站式因子清洗（借鉴 Alphalens `clean_factor_and_forward_returns`）

    Args:
        factor: 含 code, date, factor_col 的 DataFrame
        forward_returns: 含 code, date, ret 的 DataFrame
        winsorize: 缩尾方法
        standardize: 标准化方法
        neutralize: 是否中性化
        industry_col: 行业列
        mcap_col: 市值对数列
        factor_col: 因子列名

    Returns:
        (清洗后的因子 DataFrame, 前瞻收益 DataFrame)
    """
    df = factor.copy()
    fwd = forward_returns.copy()

    # 1) 合并
    df["date"] = pd.to_datetime(df["date"])
    fwd["date"] = pd.to_datetime(fwd["date"])
    merged = df.merge(fwd, on=["code", "date"], how="inner")
    if merged.empty:
        return df, fwd

    # 2) 缺失值清理
    merged = merged.dropna(subset=[factor_col, "ret"])

    # 3) 缩尾
    if winsorize is not None and winsorize != "none":
        method_map = {
            "zscore": winsorize_zscore,
            "quantile": winsorize_quantile,
            "mad": winsorize_mad,
        }
        fn = method_map.get(winsorize)
        if fn is not None:
            merged[factor_col] = merged.groupby("date")[factor_col].transform(fn)

    # 4) 标准化
    if standardize is not None and standardize != "none":
        method_map = {
            "zscore": standardize_zscore,
            "rank": standardize_rank,
        }
        fn = method_map.get(standardize)
        if fn is not None:
            merged[factor_col] = merged.groupby("date")[factor_col].transform(fn)

    # 5) 中性化
    if neutralize:
        if industry_col not in merged.columns:
            merged[industry_col] = "unknown"
        if mcap_col not in merged.columns:
            merged[mcap_col] = 0.0
        merged[f"{factor_col}_neutral"] = neutralize_industry_mcap(
            merged, factor_col, industry_col, mcap_col,
            neutralize_industry=True, neutralize_mcap=True,
        )
        merged[factor_col] = merged[f"{factor_col}_neutral"]

    # 6) 拆回
    factor_out = merged[["code", "date", factor_col]].copy()
    fwd_out = merged[["code", "date", "ret"]].copy()
    return factor_out, fwd_out


__all__ = [
    "winsorize_zscore", "winsorize_quantile", "winsorize_mad",
    "standardize_zscore", "standardize_rank",
    "neutralize_industry_mcap",
    "clean_factor",
]
