"""
因子预处理：去极值 + 标准化

借鉴来源：
- AlphaPurify：40+ 预处理方法（Winsorization / Neutralization / Standardization），
  含 MAD、分位数、ridge、lasso、PCA 等
- Qlib processor：Normalize / RobustZScoreNorm / Fillna 等声明式处理器

优化点：
jingni-trader 现有因子引擎在 IC 分析与融合前未做去极值和标准化，
极端值会扭曲 IC 与 IC-IR 加权，且不同量纲因子无法直接加权融合。
本模块提供截面（per date）向量化预处理：

  - winsorize_mad: MAD 法去极值（抗异常值，比 3σ 更稳健）
  - winsorize_quantile: 分位数法去极值（如 1%/99% 截断）
  - standardize_zscore: Z-score 标准化（截面均值 0、标准差 1）
  - preprocess_factor: 一站式 pipeline（去极值 → 标准化）

所有操作均按 date 分组向量化，避免逐日 Python 循环。
"""
from typing import Optional
import numpy as np
import pandas as pd


def winsorize_mad(
    factor_df: pd.DataFrame,
    factor_col: str,
    n: float = 3.0,
    group_col: str = "date",
) -> pd.Series:
    """
    MAD 法去极值（截面）。

    原理：median ± n * 1.4826 * MAD，其中 MAD = median(|x - median|)。
    1.4826 是使 MAD 在正态分布下与标准差一致的常数。
    比 3σ 更稳健，因为中位数不受极端值影响。

    参数:
        factor_df: 含 group_col 与 factor_col 的 DataFrame
        factor_col: 待处理的因子列
        n: 截断倍数（默认 3.0，等价于 3σ）
        group_col: 分组列（默认按日期截面）

    返回:
        去极值后的 Series
    """
    g = factor_df.groupby(group_col)[factor_col]
    median = g.transform("median")
    mad = (factor_df[factor_col] - median).groupby(factor_df[group_col]).transform(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    # mad=0 时（如一半以上相同值）退化为 0，避免除零
    scale = 1.4826 * mad
    lower = median - n * scale
    upper = median + n * scale
    return factor_df[factor_col].clip(lower=lower, upper=upper)


def winsorize_quantile(
    factor_df: pd.DataFrame,
    factor_col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    group_col: str = "date",
) -> pd.Series:
    """
    分位数法去极值（截面）。

    参数:
        lower_q: 下分位数（默认 1%）
        upper_q: 上分位数（默认 99%）
    """
    g = factor_df.groupby(group_col)[factor_col]
    lower = g.transform("quantile", lower_q)
    upper = g.transform("quantile", upper_q)
    return factor_df[factor_col].clip(lower=lower, upper=upper)


def standardize_zscore(
    factor_df: pd.DataFrame,
    factor_col: str,
    group_col: str = "date",
) -> pd.Series:
    """
    Z-score 标准化（截面）：(x - mean) / std。

    标准化后因子均值为 0、标准差为 1，使不同量纲因子可加权融合。
    std=0 时（如因子值全部相同）返回 0，避免除零：用 std.replace(0, 1)，
    此时 x-mean=0，0/1=0。
    """
    g = factor_df.groupby(group_col)[factor_col]
    mean = g.transform("mean")
    std = g.transform("std")
    # std=0 时替换为 1，避免除零；此时 x-mean=0，结果为 0
    std_safe = std.replace(0, 1.0)
    return (factor_df[factor_col] - mean) / std_safe


def preprocess_factor(
    factor_df: pd.DataFrame,
    factor_col: str,
    winsorize_method: str = "mad",
    winsorize_n: float = 3.0,
    winsorize_lower_q: float = 0.01,
    winsorize_upper_q: float = 0.99,
    standardize: bool = True,
    group_col: str = "date",
) -> pd.Series:
    """
    一站式因子预处理 pipeline：去极值 → 标准化。

    参数:
        winsorize_method: "mad" 或 "quantile" 或 None（跳过去极值）
        winsorize_n: MAD 法倍数
        winsorize_lower_q / winsorize_upper_q: 分位数法上下界
        standardize: 是否标准化
    """
    out = factor_df[factor_col].astype(float).copy()

    if winsorize_method == "mad":
        out = winsorize_mad(factor_df, factor_col, n=winsorize_n, group_col=group_col)
    elif winsorize_method == "quantile":
        out = winsorize_quantile(
            factor_df, factor_col,
            lower_q=winsorize_lower_q, upper_q=winsorize_upper_q,
            group_col=group_col,
        )

    if standardize:
        tmp = factor_df.copy()
        tmp[factor_col] = out
        out = standardize_zscore(tmp, factor_col, group_col=group_col)

    return out
