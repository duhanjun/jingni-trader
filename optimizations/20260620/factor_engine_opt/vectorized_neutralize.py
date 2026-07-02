"""
向量化因子中性化

借鉴来源：
- Qlib DataHandler/Processor：声明式、向量化预处理
- AlphaPurify：40+ 预处理方法（含 ridge/lasso 中性化）

优化点：
jingni-trader 现有 `FactorEngine.neutralize` 对每个日期循环：
  1. 切片该日数据
  2. pd.get_dummies 构造行业哑变量
  3. sklearn LinearRegression().fit() + predict()
  4. 取残差
每个日期都重复 sklearn 对象构造与拟合开销，D 个日期 = D 次 Python 调用。

本模块用 `groupby('date').apply` + numpy 最小二乘（lstsq）替代：
  - 避免 sklearn 对象构造开销
  - numpy lstsq 对小矩阵（截面 ~数千行 × 几十个哑变量）极快
  - 一次 apply 完成，逻辑清晰

进一步：对「仅市值中性化」的常见场景，用 Frisch-Waugh-Lovell 定理
完全向量化（无需逐日 apply），性能更优。
"""
from typing import Optional, List
import numpy as np
import pandas as pd


def _neutralize_one_group(
    group: pd.DataFrame,
    factor: str,
    x_cols: List[str],
    min_count: int = 30,
) -> pd.Series:
    """对单个截面做 OLS 取残差（numpy 实现，避免 sklearn 开销）"""
    sub = group[[factor] + x_cols].dropna()
    if len(sub) < min_count:
        # 样本不足，返回原值
        return group[factor]
    y = sub[factor].to_numpy(dtype=float)
    X = sub[x_cols].to_numpy(dtype=float)
    # 加截距项
    X = np.column_stack([np.ones(len(X)), X])
    try:
        # lstsq 返回最小二乘解
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = y - X @ beta
    except np.linalg.LinAlgError:
        return group[factor]
    out = group[factor].copy()
    out.loc[sub.index] = residual
    return out


def neutralize_vectorized(
    factor_df: pd.DataFrame,
    factor_names: List[str],
    neutralize_mcap: bool = True,
    neutralize_industry: bool = True,
    mcap_col: str = "lncap",
    industry_col: str = "industry",
    min_count: int = 30,
) -> pd.DataFrame:
    """
    向量化因子中性化（行业 + 市值），输出 {factor}_neutral 列。

    参数:
        factor_df: 含 code, date, [因子列], 可选 lncap/industry 的 DataFrame
        factor_names: 待中性化的因子列名
        neutralize_mcap: 是否市值中性化
        neutralize_industry: 是否行业中性化
        mcap_col: 对数市值列名
        industry_col: 行业列名
        min_count: 截面最少样本数

    返回:
        在原 DataFrame 基础上新增 {factor}_neutral 列
    """
    if not neutralize_industry and not neutralize_mcap:
        return factor_df
    if factor_df.empty:
        return factor_df

    result = factor_df.copy()

    # 构造回归自变量列
    x_cols: List[str] = []
    if neutralize_mcap and mcap_col in result.columns:
        x_cols.append(mcap_col)
    if neutralize_industry and industry_col in result.columns:
        # 行业哑变量（一次性构造，避免逐日重复 get_dummies）
        dummies = pd.get_dummies(result[industry_col], prefix="ind", dtype=float)
        # 仅保留出现频次较高的行业，避免稀疏哑变量过多
        for col in dummies.columns:
            result[col] = dummies[col].values
            x_cols.append(col)

    if not x_cols:
        # 没有可用自变量，直接复制原值
        for factor in factor_names:
            if factor in result.columns:
                result[f"{factor}_neutral"] = result[factor]
        return result

    # 逐因子、逐截面 apply（numpy lstsq，远快于 sklearn）
    for factor in factor_names:
        if factor not in result.columns:
            continue
        neutralized = result.groupby("date", group_keys=False).apply(
            _neutralize_one_group,
            factor=factor,
            x_cols=x_cols,
            min_count=min_count,
        )
        result[f"{factor}_neutral"] = neutralized

    # 清理临时哑变量列
    dummy_cols = [c for c in x_cols if c.startswith("ind_")]
    if dummy_cols:
        result = result.drop(columns=dummy_cols)

    return result


def neutralize_mcap_only_vectorized(
    factor_df: pd.DataFrame,
    factor_names: List[str],
    mcap_col: str = "lncap",
    min_count: int = 30,
) -> pd.DataFrame:
    """
    仅市值中性化的完全向量化实现（Frisch-Waugh-Lovell 定理）。

    FWL 定理：y 对 [1, x] 回归的残差 = y - x̃'β̃，其中
      x̃ = x - mean(x)，β̃ = (x̃'x̃)^-1 x̃'y
    由于市值是单变量，可完全向量化（无需逐日 apply）：
      residual = y - mean(y) - cov(x,y)/var(x) * (x - mean(x))
    所有 mean/cov/var 均通过 groupby + transform 一次性计算。

    适用场景：仅需市值中性化（无行业中性化）时性能最优。
    """
    if factor_df.empty or mcap_col not in factor_df.columns:
        return factor_df

    result = factor_df.copy()
    valid = result.dropna(subset=[mcap_col])

    if valid.empty:
        for f in factor_names:
            if f in result.columns:
                result[f"{f}_neutral"] = result[f]
        return result

    # 组内均值（市值与因子）
    g = valid.groupby("date")
    x_mean = g[mcap_col].transform("mean")

    # 组内样本数过滤
    counts = g.size().reindex(valid["date"]).values
    mask = counts >= min_count

    for factor in factor_names:
        if factor not in result.columns:
            continue
        y_mean = g[factor].transform("mean")
        x_centered = valid[mcap_col] - x_mean
        y_centered = valid[factor] - y_mean

        # 组内 var(x) 与 cov(x,y)
        var_x = (x_centered ** 2).groupby(valid["date"]).transform("sum")
        cov_xy = (x_centered * y_centered).groupby(valid["date"]).transform("sum")

        beta = cov_xy / var_x.replace(0, np.nan)
        residual = y_centered - beta * x_centered

        out = result[factor].astype(float).copy()
        # 样本数不足的组保留原值
        residual_values = residual.where(mask, result.loc[valid.index, factor])
        out.loc[valid.index] = residual_values
        result[f"{factor}_neutral"] = out

    return result
