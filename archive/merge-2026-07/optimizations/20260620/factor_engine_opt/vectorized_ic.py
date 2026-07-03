"""
向量化 IC 分析

借鉴来源：
- AlphaPurify：向量化 IC 计算，4M 行 25 秒完成
- Qlib：表达式引擎避免逐行 Python 调用

优化点：
jingni-trader 现有 `FactorEngine._calc_ic` 对每个日期循环调用
`scipy.stats.spearmanr`，存在显著 Python 循环开销。本模块通过
`groupby + transform(rank)` + 向量化 Pearson 公式，将逐日 IC 计算
压缩为几次整表运算，大幅减少 Python 层循环。

核心公式：
Spearman rank IC = Pearson(x_rank, y_rank) per date
Pearson(x, y) per group = cov(x,y) / (std(x) * std(y))，可由
  (x - mean_x) · (y - mean_y) / sqrt(Σ(x-mean_x)² · Σ(y-mean_y)²)
  完全向量化计算。
"""
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd


def _rank_within_group(df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
    """在每组内对指定列排名（average rank，与 scipy spearmanr 默认一致）"""
    return df.groupby(group_col)[col].rank(method="average")


def _pearson_by_group_vectorized(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    min_count: int = 10,
) -> pd.Series:
    """
    向量化计算每组内的 Pearson 相关系数。

    返回: Series，index 为 group_col 的唯一值，value 为该组的相关系数。
          样本数 < min_count 的组返回 NaN。
    """
    # 仅保留有效观测
    valid = df[[group_col, x_col, y_col]].dropna()
    if valid.empty:
        return pd.Series(dtype=float)

    # 组内中心化（去均值）
    g = valid.groupby(group_col)
    x_centered = valid[x_col] - g[x_col].transform("mean")
    y_centered = valid[y_col] - g[y_col].transform("mean")

    # 组内样本数
    counts = g.size()

    # 分子：Σ (x-x̄)(y-ȳ)
    num = (x_centered * y_centered).groupby(valid[group_col]).sum()

    # 分母：sqrt(Σ(x-x̄)² · Σ(y-ȳ)²)
    den_x = (x_centered ** 2).groupby(valid[group_col]).sum()
    den_y = (y_centered ** 2).groupby(valid[group_col]).sum()
    den = np.sqrt(den_x * den_y)

    corr = num / den.replace(0, np.nan)

    # 样本数不足的组置 NaN
    corr = corr.where(counts >= min_count, np.nan)
    return corr


def calc_ic_series_vectorized(
    data: pd.DataFrame,
    factor_col: str,
    forward_col: str,
    ic_type: str = "spearman",
    min_count: int = 10,
) -> pd.Series:
    """
    向量化计算单因子的 IC 时间序列。

    参数:
        data: 含 date, factor_col, forward_col 的 DataFrame
        factor_col: 因子列名
        forward_col: 未来收益列名
        ic_type: "spearman" (Rank IC) 或 "pearson" (普通 IC)
        min_count: 每个截面最少样本数，不足返回 NaN

    返回:
        Series，index 为 date，value 为当日 IC
    """
    if factor_col not in data.columns or forward_col not in data.columns:
        return pd.Series(dtype=float)

    work = data[["date", factor_col, forward_col]].copy()

    if ic_type == "spearman":
        # Rank IC：先排名再做 Pearson
        work["_x_rank"] = _rank_within_group(work, factor_col, "date")
        work["_y_rank"] = _rank_within_group(work, forward_col, "date")
        ic = _pearson_by_group_vectorized(
            work, "_x_rank", "_y_rank", "date", min_count=min_count
        )
    else:
        # 普通 IC
        ic = _pearson_by_group_vectorized(
            work, factor_col, forward_col, "date", min_count=min_count
        )

    ic = ic.dropna()
    ic.index.name = "date"
    return ic


def calc_ic_stats_vectorized(ic_series: pd.Series) -> Dict[str, float]:
    """由 IC 序列计算统计量（与现有 engine.ic_analysis 输出对齐）"""
    if ic_series is None or ic_series.empty:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_t_stat": 0.0,
        }
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_positive_ratio = (ic_series > 0).mean()
    n = len(ic_series)
    ic_t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0
    return {
        "ic_mean": round(float(ic_mean), 6),
        "ic_std": round(float(ic_std), 6),
        "ic_ir": round(float(ic_ir), 4),
        "ic_positive_ratio": round(float(ic_positive_ratio), 4),
        "ic_t_stat": round(float(ic_t_stat), 4),
    }


def ic_analysis_vectorized(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    forward_cols: Optional[List[str]] = None,
    ic_type: str = "spearman",
    min_count: int = 10,
) -> Dict[str, Any]:
    """
    向量化批量 IC 分析（对标 FactorEngine.ic_analysis）。

    参数:
        factor_df: 含 code, date, [因子列] 的 DataFrame
        forward_returns: 含 code, date, [ret_forward_*] 的 DataFrame
        factor_names: 待分析的因子列名列表
        forward_cols: 待分析的未来收益列名列表，默认
                      ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']
        ic_type: "spearman" 或 "pearson"
        min_count: 截面最少样本数

    返回:
        {forward_col: [{factor, forward_period, ic_mean, ic_std, ic_ir, ...}, ...]}
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns if c not in ("code", "date", "industry")
        ]
    if forward_cols is None:
        forward_cols = [
            c for c in forward_returns.columns
            if c.startswith("ret_forward_")
        ]

    # 一次性 merge，避免每个因子重复 merge
    # 仅 merge factor_df 中不存在的未来收益列，避免列名冲突（_x/_y 后缀）
    forward_cols_to_merge = [
        c for c in forward_cols
        if c in forward_returns.columns and c not in factor_df.columns
    ]
    if forward_cols_to_merge:
        merge_cols = ["code", "date"] + forward_cols_to_merge
        data = factor_df.merge(
            forward_returns[merge_cols], on=["code", "date"], how="inner"
        )
    else:
        # 所有未来收益列已在 factor_df 中，直接使用
        data = factor_df.copy()

    results: Dict[str, Any] = {}
    for forward_col in forward_cols:
        if forward_col not in data.columns:
            continue
        ic_results = []
        for factor in factor_names:
            if factor not in data.columns:
                continue
            ic_series = calc_ic_series_vectorized(
                data, factor, forward_col, ic_type=ic_type, min_count=min_count
            )
            if ic_series.empty:
                continue
            stats = calc_ic_stats_vectorized(ic_series)
            ic_results.append({
                "factor": factor,
                "forward_period": forward_col,
                **stats,
            })
        results[forward_col] = ic_results

    return results
