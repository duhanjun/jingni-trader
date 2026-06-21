"""
向量化 IC 分析模块

借鉴来源:
- Microsoft Qlib: IC 分析是因子筛选核心，但原版实现用 Python 逐日循环，性能差
- VectorBT: 向量化思想——用 groupby + 矩阵运算替代 for 循环

优化点:
jingni-trader factor-engine._calc_ic 使用 `for dt in dates` 逐日循环计算 Spearman IC，
当股票数 × 日期数 × 因子数 较大时性能极差。

本模块用 pandas groupby + rank + corr 一次性向量化计算所有日期的 IC 序列，
并提供 IC 衰减分析(多持有期)与分层 IC(quantile IC)。
"""
from __future__ import annotations
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("vectorized-ic")


def calc_ic_vectorized(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
    forward_col: str = "ret_forward_5d",
    ic_type: str = "spearman",
    min_stocks: int = 10,
) -> Dict[str, pd.Series]:
    """
    向量化计算多因子的 IC 时间序列

    参数:
        factor_df: 含 code, date, [因子列]
        forward_returns: 含 code, date, [forward_col]
        factor_names: 因子名列表
        forward_col: 前瞻收益列名
        ic_type: "spearman" 或 "pearson"
        min_stocks: 截面最小股票数

    返回:
        {因子名: IC 时间序列(Series, index=date)}
    """
    # 过滤掉不存在的因子列，避免 KeyError
    available_factors = [f for f in factor_names if f in factor_df.columns]
    if not available_factors:
        return {}

    merged = factor_df[["code", "date"] + available_factors].merge(
        forward_returns[["code", "date", forward_col]],
        on=["code", "date"], how="inner",
    )
    merged = merged.dropna(subset=[forward_col])

    result: Dict[str, pd.Series] = {}
    for factor in available_factors:
        if factor not in merged.columns:
            continue
        sub = merged.dropna(subset=[factor])
        if sub.empty:
            continue

        if ic_type == "spearman":
            # 向量化: 按日期分组排名后再算 Pearson 相关 = Spearman
            sub = sub.copy()
            sub["_f_rank"] = sub.groupby("date")[factor].rank()
            sub["_r_rank"] = sub.groupby("date")[forward_col].rank()
            x_col, y_col = "_f_rank", "_r_rank"
        else:
            x_col, y_col = factor, forward_col

        # 过滤截面股票数不足的日期
        counts = sub.groupby("date").size()
        valid_dates = counts[counts >= min_stocks].index
        sub = sub[sub["date"].isin(valid_dates)]
        if sub.empty:
            continue

        # 向量化协方差/方差计算 IC
        grp = sub.groupby("date")
        mean_x = grp[x_col].transform("mean")
        mean_y = grp[y_col].transform("mean")
        dx = sub[x_col] - mean_x
        dy = sub[y_col] - mean_y
        cov = (dx * dy).groupby(sub["date"]).sum()
        var_x = (dx ** 2).groupby(sub["date"]).sum()
        var_y = (dy ** 2).groupby(sub["date"]).sum()
        denom = np.sqrt(var_x * var_y)
        ic_series = (cov / denom.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        result[factor] = ic_series

    return result


def calc_ic_stats(ic_series_dict: Dict[str, pd.Series]) -> List[Dict[str, Any]]:
    """
    汇总 IC 序列的统计量

    返回每个因子的: ic_mean, ic_std, ic_ir, ic_positive_ratio, ic_t_stat
    """
    stats_list = []
    for factor, ic in ic_series_dict.items():
        ic = ic.dropna()
        if len(ic) < 2:
            continue
        ic_mean = ic.mean()
        ic_std = ic.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        n = len(ic)
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
        stats_list.append({
            "factor": factor,
            "ic_mean": round(float(ic_mean), 6),
            "ic_std": round(float(ic_std), 6),
            "ic_ir": round(float(ic_ir), 4),
            "ic_positive_ratio": round(float((ic > 0).mean()), 4),
            "ic_t_stat": round(float(t_stat), 4),
            "ic_count": int(n),
        })
    return stats_list


def calc_ic_decay(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
    periods: List[int] = [1, 5, 10, 20, 60],
    ic_type: str = "spearman",
) -> pd.DataFrame:
    """
    IC 衰减分析: 计算因子在不同持有期的 IC，观察预测能力衰减

    返回:
        DataFrame, index=因子名, columns=持有期, values=ic_mean
    """
    decay_data = {}
    for p in periods:
        col = f"ret_forward_{p}d"
        if col not in forward_returns.columns:
            continue
        ic_dict = calc_ic_vectorized(
            factor_df, forward_returns, factor_names, col, ic_type,
        )
        decay_data[p] = {f: float(ic.mean()) for f, ic in ic_dict.items()}

    if not decay_data:
        return pd.DataFrame()
    df = pd.DataFrame(decay_data)
    df.index.name = "factor"
    return df


def calc_quantile_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_name: str,
    forward_col: str = "ret_forward_5d",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    分层 IC: 计算各分组的平均收益，观察因子单调性

    返回:
        DataFrame, columns=[quantile, mean_return, count]
    """
    merged = factor_df[["code", "date", factor_name]].merge(
        forward_returns[["code", "date", forward_col]],
        on=["code", "date"], how="inner",
    ).dropna(subset=[factor_name, forward_col])

    merged["quantile"] = merged.groupby("date")[factor_name].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop") + 1
    )
    result = merged.groupby("quantile")[forward_col].agg(["mean", "count"]).reset_index()
    result.columns = ["quantile", "mean_return", "count"]
    return result


# ── 兼容旧接口: 与 jingni-trader FactorEngine._calc_ic 等价 ──────────────
def calc_ic_legacy(
    data: pd.DataFrame,
    factor_col: str,
    forward_col: str,
    ic_type: str = "spearman",
) -> Optional[pd.Series]:
    """
    逐日循环版 IC 计算(与 jingni-trader 原实现等价)
    保留用于性能对比测试
    """
    if forward_col not in data.columns:
        return None
    ic_list = []
    dates = sorted(data["date"].unique())
    for dt in dates:
        cross = data[data["date"] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < 10:
            continue
        if ic_type == "spearman":
            ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col], nan_policy="omit")
        else:
            ic, _ = stats.pearsonr(cross[factor_col].fillna(0), cross[forward_col].fillna(0))
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})
    if not ic_list:
        return None
    ic_df = pd.DataFrame(ic_list)
    ic_df["date"] = pd.to_datetime(ic_df["date"])
    return ic_df.set_index("date")["ic"]


def benchmark_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
    forward_col: str = "ret_forward_5d",
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    性能基准: 对比向量化版与逐日循环版的耗时与结果一致性

    返回:
        {
            "vectorized_time": float,
            "legacy_time": float,
            "speedup": float,
            "max_abs_diff": float,  # IC 均值最大绝对差
            "results": {...},
        }
    """
    # 向量化版
    t0 = time.perf_counter()
    vec_ic = calc_ic_vectorized(
        factor_df, forward_returns, factor_names, forward_col, ic_type,
    )
    vec_time = time.perf_counter() - t0

    # 逐日循环版
    merged = factor_df[["code", "date"] + factor_names].merge(
        forward_returns[["code", "date", forward_col]], on=["code", "date"], how="inner",
    )
    t0 = time.perf_counter()
    legacy_ic = {}
    for f in factor_names:
        legacy_ic[f] = calc_ic_legacy(merged, f, forward_col, ic_type)
    legacy_time = time.perf_counter() - t0

    # 一致性校验
    max_diff = 0.0
    for f in factor_names:
        if f in vec_ic and f in legacy_ic and legacy_ic[f] is not None:
            v_mean = float(vec_ic[f].mean())
            l_mean = float(legacy_ic[f].mean())
            max_diff = max(max_diff, abs(v_mean - l_mean))

    speedup = legacy_time / vec_time if vec_time > 0 else float("inf")
    return {
        "vectorized_time_sec": round(vec_time, 4),
        "legacy_time_sec": round(legacy_time, 4),
        "speedup": round(speedup, 2),
        "max_abs_ic_mean_diff": round(max_diff, 8),
        "n_factors": len(factor_names),
        "n_dates": int(factor_df["date"].nunique()),
    }
