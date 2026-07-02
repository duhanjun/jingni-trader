"""
向量化 IC 分析（借鉴 VectorBT 的向量化思想）

核心优化点：
    现有 factor-engine 的 _calc_ic() 通过 `for dt in dates` 逐日循环计算
    Spearman/Pearson 相关性，在 1000+ 交易日 × 4000+ 股票规模下非常慢。

    本模块用 pandas groupby + transform 向量化实现，避免 Python 层循环，
    在保持数值结果一致的前提下大幅提升性能。

借鉴来源：
    - VectorBT: 将逐 bar 循环替换为矩阵/向量化运算
    - Qlib: IC/ICIR/Rank IC 的标准化定义

与现有 factor-engine 的关系：
    不修改 main 分支代码；作为独立优化验证模块。
    输出与 engine.ic_analysis() 等价，可直接对比正确性与性能。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("vectorized-ic")


def _rank_within_group(s: pd.Series) -> pd.Series:
    """截面排名（按 date 分组），返回 1..n 的秩"""
    return s.groupby(level='date').rank()


def ic_analysis_vectorized(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
    min_stocks: int = 10,
) -> Dict[str, Any]:
    """
    向量化 IC 分析。

    参数:
        factor_df: 含 code, date, [因子列] 的 DataFrame
        forward_returns: 含 code, date, ret_forward_1d/5d/20d 的 DataFrame
        factor_names: 待分析的因子列名；None 表示自动推断
        ic_type: "spearman" (Rank IC) 或 "pearson" (普通 IC)
        min_stocks: 截面最少股票数，低于此数跳过该日

    返回:
        与现有 engine.ic_analysis() 结构一致的结果字典：
        {
            "ret_forward_1d": [{factor, forward_period, ic_mean, ic_std, ic_ir, ...}, ...],
            "ret_forward_5d": [...],
            "ret_forward_20d": [...],
        }
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    # 合并因子与远期收益
    fwd_cols = [c for c in forward_returns.columns if c.startswith('ret_forward_')]
    data = factor_df.merge(
        forward_returns[['code', 'date'] + fwd_cols],
        on=['code', 'date'],
        how='inner',
    )
    if data.empty:
        return {}

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns
            if c not in ('code', 'date', 'industry')
        ]

    # 构造 MultiIndex 便于 groupby
    data = data.set_index(['date', 'code']).sort_index()

    # 预计算每个因子的截面排名（Spearman 用），一次性完成
    rank_cache: Dict[str, pd.Series] = {}
    if ic_type == "spearman":
        for f in factor_names:
            if f in data.columns:
                rank_cache[f] = data[f].groupby(level='date').rank()

    results: Dict[str, Any] = {}
    for fwd_col in fwd_cols:
        if fwd_col not in data.columns:
            continue

        # 远期收益的截面排名（Spearman）
        if ic_type == "spearman":
            fwd_rank = data[fwd_col].groupby(level='date').rank()
        else:
            fwd_rank = data[fwd_col]

        # 截面股票数（用于过滤 min_stocks）
        counts_per_day = data[fwd_col].groupby(level='date').count()
        valid_dates = counts_per_day[counts_per_day >= min_stocks].index

        ic_results: List[Dict[str, Any]] = []
        for factor in factor_names:
            if factor not in data.columns:
                continue

            if ic_type == "spearman":
                x = rank_cache.get(factor)
                if x is None:
                    continue
                y = fwd_rank
            else:
                x = data[factor]
                y = data[fwd_col]

            # 向量化逐日 IC：用 groupby + corr 一次性算出每日 IC 序列
            df_pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()
            if df_pair.empty:
                continue

            # 仅保留有效日期
            df_pair = df_pair[df_pair.index.get_level_values('date').isin(valid_dates)]
            if df_pair.empty:
                continue

            # 核心向量化：按 date 分组求 corr，得到每日 IC 序列
            ic_series = df_pair.groupby(level='date').apply(
                lambda g: g['x'].corr(g['y']) if len(g) >= min_stocks else np.nan
            ).dropna()

            if ic_series.empty:
                continue

            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            ic_positive_ratio = (ic_series > 0).mean()
            n = len(ic_series)
            ic_t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0

            ic_results.append({
                "factor": factor,
                "forward_period": fwd_col,
                "ic_mean": round(float(ic_mean), 6),
                "ic_std": round(float(ic_std), 6),
                "ic_ir": round(float(ic_ir), 4),
                "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                "ic_t_stat": round(float(ic_t_stat), 4),
            })

        results[fwd_col] = ic_results

    logger.info(f"向量化 IC 分析完成，共分析 {len(factor_names)} 个因子")
    return results


def ic_analysis_loop_baseline(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
    min_stocks: int = 10,
) -> Dict[str, Any]:
    """
    逐日循环版 IC 分析（复刻现有 factor-engine._calc_ic 的逻辑），用于性能对比基线。
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    fwd_cols = [c for c in forward_returns.columns if c.startswith('ret_forward_')]
    data = factor_df.merge(
        forward_returns[['code', 'date'] + fwd_cols],
        on=['code', 'date'],
        how='inner',
    )
    if data.empty:
        return {}

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns
            if c not in ('code', 'date', 'industry')
        ]

    results: Dict[str, Any] = {}
    dates = sorted(data['date'].unique())

    for fwd_col in fwd_cols:
        if fwd_col not in data.columns:
            continue
        ic_results: List[Dict[str, Any]] = []
        for factor in factor_names:
            if factor not in data.columns:
                continue
            ic_list = []
            for dt in dates:
                cross = data[data['date'] == dt].dropna(subset=[factor, fwd_col])
                if len(cross) < min_stocks:
                    continue
                if ic_type == "spearman":
                    ic, _ = stats.spearmanr(cross[factor], cross[fwd_col], nan_policy='omit')
                else:
                    ic, _ = stats.pearsonr(cross[factor].fillna(0), cross[fwd_col].fillna(0))
                if not np.isnan(ic):
                    ic_list.append(ic)
            if not ic_list:
                continue
            ic_arr = np.array(ic_list)
            ic_mean = ic_arr.mean()
            ic_std = ic_arr.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
            n = len(ic_arr)
            ic_results.append({
                "factor": factor,
                "forward_period": fwd_col,
                "ic_mean": round(float(ic_mean), 6),
                "ic_std": round(float(ic_std), 6),
                "ic_ir": round(float(ic_ir), 4),
                "ic_positive_ratio": round(float((ic_arr > 0).mean()), 4),
                "ic_t_stat": round(float(ic_mean / (ic_std / np.sqrt(n))) if ic_std > 0 else 0, 4),
            })
        results[fwd_col] = ic_results

    return results


def benchmark_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    对比向量化版与循环版的性能与结果一致性。

    返回:
        {
            "vectorized_time": float,
            "loop_time": float,
            "speedup": float,
            "max_ic_mean_diff": float,   # 两版 ic_mean 最大差异
            "results_match": bool,       # 是否在容差内一致
        }
    """
    t0 = time.perf_counter()
    vec_res = ic_analysis_vectorized(factor_df, forward_returns, factor_names, ic_type)
    t1 = time.perf_counter()
    loop_res = ic_analysis_loop_baseline(factor_df, forward_returns, factor_names, ic_type)
    t2 = time.perf_counter()

    vec_time = t1 - t0
    loop_time = t2 - t1
    speedup = loop_time / vec_time if vec_time > 0 else float('inf')

    # 比较结果一致性
    max_diff = 0.0
    for fwd_col in vec_res:
        vec_map = {item['factor']: item['ic_mean'] for item in vec_res[fwd_col]}
        loop_map = {item['factor']: item['ic_mean'] for item in loop_res.get(fwd_col, [])}
        for f, v in vec_map.items():
            if f in loop_map:
                max_diff = max(max_diff, abs(v - loop_map[f]))

    return {
        "vectorized_time_sec": round(vec_time, 4),
        "loop_time_sec": round(loop_time, 4),
        "speedup": round(speedup, 2),
        "max_ic_mean_diff": round(max_diff, 6),
        "results_match": max_diff < 1e-3,
        "vectorized_n_factors": sum(len(v) for v in vec_res.values()),
        "loop_n_factors": sum(len(v) for v in loop_res.values()),
    }
