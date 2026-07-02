"""
向量化 IC 分析

借鉴来源：
- Qlib 的表达式引擎和高效因子计算思想
- Qlib 论文中提到的"高性能基础设施"设计

对照 jingni-trader 现有实现：
- skills/factor-engine/engine.py 的 _calc_ic 方法
  使用 Python for-loop 遍历每个日期，逐日计算 Spearman/Pearson 相关系数

本模块的核心改进：
1. 用 pandas groupby + transform 替代逐日 for-loop
2. 用向量化 rank 计算替代 scipy.stats.spearmanr 逐日调用
3. 一次性计算所有日期的 IC，再聚合统计量

注意：Spearman IC 本质是 rank 后的 Pearson IC，
因此可以先用 groupby+rank 向量化计算 rank，再用向量化 corr 计算 IC。
"""
from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd
from scipy import stats


def vectorized_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
    min_samples: int = 10,
) -> Dict[str, Any]:
    """
    向量化 IC 分析

    参数:
        factor_df: 包含 date, code, [因子列] 的 DataFrame
        forward_returns: 包含 date, code, ret_forward_1d/5d/20d 的 DataFrame
        factor_names: 要分析的因子名列表
        ic_type: "spearman" 或 "pearson"
        min_samples: 计算单日 IC 的最小样本数

    返回:
        {
            "ret_forward_1d": [{factor, ic_mean, ic_std, ic_ir, ...}, ...],
            "ret_forward_5d": [...],
            "ret_forward_20d": [...],
        }
    """
    if factor_df.empty or forward_returns.empty:
        return {}

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns
            if c not in ('code', 'date', 'industry')
        ]

    # 合并因子和前向收益
    forward_cols = [c for c in forward_returns.columns
                    if c.startswith('ret_forward_')]
    data = factor_df.merge(
        forward_returns[['code', 'date'] + forward_cols],
        on=['code', 'date'],
        how='inner',
    )

    results: Dict[str, Any] = {}
    for forward_col in forward_cols:
        if forward_col not in data.columns:
            continue

        ic_results = []
        for factor in factor_names:
            if factor not in data.columns:
                continue

            ic_series = _calc_ic_vectorized(
                data, factor, forward_col, ic_type, min_samples
            )
            if ic_series is None or ic_series.empty:
                continue

            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
            ic_positive_ratio = float((ic_series > 0).mean())
            ic_t_stat = float(
                ic_mean / (ic_std / np.sqrt(len(ic_series)))
            ) if ic_std > 0 else 0.0

            ic_results.append({
                "factor": factor,
                "forward_period": forward_col,
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "ic_ir": round(ic_ir, 4),
                "ic_positive_ratio": round(ic_positive_ratio, 4),
                "ic_t_stat": round(ic_t_stat, 4),
                "n_dates": int(len(ic_series)),
            })

        results[forward_col] = ic_results

    return results


def _calc_ic_vectorized(
    data: pd.DataFrame,
    factor_col: str,
    forward_col: str,
    ic_type: str,
    min_samples: int,
) -> Optional[pd.Series]:
    """
    向量化计算单个因子的 IC 时间序列

    核心优化：
    - Spearman IC = rank(factor) 与 rank(forward_return) 的 Pearson 相关
    - 用 groupby('date').rank() 一次性计算所有日期的 rank
    - 用 groupby('date').apply(corr) 一次性计算所有日期的 IC
    """
    # 过滤无效值
    valid = data.dropna(subset=[factor_col, forward_col])
    if valid.empty:
        return None

    # 计算每日样本数
    daily_counts = valid.groupby('date').size()
    valid_dates = daily_counts[daily_counts >= min_samples].index
    valid = valid[valid['date'].isin(valid_dates)]
    if valid.empty:
        return None

    if ic_type == "spearman":
        # 向量化 rank：按日期分组排名
        valid = valid.copy()
        valid['_f_rank'] = valid.groupby('date')[factor_col].rank(pct=True)
        valid['_r_rank'] = valid.groupby('date')[forward_col].rank(pct=True)
        # 向量化 Pearson 相关：按日期分组计算
        ic_series = _grouped_corr(valid, '_f_rank', '_r_rank')
    else:
        # Pearson IC
        ic_series = _grouped_corr(valid, factor_col, forward_col)

    return ic_series


def _grouped_corr(df: pd.DataFrame, col_a: str, col_b: str) -> pd.Series:
    """
    向量化计算按日期分组的相关系数

    优化点：用 numpy 一次性计算，避免 scipy.stats 逐日调用的开销
    """
    # 方法：用 pandas groupby + cov 计算协方差，再推导相关系数
    # cov(A,B) / (std(A) * std(B)) = corr(A,B)
    grouped = df.groupby('date')
    cov = grouped[[col_a, col_b]].cov()

    # cov 是一个多级索引 DataFrame，提取 A-B 的协方差
    # 索引: (date, col_a) -> col_b
    try:
        cov_ab = cov.xs(col_a, level=1)[col_b]
        var_a = grouped[col_a].var()
        var_b = grouped[col_b].var()
        # 防止除零
        denom = np.sqrt(var_a * var_b)
        ic = cov_ab / denom.where(denom > 0, np.nan)
        ic = ic.dropna()
        return ic
    except KeyError:
        # 回退到 apply 方式
        def _corr(g):
            if len(g) < 2:
                return np.nan
            return g[col_a].corr(g[col_b])
        return grouped.apply(_corr).dropna()


# ------------------------------------------------------------------
# 对照基准：复刻 jingni-trader 现有的循环式 IC 计算
# ------------------------------------------------------------------
def loop_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    ic_type: str = "spearman",
    min_samples: int = 10,
) -> Dict[str, Any]:
    """循环式 IC 计算（对照基准，复刻 jingni-trader 现有实现）"""
    if factor_df.empty or forward_returns.empty:
        return {}

    if factor_names is None:
        factor_names = [
            c for c in factor_df.columns
            if c not in ('code', 'date', 'industry')
        ]

    forward_cols = [c for c in forward_returns.columns
                    if c.startswith('ret_forward_')]
    data = factor_df.merge(
        forward_returns[['code', 'date'] + forward_cols],
        on=['code', 'date'],
        how='inner',
    )

    results: Dict[str, Any] = {}
    for forward_col in forward_cols:
        if forward_col not in data.columns:
            continue

        ic_results = []
        for factor in factor_names:
            if factor not in data.columns:
                continue

            ic_list = []
            dates = sorted(data['date'].unique())
            for dt in dates:
                cross = data[data['date'] == dt].dropna(subset=[factor, forward_col])
                if len(cross) < min_samples:
                    continue
                if ic_type == "spearman":
                    ic, _ = stats.spearmanr(cross[factor], cross[forward_col], nan_policy='omit')
                else:
                    ic, _ = stats.pearsonr(cross[factor].fillna(0), cross[forward_col].fillna(0))
                if not np.isnan(ic):
                    ic_list.append({"date": dt, "ic": ic})

            if not ic_list:
                continue

            ic_df = pd.DataFrame(ic_list).set_index('date')['ic']
            ic_mean = float(ic_df.mean())
            ic_std = float(ic_df.std())
            ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
            ic_positive_ratio = float((ic_df > 0).mean())
            ic_t_stat = float(
                ic_mean / (ic_std / np.sqrt(len(ic_df)))
            ) if ic_std > 0 else 0.0

            ic_results.append({
                "factor": factor,
                "forward_period": forward_col,
                "ic_mean": round(ic_mean, 6),
                "ic_std": round(ic_std, 6),
                "ic_ir": round(ic_ir, 4),
                "ic_positive_ratio": round(ic_positive_ratio, 4),
                "ic_t_stat": round(ic_t_stat, 4),
                "n_dates": int(len(ic_df)),
            })

        results[forward_col] = ic_results

    return results


def benchmark_ic(
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Optional[List[str]] = None,
    runs: int = 3,
) -> Dict[str, Any]:
    """对比向量化 IC vs 循环 IC 的性能与正确性"""
    # 正确性
    vec_result = vectorized_ic(factor_df, forward_returns, factor_names)
    loop_result = loop_ic(factor_df, forward_returns, factor_names)

    # 性能
    vec_times = []
    loop_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        vectorized_ic(factor_df, forward_returns, factor_names)
        vec_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        loop_ic(factor_df, forward_returns, factor_names)
        loop_times.append(time.perf_counter() - t0)

    # 比较 IC 均值（验证正确性）
    ic_diffs = []
    for period in vec_result:
        for v_item, l_item in zip(
            sorted(vec_result[period], key=lambda x: x['factor']),
            sorted(loop_result[period], key=lambda x: x['factor']),
        ):
            ic_diffs.append(abs(v_item['ic_mean'] - l_item['ic_mean']))

    return {
        "vectorized": {
            "median_time": float(np.median(vec_times)),
            "min_time": float(np.min(vec_times)),
            "times": vec_times,
        },
        "loop": {
            "median_time": float(np.median(loop_times)),
            "min_time": float(np.min(loop_times)),
            "times": loop_times,
        },
        "speedup": float(np.median(loop_times) / np.median(vec_times)) if np.median(vec_times) > 0 else 0,
        "max_ic_diff": float(max(ic_diffs)) if ic_diffs else 0.0,
        "mean_ic_diff": float(np.mean(ic_diffs)) if ic_diffs else 0.0,
        "n_factors": len(vec_result.get('ret_forward_5d', [])),
    }
