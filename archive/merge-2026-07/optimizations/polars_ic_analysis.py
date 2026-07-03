"""
Polars 向量化 IC 分析模块

借鉴来源：
- AKQuant: Polars 驱动的高性能因子计算引擎
- Microsoft Qlib: 向量化因子 IC 分析

优化点：
原 factor-engine/engine.py 的 _calc_ic 方法通过 Python for 循环逐日计算 IC：
    for dt in dates:                      # Python 循环，慢
        cross = data[data['date'] == dt]  # pandas 布尔索引，每次拷贝
        ic, _ = stats.spearmanr(...)      # 逐日调用 scipy

本模块使用 Polars 的 group_by + corr 实现全向量化计算：
- 分组在 Rust 层完成，多线程并行
- Spearman IC = 对因子值和收益率分别排名后计算 Pearson 相关
- 单次表达式完成所有日期的 IC 计算，无 Python 循环
"""
from typing import Optional, List, Dict, Any
import numpy as np
import polars as pl


def calc_ic_series_polars(
    data: pl.DataFrame,
    factor_col: str,
    forward_col: str,
    ic_type: str = "spearman",
    min_samples: int = 10,
) -> pl.DataFrame:
    """
    向量化计算单因子的 IC 时间序列

    参数:
        data: 包含 date, factor_col, forward_col 的 Polars DataFrame
        factor_col: 因子列名
        forward_col: 未来收益率列名
        ic_type: "spearman" 或 "pearson"
        min_samples: 每个截面最少样本数，低于此值跳过

    返回:
        Polars DataFrame，列为 [date, ic, n]
    """
    df = data.select(["date", factor_col, forward_col]).drop_nulls()

    if df.height == 0:
        return pl.DataFrame(schema={"date": pl.Date, "ic": pl.Float64, "n": pl.Int64})

    if ic_type == "spearman":
        # Spearman = Pearson(rank(x), rank(y))
        # 在每个日期截面内排名，再计算 Pearson 相关
        df = df.with_columns(
            pl.col(factor_col).rank(method="average").over("date").alias("_f_rank"),
            pl.col(forward_col).rank(method="average").over("date").alias("_r_rank"),
        )
        ic_df = (
            df.group_by("date")
            .agg(
                pl.corr("_f_rank", "_r_rank").alias("ic"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_samples)
            .filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
        )
    else:
        # Pearson IC
        ic_df = (
            df.group_by("date")
            .agg(
                pl.corr(factor_col, forward_col).alias("ic"),
                pl.len().alias("n"),
            )
            .filter(pl.col("n") >= min_samples)
            .filter(pl.col("ic").is_not_null() & pl.col("ic").is_not_nan())
        )

    return ic_df.sort("date")


def ic_summary_stats(ic_series: pl.DataFrame) -> Dict[str, float]:
    """
    计算 IC 序列的统计摘要

    返回: ic_mean, ic_std, ic_ir, ic_positive_ratio, ic_t_stat
    """
    if ic_series.height == 0:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_t_stat": 0.0,
        }

    ic_col = ic_series["ic"]
    n = ic_col.len()
    ic_mean = ic_col.mean()
    ic_std = ic_col.std()
    ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else 0.0
    ic_positive_ratio = (ic_col > 0).sum() / n if n > 0 else 0.0
    ic_t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std and ic_std > 0 and n > 0 else 0.0

    return {
        "ic_mean": round(float(ic_mean), 6),
        "ic_std": round(float(ic_std), 6),
        "ic_ir": round(float(ic_ir), 4),
        "ic_positive_ratio": round(float(ic_positive_ratio), 4),
        "ic_t_stat": round(float(ic_t_stat), 4),
    }


def batch_ic_analysis_polars(
    data: pl.DataFrame,
    factor_names: List[str],
    forward_col: str,
    ic_type: str = "spearman",
    min_samples: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    批量向量化 IC 分析：对多个因子一次性计算 IC

    借鉴 Qlib 的批量因子评估思路，避免重复的日期分组开销。

    参数:
        data: Polars DataFrame
        factor_names: 因子名列表
        forward_col: 未来收益率列名
        ic_type: "spearman" 或 "pearson"

    返回:
        {factor_name: {ic_stats, ic_series}}
    """
    results: Dict[str, Dict[str, Any]] = {}
    for factor in factor_names:
        if factor not in data.columns:
            continue
        ic_series = calc_ic_series_polars(data, factor, forward_col, ic_type, min_samples)
        stats = ic_summary_stats(ic_series)
        results[factor] = {
            "ic_stats": stats,
            "ic_series": ic_series,
        }
    return results
