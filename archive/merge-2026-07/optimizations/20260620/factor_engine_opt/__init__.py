"""
因子引擎优化模块（20260620）

借鉴来源：
- AlphaPurify (Polars 向量化，2026-05 发布)：通过向量化 + 多进程大幅加速因子分析
- Microsoft Qlib (高性能 DataServer，比 Pandas 快 10x)：向量化表达式引擎

本模块提供以下优化：
- vectorized_ic: 向量化 IC 分析（替代逐日 Python 循环）
- vectorized_neutralize: 向量化因子中性化（替代逐日 sklearn 调用）
- preprocessing: 因子预处理（去极值 + 标准化）
"""
from .vectorized_ic import (
    calc_ic_series_vectorized,
    calc_ic_stats_vectorized,
    ic_analysis_vectorized,
)
from .vectorized_neutralize import neutralize_vectorized
from .preprocessing import (
    winsorize_mad,
    winsorize_quantile,
    standardize_zscore,
    preprocess_factor,
)

__all__ = [
    "calc_ic_series_vectorized",
    "calc_ic_stats_vectorized",
    "ic_analysis_vectorized",
    "neutralize_vectorized",
    "winsorize_mad",
    "winsorize_quantile",
    "standardize_zscore",
    "preprocess_factor",
]
