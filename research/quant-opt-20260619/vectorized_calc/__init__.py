"""
向量化因子计算 - jingni-trader 优化方向

借鉴 vectorbt (https://github.com/polakowo/vectorbt) 的核心思想：
- 避免 Python 级别的 for 循环
- 用 Numba JIT 把"逐只股票"的循环编译为机器码
- 用 2D ndarray 一次性表达"股票 × 时间"矩阵
- 利用 NumPy broadcasting 实现截面操作

同时借鉴 jingni-trader 现有的 factor_calculator 设计（继承 BaseFactorCalculator），
保持 API 兼容。
"""
from .vector_ops import (
    numba_mean,
    numba_std,
    numba_ma,
    numba_ema,
    numba_rsi,
    numba_rolling_corr,
    numba_cross_section_rank,
)
from .vectorized_engine import VectorizedFactorEngine, VectorizedFactorCalculator

__all__ = [
    "numba_mean", "numba_std", "numba_ma", "numba_ema", "numba_rsi",
    "numba_rolling_corr", "numba_cross_section_rank",
    "VectorizedFactorEngine", "VectorizedFactorCalculator",
]
