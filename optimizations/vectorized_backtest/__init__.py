"""vectorized_backtest 包: 向量化回测引擎。"""
from .vectorized_engine import (
    BacktestResult,
    CostModel,
    VectorizedBacktester,
    compute_metrics,
    compute_limit_flags,
    detect_board,
)

__all__ = [
    "BacktestResult",
    "CostModel",
    "VectorizedBacktester",
    "compute_metrics",
    "compute_limit_flags",
    "detect_board",
]
