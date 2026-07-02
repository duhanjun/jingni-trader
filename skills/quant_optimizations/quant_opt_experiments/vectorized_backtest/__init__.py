"""Vectorized Backtest Engine 入口"""
from .engine import (
    vectorized_backtest_single,
    vectorized_backtest_multi,
    VectorizedBacktestResult,
    calc_metrics,
    ma_cross_signals,
    rank_topk_signals,
)

__all__ = [
    "vectorized_backtest_single",
    "vectorized_backtest_multi",
    "VectorizedBacktestResult",
    "calc_metrics",
    "ma_cross_signals",
    "rank_topk_signals",
]