"""向量化回测引擎（quant_opt.vectorized_backtest）"""
from .engine import (
    PortfolioWeight,
    signals_to_weights,
    vectorized_backtest,
)

__all__ = ["PortfolioWeight", "signals_to_weights", "vectorized_backtest"]
