"""向量化回测引擎"""
from .engine import VectorizedBacktestEngine, VectorizedBacktestResult, run_vectorized_adapter

__all__ = ["VectorizedBacktestEngine", "VectorizedBacktestResult", "run_vectorized_adapter"]