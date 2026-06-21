"""factor_expr 包: 因子表达式引擎与注册机制。"""
from .expression_engine import ExpressionEngine, ExpressionParser
from .factor_registry import FactorMeta, FactorRegistry
from .builtin_factors import register_builtins

__all__ = [
    "ExpressionEngine",
    "ExpressionParser",
    "FactorMeta",
    "FactorRegistry",
    "register_builtins",
]
