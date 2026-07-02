"""
因子表达式引擎模块
借鉴来源: quant-stream 表达式语言 + Qlib Alpha158
提供声明式因子定义能力，支持 RANK(DELTA($close, 5)) 等嵌套表达式
"""
from .operators import OperatorRegistry
from .parser import FactorExpressionParser
from .engine import FactorExpressionEngine

__all__ = ["OperatorRegistry", "FactorExpressionParser", "FactorExpressionEngine"]