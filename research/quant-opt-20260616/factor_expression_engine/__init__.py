"""
因子表达式引擎 (Factor Expression Engine)

借鉴自开源项目：
- Microsoft Qlib: qlib/data/ops.py 的 Expression Engine
- AKQuant: akquant.factor.FactorEngine 的 Alpha101 风格 DSL

设计目标：
- 提供类似 WorldQuant Alpha101 的字符串 DSL，用户用公式即可定义因子
- 无重依赖：仅依赖 pandas / numpy，便于在 jingni-trader 内嵌
- 自动按 (code, date) 区分时间序列算子和横截面算子
- 算子注册表可扩展，便于新增自定义算子
"""

from .expression_engine import FactorExpressionEngine, compile_formula
from .operators import OPERATORS, register_operator

__all__ = [
    "FactorExpressionEngine",
    "compile_formula",
    "OPERATORS",
    "register_operator",
]
