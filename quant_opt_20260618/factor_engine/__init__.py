"""
因子引擎验证模块包

借鉴来源：
  - Qlib 表达式引擎 / Alpha158 / Alpha360 数据集
  - AKQuant FactorEngine (Polars + 算子)
  - WorldQuant Alpha101 (公式风格)
  - FactorEngine 论文 (arXiv:2603.16365v1)

本包内的代码仅作为优化方向的 PoC 与对照测试使用，
所有新代码放在 feat/quant-opt-20260618 分支的
quant_opt_20260618/ 目录中，不修改 main 分支现有实现。
"""
from .expression_engine import (
    calc_factor,
    calc_factors,
    list_operators,
    parse_formula,
    OPERATORS,
    COLUMN_ALIASES,
    OperatorSpec,
    FormulaError,
)

__all__ = [
    "calc_factor",
    "calc_factors",
    "list_operators",
    "parse_formula",
    "OPERATORS",
    "COLUMN_ALIASES",
    "OperatorSpec",
    "FormulaError",
]
