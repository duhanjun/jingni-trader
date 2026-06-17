"""
可组合因子表达式引擎（quant_opt.expression_engine）

详细使用参见 engine.py。本包通过 __init__ 暴露核心 API。
"""
from .engine import (
    Expression,
    Evaluator,
    F,
    Feature,
    Ref,
    Delta,
    TsMean,
    TsStd,
    TsRank,
    Rank,
    Zscore,
    Mean,
    builtin_a_share_factors,
)

__all__ = [
    "Expression",
    "Evaluator",
    "F",
    "Feature",
    "Ref",
    "Delta",
    "TsMean",
    "TsStd",
    "TsRank",
    "Rank",
    "Zscore",
    "Mean",
    "builtin_a_share_factors",
]
