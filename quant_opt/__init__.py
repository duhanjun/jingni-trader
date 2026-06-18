"""
Top-level package for the ``quant_opt`` validation work.
"""
from .expression_engine.expr_engine import (
    BinaryOp,
    ExpressionEvaluator,
    FieldRef,
    FuncCall,
    Number,
    OperatorRegistry,
    UnaryOp,
    DEFAULT_EVALUATOR,
    evaluate_formula,
    list_operators,
    parse_formula,
    tokenize,
)
from .expression_engine.alpha_catalog import ALPHA_CATALOG, get_catalog, get_formula
from .ic_analysis.vectorized_ic import (
    batch_ic,
    compute_ic_series,
    rank_ic_decay,
    summarize_ic,
)
from .cv_splitter.purged_cv import (
    CVSplit,
    LeakageReport,
    PurgedFold,
    PurgedKFold,
    TimeSeriesCV,
    leakage_check,
)

__all__ = [
    # expression engine
    "BinaryOp", "ExpressionEvaluator", "FieldRef", "FuncCall", "Number",
    "OperatorRegistry", "UnaryOp", "DEFAULT_EVALUATOR", "evaluate_formula",
    "list_operators", "parse_formula", "tokenize",
    # catalog
    "ALPHA_CATALOG", "get_catalog", "get_formula",
    # ic
    "batch_ic", "compute_ic_series", "rank_ic_decay", "summarize_ic",
    # cv + leakage
    "CVSplit", "LeakageReport", "PurgedFold", "PurgedKFold", "TimeSeriesCV",
    "leakage_check",
]
