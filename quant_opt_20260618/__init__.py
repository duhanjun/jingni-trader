"""
quant_opt_20260618
==================

Validation package for the 2026-06-18 quant optimization spike.

Modules
-------
- :mod:`quant_opt_20260618.expression_dsl`  : Qlib-inspired factor
  expression DSL (tokenizer / parser / operators / evaluator).
- :mod:`quant_opt_20260618.ic_vectorized`    : Vectorized Pearson/Rank
  IC computation (drop-in replacement for the legacy loop).
- :mod:`quant_opt_20260618.factor_validator` : IC + bootstrap-based
  rule significance test (Jesse-inspired).

All code in this package is **experimental** and lives on the
``feat/quant-opt-20260618`` branch.  It does **not** modify main.
"""
from .ic_vectorized import (
    ic_series_pearson,
    ic_series_spearman,
    ic_summary,
    ic_analysis_batch,
)
from .factor_validator import validate_factor, FactorVerdict

__all__ = [
    "ic_series_pearson",
    "ic_series_spearman",
    "ic_summary",
    "ic_analysis_batch",
    "validate_factor",
    "FactorVerdict",
]
