"""
Pre-defined alpha factor formulas (Alpha158-style catalog)

Borrowed idea: Qlib ships with a pre-built catalog of 158 / 360
technical alpha factors. We expose a smaller but representative
catalog (~30 formulas) that exercises all major operator families.

The catalog is a list of (name, formula) tuples and can be
extended in user code. Each entry has been picked to be
intuitively meaningful for A-share quant researchers.
"""
from __future__ import annotations

from typing import List, Tuple

# (display_name, formula_string)
ALPHA_CATALOG: List[Tuple[str, str]] = [
    # ---- momentum / reversal -----------------------------------------------
    ("MOM_5",  "Ref($close, 5) / $close - 1"),
    ("MOM_20", "Ref($close, 20) / $close - 1"),
    ("REV_5",  "-($close / Ref($close, 5) - 1)"),
    ("REV_20", "-($close / Ref($close, 20) - 1)"),

    # ---- moving average convergence ----------------------------------------
    ("MA_RATIO_5_20",  "Mean($close, 5) / Mean($close, 20) - 1"),
    ("MA_RATIO_5_60",  "Mean($close, 5) / Mean($close, 60) - 1"),
    ("BIAS_20",        "($close - Mean($close, 20)) / Mean($close, 20)"),
    ("BIAS_60",        "($close - Mean($close, 60)) / Mean($close, 60)"),

    # ---- volatility --------------------------------------------------------
    ("VOL_20",         "Std($close, 20) / Mean($close, 20)"),
    ("VOL_60",         "Std($close, 60) / Mean($close, 60)"),
    ("HIGH_LOW_20",    "(Max($high, 20) - Min($low, 20)) / Mean($close, 20)"),

    # ---- volume ------------------------------------------------------------
    ("VOL_RATIO_5_20", "Mean($volume, 5) / Mean($volume, 20) - 1"),
    ("AMOUNT_20",      "Mean($amount, 20)"),
    ("AMOUNT_RATIO_5_20", "Mean($amount, 5) / Mean($amount, 20) - 1"),
    ("PV_CORR_10",     "Corr($close, $volume, 10)"),

    # ---- EMA / trend -------------------------------------------------------
    ("EMA_RATIO_5_20", "EMA($close, 5) / EMA($close, 20) - 1"),
    ("EMA_RATIO_10_60", "EMA($close, 10) / EMA($close, 60) - 1"),
    ("TSRANK_CLOSE_20", "TsRank($close, 20)"),

    # ---- price / range / position -----------------------------------------
    ("DAILY_RANGE",    "($high - $low) / $close"),
    ("CLOSE_POSITION", "($close - $low) / ($high - $low + 1e-12)"),

    # ---- cross-sectional ----------------------------------------------------
    ("RANK_RET_1",     "Rank($close / Ref($close, 1) - 1)"),
    ("RANK_AMOUNT_20", "Rank(Mean($amount, 20))"),
    ("ZSCORE_MOM_20",  "Zscore(Ref($close, 20) / $close - 1)"),
    ("SCALE_VOL_20",   "Scale(Std($close, 20))"),

    # ---- log returns / signed power ----------------------------------------
    ("LOG_RET_1",      "Log($close / Ref($close, 1))"),
    ("LOG_RET_5",      "Log($close / Ref($close, 5))"),
    ("SIGNED_POW_RET_5", "SignedPower($close / Ref($close, 5) - 1, 0.5)"),

    # ---- longer-term reversals --------------------------------------------
    ("REV_60",         "-($close / Ref($close, 60) - 1)"),
    ("MOM_60",         "Ref($close, 60) / $close - 1"),

    # ---- mean-reversion of high/low ---------------------------------------
    ("CLOSE_VS_HIGH_20", "$close / Max($high, 20) - 1"),
    ("CLOSE_VS_LOW_20",  "$close / Min($low, 20) - 1"),
]


def get_catalog() -> List[Tuple[str, str]]:
    """Return a copy of the alpha catalog."""
    return list(ALPHA_CATALOG)


def get_formula(name: str) -> str:
    """Look up a single formula by display name."""
    for n, f in ALPHA_CATALOG:
        if n == name:
            return f
    raise KeyError(f"Unknown factor: {name}")
