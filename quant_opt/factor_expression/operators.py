"""
Built-in operators for the factor expression engine.

The engine is **extensible**: callers may register new operators via
``engine.register(name, fn, group)``.  ``group`` is either ``"ts"``
(time-series, applied per ``code``), ``"cs"`` (cross-section, applied
per ``date``) or ``"el"`` (element-wise, applied to the whole frame).

The shipped operators mirror the most commonly used subset of Qlib
(``qlib.data.ops``) and AKQuant (``akquant.factor``) so existing
expressions can be ported almost verbatim.

Categories
----------

* **Time-series** (``ts``):  Ref, Delta, MA, EMA, Std, Sum, Min, Max,
  TsRank, Product, IfElse
* **Cross-section** (``cs``):  Rank, Demean, Zscore, CsMean, CsStd,
  CsMin, CsMax
* **Element-wise** (``el``): Abs, Log, Sign, SignPower, Power, Sqrt,
  Clip, Where, Add, Sub, Mul, Div, Neg
"""
from __future__ import annotations

from typing import Callable, Dict, Set

import numpy as np
import pandas as pd


# ── Type aliases ─────────────────────────────────────────────────────


OperatorFn = Callable[..., "pd.Series | float"]
OPS: Dict[str, OperatorFn] = {}
GROUPS: Dict[str, str] = {}  # name -> "ts" | "cs" | "el"
ALIASES: Dict[str, str] = {}  # alias -> canonical name


def _register(name: str, group: str, fn: OperatorFn, *aliases: str) -> None:
    if name in OPS:
        return
    OPS[name] = fn
    GROUPS[name] = group
    for a in aliases:
        if a in ALIASES:
            continue
        ALIASES[a] = name


_DEFAULT_REGISTERED = False


def register_default_operators() -> None:
    """Populate the global registry with the default operator set.

    Safe to call multiple times — already-registered operators are left
    untouched.
    """
    global _DEFAULT_REGISTERED
    if _DEFAULT_REGISTERED:
        return
    _DEFAULT_REGISTERED = True


# ── Element-wise helpers ─────────────────────────────────────────────


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if np.isscalar(x):
        return pd.Series([x])
    return pd.Series(x)


# ── Element-wise operators ───────────────────────────────────────────


def op_abs(x):
    return _to_series(x).abs()


def op_log(x):
    s = _to_series(x)
    return np.log(s.replace(0, np.nan))


def op_sign(x):
    return np.sign(_to_series(x))


def op_sqrt(x):
    return np.sqrt(_to_series(x).abs())


def op_power(x, p):
    return _to_series(x) ** _to_series(p)


def op_clip(x, lo, hi):
    return _to_series(x).clip(lower=_to_series(lo), upper=_to_series(hi))


def op_where(cond, a, b):
    c = _to_series(cond).astype(bool)
    A = _to_series(a)
    B = _to_series(b)
    return pd.Series(np.where(c, A, B), index=A.index)


def op_neg(x):
    return -_to_series(x)


# ── Time-series operators (per code) ────────────────────────────────


def _ts_grouped(series: pd.Series, code: pd.Series) -> pd.Series:
    return series.groupby(code)


def op_ref(series, d: float, code):
    return _ts_grouped(series, code).shift(int(d))


def op_delta(series, d: float, code):
    return series - op_ref(series, d, code)


def _rolling_reset(series_groupby, d: int, min_periods: int):
    """rolling() returns a MultiIndex; flatten to the original index."""
    return series_groupby.rolling(d, min_periods=min_periods).mean().reset_index(level=0, drop=True)


def op_ma(series, d: float, code):
    return _rolling_reset(_ts_grouped(series, code), int(d), 1)


def op_ema(series, d: float, code):
    # EWM is per-column; we group via apply to keep alignment.
    return _ts_grouped(series, code).apply(
        lambda s: s.ewm(span=int(d), adjust=False, min_periods=1).mean()
    ).reset_index(level=0, drop=True)


def op_std(series, d: float, code):
    return _ts_grouped(series, code).rolling(int(d), min_periods=2).std().reset_index(level=0, drop=True)


def op_sum(series, d: float, code):
    return _ts_grouped(series, code).rolling(int(d), min_periods=1).sum().reset_index(level=0, drop=True)


def op_min(series, d: float, code):
    return _ts_grouped(series, code).rolling(int(d), min_periods=1).min().reset_index(level=0, drop=True)


def op_max(series, d: float, code):
    return _ts_grouped(series, code).rolling(int(d), min_periods=1).max().reset_index(level=0, drop=True)


def op_ts_rank(series, d: float, code):
    def _rank(s):
        return s.rank(pct=True)

    return _ts_grouped(series, code).rolling(int(d), min_periods=1).apply(_rank, raw=False).reset_index(level=0, drop=True)


def op_product(series, d: float, code):
    return _ts_grouped(series, code).rolling(int(d), min_periods=1).apply(np.prod, raw=True).reset_index(level=0, drop=True)


def op_ifelse(cond, a, b, code):
    # like where, but condition is a *time-series* (per code) — we still
    # apply it element-wise because we already have aligned series.
    return op_where(cond, a, b)


# ── Cross-sectional operators (per date) ─────────────────────────────


def op_rank(series, date):
    return series.groupby(date).rank(pct=True)


def op_demean(series, date):
    return series - series.groupby(date).transform("mean")


def op_zscore(series, date):
    g = series.groupby(date)
    return (series - g.transform("mean")) / g.transform("std")


def op_cs_mean(series, date):
    return series.groupby(date).transform("mean")


def op_cs_std(series, date):
    return series.groupby(date).transform("std")


def op_cs_min(series, date):
    return series.groupby(date).transform("min")


def op_cs_max(series, date):
    return series.groupby(date).transform("max")


# ── Public registration ───────────────────────────────────────────────


def register_default_operators() -> None:
    """Populate the global registry with the default operator set."""

    # element-wise
    _register("Abs", "el", op_abs, "abs")
    _register("Log", "el", op_log, "log", "Ln")
    _register("Sign", "el", op_sign, "sign")
    _register("Sqrt", "el", op_sqrt, "sqrt")
    _register("Power", "el", op_power, "Pow")
    _register("Clip", "el", op_clip)
    _register("Where", "el", op_where)
    _register("Neg", "el", op_neg, "Negate")

    # time-series
    _register("Ref", "ts", op_ref, "Delay", "Lag")
    _register("Delta", "ts", op_delta)
    _register("MA", "ts", op_ma, "Mean", "Ts_Mean", "TsMean")
    _register("EMA", "ts", op_ema, "Ema")
    _register("Std", "ts", op_std, "Ts_Std", "TsStd")
    _register("Sum", "ts", op_sum, "Ts_Sum", "TsSum")
    _register("Min", "ts", op_min, "Ts_Min", "TsMin")
    _register("Max", "ts", op_max, "Ts_Max", "TsMax")
    _register("TsRank", "ts", op_ts_rank, "Ts_Rank")
    _register("Product", "ts", op_product, "Prod")
    _register("IfElse", "ts", op_ifelse)

    # cross-section
    _register("Rank", "cs", op_rank, "CsRank", "Cs_Rank")
    _register("Demean", "cs", op_demean, "CsDemean")
    _register("Zscore", "cs", op_zscore, "CsZscore", "ZScore")
    _register("CsMean", "cs", op_cs_mean, "Mean")
    _register("CsStd", "cs", op_cs_std)
    _register("CsMin", "cs", op_cs_min)
    _register("CsMax", "cs", op_cs_max)


def known_operators() -> Set[str]:
    return set(OPS) | set(ALIASES)
