"""
Expression DSL - Built-in Operators
====================================

Set of stock-style operators exposed by the DSL.  Each operator is a
plain function that takes a *per-stock* ``pandas.Series`` and returns a
``pandas.Series``.  The :class:`Evaluator` is responsible for
``groupby(code)`` routing so operators only need to worry about a
single time series.

The vocabulary is a pragmatic subset of the Qlib operator set, tuned
to A-share alpha factors:

- 时间维度：``Ref``, ``Delta``, ``Mean``, ``Std``, ``Sum``, ``EwmMean``
- 截面维度：``Rank`` (cross-section), ``Scale`` (CS scale)
- 关系：``Corr``, ``Cov``
- 表达式：``Sign``, ``Abs``, ``Log``, ``SignPower``
- A 股习惯：``PctChange``

Custom operators can be registered via :func:`register_operator`.
"""
from typing import Callable, Dict

import numpy as np
import pandas as pd

# All operators take pd.Series and return pd.Series.  In addition,
# operators marked ``needs_window=True`` validate that their last
# argument is a positive integer window size.


class _WindowError(ValueError):
    pass


# --- core time-series operators --------------------------------------------


def Ref(x: pd.Series, n: int) -> pd.Series:
    """``Ref(x, n)`` — value of *x* n periods ago."""
    return x.shift(int(n))


def Delta(x: pd.Series, n: int) -> pd.Series:
    """``Delta(x, n)`` — ``x - Ref(x, n)``."""
    return x - x.shift(int(n))


def Mean(x: pd.Series, n: int) -> pd.Series:
    """``Mean(x, n)`` — rolling mean over the last *n* periods (inclusive)."""
    n = int(n)
    if n <= 0:
        raise _WindowError(f"Mean 窗口必须为正整数: {n}")
    return x.rolling(window=n, min_periods=_min_periods(n, 1)).mean()


def Std(x: pd.Series, n: int) -> pd.Series:
    """``Std(x, n)`` — rolling population standard deviation."""
    n = int(n)
    if n <= 0:
        raise _WindowError(f"Std 窗口必须为正整数: {n}")
    return x.rolling(window=n, min_periods=_min_periods(n, 2)).std()


def Sum(x: pd.Series, n: int) -> pd.Series:
    n = int(n)
    if n <= 0:
        raise _WindowError(f"Sum 窗口必须为正整数: {n}")
    return x.rolling(window=n, min_periods=_min_periods(n, 1)).sum()


def EwmMean(x: pd.Series, n: int, half_life: float = None) -> pd.Series:
    """Exponentially weighted moving mean.

    ``n`` is used to derive ``halflife`` when ``half_life`` is not given.
    The relation is ``alpha = 1 - exp(-ln(2) / halflife)``.
    """
    n = int(n)
    if half_life is None:
        if n <= 0:
            raise _WindowError("EwmMean 窗口必须为正整数")
        half_life = (n - 1) / np.log(2)
    return x.ewm(halflife=half_life, adjust=False).mean()


def PctChange(x: pd.Series, n: int) -> pd.Series:
    n = int(n)
    if n <= 0:
        raise _WindowError("PctChange 窗口必须为正整数")
    return x.pct_change(periods=n)


# --- element-wise operators ------------------------------------------------


def Sign(x: pd.Series) -> pd.Series:
    return np.sign(x)


def Abs(x: pd.Series) -> pd.Series:
    return x.abs()


def Log(x: pd.Series) -> pd.Series:
    return np.log(x.where(x > 0))


def SignPower(x: pd.Series, p: float) -> pd.Series:
    return np.sign(x) * (x.abs() ** p)


# --- bivariate rolling operators ------------------------------------------


def Corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    n = int(n)
    if n <= 1:
        raise _WindowError("Corr 窗口必须 >= 2")
    return x.rolling(n, min_periods=_min_periods(n, 2)).corr(y)


def Cov(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    n = int(n)
    if n <= 1:
        raise _WindowError("Cov 窗口必须 >= 2")
    return x.rolling(n, min_periods=_min_periods(n, 2)).cov(y)


def _min_periods(n, lower: int) -> int:
    """Compute a valid ``min_periods`` for ``x.rolling(window=n, ...)``.

    Rules:
    - ``1 <= min_periods <= window``
    - Use a fraction of the window (n // 2) so the rolling statistic
      doesn't need a fully-warm window.
    - Never go below ``lower`` (1 for mean-like, 2 for std/corr).
    - Never exceed ``n`` (the window itself).
    - Always returns ``int`` (caller may pass a float from the DSL).
    """
    n = int(n)
    lower = int(min(lower, n))
    return int(max(lower, min(n, max(1, n // 2))))


# --- cross-section (operate on the whole frame) --------------------------


def _cross_section_required(name: str):
    def _raise(*_args, **_kwargs):
        raise RuntimeError(
            f"操作符 {name!r} 必须在截面上下文使用，请通过 Evaluator(group_key=...) 调用"
        )
    return _raise


# --- registry --------------------------------------------------------------


_BUILTIN_OPERATORS: Dict[str, Callable] = {
    "Ref": Ref,
    "Delta": Delta,
    "Mean": Mean,
    "Std": Std,
    "Sum": Sum,
    "EwmMean": EwmMean,
    "PctChange": PctChange,
    "Sign": Sign,
    "Abs": Abs,
    "Log": Log,
    "SignPower": SignPower,
    "Corr": Corr,
    "Cov": Cov,
    "Rank": _cross_section_required("Rank"),
    "Scale": _cross_section_required("Scale"),
}


def register_operator(name: str, func: Callable) -> None:
    """Register a custom operator at runtime."""
    if not name.isidentifier():
        raise ValueError(f"操作符名必须是合法标识符: {name!r}")
    _BUILTIN_OPERATORS[name] = func


def get_operator(name: str) -> Callable:
    if name not in _BUILTIN_OPERATORS:
        raise KeyError(f"未知操作符: {name!r}")
    return _BUILTIN_OPERATORS[name]


def list_operators() -> list:
    return sorted(_BUILTIN_OPERATORS)
