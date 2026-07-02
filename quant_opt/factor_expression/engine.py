"""
Factor engine: walks the AST and materialises results.

The engine is **stateless** apart from the operator registry and a
small **sub-expression cache** (keyed by the AST node id).  This is
the same pattern that Qlib uses internally (``FeatureRowName`` nodes
share a common cache so ``MA(MA(Close, 5), 5)`` does not re-evaluate
the inner MA when it appears in two expressions).

Public API
----------

>>> from quant_opt.factor_expression import FactorEngine
>>> engine = FactorEngine()
>>> out = engine.calc(df, "Rank(MA($close, 20) - MA($close, 5))")
>>> out = engine.calc_many(df, ["MA($close, 20)", "Rank(Delta($close, 5))"], names=["fast", "slow"])
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .operators import GROUPS, OPS, ALIASES, register_default_operators
from .parser import Node, parse


class ExpressionError(ValueError):
    """Raised for any semantic error in a factor expression."""


class FactorEngine:
    """Stateless factor expression evaluator.

    Parameters
    ----------
    code_col, date_col : str
        Names of the columns identifying the panel.  Defaults are
        ``"code"`` and ``"date"``.
    cache : bool
        If ``True`` (default) reuse intermediate results for the same
        AST node within a single :py:meth:`calc` call.
    """

    def __init__(
        self,
        code_col: str = "code",
        date_col: str = "date",
        cache: bool = True,
    ) -> None:
        self.code_col = code_col
        self.date_col = date_col
        self.cache_enabled = cache
        register_default_operators()
        # user-registered operators (added on top of defaults)
        self._user_ops: Dict[str, Callable] = {}
        self._user_groups: Dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────

    def register(self, name: str, group: str, fn: Callable) -> None:
        """Register a custom operator at runtime."""
        if group not in {"ts", "cs", "el"}:
            raise ValueError(f"group must be ts/cs/el, got {group!r}")
        if name in self._user_ops:
            raise ValueError(f"operator {name!r} already registered")
        self._user_ops[name] = fn
        self._user_groups[name] = group

    def calc(
        self,
        df: pd.DataFrame,
        expression: str,
        name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Evaluate a single expression.  Returns a copy of ``df`` with
        one extra column ``name`` (default: ``f"f_0"``)."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df.empty:
            raise ExpressionError("input DataFrame is empty")
        for col in (self.code_col, self.date_col):
            if col not in df.columns:
                raise ExpressionError(f"column {col!r} not in input frame")

        ast = parse(expression)
        self._cache: Dict[int, pd.Series] = {}
        try:
            result = self._eval(ast, df)
        finally:
            self._cache = {}

        if not isinstance(result, pd.Series):
            # scalar — broadcast
            result = pd.Series(result, index=df.index)

        # align order
        result = result.reindex(df.index)
        out = df.copy()
        out[name or "f_0"] = result
        return out

    def calc_many(
        self,
        df: pd.DataFrame,
        expressions: Sequence[str],
        names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Evaluate many expressions, reusing intermediates where the
        same AST node id is encountered."""
        if names is None:
            names = [f"f_{i}" for i in range(len(expressions))]
        if len(names) != len(expressions):
            raise ValueError("expressions and names length mismatch")
        out = df.copy()
        for expr, nm in zip(expressions, names):
            tmp = self.calc(df, expr, name=nm)
            out[nm] = tmp[nm]
        return out

    # ── AST walker ────────────────────────────────────────────────────

    def _eval(self, node: Node, df: pd.DataFrame) -> Any:
        if self.cache_enabled and id(node) in self._cache:
            return self._cache[id(node)]

        if node.op is None:
            value = self._eval_leaf(node, df)
        else:
            value = self._eval_op(node, df)

        if self.cache_enabled and isinstance(value, pd.Series):
            self._cache[id(node)] = value
        return value

    # ── Leaves ────────────────────────────────────────────────────────

    def _eval_leaf(self, node: Node, df: pd.DataFrame) -> Any:
        if node.value is not None:
            return float(node.value)
        if node.name is not None:
            col = node.name.lstrip("$")
            if col not in df.columns:
                # try case-insensitive
                ci_map = {c.lower(): c for c in df.columns}
                if col.lower() in ci_map:
                    col = ci_map[col.lower()]
                else:
                    raise ExpressionError(
                        f"column {col!r} not in input frame "
                        f"(have: {list(df.columns)})"
                    )
            return df[col]
        raise ExpressionError(f"empty AST node: {node}")

    # ── Operator dispatch ────────────────────────────────────────────

    def _eval_op(self, node: Node, df: pd.DataFrame) -> Any:
        op = node.op
        args = node.args

        # ── unary (must come BEFORE the binary check) ───────────────
        if op == "-" and len(args) == 1:
            return -self._to_series(self._eval(args[0], df))
        if op == "+" and len(args) == 1:
            return self._to_series(self._eval(args[0], df))

        # ── arithmetic ────────────────────────────────────────────────
        if op == "+":
            return self._add(self._eval(args[0], df), self._eval(args[1], df))
        if op == "-":
            return self._sub(self._eval(args[0], df), self._eval(args[1], df))
        if op == "*":
            return self._mul(self._eval(args[0], df), self._eval(args[1], df))
        if op == "/":
            return self._div(self._eval(args[0], df), self._eval(args[1], df))

        # ── named function ────────────────────────────────────────────
        canonical = ALIASES.get(op, op)
        if canonical in self._user_ops:
            fn = self._user_ops[canonical]
            group = self._user_groups[canonical]
        elif canonical in OPS:
            fn = OPS[canonical]
            group = GROUPS[canonical]
        else:
            raise ExpressionError(f"unknown operator {op!r}")

        # evaluate all argument expressions
        if group == "ts":
            vals = [self._eval(a, df) for a in args]
            return self._apply_ts(canonical, fn, vals, df)
        if group == "cs":
            vals = [self._eval(a, df) for a in args]
            return self._apply_cs(canonical, fn, vals, df)
        # element-wise
        vals = [self._eval(a, df) for a in args]
        return self._apply_el(canonical, fn, vals)

    # ── Application helpers ──────────────────────────────────────────

    @staticmethod
    def _to_series(x):
        """Wrap ``x`` in a Series *only* if it is not already a Series
        and not a numpy scalar / Python scalar.  Returning the scalar
        unchanged lets pandas broadcast it naturally against the
        aligned Series on the other side of the operator."""
        if isinstance(x, pd.Series):
            return x
        if np.isscalar(x) or isinstance(x, (int, float, np.number)):
            return x
        return pd.Series(x)

    @staticmethod
    def _align(a, b):
        if isinstance(a, pd.Series) and isinstance(b, pd.Series):
            a, b = a.align(b, join="outer")
        return a, b

    def _add(self, a, b):
        a, b = self._align(a, b)
        return self._to_series(a) + self._to_series(b)

    def _sub(self, a, b):
        a, b = self._align(a, b)
        return self._to_series(a) - self._to_series(b)

    def _mul(self, a, b):
        a, b = self._align(a, b)
        return self._to_series(a) * self._to_series(b)

    def _div(self, a, b):
        a, b = self._align(a, b)
        if isinstance(b, pd.Series):
            b = b.replace(0, np.nan)
        return self._to_series(a) / self._to_series(b)

    def _apply_el(self, name, fn, vals):
        series_vals = [self._to_series(v) for v in vals]
        return fn(*series_vals)

    def _apply_ts(self, name, fn, vals, df):
        code = df[self.code_col]
        # first arg is the series; rest are scalars/params
        series = self._to_series(vals[0])
        # broadcast scalars into lists matching series length
        args = [series] + [v for v in vals[1:]]
        # special-case functions that need code passed explicitly
        if name in {"Ref", "Delta", "MA", "EMA", "Std", "Sum",
                    "Min", "Max", "TsRank", "Product", "IfElse"}:
            return fn(*args, code)
        return fn(*args)

    def _apply_cs(self, name, fn, vals, df):
        date = df[self.date_col]
        series = self._to_series(vals[0])
        return fn(series, date)
