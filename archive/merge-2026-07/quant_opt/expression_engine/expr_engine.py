"""
Factor Expression Engine

Inspired by Microsoft Qlib's expression engine
(https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py)

Provides a mini-DSL for defining alpha factors as string formulas such as:
    "Ref($close, 5) / Ref($close, 1) - 1"
    "Mean($close, 20) - $close"
    "Rank(Mean($volume, 5))"

The engine consists of three components:
1. Tokenizer   - lex the formula into tokens
2. AST builder - parse tokens into a tree of operators
3. Evaluator   - evaluate the AST against a multi-index (code, date) DataFrame

Operators are vectorized and operate on a (code, date) panel.
The engine supports both cross-sectional (per date) and time-series operators.

This module is intentionally pure (no I/O, no global state) so that it
can be plugged into the existing factor-engine without touching the
main `engine.py` flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldRef:
    """Reference to a data column, e.g. $close, $volume, $open."""
    name: str

    def __repr__(self) -> str:
        return f"${self.name}"


@dataclass(frozen=True)
class Number:
    value: float

    def __repr__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class UnaryOp:
    name: str
    operand: "Node"


@dataclass(frozen=True)
class BinaryOp:
    name: str
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class FuncCall:
    name: str
    args: Tuple["Node", ...]


Node = Union[FieldRef, Number, UnaryOp, BinaryOp, FuncCall]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS: Sequence[Tuple[str, str]] = [
    ("WS",      r"\s+"),
    ("FIELD",   r"\$[A-Za-z_][A-Za-z0-9_]*"),
    ("NUM",     r"\d+(\.\d+)?([eE][+-]?\d+)?"),
    ("NAME",    r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP",      r"[\+\-\*/\^\,\(\)]"),
]


def tokenize(formula: str) -> List[Tuple[str, str]]:
    """Convert a formula string into a list of (type, value) tokens.

    Whitespace tokens are dropped.
    """
    import re
    regex = "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS)
    tokens: List[Tuple[str, str]] = []
    for match in re.finditer(regex, formula):
        kind = match.lastgroup or ""
        value = match.group()
        if kind == "WS":
            continue
        tokens.append((kind, value))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(ValueError):
    """Raised when a formula cannot be parsed."""


class _Parser:
    """Recursive-descent parser for the factor expression DSL.

    Grammar (simplified):
        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := unary ('^' unary)*
        unary  := '-' unary | atom
        atom   := NUMBER | FIELD | NAME '(' args? ')' | '(' expr ')'
    """

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[Tuple[str, str]]:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, kind: Optional[str] = None) -> Tuple[str, str]:
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of formula")
        if kind is not None and tok[0] != kind:
            raise ParseError(f"Expected {kind}, got {tok}")
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self._expr()
        if self.pos != len(self.tokens):
            raise ParseError(f"Trailing tokens at position {self.pos}")
        return node

    def _expr(self) -> Node:
        node = self._term()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in ("+", "-"):
                break
            self._consume()
            rhs = self._term()
            node = BinaryOp(tok[1], node, rhs)
        return node

    def _term(self) -> Node:
        node = self._factor()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] not in ("*", "/"):
                break
            self._consume()
            rhs = self._factor()
            node = BinaryOp(tok[1], node, rhs)
        return node

    def _factor(self) -> Node:
        node = self._unary()
        while True:
            tok = self._peek()
            if tok is None or tok[0] != "OP" or tok[1] != "^":
                break
            self._consume()
            rhs = self._unary()
            node = BinaryOp("^", node, rhs)
        return node

    def _unary(self) -> Node:
        tok = self._peek()
        if tok is not None and tok[0] == "OP" and tok[1] == "-":
            self._consume()
            return UnaryOp("-", self._unary())
        return self._atom()

    def _atom(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise ParseError("Unexpected end of formula")
        kind, val = tok
        if kind == "NUM":
            self._consume()
            return Number(float(val))
        if kind == "FIELD":
            self._consume()
            return FieldRef(val[1:])  # strip leading '$'
        if kind == "OP" and val == "(":
            self._consume()
            node = self._expr()
            self._consume("OP")  # ')'
            if self._peek() is not None and self._peek()[0] == "OP" and self._peek()[1] == ")":
                self._consume()
            return node
        if kind == "NAME":
            self._consume()
            if self._peek() is None or self._peek()[0] != "OP" or self._peek()[1] != "(":
                raise ParseError(f"Expected '(' after function name, got {self._peek()}")
            self._consume("OP")
            args: List[Node] = []
            if not (self._peek() and self._peek()[0] == "OP" and self._peek()[1] == ")"):
                args.append(self._expr())
                while self._peek() and self._peek()[0] == "OP" and self._peek()[1] == ",":
                    self._consume()
                    args.append(self._expr())
            self._consume("OP")
            return FuncCall(val, tuple(args))
        raise ParseError(f"Unexpected token {tok}")


def parse_formula(formula: str) -> Node:
    """Parse a formula string into an AST."""
    return _Parser(tokenize(formula)).parse()


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

# Each operator receives a panel DataFrame (index=[code, date]) and returns
# a Series/array of the same shape. Cross-sectional operators work per date;
# time-series operators work per code.

class OperatorRegistry:
    """Holds the implementations of built-in factor operators."""

    def __init__(self):
        self._fns: Dict[str, Callable[..., pd.Series]] = {}
        self._register_defaults()

    def register(self, name: str, fn: Callable[..., pd.Series]) -> None:
        self._fns[name] = fn

    def has(self, name: str) -> bool:
        return name in self._fns

    def call(self, name: str, *args, **kwargs) -> pd.Series:
        if name not in self._fns:
            raise KeyError(f"Unknown operator/function: {name}")
        return self._fns[name](*args, **kwargs)

    # -- default operators ---------------------------------------------------

    def _register_defaults(self) -> None:
        ts = self  # alias

        def _to_scalar(x):
            """Accept a scalar literal or a constant Series and return a scalar."""
            if isinstance(x, pd.Series):
                if len(x) == 0:
                    return np.nan
                # detect constant series
                non_na = x.dropna()
                if len(non_na) == 0:
                    return np.nan
                first = non_na.iloc[0]
                if (non_na == first).all():
                    return float(first)
                return x  # not constant - treat as Series
            try:
                return float(x)
            except (TypeError, ValueError):
                return x

        # --- time-series operators (per code) --------------------------------
        def _ref(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").shift(n)

        def _delta(x, n):
            n = int(_to_scalar(n))
            return x - _ref(x, n)

        def _mean(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)

        def _std(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").rolling(n, min_periods=2).std().reset_index(level=0, drop=True)

        def _sum(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").rolling(n, min_periods=1).sum().reset_index(level=0, drop=True)

        def _max(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").rolling(n, min_periods=1).max().reset_index(level=0, drop=True)

        def _min(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").rolling(n, min_periods=1).min().reset_index(level=0, drop=True)

        def _ema(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").transform(lambda s: s.ewm(span=n, adjust=False).mean())

        def _ts_rank(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="code").rolling(n, min_periods=2).apply(
                lambda s: s.rank(pct=True).iloc[-1], raw=False
            ).reset_index(level=0, drop=True)

        def _corr(x, y, n):
            """Rolling time-series correlation between two equal-length series."""
            n = int(_to_scalar(n))
            xdf = x.unstack(level="code")
            ydf = y.unstack(level="code")
            out = xdf.rolling(n, min_periods=2).corr(ydf)
            # out has index (date, code); swap to match x's (code, date) order
            stacked = out.stack(future_stack=True)
            stacked.index = stacked.index.swaplevel("code", "date")
            return stacked.reindex(x.index)

        # --- cross-sectional operators (per date) ----------------------------
        def _rank(x):
            return x.groupby(level="date").rank(pct=True)

        def _scale(x):
            return x.groupby(level="date").apply(lambda s: s / s.abs().sum()).reset_index(level=0, drop=True)

        def _zscore(x):
            return x.groupby(level="date").transform(lambda s: (s - s.mean()) / s.std(ddof=0))

        def _quantile(x, n):
            n = int(_to_scalar(n))
            return x.groupby(level="date").transform(lambda s: pd.qcut(s, n, labels=False, duplicates="drop"))

        def _cs_mean(x):
            return x.groupby(level="date").transform("mean")

        def _cs_std(x):
            return x.groupby(level="date").transform("std")

        def _neutralize(x, lncap):
            """Cross-sectional OLS neutralization vs. lncap (per date)."""
            res = pd.Series(index=x.index, dtype=float)
            for dt, grp in x.groupby(level="date"):
                y = grp.values
                w = lncap.xs(dt).reindex(grp.index.get_level_values("code")).values
                mask = ~(np.isnan(y) | np.isnan(w))
                if mask.sum() < 5:
                    res.loc[grp.index] = y
                    continue
                coef = np.polyfit(w[mask], y[mask], 1)
                pred = np.polyval(coef, w)
                res.loc[grp.index] = y - pred
            return res

        # --- element-wise wrappers -------------------------------------------
        def _abs(x):
            return x.abs()

        def _log(x):
            return np.log(x.replace(0, np.nan))

        def _sign(x):
            return np.sign(x)

        def _signedpower(x, e):
            e = float(_to_scalar(e))
            return np.sign(x) * (np.abs(x) ** e)

        self.register("Ref", _ref)
        self.register("Delta", _delta)
        self.register("Mean", _mean)
        self.register("Std", _std)
        self.register("Sum", _sum)
        self.register("Max", _max)
        self.register("Min", _min)
        self.register("EMA", _ema)
        self.register("TsRank", _ts_rank)
        self.register("Corr", _corr)

        self.register("Rank", _rank)
        self.register("Scale", _scale)
        self.register("Zscore", _zscore)
        self.register("Quantile", _quantile)
        self.register("CsMean", _cs_mean)
        self.register("CsStd", _cs_std)
        self.register("Neutralize", _neutralize)

        self.register("Abs", _abs)
        self.register("Log", _log)
        self.register("Sign", _sign)
        self.register("SignedPower", _signedpower)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ExpressionEvaluator:
    """Evaluate a parsed AST against a (code, date) panel of data."""

    def __init__(self, registry: Optional[OperatorRegistry] = None):
        self.registry = registry or OperatorRegistry()

    def evaluate(self, node: Node, data: pd.DataFrame) -> pd.Series:
        panel = self._ensure_panel(data)

        if isinstance(node, FieldRef):
            if node.name not in panel.columns:
                raise KeyError(f"Column ${node.name} not in data")
            return panel[node.name]

        if isinstance(node, Number):
            return pd.Series(node.value, index=panel.index, dtype=float)

        if isinstance(node, UnaryOp):
            v = self.evaluate(node.operand, data)
            if node.name == "-":
                return -v
            return self.registry.call(node.name, v)

        if isinstance(node, BinaryOp):
            l = self.evaluate(node.left, data)
            r = self.evaluate(node.right, data)
            if node.name == "+":
                return l + r
            if node.name == "-":
                return l - r
            if node.name == "*":
                return l * r
            if node.name == "/":
                return l / r
            if node.name == "^":
                return l ** r
            return self.registry.call(node.name, l, r)

        if isinstance(node, FuncCall):
            args = tuple(self.evaluate(a, data) for a in node.args)
            return self.registry.call(node.name, *args)

        raise TypeError(f"Unknown AST node: {type(node)}")

    @staticmethod
    def _ensure_panel(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.MultiIndex) or set(data.index.names) != {"code", "date"}:
            if {"code", "date"}.issubset(data.columns):
                df = data.set_index(["code", "date"]).sort_index()
                return df
            raise ValueError("Data must be indexed by (code, date) or contain those columns")
        return data.sort_index()


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY = OperatorRegistry()
DEFAULT_EVALUATOR = ExpressionEvaluator(DEFAULT_REGISTRY)


def evaluate_formula(formula: str, data: pd.DataFrame,
                     evaluator: Optional[ExpressionEvaluator] = None) -> pd.Series:
    """High-level helper: parse + evaluate a factor formula."""
    node = parse_formula(formula)
    ev = evaluator or DEFAULT_EVALUATOR
    return ev.evaluate(node, data)


def list_operators() -> List[str]:
    """Return the names of all registered operators (handy for docs/UI)."""
    return sorted(DEFAULT_REGISTRY._fns.keys())
