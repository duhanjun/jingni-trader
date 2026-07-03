"""
Factor Expression Engine
========================

Declarative factor expressions inspired by:

* **Qlib** ``qlib.data.ops`` — MA / Ref / Std / Log / Abs / Rank / ...
* **AKQuant** ``akquant.factor`` — ``Rank(Ts_Mean(Close, 5))`` style
  Alpha101 DSL on top of a columnar engine.

The goal is to let users write::

    factor = engine.calc(df, "Rank(Delta(Log(Close), 5)) - 0.5*Std(Close, 20)")

instead of writing imperative pandas code for every new factor.

Design
------

``FactorEngine`` takes a long-format ``DataFrame`` (columns
``code, date, open, high, low, close, volume, ...``) and returns a new
DataFrame with one column per requested expression.  The engine is
group-aware: every time-series (prefix ``Ts``/``Ref``/``Delta``/``MA``/...)
operator is applied **per ``code``** while every cross-sectional (prefix
``Cs``/``Rank``/``Demean``/``Zscore``) operator is applied **per ``date``**.

Parsing uses a small **Pratt top-down parser** (no external deps) so
the grammar is easy to extend and inspect.  The parser emits a tree of
``Node`` objects that the engine walks to materialise results.  A
sub-expression cache keyed by ``(expression, group_signature)`` avoids
re-computing the same intermediate, following the Qlib pattern.
"""
from .engine import FactorEngine, ExpressionError
from .parser import parse, Node

__all__ = ["FactorEngine", "ExpressionError", "parse", "Node"]
