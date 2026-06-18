"""
Expression DSL - Evaluator
==========================

Walk an AST produced by :mod:`expression_dsl.parser` and evaluate it
against a long-format ``pandas.DataFrame`` (``code``/``date``/``open``
/``high``/``low``/``close``/``volume``/``amount``/...).

Design
------
The evaluator is *per-stock* at the operator level.  All time-series
operators (those registered in :mod:`expression_dsl.operators`) take
a single-stock ``pd.Series`` and return a single-stock ``pd.Series``.
The :class:`Evaluator` is responsible for the ``groupby('code')``
routing so that windows never cross stock boundaries.  After the
final value is computed per stock, results are concatenated and
aligned back to the original frame's row order.

This mirrors the design of ``FactorEngine.compute_a_share_factors``
in jingni-trader but additionally:

- **Cross-section** operators (currently ``Rank`` and ``Scale``) run
  per ``date`` after the per-stock computation finishes.
- The whole pipeline is expression-driven — there is no hard-coded
  list of factors to compute.

Returned values are always a ``pd.Series`` aligned with the input
frame's integer index.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd

from .operators import get_operator
from .parser import (
    AstNode,
    BinaryOpNode,
    CallNode,
    FieldNode,
    NumberNode,
    UnaryOpNode,
    parse,
)

# columns automatically exposed as ``$field`` references
DEFAULT_FIELDS = ("open", "high", "low", "close", "volume", "amount",
                  "turnover_rate", "change_pct")


class EvalError(ValueError):
    pass


class Evaluator:
    """Evaluate an expression AST against a long-format frame.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format frame.  Must contain ``code`` and ``date`` columns.
    fields : tuple[str, ...], optional
        Additional columns to expose as ``$field`` references.
    """

    def __init__(self, data: pd.DataFrame, fields: Optional[tuple] = None):
        if "code" not in data.columns or "date" not in data.columns:
            raise EvalError("输入数据必须包含 code 与 date 列")
        # 保序——输出的 index 必须与输入对齐
        self.data = data.reset_index(drop=True)
        self.fields = tuple(fields) if fields else DEFAULT_FIELDS

    def evaluate(self, ast: AstNode) -> pd.Series:
        """Evaluate ``ast`` and return a Series aligned with ``self.data``."""
        # Split into per-stock DataFrames so operators never cross stock boundaries.
        # Each sub-frame's integer position is recorded so we can place the
        # per-stock result back at the right global index.
        pieces = []
        for code, sub in self.data.groupby("code", sort=False):
            sub = sub.reset_index(drop=False)  # preserve the original index
            sub_ev = _PerStockEvaluator(sub, self.fields)
            res = sub_ev.evaluate(ast)
            res.index = sub["index"].values  # re-attach original row index
            res.name = code
            pieces.append(res)
        # Concatenate, then sort by the original global index to match self.data
        result = pd.concat(pieces).sort_index()
        return result.reindex(self.data.index)


class _PerStockEvaluator:
    """Per-stock evaluator.  Operators are pure per-stock functions."""

    def __init__(self, data: pd.DataFrame, fields: tuple):
        if "date" not in data.columns:
            raise EvalError("per-stock data 缺少 date 列")
        # 保留 "index" 列（来自上层 reset_index(drop=False)）
        # Make sure data is sorted by date for window correctness
        self.orig_index = data["index"].values if "index" in data.columns else None
        data = data.drop(columns=["index"]) if "index" in data.columns else data
        self.data = data.sort_values("date").reset_index(drop=True)
        self.fields = fields

    def evaluate(self, ast: AstNode) -> pd.Series:
        out = self._eval(ast)
        if not isinstance(out, pd.Series):
            raise EvalError("表达式最终结果不是 Series（可能未使用字段或函数）")
        # Realign: out is in sorted-by-date order; map back to orig_index
        if self.orig_index is not None:
            # out.index is 0..N-1 (sorted); map to self.orig_index which
            # is the original row index for this stock
            mapped = pd.Series(out.values, index=self.orig_index)
            return mapped.sort_index()
        return out.reindex(self.data.index)

    # --- dispatchers ---------------------------------------------------

    def _eval(self, node: AstNode) -> Any:
        if isinstance(node, FieldNode):
            if node.name not in self.data.columns:
                raise EvalError(f"未知字段 ${node.name}")
            return self.data[node.name]
        if isinstance(node, NumberNode):
            return node.value
        if isinstance(node, BinaryOpNode):
            return self._eval_binary(node)
        if isinstance(node, UnaryOpNode):
            return self._eval_unary(node)
        if isinstance(node, CallNode):
            return self._eval_call(node)
        raise EvalError(f"无法求值的节点: {node!r}")

    def _eval_binary(self, node: BinaryOpNode) -> pd.Series:
        left = self._eval(node.left)
        right = self._eval(node.right)
        if node.op == "+":
            return self._align_pair(left, right, lambda a, b: a + b)
        if node.op == "-":
            return self._align_pair(left, right, lambda a, b: a - b)
        if node.op == "*":
            return self._align_pair(left, right, lambda a, b: a * b)
        if node.op == "/":
            return self._align_pair(left, right, lambda a, b: a / b)
        if node.op == "^":
            return self._align_pair(left, right, lambda a, b: a ** b)
        raise EvalError(f"未知二元操作符: {node.op}")

    def _eval_unary(self, node: UnaryOpNode) -> pd.Series:
        v = self._eval(node.operand)
        if not isinstance(v, pd.Series):
            return -v
        return -v

    @staticmethod
    def _align_pair(left, right, op):
        if isinstance(left, pd.Series) and isinstance(right, pd.Series):
            if not left.index.equals(right.index):
                right = right.reindex(left.index)
            return op(left, right)
        if isinstance(left, pd.Series):
            return op(left, right)
        if isinstance(right, pd.Series):
            return op(left, right)
        return op(left, right)

    def _eval_call(self, node: CallNode) -> Any:
        func = get_operator(node.func)
        args = [self._eval(a) for a in node.args]
        return func(*args)


# ---------------------------------------------------------------------------
# Global (cross-stock) helpers for Rank/Scale.  Used by the public helpers
# ``evaluate_rank`` / ``evaluate_scale`` which can also be called directly
# from the top-level ``evaluate`` function.
# ---------------------------------------------------------------------------


def _cross_section_apply(expr: str, data: pd.DataFrame, kind: str) -> pd.Series:
    """Evaluate ``expr`` per stock then apply a per-date transform.

    ``expr`` must be a top-level ``Rank(...)`` or ``Scale(...)`` call.
    The inner argument is evaluated per stock; the final transform is
    applied across all stocks for each date.
    """
    ast = parse(expr)
    if not isinstance(ast, CallNode) or ast.func not in ("Rank", "Scale"):
        raise EvalError(f"_cross_section_apply 仅支持 Rank/Scale 顶层调用，得到 {ast!r}")
    if len(ast.args) != 1:
        raise EvalError("Rank/Scale 顶层调用仅接受 1 个参数")

    # Evaluate the inner expression per stock
    ev = Evaluator(data)
    # Use a stripped AST without the Rank/Scale wrapper to avoid the
    # cross-section operator being called per-stock.
    inner_ast = ast.args[0]
    per_stock = ev.evaluate(inner_ast)
    # `per_stock` is aligned with `data`; group by date.
    grp = data.assign(**{"__x__": per_stock}).groupby("date")["__x__"]
    if kind == "rank":
        out = grp.rank(pct=True)
    elif kind == "scale":
        out = grp.transform(lambda s: s / s.abs().sum() if s.abs().sum() != 0 else s * 0)
    else:
        raise EvalError(f"未知 cross-section 类型: {kind}")
    return out.reset_index(drop=True)


def evaluate(expr: str, data: pd.DataFrame) -> pd.Series:
    """Convenience helper: parse + evaluate in one call.

    Handles ``Rank`` and ``Scale`` (which need cross-section context) by
    switching to the global evaluator automatically.
    """
    ast = parse(expr)
    # If the top-level is a Rank/Scale call, use the cross-section path
    if isinstance(ast, CallNode) and ast.func in ("Rank", "Scale"):
        return _cross_section_apply(expr, data, ast.func.lower())
    return Evaluator(data).evaluate(ast)


# ---------------------------------------------------------------------------
# Convenience factor library: 25+ canonical A-share alpha factors expressed
# in the DSL.  These are *not* executed here — they exist so callers can
# iterate / cherry-pick / diff against hard-coded versions in the existing
# ``FactorEngine``.
# ---------------------------------------------------------------------------
ALPHA158_LITE: dict = {
    # 价量反转
    "REVS5":   "-PctChange($close, 5)",
    "REVS20":  "-PctChange($close, 20)",
    # 量能
    "VOL_CHG": "Mean($volume, 5) / Mean($volume, 20) - 1",
    "AMOUNT_CHG": "Mean($amount, 5) / Mean($amount, 20) - 1",
    # 波动
    "VOL20":   "Std(PctChange($close, 1), 20)",
    "RNG20":   "(Mean($high, 20) - Mean($low, 20)) / Mean($close, 20)",
    # 趋势
    "MA_RATIO_5_20": "Mean($close, 5) / Mean($close, 20) - 1",
    "MA_RATIO_5_60": "Mean($close, 5) / Mean($close, 60) - 1",
    # 资金流 (代理)
    "MOMENTUM_VOL": "PctChange($close, 20) * Log(Mean($volume, 20) + 1)",
    # 相关性
    "CORR_PV_20": "Corr($close, $volume, 20)",
    "CORR_PR_5":  "Corr(PctChange($close, 1), Mean($close, 5), 5)",
    # 残差
    "CLOSE_MA5_RESID": "$close / Mean($close, 5) - 1",
    "CLOSE_MA20_RESID": "$close / Mean($close, 20) - 1",
    # 量价背离
    "PV_DIVERGENCE": "PctChange($close, 5) - PctChange(Mean($volume, 5), 5)",
    # EWM
    "EWM_10":  "EwmMean($close, 10)",
    "EWM_30":  "EwmMean($close, 30)",
    # 复合
    "QUALITY_MOM": "PctChange($close, 60) / Std(PctChange($close, 1), 60)",
    "MEAN_REV_5":  "$close / Mean($close, 5) - 1",
    "MEAN_REV_20": "$close / Mean($close, 20) - 1",
    "MEAN_REV_60": "$close / Mean($close, 60) - 1",
    "TREND_20_60": "Mean($close, 20) / Mean($close, 60) - 1",
    "TREND_5_20":  "Mean($close, 5) / Mean($close, 20) - 1",
    "VOLATILITY_60": "Std(PctChange($close, 1), 60)",
    "SKEW_20":     "Sign($close - Mean($close, 20)) * Abs($close - Mean($close, 20)) ^ 1.5",
    "KURT_20":     "Abs($close - Mean($close, 20)) ^ 2 - Std($close, 20) ^ 2",
    "TURNOVER_MA": "Mean($turnover_rate, 5) - Mean($turnover_rate, 20)",
    "AMH":         "($high - $low) / $open",
}


if __name__ == "__main__":  # manual smoke test
    import sys
    if len(sys.argv) < 2:
        print("usage: python evaluator.py '<expr>'")
        sys.exit(0)
    expr = sys.argv[1]
    n = 50
    rng = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "code": ["X"] * n,
        "date": rng,
        "close": np.cumsum(np.random.randn(n)) + 100,
        "volume": np.random.randint(1000, 10000, n).astype(float),
        "amount": np.random.randint(1_000_000, 10_000_000, n).astype(float),
        "open": np.cumsum(np.random.randn(n)) + 100,
        "high": np.cumsum(np.random.randn(n)) + 101,
        "low": np.cumsum(np.random.randn(n)) + 99,
        "turnover_rate": np.random.rand(n),
    })
    print(evaluate(expr, df).tail())
