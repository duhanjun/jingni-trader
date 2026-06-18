"""
Tests for the expression DSL
=============================

Three categories of tests:

1. **Tokenizer** – lexing edge cases.
2. **Parser** – operator precedence, nested calls, error reporting.
3. **Evaluator** – numeric equivalence against naive ``pandas``
   implementations (the gold standard for correctness).
"""
import math
import unittest

import numpy as np
import pandas as pd

from quant_opt_20260618.expression_dsl import (
    tokenize, Token, TokenType, TokenizeError,
    parse, ParseError, FieldNode, NumberNode,
    BinaryOpNode, UnaryOpNode, CallNode,
    Evaluator, evaluate, ALPHA158_LITE, list_operators, register_operator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _toy_df(n: int = 100, codes=("AAA", "BBB")) -> pd.DataFrame:
    """Build a tiny long-format frame for evaluator tests."""
    rng = np.random.default_rng(0)
    rows = []
    for code in codes:
        # 用时间索引——保证 groupby('date') 行为确定
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = np.cumsum(rng.standard_normal(n)) + 100
        vol = rng.integers(1000, 10000, n).astype(float)
        amt = rng.integers(1_000_000, 10_000_000, n).astype(float)
        op = close + rng.standard_normal(n) * 0.1
        hi = close + np.abs(rng.standard_normal(n))
        lo = close - np.abs(rng.standard_normal(n))
        tr = rng.random(n)
        for i in range(n):
            rows.append({
                "code": code, "date": dates[i],
                "open": op[i], "high": hi[i], "low": lo[i],
                "close": close[i], "volume": vol[i], "amount": amt[i],
                "turnover_rate": tr[i], "change_pct": (close[i] - op[i]) / op[i],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TestTokenizer(unittest.TestCase):

    def test_basic_field_and_number(self):
        toks = tokenize("$close - 1.5")
        self.assertEqual(toks[:-1], [
            Token(TokenType.FIELD, "close", 0),
            Token(TokenType.OP, "-", 7),
            Token(TokenType.NUMBER, "1.5", 9),
        ])
        self.assertEqual(toks[-1].type, TokenType.EOF)

    def test_function_call(self):
        toks = tokenize("Mean($close, 20)")
        types = [t.type for t in toks]
        self.assertEqual(types, [
            TokenType.IDENT, TokenType.LPAREN, TokenType.FIELD,
            TokenType.COMMA, TokenType.NUMBER, TokenType.RPAREN,
            TokenType.EOF,
        ])

    def test_whitespace_ignored(self):
        toks = tokenize("  $close \t  /  $open  ")
        self.assertEqual([t.value for t in toks if t.type != TokenType.EOF],
                         ["close", "/", "open"])

    def test_unterminated_field_raises(self):
        with self.assertRaises(TokenizeError):
            tokenize("$")

    def test_unknown_char_raises(self):
        with self.assertRaises(TokenizeError):
            tokenize("$close & $open")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser(unittest.TestCase):

    def test_field_and_number(self):
        ast = parse("$close")
        self.assertIsInstance(ast, FieldNode)
        self.assertEqual(ast.name, "close")

        ast = parse("3.14")
        self.assertIsInstance(ast, NumberNode)
        self.assertAlmostEqual(ast.value, 3.14)

    def test_precedence(self):
        # 1 + 2 * 3 should parse as 1 + (2*3)
        ast = parse("1 + 2 * 3")
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, "+")
        self.assertIsInstance(ast.left, NumberNode)
        self.assertIsInstance(ast.right, BinaryOpNode)
        self.assertEqual(ast.right.op, "*")

    def test_parens(self):
        ast = parse("(1 + 2) * 3")
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, "*")
        self.assertEqual(ast.left.op, "+")

    def test_nested_call(self):
        # ``Mean($close / $open, 20)`` — first arg is ``$close / $open``
        ast = parse("Mean($close / $open, 20)")
        self.assertIsInstance(ast, CallNode)
        self.assertEqual(ast.func, "Mean")
        self.assertEqual(len(ast.args), 2)
        self.assertIsInstance(ast.args[0], BinaryOpNode)
        self.assertIsInstance(ast.args[1], NumberNode)

    def test_unary_minus(self):
        ast = parse("-PctChange($close, 5)")
        self.assertIsInstance(ast, UnaryOpNode)
        self.assertEqual(ast.op, "-")

    def test_trailing_token_error(self):
        with self.assertRaises(ParseError):
            parse("Mean($close, 20) junk")

    def test_missing_paren_error(self):
        with self.assertRaises(ParseError):
            parse("Mean($close, 20")


# ---------------------------------------------------------------------------
# Evaluator – numeric equivalence
# ---------------------------------------------------------------------------


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.df = _toy_df()

    def _first_stock(self, series: pd.Series) -> pd.Series:
        return series[self.df["code"] == "AAA"].reset_index(drop=True)

    # ---- arithmetic & fields ----

    def test_field_passthrough(self):
        out = evaluate("$close", self.df)
        np.testing.assert_array_equal(out.values, self.df["close"].values)

    def test_arithmetic(self):
        out = evaluate("$close - $open", self.df)
        np.testing.assert_array_almost_equal(out.values,
                                             (self.df["close"] - self.df["open"]).values)

    def test_unary_minus(self):
        out = evaluate("-$close", self.df)
        np.testing.assert_array_equal(out.values, -self.df["close"].values)

    def test_div(self):
        out = evaluate("$close / $open", self.df)
        np.testing.assert_array_almost_equal(out.values,
                                             (self.df["close"] / self.df["open"]).values)

    # ---- rolling ops ----

    def test_ref(self):
        out = evaluate("Ref($close, 5)", self.df)
        for code in self.df["code"].unique():
            sub = self.df[self.df["code"] == code].sort_values("date")
            np.testing.assert_array_equal(
                out[self.df["code"] == code].values,
                sub["close"].shift(5).values,
            )

    def test_mean_rolling(self):
        out = evaluate("Mean($close, 20)", self.df)
        for code in self.df["code"].unique():
            sub = self.df[self.df["code"] == code].sort_values("date")
            expected = sub["close"].rolling(20, min_periods=10).mean()
            np.testing.assert_array_almost_equal(
                out[self.df["code"] == code].values,
                expected.values,
                decimal=10,
            )

    def test_std_rolling(self):
        out = evaluate("Std($close, 20)", self.df)
        for code in self.df["code"].unique():
            sub = self.df[self.df["code"] == code].sort_values("date")
            expected = sub["close"].rolling(20, min_periods=10).std()
            np.testing.assert_array_almost_equal(
                out[self.df["code"] == code].values,
                expected.values,
                decimal=10,
            )

    def test_pct_change(self):
        out = evaluate("PctChange($close, 5)", self.df)
        for code in self.df["code"].unique():
            sub = self.df[self.df["code"] == code].sort_values("date")
            expected = sub["close"].pct_change(5)
            np.testing.assert_array_almost_equal(
                out[self.df["code"] == code].values,
                expected.values,
                decimal=10,
            )

    def test_corr(self):
        out = evaluate("Corr($close, $volume, 20)", self.df)
        # Spot-check a few cells
        for code in self.df["code"].unique():
            sub = self.df[self.df["code"] == code].sort_values("date")
            expected = sub["close"].rolling(20, min_periods=10).corr(sub["volume"])
            np.testing.assert_array_almost_equal(
                out[self.df["code"] == code].values,
                expected.values,
                decimal=6,
            )

    # ---- element-wise ----

    def test_log_of_positive(self):
        out = evaluate("Log($close)", self.df)
        np.testing.assert_array_almost_equal(out.values, np.log(self.df["close"]).values)

    def test_abs(self):
        out = evaluate("Abs(-$close)", self.df)
        np.testing.assert_array_almost_equal(out.values, np.abs(self.df["close"]).values)

    def test_signpower(self):
        out = evaluate("SignPower($close, 2)", self.df)
        np.testing.assert_array_almost_equal(
            out.values,
            np.sign(self.df["close"]) * np.power(np.abs(self.df["close"]), 2).values
        )

    # ---- cross-section ----

    def test_rank(self):
        out = evaluate("Rank($close)", self.df)
        for dt in self.df["date"].unique():
            mask = self.df["date"] == dt
            r = out[mask].rank(pct=True).values
            np.testing.assert_array_almost_equal(out[mask].values, r, decimal=10)

    def test_scale(self):
        out = evaluate("Scale($close)", self.df)
        for dt in self.df["date"].unique():
            mask = self.df["date"] == dt
            sub = self.df.loc[mask, "close"]
            expected = sub / sub.abs().sum()
            np.testing.assert_array_almost_equal(out[mask].values, expected.values, decimal=10)

    # ---- error handling ----

    def test_unknown_field(self):
        with self.assertRaises(Exception):
            evaluate("$does_not_exist", self.df)

    def test_unknown_operator(self):
        with self.assertRaises(KeyError):
            evaluate("NoSuchFn($close, 20)", self.df)

    def test_invalid_window(self):
        with self.assertRaises(Exception):
            evaluate("Mean($close, 0)", self.df)


# ---------------------------------------------------------------------------
# Alpha158 lite smoke
# ---------------------------------------------------------------------------


class TestAlpha158Lite(unittest.TestCase):

    def setUp(self):
        self.df = _toy_df(n=200)

    def test_all_factors_compute(self):
        for name, expr in ALPHA158_LITE.items():
            with self.subTest(factor=name):
                out = evaluate(expr, self.df)
                # NaN is fine (warm-up window) but no exceptions
                self.assertEqual(len(out), len(self.df))
                # at least 50% of values finite for windows up to 60
                self.assertGreater(out.notna().mean(), 0.5, f"{name} 全空")


if __name__ == "__main__":
    unittest.main(verbosity=2)
