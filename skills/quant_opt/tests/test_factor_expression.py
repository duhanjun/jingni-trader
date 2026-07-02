"""
SafeExpressionEngine 单元测试
============================

覆盖：
1. 基本 AST 校验（白名单通过 / 黑名单拒绝）
2. Alpha101 风格时序/横截面算子的数值正确性
3. 危险操作（__import__、exec、getattr）的拦截
4. 性能基准（与硬编码 groupby 实现对比）
"""

import sys
import os
import time
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT_OPT_DIR = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(QUANT_OPT_DIR)
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from skills.quant_opt.factor_expression_engine import (
    SafeExpressionEngine,
    FactorEngine,
    ExpressionContext,
    UnsafeExpressionError,
)


def make_synth_data(n_codes: int = 5, n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """生成合成 A 股数据"""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    for code_i in range(n_codes):
        code = f"{code_i:06d}.SZ"
        base = 10.0 + code_i * 5
        ret = rng.normal(0.001, 0.02, n_days)
        prices = base * np.cumprod(1 + ret)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": prices[i] * 0.99,
                "high": prices[i] * 1.01,
                "low": prices[i] * 0.99,
                "close": prices[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


class TestASTValidation(unittest.TestCase):
    """AST 白名单校验测试"""

    def setUp(self):
        self.engine = SafeExpressionEngine()

    def test_simple_arithmetic_passes(self):
        compiled = self.engine.parse("1 + 2 * 3")
        self.assertEqual(compiled.expr, "1 + 2 * 3")

    def test_function_call_passes(self):
        compiled = self.engine.parse("abs(-1.5) + max(1, 2, 3)")
        ctx = ExpressionContext(data=make_synth_data())
        result = self.engine.compute(compiled, ctx)
        # 当结果是标量时，会被广播到所有行
        self.assertTrue(isinstance(result, pd.Series))
        # 每行的值应等于 abs(-1.5) + 3 = 4.5
        self.assertAlmostEqual(float(result.iloc[0]), abs(-1.5) + 3)

    def test_unsafe_import_blocked(self):
        with self.assertRaises(UnsafeExpressionError):
            self.engine.parse("__import__('os').system('echo hi')")

    def test_unsafe_dunder_blocked(self):
        with self.assertRaises(UnsafeExpressionError):
            self.engine.parse("__class__")

    def test_unsafe_function_blocked(self):
        with self.assertRaises(UnsafeExpressionError):
            self.engine.parse("eval('1+1')")

    def test_unsafe_exec_blocked(self):
        with self.assertRaises(UnsafeExpressionError):
            self.engine.parse("exec('print(1)')")

    def test_keyword_arguments_blocked(self):
        with self.assertRaises(UnsafeExpressionError):
            self.engine.parse("abs(x=1)")

    def test_length_limit(self):
        engine = SafeExpressionEngine(max_expression_length=10)
        with self.assertRaises(UnsafeExpressionError):
            engine.parse("1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11")


class TestAlpha101Operators(unittest.TestCase):
    """Alpha101 风格算子的数值正确性测试"""

    def setUp(self):
        self.data = make_synth_data(n_codes=3, n_days=40)
        self.engine = SafeExpressionEngine()
        self.fengine = FactorEngine()

    def test_ts_mean_values(self):
        out = self.fengine.compute(self.data, "Ts_Mean($close, 5)", name="ts_mean_5")
        expected = self.data.groupby("code")["close"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        np.testing.assert_allclose(out["ts_mean_5"].values, expected.values, rtol=1e-6)

    def test_delay_values(self):
        out = self.fengine.compute(self.data, "Delay($close, 1)", name="delay_1")
        expected = self.data.groupby("code")["close"].shift(1)
        np.testing.assert_allclose(
            out["delay_1"].fillna(-1).values,
            expected.fillna(-1).values,
            rtol=1e-6,
        )

    def test_delta_values(self):
        out = self.fengine.compute(self.data, "Delta($close, 3)", name="delta_3")
        expected = self.data["close"] - self.data.groupby("code")["close"].shift(3)
        np.testing.assert_allclose(
            out["delta_3"].fillna(0).values,
            expected.fillna(0).values,
            atol=1e-6,
        )

    def test_rank_values(self):
        out = self.fengine.compute(self.data, "Rank($close)", name="rank_close")
        self.assertTrue((out["rank_close"] >= 0).all())
        self.assertTrue((out["rank_close"] <= 1).all())
        for d, g in out.groupby("date"):
            counts = g["rank_close"].value_counts()
            self.assertGreater(len(counts), 1)

    def test_complex_formula(self):
        out = self.fengine.compute(
            self.data,
            "Rank(Ts_Mean(Delta($close, 1), 5))",
            name="complex"
        )
        self.assertIn("complex", out.columns)
        valid = out["complex"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())

    def test_compute_many(self):
        formulas = {
            "ret_5": "Delta($close, 5) / $close",
            "rank_close": "Rank($close)",
            "ts_mean_5": "Ts_Mean($close, 5)",
        }
        out = self.fengine.compute_many(self.data, formulas)
        self.assertIn("ret_5", out.columns)
        self.assertIn("rank_close", out.columns)
        self.assertIn("ts_mean_5", out.columns)


class TestSecurity(unittest.TestCase):
    """安全相关测试"""

    def setUp(self):
        self.engine = SafeExpressionEngine()
        self.data = make_synth_data()

    def test_cannot_read_globals(self):
        with self.assertRaises(UnsafeExpressionError):
            self.engine.parse("__builtins__")

    def test_cannot_infinite_loop(self):
        with self.assertRaises((UnsafeExpressionError, SyntaxError)):
            self.engine.parse("while True: pass")

    def test_safe_with_arithmetic_on_field(self):
        compiled = self.engine.parse("$close * 2 + 1")
        ctx = ExpressionContext(data=self.data)
        result = self.engine.compute(compiled, ctx)
        expected = (self.data["close"] * 2 + 1).values
        np.testing.assert_allclose(
            np.asarray(result, dtype=float),
            expected,
            rtol=1e-6,
        )


class TestPerformance(unittest.TestCase):
    """性能基准测试"""

    def test_factor_computation_perf(self):
        data = make_synth_data(n_codes=10, n_days=240)
        engine = FactorEngine()

        t0 = time.time()
        out = engine.compute(data, "Rank(Ts_Mean($close, 20))", name="alpha1_like")
        elapsed = time.time() - t0

        self.assertGreater(out.shape[0], 0)
        self.assertLess(
            elapsed, 5.0,
            f"Too slow: {elapsed:.2f}s for 10 codes x 240 days"
        )

    def test_batch_factor_perf(self):
        data = make_synth_data(n_codes=10, n_days=240)
        engine = FactorEngine()
        formulas = {
            "f1": "Rank(Ts_Mean($close, 5))",
            "f2": "Rank(Ts_Mean($close, 10))",
            "f3": "Rank(Ts_Mean($close, 20))",
            "f4": "Delta($close, 5) / $close",
            "f5": "StdDev(Delta($close, 1), 20)",
        }
        t0 = time.time()
        out = engine.compute_many(data, formulas)
        elapsed = time.time() - t0

        self.assertEqual(len([c for c in out.columns if c.startswith("f")]), 5)
        self.assertLess(elapsed, 10.0, f"Too slow: {elapsed:.2f}s for 5 factors")


if __name__ == "__main__":
    unittest.main(verbosity=2)
