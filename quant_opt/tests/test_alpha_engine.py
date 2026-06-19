"""alpha_expression_engine 单元测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
import pandas as pd

from quant_opt.alpha_expression_engine import (
    ExpressionEngine, EvalContext, compute_factors, list_builtin_factors
)


def make_data(n_stocks: int = 5, n_days: int = 60) -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for s in range(n_stocks):
        close = 10 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
        for i, d in enumerate(dates):
            p = close[i]
            rows.append({
                "code": f"{s:06d}.SH",
                "date": d,
                "open": p * 0.99,
                "high": p * 1.01,
                "low": p * 0.98,
                "close": p,
                "volume": int(1e6 * np.random.uniform(0.5, 2)),
                "amount": p * int(1e6),
                "turnover_rate": np.random.uniform(0.5, 5),
                "change_pct": np.random.uniform(-3, 3),
            })
    return pd.DataFrame(rows)


class TestAlphaEngine(unittest.TestCase):

    def test_field_access(self):
        data = make_data()
        ctx = EvalContext(data=data)
        engine = ExpressionEngine()
        s = engine.eval("$close", ctx)
        self.assertEqual(len(s), len(data))

    def test_simple_arithmetic(self):
        data = make_data()
        ctx = EvalContext(data=data)
        engine = ExpressionEngine()
        s = engine.eval("$close - $open", ctx)
        self.assertEqual(len(s), len(data))
        # 中位数应 >= 0 (close >= open 概率高)
        self.assertGreater(s.median(), -1.0)

    def test_ret_builtin(self):
        data = make_data()
        ctx = EvalContext(data=data)
        engine = ExpressionEngine()
        s = engine.eval("Ret($close, 5)", ctx)
        self.assertEqual(len(s), len(data))
        # 前 5 个应该是 NaN
        self.assertTrue(s.head(5).isna().all())

    def test_nested_expression(self):
        data = make_data()
        ctx = EvalContext(data=data)
        engine = ExpressionEngine()
        s = engine.eval("Rank(Delta($close, 1))", ctx)
        self.assertEqual(len(s), len(data))

    def test_builtin_factors(self):
        data = make_data()
        expressions = {
            "ret_1d": "ret_1d",
            "ret_5d": "ret_5d",
            "vol_20d": "volatility_20d",
        }
        df, results = compute_factors(data, expressions)
        self.assertEqual(df.shape[1], 2 + 3)  # code, date, 3 factors
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertGreater(len(r.series), 0)

    def test_security_safety(self):
        engine = ExpressionEngine()
        with self.assertRaises(ValueError):
            engine.parse("__import__('os').system('rm -rf /')")
        with self.assertRaises(ValueError):
            engine.parse("open('x', 'w')")

    def test_unregistered_operator(self):
        data = make_data()
        ctx = EvalContext(data=data)
        engine = ExpressionEngine()
        with self.assertRaises(ValueError):
            engine.eval("BadOp($close, 1)", ctx)

    def test_custom_factor(self):
        engine = ExpressionEngine(custom_factors={"my_factor": "Ret($close, 3)"})
        data = make_data()
        ctx = EvalContext(data=data)
        s = engine.eval("my_factor", ctx)
        self.assertEqual(len(s), len(data))

    def test_list_builtin_factors(self):
        factors = list_builtin_factors()
        self.assertGreater(len(factors), 10)
        self.assertIn("ret_20d", factors)
        self.assertIn("volatility_20d", factors)


if __name__ == "__main__":
    unittest.main()
