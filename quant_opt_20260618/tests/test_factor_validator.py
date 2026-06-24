"""
Tests for the factor validator
===============================
"""
import unittest

import numpy as np
import pandas as pd

from quant_opt_20260618.factor_validator import validate_factor, _bootstrap_pvalue


def _panel(n_dates=200, n_stocks=40, signal=0.3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    rows = []
    for c in codes:
        f = rng.standard_normal(n_dates)
        r = rng.standard_normal(n_dates) + signal * f
        for i, d in enumerate(dates):
            rows.append({"code": c, "date": d, "factor": f[i], "ret": r[i]})
    return pd.DataFrame(rows)


class TestValidator(unittest.TestCase):

    def test_strong_signal_accepts(self):
        df = _panel(signal=0.5)
        v = validate_factor(df["factor"], df["ret"], df["date"], n_bootstrap=200)
        self.assertIn(v.decision, ("ACCEPT", "REVIEW"))
        self.assertGreater(abs(v.ic_ir), 0.3)
        self.assertLess(v.bootstrap_p, 0.10)

    def test_pure_noise_rejects(self):
        df = _panel(signal=0.0)
        v = validate_factor(df["factor"], df["ret"], df["date"], n_bootstrap=200)
        # 无信号时应该落入 REJECT 或 REVIEW
        self.assertIn(v.decision, ("REJECT", "REVIEW"))
        # bootstrap p 接近 0.5
        self.assertGreater(v.bootstrap_p, 0.05)

    def test_short_series_high_p(self):
        df = _panel(n_dates=15, n_stocks=10, signal=0.5)
        v = validate_factor(df["factor"], df["ret"], df["date"], n_bootstrap=200)
        # 数据太少 -> p=1
        self.assertEqual(v.bootstrap_p, 1.0)

    def test_bootstrap_pvalue_deterministic_with_seed(self):
        ic = pd.Series(np.random.default_rng(0).standard_normal(100))
        a = _bootstrap_pvalue(ic, n_bootstrap=200, block_size=5, seed=42)
        b = _bootstrap_pvalue(ic, n_bootstrap=200, block_size=5, seed=42)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)