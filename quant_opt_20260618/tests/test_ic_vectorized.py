"""
Tests for the vectorized IC module
===================================
"""
import time
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from quant_opt_20260618.ic_vectorized import (
    ic_series_pearson,
    ic_series_spearman,
    ic_summary,
    ic_analysis_batch,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _build_panel(n_dates: int = 60, n_stocks: int = 30,
                 signal_strength: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    codes = [f"S{i:03d}" for i in range(n_stocks)]
    rows = []
    for c in codes:
        f = rng.standard_normal(n_dates)
        r = rng.standard_normal(n_dates) + signal_strength * f
        for i in range(n_dates):
            rows.append({"code": c, "date": dates[i],
                         "factor": f[i], "ret": r[i]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


class TestICVectorizedCorrectness(unittest.TestCase):

    def test_pearson_matches_scipy(self):
        df = _build_panel()
        # reference: loop with scipy
        dates = sorted(df["date"].unique())
        ref = {}
        for dt in dates:
            sub = df[df["date"] == dt].dropna(subset=["factor", "ret"])
            if len(sub) < 10:
                continue
            ic, _ = stats.pearsonr(sub["factor"], sub["ret"])
            ref[dt] = ic

        ours = ic_series_pearson(df["factor"], df["ret"], df["date"])
        for dt, expected in ref.items():
            self.assertAlmostEqual(ours.loc[dt], expected, places=6,
                                   msg=f"mismatch at {dt}")

    def test_spearman_matches_scipy(self):
        df = _build_panel()
        dates = sorted(df["date"].unique())
        ref = {}
        for dt in dates:
            sub = df[df["date"] == dt].dropna(subset=["factor", "ret"])
            if len(sub) < 10:
                continue
            ic, _ = stats.spearmanr(sub["factor"], sub["ret"])
            ref[dt] = ic

        ours = ic_series_spearman(df["factor"], df["ret"], df["date"])
        for dt, expected in ref.items():
            self.assertAlmostEqual(ours.loc[dt], expected, places=4,
                                   msg=f"mismatch at {dt}")

    def test_summary_keys(self):
        df = _build_panel()
        ic = ic_series_pearson(df["factor"], df["ret"], df["date"])
        s = ic_summary(ic)
        for k in ("ic_mean", "ic_std", "ic_ir", "ic_positive_ratio", "ic_t_stat"):
            self.assertIn(k, s)
            self.assertIsInstance(s[k], float)

    def test_batch(self):
        df = _build_panel()
        # fabricate a second factor
        df["factor2"] = df["factor"] * 0.5 + np.random.default_rng(1).standard_normal(len(df)) * 0.1
        result = ic_analysis_batch(df, df["ret"], ["factor", "factor2"])
        self.assertEqual(set(result.keys()), {"factor", "factor2"})

    def test_min_obs_threshold(self):
        # Build a panel where some dates have < 10 obs to ensure they are dropped.
        df = _build_panel(n_dates=10, n_stocks=2)  # only 2 stocks
        ic = ic_series_pearson(df["factor"], df["ret"], df["date"], min_obs=10)
        # All NaN — no date has >= 10 obs
        self.assertTrue(ic.isna().all())

    def test_constant_cross_section_returns_nan(self):
        # build a date where factor is constant
        rows = []
        for d in range(3):
            for s in range(20):
                rows.append({"code": f"S{s}", "date": pd.Timestamp(f"2024-01-{d+1}"),
                             "factor": 1.0 if d == 1 else float(s),
                             "ret": float(s)})
        df = pd.DataFrame(rows)
        ic = ic_series_pearson(df["factor"], df["ret"], df["date"], min_obs=5)
        # the constant day should be NaN
        self.assertTrue(pd.isna(ic.iloc[1]))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestICVectorizedEdges(unittest.TestCase):

    def test_empty_input(self):
        df = pd.DataFrame({"code": [], "date": [], "factor": [], "ret": []})
        ic = ic_series_pearson(df["factor"], df["ret"], df["date"])
        self.assertEqual(len(ic), 0)

    def test_nan_in_factor_dropped(self):
        df = _build_panel()
        df.loc[df.index[:5], "factor"] = np.nan
        ic = ic_series_pearson(df["factor"], df["ret"], df["date"])
        # Should not raise
        self.assertGreater(len(ic), 0)

    def test_misaligned_index(self):
        # Shuffle the rows of the whole frame — factor, ret, date all move together
        df = _build_panel()
        df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
        ic = ic_series_pearson(df_shuffled["factor"], df_shuffled["ret"], df_shuffled["date"])
        ref = ic_series_pearson(df["factor"], df["ret"], df["date"])
        pd.testing.assert_series_equal(ic.sort_index(), ref.sort_index(),
                                       check_names=False)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestICVectorizedPerformance(unittest.TestCase):

    def test_speedup_vs_loop(self):
        # 5,000 stocks × 250 dates — typical A-share universe
        df = _build_panel(n_dates=250, n_stocks=5000, seed=2)
        # Sample a subset for the loop test (loop is much slower)
        n_loop_dates = 40
        df_loop = df[df["date"] < df["date"].unique()[n_loop_dates]]

        # reference loop
        t0 = time.perf_counter()
        for dt in df_loop["date"].unique():
            sub = df_loop[df_loop["date"] == dt]
            if len(sub) >= 10:
                stats.pearsonr(sub["factor"], sub["ret"])
        t_loop = time.perf_counter() - t0

        # vectorized on the full panel
        t0 = time.perf_counter()
        ic_series_pearson(df["factor"], df["ret"], df["date"])
        t_vec = time.perf_counter() - t0

        # Extrapolate the loop cost to the full panel for an honest comparison
        t_loop_extrap = t_loop * (df["date"].nunique() / n_loop_dates)
        speedup = t_loop_extrap / max(t_vec, 1e-9)
        print(f"\n[perf] loop(extrap)={t_loop_extrap:.2f}s  vec={t_vec:.3f}s  speedup={speedup:.1f}x")
        # We expect at least 3× speedup; the win is in correctness + per-call overhead
        # (the pure-numpy loop is the lower bound — production loops with scipy.stats
        # tend to be much slower than pure-numpy Pearson).
        self.assertGreater(speedup, 3.0,
                           f"vectorized version should be at least 3× faster, got {speedup:.1f}x")


if __name__ == "__main__":
    unittest.main(verbosity=2)
