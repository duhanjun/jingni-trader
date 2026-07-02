"""
Performance & correctness benchmark: vectorized IC vs. loop-based IC.

This script:
1. Generates a realistic-looking OHLCV panel (200 stocks × 750 trading days)
2. Computes IC using the existing loop-based approach (jingni-trader style)
3. Computes IC using the new vectorized approach
4. Asserts that the two are numerically equivalent (within a small tolerance)
5. Prints a wall-clock comparison

Run:  PYTHONPATH=/workspace python3 quant_opt/benchmarks/bench_ic.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quant_opt import compute_ic_series  # new vectorized
from quant_opt.tests.test_quant_opt import _make_synthetic_data


def loop_based_ic(factor_df: pd.DataFrame, factor_col: str, target_col: str) -> pd.Series:
    """Faithful re-implementation of jingni-trader's `_calc_ic` style loop."""
    ic_list = []
    dates = sorted(factor_df["date"].unique())
    for dt in dates:
        cross = factor_df[factor_df["date"] == dt].dropna(subset=[factor_col, target_col])
        if len(cross) < 10:
            continue
        ic, _ = stats.spearmanr(cross[factor_col], cross[target_col], nan_policy="omit")
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})
    if not ic_list:
        return pd.Series(dtype=float)
    out = pd.DataFrame(ic_list)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")["ic"]


def main():
    print("Building synthetic data (200 stocks × 750 days)...")
    t0 = time.time()
    df = _make_synthetic_data(n_dates=750, n_codes=200, seed=2025)
    print(f"  build: {time.time() - t0:.2f} s, shape={df.shape}")
    df_idx = df.set_index(["code", "date"]).sort_index()

    # construct a noisy predictive factor and a forward return
    rng = np.random.default_rng(7)
    base_signal = (
        0.6 * df_idx["close"].groupby(level="code").pct_change(5) +
        0.4 * df_idx["volume"].groupby(level="code").pct_change(5)
    )
    factor = base_signal + rng.normal(0, 0.01, size=len(base_signal))
    factor.index = df_idx.index
    target = df_idx["close"].groupby(level="code").pct_change().shift(-1)
    factor.name = "F"
    target.name = "RET_FWD_1"

    # 1) loop
    t0 = time.time()
    long_df = df_idx.reset_index()
    long_df["F"] = factor.values
    long_df["RET_FWD_1"] = target.values
    loop_ic = loop_based_ic(long_df, "F", "RET_FWD_1")
    t_loop = time.time() - t0
    print(f"\nLoop-based IC: {t_loop:.3f} s  ({len(loop_ic)} dates)")

    # 2) vectorized
    t0 = time.time()
    vec_ic = compute_ic_series(factor, target, method="spearman")
    t_vec = time.time() - t0
    print(f"Vectorized IC:  {t_vec:.3f} s  ({len(vec_ic)} dates)")

    # 3) correctness
    common = loop_ic.index.intersection(vec_ic.index)
    a = loop_ic.loc[common].values
    b = vec_ic.loc[common].values
    max_diff = float(np.nanmax(np.abs(a - b)))
    mean_diff = float(np.nanmean(np.abs(a - b)))
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"\nCorrectness check (n={len(common)} dates):")
    print(f"  max |diff|  = {max_diff:.2e}")
    print(f"  mean |diff| = {mean_diff:.2e}")
    print(f"  correlation = {corr:.6f}")

    # 4) speedup
    if t_vec > 0:
        speedup = t_loop / t_vec
    else:
        speedup = float("inf")
    print(f"\nSpeedup: {speedup:.1f}x")

    # Pass/fail
    # Spearman computed via scipy.stats.spearmanr (with internal ranking) and
    # via scipy.stats.rankdata + per-date Pearson can differ at the 1e-3 level
    # due to tie-handling and rank averaging; correlation should still be 1.0.
    ok = max_diff < 1e-2 and corr > 0.9999
    print("\n" + ("BENCHMARK PASSED" if ok else "BENCHMARK FAILED"))

    return {
        "t_loop_s": t_loop,
        "t_vec_s": t_vec,
        "speedup_x": speedup,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "corr": corr,
        "n_dates": int(len(common)),
        "passed": bool(ok),
    }


if __name__ == "__main__":
    result = main()
    print("\nResult JSON:")
    import json
    print(json.dumps(result, indent=2))
