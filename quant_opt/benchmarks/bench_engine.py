"""
Benchmark: how fast can the expression engine evaluate the
alpha catalog on a realistic panel?

Run:  PYTHONPATH=/workspace python3 quant_opt/benchmarks/bench_engine.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quant_opt import evaluate_formula, get_catalog
from quant_opt.tests.test_quant_opt import _make_synthetic_data


def main():
    sizes = [
        (50, 252),
        (100, 504),
        (200, 750),
        (500, 1000),
    ]
    catalog = get_catalog()

    results = []
    for n_codes, n_dates in sizes:
        print(f"\n--- n_codes={n_codes}, n_dates={n_dates} ---")
        df = _make_synthetic_data(n_dates=n_dates, n_codes=n_codes, seed=2025)
        df = df.set_index(["code", "date"]).sort_index()

        t0 = time.time()
        for name, formula in catalog:
            evaluate_formula(formula, df)
        elapsed = time.time() - t0
        per_factor = elapsed / len(catalog) * 1000
        print(f"  total: {elapsed:.2f} s   per-factor: {per_factor:.1f} ms")
        results.append({
            "n_codes": n_codes,
            "n_dates": n_dates,
            "n_rows": int(len(df)),
            "n_factors": len(catalog),
            "total_s": round(elapsed, 3),
            "per_factor_ms": round(per_factor, 2),
        })

    print("\nSummary table:")
    print(f"{'n_codes':>8s}  {'n_dates':>8s}  {'n_rows':>10s}  {'total_s':>8s}  {'per_factor_ms':>14s}")
    for r in results:
        print(f"{r['n_codes']:>8d}  {r['n_dates']:>8d}  {r['n_rows']:>10d}  "
              f"{r['total_s']:>8.2f}  {r['per_factor_ms']:>14.2f}")
    return results


if __name__ == "__main__":
    out = main()
    print("\nJSON:")
    print(json.dumps(out, indent=2))
