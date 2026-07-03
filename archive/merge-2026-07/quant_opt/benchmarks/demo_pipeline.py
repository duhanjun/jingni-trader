"""
End-to-end demonstration: build 30+ alpha factors via the expression
engine, run IC analysis using the vectorized IC module, and
leakage-check the resulting panel.

Run:  PYTHONPATH=/workspace python3 quant_opt/benchmarks/demo_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quant_opt import (
    evaluate_formula,
    get_catalog,
    batch_ic,
    rank_ic_decay,
    leakage_check,
    TimeSeriesCV,
    PurgedKFold,
)
from quant_opt.tests.test_quant_opt import _make_synthetic_data


def main():
    print("=" * 70)
    print(" End-to-end factor pipeline demo (Qlib-inspired)")
    print("=" * 70)

    # 1) build data
    print("\n[1] Building OHLCV panel (50 stocks x 252 days)...")
    t0 = time.time()
    df = _make_synthetic_data(n_dates=252, n_codes=50, seed=99)
    df = df.set_index(["code", "date"]).sort_index()
    print(f"    shape={df.shape}  build_time={time.time() - t0:.2f} s")

    # 2) evaluate the alpha catalog
    print("\n[2] Evaluating alpha catalog...")
    catalog = get_catalog()
    t0 = time.time()
    factor_data = {}
    for name, formula in catalog:
        try:
            factor_data[name] = evaluate_formula(formula, df)
        except Exception as e:  # noqa: BLE001
            print(f"    ! {name} failed: {e}")
    elapsed = time.time() - t0
    print(f"    computed {len(factor_data)} factors in {elapsed:.2f} s")

    factor_df = pd.DataFrame(factor_data, index=df.index)
    # add a forward return column
    target = df["close"].groupby(level="code").pct_change().shift(-1)
    target.name = "ret_fwd_1d"

    # 3) leakage check
    print("\n[3] Leakage check...")
    rep = leakage_check(df.reset_index())
    print(f"    {rep}")

    # 4) IC summary
    print("\n[4] IC summary (Spearman, top 10 by |IC_IR|):")
    ic_results = batch_ic(factor_df, target, method="spearman")
    ic_table = pd.DataFrame(ic_results).T
    ic_table["abs_ic_ir"] = ic_table["ic_ir"].abs()
    ic_table = ic_table.sort_values("abs_ic_ir", ascending=False)
    print(ic_table[["ic_mean", "ic_std", "ic_ir", "ic_positive_ratio", "n_periods"]]
          .head(10).to_string())

    # 5) decay analysis on the best factor
    best_name = ic_table.index[0]
    print(f"\n[5] IC decay for best factor ({best_name}):")
    decay = rank_ic_decay(factor_df[best_name], target, max_lag=10)
    print(decay.to_string())

    # 6) time-series CV sanity
    print("\n[6] Time-series CV (train=120, valid=30, test=30, step=30, purge=5):")
    dates = sorted(df.index.get_level_values("date").unique())
    cv = TimeSeriesCV(train_size=120, valid_size=30, test_size=30, step=30, purge_gap=5)
    splits = list(cv.split(dates))
    print(f"    n_splits={len(splits)}")
    for s in splits[:3]:
        print(f"    fold {s.fold_id}: train {s.train_period[0].date()}→{s.train_period[1].date()}, "
              f"valid {s.valid_period[0].date()}→{s.valid_period[1].date()}, "
              f"test  {s.test_period[0].date()}→{s.test_period[1].date()}")

    # 7) Purged K-Fold sanity
    print("\n[7] Purged K-Fold (5 splits, purge_gap=3):")
    pkf = PurgedKFold(n_splits=5, purge_gap=3)
    folds = list(pkf.split(len(dates)))
    print(f"    n_folds={len(folds)}")
    for f in folds:
        print(f"    fold {f.fold_id}: train_size={len(f.train_idx)} test_size={len(f.test_idx)}")

    return {
        "n_factors": len(factor_data),
        "n_dates": int(len(dates)),
        "n_codes": int(df.index.get_level_values("code").nunique()),
        "build_time_s": float(elapsed),
        "best_factor": best_name,
        "best_factor_ic_ir": float(ic_table.loc[best_name, "ic_ir"]),
        "n_splits_ts_cv": int(len(splits)),
        "n_folds_purged": int(len(folds)),
        "leakage_clean": bool(rep.is_clean),
    }


if __name__ == "__main__":
    result = main()
    print("\nSummary JSON:")
    print(json.dumps(result, indent=2))
