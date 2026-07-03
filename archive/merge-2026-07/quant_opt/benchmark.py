"""
benchmark.py
============

性能基准测试：对比循环版与向量化版在更大规模数据上的速度。

输出结果到 quant_opt/results/benchmark.json + 终端。
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "quant_opt" / "backtest"))
sys.path.insert(0, str(ROOT / "quant_opt" / "factor"))

from vectorized_adapter import VectorizedAdapter, build_test_data
from ic_analysis_vectorized import batch_ic_analysis

import pandas as pd
import numpy as np


def benchmark_backtest_scaling():
    """回测速度基准：n_stocks × n_days。"""
    print("=" * 60)
    print("Backtest scaling benchmark (vectorized)")
    print("=" * 60)
    sizes = [
        (50, 252),
        (100, 252),
        (200, 504),
        (500, 504),
        (1000, 504),
    ]
    results = []
    print(f"{'n_stocks':<10} {'n_days':<8} {'time (s)':<10} {'equity (M)':<12} {'n_trades':<10}")
    for n_stocks, n_days in sizes:
        data = build_test_data(n_stocks=n_stocks, n_days=n_days, seed=42)
        # 买入前 20% 股票并持有
        top_n = max(1, n_stocks // 5)
        codes = sorted(data["code"].unique())[:top_n]
        signals = []
        for dt in sorted(data["date"].unique()):
            for code in codes:
                signals.append({"date": dt, "code": code, "signal": 1})
        signals = pd.DataFrame(signals)
        adapter = VectorizedAdapter()

        t0 = time.perf_counter()
        result = adapter.run_backtest(data, signals)
        elapsed = time.perf_counter() - t0

        equity = result["equity_curve"]["equity"].iloc[-1]
        n_trades = len(result["trades"])
        results.append({
            "n_stocks": n_stocks, "n_days": n_days,
            "elapsed_s": elapsed,
            "equity_final": equity,
            "n_trades": n_trades,
        })
        print(f"{n_stocks:<10} {n_days:<8} {elapsed:<10.3f} {equity / 1e6:<12.2f} {n_trades:<10}")
    return results


def benchmark_ic_scaling():
    """IC 分析速度基准：n_factors × n_stocks × n_dates。"""
    print("\n" + "=" * 60)
    print("IC analysis scaling benchmark")
    print("=" * 60)
    configs = [
        (10, 100, 252),
        (20, 100, 500),
        (50, 200, 500),
    ]
    results = []
    print(f"{'n_factors':<10} {'n_stocks':<10} {'n_dates':<10} {'time (s)':<10}")
    for n_factors, n_stocks, n_dates in configs:
        dates = pd.bdate_range("2020-01-01", periods=n_dates)
        rows = []
        rng = np.random.default_rng(42)
        for dt in dates:
            for ci in range(n_stocks):
                code = f"{ci:06d}.SH"
                row = {"date": dt, "code": code}
                for fi in range(n_factors):
                    row[f"f{fi}"] = rng.normal(0, 1)
                row["ret_forward_5d"] = rng.normal(0, 0.04)
                rows.append(row)
        df = pd.DataFrame(rows)
        factor_names = [f"f{i}" for i in range(n_factors)]
        t0 = time.perf_counter()
        res = batch_ic_analysis(df, df, factor_names, ["ret_forward_5d"], min_obs=20)
        elapsed = time.perf_counter() - t0
        results.append({
            "n_factors": n_factors, "n_stocks": n_stocks, "n_dates": n_dates,
            "elapsed_s": elapsed,
        })
        print(f"{n_factors:<10} {n_stocks:<10} {n_dates:<10} {elapsed:<10.3f}")
    return results


def main():
    out_dir = ROOT / "quant_opt" / "results"
    out_dir.mkdir(exist_ok=True)

    bt_results = benchmark_backtest_scaling()
    ic_results = benchmark_ic_scaling()

    with open(out_dir / "benchmark.json", "w") as f:
        json.dump({
            "backtest_scaling": bt_results,
            "ic_scaling": ic_results,
        }, f, indent=2)
    print(f"\nBenchmark results saved to {out_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
