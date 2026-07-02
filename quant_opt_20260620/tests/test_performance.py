"""
性能对比测试: 向量化 vs 逐日循环 (回测引擎 + 因子中性化 + IC分析)

每个测试返回 dict: {name, passed, details, metrics}
"""
from __future__ import annotations
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data import generate_panel, generate_signals, generate_factor_panel
from vectorized_backtest import BaselineLoopAdapter, LookaheadFixedAdapter, VectorizedAdapter
from vectorized_factor import (
    neutralize_loop,
    neutralize_vectorized,
    ic_analysis_loop,
    ic_analysis_vectorized,
    time_function,
)


def test_backtest_performance() -> dict:
    """回测引擎性能对比: 基线(逐日循环) vs 修复版(逐日循环) vs 向量化版"""
    panel = generate_panel(n_codes=80, n_days=300, seed=42)
    signals = generate_signals(panel, strategy="reversal", top_pct=0.2, seed=42)

    runs = 2
    t0 = time.perf_counter()
    for _ in range(runs):
        BaselineLoopAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    base_t = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    fixed_t = (time.perf_counter() - t0) / runs

    t0 = time.perf_counter()
    for _ in range(runs):
        VectorizedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
    vect_t = (time.perf_counter() - t0) / runs

    speedup_vs_base = base_t / vect_t if vect_t > 0 else 0
    speedup_vs_fixed = fixed_t / vect_t if vect_t > 0 else 0

    # 向量化应显著快于逐日循环
    passed = vect_t < fixed_t and speedup_vs_fixed > 1.5
    return {
        "name": "回测引擎性能 (80股票×300天)",
        "passed": passed,
        "details": (
            f"基线(逐日循环)={base_t*1000:.1f}ms; "
            f"修复版(逐日循环)={fixed_t*1000:.1f}ms; "
            f"向量化版={vect_t*1000:.1f}ms; "
            f"加速比(向量化/基线)={speedup_vs_base:.1f}x, "
            f"加速比(向量化/修复版)={speedup_vs_fixed:.1f}x. "
            f"{'✓ 向量化显著更快' if passed else '✗ 加速不明显'}"
        ),
        "metrics": {
            "baseline_ms": base_t * 1000,
            "fixed_ms": fixed_t * 1000,
            "vectorized_ms": vect_t * 1000,
            "speedup_vs_baseline": speedup_vs_base,
            "speedup_vs_fixed": speedup_vs_fixed,
        },
    }


def test_backtest_scalability() -> dict:
    """规模扩展性: 股票数翻倍时各版本耗时增长"""
    results = {}
    for n_codes in [40, 80, 160]:
        panel = generate_panel(n_codes=n_codes, n_days=200, seed=42)
        signals = generate_signals(panel, strategy="reversal", top_pct=0.2, seed=42)

        t0 = time.perf_counter()
        LookaheadFixedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
        fixed_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        VectorizedAdapter().run_backtest(data=panel, signals=signals, t_plus_1=True, price_limit=True)
        vect_t = time.perf_counter() - t0

        results[n_codes] = {"fixed_ms": fixed_t * 1000, "vect_ms": vect_t * 1000,
                            "speedup": fixed_t / vect_t if vect_t > 0 else 0}

    # 向量化版在更大规模下加速比应更高 (循环版 O(n) 增长更陡)
    speedup_40 = results[40]["speedup"]
    speedup_160 = results[160]["speedup"]
    passed = speedup_160 >= speedup_40
    return {
        "name": "回测规模扩展性 (40→80→160股票)",
        "passed": passed,
        "details": (
            f"40股: 修复版={results[40]['fixed_ms']:.0f}ms, 向量化={results[40]['vect_ms']:.0f}ms, "
            f"加速={speedup_40:.1f}x; "
            f"160股: 修复版={results[160]['fixed_ms']:.0f}ms, 向量化={results[160]['vect_ms']:.0f}ms, "
            f"加速={speedup_160:.1f}x. "
            f"{'✓ 规模越大向量化优势越明显' if passed else '△ 加速比未随规模提升'}"
        ),
        "metrics": results,
    }


def test_neutralize_performance() -> dict:
    """因子中性化性能: 逐日循环(sklearn) vs 向量化(groupby+lstsq)"""
    panel = generate_panel(n_codes=60, n_days=200, seed=42)
    factor_df = generate_factor_panel(panel, n_extra_factors=3, seed=42)
    factor_cols = ["reversal_20d", "volatility_20d", "turnover_20d"]

    res_loop = time_function(neutralize_loop, 2, factor_df, factor_cols, True, True)
    res_vect = time_function(neutralize_vectorized, 2, factor_df, factor_cols, True, True)

    loop_t = res_loop["mean_time"]
    vect_t = res_vect["mean_time"]
    speedup = loop_t / vect_t if vect_t > 0 else 0

    # 正确性: 两版残差相关性应高
    common_cols = [f"{c}_neutral" for c in factor_cols]
    corrs = []
    for col in common_cols:
        if col in res_loop["result"].columns and col in res_vect["result"].columns:
            a = res_loop["result"][col].dropna()
            b = res_vect["result"][col].reindex(a.index).dropna()
            common = a.index.intersection(b.index)
            if len(common) > 10:
                corrs.append(float(a.loc[common].corr(b.loc[common])))

    avg_corr = float(np.mean(corrs)) if corrs else 0.0
    passed = speedup > 1.0 and avg_corr > 0.95
    return {
        "name": "因子中性化性能 (60股票×200天×3因子)",
        "passed": passed,
        "details": (
            f"逐日循环(sklearn)={loop_t*1000:.0f}ms; "
            f"向量化(groupby+lstsq)={vect_t*1000:.0f}ms; "
            f"加速={speedup:.1f}x; 残差相关性={avg_corr:.4f}. "
            f"{'✓ 向量化更快且结果一致' if passed else '✗ 未达预期'}"
        ),
        "metrics": {"loop_ms": loop_t * 1000, "vect_ms": vect_t * 1000,
                    "speedup": speedup, "residual_corr": avg_corr},
    }


def test_ic_analysis_performance() -> dict:
    """IC分析性能: 逐日循环(scipy) vs 向量化(groupby+corr)"""
    panel = generate_panel(n_codes=60, n_days=200, seed=42)
    factor_df = generate_factor_panel(panel, n_extra_factors=2, seed=42)
    # 构造前向收益
    price = panel.sort_values(["code", "date"]).copy()
    price["ret_forward_5d"] = price.groupby("code")["close"].transform(lambda x: x.shift(-5) / x - 1)
    data = factor_df.merge(price[["code", "date", "ret_forward_5d"]], on=["code", "date"], how="inner")
    factor_names = ["reversal_20d", "volatility_20d", "turnover_20d"]

    res_loop = time_function(ic_analysis_loop, 2, data, factor_names, "ret_forward_5d", "spearman")
    res_vect = time_function(ic_analysis_vectorized, 2, data, factor_names, "ret_forward_5d", "spearman")

    loop_t = res_loop["mean_time"]
    vect_t = res_vect["mean_time"]
    speedup = loop_t / vect_t if vect_t > 0 else 0

    # 正确性: IC均值应接近
    ic_diffs = []
    for f in factor_names:
        if f in res_loop["result"] and f in res_vect["result"]:
            ic_diffs.append(abs(res_loop["result"][f]["ic_mean"] - res_vect["result"][f]["ic_mean"]))
    max_diff = float(max(ic_diffs)) if ic_diffs else 1.0

    passed = speedup > 1.0 and max_diff < 0.01
    return {
        "name": "IC分析性能 (60股票×200天×3因子)",
        "passed": passed,
        "details": (
            f"逐日循环(scipy)={loop_t*1000:.0f}ms; "
            f"向量化(groupby+corr)={vect_t*1000:.0f}ms; "
            f"加速={speedup:.1f}x; IC均值最大差异={max_diff:.6f}. "
            f"{'✓ 向量化更快且IC一致' if passed else '✗ 未达预期'}"
        ),
        "metrics": {"loop_ms": loop_t * 1000, "vect_ms": vect_t * 1000,
                    "speedup": speedup, "max_ic_diff": max_diff},
    }


def run_all() -> list:
    return [
        test_backtest_performance(),
        test_backtest_scalability(),
        test_neutralize_performance(),
        test_ic_analysis_performance(),
    ]
