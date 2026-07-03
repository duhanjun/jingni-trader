"""
性能对比测试

对比 Polars 向量化实现与原 pandas 逐日循环实现的性能差异。
测试不同数据规模下的加速比。
"""
import sys
import os
import time
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optimizations.polars_ic_analysis import calc_ic_series_polars
from optimizations.polars_neutralize import (
    neutralize_mcap_polars,
    neutralize_industry_mcap_polars,
)
from optimizations.vectorized_metrics import calc_enhanced_metrics
from optimizations.tests import generate_panel_data, generate_equity_curve
from optimizations.tests.test_correctness import (
    ref_calc_ic_pandas,
    ref_neutralize_mcap_pandas,
    ref_neutralize_industry_mcap_pandas,
)


def _time_fn(fn, *args, repeat=3, **kwargs):
    """计时函数，取中位数"""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return np.median(times)


def test_ic_performance():
    """IC 分析性能对比"""
    print("\n--- IC 分析性能对比 ---")
    print(f"{'规模(股票×天)':<20} {'pandas(s)':<12} {'polars(s)':<12} {'加速比':<10}")
    print("-" * 54)

    for n_stocks, n_days in [(100, 100), (300, 250), (500, 250), (1000, 250)]:
        data = generate_panel_data(n_stocks=n_stocks, n_days=n_days, seed=42)
        data_pd = data.to_pandas()

        t_pandas = _time_fn(
            ref_calc_ic_pandas, data_pd, "factor_1", "ret_forward_5d", "spearman", repeat=2
        )
        t_polars = _time_fn(
            calc_ic_series_polars, data, "factor_1", "ret_forward_5d", "spearman", repeat=3
        )

        speedup = t_pandas / t_polars if t_polars > 0 else float("inf")
        label = f"{n_stocks}×{n_days}"
        print(f"{label:<20} {t_pandas:<12.4f} {t_polars:<12.4f} {speedup:<10.1f}x")

        # Polars 应该更快（至少在较大规模下）
        if n_stocks >= 300:
            assert speedup > 1.0, f"Polars 未实现加速: {speedup}x"


def test_neutralize_mcap_performance():
    """市值中性化性能对比"""
    print("\n--- 市值中性化性能对比 ---")
    print(f"{'规模(股票×天)':<20} {'pandas(s)':<12} {'polars(s)':<12} {'加速比':<10}")
    print("-" * 54)

    for n_stocks, n_days in [(100, 100), (300, 250), (500, 250), (1000, 250)]:
        data = generate_panel_data(n_stocks=n_stocks, n_days=n_days, seed=42)
        data_pd = data.to_pandas().sort_values(["date", "code"]).reset_index(drop=True)

        t_pandas = _time_fn(
            ref_neutralize_mcap_pandas, data_pd, "factor_1", "lncap", repeat=2
        )
        t_polars = _time_fn(
            neutralize_mcap_polars, data, "factor_1", "lncap", repeat=3
        )

        speedup = t_pandas / t_polars if t_polars > 0 else float("inf")
        label = f"{n_stocks}×{n_days}"
        print(f"{label:<20} {t_pandas:<12.4f} {t_polars:<12.4f} {speedup:<10.1f}x")

        if n_stocks >= 300:
            assert speedup > 1.0, f"Polars 未实现加速: {speedup}x"


def test_neutralize_industry_mcap_performance():
    """行业+市值中性化性能对比"""
    print("\n--- 行业+市值中性化性能对比 (FWL 向量化 vs 逐日 OLS) ---")
    print(f"{'规模(股票×天)':<20} {'pandas(s)':<12} {'polars(s)':<12} {'加速比':<10}")
    print("-" * 54)

    for n_stocks, n_days in [(100, 100), (300, 250), (500, 250), (1000, 250)]:
        data = generate_panel_data(n_stocks=n_stocks, n_days=n_days, seed=42)
        data_pd = data.to_pandas().sort_values(["date", "code"]).reset_index(drop=True)

        t_pandas = _time_fn(
            ref_neutralize_industry_mcap_pandas,
            data_pd, "factor_1", "industry", "lncap", repeat=2
        )
        t_polars = _time_fn(
            neutralize_industry_mcap_polars,
            data, "factor_1", "industry", "lncap", repeat=3
        )

        speedup = t_pandas / t_polars if t_polars > 0 else float("inf")
        label = f"{n_stocks}×{n_days}"
        print(f"{label:<20} {t_pandas:<12.4f} {t_polars:<12.4f} {speedup:<10.1f}x")

        if n_stocks >= 300:
            assert speedup > 1.0, f"Polars 未实现加速: {speedup}x"


def test_metrics_performance():
    """绩效指标计算性能对比（增强版 vs 原基础版）"""
    print("\n--- 绩效指标计算性能对比 ---")
    print(f"{'天数':<12} {'基础版(s)':<14} {'增强版(s)':<14} {'指标数':<10}")
    print("-" * 50)

    def ref_basic_metrics(eq):
        """原 _calc_metrics 参考实现"""
        returns = eq.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
        win_rate = (returns > 0).mean()
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        return {
            "total_return": total_return, "annual_return": annual_return,
            "volatility": volatility, "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown, "win_rate": win_rate,
            "calmar_ratio": calmar,
        }

    for n_days in [250, 500, 1000, 2500]:
        eq_df = generate_equity_curve(n_days=n_days, seed=42)
        eq = eq_df.set_index("date")["equity"]

        t_basic = _time_fn(ref_basic_metrics, eq, repeat=5)
        t_enhanced = _time_fn(calc_enhanced_metrics, eq, repeat=5)
        n_basic = len(ref_basic_metrics(eq))
        n_enhanced = len(calc_enhanced_metrics(eq))

        print(f"{n_days:<12} {t_basic:<14.6f} {t_enhanced:<14.6f} {n_enhanced:<10}")

    print(f"\n  增强版指标数: {len(calc_enhanced_metrics(generate_equity_curve().set_index('date')['equity']))} 个 (原基础版 7 个)")


def run_all_performance_tests():
    """运行所有性能测试"""
    print("=" * 60)
    print("性能对比测试")
    print("=" * 60)
    results = {}
    passed = 0
    failed = 0

    for name, test_fn in [
        ("IC 分析", test_ic_performance),
        ("市值中性化", test_neutralize_mcap_performance),
        ("行业+市值中性化", test_neutralize_industry_mcap_performance),
        ("绩效指标", test_metrics_performance),
    ]:
        try:
            print(f"\n[测试] {name}")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"性能测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return passed, failed


if __name__ == "__main__":
    run_all_performance_tests()
