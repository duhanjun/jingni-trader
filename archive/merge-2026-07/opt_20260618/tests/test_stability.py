"""
测试 4: 多源稳健性测试

验证目标：
1. 时间窗口切片测试能在多个子区间独立运行回测
2. 参数扫描能枚举 param_grid 笛卡尔积
3. 股票池抽样能产生多样化的回测结果
4. Bootstrap 块状抽样能产生稳定的分布
5. 稳定性评分能输出有意义的指标
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from stability_test import StabilityTester, StabilityResult
from strategy_api import TopkDropoutStrategy
from vectorized_backtest import VectorizedBacktestEngine
from synthetic_data import generate_synthetic_data


def build_backtest_fn(topk: int = 10):
    """构造一个闭包回测函数，供 StabilityTester 调用"""
    strategy = TopkDropoutStrategy(topk=topk)
    engine = VectorizedBacktestEngine()

    def fn(data, factors, params=None):
        if params and "topk" in params:
            nonlocal_strategy = TopkDropoutStrategy(**{k: v for k, v in params.items() if k in ["topk", "n_drop"]})
            signals = nonlocal_strategy.generate_signals(data, factors)
        else:
            signals = strategy.generate_signals(data, factors)
        return engine.run(data, signals)
    return fn


def test_time_window_test():
    """时间窗口切片测试"""
    print("=" * 60)
    print("测试 4.1: 时间窗口切片")
    print("=" * 60)
    # 生成 2 年数据
    data, factors = generate_synthetic_data(n_stocks=40, n_days=500, seed=42)

    fn = build_backtest_fn(topk=10)
    tester = StabilityTester(backtest_fn=fn)
    tester.add_time_window_test(
        start="2024-01-01", end="2025-12-31",
        window_months=6, step_months=3,
    )
    results = tester.run(data, factors)
    print(f"  生成 {len(results)} 个时间窗口回测结果")

    valid = [r for r in results if r.metrics]
    print(f"  有效结果: {len(valid)}")

    # 检查返回的窗口数
    assert len(results) >= 2, f"应至少 2 个窗口，实际 {len(results)}"
    # 检查每个窗口的 metrics 完整
    for r in valid[:3]:
        print(f"  {r.name}: sharpe={r.metrics.get('sharpe_ratio', 0):.3f}, "
              f"年化={r.metrics.get('annual_return', 0):.3%}")
    print("  ✓ 时间窗口切片测试正常")
    return results


def test_param_sweep_test():
    """参数扫描测试"""
    print("=" * 60)
    print("测试 4.2: 参数扫描")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=40, n_days=200, seed=42)

    fn = build_backtest_fn()
    tester = StabilityTester(backtest_fn=fn)
    tester.add_param_sweep_test({"topk": [5, 10, 20, 30]})
    results = tester.run(data, factors)
    print(f"  生成 {len(results)} 个参数组合结果")
    assert len(results) == 4, f"应 4 个组合（topk=5,10,20,30），实际 {len(results)}"

    for r in results:
        print(f"  {r.name}: sharpe={r.metrics.get('sharpe_ratio', 0):.3f}, "
              f"年化={r.metrics.get('annual_return', 0):.3%}")
    print("  ✓ 参数扫描测试正常")
    return results


def test_universe_sample_test():
    """股票池抽样测试"""
    print("=" * 60)
    print("测试 4.3: 股票池抽样")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=60, n_days=200, seed=42)

    fn = build_backtest_fn(topk=8)
    tester = StabilityTester(backtest_fn=fn)
    tester.add_universe_sample_test(n_samples=5, sample_frac=0.7, random_state=42)
    results = tester.run(data, factors)
    print(f"  生成 {len(results)} 个抽样结果")
    assert len(results) == 5, f"应 5 个抽样，实际 {len(results)}"

    for r in results:
        n_codes = r.extra.get("n_codes", 0)
        print(f"  {r.name}: n_codes={n_codes}, "
              f"年化={r.metrics.get('annual_return', 0):.3%}, "
              f"夏普={r.metrics.get('sharpe_ratio', 0):.3f}")
    print("  ✓ 股票池抽样测试正常")
    return results


def test_bootstrap_test():
    """Bootstrap 稳健性测试"""
    print("=" * 60)
    print("测试 4.4: Bootstrap 抽样")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=30, n_days=200, seed=42)

    fn = build_backtest_fn(topk=5)
    tester = StabilityTester(backtest_fn=fn)
    tester.add_bootstrap_test(n_bootstrap=10, block_size=20, random_state=42)
    results = tester.run(data, factors)
    print(f"  生成 {len(results)} 个 bootstrap 样本")
    assert len(results) == 10, f"应 10 个 bootstrap，实际 {len(results)}"

    for r in results[:3]:
        n_days = r.extra.get("n_days", 0)
        print(f"  {r.name}: n_days={n_days}, "
              f"年化={r.metrics.get('annual_return', 0):.3%}, "
              f"夏普={r.metrics.get('sharpe_ratio', 0):.3f}")
    print("  ✓ Bootstrap 测试正常")
    return results


def test_stability_score():
    """稳定性评分"""
    print("=" * 60)
    print("测试 4.5: 稳定性评分")
    print("=" * 60)
    # 构造稳定的模拟结果
    stable_results = [
        StabilityResult(
            name=f"t{i}",
            metrics={
                "sharpe_ratio": 1.5 + np.random.randn() * 0.1,
                "max_drawdown": -0.10 + np.random.randn() * 0.01,
                "annual_return": 0.15 + np.random.randn() * 0.01,
            }
        )
        for i in range(10)
    ]
    # 构造不稳定的模拟结果
    unstable_results = [
        StabilityResult(
            name=f"t{i}",
            metrics={
                "sharpe_ratio": np.random.choice([2.0, -1.0]),
                "max_drawdown": -np.random.uniform(0.05, 0.5),
                "annual_return": np.random.uniform(-0.3, 0.5),
            }
        )
        for i in range(10)
    ]
    fn = lambda d, f: {}
    tester = StabilityTester(backtest_fn=fn)

    s1 = tester.stability_score(stable_results)
    s2 = tester.stability_score(unstable_results)
    print(f"  稳定场景: score={s1['stability_score']:.3f}, "
          f"sharpe_consistency={s1['sharpe_consistency']:.2f}, "
          f"drawdown_range={s1['drawdown_range']:.3f}")
    print(f"  不稳定场景: score={s2['stability_score']:.3f}, "
          f"sharpe_consistency={s2['sharpe_consistency']:.2f}, "
          f"drawdown_range={s2['drawdown_range']:.3f}")

    assert s1["stability_score"] > s2["stability_score"], "稳定场景评分应高于不稳定场景"
    print(f"  ✓ 稳定性评分差异: {s1['stability_score'] - s2['stability_score']:.3f}")
    return True


def test_summarize():
    """汇总 DataFrame"""
    print("=" * 60)
    print("测试 4.6: summarize 输出")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=30, n_days=300, seed=1)
    fn = build_backtest_fn()
    tester = StabilityTester(backtest_fn=fn)
    tester.add_param_sweep_test({"topk": [5, 10, 20]})
    results = tester.run(data, factors)
    summary = tester.summarize(results)
    print(f"  汇总表 shape: {summary.shape}")
    print(f"  列名: {list(summary.columns)[:8]}")
    assert "name" in summary.columns
    assert "annual_return" in summary.columns or len(summary) == 0
    print("  ✓ summarize 输出格式正确")
    return True


def main():
    print("\n" + "=" * 60)
    print("【测试 4: 多源稳健性测试验证】")
    print("=" * 60 + "\n")

    results = {}
    try:
        test_time_window_test()
        results["time_window"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["time_window"] = False

    try:
        test_param_sweep_test()
        results["param_sweep"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["param_sweep"] = False

    try:
        test_universe_sample_test()
        results["universe_sample"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["universe_sample"] = False

    try:
        test_bootstrap_test()
        results["bootstrap"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["bootstrap"] = False

    try:
        test_stability_score()
        results["stability_score"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["stability_score"] = False

    try:
        test_summarize()
        results["summarize"] = True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        results["summarize"] = False

    print("\n" + "=" * 60)
    print("测试 4 总结")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'通过' if v else '失败'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
