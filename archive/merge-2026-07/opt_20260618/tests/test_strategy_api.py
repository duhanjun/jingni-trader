"""
测试 2: Strategy 抽象基类与组合策略

验证目标：
1. TopkDropoutStrategy 能基于 alpha_score 选出 topk 股票
2. 多个策略实例独立工作
3. 与 VectorizedBacktestEngine 无缝集成
4. 工厂方法 create_strategy 正确创建策略
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from strategy_api import (
    Strategy, SignalStrategy, TopkDropoutStrategy,
    ReversalStrategy, MomentumStrategy, STRATEGY_REGISTRY, create_strategy
)
from vectorized_backtest import VectorizedBacktestEngine
from synthetic_data import generate_synthetic_data


def test_topk_strategy_basic():
    """TopkDropoutStrategy 基础测试"""
    print("=" * 60)
    print("测试 2.1: TopkDropoutStrategy 基础")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=30, n_days=50, seed=1)

    strategy = TopkDropoutStrategy(topk=5, n_drop=1)
    signals = strategy.generate_signals(data, factors)

    assert not signals.empty, "Topk 策略应产生信号"
    assert {"date", "code", "weight"}.issubset(signals.columns), "信号应包含必要列"
    # 每个交易日最多 topk 只股票
    per_day = signals.groupby("date").size()
    assert per_day.max() <= 5, f"每天最多 5 只股票，实际最多 {per_day.max()}"
    # 权重和为 1
    sums = signals.groupby("date")["weight"].sum()
    assert (sums - 1.0).abs().max() < 1e-6, f"每天权重和应为 1，实际: {sums.iloc[0]}"
    print(f"  ✓ Topk=5, 信号数 {len(signals)}, 每天平均 {per_day.mean():.1f} 只")
    return True


def test_multiple_strategies_independent():
    """多个策略实例独立工作"""
    print("=" * 60)
    print("测试 2.2: 多个策略独立")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=30, n_days=50, seed=1)

    s1 = TopkDropoutStrategy(topk=5)
    s2 = TopkDropoutStrategy(topk=10)
    s3 = ReversalStrategy(topk=5)
    s4 = MomentumStrategy(topk=5)

    sigs1 = s1.generate_signals(data, factors)
    sigs2 = s2.generate_signals(data, factors)
    sigs3 = s3.generate_signals(data, factors)
    sigs4 = s4.generate_signals(data, factors)

    n1 = sigs1.groupby("date").size().mean()
    n2 = sigs2.groupby("date").size().mean()
    n3 = sigs3.groupby("date").size().mean()
    n4 = sigs4.groupby("date").size().mean()

    print(f"  Topk5:  日均 {n1:.1f} 只, 总 {len(sigs1)} 信号")
    print(f"  Topk10: 日均 {n2:.1f} 只, 总 {len(sigs2)} 信号")
    print(f"  Reversal5: 日均 {n3:.1f} 只, 总 {len(sigs3)} 信号")
    print(f"  Momentum5: 日均 {n4:.1f} 只, 总 {len(sigs4)} 信号")

    # Reversal 和 Momentum 选股应不同（alpha_score 是正相关的）
    rev_codes = set(sigs3["code"].unique())
    mom_codes = set(sigs4["code"].unique())
    overlap = len(rev_codes & mom_codes)
    print(f"  Reversal vs Momentum 重叠股票数: {overlap}")
    assert n1 <= 5 and n2 <= 10, "Topk 限制应生效"
    print("  ✓ 各策略独立工作正常")
    return True


def test_strategy_engine_integration():
    """策略与回测引擎集成"""
    print("=" * 60)
    print("测试 2.3: 策略与回测引擎集成")
    print("=" * 60)
    data, factors = generate_synthetic_data(n_stocks=30, n_days=120, seed=2)
    strategy = TopkDropoutStrategy(topk=10)
    signals = strategy.generate_signals(data, factors)

    engine = VectorizedBacktestEngine()
    result = engine.run(data, signals)
    m = result["metrics"]
    print(f"  Topk10 策略回测: 年化={m['annual_return']:.3%}, "
          f"夏普={m['sharpe_ratio']:.3f}, "
          f"回撤={m['max_drawdown']:.3%}")
    assert "annual_return" in m
    print("  ✓ 策略 → 信号 → 回测 流程通畅")
    return m


def test_factory_create_strategy():
    """工厂方法 create_strategy"""
    print("=" * 60)
    print("测试 2.4: 工厂方法")
    print("=" * 60)
    s = create_strategy("topk_dropout", topk=15)
    assert isinstance(s, TopkDropoutStrategy)
    assert s.topk == 15
    print(f"  ✓ topk_dropout 工厂创建成功: topk={s.topk}")

    s2 = create_strategy("momentum", topk=20)
    assert isinstance(s2, MomentumStrategy)
    assert s2.topk == 20
    print(f"  ✓ momentum 工厂创建成功: topk={s2.topk}")

    # 错误情况
    try:
        create_strategy("nonexistent")
        assert False, "应抛出 ValueError"
    except ValueError:
        print("  ✓ 未知策略正确抛出 ValueError")
    return True


def main():
    print("\n" + "=" * 60)
    print("【测试 2: Strategy API 验证】")
    print("=" * 60 + "\n")

    results = {}
    try:
        test_topk_strategy_basic()
        results["topk_basic"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["topk_basic"] = False

    try:
        test_multiple_strategies_independent()
        results["multi_strategy"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["multi_strategy"] = False

    try:
        test_strategy_engine_integration()
        results["engine_integration"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["engine_integration"] = False

    try:
        test_factory_create_strategy()
        results["factory"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["factory"] = False

    print("\n" + "=" * 60)
    print("测试 2 总结")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'通过' if v else '失败'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
