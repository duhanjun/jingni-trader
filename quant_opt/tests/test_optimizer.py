"""
test_optimizer.py
=================

回测参数优化器的测试。

测试覆盖：
- 网格搜索功能
- 最佳参数识别
- 热力图构造
- 边界条件
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backtest"))
sys.path.insert(0, os.path.join(_HERE, "..", "factor"))

from vectorized_adapter import VectorizedAdapter, build_test_data
from optimizer import BacktestOptimizer, _signal_factory_momentum


def test_optimizer_basic():
    """基本功能：能找到最佳参数。"""
    data = build_test_data(n_stocks=20, n_days=126, seed=42)
    opt = BacktestOptimizer(VectorizedAdapter())
    result = opt.optimize(
        data=data,
        signals_factory=lambda p: _signal_factory_momentum(data, **p),
        param_grid={
            "lookback": [5, 10, 20],
            "top_pct": [0.1, 0.2, 0.3],
        },
        maximize="sharpe_ratio",
    )
    assert "best_params" in result
    assert "best_value" in result
    assert "heatmap" in result
    assert "all_results" in result
    assert len(result["all_results"]) == 3 * 3  # 9 组合
    assert result["heatmap"].shape == (3, 3)  # 3 lookback × 3 top_pct
    print(f"  Best params: {result['best_params']}, value: {result['best_value']:.3f}")
    print(f"  Heatmap shape: {result['heatmap'].shape}")
    print(f"  Heatmap:\n{result['heatmap']}")


def test_optimizer_finds_maximize_target():
    """优化器应能找到最大化目标的参数。"""
    data = build_test_data(n_stocks=30, n_days=252, seed=42)
    opt = BacktestOptimizer(VectorizedAdapter())
    result = opt.optimize(
        data=data,
        signals_factory=lambda p: _signal_factory_momentum(data, **p),
        param_grid={"lookback": [5, 10, 20, 60], "top_pct": [0.1, 0.2, 0.3]},
        maximize="sharpe_ratio",
    )
    # 找出 all_results 中 sharpe 最大的
    best = max(result["all_results"], key=lambda r: r["value"])
    # 优化器报告的最佳应该与手工找的最大一致
    assert result["best_params"] == best["params"]
    assert result["best_value"] == best["value"]


def test_optimizer_handles_failures():
    """某个参数组合失败时仍能继续。"""
    data = build_test_data(n_stocks=20, n_days=60, seed=1)
    opt = BacktestOptimizer(VectorizedAdapter())

    def bad_factory(p):
        if p.get("lookback") == 5:
            raise ValueError("simulated failure")
        return _signal_factory_momentum(data, **p)

    result = opt.optimize(
        data=data,
        signals_factory=bad_factory,
        param_grid={"lookback": [5, 10], "top_pct": [0.1, 0.2]},
        maximize="sharpe_ratio",
    )
    # 4 个组合中有 2 个会失败，2 个应成功
    successes = [r for r in result["all_results"] if "metrics" in r]
    failures = [r for r in result["all_results"] if "error" in r]
    assert len(successes) == 2
    assert len(failures) == 2
    print(f"  Failures handled: {len(failures)} caught, {len(successes)} succeeded")


def test_optimizer_single_param():
    """单参数时热力图应为空（只支持 2D）。"""
    data = build_test_data(n_stocks=20, n_days=60, seed=2)
    opt = BacktestOptimizer(VectorizedAdapter())
    result = opt.optimize(
        data=data,
        signals_factory=lambda p: _signal_factory_momentum(data, **p),
        param_grid={"lookback": [5, 10, 20]},
        maximize="sharpe_ratio",
    )
    assert result["heatmap"].empty
    assert len(result["all_results"]) == 3


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        print(f"\n[ {t.__name__} ]")
        try:
            t()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
