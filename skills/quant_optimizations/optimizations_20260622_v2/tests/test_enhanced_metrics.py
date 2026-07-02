"""
扩展绩效指标测试

测试内容：
1. VaR / CVaR 计算正确性
2. 信息比率计算
3. Beta / Alpha 计算
4. 最大回撤持续期
5. 捕获率
6. 完整指标计算
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from skills.quant_optimizations.optimizations_20260622_v2.enhanced_metrics import (
    calc_var,
    calc_cvar,
    calc_information_ratio,
    calc_beta_alpha,
    calc_max_drawdown_duration,
    calc_capture_ratios,
    calc_tail_ratio,
    calc_full_metrics,
)


def test_var_cvar():
    """测试 1：VaR / CVaR"""
    print("\n=== 测试 1: VaR / CVaR ===")
    np.random.seed(42)
    returns = pd.Series(np.random.randn(1000) * 0.02)

    var_95 = calc_var(returns, 0.95)
    cvar_95 = calc_cvar(returns, 0.95)
    var_99 = calc_var(returns, 0.99)
    cvar_99 = calc_cvar(returns, 0.99)

    print(f"  VaR 95%: {var_95:.4f}")
    print(f"  CVaR 95%: {cvar_95:.4f}")
    print(f"  VaR 99%: {var_99:.4f}")
    print(f"  CVaR 99%: {cvar_99:.4f}")

    # VaR 应为负数（损失）
    assert var_95 < 0, "VaR 95% 应为负数"
    assert var_99 < var_95, "VaR 99% 应比 VaR 95% 更极端"
    # CVaR 应比 VaR 更极端（更负）
    assert cvar_95 <= var_95, "CVaR 应 <= VaR"
    assert cvar_99 <= var_99, "CVaR 应 <= VaR"
    print("  ✓ VaR / CVaR 计算正确")


def test_information_ratio():
    """测试 2：信息比率"""
    print("\n=== 测试 2: 信息比率 ===")
    np.random.seed(42)
    benchmark = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2024-01-01", periods=252))
    # 策略 = 基准 + 超额
    strategy = benchmark + 0.001

    ir = calc_information_ratio(strategy, benchmark)
    print(f"  信息比率: {ir:.4f}")
    assert ir > 0, "有正超额收益时 IR 应为正"
    print("  ✓ 信息比率计算正确")


def test_beta_alpha():
    """测试 3：Beta / Alpha"""
    print("\n=== 测试 3: Beta / Alpha ===")
    np.random.seed(42)
    benchmark = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2024-01-01", periods=252))
    # 策略 = 1.5 * 基准 + 0.0005 + 噪声
    strategy = 1.5 * benchmark + 0.0005 + np.random.randn(252) * 0.002

    ba = calc_beta_alpha(strategy, benchmark)
    print(f"  Beta: {ba['beta']:.4f} (期望 ~1.5)")
    print(f"  Alpha: {ba['alpha']:.4f} (期望 > 0)")
    assert abs(ba["beta"] - 1.5) < 0.2, f"Beta 应接近 1.5，实际 {ba['beta']}"
    assert ba["alpha"] > 0, "Alpha 应为正"
    print("  ✓ Beta / Alpha 计算正确")


def test_drawdown_duration():
    """测试 4：最大回撤持续期"""
    print("\n=== 测试 4: 最大回撤持续期 ===")
    # 构造一个先涨后跌再恢复的净值曲线
    equity = pd.Series(
        [100, 110, 120, 100, 80, 90, 110, 130],
        index=pd.bdate_range("2024-01-01", periods=8),
    )
    dd_info = calc_max_drawdown_duration(equity)
    print(f"  最大回撤持续期: {dd_info['max_dd_duration']} 天")
    print(f"  回撤开始: {dd_info['underwater_start']}")
    print(f"  回撤谷底: {dd_info['underwater_end']}")
    print(f"  恢复期: {dd_info['max_dd_recovery']} 天")
    assert dd_info["max_dd_duration"] > 0
    print("  ✓ 回撤持续期计算正确")


def test_capture_ratios():
    """测试 5：捕获率"""
    print("\n=== 测试 5: 捕获率 ===")
    np.random.seed(42)
    benchmark = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2024-01-01", periods=252))
    # 策略在上涨时赚更多，下跌时亏更少
    up = benchmark > 0
    strategy = benchmark.copy()
    strategy[up] = benchmark[up] * 1.2
    strategy[~up] = benchmark[~up] * 0.8

    caps = calc_capture_ratios(strategy, benchmark)
    print(f"  上行捕获率: {caps['up_capture']:.4f} (期望 ~1.2)")
    print(f"  下行捕获率: {caps['down_capture']:.4f} (期望 ~0.8)")
    assert caps["up_capture"] > 1.0, "上行捕获率应 > 1"
    assert caps["down_capture"] < 1.0, "下行捕获率应 < 1"
    print("  ✓ 捕获率计算正确")


def test_tail_ratio():
    """测试 6：尾部比率"""
    print("\n=== 测试 6: 尾部比率 ===")
    np.random.seed(42)
    # 正偏分布
    returns = pd.Series(np.random.exponential(0.01, 1000) - 0.005)
    tr = calc_tail_ratio(returns)
    print(f"  尾部比率: {tr:.4f}")
    assert tr > 0
    print("  ✓ 尾部比率计算正确")


def test_full_metrics():
    """测试 7：完整指标计算"""
    print("\n=== 测试 7: 完整指标 ===")
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003, index=pd.bdate_range("2024-01-01", periods=252))
    equity = (1 + returns).cumprod() * 1_000_000

    metrics = calc_full_metrics(equity)
    print(f"  指标数量: {len(metrics)}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    # 验证关键指标存在
    required = [
        "total_return", "annual_return", "volatility", "max_drawdown",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "var_95", "cvar_95", "var_99", "cvar_99",
        "max_dd_duration", "tail_ratio", "win_rate",
    ]
    for key in required:
        assert key in metrics, f"缺少指标: {key}"
    print("  ✓ 完整指标计算正确")


def test_full_metrics_with_benchmark():
    """测试 8：带基准的完整指标"""
    print("\n=== 测试 8: 带基准的完整指标 ===")
    np.random.seed(42)
    benchmark_returns = pd.Series(np.random.randn(252) * 0.01, index=pd.bdate_range("2024-01-01", periods=252))
    returns = benchmark_returns + 0.0005
    equity = (1 + returns).cumprod() * 1_000_000

    metrics = calc_full_metrics(equity, returns, benchmark_returns)
    assert "information_ratio" in metrics
    assert "beta" in metrics
    assert "alpha" in metrics
    assert "up_capture" in metrics
    assert "down_capture" in metrics
    print(f"  信息比率: {metrics['information_ratio']:.4f}")
    print(f"  Beta: {metrics['beta']:.4f}")
    print(f"  Alpha: {metrics['alpha']:.4f}")
    print("  ✓ 带基准的完整指标计算正确")


def test_edge_cases():
    """测试 9：边界条件"""
    print("\n=== 测试 9: 边界条件 ===")

    # 空数据
    assert calc_var(pd.Series(dtype=float)) == 0.0
    assert calc_cvar(pd.Series(dtype=float)) == 0.0
    assert calc_full_metrics(pd.Series(dtype=float)) == {}
    print("  ✓ 空数据正确处理")

    # 单点数据
    single = pd.Series([1.0])
    assert calc_full_metrics(single) == {}
    print("  ✓ 单点数据正确处理")

    # 恒定净值（零波动）
    constant = pd.Series([1.0] * 100)
    metrics = calc_full_metrics(constant)
    assert metrics["volatility"] == 0.0
    assert metrics["sharpe_ratio"] == 0.0
    print("  ✓ 恒定净值正确处理")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_var_cvar()
    test_information_ratio()
    test_beta_alpha()
    test_drawdown_duration()
    test_capture_ratios()
    test_tail_ratio()
    test_full_metrics()
    test_full_metrics_with_benchmark()
    test_edge_cases()
    print("\n🎉 全部扩展指标测试通过")