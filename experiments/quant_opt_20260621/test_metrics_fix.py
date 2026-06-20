"""
回测指标修正测试

测试内容:
  1. 年化收益一致性: total_return 与 annual_return 一致
  2. Sharpe 一致性: 用 annual_return 而非 returns.mean()*252
  3. 最大回撤正确性
  4. Information Ratio / Tracking Error
  5. Trade pair 胜率
  6. 边界条件: 单点序列、全零收益、空 trades
  7. 与原 base_backtest.py 对比, 验证 bug 修复
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "skills", "backtest-engine", "scripts", "base"))

from metrics_fix import (
    annual_return, total_return, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, tracking_error, information_ratio,
    win_rate_from_trades, profit_loss_ratio, calc_all_metrics,
    monthly_returns, rolling_sharpe, safe_div,
)


def make_equity(n_days: int = 252, annual_ret: float = 0.1, vol: float = 0.15, seed: int = 42) -> pd.Series:
    """生成净值序列"""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(annual_ret / 252, vol / np.sqrt(252), n_days)
    equity = pd.Series(
        (1 + daily_ret).cumprod() * 1_000_000,
        index=pd.bdate_range("2024-01-01", periods=n_days),
    )
    return equity


def test_annual_return_consistency():
    """测试1: 年化收益一致性"""
    print("\n[Test 1] 年化收益一致性测试")
    equity = make_equity(n_days=252, annual_ret=0.1, vol=0.0)  # 零波动, 固定收益
    tr = total_return(equity)
    ar = annual_return(equity)
    # 零波动时, 252 日 total_return ≈ annual_return
    assert abs(tr - ar) < 0.01, f"零波动时 total_return ({tr:.4f}) 应接近 annual_return ({ar:.4f})"
    print(f"  total_return={tr:.4%}, annual_return={ar:.4%}")
    print("  [PASS] 年化收益一致性测试通过")


def test_sharpe_consistency():
    """测试2: Sharpe 一致性 - 用几何年化而非 returns.mean()*252"""
    print("\n[Test 2] Sharpe 一致性测试")
    equity = make_equity(n_days=504, annual_ret=0.15, vol=0.2, seed=1)
    sr = sharpe_ratio(equity, risk_free=0.03)

    # 旧实现 (base_backtest.py) 用 returns.mean()*252 作为年化收益
    rets = equity.pct_change().dropna()
    old_ann_ret = rets.mean() * 252
    old_vol = rets.std() * np.sqrt(252)
    old_sharpe = (old_ann_ret - 0.03) / old_vol

    # 新实现用几何年化
    new_ann_ret = annual_return(equity)
    new_vol = rets.std() * np.sqrt(252)
    new_sharpe = (new_ann_ret - 0.03) / new_vol

    print(f"  旧 Sharpe (算术年化): {old_sharpe:.4f}")
    print(f"  新 Sharpe (几何年化): {new_sharpe:.4f}")
    assert abs(sr - new_sharpe) < 1e-6, "sharpe_ratio 应使用几何年化"
    # 几何年化通常略低于算术年化
    assert new_ann_ret <= old_ann_ret + 1e-9, "几何年化应 <= 算术年化"
    print("  [PASS] Sharpe 一致性测试通过 - 使用几何年化")


def test_max_drawdown():
    """测试3: 最大回撤正确性"""
    print("\n[Test 3] 最大回撤正确性测试")
    # 构造已知回撤序列
    equity = pd.Series(
        [100, 110, 120, 90, 95, 85, 100, 105],
        index=pd.bdate_range("2024-01-01", periods=8),
    )
    mdd = max_drawdown(equity)
    # 峰值 120, 谷底 85, 回撤 = 85/120 - 1 = -0.2917
    expected = 85 / 120 - 1
    assert abs(mdd - expected) < 1e-6, f"最大回撤应为 {expected:.4f}, 实际 {mdd:.4f}"
    print(f"  最大回撤: {mdd:.4%} (峰值 120 -> 谷底 85)")
    print("  [PASS] 最大回撤正确性测试通过")


def test_information_ratio():
    """测试4: Information Ratio / Tracking Error"""
    print("\n[Test 4] Information Ratio 测试")
    rng = np.random.default_rng(42)
    n = 252
    strategy_ret = pd.Series(rng.normal(0.001, 0.015, n), index=pd.bdate_range("2024-01-01", periods=n))
    benchmark_ret = pd.Series(rng.normal(0.0005, 0.012, n), index=pd.bdate_range("2024-01-01", periods=n))

    te = tracking_error(strategy_ret, benchmark_ret)
    ir = information_ratio(strategy_ret, benchmark_ret)

    # 手算验证
    excess = strategy_ret - benchmark_ret
    expected_te = excess.std() * np.sqrt(252)
    expected_ir = (excess.mean() * 252) / expected_te

    assert abs(te - expected_te) < 1e-6, f"跟踪误差应为 {expected_te:.4f}, 实际 {te:.4f}"
    assert abs(ir - expected_ir) < 1e-6, f"信息比率应为 {expected_ir:.4f}, 实际 {ir:.4f}"
    print(f"  跟踪误差: {te:.4f}, 信息比率: {ir:.4f}")
    print("  [PASS] Information Ratio 测试通过")


def test_trade_pair_win_rate():
    """测试5: Trade pair 胜率 - 修复旧实现 buy/sell pnl 无意义问题"""
    print("\n[Test 5] Trade pair 胜率测试")
    # 构造已知交易: 买 100 @ 10, 卖 100 @ 12 (盈利), 买 100 @ 20, 卖 100 @ 18 (亏损)
    trades = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=4),
        "code": ["000001.SZ"] * 4,
        "action": ["buy", "sell", "buy", "sell"],
        "price": [10.0, 12.0, 20.0, 18.0],
        "shares": [100, 100, 100, 100],
        "amount": [1000, 1200, 2000, 1800],
        "commission": [5, 5, 5, 5],
        "tax": [0, 1.2, 0, 1.8],
        "pnl": [-1005, 1193.8, -2005, 1793.2],
    })
    wr = win_rate_from_trades(trades)
    # 2 次完整交易, 1 盈 1 亏, 胜率 0.5
    assert abs(wr - 0.5) < 1e-6, f"胜率应为 0.5, 实际 {wr}"
    plr = profit_loss_ratio(trades)
    # 盈利 200, 亏损 -200, 盈亏比 1.0
    assert abs(plr - 1.0) < 1e-6, f"盈亏比应为 1.0, 实际 {plr}"
    print(f"  胜率: {wr:.2%} (1 盈 1 亏)")
    print(f"  盈亏比: {plr:.4f}")
    print("  [PASS] Trade pair 胜率测试通过")


def test_edge_cases():
    """测试6: 边界条件"""
    print("\n[Test 6] 边界条件测试")

    # 单点序列
    equity = pd.Series([1_000_000], index=pd.bdate_range("2024-01-01", periods=1))
    assert total_return(equity) == 0.0
    assert annual_return(equity) == 0.0
    assert max_drawdown(equity) == 0.0
    assert sharpe_ratio(equity) == 0.0
    print("  [PASS] 单点序列处理正确")

    # 全零收益
    equity = pd.Series(
        [1_000_000] * 100,
        index=pd.bdate_range("2024-01-01", periods=100),
    )
    assert total_return(equity) == 0.0
    assert annual_return(equity) == 0.0
    assert max_drawdown(equity) == 0.0
    assert sharpe_ratio(equity) == 0.0  # vol=0 时返回 0
    print("  [PASS] 全零收益处理正确")

    # 空 trades
    assert win_rate_from_trades(pd.DataFrame()) == 0.0
    assert profit_loss_ratio(pd.DataFrame()) == 0.0
    print("  [PASS] 空 trades 处理正确")

    # safe_div
    assert safe_div(1, 0) == 0.0
    assert safe_div(1, 0, default=-1) == -1
    assert safe_div(1, 2) == 0.5
    print("  [PASS] safe_div 处理正确")


def test_compare_with_old_implementation():
    """测试7: 与旧实现对比, 验证 bug 修复"""
    print("\n[Test 7] 与旧实现对比测试")
    try:
        from base_backtest import BaseBacktestMetrics
    except ImportError:
        print("  [SKIP] 无法导入旧实现 base_backtest.py, 跳过对比")
        return

    equity = make_equity(n_days=504, annual_ret=0.12, vol=0.18, seed=7)

    # 旧实现的 Sharpe (用 returns.mean()*252)
    rets = equity.pct_change().dropna()
    old_sharpe = BaseBacktestMetrics.calc_sharpe(rets, risk_free=0.03)
    new_sharpe = sharpe_ratio(equity, risk_free=0.03)

    # 旧实现的年化收益 (用 total_return ** (1/n_years) - 1)
    old_ann = BaseBacktestMetrics.calc_annual_return(equity)
    new_ann = annual_return(equity)

    print(f"  旧 annual_return: {old_ann:.4%}")
    print(f"  新 annual_return: {new_ann:.4%}")
    print(f"  旧 Sharpe: {old_sharpe:.4f}")
    print(f"  新 Sharpe: {new_sharpe:.4f}")

    # 两者应接近但不完全相同 (新实现修正了一致性)
    assert abs(old_ann - new_ann) < 0.01, "年化收益应接近"
    # 新 Sharpe 应使用几何年化, 与 annual_return 一致
    expected_new_sharpe = (new_ann - 0.03) / (rets.std() * np.sqrt(252))
    assert abs(new_sharpe - expected_new_sharpe) < 1e-6, "新 Sharpe 应基于几何年化"
    print("  [PASS] 新实现修正了 Sharpe 与 annual_return 的一致性")


def test_full_metrics():
    """测试8: 完整指标计算"""
    print("\n[Test 8] 完整指标计算测试")
    equity = make_equity(n_days=504, annual_ret=0.15, vol=0.2, seed=11)
    benchmark = make_equity(n_days=504, annual_ret=0.08, vol=0.15, seed=22)

    trades = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=4),
        "code": ["000001.SZ"] * 4,
        "action": ["buy", "sell", "buy", "sell"],
        "price": [10.0, 11.0, 12.0, 10.5],
        "shares": [1000, 1000, 1000, 1000],
        "amount": [10000, 11000, 12000, 10500],
        "commission": [5, 5, 5, 5],
        "tax": [0, 11, 0, 10.5],
        "pnl": [-10005, 10984, -12005, 10484.5],
    })

    metrics = calc_all_metrics(equity, trades=trades, benchmark=benchmark)
    assert "total_return" in metrics
    assert "annual_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "sortino_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "calmar_ratio" in metrics
    assert "tracking_error" in metrics
    assert "information_ratio" in metrics
    assert "trade_win_rate" in metrics
    assert "profit_loss_ratio" in metrics
    assert "monthly_mean_return" in metrics
    assert "best_month" in metrics
    assert "worst_month" in metrics

    print(f"  总收益: {metrics['total_return']:.4%}")
    print(f"  年化收益: {metrics['annual_return']:.4%}")
    print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino: {metrics['sortino_ratio']:.4f}")
    print(f"  最大回撤: {metrics['max_drawdown']:.4%}")
    print(f"  Calmar: {metrics['calmar_ratio']:.4f}")
    print(f"  跟踪误差: {metrics['tracking_error']:.4f}")
    print(f"  信息比率: {metrics['information_ratio']:.4f}")
    print(f"  交易胜率: {metrics['trade_win_rate']:.4%}")
    print(f"  盈亏比: {metrics['profit_loss_ratio']:.4f}")
    print(f"  月均收益: {metrics['monthly_mean_return']:.4%}")
    print(f"  最佳月份: {metrics['best_month']:.4%}")
    print(f"  最差月份: {metrics['worst_month']:.4%}")
    print("  [PASS] 完整指标计算测试通过")


if __name__ == "__main__":
    test_annual_return_consistency()
    test_sharpe_consistency()
    test_max_drawdown()
    test_information_ratio()
    test_trade_pair_win_rate()
    test_edge_cases()
    test_compare_with_old_implementation()
    test_full_metrics()
    print("\n=== 所有指标修正测试通过 ===")
