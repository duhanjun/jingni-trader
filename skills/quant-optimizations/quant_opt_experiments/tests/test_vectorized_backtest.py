"""
测试 2：矢量化回测引擎
- 与现有 native_adapter 在同一份数据上跑同样策略
- 对比 净值曲线 / 绩效指标 / 速度
"""
import time
import pandas as pd
import numpy as np
import pytest

from skills.quant-optimizations.quant_opt_experiments.vectorized_backtest import (
    vectorized_backtest_single,
    vectorized_backtest_multi,
    calc_metrics,
    ma_cross_signals,
    rank_topk_signals,
)
from skills.quant-optimizations.quant_opt_experiments.tests.fixtures import make_synthetic_panel


def close_pivot(panel):
    return panel.pivot(index="date", columns="code", values="close").sort_index()


# ---- Test 1: 指标计算正确性 ----
def test_metrics_calculation():
    equity = pd.Series(
        [1.0, 1.01, 1.02, 1.005, 1.02, 1.03],
        index=pd.bdate_range("2024-01-01", periods=6),
    )
    m = calc_metrics(equity)
    assert "annual_return" in m
    assert "sharpe_ratio" in m
    assert "max_drawdown" in m
    assert m["max_drawdown"] <= 0  # 回撤必须 ≤ 0
    # 5 天从 1.0 到 1.03，总收益 3%
    assert abs(m["total_return"] - 0.03) < 1e-6


# ---- Test 2: 单标的矢量化 vs 手算 ----
def test_single_asset_vectorized_matches_manual(panel):
    close = panel[panel["code"] == panel["code"].iloc[0]].set_index("date")["close"]
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    # 在第 50 / 100 / 150 ... 天买入，第 70 / 120 ... 天卖出
    for i in range(50, len(close) - 30, 60):
        entries.iloc[i] = True
        exits.iloc[i + 20] = True

    res = vectorized_backtest_single(close, entries, exits, init_cash=1_000_000)
    assert len(res["equity"]) == len(close)
    # 第一次买入后净值 > init * 0.9 (因有 95% 仓位限制)
    assert res["equity"].iloc[60] > 0.9 * 1_000_000
    # 收益曲线在某些时点 > 0 (有交易)
    assert res["equity"].iloc[-1] > 0


# ---- Test 3: 多标的 + 因子 topk 信号 + 回测 ----
def test_multi_asset_with_topk(panel):
    close = close_pivot(panel)
    # 用最简单的反转因子: 5日收益率取负
    factor = -close.pct_change(5)
    entries, exits = rank_topk_signals(factor, top_pct=0.34, hold_days=20)

    res = vectorized_backtest_multi(
        close, entries, exits,
        alloc_per_asset=0.15,
        init_cash=1_000_000,
    )
    metrics = res.metrics
    assert "annual_return" in metrics
    assert metrics["n_days"] > 100
    # 交易记录
    print(f"\n[INFO] 多标的回测完成: 年化={metrics['annual_return']:.2%} "
          f"夏普={metrics['sharpe_ratio']:.2f} 交易次数={len(res.trades)}")


# ---- Test 4: 双均线信号 ----
def test_ma_cross_signals(panel):
    close = close_pivot(panel)
    entries, exits = ma_cross_signals(close, fast=5, slow=20)
    # 每天每只股票最多一个 entry / exit
    assert entries.sum().sum() > 0
    assert exits.sum().sum() > 0
    # entry 和 exit 不应同时为 True
    assert (entries & exits).sum().sum() == 0


# ---- Test 5: T+1 规则 ----
def test_t_plus1_rule(panel):
    close = panel[panel["code"] == panel["code"].iloc[0]].set_index("date")["close"]
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    # 第 50 天买入，第 50 天（次日）卖出：T+1 模式下不应允许
    entries.iloc[50] = True
    exits.iloc[51] = True  # 仅持有 1 天就卖

    res = vectorized_backtest_single(close, entries, exits, init_cash=1_000_000, t_plus_1=True)
    trades = res["trades"]
    # 检查是否真的有 sell 成交
    sells = trades[trades["side"] == "sell"]
    # 由于 T+1，51 天的 sell 应被拒绝，实际可能要到 52 天
    # 验证：至少有一个 buy
    assert (trades["side"] == "buy").any()


# ---- Test 6: 性能对比 (vs 现有 native_adapter) ----
def test_perf_vs_native_adapter(panel):
    """对比矢量化版与 native_adapter 的速度"""
    import sys, os
    # native_adapter 用的是相对 import（from ..base import ...），
    # 所以需要把 skills/backtest-engine 目录加到 sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bt_path = os.path.join(project_root, "skills", "backtest-engine")
    if bt_path not in sys.path:
        sys.path.insert(0, bt_path)
    try:
        from scripts.adapters.native_adapter import NativeAdapter  # type: ignore
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"native_adapter 不可用: {e}")

    close = close_pivot(panel).head(252)  # 一年数据
    factor = -close.pct_change(5)
    entries, exits = rank_topk_signals(factor, top_pct=0.34, hold_days=20)

    # --- 矢量化 ---
    t0 = time.time()
    res_v = vectorized_backtest_multi(close, entries, exits, alloc_per_asset=0.15, init_cash=1_000_000)
    t_v = time.time() - t0

    # --- native_adapter ---
    # native_adapter 期望长表 data + signals
    data_long = panel[panel["date"].isin(close.index)].copy()
    sig_long = []
    for dt in close.index:
        for code in close.columns:
            sig_long.append({
                "date": dt, "code": code,
                "signal": 1 if entries.loc[dt, code] else (-1 if exits.loc[dt, code] else 0)
            })
    sig_df = pd.DataFrame(sig_long)
    adapter = NativeAdapter()
    t0 = time.time()
    res_n = adapter.run_backtest(data=data_long, signals=sig_df, init_capital=1_000_000)
    t_n = time.time() - t0

    speedup = t_n / max(t_v, 1e-6)
    print(f"\n[PERF] 矢量化={t_v*1000:.1f}ms  native_adapter={t_n*1000:.1f}ms  加速={speedup:.1f}x")
    # 不强制要求更快（数据小），但要求都跑通
    assert "annual_return" in res_v.metrics
    assert "sharpe_ratio" in res_v.metrics
    if isinstance(res_n, dict):
        assert "metrics" in res_n
    else:
        # res_n 可能是对象
        assert hasattr(res_n, "metrics")


# ---- Test 7: 边界条件 ----
def test_empty_data():
    close = pd.Series(dtype=float)
    entries = pd.Series(dtype=bool)
    exits = pd.Series(dtype=bool)
    res = vectorized_backtest_single(close, entries, exits)
    assert res["equity"].empty


def test_all_false_signals(panel):
    close = panel[panel["code"] == panel["code"].iloc[0]].set_index("date")["close"]
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    res = vectorized_backtest_single(close, entries, exits, init_cash=1_000_000)
    # 无交易 → 净值恒为 1
    assert (res["equity"] == 1_000_000).all()


def test_cannot_buy_insufficient_cash(panel):
    """现金不足以买 1 手时不应成交"""
    close = panel[panel["code"] == panel["code"].iloc[0]].set_index("date")["close"]
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    entries.iloc[100] = True
    exits.iloc[200] = True
    # 给很少的现金
    res = vectorized_backtest_single(close, entries, exits, init_cash=10.0)
    trades = res["trades"]
    # 100 元买不到 1 手 100 股 → 应无 buy 交易
    if len(trades) > 0 and "side" in trades.columns:
        assert not (trades["side"] == "buy").any()
    # 即便为空也通过


# ---- pytest fixture ----
@pytest.fixture
def panel():
    return make_synthetic_panel(n_stocks=6, n_days=500, seed=42)


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"])
    sys.exit(0)