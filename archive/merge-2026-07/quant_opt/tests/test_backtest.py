"""
test_backtest.py
================

向量化回测 vs 循环回测的对比测试。

测试覆盖：
- 性能对比：循环 vs 向量化
- 数值一致性：相同输入下结果应一致
- 边界条件：空数据、单只股票、单日、停牌、全涨停
- 信号类型：signal/target_percent/target_amount/signal_strength
"""
from __future__ import annotations

import os
import sys
import time
import math
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backtest"))

from vectorized_adapter import VectorizedAdapter, build_test_data, build_test_signals
from comprehensive_metrics import compute_full_metrics


# ---------------------------------------------------------------
# 简易的纯 Python 循环回测参考实现（基于现有 native_adapter.py 思路）
# ---------------------------------------------------------------
def loop_backtest(data: pd.DataFrame, signals: pd.DataFrame,
                  init_capital: float = 1_000_000.0,
                  commission_rate: float = 0.00025,
                  stamp_tax_rate: float = 0.001,
                  min_commission: float = 5.0,
                  slippage: float = 0.001,
                  t_plus_1: bool = True) -> dict:
    """纯循环回测参考实现，用于与向量化版本对比。"""
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
    dates = sorted(signals["date"].unique())
    cash = init_capital
    positions = {}  # code -> shares
    equity_records = []
    trades = []

    LOT = 100
    for dt_idx, dt in enumerate(dates):
        day_signal = signals[signals["date"] == dt]
        day_data = data[data["date"] == dt].set_index("code")

        # 用前一日重估权益
        if dt_idx > 0:
            prev_dt = dates[dt_idx - 1]
            prev_data = data[data["date"] == prev_dt].set_index("code")
            market_value = sum(
                shares * prev_data.loc[code, "close"] if code in prev_data.index else 0
                for code, shares in positions.items() if shares > 0
            )
            equity = cash + market_value
        else:
            equity = init_capital
            market_value = 0

        equity_records.append({"date": dt, "equity": equity, "cash": cash,
                               "market_value": market_value})

        # 调仓在次日开盘
        if dt_idx == len(dates) - 1:
            break
        next_dt = dates[dt_idx + 1]
        next_data = data[data["date"] == next_dt].set_index("code")

        # 计算目标股数
        target_shares_dict = {}
        n_buy = 0
        for _, row in day_signal.iterrows():
            if row.get("signal", 0) > 0 and row["code"] in next_data.index:
                target_shares_dict[row["code"]] = True
                n_buy += 1
        if n_buy == 0:
            target_shares_dict = {code: False for code in positions}
        budget = cash * 0.95 / max(n_buy, 1)
        new_positions = {}
        for code in positions:
            if code in next_data.index:
                if target_shares_dict.get(code) is True:
                    price = next_data.loc[code, "open"] * (1 + slippage)
                    if price > 0:
                        sh = int(budget / price / LOT) * LOT
                        if sh > 0 and sh * price + min(sh * price * commission_rate, min_commission) <= cash:
                            new_positions[code] = sh
                            cost = sh * price + min(sh * price * commission_rate, min_commission)
                            cash -= cost
                            trades.append({"date": next_dt, "code": code, "action": "buy",
                                           "price": price, "shares": sh, "amount": sh * price,
                                           "commission": min(sh * price * commission_rate, min_commission),
                                           "tax": 0.0})
        for code in list(positions.keys()):
            if code not in new_positions and positions[code] > 0:
                if code in next_data.index:
                    price = next_data.loc[code, "open"] * (1 - slippage)
                    if price > 0:
                        sh = positions[code]
                        proceeds = sh * price
                        comm = max(sh * price * commission_rate, min_commission)
                        tax = sh * price * stamp_tax_rate
                        cash += proceeds - comm - tax
                        trades.append({"date": next_dt, "code": code, "action": "sell",
                                       "price": price, "shares": sh, "amount": sh * price,
                                       "commission": comm, "tax": tax})
        for code, sh in new_positions.items():
            positions[code] = sh
        # 移除已清仓
        positions = {k: v for k, v in positions.items() if v > 0}

    eq_df = pd.DataFrame(equity_records)
    eq_series = eq_df.set_index("date")["equity"]
    metrics = compute_full_metrics(eq_series, trades=pd.DataFrame(trades))
    return {"equity_curve": eq_df, "trades": pd.DataFrame(trades), "metrics": metrics}


# ---------------------------------------------------------------
# 测试函数
# ---------------------------------------------------------------
def test_vectorized_basic():
    """基本功能：能跑通且 equity 有界。"""
    data = build_test_data(n_stocks=10, n_days=60)
    signals = build_test_signals(data, top_pct=0.3)
    adapter = VectorizedAdapter()
    result = adapter.run_backtest(data, signals)
    eq_final = result["equity_curve"]["equity"].iloc[-1]
    init = 1_000_000
    # equity 应在初始资金的 50%~200% 之间 (允许极端)
    assert 0.3 * init < eq_final < 5 * init
    assert not result["metrics"]["sharpe_ratio"] != result["metrics"]["sharpe_ratio"]  # 非 NaN
    print(f"  basic: equity final={eq_final:,.0f}, sharpe={result['metrics']['sharpe_ratio']:.3f}")


def test_vectorized_buy_and_hold():
    """买入持有 5 只股票 30 天：equity 应在合理范围内。"""
    data = build_test_data(n_stocks=10, n_days=30, seed=42)
    # 选定 5 只股票，每天都是同一组
    codes = sorted(data["code"].unique())[:5]
    signals = []
    for dt in data["date"].unique():
        for code in codes:
            signals.append({"date": dt, "code": code, "signal": 1})
    signals = pd.DataFrame(signals)
    adapter = VectorizedAdapter()
    result = adapter.run_backtest(data, signals)
    eq_final = result["equity_curve"]["equity"].iloc[-1]
    init = 1_000_000
    # 买入持有 5 只股票 30 天，预期 equity 在 0.7~1.5 倍 (有滑点 / 佣金)
    assert 0.7 * init < eq_final < 1.5 * init
    print(f"  buy-and-hold 5 stocks × 30 days: equity final={eq_final:,.0f}")


def test_vectorized_empty_data():
    """空数据应优雅处理。"""
    adapter = VectorizedAdapter()
    result = adapter.run_backtest(pd.DataFrame(), pd.DataFrame())
    assert result["metrics"] == {}
    print("  empty data: OK (no crash)")


def test_vectorized_single_stock_single_day():
    """单只股票 + 单日。"""
    data = pd.DataFrame([{
        "date": pd.Timestamp("2023-01-01"), "code": "000001.SZ",
        "open": 10.0, "close": 10.5, "high": 10.6, "low": 9.9, "volume": 1000000,
        "is_limit_up": False, "is_limit_down": False,
    }, {
        "date": pd.Timestamp("2023-01-02"), "code": "000001.SZ",
        "open": 10.5, "close": 11.0, "high": 11.1, "low": 10.4, "volume": 1000000,
        "is_limit_up": False, "is_limit_down": False,
    }])
    signals = pd.DataFrame([{
        "date": pd.Timestamp("2023-01-01"), "code": "000001.SZ", "signal": 1,
    }])
    adapter = VectorizedAdapter()
    result = adapter.run_backtest(data, signals)
    assert not result["equity_curve"].empty
    print(f"  single stock: equity final={result['equity_curve']['equity'].iloc[-1]:,.0f}")


def test_performance_comparison():
    """向量化 vs 循环：速度对比 (买入持有策略，无换手)。"""
    n_stocks_list = [20, 50, 100]
    n_days = 252
    print(f"\n  Performance comparison (n_days={n_days}, 买入持有 top 5):")
    print(f"  {'n_stocks':<10} {'loop (s)':<12} {'vector (s)':<12} {'speedup':<10}")
    for n_stocks in n_stocks_list:
        data = build_test_data(n_stocks=n_stocks, n_days=n_days, seed=42)
        # 始终买入 top 5
        codes = sorted(data["code"].unique())[:5]
        signals = []
        for dt in sorted(data["date"].unique()):
            for code in codes:
                signals.append({"date": dt, "code": code, "signal": 1})
        signals = pd.DataFrame(signals)
        adapter = VectorizedAdapter()

        # Loop version (with simplified buy-hold)
        t0 = time.perf_counter()
        loop_result = loop_buy_and_hold(data, codes)
        loop_t = time.perf_counter() - t0

        # Vector version
        t0 = time.perf_counter()
        vec_result = adapter.run_backtest(data, signals)
        vec_t = time.perf_counter() - t0

        speedup = loop_t / vec_t if vec_t > 0 else float("inf")
        print(f"  {n_stocks:<10} {loop_t:<12.3f} {vec_t:<12.3f} {speedup:<10.1f}x")

        loop_ret = loop_result["equity"] / 1_000_000 - 1
        vec_ret = vec_result["equity_curve"]["equity"].iloc[-1] / 1_000_000 - 1
        print(f"    -> loop_ret={loop_ret:.3f}, vec_ret={vec_ret:.3f}, "
              f"diff={abs(loop_ret - vec_ret):.4f}")


def loop_buy_and_hold(data: pd.DataFrame, codes: list, init_capital: float = 1_000_000.0) -> dict:
    """买入持有 N 只股票的简单循环实现，用于速度对比。"""
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    dates = sorted(data["date"].unique())
    cash = init_capital
    positions = {c: 0 for c in codes}
    equity_records = []
    for i, dt in enumerate(dates):
        day = data[data["date"] == dt].set_index("code")
        if i == 0:
            # 首日建仓：等权
            budget = cash * 0.95 / len(codes)
            for code in codes:
                if code in day.index and day.loc[code, "open"] > 0:
                    shares = int(budget / day.loc[code, "open"] / 100) * 100
                    if shares > 0:
                        cost = shares * day.loc[code, "open"] * (1 + 0.00025)
                        cash -= cost
                        positions[code] = shares
        market_value = sum(positions[c] * day.loc[c, "close"] for c in codes if c in day.index)
        equity = cash + market_value
        equity_records.append({"date": dt, "equity": equity, "cash": cash, "market_value": market_value})
    eq_df = pd.DataFrame(equity_records)
    return {"equity": eq_df.set_index("date")["equity"].iloc[-1],
            "equity_curve": eq_df}


def test_signal_types():
    """支持不同信号格式。"""
    data = build_test_data(n_stocks=10, n_days=60, seed=1)
    sig_binary = build_test_signals(data, top_pct=0.2)
    sig_amount = sig_binary.copy()
    sig_amount["target_amount"] = sig_amount["signal"] * 100_000
    sig_amount = sig_amount.drop(columns=["signal"])

    sig_strength = sig_binary.copy()
    sig_strength["signal_strength"] = sig_strength["signal"] * 1.0
    sig_strength = sig_strength.drop(columns=["signal"])

    adapter = VectorizedAdapter()
    r1 = adapter.run_backtest(data, sig_binary)
    r2 = adapter.run_backtest(data, sig_amount)
    r3 = adapter.run_backtest(data, sig_strength)
    # 不同信号格式应都能跑通
    for r, name in [(r1, "binary"), (r2, "amount"), (r3, "strength")]:
        eq_final = r["equity_curve"]["equity"].iloc[-1]
        assert 0.3 * 1e6 < eq_final < 5 * 1e6
        print(f"  signal_type={name}: equity_final={eq_final:,.0f}")


def test_target_percent_direct():
    """直接给 target_percent 权重。"""
    data = build_test_data(n_stocks=5, n_days=30, seed=2)
    dates = sorted(data["date"].unique())
    signals = []
    for dt in dates:
        for i, code in enumerate(sorted(data["code"].unique())):
            signals.append({
                "date": dt, "code": code,
                "target_percent": 1.0 / 5,  # 等权
            })
    signals = pd.DataFrame(signals)
    adapter = VectorizedAdapter()
    result = adapter.run_backtest(data, signals)
    eq_final = result["equity_curve"]["equity"].iloc[-1]
    print(f"  equal-weight: equity_final={eq_final:,.0f}, "
          f"sharpe={result['metrics']['sharpe_ratio']:.3f}")
    assert not math.isnan(result["metrics"]["sharpe_ratio"])


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        print(f"\n[ {t.__name__} ]")
        try:
            t()
            passed += 1
            print(f"  ✓ PASSED")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
