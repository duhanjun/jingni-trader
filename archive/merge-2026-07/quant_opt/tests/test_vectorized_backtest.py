"""
向量化回测测试 & 性能基准
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_opt.vectorized_backtest import (
    PortfolioWeight,
    signals_to_weights,
    vectorized_backtest,
)
from quant_opt.tests._fixtures import (
    make_signals,
    make_synthetic_a_share_data,
)


# ---------------------------------------------------------------------------
# 正确性测试
# ---------------------------------------------------------------------------
def test_portfolio_weight_normalization():
    data = make_synthetic_a_share_data(n_stocks=10, n_days=20)
    sig = make_signals(data, n_dates=20, top_quantile=0.3)
    pw = signals_to_weights(sig, top_quantile=0.3, long_only=True)
    assert isinstance(pw, PortfolioWeight)
    # 每日权重和应 ≤ 1
    row_sums = pw.weight_frame.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-3).all()


def test_vectorized_backtest_runs():
    data = make_synthetic_a_share_data(n_stocks=20, n_days=60)
    sig = make_signals(data, n_dates=60, top_quantile=0.2)
    pw = signals_to_weights(sig, top_quantile=0.2, long_only=True)
    result = vectorized_backtest(
        price_df=data, weights=pw, init_capital=1_000_000.0
    )
    assert "equity_curve" in result
    assert "trades" in result
    assert "metrics" in result
    assert not result["equity_curve"].empty
    # 初始权益应等于 init_capital
    eq_first = result["equity_curve"]["equity"].iloc[0]
    assert abs(eq_first - 1_000_000.0) < 1.0


def test_metrics_calculated():
    data = make_synthetic_a_share_data(n_stocks=20, n_days=100)
    sig = make_signals(data, n_dates=100, top_quantile=0.2)
    pw = signals_to_weights(sig)
    result = vectorized_backtest(price_df=data, weights=pw)
    m = result["metrics"]
    for k in ["total_return", "annual_return", "sharpe_ratio",
              "max_drawdown", "calmar_ratio", "win_rate"]:
        assert k in m
        assert isinstance(m[k], float)


def test_long_short_weights():
    data = make_synthetic_a_share_data(n_stocks=20, n_days=30)
    # 构造做多做空信号
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start=data["date"].min(), periods=30)
    codes = data["code"].unique()
    rows = []
    for d in dates:
        ch = rng.choice(codes, size=10, replace=False)
        for c in codes:
            if c in ch[:5]:
                rows.append({"date": d, "code": c, "signal": 1})
            elif c in ch[5:]:
                rows.append({"date": d, "code": c, "signal": -1})
            else:
                rows.append({"date": d, "code": c, "signal": 0})
    sig = pd.DataFrame(rows)
    pw = signals_to_weights(sig, top_quantile=0.5, bottom_quantile=0.5, long_only=False)
    # 每日权重绝对值之和应 = 1
    assert (pw.weight_frame.abs().sum(axis=1) - 1.0).abs().max() < 1e-3


# ---------------------------------------------------------------------------
# 性能基准
# ---------------------------------------------------------------------------
def test_performance_benchmark():
    """对比向量化版与（参考）逐日循环版的性能"""
    n_stocks = 100
    n_days = 500
    data = make_synthetic_a_share_data(n_stocks=n_stocks, n_days=n_days, seed=2024)
    sig = make_signals(data, n_dates=n_days, top_quantile=0.1)
    pw = signals_to_weights(sig, top_quantile=0.1)

    # 1) 向量化版
    t0 = time.perf_counter()
    result = vectorized_backtest(price_df=data, weights=pw, init_capital=1_000_000.0)
    t_vec = time.perf_counter() - t0
    print(f"\n[bench] 向量化回测 ({n_stocks}股×{n_days}日): {t_vec:.3f}s")

    # 2) 参考：纯 Python 逐日循环版
    t0 = time.perf_counter()
    _ref_loop_backtest(data, sig, top_quantile=0.1)
    t_loop = time.perf_counter() - t0
    print(f"[bench] 纯 Python 循环回测: {t_loop:.3f}s")
    print(f"[bench] 加速比: {t_loop / max(t_vec, 1e-6):.1f}x")

    # 向量化应明显快于纯循环
    assert t_vec < t_loop


def _ref_loop_backtest(data: pd.DataFrame, sig: pd.DataFrame, top_quantile: float):
    """简单参考实现：纯 Python 逐日循环，不做完整交易细节"""
    dates = sorted(sig["date"].unique())
    codes = data["code"].unique()
    cash = 1_000_000.0
    holdings = {c: 0 for c in codes}
    eq_records = []
    for dt in dates:
        day_sig = sig[sig["date"] == dt]
        day_data = data[data["date"] == dt].set_index("code")
        # 等权买入
        selected = day_sig[day_sig["signal"] > 0]["code"].tolist()
        if not selected:
            eq_records.append({"date": dt, "equity": cash})
            continue
        per_stock = cash / len(selected)
        for c in selected:
            if c in day_data.index:
                price = day_data.loc[c, "close"]
                shares = int(per_stock / price / 100) * 100
                if shares > 0:
                    holdings[c] = holdings.get(c, 0) + shares
                    cash -= shares * price
        mv = sum(holdings[c] * day_data.loc[c, "close"] if c in day_data.index else 0 for c in holdings)
        eq_records.append({"date": dt, "equity": cash + mv})
    return pd.DataFrame(eq_records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
