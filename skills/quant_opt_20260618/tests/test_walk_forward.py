"""
Walk-Forward Backtest 测试
==========================
测试目标:
  1. 分段正确性
  2. OOS 综合指标
  3. 边界条件
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.quant_opt_20260618.walk_forward.walk_forward import (
    WalkForwardBacktest,
    WalkForwardSegment,
)


def make_synthetic(n_stocks: int = 30, n_days: int = 504, seed: int = 11):
    """合成数据 (两年数据)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"W{str(i).zfill(6)}" for i in range(n_stocks)]
    n = n_stocks
    log_prices = np.zeros((n, n_days))
    log_prices[:, :5] = np.cumsum(rng.normal(0, 0.02, (n, 5)), axis=1) + 4.0
    for t in range(5, n_days):
        if t < 10:
            log_prices[:, t] = log_prices[:, t - 1] + rng.normal(0, 0.02, n)
        else:
            past_5 = log_prices[:, t - 5] - log_prices[:, t - 10]
            future_5 = -0.4 * past_5 + rng.normal(0, 0.04, n)  # 弱反转 + 大噪声
            log_prices[:, t] = log_prices[:, t - 5] + future_5

    rows = []
    for i, code in enumerate(codes):
        for j, dt in enumerate(dates):
            price = float(np.exp(log_prices[i, j]))
            rows.append({
                "date": dt, "code": code,
                "open": price, "close": price,
                "high": price * 1.01, "low": price * 0.99,
                "is_limit_up": False, "is_limit_down": False,
                "volume": float(rng.uniform(1e6, 1e7)),
            })
    return pd.DataFrame(rows)


def simple_signal_fn(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """简单信号: 基于训练集均值的反转因子 (top-10)"""
    train = train_data.copy().sort_values(["code", "date"])
    train["factor"] = -train.groupby("code")["close"].transform(lambda x: x.pct_change(5))
    # 用训练集最后一天的因子
    last_date = train["date"].max()
    last_factors = train[train["date"] == last_date][["code", "factor"]].dropna()

    test = test_data.copy()
    test = test.merge(last_factors, on="code", how="left")
    test = test.dropna(subset=["factor"])
    test["rank"] = test.groupby("date")["factor"].rank(method="first", ascending=False)
    sig = test[test["rank"] <= 10][["date", "code"]].copy()
    sig["signal"] = 1
    return sig


def test_walk_forward_basic():
    """基础 walk-forward 测试"""
    data = make_synthetic(n_stocks=30, n_days=504)
    wf = WalkForwardBacktest(train_size=252, test_size=63, step=63, expanding=True)
    res = wf.run(data, simple_signal_fn)

    segments = res["segments"]
    oos_eq = res["oos_equity_curve"]
    summary = res["summary"]

    print(f"\n[test_walk_forward_basic] 段数: {len(segments)}")
    for s in segments:
        print(f"  [{s['segment_id']}] train={s['train_start']}~{s['train_end']}  test={s['test_start']}~{s['test_end']}  oos_ret={s['oos_total_return']:+.4f}  sharpe={s['oos_sharpe']:+.2f}  trades={s['oos_n_trades']}")
    print(f"OOS equity curve shape: {oos_eq.shape}")
    print(f"Summary: {summary}")
    assert len(segments) > 0
    assert not oos_eq.empty
    assert "oos_sharpe_ratio" in summary
    print("[test_walk_forward_basic] PASSED")


def test_walk_forward_rolling():
    """Rolling Window (固定窗口)"""
    data = make_synthetic(n_stocks=30, n_days=504)
    wf = WalkForwardBacktest(train_size=252, test_size=63, step=63, expanding=False)
    res = wf.run(data, simple_signal_fn)
    segments = res["segments"]
    print(f"\n[test_walk_forward_rolling] Rolling Window 段数: {len(segments)}")
    for s in segments:
        print(f"  [{s['segment_id']}] train={s['n_train_days']}d  test={s['n_test_days']}d  ret={s['oos_total_return']:+.4f}")
    assert len(segments) > 0
    # 固定窗口下，每段 train_size 都应该 = 252
    for s in segments:
        assert s["n_train_days"] == 252, f"expected 252, got {s['n_train_days']}"
    print("[test_walk_forward_rolling] PASSED")


def test_walk_forward_segment_metrics():
    """段指标应在合理范围内"""
    data = make_synthetic(n_stocks=30, n_days=504)
    wf = WalkForwardBacktest(train_size=252, test_size=63, step=63)
    res = wf.run(data, simple_signal_fn)

    for s in res["segments"]:
        # 总收益、sharpe、mdd 应为有限数
        assert np.isfinite(s["oos_total_return"])
        assert np.isfinite(s["oos_sharpe"])
        assert np.isfinite(s["oos_max_drawdown"])
        assert -1.0 <= s["oos_max_drawdown"] <= 0.0
    print(f"\n[test_walk_forward_segment_metrics] 段数: {len(res['segments'])}, 所有指标有限")
    print("[test_walk_forward_segment_metrics] PASSED")


def test_edge_cases():
    """边界条件"""
    # 1) 数据不够长
    short_data = make_synthetic(n_stocks=10, n_days=100)
    wf = WalkForwardBacktest(train_size=252, test_size=63)
    res = wf.run(short_data, simple_signal_fn)
    assert res["segments"] == []
    print("\n[test_edge_cases] 短数据: PASSED (0 段)")

    # 2) 非法参数
    try:
        WalkForwardBacktest(train_size=10, test_size=5)
        assert False
    except ValueError:
        print("[test_edge_cases] 非法参数: PASSED (正确抛错)")


if __name__ == "__main__":
    test_walk_forward_basic()
    test_walk_forward_rolling()
    test_walk_forward_segment_metrics()
    test_edge_cases()
    print("\n=== Walk-Forward Backtest 所有测试通过 ===")
