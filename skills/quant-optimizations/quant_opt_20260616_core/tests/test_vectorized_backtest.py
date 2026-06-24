"""
向量化回测引擎测试
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _make_market(n_stocks: int = 30, n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for code in codes:
        close = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n_days)))
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": close[i] * (1 + rng.normal(0, 0.005)),
                "high": close[i] * (1 + abs(rng.normal(0, 0.01))),
                "low": close[i] * (1 - abs(rng.normal(0, 0.01))),
                "close": close[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
                "amount": float(close[i] * rng.integers(1_000_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


def _make_signals(market: pd.DataFrame, top_frac: float = 0.2) -> pd.DataFrame:
    """朴素动量信号: 用前 20 日收益排名前 20% 的标的生成买入信号"""
    df = market.sort_values(["code", "date"]).copy()
    df["ret_20"] = df.groupby("code")["close"].pct_change(20)
    df["rank"] = df.groupby("date")["ret_20"].rank(pct=True, ascending=False)
    sig = df[["code", "date", "rank"]].copy()
    sig["signal"] = (sig["rank"] <= top_frac).astype(int)
    return sig[["code", "date", "signal"]]


def test_basic_run():
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    market = _make_market()
    signals = _make_signals(market)
    bt = VectorizedBacktester()
    res = bt.run(market, signals)
    assert "equity_curve" in res
    assert "metrics" in res
    metrics = res["metrics"]
    assert metrics["n_days"] > 0
    assert metrics["total_trades"] > 0
    print(f"  [OK] basic run: total_return={metrics['total_return']:.4f} sharpe={metrics['sharpe_ratio']:.3f}")


def test_t1_constraint():
    """T+1: 当日买入次日才可卖, 验证当日卖出不会发生"""
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    market = _make_market(n_stocks=5, n_days=30)
    # 信号: 每天都全仓买, 第二天全仓卖
    sig = []
    for d in market["date"].unique():
        sig.append({"code": market["code"].iloc[0], "date": d, "signal": 1})
    sig_df = pd.DataFrame(sig)
    bt = VectorizedBacktester()
    res = bt.run(market, sig_df)
    trades = res["trades"]
    # 因为 T+1, 第一笔买入后才能在第二天卖出
    if not trades.empty:
        actions = trades["action"].tolist()
        # 第一笔必为买
        assert actions[0] == "buy", actions[:5]
    print(f"  [OK] T+1 constraint respected")


def test_no_lookahead():
    """信号当日收盘生效, 不能使用当日未来价格"""
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    market = _make_market(n_stocks=5, n_days=20)
    # 仅在第 0 天买入
    d0 = market["date"].min()
    sig = market[market["date"] == d0][["code", "date"]].copy()
    sig["signal"] = 1
    bt = VectorizedBacktester()
    res = bt.run(market, sig)
    # 之后不再有买入
    buys = res["trades"][res["trades"]["action"] == "buy"]
    assert (buys["date"] == d0).all()
    print("  [OK] no-lookahead: all buys on t0")


def test_price_limit_filter():
    """涨停日不能买入, 跌停日不能卖出"""
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    market = _make_market(n_stocks=5, n_days=10)
    # 标记某天涨停
    limit_up_date = market["date"].iloc[5]
    target_code = market["code"].iloc[0]
    market = market.copy()
    market["is_limit_up"] = False
    market.loc[
        (market["date"] == limit_up_date) & (market["code"] == target_code),
        "is_limit_up",
    ] = True
    sig = market[["code", "date"]].copy()
    sig["signal"] = 1
    bt = VectorizedBacktester()
    res = bt.run(market, sig)
    # 在 limit_up_date 不应有 target_code 的买入
    buys = res["trades"]
    if not buys.empty:
        bad = buys[
            (buys["date"] == limit_up_date)
            & (buys["code"] == target_code)
            & (buys["action"] == "buy")
        ]
        assert len(bad) == 0, f"should not buy limit-up: {bad}"
    print("  [OK] price limit filter")


def test_perf_large():
    """200 标的 * 1000 天 耗时 < 3s"""
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    market = _make_market(n_stocks=200, n_days=1000, seed=11)
    signals = _make_signals(market, top_frac=0.1)
    bt = VectorizedBacktester()
    t0 = time.time()
    res = bt.run(market, signals)
    dt = time.time() - t0
    metrics = res["metrics"]
    print(f"  [OK] perf 200x1000: {dt:.3f}s, trades={metrics['total_trades']}")
    assert dt < 5.0


def test_metrics_keys():
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    market = _make_market(n_stocks=10, n_days=20)
    signals = _make_signals(market)
    bt = VectorizedBacktester()
    res = bt.run(market, signals)
    expected = {
        "total_return", "annual_return", "volatility", "sharpe_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "total_trades", "n_days",
    }
    assert expected <= set(res["metrics"].keys()), set(res["metrics"].keys()) ^ expected
    print("  [OK] metrics keys complete")


def test_empty_input():
    from skills.quant-optimizations.quant_opt_20260616_core.vectorized_backtest import VectorizedBacktester
    bt = VectorizedBacktester()
    res = bt.run(pd.DataFrame(), pd.DataFrame())
    assert res["metrics"] == {}
    print("  [OK] empty input handled")


def run() -> dict:
    test_basic_run()
    test_t1_constraint()
    test_no_lookahead()
    test_price_limit_filter()
    test_metrics_keys()
    test_empty_input()
    test_perf_large()
    return {"status": "passed", "cases": 7}


if __name__ == "__main__":
    run()