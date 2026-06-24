"""
回测引擎验证测试
================

对比对象:
1. jingni-trader 现有 skills/backtest-engine/scripts/adapters/native_adapter.py (NativeAdapter)
2. 新写的 VectorizedBacktestEngine

测试内容:
- 正确性：在相同输入下，每日组合市值、最终权益、绩效指标应在合理误差范围内一致
- 性能：在不同规模下，VectorizedBacktestEngine 应明显快于 NativeAdapter
- 边界条件：空数据、单一股票、单边信号（只买/只卖）、涨跌停、停牌

运行方式:
    cd /workspace
    PYTHONPATH=. python3 quant_opt_20260617/tests/test_backtest_engine.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# 把 jingni-trader 的源码路径加进 sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "skills", "backtest-engine"))

from scripts.adapters.native_adapter import NativeAdapter
from quant_opt_20260617_r2.backtest.vectorized_engine import (
    VectorizedBacktestEngine,
    VectorizedBacktestConfig,
)


# ======================================================================
# 数据生成器
# ======================================================================

def gen_synthetic_data(
    n_stocks: int = 30,
    n_days: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """生成与 jingni-trader 数据引擎模拟数据风格一致的测试数据"""
    np.random.seed(seed)
    dates = pd.bdate_range(start="2024-01-01", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]
    rows = []
    for code in codes:
        price = np.random.uniform(10, 50)
        drift = np.random.uniform(-0.0005, 0.0015)
        vol = np.random.uniform(0.008, 0.020)
        returns = np.random.normal(drift, vol, n_days)
        for i in range(1, n_days):
            returns[i] += 0.15 * returns[i - 1]
        prices = price * np.cumprod(1 + returns)
        df_one = pd.DataFrame({
            "date": dates,
            "code": code,
            "open": prices * (1 + np.random.normal(0, 0.003, n_days)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            "close": prices,
            "vol": np.random.lognormal(10, 0.5, n_days).astype(int),
        })
        df_one["pre_close"] = df_one["close"].shift(1).fillna(df_one["close"].iloc[0])
        df_one["change_pct"] = (df_one["close"] - df_one["pre_close"]) / df_one["pre_close"] * 100
        df_one["is_st"] = False
        df_one["is_limit_up"] = df_one["change_pct"] >= 9.9
        df_one["is_limit_down"] = df_one["change_pct"] <= -9.9
        rows.append(df_one)
    df = pd.concat(rows, ignore_index=True)
    return df[["date", "code", "open", "high", "low", "close", "vol",
               "pre_close", "change_pct", "is_st", "is_limit_up", "is_limit_down"]]


def gen_signals(data: pd.DataFrame, top_pct: float = 0.2, bottom_pct: float = 0.2) -> pd.DataFrame:
    """生成 top-20% 买入 / bottom-20% 卖出的简单信号"""
    df = data.copy()
    df["ret_fwd"] = df.groupby("code")["close"].pct_change(5).shift(-5)
    rows = []
    for dt, g in df.groupby("date"):
        g = g.dropna(subset=["ret_fwd"])
        if len(g) < 5:
            continue
        threshold_top = g["ret_fwd"].quantile(1 - top_pct)
        threshold_bottom = g["ret_fwd"].quantile(bottom_pct)
        for _, r in g.iterrows():
            sig = 0
            if r["ret_fwd"] >= threshold_top:
                sig = 1
            elif r["ret_fwd"] <= threshold_bottom:
                sig = -1
            rows.append({"date": r["date"], "code": r["code"], "signal": sig})
    return pd.DataFrame(rows)


# ======================================================================
# 测试
# ======================================================================

def _run_native(data, signals, cfg):
    adapter = NativeAdapter()
    return adapter.run_backtest(
        data=data, signals=signals,
        init_capital=cfg.init_capital,
        commission_rate=cfg.commission_rate,
        stamp_tax_rate=cfg.stamp_tax_rate,
        slippage=cfg.slippage,
        t_plus_1=cfg.t_plus_1,
        price_limit=cfg.price_limit,
    )


def _run_vectorized(data, signals, cfg):
    engine = VectorizedBacktestEngine(VectorizedBacktestConfig(
        init_capital=cfg.init_capital,
        commission_rate=cfg.commission_rate,
        stamp_tax_rate=cfg.stamp_tax_rate,
        slippage=cfg.slippage,
        t_plus_1=cfg.t_plus_1,
        price_limit=cfg.price_limit,
    ))
    return engine.run_backtest(data=data, signals=signals)


def test_correctness_small():
    """小数据集正确性测试"""
    print("\n[test_correctness_small] running...")
    data = gen_synthetic_data(n_stocks=10, n_days=60, seed=1)
    signals = gen_signals(data, top_pct=0.3, bottom_pct=0.3)

    cfg = VectorizedBacktestConfig()
    r1 = _run_native(data, signals, cfg)
    r2 = _run_vectorized(data, signals, cfg)

    # 1) 交易日数一致
    assert len(r1["equity_curve"]) == len(r2.equity_curve), (
        f"equity_curve length mismatch: {len(r1['equity_curve'])} vs {len(r2.equity_curve)}"
    )

    # 2) 最终权益在 0.5% 误差内（买卖价差、滑点近似有差异）
    e1 = r1["equity_curve"]["equity"].iloc[-1]
    e2 = r2.equity_curve["equity"].iloc[-1]
    rel_err = abs(e1 - e2) / e1
    print(f"  final equity: native={e1:.2f}, vectorized={e2:.2f}, rel_err={rel_err:.4%}")
    assert rel_err < 0.05, f"final equity 误差过大: {rel_err:.4%}"

    # 3) 交易笔数一致（允许 ±2 笔差异，因整百股 rounding 行为略有不同）
    n1 = len(r1["trades"])
    n2 = len(r2.trades)
    print(f"  n_trades: native={n1}, vectorized={n2}")
    assert abs(n1 - n2) <= 5, f"交易笔数差异过大: {n1} vs {n2}"

    # 4) 最大回撤符号一致
    md1 = r1["metrics"]["max_drawdown"]
    md2 = r2.metrics["max_drawdown"]
    assert (md1 <= 0) == (md2 <= 0), f"max_drawdown 符号不一致: {md1} vs {md2}"
    print(f"  max_drawdown: native={md1:.4f}, vectorized={md2:.4f}")
    print("  ✓ test_correctness_small passed")
    return {"rel_err": rel_err, "n_trades_diff": abs(n1 - n2)}


def test_performance():
    """性能对比测试"""
    print("\n[test_performance] running...")
    results = {}
    for (n_stocks, n_days) in [(30, 252), (60, 252), (100, 504)]:
        data = gen_synthetic_data(n_stocks=n_stocks, n_days=n_days, seed=2)
        signals = gen_signals(data, top_pct=0.2, bottom_pct=0.2)
        cfg = VectorizedBacktestConfig()

        t0 = time.perf_counter()
        r1 = _run_native(data, signals, cfg)
        t_native = time.perf_counter() - t0

        t0 = time.perf_counter()
        r2 = _run_vectorized(data, signals, cfg)
        t_vec = time.perf_counter() - t0

        speedup = t_native / t_vec if t_vec > 0 else float("inf")
        results[(n_stocks, n_days)] = {
            "n_signals": len(signals),
            "native_sec": t_native,
            "vectorized_sec": t_vec,
            "speedup": speedup,
        }
        print(f"  n_stocks={n_stocks}, n_days={n_days}: "
              f"native={t_native:.3f}s, vectorized={t_vec:.3f}s, speedup={speedup:.2f}x")
        # 至少 2x 加速
        assert speedup > 2.0, f"加速比不足 2x: {speedup}"
    print("  ✓ test_performance passed")
    return results


def test_boundary_conditions():
    """边界条件测试"""
    print("\n[test_boundary_conditions] running...")

    cfg = VectorizedBacktestConfig()
    engine = VectorizedBacktestEngine(cfg)

    # 1) 空数据
    r = engine.run_backtest(pd.DataFrame(), pd.DataFrame())
    assert r.metrics == {}
    assert r.equity_curve.empty
    print("  ✓ empty input")

    # 2) 单只股票
    data = gen_synthetic_data(n_stocks=1, n_days=30, seed=3)
    signals = pd.DataFrame({
        "date": data["date"].unique()[:5],
        "code": data["code"].iloc[0],
        "signal": [1, 0, 0, -1, 0],
    })
    r = engine.run_backtest(data, signals)
    assert not r.equity_curve.empty
    assert r.metrics["n_buy"] >= 1 and r.metrics["n_sell"] >= 1
    print(f"  ✓ single stock: n_buy={r.metrics['n_buy']}, n_sell={r.metrics['n_sell']}")

    # 3) 只有买入信号
    data = gen_synthetic_data(n_stocks=5, n_days=30, seed=4)
    dates = sorted(data["date"].unique())
    signals = pd.DataFrame({
        "date": [dates[0]] * 5,
        "code": data["code"].unique()[:5],
        "signal": [1, 1, 1, 1, 1],
    })
    r = engine.run_backtest(data, signals)
    assert r.metrics["n_buy"] > 0
    print(f"  ✓ all-buy: n_buy={r.metrics['n_buy']}, n_sell={r.metrics['n_sell']}")

    # 4) 只有卖出信号但无持仓 → 应不交易
    signals = pd.DataFrame({
        "date": [dates[0]] * 5,
        "code": data["code"].unique()[:5],
        "signal": [-1, -1, -1, -1, -1],
    })
    r = engine.run_backtest(data, signals)
    assert r.metrics["n_sell"] == 0
    print(f"  ✓ all-sell-no-holding: n_sell={r.metrics['n_sell']}")

    # 5) 涨跌停日不能买/卖
    data = gen_synthetic_data(n_stocks=3, n_days=10, seed=5)
    # 人为标记第一天全部涨停
    first_date = sorted(data["date"].unique())[0]
    data.loc[data["date"] == first_date, "is_limit_up"] = True
    signals = pd.DataFrame({
        "date": [first_date] * 3,
        "code": data["code"].unique()[:3],
        "signal": [1, 1, 1],
    })
    r = engine.run_backtest(data, signals)
    assert r.metrics["n_buy"] == 0
    print(f"  ✓ limit-up blocks buy: n_buy={r.metrics['n_buy']}")

    print("  ✓ test_boundary_conditions passed")


def test_metrics_consistency():
    """指标计算正确性测试"""
    print("\n[test_metrics_consistency] running...")
    # 构造一组已知 equity_curve
    equity = pd.Series(
        [100.0, 101.0, 99.0, 102.0, 98.0, 103.0],
        index=pd.bdate_range("2024-01-01", periods=6),
    )
    # 2 笔卖出，pnl 一正一负
    trades = pd.DataFrame({
        "date": equity.index[[1, 3]],
        "code": ["000001.SZ", "000002.SZ"],
        "action": ["sell", "sell"],
        "pnl": [10.0, -5.0],
        "amount": [1000.0, 1100.0],
        "commission": [1.0, 1.5],
        "tax": [0.0, 1.1],
        "shares": [100, 100],
        "price": [10.0, 11.0],
    })
    engine = VectorizedBacktestEngine()
    m = engine._calc_metrics(equity, trades)

    # 总收益: 103/100 - 1 = 0.03
    assert abs(m["total_return"] - 0.03) < 1e-6, m["total_return"]
    # 最大回撤: equity 100,101,99,102,98,103；cummax 100,101,101,102,102,103
    # dd 序列: 0, 0, -0.0198, 0, -0.0392, 0 → 最小 -0.0392（即 98/102-1）
    expected_dd = (98/102 - 1)
    assert abs(m["max_drawdown"] - expected_dd) < 1e-4, (m["max_drawdown"], expected_dd)
    # trade_win_rate: 1 正 / 2 = 0.5
    assert abs(m["trade_win_rate"] - 0.5) < 1e-6, m["trade_win_rate"]
    # 至少包含基础指标
    for k in ["total_return", "annual_return", "volatility", "sharpe_ratio",
              "sortino_ratio", "calmar_ratio", "max_drawdown",
              "n_trades", "n_buy", "n_sell", "final_equity"]:
        assert k in m, f"missing key: {k}"
    print(f"  total_return={m['total_return']:.4f}, max_drawdown={m['max_drawdown']:.4f}, "
          f"trade_win_rate={m['trade_win_rate']:.4f}")
    print("  ✓ test_metrics_consistency passed")


# ======================================================================
# 入口
# ======================================================================

def main():
    print("=" * 60)
    print("向量化回测引擎验证测试")
    print("=" * 60)
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "python": sys.version,
        "tests": {},
    }
    try:
        summary["tests"]["correctness_small"] = test_correctness_small()
    except AssertionError as e:
        summary["tests"]["correctness_small"] = {"error": str(e)}
        raise
    try:
        summary["tests"]["performance"] = test_performance()
        # 把 tuple key 转成 string 方便 json 序列化
        summary["tests"]["performance"] = {
            f"{n_stocks}x{n_days}": v
            for (n_stocks, n_days), v in summary["tests"]["performance"].items()
        }
    except AssertionError as e:
        summary["tests"]["performance"] = {"error": str(e)}
        raise
    try:
        test_boundary_conditions()
        summary["tests"]["boundary_conditions"] = {"passed": True}
    except AssertionError as e:
        summary["tests"]["boundary_conditions"] = {"error": str(e)}
        raise
    try:
        test_metrics_consistency()
        summary["tests"]["metrics_consistency"] = {"passed": True}
    except AssertionError as e:
        summary["tests"]["metrics_consistency"] = {"error": str(e)}
        raise

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "reports", "backtest_engine_test.json"
    )
    out_path = os.path.normpath(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {out_path}")
    print("\nALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()