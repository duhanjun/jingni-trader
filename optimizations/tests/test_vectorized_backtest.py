"""
OPTIMIZATION 1 验证：向量化回测引擎
====================================
测试内容：
(a) 正确性：原始 vs 向量化，equity_curve 逐日一致、trades 一致、最终现金一致
(b) 性能：n_stocks=80, n_days=400，打印两者耗时与加速比，断言向量化更快
(c) 边界：空信号、单只股票、单日、全卖无持仓、全买无现金

运行：python tests/test_vectorized_backtest.py
"""
import sys
import os
import time

# 让 tests/ 脚本能直接 import optimizations 目录下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_generator import generate_test_data
from vectorized_backtest import run_original_backtest, VectorizedBacktest


def _check_equity_identical(r_orig, r_vec, label=""):
    """断言两条 equity_curve 逐日一致"""
    eq_o = r_orig["equity_curve"].reset_index(drop=True)
    eq_v = r_vec["equity_curve"].reset_index(drop=True)
    assert len(eq_o) == len(eq_v), f"[{label}] equity 行数不一致 {len(eq_o)} vs {len(eq_v)}"
    for col in ["equity", "cash", "market_value"]:
        a = eq_o[col].to_numpy(dtype=float)
        b = eq_v[col].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-6), (
            f"[{label}] equity_curve.{col} 不一致, max diff={np.max(np.abs(a-b))}"
        )
    # position_count 也应一致
    assert list(eq_o["position_count"]) == list(eq_v["position_count"]), \
        f"[{label}] position_count 不一致"


def _check_trades_identical(r_orig, r_vec, label=""):
    """断言 trades 笔数与总金额一致"""
    t_o, t_v = r_orig["trades"], r_vec["trades"]
    assert len(t_o) == len(t_v), f"[{label}] trades 笔数不一致 {len(t_o)} vs {len(t_v)}"
    if len(t_o) == 0:
        return
    assert t_o["amount"].sum() == t_v["amount"].sum(), f"[{label}] trades amount 总和不一致"
    assert t_o["commission"].sum() == t_v["commission"].sum(), f"[{label}] commission 总和不一致"
    # 最终现金（equity_curve 最后一行 cash）应一致
    assert np.isclose(
        r_orig["equity_curve"]["cash"].iloc[-1],
        r_vec["equity_curve"]["cash"].iloc[-1],
        rtol=1e-9, atol=1e-6,
    ), f"[{label}] 最终 cash 不一致"


# ---------------- (a) 正确性 ----------------
def test_correctness():
    print("\n=== [1a] 正确性测试：原始 vs 向量化 ===")
    data, signals = generate_test_data(n_stocks=50, n_days=300, seed=42)
    r_orig = run_original_backtest(data, signals)
    r_vec = VectorizedBacktest().run_backtest(data, signals)

    _check_equity_identical(r_orig, r_vec, "correctness")
    _check_trades_identical(r_orig, r_vec, "correctness")

    # 最终权益一致
    final_o = r_orig["equity_curve"]["equity"].iloc[-1]
    final_v = r_vec["equity_curve"]["equity"].iloc[-1]
    assert np.isclose(final_o, final_v, rtol=1e-9, atol=1e-6), \
        f"最终权益不一致 {final_o} vs {final_v}"

    print(f"  equity_curve 行数: {len(r_orig['equity_curve'])}")
    print(f"  trades 笔数: {len(r_orig['trades'])}")
    print(f"  最终权益: orig={final_o:.4f}  vec={final_v:.4f}")
    print(f"  逐日 equity 最大绝对差: "
          f"{np.max(np.abs(r_orig['equity_curve']['equity'].to_numpy() - r_vec['equity_curve']['equity'].to_numpy())):.2e}")
    print("  [PASS] 正确性：原始与向量化结果一致")


# ---------------- (b) 性能 ----------------
def test_performance():
    print("\n=== [1b] 性能测试：n_stocks=80, n_days=400 ===")
    data, signals = generate_test_data(n_stocks=80, n_days=400, seed=2024)

    t0 = time.perf_counter()
    r_orig = run_original_backtest(data, signals)
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_vec = VectorizedBacktest().run_backtest(data, signals)
    t_vec = time.perf_counter() - t0

    # 性能测试也要保证结果一致
    _check_equity_identical(r_orig, r_vec, "perf")
    _check_trades_identical(r_orig, r_vec, "perf")

    speedup = t_orig / t_vec if t_vec > 0 else float("inf")
    print(f"  原始耗时:   {t_orig:.3f}s")
    print(f"  向量化耗时: {t_vec:.3f}s")
    print(f"  加速比:     {speedup:.2f}x")
    assert t_vec < t_orig, f"向量化应更快 (orig={t_orig}, vec={t_vec})"
    print("  [PASS] 向量化更快且结果一致")
    return {"orig": t_orig, "vec": t_vec, "speedup": speedup}


# ---------------- (c) 边界 ----------------
def test_boundary_empty_signals():
    print("\n=== [1c-1] 边界：空信号 ===")
    data, _ = generate_test_data(n_stocks=5, n_days=20, seed=1)
    empty_signals = pd.DataFrame(columns=["code", "date", "signal"])
    r_orig = run_original_backtest(data, empty_signals)
    r_vec = VectorizedBacktest().run_backtest(data, empty_signals)
    assert r_orig["equity_curve"].empty and r_vec["equity_curve"].empty
    print("  [PASS] 空信号返回空结果")


def test_boundary_single_stock_single_day():
    print("\n=== [1c-2] 边界：单只股票单日 ===")
    data, _ = generate_test_data(n_stocks=1, n_days=1, seed=2)
    # 构造一个买入信号
    sig = pd.DataFrame([{"code": data["code"].iloc[0], "date": data["date"].iloc[0], "signal": 1}])
    r_orig = run_original_backtest(data, sig)
    r_vec = VectorizedBacktest().run_backtest(data, sig)
    _check_equity_identical(r_orig, r_vec, "single-day")
    _check_trades_identical(r_orig, r_vec, "single-day")
    print(f"  trades: {len(r_orig['trades'])}, final cash close: "
          f"{np.isclose(r_orig['equity_curve']['cash'].iloc[-1], r_vec['equity_curve']['cash'].iloc[-1])}")
    print("  [PASS] 单股单日一致")


def test_boundary_all_sell_no_position():
    print("\n=== [1c-3] 边界：全卖但无持仓 ===")
    data, _ = generate_test_data(n_stocks=5, n_days=10, seed=3)
    # 全部卖出信号，但没有任何持仓 -> 不应产生 sell trade
    sig = pd.DataFrame([
        {"code": c, "date": data["date"].iloc[0], "signal": -1}
        for c in data["code"].unique()
    ])
    r_orig = run_original_backtest(data, sig)
    r_vec = VectorizedBacktest().run_backtest(data, sig)
    _check_equity_identical(r_orig, r_vec, "all-sell-no-pos")
    _check_trades_identical(r_orig, r_vec, "all-sell-no-pos")
    # 无持仓时不应有 sell 成交
    if not r_orig["trades"].empty:
        assert (r_orig["trades"]["action"] != "sell").all(), "无持仓不应有卖出成交"
    print("  [PASS] 全卖无持仓一致")


def test_boundary_all_buy_no_cash():
    print("\n=== [1c-4] 边界：全买但几乎无现金 ===")
    data, _ = generate_test_data(n_stocks=5, n_days=10, seed=4)
    sig = pd.DataFrame([
        {"code": c, "date": data["date"].iloc[0], "signal": 1}
        for c in data["code"].unique()
    ])
    # 初始资金极小，买不起任何整手
    r_orig = run_original_backtest(data, sig, init_capital=1.0)
    r_vec = VectorizedBacktest().run_backtest(data, sig, init_capital=1.0)
    _check_equity_identical(r_orig, r_vec, "all-buy-no-cash")
    _check_trades_identical(r_orig, r_vec, "all-buy-no-cash")
    print("  [PASS] 全买无现金一致")


def run_all():
    test_correctness()
    perf = test_performance()
    test_boundary_empty_signals()
    test_boundary_single_stock_single_day()
    test_boundary_all_sell_no_position()
    test_boundary_all_buy_no_cash()
    print("\n=== 全部回测测试通过 ===")
    return perf


if __name__ == "__main__":
    run_all()
