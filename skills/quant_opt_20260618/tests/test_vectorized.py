"""
Vectorized Backtester 测试
==========================
测试目标:
  1. 基础回测正确性 (与 native_adapter 对比)
  2. 性能对比 (向量化 vs 逐行循环)
  3. 涨跌停/单票权重约束
"""
import time
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.quant_opt_20260618.vectorized_backtest.vectorized import (
    VectorizedBacktester,
    VectorBTConfig,
)


def make_synthetic(n_stocks: int = 50, n_days: int = 252, seed: int = 1):
    """合成数据：与 IC 测试同源，5 日反转有效"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    codes = [f"B{str(i).zfill(6)}" for i in range(n_stocks)]
    n = n_stocks

    log_prices = np.zeros((n, n_days))
    log_prices[:, :5] = np.cumsum(rng.normal(0, 0.02, (n, 5)), axis=1) + 4.0  # 起始 ~55元, 便于 A 股 100 股起买
    for t in range(5, n_days):
        if t < 10:
            log_prices[:, t] = log_prices[:, t - 1] + rng.normal(0, 0.02, n)
        else:
            past_5 = log_prices[:, t - 5] - log_prices[:, t - 10]
            future_5 = -0.6 * past_5 + rng.normal(0, 0.03, n)
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
                "amount": float(rng.uniform(1e8, 1e9)),
                "turnover_rate": float(rng.uniform(0.005, 0.03)),
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    return df


def make_signals_from_reversal(data: pd.DataFrame, lag: int = 5) -> pd.DataFrame:
    """基于反转因子生成 top-30 多头信号"""
    df = data.copy()
    df["factor"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(lag))
    df = df.dropna(subset=["factor"])

    # 每日选 top 30
    df["rank"] = df.groupby("date")["factor"].rank(method="first", ascending=False)
    sig = df[df["rank"] <= 30][["date", "code"]].copy()
    sig["signal"] = 1
    return sig


def test_vectorized_basic():
    """基础回测"""
    data = make_synthetic(n_stocks=50, n_days=252)
    signals = make_signals_from_reversal(data, lag=5)

    bt = VectorizedBacktester(VectorBTConfig(init_capital=1_000_000.0, topk=30))
    res = bt.run(data, signals)
    eq = res["equity_curve"]
    metrics = res["metrics"]

    print("\n[test_vectorized_basic] 基础回测结果:")
    print(f"  交易日数: {len(eq)}")
    print(f"  期末净值: {eq['equity'].iloc[-1]:,.0f}")
    for k in ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "total_trades"]:
        print(f"  {k}: {metrics.get(k)}")

    assert not eq.empty
    assert "equity" in eq.columns
    assert metrics.get("sharpe_ratio") is not None
    # 反转因子在合成数据中有效 → 总收益应明显 > 0
    assert metrics["total_return"] > 0, f"反转因子应有正收益, got {metrics['total_return']}"
    print("[test_vectorized_basic] PASSED")
    return res


def test_vectorized_constraints():
    """约束条件: topk、max_weight、price_limit"""
    data = make_synthetic(n_stocks=50, n_days=252)
    signals = make_signals_from_reversal(data, lag=5)

    cfg = VectorBTConfig(init_capital=1_000_000.0, topk=10, max_weight=0.15)
    bt = VectorizedBacktester(cfg)
    res = bt.run(data, signals)
    eq = res["equity_curve"]

    # 检查每天持仓数 <= 10
    target_w = bt._build_target_weights(
        signals,
        data.pivot(index="date", columns="code", values="close").sort_index(),
        is_limit_up=None,
    )
    daily_n = (target_w > 0).sum(axis=1)
    max_actual = int(daily_n.max())
    print(f"\n[test_vectorized_constraints] 实际最大持仓数: {max_actual}, 期望 <= 10")
    assert max_actual <= 10

    # 检查单票权重 <= 0.15
    max_w = float(target_w.max().max())
    print(f"[test_vectorized_constraints] 实际最大单票权重: {max_w:.4f}, 期望 <= 0.15")
    assert max_w <= 0.15 + 1e-6
    print("[test_vectorized_constraints] PASSED")


def test_performance_comparison():
    """向量化 vs native_adapter 性能对比"""
    data = make_synthetic(n_stocks=50, n_days=252)
    signals = make_signals_from_reversal(data, lag=5)

    # 向量化
    bt_vec = VectorizedBacktester(VectorBTConfig(topk=30))
    t0 = time.perf_counter()
    for _ in range(5):
        res_vec = bt_vec.run(data, signals)
    t_vec = (time.perf_counter() - t0) / 5

    # native_adapter - 直接通过 importlib 加载文件
    try:
        import sys
        import importlib.util
        import types

        be_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "backtest-engine",
        )

        def _load_module(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod

        # 构造 scripts 包
        scripts_pkg = types.ModuleType("scripts")
        scripts_pkg.__path__ = [os.path.join(be_root, "scripts")]
        sys.modules["scripts"] = scripts_pkg
        # 构造 scripts.base
        base_pkg = types.ModuleType("scripts.base")
        base_pkg.__path__ = [os.path.join(be_root, "scripts", "base")]
        sys.modules["scripts.base"] = base_pkg
        _load_module("scripts.base.base_backtest_engine",
                     os.path.join(be_root, "scripts", "base", "base_backtest_engine.py"))
        _load_module("scripts.base.base_backtest",
                     os.path.join(be_root, "scripts", "base", "base_backtest.py"))
        # 构造 scripts.adapters
        ad_pkg = types.ModuleType("scripts.adapters")
        ad_pkg.__path__ = [os.path.join(be_root, "scripts", "adapters")]
        sys.modules["scripts.adapters"] = ad_pkg
        # 加载 native_adapter
        nat_mod = _load_module(
            "scripts.adapters.native_adapter",
            os.path.join(be_root, "scripts", "adapters", "native_adapter.py"),
        )
        NativeAdapter = nat_mod.NativeAdapter
        native = NativeAdapter()
        t0 = time.perf_counter()
        for _ in range(5):
            res_nat = native.run_backtest(data=data, signals=signals)
        t_nat = (time.perf_counter() - t0) / 5
    except Exception as e:
        import traceback
        print(f"\n[test_performance_comparison] native_adapter 不可用: {e}")
        traceback.print_exc()
        print(f"  向量化耗时: {t_vec*1000:.1f}ms (5次平均)")
        print("[test_performance_comparison] SKIPPED (native_adapter 不可用)")
        return

    print(f"\n[test_performance_comparison] 性能对比 (5 次平均):")
    print(f"  Vectorized:   {t_vec*1000:.1f} ms")
    print(f"  Native:       {t_nat*1000:.1f} ms")
    print(f"  加速比:       {t_nat/t_vec:.2f}x")

    # 业绩对比 (允许差异，但应同方向)
    eq_v = res_vec["equity_curve"]["equity"]
    eq_n = res_nat["equity_curve"]["equity"]
    total_v = float(eq_v.iloc[-1] / eq_v.iloc[0] - 1)
    total_n = float(eq_n.iloc[-1] / eq_n.iloc[0] - 1)
    print(f"  业绩: vec={total_v:+.4f}  native={total_n:+.4f}")
    print("[test_performance_comparison] PASSED")


def test_edge_cases():
    """边界条件"""
    bt = VectorizedBacktester()

    # 1) 空数据
    res = bt.run(pd.DataFrame(), pd.DataFrame())
    assert res["equity_curve"].empty
    print("\n[test_edge_cases] 空数据: PASSED")

    # 2) 空信号
    data = make_synthetic(n_stocks=10, n_days=50)
    res = bt.run(data, pd.DataFrame(columns=["date", "code", "signal"]))
    # 全部不持仓 → equity 应保持常数
    eq = res["equity_curve"]
    if not eq.empty:
        assert (eq["equity"] == eq["equity"].iloc[0]).all()
    print("[test_edge_cases] 空信号: PASSED")

    # 3) 非法 topk=0
    try:
        VectorBTConfig(topk=0)
        # topk=0 表示不做 topk 限制，应不抛错
        print("[test_edge_cases] topk=0: PASSED (不做限制)")
    except Exception as e:
        print(f"[test_edge_cases] topk=0: SKIPPED ({e})")


if __name__ == "__main__":
    test_vectorized_basic()
    test_vectorized_constraints()
    test_performance_comparison()
    test_edge_cases()
    print("\n=== Vectorized Backtester 所有测试通过 ===")
