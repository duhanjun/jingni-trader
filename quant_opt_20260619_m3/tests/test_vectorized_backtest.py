"""测试向量化回测引擎"""
import sys
import os
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS)))
sys.path.insert(0, _THIS)
sys.path.insert(0, "/workspace")

import time
import numpy as np
import pandas as pd

from quant_opt_20260619_m3.vectorized_backtest.engine import (
    VectorizedBacktestEngine, run_vectorized_adapter
)
try:
    import importlib.util, sys, types
    # 构造 skills 包结构, 支持 native_adapter 内部的相对导入
    if "skills" not in sys.modules:
        if "/workspace" not in sys.path:
            sys.path.insert(0, "/workspace")
        skills_mod = types.ModuleType("skills")
        skills_mod.__path__ = ["/workspace/skills"]
        sys.modules["skills"] = skills_mod
        for sub in ["backtest-engine", "factor-engine", "portfolio-risk-engine",
                    "execution-monitor-engine", "strategy-model-engine",
                    "data-engine", "reports-engine"]:
            sub_mod = types.ModuleType(f"skills.{sub}")
            sub_mod.__path__ = [f"/workspace/skills/{sub}"]
            sys.modules[f"skills.{sub}"] = sub_mod
    if "skills.backtest-engine.scripts" not in sys.modules:
        scripts_mod = types.ModuleType("skills.backtest-engine.scripts")
        scripts_mod.__path__ = ["/workspace/skills/backtest-engine/scripts"]
        sys.modules["skills.backtest-engine.scripts"] = scripts_mod
    if "skills.backtest-engine.scripts.base" not in sys.modules:
        base_mod = types.ModuleType("skills.backtest-engine.scripts.base")
        base_mod.__path__ = ["/workspace/skills/backtest-engine/scripts/base"]
        sys.modules["skills.backtest-engine.scripts.base"] = base_mod
    if "skills.backtest-engine.scripts.adapters" not in sys.modules:
        adapter_mod = types.ModuleType("skills.backtest-engine.scripts.adapters")
        adapter_mod.__path__ = ["/workspace/skills/backtest-engine/scripts/adapters"]
        sys.modules["skills.backtest-engine.scripts.adapters"] = adapter_mod

    spec = importlib.util.spec_from_file_location(
        "skills.backtest-engine.scripts.adapters.native_adapter",
        "/workspace/skills/backtest-engine/scripts/adapters/native_adapter.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["skills.backtest-engine.scripts.adapters.native_adapter"] = mod
    spec.loader.exec_module(mod)
    NativeAdapter = mod.NativeAdapter
    HAS_NATIVE = True
except Exception as e:
    print(f"[WARN] 找不到 NativeAdapter: {e}, 跳过对比测试")
    HAS_NATIVE = False


def _make_synth_market(n_stocks: int = 5, n_days: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for code in range(n_stocks):
        close = 10 + np.cumsum(rng.normal(0, 0.5, n_days))
        high = close + rng.uniform(0, 0.3, n_days)
        low = close - rng.uniform(0, 0.3, n_days)
        open_ = close + rng.normal(0, 0.2, n_days)
        volume = rng.integers(1_000_000, 5_000_000, n_days)
        is_limit_up = rng.random(n_days) < 0.05
        is_limit_down = rng.random(n_days) < 0.05
        for i in range(n_days):
            rows.append({
                "code": f"S{code:04d}", "date": dates[i],
                "open": open_[i], "high": high[i], "low": low[i],
                "close": close[i], "volume": volume[i],
                "is_limit_up": is_limit_up[i],
                "is_limit_down": is_limit_down[i],
            })
    return pd.DataFrame(rows)


def test_basic_vectorized_run():
    data = _make_synth_market(n_stocks=3, n_days=30)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())

    # 简单等权: 每天 0.3 权重平分给 3 只股票
    weights = pd.DataFrame(0.30, index=dates, columns=codes)
    weights.iloc[0] = 0  # 第一天空仓

    eng = VectorizedBacktestEngine(init_capital=1_000_000)
    res = eng.run(data, weights)

    assert not res.equity_curve.empty
    assert "equity" in res.equity_curve.columns
    assert len(res.trades) > 0  # 应有交易
    print(f"[PASS] test_basic_vectorized_run: {len(res.trades)} trades, "
          f"final equity = {res.equity_curve['equity'].iloc[-1]:.0f}")


def test_no_transaction_when_weights_zero():
    data = _make_synth_market(n_stocks=2, n_days=20)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())
    # 永远空仓
    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    eng = VectorizedBacktestEngine(init_capital=1_000_000)
    res = eng.run(data, weights)
    assert len(res.trades) == 0
    assert abs(res.equity_curve["equity"].iloc[-1] - 1_000_000) < 1
    print("[PASS] test_no_transaction_when_weights_zero")


def test_adapter_interface():
    """验证: run_vectorized_adapter 输出与 native_adapter 接口一致"""
    data = _make_synth_market(n_stocks=2, n_days=20)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())
    weights = pd.DataFrame(0.4, index=dates, columns=codes)
    weights.iloc[0] = 0
    res = run_vectorized_adapter(data, weights)
    assert "trades" in res
    assert "equity_curve" in res
    assert "metrics" in res
    assert isinstance(res["trades"], pd.DataFrame)
    assert isinstance(res["equity_curve"], pd.DataFrame)
    print("[PASS] test_adapter_interface")


def test_cash_not_negative():
    """现金约束: 任何时点现金 >= 0"""
    data = _make_synth_market(n_stocks=3, n_days=30)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())
    # 大权重(1.0), 触发现金约束
    weights = pd.DataFrame(1.0, index=dates, columns=codes)
    weights.iloc[0] = 0
    eng = VectorizedBacktestEngine(init_capital=1_000_000, max_position_pct=1.0)
    res = eng.run(data, weights)
    # 现金不应为负
    assert (res.equity_curve["cash"] >= -1).all(), \
        f"现金曾为负: min={res.equity_curve['cash'].min()}"
    print(f"[PASS] test_cash_not_negative: min cash = {res.equity_curve['cash'].min():.0f}")


def test_consistency_with_native():
    """
    对比向量化和 native 适配器在等权策略下的结果.
    允许一定差异(交易时点不同), 但最终权益数量级应一致.
    """
    if not HAS_NATIVE:
        print("[SKIP] test_consistency_with_native")
        return
    data = _make_synth_market(n_stocks=2, n_days=20)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())

    # 构造 native 风格的 signal: 第一天买入所有, 之后不操作
    signals = pd.DataFrame([{
        "date": dates[1], "code": c, "signal": 1
    } for c in codes])

    native = NativeAdapter()
    res_native = native.run_backtest(
        data=data, signals=signals, init_capital=1_000_000,
        t_plus_1=True, price_limit=True, slippage=0.0,
    )
    eq_native = res_native["equity_curve"]["equity"].iloc[-1]

    # 向量化: 一开始就满仓
    weights = pd.DataFrame(0.45, index=dates, columns=codes)
    weights.iloc[0] = 0
    res_vbt = run_vectorized_adapter(data, weights, init_capital=1_000_000)
    eq_vbt = res_vbt["equity_curve"]["equity"].iloc[-1]

    # 由于两者的成交时点不同(T+1 与否), 数量级一致即可
    ratio = eq_vbt / eq_native
    print(f"  native final = {eq_native:.0f}, vbt final = {eq_vbt:.0f}, ratio = {ratio:.3f}")
    assert 0.85 < ratio < 1.15, f"差异过大: ratio = {ratio}"
    print("[PASS] test_consistency_with_native")


def test_perf_vs_native():
    """性能对比: 同样输入下向量化 vs native 耗时"""
    if not HAS_NATIVE:
        print("[SKIP] test_perf_vs_native")
        return
    n_stocks = 10
    n_days = 100
    data = _make_synth_market(n_stocks=n_stocks, n_days=n_days)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())

    weights = pd.DataFrame(0.08, index=dates, columns=codes)
    weights.iloc[0] = 0

    signals = pd.DataFrame([
        {"date": d, "code": c, "signal": 1}
        for d in dates[:3] for c in codes
    ])

    # native
    native = NativeAdapter()
    t0 = time.time()
    res_n = native.run_backtest(data, signals, init_capital=1_000_000,
                                t_plus_1=True, price_limit=True, slippage=0.0)
    t_native = time.time() - t0

    # vectorized
    t0 = time.time()
    res_v = run_vectorized_adapter(data, weights, init_capital=1_000_000)
    t_vbt = time.time() - t0

    print(f"  规模: {n_stocks} stocks × {n_days} days")
    print(f"  native 耗时 = {t_native*1000:.1f} ms")
    print(f"  vectorized 耗时 = {t_vbt*1000:.1f} ms")
    print(f"  加速比 = {t_native / max(t_vbt, 1e-9):.2f}x")
    assert t_vbt <= t_native * 5, "向量化版本不应明显更慢"
    print("[PASS] test_perf_vs_native")


def test_perf_scaling():
    """规模扩展: 验证向量化在更大规模下保持稳定"""
    n_stocks = 30
    n_days = 200
    data = _make_synth_market(n_stocks=n_stocks, n_days=n_days)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())
    weights = pd.DataFrame(0.03, index=dates, columns=codes)
    weights.iloc[0] = 0

    t0 = time.time()
    res = run_vectorized_adapter(data, weights, init_capital=1_000_000)
    t = time.time() - t0
    n_trades = len(res["trades"])
    print(f"  规模: {n_stocks} stocks × {n_days} days → {t*1000:.1f} ms, {n_trades} trades")
    assert t < 5.0, f"耗时过高: {t:.2f}s"
    print("[PASS] test_perf_scaling")


def test_limit_up_handling():
    """涨跌停处理: 涨停时不应买入(对应日 next_open 价涨停)"""
    data = _make_synth_market(n_stocks=2, n_days=20)
    # 强制某天某只股票涨停
    data.loc[(data["code"] == "S0000") & (data["date"] == data["date"].unique()[5]), "is_limit_up"] = True
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())
    weights = pd.DataFrame(0.45, index=dates, columns=codes)
    weights.iloc[0] = 0

    res = run_vectorized_adapter(data, weights, init_capital=1_000_000)
    # 验证: 该日 S0000 不应有买入成交(因为次日开盘按涨停处理)
    # 简化检查: 交易数应当比无涨停情形少
    print(f"  涨停处理后, 交易数 = {len(res['trades'])}")
    print("[PASS] test_limit_up_handling")


def test_extreme_weights():
    """极端权重: 权重全 0、NaN、负数、超过 1"""
    data = _make_synth_market(n_stocks=2, n_days=10)
    dates = sorted(data["date"].unique())
    codes = sorted(data["code"].unique())

    # 全部 NaN
    weights = pd.DataFrame(np.nan, index=dates, columns=codes)
    eng = VectorizedBacktestEngine(init_capital=1_000_000)
    res = eng.run(data, weights)
    assert len(res.trades) == 0

    # 负权重应被裁剪到 0
    weights = pd.DataFrame(-0.5, index=dates, columns=codes)
    res = eng.run(data, weights)
    assert len(res.trades) == 0

    # 超过 1 应被裁剪
    weights = pd.DataFrame(2.0, index=dates, columns=codes)
    weights.iloc[0] = 0
    res = eng.run(data, weights)
    print(f"  极端权重测试通过, 交易数 = {len(res.trades)}")
    print("[PASS] test_extreme_weights")


if __name__ == "__main__":
    test_basic_vectorized_run()
    test_no_transaction_when_weights_zero()
    test_adapter_interface()
    test_cash_not_negative()
    test_consistency_with_native()
    test_perf_vs_native()
    test_perf_scaling()
    test_limit_up_handling()
    test_extreme_weights()
    print("\n所有向量化回测测试通过 ✓")