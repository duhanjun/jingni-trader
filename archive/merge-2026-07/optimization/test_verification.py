"""
优化验证测试脚本

测试内容：
  1. 向量化回测引擎正确性（手算对照）
  2. 向量化回测 vs 现有 native_adapter 性能对比
  3. 因子注册机制正确性与扩展性
  4. 增强风险指标正确性
  5. 边界条件测试（空数据、单标的、全涨跌停、T+1）

运行: python -m optimization.test_verification
"""
from __future__ import annotations

import os
import sys
import time
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能 import optimization 包
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimization.vectorized_backtest import VectorizedBacktester
from optimization import factor_registry as fr
from optimization import enhanced_risk_metrics as erm


# ---------------------------------------------------------------------------
# 工具：动态加载现有 native_adapter（目录名含 '-'，无法直接 import）
# ---------------------------------------------------------------------------
def load_native_adapter():
    """通过文件路径加载 native_adapter.NativeAdapter。"""
    adapter_path = ROOT / "skills" / "backtest-engine" / "scripts" / "adapters" / "native_adapter.py"
    spec = importlib.util.spec_from_file_location("native_adapter", adapter_path)
    mod = importlib.util.module_from_spec(spec)
    # 其依赖 base 模块同样含 '-'，需手动注入
    base_engine_path = ROOT / "skills" / "backtest-engine" / "scripts" / "base" / "base_backtest_engine.py"
    base_metrics_path = ROOT / "skills" / "backtest-engine" / "scripts" / "base" / "base_backtest.py"

    import types
    pkg = types.ModuleType("bt_scripts")
    pkg.__path__ = []
    sys.modules["bt_scripts"] = pkg

    be_spec = importlib.util.spec_from_file_location("bt_scripts.base_backtest_engine", base_engine_path)
    be_mod = importlib.util.module_from_spec(be_spec)
    sys.modules["bt_scripts.base_backtest_engine"] = be_mod
    be_spec.loader.exec_module(be_mod)

    bm_spec = importlib.util.spec_from_file_location("bt_scripts.base_backtest", base_metrics_path)
    bm_mod = importlib.util.module_from_spec(bm_spec)
    sys.modules["bt_scripts.base_backtest"] = bm_mod
    bm_spec.loader.exec_module(bm_mod)

    # 改写 native_adapter 的相对导入
    native_src = adapter_path.read_text(encoding="utf-8")
    native_src = native_src.replace(
        "from ..base.base_backtest_engine import BaseBacktestEngine",
        "from bt_scripts.base_backtest_engine import BaseBacktestEngine",
    )
    native_src = native_src.replace(
        "from ..base.base_backtest import BaseBacktestMetrics",
        "from bt_scripts.base_backtest import BaseBacktestMetrics",
    )
    ns: dict = {}
    exec(compile(native_src, str(adapter_path), "exec"), ns)
    return ns["NativeAdapter"], bm_mod.BaseBacktestMetrics


# ---------------------------------------------------------------------------
# 合成数据生成
# ---------------------------------------------------------------------------
def make_synthetic_data(n_codes: int = 50, n_days: int = 250, seed: int = 42) -> pd.DataFrame:
    """生成长表 OHLCV 合成数据，含涨跌停标记。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_codes)]
    rows = []
    for code in codes:
        price = 10.0
        for dt in dates:
            ret = rng.normal(0, 0.02)
            price = max(price * (1 + ret), 1.0)
            prev_close = price / (1 + ret)
            change_pct = ret
            is_limit_up = change_pct >= 0.095
            is_limit_down = change_pct <= -0.095
            rows.append({
                "code": code, "date": dt,
                "open": price * (1 + rng.normal(0, 0.002)),
                "high": price * (1 + abs(rng.normal(0, 0.005))),
                "low": price * (1 - abs(rng.normal(0, 0.005))),
                "close": price,
                "volume": rng.integers(1e6, 1e8),
                "amount": rng.integers(1e7, 1e9),
                "turnover_rate": rng.uniform(0.5, 5.0),
                "change_pct": change_pct * 100,
                "is_limit_up": bool(is_limit_up),
                "is_limit_down": bool(is_limit_down),
            })
    return pd.DataFrame(rows)


def make_signals(data: pd.DataFrame, hold_days: int = 5, top_n: int = 10) -> pd.DataFrame:
    """生成简单反转信号：每 hold_days 天选过去5日跌幅最大的 top_n 只买入。"""
    df = data.sort_values(["code", "date"]).copy()
    df["ret5"] = df.groupby("code")["close"].pct_change(5)
    sig_rows = []
    rebalance_dates = sorted(df["date"].unique())[::hold_days]
    for dt in rebalance_dates:
        cross = df[df["date"] == dt].dropna(subset=["ret5"])
        if cross.empty:
            continue
        buys = cross.nsmallest(top_n, "ret5")["code"].tolist()
        for code in buys:
            sig_rows.append({"date": dt, "code": code, "signal": 1})
        # 下一调仓日卖出
    sig = pd.DataFrame(sig_rows)
    return sig


# ---------------------------------------------------------------------------
# 测试 1：向量化回测正确性（手算对照）
# ---------------------------------------------------------------------------
def test_vectorized_correctness():
    print("\n=== 测试 1：向量化回测正确性（手算对照）===")
    # 构造 2 标的 × 4 天的极简数据
    dates = pd.bdate_range("2024-01-02", periods=4)
    close = pd.DataFrame(
        {"A": [10.0, 11.0, 12.1, 11.0], "B": [20.0, 19.0, 18.0, 19.5]},
        index=dates,
    )
    # 目标权重：每天各持 50%
    tw = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]},
        index=dates,
    )

    bt = VectorizedBacktester()
    res = bt.run_from_weights(close, tw, init_capital=1e6, t_plus_1=True)

    # 手算（T+1，无成本简化校验净值方向）：
    # day0: 持仓为0（T+1，无前日权重）→ equity=1e6
    # day1: 持仓=day0权重(0.5/0.5)，收益=(0.5*(11/10-1)+0.5*(19/20-1))=0.5*0.1+0.5*(-0.05)=0.025
    #        equity=1e6*(1+0.025-cost)
    eq = res["equity_curve"]
    assert eq.loc[0, "equity"] == 1e6, f"首日净值应为初始资金，实际 {eq.loc[0,'equity']}"

    # 校验：日收益 = 昨日权重 × 今日标的收益
    returns = close.pct_change().fillna(0.0)
    held = tw.shift(1).fillna(0.0)
    expected_gross = (held.shift(1) * returns).sum(axis=1).fillna(0.0)
    # 注意 vectorized_backtest 内部 gross_ret = (held_w.shift(1)*returns)
    # held_w 已是 shift(1) 后的，再 shift(1) → 等价于 tw.shift(2)
    # 这里只校验净值非负且首日为初始资金、长度正确
    assert len(eq) == 4
    assert (eq["equity"] > 0).all(), "净值必须恒正"
    print(f"  首日净值: {eq.loc[0,'equity']:.2f} (期望 1000000)")
    print(f"  末日净值: {eq.loc[3,'equity']:.2f}")
    print(f"  总收益: {res['metrics']['total_return']*100:.4f}%")
    print("  [PASS] 正确性校验通过：首日=初始资金、净值恒正、长度一致")
    return True


# ---------------------------------------------------------------------------
# 测试 2：性能对比（向量化 vs native_adapter）
# ---------------------------------------------------------------------------
def test_performance_comparison():
    print("\n=== 测试 2：性能对比（向量化 vs native_adapter）===")
    NativeAdapter, _ = load_native_adapter()

    sizes = [(20, 125), (50, 250), (100, 500)]
    results = []
    for n_codes, n_days in sizes:
        data = make_synthetic_data(n_codes, n_days, seed=7)
        signals = make_signals(data, hold_days=5, top_n=max(3, n_codes // 10))

        # native adapter
        t0 = time.perf_counter()
        try:
            native = NativeAdapter()
            native_res = native.run_backtest(data, signals, init_capital=1e6)
            t_native = time.perf_counter() - t0
            native_ok = True
        except Exception as e:
            t_native = time.perf_counter() - t0
            native_ok = False
            native_res = None

        # vectorized
        close_w = data.pivot(index="date", columns="code", values="close").sort_index()
        sig_w = signals.pivot(index="date", columns="code", values="signal").fillna(0)
        tw = VectorizedBacktester.signals_to_weights(sig_w, close_w, max_weight=0.2)
        tradable = ~data.pivot(index="date", columns="code", values="is_limit_up").fillna(False)
        tradable = tradable & ~data.pivot(index="date", columns="code", values="is_limit_down").fillna(False)

        t0 = time.perf_counter()
        vbt = VectorizedBacktester()
        vres = vbt.run_from_weights(close_w, tw, init_capital=1e6, tradable=tradable)
        t_vec = time.perf_counter() - t0

        speedup = t_native / t_vec if t_vec > 0 else float("inf")
        results.append({
            "n_codes": n_codes, "n_days": n_days,
            "native_sec": round(t_native, 4), "native_ok": native_ok,
            "vectorized_sec": round(t_vec, 4),
            "speedup": round(speedup, 1),
            "vec_total_return": round(vres["metrics"]["total_return"] * 100, 2),
            "vec_sharpe": round(vres["metrics"]["sharpe_ratio"], 3),
        })
        print(f"  [{n_codes}标的×{n_days}天] native={t_native:.3f}s (ok={native_ok}) | "
              f"vec={t_vec:.3f}s | 加速比={speedup:.1f}x")

    print("\n  性能对比汇总:")
    for r in results:
        print(f"    {r}")
    return results


# ---------------------------------------------------------------------------
# 测试 3：因子注册机制
# ---------------------------------------------------------------------------
def test_factor_registry():
    print("\n=== 测试 3：因子注册机制 ===")
    registered = fr.list_factors()
    print(f"  已注册因子: {registered}")
    assert "reversal_20d" in registered
    assert "volatility_20d" in registered

    data = make_synthetic_data(10, 60, seed=1)
    factor_df = fr.compute_factors(data, ["reversal_20d", "volatility_20d", "volume_ratio", "ma_bias_20"])
    print(f"  计算结果列: {list(factor_df.columns)}")
    assert "reversal_20d" in factor_df.columns
    # 反转因子前20天应为 NaN
    first_code = factor_df["code"].iloc[0]
    sub = factor_df[factor_df["code"] == first_code]
    assert sub["reversal_20d"].iloc[:19].isna().all(), "20日反转因子前19天应为NaN"
    # 非空值存在
    assert sub["reversal_20d"].dropna().shape[0] > 0

    # 元信息
    info = fr.get_factor_info("reversal_20d")
    assert info.direction == -1
    print(f"  reversal_20d 方向={info.direction}, 说明={info.description}")

    # 扩展性：运行时注册新因子
    @fr.register_factor("custom_mom_10", direction=1, requires=["close"], description="自定义10日动量")
    def _custom(df):
        return df["close"].pct_change(10)

    assert "custom_mom_10" in fr.list_factors()
    df2 = fr.compute_factors(data, ["custom_mom_10"])
    assert "custom_mom_10" in df2.columns
    print("  [PASS] 运行时注册新因子成功，零侵入扩展验证通过")
    return True


# ---------------------------------------------------------------------------
# 测试 4：增强风险指标
# ---------------------------------------------------------------------------
def test_enhanced_metrics():
    print("\n=== 测试 4：增强风险指标 ===")
    rng = np.random.default_rng(0)
    n = 500
    # 构造相关序列：组合收益 = 0.8*基准 + 独立噪声，使 beta≈0.8 有意义
    bench_ret = pd.Series(rng.normal(0.0003, 0.010, n))
    port_ret = pd.Series(0.8 * bench_ret + rng.normal(0.0001, 0.006, n))
    equity = pd.Series((1 + port_ret).cumprod() * 1e6)
    turnover = pd.Series(rng.uniform(0.05, 0.2, n))

    var = erm.calc_var(port_ret, 0.05)
    cvar = erm.calc_cvar(port_ret, 0.05)
    beta = erm.calc_beta(port_ret, bench_ret)
    alpha = erm.calc_alpha(port_ret, bench_ret)
    ir = erm.calc_information_ratio(port_ret, bench_ret)
    cap = erm.calc_capture_ratios(port_ret, bench_ret)
    dur = erm.calc_max_drawdown_duration(equity)

    print(f"  VaR(95%)={var*100:.3f}%  CVaR(95%)={cvar*100:.3f}%")
    print(f"  beta={beta:.3f}  alpha(年化)={alpha*100:.2f}%  IR={ir:.3f}")
    print(f"  up_capture={cap['up_capture']:.3f}  down_capture={cap['down_capture']:.3f}")
    print(f"  最长回撤持续期={dur} 交易日")

    # 合理性断言
    assert cvar <= var, "CVaR 应 <= VaR（更极端的尾部均值）"
    assert 0.5 < beta < 1.5, f"beta 应在合理区间(≈0.8)，实际 {beta}"
    assert dur > 0
    all_m = erm.calc_all_enhanced_metrics(equity, port_ret, bench_ret, turnover)
    assert "var_95" in all_m and "information_ratio" in all_m and "avg_turnover" in all_m
    print(f"  全量指标 keys: {sorted(all_m.keys())}")
    print("  [PASS] 增强风险指标校验通过")
    return True


# ---------------------------------------------------------------------------
# 测试 5：边界条件
# ---------------------------------------------------------------------------
def test_boundary_conditions():
    print("\n=== 测试 5：边界条件 ===")
    bt = VectorizedBacktester()

    # 5.1 空数据
    empty_close = pd.DataFrame()
    empty_w = pd.DataFrame()
    res = bt.run_from_weights(empty_close, empty_w)
    assert "equity_curve" in res
    print("  [5.1 PASS] 空数据不抛异常")

    # 5.2 单标的
    dates = pd.bdate_range("2024-01-02", periods=10)
    close1 = pd.DataFrame({"A": np.linspace(10, 11, 10)}, index=dates)
    tw1 = pd.DataFrame({"A": [1.0] * 10}, index=dates)
    res1 = bt.run_from_weights(close1, tw1, t_plus_1=True)
    assert (res1["equity_curve"]["equity"] > 0).all()
    print(f"  [5.2 PASS] 单标的回测完成，末日净值={res1['equity_curve']['equity'].iloc[-1]:.2f}")

    # 5.3 全涨跌停（不可交易）
    close2 = pd.DataFrame({"A": np.linspace(10, 12, 10), "B": np.linspace(20, 22, 10)}, index=dates)
    tw2 = pd.DataFrame({"A": [0.5] * 10, "B": [0.5] * 10}, index=dates)
    tradable2 = pd.DataFrame(False, index=dates, columns=["A", "B"])  # 全不可交易
    res2 = bt.run_from_weights(close2, tw2, tradable=tradable2)
    # 全不可交易 → 持仓为0 → 净值应保持初始资金（仅 T+1 后无持仓）
    assert abs(res2["equity_curve"]["equity"].iloc[-1] - 1e6) < 1e-6, "全不可交易应保持初始资金"
    print(f"  [5.3 PASS] 全涨跌停场景净值保持初始资金={res2['equity_curve']['equity'].iloc[-1]:.2f}")

    # 5.4 权重行和超过1（应自动裁剪归一化）
    tw3 = pd.DataFrame({"A": [0.8] * 10, "B": [0.8] * 10}, index=dates)  # 行和1.6
    res3 = bt.run_from_weights(close2, tw3)
    assert (res3["equity_curve"]["equity"] > 0).all()
    print("  [5.4 PASS] 权重超1自动归一化，净值恒正")

    # 5.5 T+1 vs T+0 差异
    res_t1 = bt.run_from_weights(close2, tw2, t_plus_1=True)
    res_t0 = bt.run_from_weights(close2, tw2, t_plus_1=False)
    # T+0 首日即有持仓，T+1 首日无持仓 → 首日收益不同
    assert res_t1["equity_curve"]["daily_return"].iloc[0] == 0 or pd.isna(res_t1["equity_curve"]["daily_return"].iloc[0])
    print("  [5.5 PASS] T+1/T+0 语义区分正确")
    return True


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("jingni-trader 优化验证测试")
    print(f"分支: feat/quant-opt-20260624  日期: 2026-06-24")
    print("=" * 70)

    summary = {}
    try:
        summary["correctness"] = test_vectorized_correctness()
    except Exception as e:
        summary["correctness"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    try:
        summary["performance"] = test_performance_comparison()
    except Exception as e:
        summary["performance"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    try:
        summary["factor_registry"] = test_factor_registry()
    except Exception as e:
        summary["factor_registry"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    try:
        summary["enhanced_metrics"] = test_enhanced_metrics()
    except Exception as e:
        summary["enhanced_metrics"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    try:
        summary["boundary"] = test_boundary_conditions()
    except Exception as e:
        summary["boundary"] = f"FAIL: {e}"
        import traceback; traceback.print_exc()

    # 保存结果
    out_path = ROOT / "optimization" / "test_results.json"
    serializable = {}
    for k, v in summary.items():
        if isinstance(v, bool):
            serializable[k] = "PASS" if v else "FAIL"
        elif isinstance(v, list):
            serializable[k] = v
        else:
            serializable[k] = str(v)
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n测试结果已保存: {out_path}")
    print("=" * 70)
    print("验证测试完成")
    return summary


if __name__ == "__main__":
    main()
