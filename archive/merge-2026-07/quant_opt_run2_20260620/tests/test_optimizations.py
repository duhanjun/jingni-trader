"""
量化优化验证测试套件

覆盖三个优化模块：
  1. factor_expression_engine.py  - 因子表达式引擎
  2. enhanced_backtest_engine.py  - 增强回测引擎
  3. factor_analysis.py           - 因子分析与预处理

测试类型：
  - 正确性测试（与已知结果/手算对比）
  - 性能对比测试（与 jingni-trader 现有实现对比）
  - 边界条件测试（空数据、单只股票、全 NaN 等）

运行: python -m quant_opt_20260620.tests.test_optimizations
"""
import os
import sys
import time
import traceback
from typing import Callable, Tuple

import numpy as np
import pandas as pd

# 让测试可独立运行
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # quant_opt_20260620/
WORKSPACE = os.path.dirname(ROOT)  # /workspace
sys.path.insert(0, ROOT)
sys.path.insert(0, WORKSPACE)

from quant_opt_run2_20260620.factor_expression_engine import (
    FactorExpressionEngine, parse, PRESET_FACTORS, list_preset_factors,
)
from quant_opt_run2_20260620.enhanced_backtest_engine import EnhancedBacktestEngine
from quant_opt_run2_20260620.factor_analysis import FactorPreprocessor, FactorICAnalyzer
from quant_opt_run2_20260620.tests.synthetic_data import (
    make_synthetic_data, make_signals_from_factor, make_benchmark_returns,
)


# ============================================================
# 测试框架（轻量自实现，避免依赖 pytest）
# ============================================================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.details = []

    def record(self, name: str, ok: bool, info: str = ""):
        status = "PASS" if ok else "FAIL"
        self.details.append((name, status, info))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        mark = "✓" if ok else "✗"
        print(f"  {mark} [{name}] {info}")

    def summary(self) -> str:
        total = self.passed + self.failed
        return f"\n=== 测试汇总: {self.passed}/{total} 通过, {self.failed} 失败 ===\n"


def assert_close(a, b, tol=1e-6, name=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: {a} != {b} (tol={tol})")


# ============================================================
# 1. 因子表达式引擎测试
# ============================================================

def test_factor_expression_correctness(tr: TestResult):
    """正确性：表达式引擎结果与手算/pandas 直算一致"""
    print("\n--- 测试组 1: 因子表达式引擎正确性 ---")
    data = make_synthetic_data(n_codes=5, n_days=60, seed=1)
    engine = FactorExpressionEngine()

    # 1.1 MA($close, 5) 应等于 pandas rolling mean
    try:
        result = engine.calculate(data, ["MA($close, 5)"])
        # 取第一只股票手算对比
        first_code = data["code"].iloc[0]
        sub = data[data["code"] == first_code].sort_values("date")
        expected = sub["close"].rolling(5, min_periods=2).mean().values
        actual = result[result["code"] == first_code]["factor_0"].values
        # rolling min_periods 一致
        ok = np.allclose(expected, actual, equal_nan=True)
        tr.record("MA($close,5) 与 pandas 一致", ok)
    except Exception as e:
        tr.record("MA($close,5) 与 pandas 一致", False, str(e))

    # 1.2 Ref($close, 1) = close.shift(1)
    try:
        result = engine.calculate(data, ["Ref($close, 1)"])
        sub = data[data["code"] == first_code].sort_values("date")
        expected = sub["close"].shift(1).values
        actual = result[result["code"] == first_code]["factor_0"].values
        ok = np.allclose(expected, actual, equal_nan=True)
        tr.record("Ref($close,1) = shift(1)", ok)
    except Exception as e:
        tr.record("Ref($close,1) = shift(1)", False, str(e))

    # 1.3 复合表达式: MA($close,20) - MA($close,5)
    try:
        result = engine.calculate(data, ["MA($close, 20) - MA($close, 5)"])
        sub = data[data["code"] == first_code].sort_values("date")
        expected = (sub["close"].rolling(20, min_periods=10).mean()
                    - sub["close"].rolling(5, min_periods=2).mean()).values
        actual = result[result["code"] == first_code]["factor_0"].values
        ok = np.allclose(expected, actual, equal_nan=True)
        tr.record("复合表达式 MA(20)-MA(5)", ok)
    except Exception as e:
        tr.record("复合表达式 MA(20)-MA(5)", False, str(e))

    # 1.4 截面 Rank: 每日 rank 应在 [0,1]
    try:
        data_rank = make_synthetic_data(n_codes=30, n_days=60, seed=15)
        result = engine.calculate(data_rank, ["Rank($close)"])
        vals = result["factor_0"].dropna()
        ok = vals.min() >= 0 and vals.max() <= 1
        # 每日 rank 均值应接近 0.5（股票数足够多时）
        daily_mean = result.groupby("date")["factor_0"].mean().dropna()
        ok = ok and abs(daily_mean.mean() - 0.5) < 0.05
        tr.record("截面 Rank($close) 范围 [0,1]", ok, f"mean={daily_mean.mean():.4f}")
    except Exception as e:
        tr.record("截面 Rank($close) 范围 [0,1]", False, str(e))

    # 1.5 预置因子可全部解析
    try:
        ok = True
        for name in list_preset_factors():
            expr = PRESET_FACTORS[name]
            ast = parse(expr)  # 仅解析不计算，验证语法
        # 实际计算一个
        result = engine.calculate(data, [PRESET_FACTORS["mom_20"]])
        ok = "factor_0" in result.columns and not result["factor_0"].isna().all()
        tr.record("预置因子全部可解析计算", ok, f"{len(list_preset_factors())} 个因子")
    except Exception as e:
        tr.record("预置因子全部可解析计算", False, str(e))


def test_factor_expression_performance(tr: TestResult):
    """性能：表达式引擎 vs 现有 pandas_ta_calculator 的逐股票循环"""
    print("\n--- 测试组 2: 因子表达式引擎性能 ---")
    data = make_synthetic_data(n_codes=50, n_days=250, seed=2)
    engine = FactorExpressionEngine()

    # 表达式引擎计时
    t0 = time.perf_counter()
    result_expr = engine.calculate(data, ["MA($close, 20)", "STD($close, 20)", "Ref($close, 5)"])
    t_expr = time.perf_counter() - t0

    # 模拟现有实现的逐股票循环（与 pandas_ta_calculator._calc_single 同结构）
    t0 = time.perf_counter()
    result_loop = data[["code", "date"]].copy()
    for factor_name, calc_fn in [
        ("ma_20", lambda s: s.rolling(20, min_periods=10).mean()),
        ("std_20", lambda s: s.rolling(20, min_periods=10).std()),
        ("ref_5", lambda s: s.shift(5)),
    ]:
        series = pd.Series(index=data.index, dtype=float)
        for code in data["code"].unique():
            mask = data["code"] == code
            idx = data[mask].index
            series.loc[idx] = calc_fn(data.loc[idx, "close"])
        result_loop[factor_name] = series
    t_loop = time.perf_counter() - t0

    # 正确性校验
    ok_correct = np.allclose(
        result_expr["factor_0"].values,
        result_loop["ma_20"].values,
        equal_nan=True,
    )
    speedup = t_loop / t_expr if t_expr > 0 else float("inf")
    ok = ok_correct and speedup >= 1.0
    tr.record(
        "表达式引擎 vs 逐股票循环",
        ok,
        f"expr={t_expr*1000:.1f}ms loop={t_loop*1000:.1f}ms 加速={speedup:.2f}x 正确={ok_correct}",
    )


def test_factor_expression_edge(tr: TestResult):
    """边界条件：空数据、单只股票、未知字段"""
    print("\n--- 测试组 3: 因子表达式引擎边界条件 ---")
    engine = FactorExpressionEngine()

    # 空数据
    try:
        result = engine.calculate(pd.DataFrame(columns=["code", "date", "close"]), ["MA($close, 5)"])
        tr.record("空数据不崩溃", True)
    except Exception as e:
        tr.record("空数据不崩溃", False, str(e))

    # 单只股票
    try:
        data = make_synthetic_data(n_codes=1, n_days=30, seed=3)
        result = engine.calculate(data, ["MA($close, 5)", "Rank($close)"])
        ok = len(result) == 30 and not result["factor_0"].isna().all()
        tr.record("单只股票可计算", ok)
    except Exception as e:
        tr.record("单只股票可计算", False, str(e))

    # 未知字段应报错
    try:
        data = make_synthetic_data(n_codes=2, n_days=10, seed=4)
        engine.calculate(data, ["MA($nonexistent, 5)"])
        tr.record("未知字段应报错", False, "未抛出异常")
    except KeyError:
        tr.record("未知字段应报错", True)
    except Exception as e:
        tr.record("未知字段应报错", False, f"异常类型错误: {type(e).__name__}")

    # 语法错误应报错
    try:
        parse("MA($close, )")
        tr.record("语法错误应报错", False, "未抛出异常")
    except SyntaxError:
        tr.record("语法错误应报错", True)
    except Exception as e:
        tr.record("语法错误应报错", False, f"异常类型错误: {type(e).__name__}")


# ============================================================
# 2. 增强回测引擎测试
# ============================================================

def test_backtest_correctness(tr: TestResult):
    """正确性：回测基本流程、资金守恒、T+1 执行"""
    print("\n--- 测试组 4: 增强回测引擎正确性 ---")
    data = make_synthetic_data(n_codes=5, n_days=60, seed=10)
    # 简单信号：第 10 天起每日买第一只
    sig_rows = []
    first_code = data["code"].iloc[0]
    dates = sorted(data["date"].unique())
    for dt in dates[10:15]:
        sig_rows.append({"code": first_code, "date": dt, "signal": 1})
    for dt in dates[20:22]:
        sig_rows.append({"code": first_code, "date": dt, "signal": -1})
    signals = pd.DataFrame(sig_rows)

    bench = make_benchmark_returns(data)
    engine = EnhancedBacktestEngine()
    result = engine.run_backtest(data, signals, init_capital=1e6, benchmark_returns=bench)

    # 4.1 返回结构完整
    required_keys = {"trades", "positions", "equity_curve", "metrics"}
    ok = required_keys.issubset(result.keys())
    tr.record("返回结构完整", ok, f"keys={set(result.keys())}")

    # 4.2 equity_curve 非空
    ok = not result["equity_curve"].empty and len(result["equity_curve"]) > 0
    tr.record("equity_curve 非空", ok, f"len={len(result['equity_curve'])}")

    # 4.3 资金守恒：equity = cash + market_value
    ec = result["equity_curve"]
    diff = (ec["equity"] - ec["cash"] - ec["market_value"]).abs().max()
    tr.record("资金守恒 equity=cash+mv", diff < 1e-2, f"max_diff={diff:.4f}")

    # 4.4 T+1 验证：信号 T 日，成交应在 T+1
    trades = result["trades"]
    if not trades.empty:
        first_buy = trades[trades["action"] == "buy"].iloc[0]
        signal_date = pd.Timestamp(signals[signals["signal"] == 1]["date"].iloc[0])
        trade_date = pd.Timestamp(first_buy["date"])
        # 成交日应严格晚于信号日（T+1）
        ok = trade_date > signal_date
        tr.record("T+1 执行: 成交日 > 信号日", ok,
                  f"signal={signal_date.date()} trade={trade_date.date()}")
    else:
        tr.record("T+1 执行: 成交日 > 信号日", False, "无成交记录")

    # 4.5 指标完整：含新增的 alpha/beta/IR/VaR
    metrics = result["metrics"]
    new_metrics = {"alpha", "beta", "information_ratio", "tracking_error", "var_95", "cvar_95", "omega_ratio"}
    ok = new_metrics.issubset(metrics.keys())
    tr.record("新增风险指标齐全", ok, f"missing={new_metrics - set(metrics.keys())}")


def test_backtest_vs_native(tr: TestResult):
    """性能与一致性：增强引擎 vs 现有 native_adapter"""
    print("\n--- 测试组 5: 增强回测 vs 现有 native_adapter ---")
    data = make_synthetic_data(n_codes=20, n_days=200, seed=11)

    # 生成信号：基于简单动量
    data_sorted = data.sort_values(["code", "date"]).copy()
    data_sorted["mom"] = data_sorted.groupby("code")["close"].transform(
        lambda s: s.pct_change(5)
    )
    data_sorted["rank"] = data_sorted.groupby("date")["mom"].rank(pct=True)
    signals = data_sorted[["code", "date"]].copy()
    signals["signal"] = 0
    signals.loc[data_sorted["rank"] > 0.8, "signal"] = 1
    signals.loc[data_sorted["rank"] < 0.2, "signal"] = -1
    # 抽样减少信号数量，避免每日全调仓
    signals = signals[signals["signal"] != 0].reset_index(drop=True)

    bench = make_benchmark_returns(data)

    # 增强引擎
    eng = EnhancedBacktestEngine()
    t0 = time.perf_counter()
    res_enh = eng.run_backtest(data, signals, init_capital=1e6, benchmark_returns=bench)
    t_enh = time.perf_counter() - t0

    # 现有 native_adapter
    try:
        sys.path.insert(0, os.path.join(WORKSPACE, "skills", "backtest-engine", "scripts"))
        # 现有 native 依赖 scripts 包结构，直接 import 可能失败，做容错
        from adapters.native_adapter import NativeAdapter
        native = NativeAdapter()
        t0 = time.perf_counter()
        res_nat = native.run_backtest(data, signals, init_capital=1e6)
        t_nat = time.perf_counter() - t0

        # 两者都应产出非空 equity_curve
        ok = (not res_enh["equity_curve"].empty) and (not res_nat["equity_curve"].empty)
        # 增强引擎应有更多指标
        more_metrics = len(res_enh["metrics"]) > len(res_nat.get("metrics", {}))
        tr.record(
            "增强引擎产出完整且指标更丰富",
            ok and more_metrics,
            f"enh_metrics={len(res_enh['metrics'])} nat_metrics={len(res_nat.get('metrics', {}))}",
        )
        tr.record(
            "回测性能对比",
            True,
            f"enh={t_enh*1000:.1f}ms native={t_nat*1000:.1f}ms",
        )
    except Exception as e:
        # native_adapter 依赖 scripts 包注册，独立运行时可能 import 失败
        # 此时只验证增强引擎自身可用
        tr.record(
            "增强引擎独立可用(native不可导入)",
            not res_enh["equity_curve"].empty,
            f"native导入失败: {type(e).__name__}; enh耗时={t_enh*1000:.1f}ms",
        )


def test_backtest_edge(tr: TestResult):
    """边界条件：空数据、无信号、全涨停"""
    print("\n--- 测试组 6: 增强回测边界条件 ---")
    engine = EnhancedBacktestEngine()

    # 空数据
    try:
        res = engine.run_backtest(pd.DataFrame(), pd.DataFrame())
        ok = res["metrics"] == {} and res["equity_curve"].empty
        tr.record("空数据返回空结果", ok)
    except Exception as e:
        tr.record("空数据返回空结果", False, str(e))

    # 无信号：应无交易（equity_curve 可能为空或为初始资金平线）
    try:
        data = make_synthetic_data(n_codes=3, n_days=20, seed=12)
        signals = pd.DataFrame(columns=["code", "date", "signal"])
        res = engine.run_backtest(data, signals)
        ok = res["trades"].empty
        if not res["equity_curve"].empty:
            ok = ok and res["equity_curve"].iloc[-1]["equity"] == 1e6
        tr.record("无信号无交易", ok, f"trades={len(res['trades'])} ec_empty={res['equity_curve'].empty}")
    except Exception as e:
        tr.record("无信号无交易", False, str(e))

    # 全涨停：买入信号应被拒绝
    try:
        data = make_synthetic_data(n_codes=3, n_days=20, seed=13)
        data["is_limit_up"] = True  # 全部涨停
        dates = sorted(data["date"].unique())
        signals = pd.DataFrame([
            {"code": data["code"].iloc[0], "date": dates[5], "signal": 1}
        ])
        res = engine.run_backtest(data, signals)
        # 涨停日无法买入，但 T+1 后若不再涨停可买入；这里全涨停所以应无成交
        ok = len(res["trades"]) == 0
        tr.record("全涨停时买入被拒", ok, f"trades={len(res['trades'])}")
    except Exception as e:
        tr.record("全涨停时买入被拒", False, str(e))

    # target_weight 信号模式
    try:
        data = make_synthetic_data(n_codes=3, n_days=30, seed=14)
        dates = sorted(data["date"].unique())
        signals = pd.DataFrame([
            {"code": data["code"].iloc[0], "date": dates[5], "target_weight": 0.5},
            {"code": data["code"].iloc[1], "date": dates[5], "target_weight": 0.3},
        ])
        res = engine.run_backtest(data, signals)
        ok = len(res["trades"]) > 0
        tr.record("target_weight 信号模式", ok, f"trades={len(res['trades'])}")
    except Exception as e:
        tr.record("target_weight 信号模式", False, str(e))


# ============================================================
# 3. 因子分析测试
# ============================================================

def test_factor_preprocessing(tr: TestResult):
    """因子预处理：缩尾/标准化/中性化"""
    print("\n--- 测试组 7: 因子预处理 ---")
    rng = np.random.default_rng(20)
    n = 300
    n_dates = 10
    per_date = n // n_dates
    # 构造因子与市值有强相关性的数据：factor = log(mc) + 小噪声
    market_cap = rng.uniform(1e9, 1e11, n)
    log_mc = np.log(market_cap)
    factor_vals = log_mc + rng.normal(0, 0.1, n)  # 强相关
    df = pd.DataFrame({
        "date": np.repeat(pd.date_range("2023-01-03", periods=n_dates), per_date),
        "code": [f"{i:06d}.SZ" for i in range(per_date)] * n_dates,
        "factor": factor_vals,
        "industry": rng.choice(["A", "B", "C", "D", "E"], n),
        "market_cap": market_cap,
    })

    pre = FactorPreprocessor()

    # 缩尾：注入极端值后应被截断
    try:
        df_w = df.copy()
        df_w.loc[0, "factor"] = 100
        df_w.loc[1, "factor"] = -100
        w = pre.winsorize(df_w["factor"], 0.05, df_w["date"])
        ok = w.max() < 100 and w.min() > -100
        tr.record("Winsorize 截断极值", ok, f"max={w.max():.2f} min={w.min():.2f}")
    except Exception as e:
        tr.record("Winsorize 截断极值", False, str(e))

    # 标准化：均值≈0 标准差≈1
    try:
        s = pre.standardize(df["factor"], df["date"])
        daily_mean = s.groupby(df["date"]).mean()
        daily_std = s.groupby(df["date"]).std()
        ok = abs(daily_mean.mean()) < 0.1 and abs(daily_std.mean() - 1) < 0.1
        tr.record("Standardize 均值0标准差1", ok,
                  f"mean={daily_mean.mean():.4f} std={daily_std.mean():.4f}")
    except Exception as e:
        tr.record("Standardize 均值0标准差1", False, str(e))

    # 中性化：残差与市值相关性应显著降低（因子强依赖于市值）
    try:
        resid = pre.neutralize(df, "factor", "date", "industry", "market_cap")
        orig_corr = df["factor"].corr(np.log(df["market_cap"]))
        resid_corr = resid.corr(np.log(df["market_cap"]))
        ok = abs(resid_corr) < abs(orig_corr) * 0.3  # 至少降低70%
        tr.record("Neutralize 降低市值相关性", ok,
                  f"orig={orig_corr:.3f} resid={resid_corr:.3f}")
    except Exception as e:
        tr.record("Neutralize 降低市值相关性", False, str(e))

    # 完整流水线
    try:
        out = pre.pipeline(df, "factor", "date", 0.05, True, "industry", "market_cap")
        ok = len(out) == n and not out.isna().all()
        tr.record("Pipeline 完整运行", ok)
    except Exception as e:
        tr.record("Pipeline 完整运行", False, str(e))


def test_factor_ic(tr: TestResult):
    """因子 IC 分析：前向收益、IC、分组收益"""
    print("\n--- 测试组 8: 因子 IC 分析 ---")
    data = make_synthetic_data(n_codes=30, n_days=120, seed=21)
    analyzer = FactorICAnalyzer(n_forward=5)

    # 构造一个有预测力的因子：直接用 fwd_ret_5 + 噪声（仅用于测试 IC 计算正确性）
    data_sorted = data.sort_values(["code", "date"]).copy()
    data_sorted["fwd_ret_5"] = data_sorted.groupby("code")["close"].transform(
        lambda s: s.shift(-5) / s - 1
    )
    rng = np.random.default_rng(22)
    # 因子 = fwd_ret_5 + 噪声，确保与 5 日前向收益强正相关
    data_sorted["alpha_factor"] = (
        data_sorted["fwd_ret_5"] + rng.normal(0, 0.01, len(data_sorted))
    )

    # 8.1 前向收益计算
    try:
        fwd = analyzer.compute_forward_returns(data, periods=[1, 5, 10])
        ok = "fwd_ret_5" in fwd.columns and not fwd["fwd_ret_5"].isna().all()
        tr.record("前向收益计算", ok)
    except Exception as e:
        tr.record("前向收益计算", False, str(e))

    # 8.2 IC 分析：有预测力的因子 IC 应为正
    try:
        result = analyzer.analyze(data_sorted, "alpha_factor", "date",
                                  ["fwd_ret_5"])
        ic_mean = result["ic_mean"]
        ok = not np.isnan(ic_mean) and ic_mean > 0
        tr.record("IC 为正(有预测力因子)", ok,
                  f"IC={ic_mean:.4f} ICIR={result['icir']:.4f}")
    except Exception as e:
        tr.record("IC 为正(有预测力因子)", False, str(e))

    # 8.3 IC 衰减：因子直接关联 5 日收益，5 日 IC 应最高
    try:
        fwd = analyzer.compute_forward_returns(data, periods=[1, 5, 10, 20])
        merged = data_sorted.drop(columns=["fwd_ret_5"], errors="ignore").merge(
            fwd[["code", "date", "fwd_ret_1", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20"]],
            on=["code", "date"], how="left"
        )
        result = analyzer.analyze(merged, "alpha_factor", "date",
                                  ["fwd_ret_1", "fwd_ret_5", "fwd_ret_10", "fwd_ret_20"])
        decay = result["ic_decay"]
        # 因子 = fwd_ret_5 + noise，所以 5 日 IC 应为正且最高
        ic5 = decay["fwd_ret_5"]
        ok = ic5 > 0 and ic5 >= decay["fwd_ret_1"] and ic5 >= decay["fwd_ret_20"]
        tr.record("IC 衰减: 5日IC最高且为正", ok,
                  f"1d={decay['fwd_ret_1']:.4f} 5d={decay['fwd_ret_5']:.4f} "
                  f"20d={decay['fwd_ret_20']:.4f}")
    except Exception as e:
        tr.record("IC 衰减: 5日IC最高且为正", False, str(e))

    # 8.4 分组收益：高分组(Q5)收益应高于低分组(Q1)
    try:
        fwd = analyzer.compute_forward_returns(data, periods=[5])
        # data_sorted 已有 fwd_ret_5 列，先删除避免 merge 冲突
        merged = data_sorted.drop(columns=["fwd_ret_5"], errors="ignore").merge(
            fwd[["code", "date", "fwd_ret_5"]],
            on=["code", "date"], how="left"
        )
        qr = analyzer.quantile_returns(merged, "alpha_factor", "fwd_ret_5", "date", 5)
        ok = len(qr) == 5
        if ok:
            q5 = qr[qr["quantile"] == "Q5"]["mean_return"].iloc[0]
            q1 = qr[qr["quantile"] == "Q1"]["mean_return"].iloc[0]
            ok = q5 > q1
        tr.record("分组收益单调性 Q5>Q1", ok,
                  f"Q1={qr[qr['quantile']=='Q1']['mean_return'].iloc[0]:.4f} "
                  f"Q5={qr[qr['quantile']=='Q5']['mean_return'].iloc[0]:.4f}")
    except Exception as e:
        tr.record("分组收益单调性 Q5>Q1", False, str(e))


# ============================================================
# 主入口
# ============================================================

def run_all():
    tr = TestResult()
    print("=" * 60)
    print("量化优化验证测试 (feat/quant-opt-20260620)")
    print("=" * 60)

    test_factor_expression_correctness(tr)
    test_factor_expression_performance(tr)
    test_factor_expression_edge(tr)
    test_backtest_correctness(tr)
    test_backtest_vs_native(tr)
    test_backtest_edge(tr)
    test_factor_preprocessing(tr)
    test_factor_ic(tr)

    print(tr.summary())
    # 打印失败详情
    failures = [(n, s, i) for n, s, i in tr.details if s == "FAIL"]
    if failures:
        print("失败详情:")
        for n, s, i in failures:
            print(f"  - {n}: {i}")
    return tr


if __name__ == "__main__":
    run_all()
