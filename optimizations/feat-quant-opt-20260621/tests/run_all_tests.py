"""
优化验证主测试脚本

测试内容:
1. 向量化回测引擎 vs 原生回测引擎: 正确性 + 性能对比
2. 因子表达式引擎: 正确性 + 与硬编码因子对比
3. 因子分层分析: 完整流程验证
4. 边界条件测试
"""
import os
import sys
import time
import json
import traceback
from pathlib import Path

# 设置路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # /workspace
OPT_DIR = SCRIPT_DIR.parent  # /workspace/optimizations/feat-quant-opt-20260621

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

# 导入测试数据生成器
from tests.synthetic_data import (
    generate_synthetic_a_share_data,
    generate_random_signals,
    generate_momentum_signals,
)

# 导入优化模块
from vectorized_backtest.engine import VectorizedBacktestEngine
from factor_expression.engine import FactorExpressionEngine
from factor_analysis.engine import FactorLayeredAnalysis

# 导入原版回测引擎 (用于对比)
from original_impl.backtest_engine_test_stub import NativeAdapterStub


REPORT_DIR = OPT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def test_vectorized_backtest_correctness(data: pd.DataFrame, signals: pd.DataFrame):
    """测试1: 向量化回测引擎正确性"""
    section("测试1: 向量化回测引擎 - 正确性验证")

    # 基准数据
    bench_data = data[data["code"] == "000300.SH"][["date", "close"]].copy()
    stock_data = data[data["code"] != "000300.SH"].copy()

    # 新引擎
    new_engine = VectorizedBacktestEngine(
        init_capital=1_000_000,
        commission_rate=0.00025,
        stamp_tax_rate=0.001,
        slippage=0.001,
        t_plus_1=True,
        price_limit=True,
        max_volume_pct=0.10,
    )
    new_result = new_engine.run_backtest(
        data=stock_data,
        signals=signals,
        benchmark_data=bench_data,
    )

    # 原引擎 (通过 stub 调用)
    old_engine = NativeAdapterStub()
    old_result = old_engine.run_backtest(
        data=stock_data,
        signals=signals,
        init_capital=1_000_000,
        commission_rate=0.00025,
        stamp_tax_rate=0.001,
        t_plus_1=True,
        price_limit=True,
        slippage=0.001,
    )

    # 对比关键指标
    new_metrics = new_result["metrics"]
    old_metrics = old_result["metrics"]

    print(f"\n[新引擎] 指标:")
    for k, v in new_metrics.items():
        print(f"  {k}: {v}")

    print(f"\n[原引擎] 指标:")
    for k, v in old_metrics.items():
        print(f"  {k}: {v}")

    # 正确性检查
    checks = []

    # 1) 净值曲线非空
    checks.append(("equity_curve_non_empty", len(new_result["equity_curve"]) > 0))

    # 2) 初始资金接近
    if len(new_result["equity_curve"]) > 0:
        first_equity = new_result["equity_curve"]["equity"].iloc[0]
        checks.append(("initial_equity_close_to_1m", abs(first_equity - 1_000_000) < 50_000))

    # 3) 交易记录非空
    checks.append(("trades_non_empty", len(new_result["trades"]) > 0))

    # 4) T+1 验证: 检查是否有同日买卖
    if not new_result["trades"].empty:
        trades = new_result["trades"]
        # 找出每只股票的买入和卖出
        buys = trades[trades["action"] == "buy"][["date", "code"]].rename(columns={"date": "buy_date"})
        sells = trades[trades["action"] == "sell"][["date", "code"]].rename(columns={"date": "sell_date"})
        if not buys.empty and not sells.empty:
            merged = pd.merge(buys, sells, on="code", how="inner")
            # 检查是否有 sell_date <= buy_date (违反 T+1)
            violations = (merged["sell_date"] <= merged["buy_date"]).sum()
            checks.append(("t_plus_1_no_violations", violations == 0))
            print(f"\n  T+1 检查: 同日买卖违规次数 = {violations}")

    # 5) 基准对比列存在
    checks.append(("benchmark_column_exists", "benchmark" in new_result["equity_curve"].columns))

    # 6) 成交量限制验证
    if not new_result["trades"].empty and "volume" in stock_data.columns:
        buy_trades = new_result["trades"][new_result["trades"]["action"] == "buy"]
        if not buy_trades.empty:
            # 检查每笔买入是否超过当日成交量的 10%
            violations = 0
            for _, trade in buy_trades.iterrows():
                day_vol = stock_data[
                    (stock_data["date"] == trade["date"]) &
                    (stock_data["code"] == trade["code"])
                ]["volume"]
                if len(day_vol) > 0 and day_vol.iloc[0] > 0:
                    if trade["shares"] > day_vol.iloc[0] * 0.10:
                        violations += 1
            checks.append(("volume_limit_respected", violations == 0))
            print(f"  成交量限制检查: 违规次数 = {violations}")

    # 7) 持仓明细记录
    checks.append(("positions_recorded", len(new_result["positions"]) > 0))

    print(f"\n正确性检查结果:")
    passed = 0
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1

    return {
        "new_metrics": new_metrics,
        "old_metrics": old_metrics,
        "checks": dict(checks),
        "passed": passed,
        "total": len(checks),
        "new_trades_count": len(new_result["trades"]),
        "old_trades_count": len(old_result["trades"]),
    }


def test_vectorized_backtest_performance(data: pd.DataFrame, signals: pd.DataFrame):
    """测试2: 性能对比"""
    section("测试2: 向量化回测引擎 - 性能对比")

    bench_data = data[data["code"] == "000300.SH"][["date", "close"]].copy()
    stock_data = data[data["code"] != "000300.SH"].copy()

    # 不同数据规模测试
    results = []
    for n_stocks in [10, 30, 50]:
        # 取前 n_stocks 只
        codes = stock_data["code"].unique()[:n_stocks]
        sub_data = stock_data[stock_data["code"].isin(codes)]
        sub_signals = signals[signals["code"].isin(codes)]

        # 新引擎
        new_engine = VectorizedBacktestEngine()
        t0 = time.perf_counter()
        new_result = new_engine.run_backtest(sub_data, sub_signals, bench_data)
        new_time = time.perf_counter() - t0

        # 原引擎
        old_engine = NativeAdapterStub()
        t0 = time.perf_counter()
        old_result = old_engine.run_backtest(
            data=sub_data, signals=sub_signals,
            init_capital=1e6, commission_rate=0.00025,
            stamp_tax_rate=0.001, t_plus_1=True,
            price_limit=True, slippage=0.001,
        )
        old_time = time.perf_counter() - t0

        speedup = old_time / new_time if new_time > 0 else float("inf")
        results.append({
            "n_stocks": n_stocks,
            "n_days": sub_data["date"].nunique(),
            "n_signals": len(sub_signals),
            "new_engine_sec": round(new_time, 4),
            "old_engine_sec": round(old_time, 4),
            "speedup": round(speedup, 2),
        })
        print(f"\n  规模 {n_stocks} 只股票:")
        print(f"    新引擎: {new_time:.4f}s | 原引擎: {old_time:.4f}s | 加速比: {speedup:.2f}x")

    return results


def test_factor_expression(data: pd.DataFrame):
    """测试3: 因子表达式引擎"""
    section("测试3: 因子表达式引擎")

    stock_data = data[data["code"] != "000300.SH"].copy()

    engine = FactorExpressionEngine()

    # 列出可用函数
    funcs = engine.list_available_functions()
    fields = engine.list_available_fields()
    print(f"\n  可用函数 ({len(funcs)}): {list(funcs.keys())}")
    print(f"  可用字段 ({len(fields)}): {fields}")

    # 定义测试因子
    factor_defs = {
        "momentum_20d": "$close / Ref($close, 20) - 1",
        "reversal_5d": "-1 * ($close / Ref($close, 5) - 1)",
        "ma5": "Mean($close, 5)",
        "ma20": "Mean($close, 20)",
        "ma_diff": "Mean($close, 5) / Mean($close, 20) - 1",
        "vol_20d": "Std($close / Ref($close,1) - 1, 20)",
        "turnover_20d": "Mean($turnover_rate, 20)",
        "volume_ratio": "$volume / Mean($volume, 20)",
        "rsi_proxy": "CSRank(Mean($close, 5) - Mean($close, 20))",
        "price_zscore": "CSZScore($close)",
        "high_low_range": "($high - $low) / $close",
        "vwap_deviation": "($close - Mean($close, 5)) / Std($close, 5)",
    }

    # 表达式校验
    print(f"\n  表达式校验:")
    for name, expr in factor_defs.items():
        validation = engine.validate_expression(expr)
        status = "OK" if validation["valid"] else "FAIL"
        deps = validation.get("dependencies", [])
        print(f"    [{status}] {name}: {expr}  deps={deps}")

    # 计算因子
    t0 = time.perf_counter()
    factors_df = engine.compute(stock_data, factor_defs)
    elapsed = time.perf_counter() - t0

    print(f"\n  因子计算完成，耗时: {elapsed:.4f}s")
    print(f"  输出形状: {factors_df.shape}")
    print(f"\n  因子统计:")
    for col in factors_df.columns:
        if col in ("code", "date"):
            continue
        s = factors_df[col]
        print(f"    {col}: mean={s.mean():.6f}, std={s.std():.6f}, "
              f"nan={s.isna().sum()}, valid={s.notna().sum()}")

    # 与原版硬编码因子对比 (momentum_20d)
    print(f"\n  与原版硬编码因子对比 (ret_20d):")
    df_sorted = stock_data.sort_values(["code", "date"]).copy()
    df_sorted["ret_20d_original"] = df_sorted.groupby("code")["close"].pct_change(20)
    df_sorted["momentum_20d_expr"] = factors_df["momentum_20d"].values

    # 计算相关性
    valid = df_sorted.dropna(subset=["ret_20d_original", "momentum_20d_expr"])
    if len(valid) > 0:
        corr = valid["ret_20d_original"].corr(valid["momentum_20d_expr"])
        max_diff = (valid["ret_20d_original"] - valid["momentum_20d_expr"]).abs().max()
        print(f"    相关性: {corr:.6f}")
        print(f"    最大差异: {max_diff:.2e}")
        correctness_ok = corr > 0.9999 and max_diff < 1e-6
        print(f"    正确性: {'PASS' if correctness_ok else 'FAIL'}")
    else:
        correctness_ok = False
        print(f"    无有效数据对比")

    return {
        "n_factors": len(factor_defs),
        "elapsed_sec": round(elapsed, 4),
        "output_shape": list(factors_df.shape),
        "factors": list(factor_defs.keys()),
        "vs_original_corr": float(corr) if len(valid) > 0 else None,
        "correctness_ok": correctness_ok,
    }


def test_factor_layered_analysis(data: pd.DataFrame):
    """测试4: 因子分层分析"""
    section("测试4: 因子分层分析")

    stock_data = data[data["code"] != "000300.SH"].copy()

    # 先用表达式引擎生成一个因子
    expr_engine = FactorExpressionEngine()
    factor_df = expr_engine.compute(stock_data, {
        "momentum_20d": "$close / Ref($close, 20) - 1",
    })

    # 分层分析
    analyzer = FactorLayeredAnalysis(
        n_quantiles=5,
        forward_periods=(1, 5, 10, 20),
    )

    t0 = time.perf_counter()
    result = analyzer.analyze(
        factor_df=factor_df,
        price_df=stock_data[["code", "date", "close"]],
        factor_col="momentum_20d",
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  分析完成，耗时: {elapsed:.4f}s")

    # IC 分析
    print(f"\n  IC 分析 (含衰减):")
    ic = result["ic_analysis"]
    for period, vals in ic.items():
        if period == "decay":
            continue
        print(f"    {period}: IC_mean={vals.get('ic_mean')}, "
              f"IC_IR={vals.get('ic_ir')}, t={vals.get('ic_t_stat')}, "
              f"positive_ratio={vals.get('ic_positive_ratio')}")
    if "decay" in ic:
        print(f"    衰减: periods={ic['decay']['periods']}, "
              f"ic_means={ic['decay']['ic_means']}")

    # 分层收益
    print(f"\n  分层收益:")
    qr = result["quantile_returns"]
    if not qr.empty:
        for period in qr["period"].unique():
            sub = qr[qr["period"] == period].sort_values("quantile")
            print(f"    {period}:")
            for _, row in sub.iterrows():
                print(f"      Q{int(row['quantile'])}: ret={row['mean_return']:.6f}, "
                      f"sharpe={row['sharpe_like']}")

    # 多空组合
    print(f"\n  多空组合:")
    ls = result["long_short_returns"]
    if not ls.empty:
        for _, row in ls.iterrows():
            print(f"    {row['period']}: L/S mean={row['long_short_mean']:.6f}, "
                  f"sharpe={row['long_short_sharpe']}, win_rate={row['win_rate']}")

    # 单调性
    print(f"\n  单调性检验:")
    mono = result["monotonicity"]
    for period, vals in mono.items():
        print(f"    {period}: rho={vals['spearman_rho']}, "
              f"p={vals['p_value']}, monotonic={vals['is_monotonic']}, "
              f"direction={vals['direction']}")

    # 换手率
    print(f"\n  分层换手率:")
    turn = result["turnover"]
    for q, vals in turn.items():
        print(f"    {q}: avg={vals['avg_turnover']}, max={vals['max_turnover']}")

    # 覆盖率
    print(f"\n  覆盖率:")
    cov = result["coverage"]
    for k, v in cov.items():
        print(f"    {k}: {v}")

    # 综合评分
    print(f"\n  综合评分:")
    summary = result["summary"]
    print(f"    总分: {summary.get('total_score')}/100, 评级: {summary.get('grade')}")
    print(f"    分项: {summary.get('scores')}")
    print(f"    解读: {summary.get('interpretation')}")

    return {
        "elapsed_sec": round(elapsed, 4),
        "ic_analysis": ic,
        "monotonicity": mono,
        "summary": summary,
        "quantile_returns_count": len(qr),
        "long_short_count": len(ls),
    }


def test_edge_cases():
    """测试5: 边界条件"""
    section("测试5: 边界条件测试")

    results = []

    # 5.1 空数据
    print("\n  [5.1] 空数据测试")
    try:
        engine = VectorizedBacktestEngine()
        result = engine.run_backtest(pd.DataFrame(), pd.DataFrame())
        ok = result["equity_curve"].empty
        results.append(("empty_data", ok))
        print(f"    {'PASS' if ok else 'FAIL'}: 空数据返回空结果")
    except Exception as e:
        results.append(("empty_data", False))
        print(f"    FAIL: 异常 {e}")

    # 5.2 单只股票
    print("\n  [5.2] 单只股票测试")
    try:
        data = generate_synthetic_a_share_data(n_stocks=1, include_benchmark=True)
        stock_data = data[data["code"] != "000300.SH"]
        signals = pd.DataFrame({
            "date": [stock_data["date"].iloc[0]],
            "code": [stock_data["code"].iloc[0]],
            "signal": [1],
        })
        engine = VectorizedBacktestEngine()
        result = engine.run_backtest(stock_data, signals)
        ok = not result["equity_curve"].empty
        results.append(("single_stock", ok))
        print(f"    {'PASS' if ok else 'FAIL'}: 单只股票回测正常")
    except Exception as e:
        results.append(("single_stock", False))
        print(f"    FAIL: 异常 {e}")

    # 5.3 涨跌停无法成交
    print("\n  [5.3] 涨跌停限制测试")
    try:
        # 构造一只股票第一天涨停
        dates = pd.bdate_range("2024-01-01", periods=5)
        data = pd.DataFrame({
            "date": dates,
            "code": "600000.SH",
            "open": [10, 11, 11, 11, 11],
            "high": [11, 11, 11, 11, 11],
            "low": [10, 10, 10, 10, 10],
            "close": [10, 11, 11, 11, 11],
            "volume": [1e6, 1e6, 1e6, 1e6, 1e6],
            "is_st": [False] * 5,
            "is_limit_up": [False, True, False, False, False],
            "is_limit_down": [False] * 5,
        })
        signals = pd.DataFrame({
            "date": [dates[1]],
            "code": ["600000.SH"],
            "signal": [1],
        })
        engine = VectorizedBacktestEngine(price_limit=True)
        result = engine.run_backtest(data, signals)
        # 涨停日不应有买入
        buy_on_limit = (
            not result["trades"].empty and
            (result["trades"]["date"] == dates[1]).any()
        )
        ok = not buy_on_limit
        results.append(("price_limit_respected", ok))
        print(f"    {'PASS' if ok else 'FAIL'}: 涨停日不买入")
    except Exception as e:
        results.append(("price_limit_respected", False))
        print(f"    FAIL: 异常 {e}")

    # 5.4 T+1 验证
    print("\n  [5.4] T+1 严格验证")
    try:
        dates = pd.bdate_range("2024-01-01", periods=3)
        data = pd.DataFrame({
            "date": dates,
            "code": "600000.SH",
            "open": [10, 10, 10],
            "high": [11, 11, 11],
            "low": [9, 9, 9],
            "close": [10, 10, 10],
            "volume": [1e6, 1e6, 1e6],
            "is_st": [False] * 3,
            "is_limit_up": [False] * 3,
            "is_limit_down": [False] * 3,
        })
        # 第0天买入，第0天卖出 (应被 T+1 拒绝)
        signals = pd.DataFrame({
            "date": [dates[0], dates[0]],
            "code": ["600000.SH", "600000.SH"],
            "signal": [1, -1],
        })
        engine = VectorizedBacktestEngine(t_plus_1=True)
        result = engine.run_backtest(data, signals)
        # 应该只有买入，没有卖出
        trades_df = result["trades"]
        if not trades_df.empty and "action" in trades_df.columns:
            sells = trades_df[trades_df["action"] == "sell"]
            ok = len(sells) == 0
        else:
            ok = True  # 无交易也视为通过 (可能买入也失败)
        results.append(("t_plus_1_strict", ok))
        print(f"    {'PASS' if ok else 'FAIL'}: T+1 阻止当日卖出 (sells={len(sells) if not trades_df.empty else 0})")
    except Exception as e:
        results.append(("t_plus_1_strict", False))
        print(f"    FAIL: 异常 {e}")

    # 5.5 表达式非法语法
    print("\n  [5.5] 表达式非法语法测试")
    try:
        engine = FactorExpressionEngine()
        # 包含 import 的恶意表达式
        validation = engine.validate_expression("__import__('os').system('ls')")
        ok = not validation["valid"]
        results.append(("malicious_expression_blocked", ok))
        print(f"    {'PASS' if ok else 'FAIL'}: 恶意表达式被拒绝")
    except Exception as e:
        # 异常也是预期行为
        results.append(("malicious_expression_blocked", True))
        print(f"    PASS: 恶意表达式抛异常: {e}")

    # 5.6 表达式未知字段
    print("\n  [5.6] 表达式未知字段测试")
    try:
        data = generate_synthetic_a_share_data(n_stocks=5, include_benchmark=False)
        engine = FactorExpressionEngine()
        result = engine.compute(data, {"bad_factor": "$nonexistent_field"})
        # 应该返回 NaN 列
        ok = result["bad_factor"].isna().all()
        results.append(("unknown_field_handles_gracefully", ok))
        print(f"    {'PASS' if ok else 'FAIL'}: 未知字段返回 NaN")
    except Exception as e:
        results.append(("unknown_field_handles_gracefully", False))
        print(f"    FAIL: 异常 {e}")

    passed = sum(1 for _, ok in results if ok)
    return {
        "checks": dict(results),
        "passed": passed,
        "total": len(results),
    }


def main():
    """主测试入口"""
    section("jingni-trader 优化验证测试套件")
    print(f"分支: feat/quant-opt-20260621")
    print(f"日期: 2026-06-21")
    print(f"测试数据: 合成 A 股日线 (50 只股票, 2022-2024)")

    # 生成测试数据
    print("\n生成测试数据...")
    t0 = time.perf_counter()
    data = generate_synthetic_a_share_data(n_stocks=50, start_date="2022-01-01", end_date="2024-12-31")
    print(f"  数据生成完成: {len(data)} 行, {data['code'].nunique()} 个代码, 耗时 {time.perf_counter()-t0:.2f}s")

    # 生成信号
    random_signals = generate_random_signals(data, n_signals_per_day=5, seed=100)
    momentum_signals = generate_momentum_signals(data, lookback=20, hold_days=5, top_n=5)
    print(f"  随机信号: {len(random_signals)} 条")
    print(f"  动量信号: {len(momentum_signals)} 条")

    all_results = {}

    # 测试1: 正确性
    try:
        all_results["correctness"] = test_vectorized_backtest_correctness(data, momentum_signals)
    except Exception as e:
        print(f"测试1异常: {e}")
        traceback.print_exc()
        all_results["correctness"] = {"error": str(e)}

    # 测试2: 性能
    try:
        all_results["performance"] = test_vectorized_backtest_performance(data, random_signals)
    except Exception as e:
        print(f"测试2异常: {e}")
        traceback.print_exc()
        all_results["performance"] = {"error": str(e)}

    # 测试3: 因子表达式
    try:
        all_results["factor_expression"] = test_factor_expression(data)
    except Exception as e:
        print(f"测试3异常: {e}")
        traceback.print_exc()
        all_results["factor_expression"] = {"error": str(e)}

    # 测试4: 因子分层分析
    try:
        all_results["factor_layered"] = test_factor_layered_analysis(data)
    except Exception as e:
        print(f"测试4异常: {e}")
        traceback.print_exc()
        all_results["factor_layered"] = {"error": str(e)}

    # 测试5: 边界条件
    try:
        all_results["edge_cases"] = test_edge_cases()
    except Exception as e:
        print(f"测试5异常: {e}")
        traceback.print_exc()
        all_results["edge_cases"] = {"error": str(e)}

    # 保存结果
    result_path = REPORT_DIR / "test_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果已保存: {result_path}")

    return all_results


if __name__ == "__main__":
    main()
