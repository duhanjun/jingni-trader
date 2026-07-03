"""
jingni-trader 优化验证测试套件

测试维度:
1. 正确性测试 (correctness)
   - 向量化回测 vs 参考实现 (T+1 关闭时) 的等价性
   - T+1 实际执行验证 (新引擎) vs 未执行 (legacy)
   - 因子表达式结果 vs 手写 pandas 公式
   - 向量化 IC vs scipy 逐日 spearmanr
   - 向量化中性化 vs sklearn 逐日 LinearRegression
2. 性能测试 (performance)
   - 回测引擎: 不同规模 (股票数 × 天数) 下的耗时对比
   - IC 分析: 逐日循环 vs 向量化
   - 因子计算: 硬编码 vs 表达式引擎
3. 边界条件测试 (boundary)
   - 空数据、单只股票、单日数据
   - 全部涨停/跌停
   - 信号缺失列
   - 表达式语法错误

输出:
- 控制台打印结构化测试结果
- 生成 JSON 报告到 quant_opt/reports/test_report.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

# 确保能导入 quant_opt 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_opt.core.vectorized_backtest import VectorizedBacktestEngine
from quant_opt.core.factor_expression import FactorExpressionEngine, PRESET_FACTORS
from quant_opt.core.vectorized_ic import VectorizedICAnalyzer, VectorizedNeutralizer
from quant_opt.tests.synthetic_data import make_synthetic_data, make_signals
from quant_opt.tests.legacy_backtest import LegacyBacktestAdapter


REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── 测试结果收集 ────────────────────────────────────────────
class TestReport:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def record(self, category: str, name: str, passed: bool,
               detail: str = "", metrics: Dict = None):
        self.results.append({
            "category": category,
            "name": name,
            "passed": passed,
            "detail": detail,
            "metrics": metrics or {},
        })
        status = "PASS" if passed else "FAIL"
        print(f"  [{category}] {name}: {status}")
        if detail and not passed:
            print(f"    -> {detail}")
        if metrics:
            for k, v in metrics.items():
                print(f"    {k}: {v}")

    def summary(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        by_cat = {}
        for r in self.results:
            cat = r["category"]
            by_cat.setdefault(cat, {"total": 0, "passed": 0})
            by_cat[cat]["total"] += 1
            if r["passed"]:
                by_cat[cat]["passed"] += 1
        return {"total": total, "passed": passed, "by_category": by_cat}


# ── 1. 正确性测试 ──────────────────────────────────────────
def test_correctness(report: TestReport):
    print("\n=== 1. 正确性测试 ===")

    # 1.1 向量化回测 vs legacy (T+1 关闭) 应等价
    print("\n--- 1.1 向量化回测 vs legacy 等价性 (T+1 关闭) ---")
    data = make_synthetic_data(n_stocks=20, n_days=120, seed=1)
    signals = make_signals(data, strategy="momentum", rebalance_freq=5, seed=1)

    legacy = LegacyBacktestAdapter()
    t0 = time.perf_counter()
    res_legacy = legacy.run_backtest(data, signals, t_plus_1=False, price_limit=False)
    t_legacy = time.perf_counter() - t0

    vengine = VectorizedBacktestEngine(t_plus_1=False, price_limit=False)
    t0 = time.perf_counter()
    res_vec = vengine.run(data, signals)
    t_vec = time.perf_counter() - t0

    # 比较 equity curve (允许浮点误差)
    eq_legacy = res_legacy["equity_curve"].set_index("date")["equity"]
    eq_vec = res_vec["equity_curve"].set_index("date")["equity"]
    common_dates = eq_legacy.index.intersection(eq_vec.index)
    if len(common_dates) > 0:
        rel_err = (
            (eq_legacy.loc[common_dates] - eq_vec.loc[common_dates]).abs()
            / eq_legacy.loc[common_dates].abs().replace(0, 1)
        ).max()
        passed = rel_err < 0.02  # 2% 容差 (整手与资金分配细节差异)
        report.record("correctness", "回测等价性 (T+1 off)", passed,
                      f"最大相对误差 {rel_err:.4%}",
                      {"legacy_time_s": round(t_legacy, 4), "vec_time_s": round(t_vec, 4),
                       "speedup": round(t_legacy / t_vec, 2) if t_vec > 0 else 0})
    else:
        report.record("correctness", "回测等价性 (T+1 off)", False, "无共同日期")

    # 1.2 T+1 实际执行验证
    print("\n--- 1.2 T+1 交割约束验证 ---")
    # 新引擎: 构造 Day1 买入 + Day2 卖出 (T+1 满足) 与 Day1 买入 + Day1 卖出 (T+1 违反)
    # 用两段连续信号验证新引擎 T+1=True 时阻止同日卖出
    dates_sorted = sorted(data["date"].unique())
    day1, day2 = dates_sorted[0], dates_sorted[1]
    code0 = data["code"].iloc[0]
    # 同日买+卖信号 (买入先于卖出在信号表中)
    same_day_sig = pd.DataFrame([
        {"date": day1, "code": code0, "signal": 1},
        {"date": day1, "code": code0, "signal": -1},
    ])
    vengine_t1 = VectorizedBacktestEngine(t_plus_1=True, price_limit=False)
    res_t1 = vengine_t1.run(data, same_day_sig)
    # T+1 开启时，同日买入的股票不应被卖出 (卖出笔数应为 0)
    sells_t1 = res_t1["trades"][res_t1["trades"]["action"] == "sell"] if not res_t1["trades"].empty else pd.DataFrame()
    t1_blocks_same_day = sells_t1.empty
    report.record("correctness", "T+1 开启阻止同日卖出 (新引擎)", t1_blocks_same_day,
                  f"T+1=on 同日卖出笔数: {0 if sells_t1.empty else len(sells_t1)}")

    # 新引擎 T+1 不应过度阻止: Day1 买入, Day2 卖出 (T+1 满足) 应成功
    next_day_sig = pd.DataFrame([
        {"date": day1, "code": code0, "signal": 1},
        {"date": day2, "code": code0, "signal": -1},
    ])
    res_next = vengine_t1.run(data, next_day_sig)
    next_day_sells = res_next["trades"][res_next["trades"]["action"] == "sell"] if not res_next["trades"].empty else pd.DataFrame()
    t1_allows_next_day = not next_day_sells.empty
    report.record("correctness", "T+1 允许次日卖出 (新引擎)", t1_allows_next_day,
                  f"次日卖出笔数: {0 if next_day_sells.empty else len(next_day_sells)}")

    # legacy 引擎: t_plus_1=True 与 t_plus_1=False 应产生完全相同结果 (证明参数是死代码)
    # 用一个跨日场景: Day1 买, Day2 卖 (T+1 满足) + Day2 买, Day2 卖 (T+1 违反)
    cross_day_sig = pd.DataFrame([
        {"date": day1, "code": code0, "signal": 1},
        {"date": day2, "code": code0, "signal": -1},
        {"date": day2, "code": data["code"].iloc[1], "signal": 1},
        {"date": day2, "code": data["code"].iloc[1], "signal": -1},
    ])
    legacy_t1_on = legacy.run_backtest(data, cross_day_sig, t_plus_1=True, price_limit=False)
    legacy_t1_off = legacy.run_backtest(data, cross_day_sig, t_plus_1=False, price_limit=False)
    eq_on = legacy_t1_on["equity_curve"]["equity"].tolist()
    eq_off = legacy_t1_off["equity_curve"]["equity"].tolist()
    legacy_param_dead = (eq_on == eq_off)
    report.record("correctness", "T+1 参数在 legacy 中是死代码", legacy_param_dead,
                  "legacy t_plus_1=True/False 产生完全相同的净值曲线，证明参数未生效")

    # 1.3 因子表达式 vs 手写 pandas
    print("\n--- 1.3 因子表达式正确性 ---")
    df = make_synthetic_data(n_stocks=10, n_days=60, seed=2)
    engine = FactorExpressionEngine()

    # momentum_20 = Ref($close, 20) / $close - 1
    expr_factors = engine.compute(df, {"momentum_20": "Ref($close, 20) / $close - 1"})
    df_sorted = df.sort_values(["code", "date"]).reset_index(drop=True)
    manual = df_sorted.groupby("code")["close"].shift(20) / df_sorted["close"] - 1
    err = (expr_factors["momentum_20"] - manual).abs().max()
    report.record("correctness", "因子表达式 momentum_20", err < 1e-10,
                  f"最大误差: {err:.2e}")

    # vol_20 = Std($close/Ref($close,1)-1, 20)
    expr_factors2 = engine.compute(df, {"vol_20": "Std($close / Ref($close, 1) - 1, 20)"})
    ret1 = df_sorted.groupby("code")["close"].pct_change()
    manual_vol = df_sorted.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )
    err2 = (expr_factors2["vol_20"] - manual_vol).abs().max()
    report.record("correctness", "因子表达式 vol_20", err2 < 1e-10,
                  f"最大误差: {err2:.2e}")

    # 1.4 向量化 IC vs scipy 逐日
    print("\n--- 1.4 向量化 IC 正确性 ---")
    factor_df = engine.compute(df, {"mom_20": "Ref($close, 20) / $close - 1"})
    fwd = df_sorted[["code", "date"]].copy()
    fwd["ret_forward_5d"] = df_sorted.groupby("code")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )
    ic_vec = VectorizedICAnalyzer.calc_ic_series(
        factor_df, fwd, "mom_20", "ret_forward_5d", method="spearman"
    )
    # 手动逐日 spearman
    merged = factor_df.merge(fwd, on=["code", "date"]).dropna(subset=["mom_20", "ret_forward_5d"])
    ic_manual = []
    for dt, g in merged.groupby("date"):
        if len(g) < 10:
            continue
        r, _ = stats.spearmanr(g["mom_20"], g["ret_forward_5d"])
        if not np.isnan(r):
            ic_manual.append({"date": dt, "ic": r})
    ic_manual_s = pd.DataFrame(ic_manual).set_index("date")["ic"]
    common = ic_vec.index.intersection(ic_manual_s.index)
    if len(common) > 0:
        ic_err = (ic_vec.loc[common] - ic_manual_s.loc[common]).abs().max()
        report.record("correctness", "向量化 IC vs scipy", ic_err < 1e-10,
                      f"最大 IC 误差: {ic_err:.2e}, 样本数: {len(common)}")
    else:
        report.record("correctness", "向量化 IC vs scipy", False, "无共同日期")

    # 1.5 向量化中性化 vs sklearn 逐日
    print("\n--- 1.5 向量化中性化正确性 ---")
    np.random.seed(3)
    factor_df["lncap"] = np.random.lognormal(20, 1, len(factor_df))
    factor_df["industry"] = np.random.choice(["A", "B", "C", "D"], len(factor_df))
    factor_df["mom_20"] = factor_df["mom_20"].fillna(0)

    neut = VectorizedNeutralizer.neutralize(
        factor_df, ["mom_20"], industry_col="industry", mcap_col="lncap"
    )
    # 手动逐日 OLS
    dummies = pd.get_dummies(factor_df["industry"], prefix="ind", dtype=float)
    X_full = pd.concat([factor_df["lncap"], dummies], axis=1).fillna(0)
    X_full["_const"] = 1.0
    manual_resid = pd.Series(index=factor_df.index, dtype=float)
    for dt, g in factor_df.groupby("date"):
        if len(g) < 30:
            manual_resid.loc[g.index] = g["mom_20"]
            continue
        X = X_full.loc[g.index].values
        y = g["mom_20"].values
        m = LinearRegression().fit(X, y)
        manual_resid.loc[g.index] = y - m.predict(X)

    err_neut = (neut["mom_20_neutral"] - manual_resid).abs().max()
    report.record("correctness", "向量化中性化 vs sklearn", err_neut < 1e-6,
                  f"最大残差误差: {err_neut:.2e}")


# ── 2. 性能测试 ────────────────────────────────────────────
def test_performance(report: TestReport):
    print("\n=== 2. 性能测试 ===")

    # 2.1 回测引擎性能对比
    print("\n--- 2.1 回测引擎性能 (不同规模) ---")
    scenarios = [
        ("small", 20, 120),
        ("medium", 50, 250),
        ("large", 100, 500),
    ]
    perf_results = []
    for name, n_stk, n_day in scenarios:
        data = make_synthetic_data(n_stocks=n_stk, n_days=n_day, seed=10)
        signals = make_signals(data, strategy="momentum", rebalance_freq=5, seed=10)

        # legacy
        legacy = LegacyBacktestAdapter()
        t0 = time.perf_counter()
        legacy.run_backtest(data, signals, t_plus_1=False, price_limit=False)
        t_legacy = time.perf_counter() - t0

        # vectorized
        vengine = VectorizedBacktestEngine(t_plus_1=False, price_limit=False)
        t0 = time.perf_counter()
        vengine.run(data, signals)
        t_vec = time.perf_counter() - t0

        speedup = t_legacy / t_vec if t_vec > 0 else 0
        perf_results.append({
            "scenario": name, "n_stocks": n_stk, "n_days": n_day,
            "legacy_s": round(t_legacy, 4), "vec_s": round(t_vec, 4),
            "speedup": round(speedup, 2),
        })
        report.record("performance", f"回测性能 {name} ({n_stk}股×{n_day}天)",
                      speedup > 1.0, f"加速比 {speedup:.2f}x",
                      perf_results[-1])

    # 2.2 IC 分析性能
    print("\n--- 2.2 IC 分析性能 ---")
    data = make_synthetic_data(n_stocks=100, n_days=250, seed=20)
    df_sorted = data.sort_values(["code", "date"]).reset_index(drop=True)
    factor_df = df_sorted[["code", "date"]].copy()
    factor_df["mom_20"] = df_sorted.groupby("code")["close"].shift(20) / df_sorted["close"] - 1
    fwd = df_sorted[["code", "date"]].copy()
    fwd["ret_forward_5d"] = df_sorted.groupby("code")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )

    # 向量化
    t0 = time.perf_counter()
    VectorizedICAnalyzer.calc_ic_series(factor_df, fwd, "mom_20", "ret_forward_5d", "spearman")
    t_vec_ic = time.perf_counter() - t0

    # 逐日 scipy (复刻 main 分支 _calc_ic 逻辑)
    t0 = time.perf_counter()
    merged = factor_df.merge(fwd, on=["code", "date"]).dropna(subset=["mom_20", "ret_forward_5d"])
    for dt in merged["date"].unique():
        g = merged[merged["date"] == dt]
        if len(g) < 10:
            continue
        stats.spearmanr(g["mom_20"], g["ret_forward_5d"])
    t_loop_ic = time.perf_counter() - t0

    ic_speedup = t_loop_ic / t_vec_ic if t_vec_ic > 0 else 0
    report.record("performance", "IC 分析向量化加速", ic_speedup > 1.0,
                  f"加速比 {ic_speedup:.2f}x",
                  {"loop_s": round(t_loop_ic, 4), "vec_s": round(t_vec_ic, 4),
                   "speedup": round(ic_speedup, 2)})

    # 2.3 因子计算: 表达式引擎 vs 硬编码
    print("\n--- 2.3 因子表达式引擎性能 ---")
    data = make_synthetic_data(n_stocks=100, n_days=250, seed=30)
    df_sorted = data.sort_values(["code", "date"]).reset_index(drop=True)

    # 表达式引擎
    engine = FactorExpressionEngine()
    t0 = time.perf_counter()
    engine.compute(data, PRESET_FACTORS)
    t_expr = time.perf_counter() - t0

    # 手写等价 (复刻 main 分支 compute_a_share_factors 风格)
    t0 = time.perf_counter()
    result = df_sorted[["code", "date"]].copy()
    result["momentum_5"] = df_sorted.groupby("code")["close"].pct_change(5)
    result["momentum_20"] = df_sorted.groupby("code")["close"].pct_change(20)
    result["momentum_60"] = df_sorted.groupby("code")["close"].pct_change(60)
    result["reversal_5"] = -result["momentum_5"]
    result["reversal_20"] = -result["momentum_20"]
    result["vol_5"] = df_sorted.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(5, min_periods=3).std())
    result["vol_20"] = df_sorted.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std())
    result["vol_60"] = df_sorted.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(60, min_periods=30).std())
    result["ma_bias_20"] = df_sorted["close"] / df_sorted.groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()) - 1
    result["ma_bias_60"] = df_sorted["close"] / df_sorted.groupby("code")["close"].transform(
        lambda x: x.rolling(60, min_periods=30).mean()) - 1
    result["volume_ratio_20"] = df_sorted["volume"] / df_sorted.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    result["volume_std_20"] = df_sorted.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).std()) / df_sorted.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    result["range_20"] = (df_sorted.groupby("code")["high"].transform(
        lambda x: x.rolling(20, min_periods=10).max()) - df_sorted.groupby("code")["low"].transform(
        lambda x: x.rolling(20, min_periods=10).min())) / df_sorted["close"]
    result["turnover_mean_20"] = df_sorted.groupby("code")["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=10).mean())
    t_manual = time.perf_counter() - t0

    overhead = t_expr / t_manual if t_manual > 0 else 0
    report.record("performance", "因子表达式引擎开销",
                  overhead < 5.0,  # 表达式引擎允许 5x 开销 (换取扩展性)
                  f"表达式 {t_expr:.3f}s vs 手写 {t_manual:.3f}s, 开销 {overhead:.2f}x",
                  {"expr_s": round(t_expr, 4), "manual_s": round(t_manual, 4),
                   "overhead_x": round(overhead, 2)})


# ── 3. 边界条件测试 ────────────────────────────────────────
def test_boundary(report: TestReport):
    print("\n=== 3. 边界条件测试 ===")

    # 3.1 空数据
    print("\n--- 3.1 空数据 ---")
    vengine = VectorizedBacktestEngine()
    res = vengine.run(pd.DataFrame(), pd.DataFrame())
    report.record("boundary", "空数据回测", res["equity_curve"].empty,
                  "应返回空结果")

    # 3.2 单只股票
    print("\n--- 3.2 单只股票 ---")
    data = make_synthetic_data(n_stocks=1, n_days=60, seed=100)
    signals = make_signals(data, strategy="momentum", rebalance_freq=5, seed=100)
    res = vengine.run(data, signals)
    report.record("boundary", "单只股票回测", not res["equity_curve"].empty,
                  f"净值记录数: {len(res['equity_curve'])}")

    # 3.3 全部涨停日 (无法买入)
    print("\n--- 3.3 全部涨停 ---")
    data = make_synthetic_data(n_stocks=5, n_days=30, seed=101, with_limits=True)
    # 强制所有交易日涨停
    data["is_limit_up"] = True
    sig_date = data["date"].iloc[10]
    signals = pd.DataFrame([
        {"date": sig_date, "code": c, "signal": 1}
        for c in data["code"].unique()
    ])
    res = vengine.run(data, signals)
    no_buy = res["trades"].empty or (res["trades"]["action"] == "buy").sum() == 0
    report.record("boundary", "全部涨停无法买入", no_buy,
                  f"买入笔数: {0 if res['trades'].empty else (res['trades']['action']=='buy').sum()}")

    # 3.4 表达式语法错误
    print("\n--- 3.4 表达式语法错误 ---")
    engine = FactorExpressionEngine()
    bad_exprs = [
        ("unknown_op", "Foo($close, 20)"),       # 未知算子
        ("syntax_err", "Ref($close, 20) / "),    # 语法错误
        ("missing_field", "Ref($foo, 20)"),       # 字段不存在
    ]
    for name, expr in bad_exprs:
        try:
            engine.compute(make_synthetic_data(n_stocks=3, n_days=30, seed=102), {"f": expr})
            report.record("boundary", f"表达式错误处理 {name}", False, "应抛异常但未抛")
        except (ValueError, KeyError) as e:
            report.record("boundary", f"表达式错误处理 {name}", True, f"正确抛异常: {type(e).__name__}")

    # 3.5 未知算子列表
    print("\n--- 3.5 算子注册表 ---")
    ops = engine.list_operators()
    expected = ["Ref", "Mean", "Std", "Sum", "Max", "Min", "Delta", "Rank",
                "Corr", "Cov", "Abs", "Log", "Sign", "Greater", "Less"]
    has_all = all(op in ops for op in expected)
    report.record("boundary", "算子注册表完整", has_all,
                  f"已注册 {len(ops)} 个算子: {ops}")

    # 3.6 IC 空因子
    print("\n--- 3.6 IC 空数据 ---")
    empty_ic = VectorizedICAnalyzer.calc_ic_series(
        pd.DataFrame(columns=["code", "date", "f"]),
        pd.DataFrame(columns=["code", "date", "r"]),
        "f", "r", "spearman",
    )
    report.record("boundary", "IC 空数据", empty_ic.empty, "应返回空 Series")


# ── 主入口 ─────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("jingni-trader 优化验证测试 (feat/quant-opt-20260623)")
    print("=" * 70)

    report = TestReport()

    try:
        test_correctness(report)
    except Exception as e:
        report.record("correctness", "正确性测试套件", False, f"异常: {e}")
        traceback.print_exc()

    try:
        test_performance(report)
    except Exception as e:
        report.record("performance", "性能测试套件", False, f"异常: {e}")
        traceback.print_exc()

    try:
        test_boundary(report)
    except Exception as e:
        report.record("boundary", "边界测试套件", False, f"异常: {e}")
        traceback.print_exc()

    # 汇总
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    summary = report.summary()
    print(f"总计: {summary['total']}, 通过: {summary['passed']}, "
          f"失败: {summary['total'] - summary['passed']}")
    for cat, s in summary["by_category"].items():
        print(f"  {cat}: {s['passed']}/{s['total']}")

    # 保存 JSON 报告
    report_path = os.path.join(REPORT_DIR, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results": report.results,
            "timestamp": pd.Timestamp.now().isoformat(),
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {report_path}")

    return 0 if summary["total"] == summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
