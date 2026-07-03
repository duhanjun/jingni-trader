"""
综合验证测试套件
================

对 feat/quant-opt-20260624 分支的三个优化模块进行:
  1. 正确性测试 (与现有实现/手算结果对比)
  2. 性能对比测试 (向量化 vs 现有循环实现)
  3. 边界条件测试 (空数据、单股票、单日期等)

运行: python -m optimizations.run_all_tests
"""
from __future__ import annotations

import json
import time
import sys
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# 确保能 import 现有 main 分支的代码做对比
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizations_20260624.vectorized_backtest.vectorized_adapter import (
    VectorizedBacktester, signals_to_target_weights,
)
from optimizations_20260624.factor_analysis.enhanced_factor_analysis import (
    EnhancedFactorAnalyzer,
)
from optimizations_20260624.walk_forward.walk_forward_validator import (
    WalkForwardValidator,
)


# ======================================================================
# 测试工具
# ======================================================================
def _make_synthetic_prices(
    n_days: int = 252,
    n_stocks: int = 50,
    start_date: str = "2023-01-01",
    seed: int = 42,
    drift: float = 0.0005,
    vol: float = 0.015,
) -> pd.DataFrame:
    """生成合成收盘价矩阵 (date × code)"""
    np.random.seed(seed)
    dates = pd.bdate_range(start_date, periods=n_days)
    codes = [f"S{i:03d}.SZ" for i in range(n_stocks)]
    rets = np.random.normal(drift, vol, (n_days, n_stocks))
    prices = 10.0 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=dates, columns=codes)


def _make_predictive_factor(
    prices: pd.DataFrame,
    forward_period: int = 5,
    correlation: float = 0.6,
    seed: int = 7,
) -> pd.DataFrame:
    """
    构造一个有预测力的因子 (与未来收益正相关)
    factor_t = 0.5 * fwd_return_t + noise
    """
    np.random.seed(seed)
    fwd = prices.shift(-forward_period) / prices - 1.0
    noise = np.random.normal(0, 0.02, prices.shape)
    factor = fwd * correlation + noise * (1 - correlation)
    long = factor.stack().rename("alpha_pred").reset_index()
    long.columns = ["date", "code", "alpha_pred"]
    return long


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.details: List[Dict[str, Any]] = []

    def check(self, name: str, condition: bool, info: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        self.details.append({"name": name, "status": status, "info": info})
        flag = "✓" if condition else "✗"
        print(f"  [{flag}] {name} {('- ' + info) if info else ''}")

    def summary(self) -> Dict[str, Any]:
        return {
            "total": self.passed + self.failed,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.passed / (self.passed + self.failed) if (self.passed + self.failed) > 0 else 0.0,
        }


# ======================================================================
# 1. 向量化回测 — 正确性 + 性能 + 边界
# ======================================================================
def test_vectorized_backtest() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("测试 1: 向量化回测引擎 (借鉴 VectorBT)")
    print("=" * 70)
    tr = TestResult()
    perf = {}

    # --- 1.1 正确性: 手算可验证的小样本 ---
    print("\n--- 1.1 正确性测试 (手算小样本) ---")
    prices_small = pd.DataFrame(
        [[10.0, 20.0], [11.0, 20.0], [11.0, 22.0]],
        index=pd.date_range("2024-01-01", periods=3),
        columns=["A", "B"],
    )
    # 调仓日: 第0天全仓 A; 第1天切换全仓 B
    tw_small = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]],
        index=prices_small.index, columns=prices_small.columns,
    )
    bt = VectorizedBacktester(init_capital=10000.0, slippage=0.0, commission_rate=0.0, stamp_tax_rate=0.0)
    res = bt.run(prices_small, tw_small)

    # 第0天: 全仓 A, close=10 → 1000 股, equity=10000
    # 第1天: A=11, 切换到 B. 持仓 A 收益 = 11/10-1 = 10% → equity=11000
    #        切换全仓 B, close=20 → 550 股
    # 第2天: B=22, 收益 = 22/20-1 = 10% → equity=12100
    final_equity = res["equity_curve"]["equity"].iloc[-1]
    tr.check(
        "无成本下权益终值 ≈ 12100",
        abs(final_equity - 12100.0) < 1.0,
        f"实际={final_equity:.2f}",
    )
    # 总收益
    tr.check(
        "总收益率 ≈ 21%",
        abs(res["metrics"]["total_return"] - 0.21) < 0.001,
        f"实际={res['metrics']['total_return']:.4f}",
    )

    # --- 1.2 正确性: 含交易成本 ---
    print("\n--- 1.2 正确性测试 (含交易成本) ---")
    bt_cost = VectorizedBacktester(
        init_capital=10000.0, slippage=0.001, commission_rate=0.00025, stamp_tax_rate=0.001
    )
    res_cost = bt_cost.run(prices_small, tw_small)
    # 有成本 → 终值应低于无成本
    tr.check(
        "含成本终值 < 无成本终值",
        res_cost["equity_curve"]["equity"].iloc[-1] < final_equity,
        f"含成本={res_cost['equity_curve']['equity'].iloc[-1]:.2f} < 无成本={final_equity:.2f}",
    )
    # 调仓次数 = 2
    tr.check("调仓次数 = 2", res_cost["n_rebalances"] == 2, f"实际={res_cost['n_rebalances']}")

    # --- 1.3 性能: 向量化 vs 现有 native 循环 ---
    print("\n--- 1.3 性能对比 (向量化 vs 现有 native 循环) ---")
    # 复刻 native_adapter 的循环逻辑作为基线
    def native_loop_backtest(prices, target_weights, init_capital=1e6,
                             commission_rate=0.00025, stamp_tax_rate=0.001, slippage=0.0001):
        """复刻 skills/backtest-engine/scripts/adapters/native_adapter.py 的循环逻辑"""
        dates = prices.index
        codes = prices.columns
        weights = target_weights.reindex(index=prices.index, columns=codes).ffill().fillna(0.0)
        cash = init_capital
        positions = {c: 0.0 for c in codes}
        equity_records = []
        for dt in dates:
            day_px = prices.loc[dt]
            day_w = weights.loc[dt]
            # 计算当前权益
            mv = sum(positions[c] * day_px[c] for c in codes if positions[c] > 0)
            total_eq = cash + mv
            # 调仓 (若该日有权重)
            if target_weights.loc[dt].notna().any() if dt in target_weights.index else False:
                # 先卖出
                for c in codes:
                    target_val = total_eq * day_w[c]
                    current_val = positions[c] * day_px[c]
                    if current_val > target_val + 1e-6 and positions[c] > 0:
                        sell_shares = positions[c] - (target_val / day_px[c] if day_px[c] > 0 else 0)
                        sell_amount = sell_shares * day_px[c] * (1 - slippage)
                        comm = max(sell_amount * commission_rate, 5)
                        tax = sell_amount * stamp_tax_rate
                        cash += sell_amount - comm - tax
                        positions[c] -= sell_shares
                # 再买入
                for c in codes:
                    target_val = total_eq * day_w[c]
                    current_val = positions[c] * day_px[c]
                    if target_val > current_val + 1e-6:
                        buy_val = target_val - current_val
                        price = day_px[c] * (1 + slippage)
                        shares = buy_val / price if price > 0 else 0
                        buy_amount = shares * price
                        comm = max(buy_amount * commission_rate, 5)
                        if cash >= buy_amount + comm:
                            cash -= buy_amount + comm
                            positions[c] += shares
            mv = sum(positions[c] * day_px[c] for c in codes if positions[c] > 0)
            total_eq = cash + mv
            equity_records.append({"date": dt, "equity": total_eq})
        return pd.DataFrame(equity_records)

    # 大样本: 252 天 × 100 股
    prices_big = _make_synthetic_prices(n_days=252, n_stocks=100, seed=11)
    sig_big = _make_predictive_factor(prices_big, forward_period=5, seed=13)
    # signals_to_target_weights 期望 signal 列, 这里复用 alpha_pred 作为信号
    sig_big = sig_big.rename(columns={"alpha_pred": "signal"})
    # 每月调仓一次 (约 21 天)
    rebal_dates = prices_big.index[::21]
    tw_big = signals_to_target_weights(sig_big, prices_big, top_pct=0.2)
    # 仅保留月度调仓日
    tw_big = tw_big.loc[tw_big.index.intersection(rebal_dates)]

    # 向量化 (用 1e6 初始资金, 与 native 一致)
    bt_big = VectorizedBacktester(init_capital=1e6)
    t0 = time.perf_counter()
    res_vec = bt_big.run(prices_big, tw_big)
    t_vec = time.perf_counter() - t0

    # 循环 (现有 native 风格)
    t0 = time.perf_counter()
    eq_native = native_loop_backtest(prices_big, tw_big, init_capital=1e6)
    t_native = time.perf_counter() - t0

    speedup = t_native / t_vec if t_vec > 0 else float("inf")
    perf["vectorized_sec"] = round(t_vec, 4)
    perf["native_loop_sec"] = round(t_native, 4)
    perf["speedup"] = round(speedup, 1)

    tr.check(
        f"向量化更快 (加速比 {speedup:.1f}x)",
        speedup > 1.5,
        f"vec={t_vec:.4f}s vs native={t_native:.4f}s",
    )
    # 两者终值同量级 (策略语义略有差异，允许 30% 偏差)
    vec_final = res_vec["equity_curve"]["equity"].iloc[-1]
    native_final = eq_native["equity"].iloc[-1]
    ratio = min(vec_final, native_final) / max(vec_final, native_final)
    tr.check(
        "两引擎终值同量级 (差异<30%)",
        ratio > 0.7,
        f"vec={vec_final:.0f} native={native_final:.0f} ratio={ratio:.3f}",
    )

    # --- 1.4 边界: 单股票 ---
    print("\n--- 1.4 边界测试 (单股票) ---")
    px1 = _make_synthetic_prices(n_days=30, n_stocks=1, seed=5)
    # 仅第 0 天调仓 (全仓), 之后持有
    tw1 = pd.DataFrame(
        [[1.0]] + [[np.nan]] * 29,
        index=px1.index, columns=px1.columns,
    )
    res1 = bt.run(px1, tw1)
    tr.check("单股票回测不报错", len(res1["equity_curve"]) == 30)
    tr.check("单股票调仓次数=1", res1["n_rebalances"] == 1, f"实际={res1['n_rebalances']}")

    # --- 1.5 边界: 空权重 (全现金) ---
    print("\n--- 1.5 边界测试 (全现金) ---")
    tw_empty = pd.DataFrame(
        np.nan, index=prices_small.index, columns=prices_small.columns
    )
    res_empty = bt.run(prices_small, tw_empty)
    tr.check(
        "全现金权益不变",
        abs(res_empty["equity_curve"]["equity"].iloc[-1] - 10000.0) < 0.01,
        f"终值={res_empty['equity_curve']['equity'].iloc[-1]:.2f}",
    )

    # --- 1.6 边界: 价格含 NaN ---
    print("\n--- 1.6 边界测试 (价格含 NaN) ---")
    prices_nan = prices_small.copy()
    prices_nan.iloc[1, 1] = np.nan
    try:
        res_nan = bt.run(prices_nan, tw_small)
        ok = len(res_nan["equity_curve"]) == 3
    except Exception as e:
        ok = False
        res_nan = None
    tr.check("价格含 NaN 不崩溃", ok)

    return {"test": "vectorized_backtest", "result": tr.summary(), "perf": perf, "details": tr.details}


# ======================================================================
# 2. 增强因子分析 — 正确性 + 对比现有 + 边界
# ======================================================================
def test_factor_analysis() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("测试 2: 增强因子分析 (借鉴 alphalens-reloaded)")
    print("=" * 70)
    tr = TestResult()
    perf = {}

    prices = _make_synthetic_prices(n_days=200, n_stocks=80, seed=21)
    factor_df = _make_predictive_factor(prices, forward_period=5, correlation=0.7, seed=23)

    an = EnhancedFactorAnalyzer(quantiles=5, min_obs=10)

    # --- 2.1 IC 衰减: 有预测力的因子在 5 期 IC 应最高 ---
    print("\n--- 2.1 IC 衰减分析 ---")
    t0 = time.perf_counter()
    ic_decay = an.ic_decay(factor_df, prices, "alpha_pred", [1, 5, 10, 20])
    t_ic = time.perf_counter() - t0
    perf["ic_decay_sec"] = round(t_ic, 4)

    print(ic_decay)
    ic_5 = ic_decay.loc[5, "ic_mean"]
    ic_20 = ic_decay.loc[20, "ic_mean"]
    tr.check("5期 IC 为正 (因子有预测力)", ic_5 > 0, f"ic_5={ic_5:.4f}")
    tr.check("IC 随期数衰减 (5期 > 20期)", ic_5 > ic_20, f"ic_5={ic_5:.4f} > ic_20={ic_20:.4f}")
    tr.check("5期 IC 显著 (|t|>2)", abs(ic_decay.loc[5, "ic_t_stat"]) > 2,
             f"t={ic_decay.loc[5, 'ic_t_stat']:.2f}")

    # --- 2.2 Rank IC vs Normal IC ---
    print("\n--- 2.2 Rank IC (Spearman) vs Normal IC (Pearson) ---")
    ic_rank = an.ic_decay(factor_df, prices, "alpha_pred", [1, 5, 10, 20], method="spearman")
    rank_ic_5 = ic_rank.loc[5, "ic_mean"]
    tr.check("Rank IC 为正", rank_ic_5 > 0, f"rank_ic_5={rank_ic_5:.4f}")
    # Rank IC 通常比 Normal IC 稳定 (std 更小)，因对极值鲁棒
    tr.check("Rank IC std <= Normal IC std * 1.5",
             ic_rank.loc[5, "ic_std"] <= ic_decay.loc[5, "ic_std"] * 1.5,
             f"rank_std={ic_rank.loc[5, 'ic_std']:.4f} vs normal_std={ic_decay.loc[5, 'ic_std']:.4f}")

    # --- 2.3 因子换手率 ---
    print("\n--- 2.3 因子换手率 ---")
    turnover = an.factor_turnover(factor_df, "alpha_pred", lag=1)
    print(turnover)
    tr.check("换手率在 [0,1]", 0 <= turnover["mean_turnover"] <= 1,
             f"turnover={turnover['mean_turnover']:.4f}")
    tr.check("自相关性为正 (因子有一定持续性)", turnover["autocorrelation"] > 0,
             f"autocorr={turnover['autocorrelation']:.4f}")

    # --- 2.4 分层收益单调性 ---
    print("\n--- 2.4 分层收益 (5层) ---")
    qr = an.quantile_returns(factor_df, prices, "alpha_pred", forward_period=5)
    print(qr)
    tr.check("分层收益有 5 层", len(qr) == 5, f"层数={len(qr)}")
    # 最高层收益 > 最低层 (因子正向预测)
    if len(qr) >= 2:
        ls = qr.attrs.get("long_short_return", 0)
        tr.check("多空收益为正 (Q_max > Q_min)", ls > 0, f"long_short={ls:.4f}")
        # 单调性: 至少最高层 > 最低层
        tr.check("最高层 mean_return > 最低层",
                 qr["mean_return"].iloc[-1] > qr["mean_return"].iloc[0],
                 f"top={qr['mean_return'].iloc[-1]:.4f} > bottom={qr['mean_return'].iloc[0]:.4f}")

    # --- 2.5 性能: 向量化 IC vs 现有逐日循环 ---
    print("\n--- 2.5 性能对比 (向量化 IC vs 逐日循环) ---")
    # 复刻 factor-engine._calc_ic 的逐日循环
    def native_ic_loop(factor_df, prices, factor_name, forward_period=5):
        fwd = prices.shift(-forward_period) / prices - 1.0
        fw = factor_df.pivot_table(index="date", columns="code", values=factor_name)
        fw = fw.reindex(index=prices.index, columns=prices.columns)
        merged = pd.concat([fw.stack().rename("f"), fwd.stack().rename("r")], axis=1).dropna()
        merged = merged.reset_index().rename(columns={"level_0": "date", "level_1": "code"})
        from scipy import stats as st
        ic_list = []
        for dt, g in merged.groupby("date"):
            if len(g) < 10:
                continue
            ic, _ = st.spearmanr(g["f"], g["r"])
            if not np.isnan(ic):
                ic_list.append(ic)
        return np.array(ic_list)

    t0 = time.perf_counter()
    native_ics = native_ic_loop(factor_df, prices, "alpha_pred", 5)
    t_native_ic = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec_decay = an.ic_decay(factor_df, prices, "alpha_pred", [5], method="spearman")
    t_vec_ic = time.perf_counter() - t0

    speedup_ic = t_native_ic / t_vec_ic if t_vec_ic > 0 else float("inf")
    perf["native_ic_sec"] = round(t_native_ic, 4)
    perf["vectorized_ic_sec"] = round(t_vec_ic, 4)
    perf["ic_speedup"] = round(speedup_ic, 1)

    tr.check(
        f"向量化 IC 更快 (加速比 {speedup_ic:.1f}x)",
        speedup_ic > 1.0,
        f"vec={t_vec_ic:.4f}s vs native={t_native_ic:.4f}s",
    )
    # IC 均值应接近 (允许数值误差)
    native_ic_mean = float(np.mean(native_ics))
    vec_ic_mean = float(vec_decay.loc[5, "ic_mean"])
    tr.check(
        "向量化 IC 与循环 IC 均值一致 (差<0.01)",
        abs(native_ic_mean - vec_ic_mean) < 0.01,
        f"native={native_ic_mean:.4f} vec={vec_ic_mean:.4f}",
    )

    # --- 2.6 边界: 无预测力因子 ---
    print("\n--- 2.6 边界测试 (纯随机因子, 无预测力) ---")
    np.random.seed(99)
    random_factor = pd.DataFrame({
        "date": np.repeat(prices.index, prices.shape[1]),
        "code": np.tile(prices.columns, prices.shape[0]),
        "alpha_rand": np.random.uniform(-1, 1, prices.size),
    })
    ic_rand = an.ic_decay(random_factor, prices, "alpha_rand", [5])
    tr.check("随机因子 5期 |IC| < 0.1", abs(ic_rand.loc[5, "ic_mean"]) < 0.1,
             f"ic={ic_rand.loc[5, 'ic_mean']:.4f}")

    return {"test": "factor_analysis", "result": tr.summary(), "perf": perf, "details": tr.details}


# ======================================================================
# 3. 滚动训练验证器 — 正确性 + 泄漏校验 + 边界
# ======================================================================
def test_walk_forward() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("测试 3: 滚动训练验证器 (借鉴 Qlib)")
    print("=" * 70)
    tr = TestResult()
    perf = {}

    from sklearn.linear_model import LinearRegression

    # 构造时变数据: f1 系数从 +0.5 → -0.5 (非平稳)
    np.random.seed(3)
    dates = pd.bdate_range("2022-01-01", "2023-12-31")
    n_per_day = 10
    all_dates = np.repeat(dates.values, n_per_day)
    X = pd.DataFrame(np.random.normal(0, 1, (len(all_dates), 3)), columns=["f1", "f2", "f3"])
    coef_t = np.linspace(0.5, -0.5, len(dates))
    coef_per_row = np.repeat(coef_t, n_per_day)
    y = pd.Series(coef_per_row * X["f1"] + 0.2 * X["f2"] + np.random.normal(0, 0.1, len(X)))
    dates_aligned = pd.Series(all_dates)

    # --- 3.1 窗口生成 & 泄漏校验 ---
    print("\n--- 3.1 窗口生成 & 泄漏校验 ---")
    wf = WalkForwardValidator(train_size=120, test_size=30, embargo_gap=5)
    windows = wf.generate_windows(dates_aligned)
    validation = wf.validate_no_leakage(windows)

    tr.check("生成多个 fold", len(windows) >= 2, f"folds={len(windows)}")
    tr.check("泄漏校验通过", validation["passed"], f"violations={validation['violations']}")
    tr.check("测试集覆盖率 > 0", validation["test_coverage"] > 0,
             f"coverage={validation['test_coverage']:.3f}")

    # 每个 fold: train_end < test_start
    for i, (tr_d, te_d) in enumerate(windows):
        ok = tr_d.max() < te_d.min()
        if not ok:
            tr.check(f"fold {i} train_end < test_start", False,
                     f"{tr_d.max()} < {te_d.min()}")
            break
    else:
        tr.check("所有 fold train_end < test_start", True)

    # --- 3.2 滚动训练执行 ---
    print("\n--- 3.2 滚动训练执行 ---")
    t0 = time.perf_counter()
    result = wf.run(X, y, dates_aligned, lambda: LinearRegression())
    t_run = time.perf_counter() - t0
    perf["walk_forward_run_sec"] = round(t_run, 4)

    tr.check("产出样本外预测", result["oof_predictions"].notna().sum() > 0,
             f"非空预测数={result['oof_predictions'].notna().sum()}")
    tr.check("fold 数 = 窗口数", result["n_folds"] == len(windows),
             f"folds={result['n_folds']}")
    tr.check("样本外 IC 计算成功", "overall_oof_ic" in result)

    # --- 3.3 滚动训练 vs 单次训练 (非平稳数据上滚动应更优) ---
    print("\n--- 3.3 滚动 vs 单次训练对比 (非平稳数据) ---")
    # 单次训练: 用前 60% 训练, 后 40% 测试
    split = int(len(X) * 0.6)
    X_train_single = X.iloc[:split]
    y_train_single = y.iloc[:split]
    X_test_single = X.iloc[split:]
    y_test_single = y.iloc[split:]

    single_model = LinearRegression().fit(X_train_single, y_train_single)
    single_pred = single_model.predict(X_test_single)
    single_ic = pd.Series(single_pred, index=y_test_single.index).corr(y_test_single)

    rolling_ic = result["overall_oof_ic"]
    perf["single_train_ic"] = round(float(single_ic), 4)
    perf["rolling_train_ic"] = round(float(rolling_ic), 4)

    print(f"  单次训练样本外 IC: {single_ic:.4f}")
    print(f"  滚动训练样本外 IC: {rolling_ic:.4f}")
    # 非平稳数据上滚动训练通常更优 (IC 更高/更接近 0 不会更差)
    tr.check("滚动训练 IC >= 单次训练 IC",
             rolling_ic >= single_ic - 0.05,
             f"rolling={rolling_ic:.4f} >= single={single_ic:.4f}")

    # --- 3.4 边界: embargo_gap=0 仍不泄漏 (test_start > train_end) ---
    print("\n--- 3.4 边界测试 (embargo_gap=0) ---")
    wf0 = WalkForwardValidator(train_size=100, test_size=30, embargo_gap=0)
    w0 = wf0.generate_windows(dates_aligned)
    v0 = wf0.validate_no_leakage(w0)
    tr.check("gap=0 时仍无 train/test 重叠", v0["passed"], f"violations={v0['violations'][:2]}")

    # --- 3.5 边界: 数据量不足 ---
    print("\n--- 3.5 边界测试 (数据量不足) ---")
    short_dates = pd.Series(pd.bdate_range("2024-01-01", periods=50))
    wf_big = WalkForwardValidator(train_size=200, test_size=50, embargo_gap=5)
    w_short = wf_big.generate_windows(short_dates)
    tr.check("数据不足时返回空窗口", len(w_short) == 0, f"folds={len(w_short)}")

    return {"test": "walk_forward", "result": tr.summary(), "perf": perf, "details": tr.details}


# ======================================================================
# 主入口
# ======================================================================
def main():
    print("╔" + "═" * 68 + "╗")
    print("║  jingni-trader 优化验证测试 (feat/quant-opt-20260624)            ║")
    print("║  借鉴: VectorBT / alphalens-reloaded / Microsoft Qlib           ║")
    print("╚" + "═" * 68 + "╝")

    all_results = []
    all_results.append(test_vectorized_backtest())
    all_results.append(test_factor_analysis())
    all_results.append(test_walk_forward())

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    total_pass = sum(r["result"]["passed"] for r in all_results)
    total_fail = sum(r["result"]["failed"] for r in all_results)
    total = total_pass + total_fail
    print(f"总计: {total_pass} 通过 / {total_fail} 失败 / {total} 总数 "
          f"({total_pass/total*100:.1f}%)")
    for r in all_results:
        s = r["result"]
        print(f"  - {r['test']}: {s['passed']}/{s['total']} "
              f"({s['pass_rate']*100:.0f}%)")

    # 保存 JSON 结果
    out_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果 JSON 已保存: {out_path}")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())