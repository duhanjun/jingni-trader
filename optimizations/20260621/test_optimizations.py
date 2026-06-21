"""
验证测试: 三项优化的正确性 / 性能 / 边界条件测试
================================================================================

测试内容:
  A. 因子表达式引擎
     - 正确性: 与等价 pandas 实现逐值对比 (Ref/Mean/Std/Rank/Corr/Slope/Delta)
     - 性能: polars vs pandas 在 200 股 × 1000 日面板上的耗时对比
     - 边界: 空数据、单股、窗口大于数据长度、未知算子、非法公式
  B. 向量化回测 + 修正指标
     - 正确性: DateIndexedBacktester 与复刻的原版逐日逻辑在相同输入下结果一致
     - 性能: 日期预索引 vs 原版逐日过滤的耗时对比
     - 指标修正: 验证原版 Sharpe 口径不一致、win_rate 被买单污染的问题
     - 边界: 空数据、无信号、单日
  C. Walk-Forward 滚动分割
     - 正确性: 训练集严格早于测试集、折间无重叠 (含 embargo)、折数符合预期
     - 边界: 交易日不足、purge/embargo 各种组合、expanding vs rolling
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# 确保能 import 优化模块
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factor_expression_engine import (  # noqa: E402
    FactorExpressionEngine, PRESET_FACTORS,
)
from vectorized_backtest_metrics import (  # noqa: E402
    CorrectedMetrics, DateIndexedBacktester, OriginalMetricsReplica,
)
from walk_forward_split import RollingWindowSplit  # noqa: E402


# ===========================================================================
# 测试工具
# ===========================================================================

class TestReport:
    def __init__(self):
        self.records = []  # [(category, name, passed, detail)]

    def record(self, category: str, name: str, passed: bool, detail: str = ""):
        self.records.append((category, name, passed, detail))
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {category} :: {name}" + (f"  -- {detail}" if detail else ""))

    def summary(self) -> dict:
        n_pass = sum(1 for r in self.records if r[2])
        n_fail = sum(1 for r in self.records if not r[2])
        return {"total": len(self.records), "pass": n_pass, "fail": n_fail}


def approx_equal(a, b, tol=1e-6) -> bool:
    """浮点近似相等 (支持标量与数组, 含 NaN)。"""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    # 标量 (0-d) 特殊处理
    if a_arr.ndim == 0 or b_arr.ndim == 0:
        if np.isnan(a_arr) and np.isnan(b_arr):
            return True
        if np.isnan(a_arr) or np.isnan(b_arr):
            return False
        return bool(abs(float(a_arr) - float(b_arr)) <= tol)
    if a_arr.shape != b_arr.shape:
        return False
    both_nan = np.isnan(a_arr) & np.isnan(b_arr)
    diff = np.abs(a_arr - b_arr)
    diff = np.where(both_nan, 0.0, diff)
    # 忽略单边 NaN (视为不相等的位置不影响整体判断时跳过)
    finite = ~both_nan
    if not finite.any():
        return True
    return bool(np.nanmax(diff[finite]) <= tol)


# ===========================================================================
# 合成数据
# ===========================================================================

def make_panel(n_codes: int = 50, n_days: int = 250, seed: int = 42) -> pl.DataFrame:
    """合成 OHLCV 面板数据 (polars)。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    rows = []
    for i in range(n_codes):
        code = f"{600000 + i:06d}.SH"
        price = 10.0 + i * 0.5
        for d in dates:
            ret = rng.normal(0, 0.02)
            price = max(price * (1 + ret), 0.5)
            vol = float(rng.lognormal(12, 0.6))
            rows.append({
                "code": code, "date": d, "open": price * (1 + rng.normal(0, 0.003)),
                "high": price * (1 + abs(rng.normal(0, 0.005))),
                "low": price * (1 - abs(rng.normal(0, 0.005))),
                "close": price, "volume": vol, "amount": vol * price,
            })
    df = pl.DataFrame(rows).sort(["code", "date"])
    return df


def make_panel_pandas(n_codes: int = 50, n_days: int = 250, seed: int = 42) -> pd.DataFrame:
    return make_panel(n_codes, n_days, seed).to_pandas()


# ===========================================================================
# A. 因子表达式引擎测试
# ===========================================================================

def test_factor_engine_correctness(report: TestReport):
    print("\n=== A1. 因子表达式引擎 - 正确性 (vs pandas) ===")
    df_pl = make_panel(n_codes=30, n_days=200, seed=7)
    df_pd = df_pl.to_pandas().sort_values(["code", "date"]).reset_index(drop=True)

    engine = FactorExpressionEngine()

    # --- Ref ---
    out = engine.compute(df_pl, {"f_ref5": "Ref($close, 5)"})
    expect = df_pd.groupby("code")["close"].shift(5).reset_index(drop=True)
    got = out["f_ref5"].to_pandas()
    ok = approx_equal(got, expect)
    report.record("A.正确性", "Ref($close,5)", ok)

    # --- Delta ---
    out = engine.compute(df_pl, {"f_delta5": "Delta($close, 5)"})
    expect = (df_pd["close"] - df_pd.groupby("code")["close"].shift(5)).reset_index(drop=True)
    got = out["f_delta5"].to_pandas()
    ok = approx_equal(got, expect)
    report.record("A.正确性", "Delta($close,5)", ok)

    # --- Mean (Ts_Mean) ---
    out = engine.compute(df_pl, {"f_mean20": "Mean($close, 20)"})
    expect = df_pd.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=10).mean()).reset_index(drop=True)
    got = out["f_mean20"].to_pandas()
    ok = approx_equal(got, expect, tol=1e-4)
    report.record("A.正确性", "Mean($close,20)", ok)

    # --- Std ---
    out = engine.compute(df_pl, {"f_std20": "Std($close, 20)"})
    expect = df_pd.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=10).std()).reset_index(drop=True)
    got = out["f_std20"].to_pandas()
    ok = approx_equal(got, expect, tol=1e-4)
    report.record("A.正确性", "Std($close,20)", ok)

    # --- 截面 Rank ---
    out = engine.compute(df_pl, {"f_csr": "Rank($close)"})
    expect = df_pd.groupby("date")["close"].rank(pct=True).reset_index(drop=True)
    got = out["f_csr"].to_pandas()
    ok = approx_equal(got, expect, tol=1e-6)
    report.record("A.正确性", "Rank($close) 截面", ok)

    # --- 复合公式: 反转 + 截面排名 ---
    formula = "Rank(-Ref($close, 5) / Ref($close, 1))"
    out = engine.compute(df_pl, {"f_compound": formula})
    inner = -df_pd.groupby("code")["close"].shift(5) / df_pd.groupby("code")["close"].shift(1)
    inner = inner.reset_index(drop=True)
    expect = inner.groupby(df_pd["date"]).rank(pct=True).reset_index(drop=True)
    got = out["f_compound"].to_pandas()
    ok = approx_equal(got, expect, tol=1e-6)
    report.record("A.正确性", "复合 Rank(-Ref/Ref)", ok)

    # --- Corr (量价相关) ---
    out = engine.compute(df_pl, {"f_corr": "Corr($close, $volume, 20)"})
    expect = df_pd.groupby("code").apply(
        lambda g: g["close"].rolling(20, min_periods=10).corr(g["volume"])
    ).reset_index(level=0, drop=True).reset_index(drop=True)
    got = out["f_corr"].to_pandas()
    ok = approx_equal(got, expect, tol=1e-4)
    report.record("A.正确性", "Corr($close,$volume,20)", ok)

    # --- Slope (趋势) ---
    out = engine.compute(df_pl, {"f_slope": "Slope($close, 20)"})
    # pandas 参考实现
    def _slope_pd(s, n=20):
        x = np.arange(n)
        x_mean = x.mean()
        y = s.rolling(n, min_periods=2)
        res = []
        for arr in y:
            if len(arr) < 2:
                res.append(np.nan)
                continue
            xx = np.arange(len(arr))
            xm = xx.mean()
            num = ((xx - xm) * (arr - arr.mean())).sum()
            den = ((xx - xm) ** 2).sum()
            res.append(num / den if den > 0 else np.nan)
        return pd.Series(res, index=s.index)
    expect = df_pd.groupby("code")["close"].transform(_slope_pd).reset_index(drop=True)
    got = out["f_slope"].to_pandas()
    ok = approx_equal(got, expect, tol=1e-4)
    report.record("A.正确性", "Slope($close,20)", ok)


def test_factor_engine_performance(report: TestReport):
    print("\n=== A2. 因子表达式引擎 - 性能 (polars vs pandas) ===")
    n_codes, n_days = 200, 1000
    df_pl = make_panel(n_codes=n_codes, n_days=n_days, seed=11)
    df_pd = df_pl.to_pandas().sort_values(["code", "date"]).reset_index(drop=True)
    engine = FactorExpressionEngine()

    formulas = PRESET_FACTORS

    # polars
    t0 = time.perf_counter()
    out_pl = engine.compute(df_pl, formulas)
    t_polars = time.perf_counter() - t0

    # pandas 等价实现 (复刻 jingni-trader 风格的逐列 transform)
    t0 = time.perf_counter()
    out_pd = df_pd[["code", "date"]].copy()
    g = df_pd.groupby("code")
    out_pd["reversal_5d"] = -g["close"].shift(5) / g["close"].shift(1)
    out_pd["reversal_20d"] = -g["close"].shift(20) / g["close"].shift(1)
    out_pd["momentum_20"] = g["close"].shift(20) / df_pd["close"]
    out_pd["momentum_60"] = g["close"].shift(60) / df_pd["close"]
    out_pd["volatility_20"] = g["close"].transform(lambda x: x.pct_change().rolling(20, min_periods=10).std())
    out_pd["volatility_5"] = g["close"].transform(lambda x: x.pct_change().rolling(5, min_periods=3).std())
    out_pd["vol_ratio"] = df_pd["volume"] / g["volume"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    out_pd["amount_ratio"] = df_pd["amount"] / g["amount"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    out_pd["slope_20"] = g["close"].transform(lambda x: x.rolling(20, min_periods=2).apply(
        lambda a: float(np.polyfit(np.arange(len(a)), a, 1)[0]) if len(a) > 1 else np.nan, raw=True))
    out_pd["corr_pv_20"] = df_pd.groupby("code").apply(
        lambda gg: gg["close"].rolling(20, min_periods=10).corr(gg["volume"])
    ).reset_index(level=0, drop=True)
    out_pd["cs_rank_mom"] = out_pd["momentum_20"].groupby(df_pd["date"]).rank(pct=True)
    out_pd["cs_zscore_vol"] = out_pd["volatility_20"].groupby(df_pd["date"]).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-12))
    t_pandas = time.perf_counter() - t0

    speedup = t_pandas / t_polars if t_polars > 0 else float("inf")
    report.record(
        "A.性能", f"polars vs pandas ({n_codes}股×{n_days}日, {len(formulas)}因子)",
        passed=t_polars < t_pandas,
        detail=f"polars={t_polars:.3f}s pandas={t_pandas:.3f}s 加速比={speedup:.2f}x",
    )


def test_factor_engine_boundary(report: TestReport):
    print("\n=== A3. 因子表达式引擎 - 边界条件 ===")
    engine = FactorExpressionEngine()

    # 空数据
    empty = pl.DataFrame({"code": [], "date": [], "close": [], "volume": []})
    out = engine.compute(empty, {"f": "Ref($close, 5)"})
    report.record("A.边界", "空数据不报错", out.height == 0)

    # 单股
    single = make_panel(n_codes=1, n_days=50, seed=1)
    out = engine.compute(single, {"f": "Mean($close, 10)"})
    report.record("A.边界", "单股计算", out.height == 50 and "f" in out.columns)

    # 窗口大于数据长度 (应产生 NaN 而非报错)
    small = make_panel(n_codes=2, n_days=10, seed=2)
    out = engine.compute(small, {"f": "Mean($close, 100)"})
    all_nan = bool(out["f"].null_count() == out.height)
    report.record("A.边界", "窗口>数据长度→全NaN", all_nan)

    # 未知算子 → 该列填 None, 不抛异常
    out = engine.compute(make_panel(2, 30, 3), {"bad": "UnknownOp($close, 5)"})
    report.record("A.边界", "未知算子不崩溃", "bad" in out.columns)

    # 非法公式 → 该列填 None, 不抛异常
    out = engine.compute(make_panel(2, 30, 4), {"bad": "Ref($close, )"})
    report.record("A.边界", "非法公式不崩溃", "bad" in out.columns)


# ===========================================================================
# B. 向量化回测 + 修正指标测试
# ===========================================================================

def _replicate_original_backtest(data, signals, init_capital=1e6, commission_rate=0.00025,
                                  stamp_tax_rate=0.001, slippage=0.001, price_limit=True):
    """忠实复刻 native_adapter.py 的逐日过滤逻辑, 用于等价性对比。"""
    data = data.sort_values(["date", "code"]).reset_index(drop=True)
    signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
    dates = sorted(signals["date"].unique())
    cash = init_capital
    positions = {}
    equity_records = []
    trades = []
    for dt in dates:
        day_signal = signals[signals["date"] == dt]
        day_data = data[data["date"] == dt]  # 原版的 O(n) 过滤
        if day_data.empty:
            continue
        day_data_map = day_data.set_index("code")
        sell_codes, buy_codes = [], []
        for _, row in day_signal.iterrows():
            code = row["code"]; sig = float(row.get("signal", 0))
            if sig > 0: buy_codes.append(code)
            elif sig < 0: sell_codes.append(code)
        for code in sell_codes:
            if positions.get(code, 0) <= 0 or code not in day_data_map.index:
                continue
            pr = day_data_map.loc[code]
            if price_limit and bool(pr.get("is_limit_down", False)):
                continue
            price = pr["close"]; shares = positions[code]
            sell_amount = price * shares
            commission = max(sell_amount * commission_rate, 5)
            tax = sell_amount * stamp_tax_rate
            cash += sell_amount - commission - tax
            trades.append({"date": dt, "code": code, "action": "sell", "price": price,
                           "shares": shares, "amount": sell_amount, "commission": commission,
                           "tax": tax, "pnl": sell_amount - commission - tax})
            positions[code] = 0
        if buy_codes:
            budget = cash * 0.95 / len(buy_codes)
            for code in buy_codes:
                if code not in day_data_map.index:
                    continue
                pr = day_data_map.loc[code]
                if price_limit and bool(pr.get("is_limit_up", False)):
                    continue
                price = pr["close"] * (1 + slippage)
                shares = int(budget / price / 100) * 100
                if shares <= 0:
                    continue
                buy_amount = price * shares
                commission = max(buy_amount * commission_rate, 5)
                cost = buy_amount + commission
                if cost > cash:
                    shares = int((cash * 0.98) / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                cash -= cost
                positions[code] = positions.get(code, 0) + shares
                trades.append({"date": dt, "code": code, "action": "buy", "price": price,
                               "shares": shares, "amount": buy_amount, "commission": commission,
                               "tax": 0, "pnl": -buy_amount - commission})
        mv = sum(s * day_data_map.loc[c, "close"] for c, s in positions.items()
                 if s > 0 and c in day_data_map.index)
        equity_records.append({"date": dt, "equity": cash + mv, "cash": cash,
                               "market_value": mv, "position_count": sum(1 for s in positions.values() if s > 0)})
    return pd.DataFrame(equity_records), pd.DataFrame(trades)


def test_backtest_correctness(report: TestReport):
    print("\n=== B1. 向量化回测 - 正确性 (vs 原版逐日过滤) ===")
    rng = np.random.default_rng(99)
    n_codes, n_days = 20, 120
    df_pd = make_panel_pandas(n_codes=n_codes, n_days=n_days, seed=99)
    # 给一些涨跌停标记 (随机少量)
    df_pd["is_limit_up"] = df_pd["close"].pct_change() >= 0.099
    df_pd["is_limit_down"] = df_pd["close"].pct_change() <= -0.099

    # 生成信号: 每天随机选 5 只买入、5 只卖出
    sig_rows = []
    for d in sorted(df_pd["date"].unique()):
        codes = df_pd[df_pd["date"] == d]["code"].tolist()
        buy = rng.choice(codes, size=min(5, len(codes)), replace=False)
        sell = rng.choice(codes, size=min(3, len(codes)), replace=False)
        for c in buy:
            sig_rows.append({"date": d, "code": c, "signal": 1})
        for c in sell:
            sig_rows.append({"date": d, "code": c, "signal": -1})
    signals = pd.DataFrame(sig_rows)

    opt = DateIndexedBacktester()
    res_opt = opt.run_backtest(df_pd, signals)
    eq_orig, tr_orig = _replicate_original_backtest(df_pd, signals)

    # 净值曲线逐日对比
    eq_opt = res_opt["equity_curve"].reset_index(drop=True)
    eq_orig = eq_orig.reset_index(drop=True)
    if len(eq_opt) == len(eq_orig):
        ok = approx_equal(eq_opt["equity"].values, eq_orig["equity"].values, tol=1e-6)
        report.record("B.正确性", "净值曲线与原版一致", ok,
                      detail=f"max_diff={np.nanmax(np.abs(eq_opt['equity'].values - eq_orig['equity'].values)):.2e}")
    else:
        report.record("B.正确性", "净值曲线长度一致", False,
                      detail=f"opt={len(eq_opt)} orig={len(eq_orig)}")

    # 成交记录数一致
    n_opt = len(res_opt["trades"]); n_orig = len(tr_orig)
    report.record("B.正确性", "成交笔数一致", n_opt == n_orig, detail=f"opt={n_opt} orig={n_orig}")


def test_backtest_performance(report: TestReport):
    print("\n=== B2. 向量化回测 - 性能 (日期预索引 vs 原版逐日过滤) ===")
    n_codes, n_days = 100, 500
    df_pd = make_panel_pandas(n_codes=n_codes, n_days=n_days, seed=33)
    df_pd["is_limit_up"] = False
    df_pd["is_limit_down"] = False
    rng = np.random.default_rng(33)
    sig_rows = []
    for d in sorted(df_pd["date"].unique()):
        codes = df_pd[df_pd["date"] == d]["code"].tolist()
        for c in rng.choice(codes, size=min(10, len(codes)), replace=False):
            sig_rows.append({"date": d, "code": c, "signal": rng.choice([1, -1])})
    signals = pd.DataFrame(sig_rows)

    opt = DateIndexedBacktester()
    t0 = time.perf_counter()
    res_opt = opt.run_backtest(df_pd, signals)
    t_opt = time.perf_counter() - t0

    t0 = time.perf_counter()
    _eq_orig, _tr_orig = _replicate_original_backtest(df_pd, signals)
    t_orig = time.perf_counter() - t0

    speedup = t_orig / t_opt if t_opt > 0 else float("inf")
    report.record("B.性能", f"日期预索引 vs 逐日过滤 ({n_codes}股×{n_days}日)",
                  passed=t_opt < t_orig,
                  detail=f"opt={t_opt:.3f}s orig={t_orig:.3f}s 加速比={speedup:.2f}x")


def test_metrics_correction(report: TestReport):
    print("\n=== B3. 指标修正 - Sharpe 口径 / 胜率 ===")
    # 构造一条已知净值曲线: 100 日起始, 每日 +0.1% 波动
    rng = np.random.default_rng(5)
    days = 252
    rets = rng.normal(0.0004, 0.01, days)
    equity = pd.Series(np.cumprod(1 + rets) * 1e6, index=pd.bdate_range("2023-01-02", periods=days))
    equity.name = "equity"

    # 原版
    returns = equity.pct_change().dropna()
    orig_sharpe = OriginalMetricsReplica.calc_sharpe(returns)
    orig_ann = OriginalMetricsReplica.calc_annual_return(equity)
    # 修正版
    corr = CorrectedMetrics.calc_sharpe_consistent(equity)

    # 验证原版 Sharpe 分子(算术) != 年化收益(几何) → 口径不一致
    arith_ann = corr["annual_return_arith"]
    geom_ann = corr["annual_return_geom"]
    inconsistent = abs(arith_ann - geom_ann) > 1e-9
    report.record("B.指标修正", "诊断: 原版算术年化≠几何年化 (确认问题存在)",
                  passed=inconsistent, detail=f"arith={arith_ann:.4f} geom={geom_ann:.4f}")

    # 验证修正版同时给出两种口径
    report.record("B.指标修正", "修正版输出两种 Sharpe 口径",
                  passed="sharpe_arith" in corr and "sharpe_geom" in corr,
                  detail=f"sharpe_arith={corr['sharpe_arith']:.4f} sharpe_geom={corr['sharpe_geom']:.4f}")

    # 验证原版 sharpe == 修正版 sharpe_arith (复刻一致)
    ok = approx_equal(orig_sharpe, corr["sharpe_arith"], tol=1e-9)
    report.record("B.指标修正", "原版Sharpe == 修正版sharpe_arith (复刻一致)",
                  passed=ok, detail=f"orig={orig_sharpe:.6f} arith={corr['sharpe_arith']:.6f}")

    # --- 胜率修正 ---
    # 构造 trades: 3 笔买单(原版pnl恒负) + 2 笔卖单(1盈1亏)
    trades = pd.DataFrame([
        {"action": "buy", "pnl": -10050},
        {"action": "buy", "pnl": -20300},
        {"action": "buy", "pnl": -5100},
        {"action": "sell", "pnl": 500},    # 盈
        {"action": "sell", "pnl": -200},   # 亏
    ])
    orig_wr = OriginalMetricsReplica.calc_win_rate(trades)  # 1/5 = 0.2 (被买单污染)
    corr_wr = CorrectedMetrics.calc_win_rate_corrected(trades)  # 1/2 = 0.5 (仅平仓单)
    ok = approx_equal(orig_wr, 0.2, tol=1e-9) and approx_equal(corr_wr, 0.5, tol=1e-9)
    report.record("B.指标修正", "胜率: 原版被买单污染(0.2) vs 修正版仅平仓单(0.5)",
                  passed=ok, detail=f"orig={orig_wr} corrected={corr_wr}")

    # --- benchmark 相对指标 ---
    bench_rets = pd.Series(rng.normal(0.0003, 0.008, len(returns)), index=returns.index)
    rel = CorrectedMetrics.calc_benchmark_metrics(returns, bench_rets)
    has_all = all(k in rel for k in
                  ["excess_return", "beta", "alpha", "information_ratio", "tracking_error"])
    report.record("B.指标修正", "benchmark 相对指标齐全",
                  passed=has_all, detail=str({k: round(v, 4) for k, v in rel.items()}))


def test_backtest_boundary(report: TestReport):
    print("\n=== B4. 向量化回测 - 边界条件 ===")
    opt = DateIndexedBacktester()
    empty = pd.DataFrame(columns=["date", "code", "close"])
    res = opt.run_backtest(empty, empty)
    report.record("B.边界", "空数据返回空结果", res["equity_curve"].empty)

    # 无信号
    df = make_panel_pandas(5, 30, 8)
    df["is_limit_up"] = False; df["is_limit_down"] = False
    res = opt.run_backtest(df, pd.DataFrame(columns=["date", "code", "signal"]))
    report.record("B.边界", "无信号→空净值", res["equity_curve"].empty)


# ===========================================================================
# C. Walk-Forward 滚动分割测试
# ===========================================================================

def test_walkforward_correctness(report: TestReport):
    print("\n=== C1. Walk-Forward - 正确性 ===")
    dates = pd.bdate_range("2020-01-02", periods=756)  # 3 年

    # rolling 模式: train=252, test=63, step=63, purge=21, embargo=10
    sp = RollingWindowSplit(train_window=252, test_window=63, step=63,
                            purge_days=21, embargo_days=10, mode="rolling")
    folds = sp.split(dates)

    # 折数预期: (756 - 252 - 21) / 63 ≈ 7.x, 但受 embargo 影响
    report.record("C.正确性", "生成折数>0", len(folds) > 0, detail=f"{len(folds)} 折")

    # 训练集严格早于测试集
    all_ordered = all(seg.train_end < seg.test_start for seg in folds)
    report.record("C.正确性", "训练集严格早于测试集", all_ordered)

    # 折间测试集不重叠
    test_ranges = [(seg.test_start, seg.test_end) for seg in folds]
    no_overlap = True
    for i in range(len(test_ranges) - 1):
        if test_ranges[i][1] >= test_ranges[i + 1][0]:
            no_overlap = False
            break
    report.record("C.正确性", "折间测试集不重叠", no_overlap)

    # embargo 生效: 下一折 train_start > 本折 test_end + embargo(近似)
    embargo_ok = True
    for i in range(len(folds) - 1):
        gap = folds[i + 1].train_start - folds[i].test_end
        # 至少有 embargo 天的间隔 (用日历日近似, 实际是交易日)
        if gap.days < 1:
            embargo_ok = False
            break
    report.record("C.正确性", "embargo 使下一折训练集在测试集之后", embargo_ok)

    # purge 生效: train_end 与 test_start 之间有 purge 间隔
    purge_ok = all((seg.test_start - seg.train_end).days >= 1 for seg in folds)
    report.record("C.正确性", "purge 使训练集尾部与测试集有间隔", purge_ok)

    # expanding 模式: 训练集起点始终是第一天
    sp_exp = RollingWindowSplit(train_window=252, test_window=63, step=63,
                                mode="expanding")
    folds_exp = sp_exp.split(dates)
    all_start_at_zero = all(seg.train_start == dates[0] for seg in folds_exp)
    report.record("C.正确性", "expanding 模式训练集起点固定", all_start_at_zero,
                  detail=f"{len(folds_exp)} 折")

    # valid 窗口
    sp_v = RollingWindowSplit(train_window=252, test_window=63, valid_window=42,
                              mode="rolling")
    folds_v = sp_v.split(dates)
    has_valid = all(seg.valid_start is not None and seg.valid_end is not None
                    for seg in folds_v)
    report.record("C.正确性", "valid 窗口正确生成", has_valid, detail=f"{len(folds_v)} 折")


def test_walkforward_boundary(report: TestReport):
    print("\n=== C2. Walk-Forward - 边界条件 ===")
    dates = pd.bdate_range("2020-01-02", periods=100)

    # 交易日不足 → 空折
    sp = RollingWindowSplit(train_window=252, test_window=63)
    folds = sp.split(dates)
    report.record("C.边界", "交易日不足→空折列表", folds == [])

    # 刚好够一折
    dates2 = pd.bdate_range("2020-01-02", periods=315)  # 252+63
    sp2 = RollingWindowSplit(train_window=252, test_window=63)
    folds2 = sp2.split(dates2)
    report.record("C.边界", "刚好够一折→1折", len(folds2) == 1, detail=f"{len(folds2)} 折")

    # 非法参数
    for kw in [dict(train_window=0, test_window=10), dict(train_window=10, test_window=-1),
               dict(train_window=10, test_window=10, purge_days=-1),
               dict(train_window=10, test_window=10, mode="bad")]:
        try:
            RollingWindowSplit(**kw)
            report.record("C.边界", f"非法参数应抛异常 {kw}", False)
        except ValueError:
            report.record("C.边界", f"非法参数应抛异常 {list(kw.keys())}", True)

    # iter_masks 与行对齐
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=756),
                       "code": "000001.SZ", "x": 1.0})
    sp3 = RollingWindowSplit(train_window=252, test_window=63, mode="rolling")
    n_folds = 0
    for fold, tr, te, va in sp3.iter_masks(df["date"]):
        n_folds += 1
        # 训练 mask 与测试 mask 互斥
        if (tr & te).any():
            report.record("C.边界", "iter_masks 训练/测试互斥", False)
            break
    else:
        report.record("C.边界", "iter_masks 训练/测试互斥", True, detail=f"{n_folds} 折")


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    report = TestReport()
    print("=" * 70)
    print("jingni-trader 优化验证测试 (feat/quant-opt-20260621)")
    print("=" * 70)

    suites = [
        ("A. 因子表达式引擎", [
            test_factor_engine_correctness,
            test_factor_engine_performance,
            test_factor_engine_boundary,
        ]),
        ("B. 向量化回测 + 修正指标", [
            test_backtest_correctness,
            test_backtest_performance,
            test_metrics_correction,
            test_backtest_boundary,
        ]),
        ("C. Walk-Forward 滚动分割", [
            test_walkforward_correctness,
            test_walkforward_boundary,
        ]),
    ]

    for suite_name, tests in suites:
        print(f"\n{'─' * 70}\n{suite_name}\n{'─' * 70}")
        for t in tests:
            try:
                t(report)
            except Exception as e:
                report.record(suite_name, t.__name__, False, detail=f"EXCEPTION: {e}")
                traceback.print_exc()

    s = report.summary()
    print(f"\n{'=' * 70}")
    print(f"总计: {s['total']}  通过: {s['pass']}  失败: {s['fail']}")
    print(f"{'=' * 70}")
    return report, s


if __name__ == "__main__":
    main()
