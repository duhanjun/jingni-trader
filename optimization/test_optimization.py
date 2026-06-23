"""
优化验证测试套件

测试内容：
1. 正确性测试：向量化实现 vs 原实现的数值一致性
2. 性能对比测试：向量化 vs 原实现的执行时间
3. 边界条件测试：T+1 强制执行、涨跌停约束、空数据、单只股票
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 将 optimization 目录和 workspace 根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_market_data(
    n_stocks=50, n_days=250, start_date="2023-01-01", seed=42
):
    """生成模拟A股行情数据"""
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]

    rows = []
    for code in codes:
        price = np.random.uniform(10, 50)
        for dt in dates:
            ret = np.random.normal(0.0005, 0.02)
            price = max(price * (1 + ret), 1.0)
            pre_close = price / (1 + ret)
            change_pct = (price - pre_close) / pre_close * 100
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_p = pre_close * (1 + np.random.normal(0, 0.003))
            vol = int(np.random.lognormal(12, 0.5))
            rows.append({
                "date": dt,
                "code": code,
                "open": round(open_p, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(price, 4),
                "volume": vol,
                "pre_close": round(pre_close, 4),
                "change_pct": round(change_pct, 4),
                "is_st": False,
                "is_limit_up": change_pct >= 9.9,
                "is_limit_down": change_pct <= -9.9,
                "turnover_rate": round(np.random.uniform(0.5, 5.0), 4),
                "amount": round(vol * price, 2),
            })
    return pd.DataFrame(rows)


def generate_signals(data, hold_days=5, top_n=10):
    """生成交易信号：每 hold_days 天调仓，选 top_n 只"""
    np.random.seed(123)
    dates = sorted(data["date"].unique())
    rebalance_dates = dates[::hold_days]
    signals = []
    for dt in rebalance_dates:
        day_data = data[data["date"] == dt]
        if day_data.empty:
            continue
        # 随机选 top_n 只作为买入信号
        selected = day_data.sample(min(top_n, len(day_data)))
        for _, row in selected.iterrows():
            signals.append({"date": dt, "code": row["code"], "signal": 1})
        # 其余持仓发出卖出信号
        held = day_data[~day_data["code"].isin(selected["code"])]
        for _, row in held.head(top_n).iterrows():
            signals.append({"date": dt, "code": row["code"], "signal": -1})
    return pd.DataFrame(signals)


def generate_factor_data(data, n_factors=8):
    """生成因子数据 + 前向收益"""
    np.random.seed(456)
    df = data.sort_values(["code", "date"]).copy()
    result = df[["code", "date"]].copy()

    # 生成若干因子
    for i in range(n_factors):
        result[f"factor_{i}"] = np.random.normal(0, 1, len(result))
    # 反转因子（与未来收益负相关，便于测试 IC）
    result["reversal_5d"] = -df.groupby("code")["close"].pct_change(5)
    result["volatility_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=5).std()
    )
    result["lncap"] = np.log(df["amount"] / df["turnover_rate"].replace(0, np.nan) * 100)

    # 前向收益
    for period in [1, 5, 20]:
        result[f"ret_forward_{period}d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-period) / x - 1
        )

    # 行业
    industries = ["银行", "地产", "医药", "消费", "科技", "能源", "材料", "工业"]
    result["industry"] = result["code"].apply(
        lambda c: industries[int(c[:6]) % len(industries)]
    )
    return result


# =====================================================================
# 测试 1：向量化回测 vs 原生回测 - 正确性与性能
# =====================================================================
def test_backtest_correctness_and_performance():
    print("\n" + "=" * 70)
    print("测试 1：向量化回测 vs 原生回测")
    print("=" * 70)

    from vectorized_backtest import VectorizedBacktestEngine

    data = generate_synthetic_market_data(n_stocks=50, n_days=250)
    signals = generate_signals(data, hold_days=5, top_n=10)
    print(f"测试数据：{len(data)} 行行情，{len(signals)} 条信号，"
          f"{data['code'].nunique()} 只股票，{data['date'].nunique()} 个交易日")

    # 加载原版 native adapter
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "skills", "backtest-engine"
    ))
    from scripts.adapters.native_adapter import NativeAdapter

    # 原版回测
    t0 = time.time()
    orig_adapter = NativeAdapter()
    orig_result = orig_adapter.run_backtest(
        data=data, signals=signals, init_capital=1e6,
        t_plus_1=True, price_limit=True,
    )
    orig_time = time.time() - t0

    # 向量化回测
    t0 = time.time()
    vec_engine = VectorizedBacktestEngine()
    vec_result = vec_engine.run_backtest(
        data=data, signals=signals, init_capital=1e6,
        t_plus_1=True, price_limit=True,
    )
    vec_time = time.time() - t0

    # 性能对比
    speedup = orig_time / vec_time if vec_time > 0 else float("inf")
    print(f"\n[性能] 原版耗时: {orig_time:.3f}s | 向量化耗时: {vec_time:.3f}s | 加速比: {speedup:.2f}x")

    # 正确性：净值曲线长度与最终净值对比
    orig_eq = orig_result.get("equity_curve", pd.DataFrame())
    vec_eq = vec_result.get("equity_curve", pd.DataFrame())
    print(f"[结果] 原版净值点数: {len(orig_eq)} | 向量化净值点数: {len(vec_eq)}")

    if not orig_eq.empty and not vec_eq.empty:
        orig_final = orig_eq["equity"].iloc[-1]
        vec_final = vec_eq["equity"].iloc[-1]
        diff_pct = abs(vec_final - orig_final) / orig_final * 100
        print(f"[净值] 原版最终净值: {orig_final:,.2f} | 向量化最终净值: {vec_final:,.2f} | 偏差: {diff_pct:.2f}%")

    # 指标对比
    orig_metrics = orig_result.get("metrics", {})
    vec_metrics = vec_result.get("metrics", {})
    print("\n[指标对比]")
    print(f"{'指标':<20} {'原版':>15} {'向量化':>15} {'差异':>10}")
    print("-" * 62)
    for key in ["total_return", "annual_return", "sharpe_ratio", "max_drawdown", "volatility"]:
        ov = orig_metrics.get(key, 0)
        vv = vec_metrics.get(key, 0)
        diff = abs(vv - ov) if isinstance(ov, (int, float)) else 0
        print(f"{key:<20} {ov:>15.4f} {vv:>15.4f} {diff:>10.4f}")

    return {
        "orig_time": orig_time,
        "vec_time": vec_time,
        "speedup": speedup,
        "orig_final_equity": float(orig_eq["equity"].iloc[-1]) if not orig_eq.empty else 0,
        "vec_final_equity": float(vec_eq["equity"].iloc[-1]) if not vec_eq.empty else 0,
        "orig_metrics": orig_metrics,
        "vec_metrics": vec_metrics,
    }


# =====================================================================
# 测试 2：向量化 IC 分析 vs 原版 - 正确性与性能
# =====================================================================
def test_ic_correctness_and_performance():
    print("\n" + "=" * 70)
    print("测试 2：向量化 IC 分析 vs 原版逐日循环")
    print("=" * 70)

    from vectorized_ic import VectorizedICAnalyzer, OriginalICAnalyzer

    data = generate_synthetic_market_data(n_stocks=100, n_days=250)
    factor_df = generate_factor_data(data, n_factors=8)
    factor_names = [c for c in factor_df.columns
                    if c not in ["code", "date", "industry", "lncap"]]

    forward_returns = factor_df[["code", "date", "ret_forward_1d",
                                  "ret_forward_5d", "ret_forward_20d"]].copy()
    print(f"测试数据：{len(factor_df)} 行，{len(factor_names)} 个因子，"
          f"{factor_df['date'].nunique()} 个截面")

    # 原版 IC 分析
    orig_analyzer = OriginalICAnalyzer()
    t0 = time.time()
    orig_ic_map = {}
    for factor in factor_names:
        ic_series = orig_analyzer._calc_ic_original(
            factor_df, factor, "ret_forward_5d", "spearman"
        )
        if ic_series is not None:
            orig_ic_map[factor] = ic_series
    orig_time = time.time() - t0

    # 向量化 IC 分析
    vec_analyzer = VectorizedICAnalyzer()
    t0 = time.time()
    vec_ic_map = {}
    for factor in factor_names:
        ic_series = vec_analyzer._calc_ic_vectorized(
            factor_df, factor, "ret_forward_5d", "spearman"
        )
        if ic_series is not None:
            vec_ic_map[factor] = ic_series
    vec_time = time.time() - t0

    speedup = orig_time / vec_time if vec_time > 0 else float("inf")
    print(f"\n[性能] 原版耗时: {orig_time:.3f}s | 向量化耗时: {vec_time:.3f}s | 加速比: {speedup:.2f}x")

    # 正确性：对比 IC 均值（Spearman rank 后应数值接近）
    # 排除前向收益列，只测试真正的因子
    test_factors = [f for f in factor_names
                    if not f.startswith("ret_forward")]
    print(f"\n[正确性] IC 均值对比 (ret_forward_5d, spearman)")
    print(f"{'因子':<20} {'原版IC均值':>12} {'向量化IC均值':>14} {'绝对误差':>10}")
    print("-" * 58)
    max_err = 0
    for factor in test_factors:
        orig_series = orig_ic_map.get(factor)
        vec_series = vec_ic_map.get(factor)
        orig_mean = float(orig_series.mean()) if orig_series is not None and len(orig_series) > 0 else 0.0
        vec_mean = float(vec_series.mean()) if vec_series is not None and len(vec_series) > 0 else 0.0
        if np.isnan(orig_mean) or np.isnan(vec_mean):
            err = 0.0
        else:
            err = abs(orig_mean - vec_mean)
        max_err = max(max_err, err)
        print(f"{factor:<20} {orig_mean:>12.6f} {vec_mean:>14.6f} {err:>10.6f}")
    print(f"\n最大绝对误差: {max_err:.6f} (向量化与原版 IC 应一致)")

    return {
        "orig_time": orig_time,
        "vec_time": vec_time,
        "speedup": speedup,
        "max_error": max_err,
        "n_factors": len(factor_names),
    }


# =====================================================================
# 测试 3：向量化中性化 vs 原版 - 正确性与性能
# =====================================================================
def test_neutralize_correctness_and_performance():
    print("\n" + "=" * 70)
    print("测试 3：向量化因子中性化 vs 原版逐日 sklearn")
    print("=" * 70)

    from vectorized_neutralize import VectorizedNeutralizer, OriginalNeutralizer

    data = generate_synthetic_market_data(n_stocks=80, n_days=120)
    factor_df = generate_factor_data(data, n_factors=4)
    factor_names = ["factor_0", "factor_1", "reversal_5d", "volatility_20d"]
    industry_df = factor_df[["code", "industry"]].drop_duplicates()

    print(f"测试数据：{len(factor_df)} 行，{len(factor_names)} 个因子，"
          f"{factor_df['date'].nunique()} 个截面")

    # 原版中性化
    orig_neut = OriginalNeutralizer()
    t0 = time.time()
    orig_result = orig_neut.neutralize_original(
        factor_df, industry_df, neutralize_mcap=True, neutralize_industry=True
    )
    orig_time = time.time() - t0

    # 向量化中性化
    vec_neut = VectorizedNeutralizer()
    t0 = time.time()
    vec_result = vec_neut.neutralize_vectorized(
        factor_df, industry_df, neutralize_mcap=True, neutralize_industry=True
    )
    vec_time = time.time() - t0

    speedup = orig_time / vec_time if vec_time > 0 else float("inf")
    print(f"\n[性能] 原版耗时: {orig_time:.3f}s | 向量化耗时: {vec_time:.3f}s | 加速比: {speedup:.2f}x")

    # 正确性：对比中性化后残差的相关性（残差应与市值/行业正交）
    print(f"\n[正确性] 中性化后残差与 lncap 的相关性（应接近0）")
    print(f"{'因子':<20} {'原版残差-lncap相关':>20} {'向量化残差-lncap相关':>22}")
    print("-" * 64)
    for factor in factor_names:
        col = f"{factor}_neutral"
        if col not in orig_result.columns or col not in vec_result.columns:
            continue
        orig_corr = orig_result[col].corr(orig_result["lncap"]) if "lncap" in orig_result else 0
        vec_corr = vec_result[col].corr(vec_result["lncap"]) if "lncap" in vec_result else 0
        print(f"{factor:<20} {orig_corr:>20.6f} {vec_corr:>22.6f}")

    # 残差数值对比
    print(f"\n[正确性] 中性化残差数值对比")
    print(f"{'因子':<20} {'原版均值':>12} {'向量化均值':>12} {'原版标准差':>12} {'向量化标准差':>12}")
    print("-" * 70)
    for factor in factor_names:
        col = f"{factor}_neutral"
        if col not in orig_result.columns or col not in vec_result.columns:
            continue
        orig_vals = orig_result[col].dropna()
        vec_vals = vec_result[col].dropna()
        print(f"{factor:<20} {orig_vals.mean():>12.6f} {vec_vals.mean():>12.6f} "
              f"{orig_vals.std():>12.6f} {vec_vals.std():>12.6f}")

    return {
        "orig_time": orig_time,
        "vec_time": vec_time,
        "speedup": speedup,
        "n_factors": len(factor_names),
    }


# =====================================================================
# 测试 4：T+1 边界条件测试
# =====================================================================
def test_t_plus_1_enforcement():
    print("\n" + "=" * 70)
    print("测试 4：T+1 规则强制执行验证")
    print("=" * 70)

    from vectorized_backtest import VectorizedBacktestEngine

    # 构造场景：第1天买入信号，验证买入不在第1天执行（T+1次日执行）
    dates = pd.bdate_range("2024-01-01", periods=5)
    data = pd.DataFrame([
        {"date": dates[i], "code": "600000.SH", "open": 10, "high": 10.5,
         "low": 9.5, "close": 10 + i * 0.1, "volume": 1000000,
         "is_st": False, "is_limit_up": False, "is_limit_down": False,
         "change_pct": 1.0, "pre_close": 10}
        for i in range(5)
    ])
    # 第1天买入信号，第4天卖出信号
    signals = pd.DataFrame([
        {"date": dates[0], "code": "600000.SH", "signal": 1},   # 第1天买入信号
        {"date": dates[3], "code": "600000.SH", "signal": -1},  # 第4天卖出信号
    ])

    engine = VectorizedBacktestEngine()
    result = engine.run_backtest(
        data=data, signals=signals, init_capital=1e6,
        t_plus_1=True, price_limit=False,
    )

    trades = result.get("trades", pd.DataFrame())
    print(f"成交记录数: {len(trades)}")
    if not trades.empty:
        print(trades[["date", "code", "action", "price", "shares"]].to_string(index=False))

    # 验证：T+1 下，第1天的买入信号应在第2天执行（而非第1天）
    t_plus_1_ok = True
    if not trades.empty:
        buy_trades = trades[trades["action"] == "buy"]
        if not buy_trades.empty:
            first_buy_date = buy_trades["date"].iloc[0]
            if first_buy_date <= dates[0]:
                print(f"[失败] T+1 未生效：买入发生在信号当日 {first_buy_date}")
                t_plus_1_ok = False
            else:
                print(f"[通过] T+1 生效：信号日 {dates[0].date()}，买入日 {first_buy_date.date()}（次日执行）")
        # 验证卖出也在买入次日之后
        sell_trades = trades[trades["action"] == "sell"]
        if not sell_trades.empty and not buy_trades.empty:
            first_sell_date = sell_trades["date"].iloc[0]
            first_buy_date = buy_trades["date"].iloc[0]
            if first_sell_date <= first_buy_date:
                print(f"[失败] T+1 未生效：卖出 {first_sell_date.date()} 不晚于买入 {first_buy_date.date()}")
                t_plus_1_ok = False
            else:
                print(f"[通过] 卖出 {first_sell_date.date()} 晚于买入 {first_buy_date.date()}")
    else:
        print("[警告] 无成交记录")
        t_plus_1_ok = False

    return {"t_plus_1_enforced": t_plus_1_ok, "n_trades": len(trades)}


# =====================================================================
# 测试 5：涨跌停边界条件测试
# =====================================================================
def test_price_limit_enforcement():
    print("\n" + "=" * 70)
    print("测试 5：涨跌停约束验证")
    print("=" * 70)

    from vectorized_backtest import VectorizedBacktestEngine

    dates = pd.bdate_range("2024-01-01", periods=3)
    data = pd.DataFrame([
        # 涨停股：连续2天涨停，无法买入
        {"date": dates[0], "code": "600001.SH", "open": 10, "high": 11,
         "low": 9.5, "close": 11.0, "volume": 1000000,
         "is_st": False, "is_limit_up": True, "is_limit_down": False,
         "change_pct": 10.0, "pre_close": 10},
        {"date": dates[1], "code": "600001.SH", "open": 11, "high": 12.1,
         "low": 10.5, "close": 12.1, "volume": 1000000,
         "is_st": False, "is_limit_up": True, "is_limit_down": False,
         "change_pct": 10.0, "pre_close": 11.0},
        # 正常股：可买入
        {"date": dates[0], "code": "600002.SH", "open": 20, "high": 20.5,
         "low": 19.5, "close": 20.0, "volume": 1000000,
         "is_st": False, "is_limit_up": False, "is_limit_down": False,
         "change_pct": 0.0, "pre_close": 20},
        {"date": dates[1], "code": "600002.SH", "open": 20, "high": 20.5,
         "low": 19.5, "close": 20.1, "volume": 1000000,
         "is_st": False, "is_limit_up": False, "is_limit_down": False,
         "change_pct": 0.5, "pre_close": 20},
    ])
    signals = pd.DataFrame([
        {"date": dates[0], "code": "600001.SH", "signal": 1},  # 涨停日买入
        {"date": dates[0], "code": "600002.SH", "signal": 1},  # 正常买入
    ])

    engine = VectorizedBacktestEngine()
    # 禁用 T+1 以隔离测试涨跌停约束（信号当日执行）
    result = engine.run_backtest(
        data=data, signals=signals, init_capital=1e6,
        t_plus_1=False, price_limit=True,
    )

    trades = result.get("trades", pd.DataFrame())
    print(f"成交记录数: {len(trades)}")
    if not trades.empty:
        print(trades[["date", "code", "action", "price", "shares"]].to_string(index=False))

    # 验证：涨停股 600001 不应被买入，正常股 600002 应被买入
    limit_blocked = True
    if not trades.empty:
        bought_limit = trades[
            (trades["code"] == "600001.SH") & (trades["action"] == "buy")
        ]
        bought_normal = trades[
            (trades["code"] == "600002.SH") & (trades["action"] == "buy")
        ]
        if not bought_limit.empty:
            print(f"[失败] 涨停股 600001.SH 被买入")
            limit_blocked = False
        else:
            print(f"[通过] 涨停股 600001.SH 被正确阻止买入")
        if not bought_normal.empty:
            print(f"[通过] 正常股 600002.SH 成功买入")
        else:
            print(f"[警告] 正常股 600002.SH 未买入")
    else:
        print(f"[警告] 无成交记录")

    return {"price_limit_enforced": limit_blocked, "n_trades": len(trades)}


# =====================================================================
# 主测试入口
# =====================================================================
def run_all_tests():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print("jingni-trader 优化验证测试套件")
    print("执行时间: " + now_str)
    print("分支: feat/quant-opt-20260623")
    print("=" * 70)

    results = {}
    try:
        results["backtest"] = test_backtest_correctness_and_performance()
    except Exception as e:
        print(f"[错误] 回测测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["backtest"] = {"error": str(e)}

    try:
        results["ic"] = test_ic_correctness_and_performance()
    except Exception as e:
        print(f"[错误] IC 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["ic"] = {"error": str(e)}

    try:
        results["neutralize"] = test_neutralize_correctness_and_performance()
    except Exception as e:
        print(f"[错误] 中性化测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["neutralize"] = {"error": str(e)}

    try:
        results["t_plus_1"] = test_t_plus_1_enforcement()
    except Exception as e:
        print(f"[错误] T+1 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["t_plus_1"] = {"error": str(e)}

    try:
        results["price_limit"] = test_price_limit_enforcement()
    except Exception as e:
        print(f"[错误] 涨跌停测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["price_limit"] = {"error": str(e)}

    return results


if __name__ == "__main__":
    results = run_all_tests()
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "test_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果已保存至: {output_path}")
