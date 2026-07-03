"""
性能测试：对比新模块与原实现的执行时间

测试维度：
1. 因子计算：Polars 向量化 vs pandas groupby.transform(lambda)
2. IC 分析：Polars groupby.agg vs Python 逐日循环 + scipy
3. 回测引擎：NumPy 向量化 vs Python 逐日循环
"""
from __future__ import annotations

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.quant_opt_20260620.factor_engine_polars import (
    PolarsFactorEngine,
    FactorDef,
    vectorized_ic_analysis,
)
from experiments.quant_opt_20260620.backtest_vectorized import (
    VectorizedBacktester,
    BacktestConfig,
)
from experiments.quant_opt_20260620.tests.data_gen import make_synthetic_data, make_signals


def _timeit(fn, *args, **kwargs):
    """简单计时器，返回 (结果, 耗时秒)"""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def bench_factor_engine():
    """因子计算性能对比"""
    print("\n[性能] 因子计算引擎对比")
    print("-" * 60)

    for n_codes, n_days in [(100, 500), (500, 500), (1000, 250)]:
        data = make_synthetic_data(n_codes=n_codes, n_days=n_days, seed=42)
        print(f"\n  数据规模: {n_codes} 只股票 × {n_days} 天 = {len(data)} 行")

        # ---- 原 pandas 实现 ----
        def orig_compute(data):
            df = data.sort_values(["code", "date"]).copy()
            result = df[["code", "date"]].copy()
            result["ret_5d"] = df.groupby("code")["close"].pct_change(5)
            result["reversal_5d"] = -result["ret_5d"]
            result["volatility_20d"] = df.groupby("code")["close"].transform(
                lambda x: x.pct_change().rolling(20, min_periods=10).std()
            )
            result["turnover_20d"] = df.groupby("code")["turnover_rate"].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result["volume_20d"] = df.groupby("code")["volume"].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result["vol_ratio"] = df["volume"] / result["volume_20d"].replace(0, np.nan)
            return result

        _, t_orig = _timeit(orig_compute, data)

        # ---- 新 Polars 实现 ----
        factors = [
            FactorDef("rev_5d", "Ts_Delta(Close, 5)", direction=-1),
            FactorDef("vol_20d", "Ts_Std(Ts_Ref(Close, 0) / Ts_Ref(Close, 1) - 1, 20)"),
            FactorDef("turnover_20d", "Ts_Mean(Turnover, 20)"),
            FactorDef("vol_ratio", "Volume / Ts_Mean(Volume, 20)"),
        ]
        engine = PolarsFactorEngine(factors=factors)
        # warmup（首次调用有 Polars 启动开销）
        _ = engine.compute(data.head(100))
        _, t_new = _timeit(engine.compute, data)

        speedup = t_orig / t_new if t_new > 0 else float("inf")
        print(f"    原实现 (pandas): {t_orig*1000:.1f} ms")
        print(f"    新实现 (Polars): {t_new*1000:.1f} ms")
        print(f"    加速比: {speedup:.2f}x")


def bench_ic_analysis():
    """IC 分析性能对比"""
    print("\n[性能] IC 分析引擎对比")
    print("-" * 60)

    from scipy import stats

    data = make_synthetic_data(n_codes=200, n_days=500, seed=42)
    df = data.sort_values(["code", "date"]).copy()

    factor_df = df[["code", "date"]].copy()
    factor_df["rev_5d"] = -df.groupby("code")["close"].pct_change(5)
    factor_df["vol_20d"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )

    fwd = df[["code", "date"]].copy()
    fwd["ret_forward_5d"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-5) / x - 1
    )

    merged = factor_df.merge(fwd, on=["code", "date"])

    # ---- 原 scipy 逐日循环 ----
    def orig_ic(merged):
        ic_list = []
        for dt, cross in merged.groupby("date"):
            valid = cross.dropna(subset=["rev_5d", "ret_forward_5d"])
            if len(valid) < 10:
                continue
            ic, _ = stats.spearmanr(valid["rev_5d"], valid["ret_forward_5d"])
            if not np.isnan(ic):
                ic_list.append(ic)
        return float(np.mean(ic_list)) if ic_list else 0.0

    _, t_orig = _timeit(orig_ic, merged)

    # ---- 新 Polars 向量化 ----
    def new_ic(factor_df, fwd):
        return vectorized_ic_analysis(factor_df, fwd, factor_names=["rev_5d", "vol_20d"], ic_type="spearman")

    _, t_new = _timeit(new_ic, factor_df, fwd)

    speedup = t_orig / t_new if t_new > 0 else float("inf")
    print(f"  数据规模: {len(merged)} 行, {merged['date'].nunique()} 个日期")
    print(f"    原实现 (scipy 逐日循环): {t_orig*1000:.1f} ms")
    print(f"    新实现 (Polars 向量化):  {t_new*1000:.1f} ms")
    print(f"    加速比: {speedup:.2f}x")


def bench_backtest():
    """回测引擎性能对比"""
    print("\n[性能] 回测引擎对比")
    print("-" * 60)

    for n_codes, n_days in [(100, 250), (500, 250), (1000, 250)]:
        data = make_synthetic_data(n_codes=n_codes, n_days=n_days, seed=42)
        signals = make_signals(data)
        print(f"\n  数据规模: {n_codes} 只股票 × {n_days} 天")

        # ---- 原 native_adapter 实现（精简版）----
        def orig_backtest(data, signals):
            data = data.sort_values(["date", "code"]).reset_index(drop=True)
            signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
            dates = sorted(signals["date"].unique())
            cash = 1e6
            positions = {}
            equity_records = []
            for dt in dates:
                day_signal = signals[signals["date"] == dt]
                day_data = data[data["date"] == dt]
                if day_data.empty:
                    continue
                day_data_map = day_data.set_index("code")
                sell_codes, buy_codes = [], []
                for _, row in day_signal.iterrows():
                    sig = row.get("signal", 0)
                    if float(sig) > 0:
                        buy_codes.append(row["code"])
                    elif float(sig) < 0:
                        sell_codes.append(row["code"])
                for code in sell_codes:
                    if code not in positions or positions[code] <= 0:
                        continue
                    if code not in day_data_map.index:
                        continue
                    price = day_data_map.loc[code, "close"]
                    shares = positions[code]
                    sell_amount = price * shares
                    commission = max(sell_amount * 0.00025, 5)
                    tax = sell_amount * 0.001
                    cash += sell_amount - commission - tax
                    positions[code] = 0
                if buy_codes:
                    budget = cash * 0.95 / len(buy_codes)
                    for code in buy_codes:
                        if code not in day_data_map.index:
                            continue
                        price = day_data_map.loc[code, "close"] * 1.001
                        shares = int(budget / price / 100) * 100
                        if shares <= 0:
                            continue
                        amount = price * shares
                        commission = max(amount * 0.00025, 5)
                        cost = amount + commission
                        if cost > cash:
                            continue
                        cash -= cost
                        positions[code] = positions.get(code, 0) + shares
                mv = sum(s * day_data_map.loc[c, "close"] for c, s in positions.items() if s > 0 and c in day_data_map.index)
                equity_records.append({"date": dt, "equity": cash + mv})
            return pd.DataFrame(equity_records)

        _, t_orig = _timeit(orig_backtest, data, signals)

        # ---- 新向量化回测 ----
        bt = VectorizedBacktester(BacktestConfig(init_capital=1e6))
        # warmup
        _ = bt.run(data.head(100), signals.head(100))
        _, t_new = _timeit(bt.run, data, signals)

        speedup = t_orig / t_new if t_new > 0 else float("inf")
        print(f"    原实现 (Python 逐日循环): {t_orig*1000:.1f} ms")
        print(f"    新实现 (NumPy 向量化):   {t_new*1000:.1f} ms")
        print(f"    加速比: {speedup:.2f}x")


if __name__ == "__main__":
    bench_factor_engine()
    bench_ic_analysis()
    bench_backtest()
    print("\n=== 性能测试完成 ===")
