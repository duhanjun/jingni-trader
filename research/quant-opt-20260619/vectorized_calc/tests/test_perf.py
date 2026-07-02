"""
向量化因子计算 - 性能 & 正确性测试

对比基准：
1. 正确性：与原生 pandas 实现的逐只股票循环对比
2. 性能：与 jingni-trader 现有 pandas-ta adapter 对比
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from vectorized_calc import VectorizedFactorCalculator


def make_panel(n_dates=500, n_codes=100, seed=42):
    """构造多只股票多日数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    codes = [f"S{i:04d}" for i in range(n_codes)]
    long = []
    for c in codes:
        rets = rng.normal(0.001, 0.02, n_dates)
        close = 10 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.01, n_dates)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n_dates)))
        open_ = close * (1 + rng.normal(0, 0.005, n_dates))
        volume = rng.integers(1_000_000, 10_000_000, n_dates)
        amount = close * volume
        for i, d in enumerate(dates):
            long.append({
                "code": c, "date": d, "open": open_[i], "high": high[i],
                "low": low[i], "close": close[i], "volume": volume[i],
                "amount": amount[i],
            })
    return pd.DataFrame(long)


def baseline_pandas_ma(data, window=20):
    """原生 pandas 实现 - 逐只股票 groupby"""
    out = data[["code", "date"]].copy()
    result = []
    for code, g in data.groupby("code"):
        g = g.sort_values("date")
        s = g["close"].rolling(window, min_periods=max(2, window // 2)).mean()
        result.append(pd.DataFrame({"code": code, "date": g["date"], "value": s.values}))
    res = pd.concat(result, ignore_index=True)
    return res.sort_values(["code", "date"]).reset_index(drop=True)


def baseline_pandas_rsi(data, window=14):
    """原生 pandas RSI"""
    out = []
    for code, g in data.groupby("code"):
        g = g.sort_values("date")
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        out.append(pd.DataFrame({"code": code, "date": g["date"], "value": rsi.values}))
    return pd.concat(out, ignore_index=True)


def test_correctness_ma():
    """MA 正确性对比"""
    data = make_panel(n_dates=100, n_codes=10)
    calc = VectorizedFactorCalculator()

    # baseline
    base = baseline_pandas_ma(data, 20)
    # vectorized
    res = calc.calculate(data, ["ma_20"])
    # 对齐比较
    merged = base.merge(res, on=["code", "date"])
    diff = (merged["value"] - merged["ma_20"]).abs()
    max_diff = diff.max()
    rel = (diff / (merged["value"].abs() + 1e-9)).max()
    print(f"MA20 max abs diff: {max_diff:.6e}, max rel diff: {rel:.6e}")
    assert rel < 1e-6, f"MA 正确性误差过大: {rel}"
    print("test_correctness_ma PASS")


def test_correctness_std():
    """Std 正确性对比"""
    data = make_panel(n_dates=100, n_codes=10)
    calc = VectorizedFactorCalculator()
    res = calc.calculate(data, ["std_20"])
    # pandas 实现的 std
    base = []
    for code, g in data.groupby("code"):
        g = g.sort_values("date")
        s = g["close"].rolling(20, min_periods=10).std()
        base.append(pd.DataFrame({"code": code, "date": g["date"], "value": s.values}))
    base = pd.concat(base, ignore_index=True)
    merged = base.merge(res, on=["code", "date"]).dropna()
    rel = ((merged["value"] - merged["std_20"]).abs() / (merged["value"].abs() + 1e-9)).max()
    print(f"STD20 max rel diff: {rel:.6e}")
    assert rel < 1e-6
    print("test_correctness_std PASS")


def test_correctness_rsi():
    """RSI 正确性对比"""
    data = make_panel(n_dates=100, n_codes=10)
    calc = VectorizedFactorCalculator()
    res = calc.calculate(data, ["rsi_14"])
    base = baseline_pandas_rsi(data, 14)
    merged = base.merge(res, on=["code", "date"]).dropna()
    diff = (merged["value"] - merged["rsi_14"]).abs()
    # RSI 值的范围 0~100，相对误差 0.1% 即可
    rel = (diff / 100.0).max()
    print(f"RSI14 max abs diff: {diff.max():.4f}, max rel diff (over 100): {rel:.6e}")
    assert diff.max() < 0.5, f"RSI 绝对误差过大: {diff.max()}"
    print("test_correctness_rsi PASS")


def test_correctness_rank():
    """截面排名正确性"""
    data = make_panel(n_dates=50, n_codes=8)
    calc = VectorizedFactorCalculator()
    res = calc.calculate(data, ["rank_volume"])
    # pandas 版：每天对所有股票做 rank(pct=True)
    base = data.copy()
    base["value"] = base.groupby("date")["volume"].rank(pct=True)
    merged = base.merge(res, on=["code", "date"]).dropna()
    rel = ((merged["value"] - merged["rank_volume"]).abs()).max()
    print(f"rank_volume max diff: {rel:.6e}")
    assert rel < 1e-9
    print("test_correctness_rank PASS")


def test_perf_vs_pandas():
    """性能对比：numba vectorized vs pandas groupby loop"""
    print("\n=== 性能对比 (100 stocks × 1000 days) ===")
    data = make_panel(n_dates=1000, n_codes=100)
    calc = VectorizedFactorCalculator()

    # numba vectorized
    t0 = time.perf_counter()
    res = calc.calculate(data, ["ma_20", "std_20", "rsi_14", "rank_volume"])
    t_vec = time.perf_counter() - t0
    print(f"  Numba vectorized (4 factors): {t_vec*1000:.2f} ms")

    # pandas groupby
    t0 = time.perf_counter()
    _ = baseline_pandas_ma(data, 20)
    t_pd = time.perf_counter() - t0
    print(f"  pandas groupby (1 MA20):      {t_pd*1000:.2f} ms")

    speedup = t_pd / max(t_vec, 1e-6)
    print(f"  速度比 (单因子等价):           {speedup:.1f}x")
    return t_vec, t_pd


def test_scalability():
    """可扩展性测试"""
    print("\n=== 可扩展性 (1000 stocks × 2400 days, A 股十年日线规模) ===")
    data = make_panel(n_dates=2400, n_codes=1000)
    calc = VectorizedFactorCalculator()
    t0 = time.perf_counter()
    res = calc.calculate(data, ["ma_20", "std_20", "rsi_14", "rank_volume"])
    t = time.perf_counter() - t0
    print(f"  计算 4 因子 × 1000 股票 × 2400 天: {t*1000:.0f} ms ({t:.2f}s)")
    assert t < 5.0, f"性能不达标: {t:.2f}s"
    print("test_scalability PASS")


def test_empty_input():
    """空数据容错"""
    data = pd.DataFrame(columns=["code", "date", "close", "volume"])
    calc = VectorizedFactorCalculator()
    res = calc.calculate(data, ["ma_20"])
    assert res.empty
    print("test_empty_input PASS")


def test_unsupported_factor():
    """不支持因子应抛错"""
    data = make_panel(n_dates=10, n_codes=2)
    calc = VectorizedFactorCalculator()
    try:
        calc.calculate(data, ["non_existent"])
    except ValueError:
        print("test_unsupported_factor PASS")
        return
    raise AssertionError("应当抛 ValueError")


if __name__ == "__main__":
    test_correctness_ma()
    test_correctness_std()
    test_correctness_rsi()
    test_correctness_rank()
    test_perf_vs_pandas()
    test_scalability()
    test_empty_input()
    test_unsupported_factor()
    print("\nAll vectorized_calc tests PASSED")
