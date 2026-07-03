"""
OPTIMIZATION 3 (part 1) 验证：向量化 IC 分析
==============================================
测试内容：
(a) 正确性：calc_ic_vectorized vs calc_ic_original，IC Series 一致（忽略 NaN）
(b) 性能：两者耗时对比，打印加速比

运行：python tests/test_vectorized_ic.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data_generator import generate_test_data
from vectorized_ic import calc_ic_original, calc_ic_vectorized


def _build_factor_df(n_stocks=60, n_days=200, seed=99):
    """构造含 factor 与 forward 收益列的 DataFrame"""
    data, _ = generate_test_data(n_stocks=n_stocks, n_days=n_days, seed=seed)
    df = data.sort_values(["code", "date"]).reset_index(drop=True)
    # 因子：5 日均值减 20 日均值（带反转意味）
    df["factor"] = (
        df.groupby("code")["close"].transform(lambda x: x.rolling(5, min_periods=1).mean())
        - df.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    )
    # forward 5 日收益
    df["forward"] = df.groupby("code")["close"].shift(-5) / df["close"] - 1
    return df


def test_correctness():
    print("\n=== [3a] IC 正确性：原始 vs 向量化 ===")
    df = _build_factor_df(n_stocks=60, n_days=200, seed=99)
    ic_o = calc_ic_original(df, "factor", "forward")
    ic_v = calc_ic_vectorized(df, "factor", "forward")

    print(f"  original  IC 点数: {len(ic_o)}, mean={ic_o.mean():.4f}")
    print(f"  vectorized IC 点数: {len(ic_v)}, mean={ic_v.mean():.4f}")

    # 索引应一致
    common = ic_o.index.intersection(ic_v.index)
    assert len(common) == len(ic_o) == len(ic_v), (
        f"IC 点数不一致 orig={len(ic_o)} vec={len(ic_v)} common={len(common)}"
    )
    diff = (ic_o.loc[common].to_numpy() - ic_v.loc[common].to_numpy())
    max_diff = float(np.nanmax(np.abs(diff)))
    print(f"  IC 最大绝对差: {max_diff:.2e}")
    assert np.allclose(ic_o.loc[common], ic_v.loc[common], rtol=1e-9, atol=1e-9), \
        f"IC 不一致, max diff={max_diff}"
    print("  [PASS] 向量化 IC 与原始一致")


def test_performance():
    print("\n=== [3b] IC 性能：n_stocks=100, n_days=300 ===")
    df = _build_factor_df(n_stocks=100, n_days=300, seed=2024)
    print(f"  数据行数: {len(df)}")

    t0 = time.perf_counter()
    ic_o = calc_ic_original(df, "factor", "forward")
    t_orig = time.perf_counter() - t0

    t0 = time.perf_counter()
    ic_v = calc_ic_vectorized(df, "factor", "forward")
    t_vec = time.perf_counter() - t0

    # 性能测试也保证一致
    common = ic_o.index.intersection(ic_v.index)
    assert np.allclose(ic_o.loc[common], ic_v.loc[common], rtol=1e-9, atol=1e-9)

    speedup = t_orig / t_vec if t_vec > 0 else float("inf")
    print(f"  原始耗时:   {t_orig:.3f}s")
    print(f"  向量化耗时: {t_vec:.3f}s")
    print(f"  加速比:     {speedup:.2f}x")
    assert t_vec < t_orig, f"向量化应更快 (orig={t_orig}, vec={t_vec})"
    print("  [PASS] 向量化 IC 更快且结果一致")
    return {"orig": t_orig, "vec": t_vec, "speedup": speedup}


def test_missing_column():
    print("\n=== [3c] 边界：缺失列返回空 Series ===")
    df = _build_factor_df(n_stocks=10, n_days=40, seed=1)
    o = calc_ic_original(df, "not_exist", "forward")
    v = calc_ic_vectorized(df, "not_exist", "forward")
    assert o.empty and v.empty
    print("  [PASS] 缺失列返回空 Series")


def run_all():
    test_correctness()
    perf = test_performance()
    test_missing_column()
    print("\n=== 全部 IC 测试通过 ===")
    return perf


if __name__ == "__main__":
    run_all()
