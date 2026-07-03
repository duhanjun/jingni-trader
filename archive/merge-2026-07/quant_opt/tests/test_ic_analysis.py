"""
test_ic_analysis.py
===================

向量化 IC 分析的单元测试。

测试覆盖：
- batch_ic_analysis 基础功能
- 已知信号的 IC 应非零
- 多 forward 周期
- 边界条件：空数据、单日
- 分位数收益分析的单调性
"""
from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "factor"))

from ic_analysis_vectorized import (
    batch_ic_analysis, quantile_returns_analysis, rolling_ic,
    _cross_section_spearman,
)


def test_spearman_perfect():
    """Spearman 在完全相关时为 1.0。"""
    f = np.array([1.0, 2, 3, 4, 5])
    r = np.array([10.0, 20, 30, 40, 50])
    assert math.isclose(_cross_section_spearman(f, r), 1.0, abs_tol=1e-6)


def test_spearman_anti():
    """Spearman 在完全负相关时为 -1.0。"""
    f = np.array([1.0, 2, 3, 4, 5])
    r = np.array([50.0, 40, 30, 20, 10])
    assert math.isclose(_cross_section_spearman(f, r), -1.0, abs_tol=1e-6)


def test_spearman_with_nan():
    """Spearman 在有 NaN 时应跳过。"""
    f = np.array([1.0, 2, np.nan, 4, 5])
    r = np.array([10.0, 20, 30, 40, 50])
    # 去除 NaN 后应该是 1.0
    assert math.isclose(_cross_section_spearman(f, r), 1.0, abs_tol=1e-6)


def test_batch_ic_with_known_signal():
    """已知动量信号 IC 应 > 0.1。"""
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "date": pd.to_datetime(np.sort(rng.choice(pd.bdate_range("2020-01-01", periods=500), n))),
        "code": rng.choice([f"{i:06d}.SH" for i in range(100)], n),
        "good_factor": rng.normal(0, 1, n),
    })
    # forward return 与 factor 相关
    df["ret_forward_5d"] = df["good_factor"] * 0.5 + rng.normal(0, 0.5, n)
    res = batch_ic_analysis(df, df, ["good_factor"], ["ret_forward_5d"])
    assert "ret_forward_5d" in res
    assert len(res["ret_forward_5d"]) == 1
    r = res["ret_forward_5d"][0]
    assert r["ic_mean"] > 0.2, f"IC should be > 0.2, got {r['ic_mean']}"
    assert r["ic_ir"] > 0.5
    print(f"  Known signal: IC={r['ic_mean']:.4f}, IR={r['ic_ir']:.4f}")


def test_batch_ic_random_signal():
    """随机信号 IC 应接近 0。"""
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame({
        "date": pd.to_datetime(np.sort(rng.choice(pd.bdate_range("2020-01-01", periods=500), n))),
        "code": rng.choice([f"{i:06d}.SH" for i in range(100)], n),
        "noise_factor": rng.normal(0, 1, n),
    })
    df["ret_forward_5d"] = rng.normal(0, 0.5, n)  # 与 factor 独立
    res = batch_ic_analysis(df, df, ["noise_factor"], ["ret_forward_5d"])
    r = res["ret_forward_5d"][0]
    assert abs(r["ic_mean"]) < 0.1, f"IC should be near 0, got {r['ic_mean']}"
    print(f"  Random signal: IC={r['ic_mean']:.4f} (expect ~0)")


def test_quantile_returns_monotonic():
    """分位数收益应单调 (强信号下)。"""
    rng = np.random.default_rng(2)
    n = 5000
    df = pd.DataFrame({
        "date": pd.to_datetime(np.sort(rng.choice(pd.bdate_range("2020-01-01", periods=500), n))),
        "code": rng.choice([f"{i:06d}.SH" for i in range(100)], n),
        "factor": rng.normal(0, 1, n),
    })
    df["ret_forward_5d"] = df["factor"] * 0.5 + rng.normal(0, 0.3, n)
    q = quantile_returns_analysis(df, df, "factor", "ret_forward_5d", quantiles=5)
    means = q["mean_ret"].drop("long_short", errors="ignore")
    # 平均收益应大致单调递增
    diffs = np.diff(means.values)
    # 至少 3/4 个 diff > 0
    assert (diffs > 0).sum() >= 3
    print(f"  Quantile mean ret:\n{means}")
    # Long-short 收益应为正
    if "long_short" in q.index:
        assert q.loc["long_short", "mean_ret"] > 0


def test_quantile_returns_empty():
    """空数据应优雅处理。"""
    df = pd.DataFrame(columns=["date", "code", "factor", "ret_forward_5d"])
    q = quantile_returns_analysis(df, df, "factor")
    assert q.empty


def test_rolling_ic():
    """滚动 IC 序列。"""
    rng = np.random.default_rng(3)
    n = 200
    f = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2020-01-01", periods=n))
    r = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2020-01-01", periods=n))
    out = rolling_ic(f, r, window=60)
    assert len(out) == n - 60 + 1
    assert out.dropna().abs().max() < 1.01  # 滚动相关系数
    print(f"  Rolling IC: {len(out)} points, range=[{out.min():.3f}, {out.max():.3f}]")


def test_batch_ic_performance():
    """IC 性能测试: 100 因子 × 500 日期 × 100 股票。"""
    rng = np.random.default_rng(4)
    n_stocks = 100
    n_dates = 500
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for dt in dates:
        for code_i in range(n_stocks):
            code = f"{code_i:06d}.SH"
            row = {"date": dt, "code": code}
            for fi in range(20):
                row[f"f{fi}"] = rng.normal(0, 1)
            row["ret_forward_1d"] = rng.normal(0, 0.02)
            row["ret_forward_5d"] = rng.normal(0, 0.04)
            rows.append(row)
    df = pd.DataFrame(rows)
    factor_names = [f"f{i}" for i in range(20)]
    forward_cols = ["ret_forward_1d", "ret_forward_5d"]
    import time
    t0 = time.perf_counter()
    res = batch_ic_analysis(df, df, factor_names, forward_cols, min_obs=20)
    elapsed = time.perf_counter() - t0
    total_calcs = sum(len(v) for v in res.values())
    print(f"  IC batch: {total_calcs} (factor, forward) pairs in {elapsed:.3f}s "
          f"({total_calcs / max(elapsed, 0.001):.0f} pairs/s)")
    # 至少跑出结果
    assert total_calcs > 0
    for fc, lst in res.items():
        print(f"  {fc}: {len(lst)} factors, "
              f"best IC={max(r['ic_mean'] for r in lst):.4f}")


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
