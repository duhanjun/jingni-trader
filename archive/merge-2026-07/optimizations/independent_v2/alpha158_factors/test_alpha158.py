"""
Alpha158 因子库与 IC 分析模块测试

验证内容:
1. 因子计算正确性: 所有因子都能计算且产出合理值
2. 因子覆盖完整性: get_available_factors 返回预期数量
3. IC 分析正确性: 对已知有效因子（动量）应判定为有效
4. IC 分析边界: 空数据、单只股票等
5. 性能: 60+ 因子 × 多股票应在合理时间内完成
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from optimizations.independent_v2.alpha158_factors import Alpha158Calculator, ICAnalyzer
from optimizations.independent_v2.data_fixtures import make_synthetic_ohlcv


# ---------- 因子计算正确性 ----------

def test_all_factors_calculable():
    """所有声明的因子都应能成功计算，不抛异常。"""
    data = make_synthetic_ohlcv(n_codes=3, n_days=120, seed=1)
    calc = Alpha158Calculator()
    all_factors = calc.get_available_factors()
    assert len(all_factors) >= 50, f"因子数量过少: {len(all_factors)}"

    result = calc.calculate(data, all_factors)
    for f in all_factors:
        assert f in result.columns, f"因子 {f} 未出现在结果中"
        # 至少应有部分非 NaN 值（前 n 日 warmup 期为 NaN 是正常的）
        non_null = result[f].notna().sum()
        assert non_null > 0, f"因子 {f} 全为 NaN"


def test_kline_factors_correctness():
    """K线形态因子计算公式应正确。"""
    data = make_synthetic_ohlcv(n_codes=1, n_days=10, seed=2)
    calc = Alpha158Calculator()
    result = calc.calculate(data, ["KMID", "KLEN"])

    # KMID = (close - open) / open
    expected_kmid = (data["close"] - data["open"]) / data["open"]
    np.testing.assert_allclose(
        result["KMID"].values, expected_kmid.values, rtol=1e-10
    )

    # KLEN = (high - low) / open
    expected_klen = (data["high"] - data["low"]) / data["open"]
    np.testing.assert_allclose(
        result["KLEN"].values, expected_klen.values, rtol=1e-10
    )


def test_trend_factor_ma5_correctness():
    """MA5 因子应为 5 日收盘均价 / 当日收盘。"""
    data = make_synthetic_ohlcv(n_codes=1, n_days=30, seed=3)
    calc = Alpha158Calculator()
    result = calc.calculate(data, ["MA5"])

    close = data.sort_values("date")["close"].values
    for i in range(4, 30):
        expected = close[i - 4:i + 1].mean() / close[i]
        actual = result.sort_values("date")["MA5"].values[i]
        if not np.isnan(actual):
            np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_factor_info_returns_metadata():
    """get_factor_info 应返回因子元信息。"""
    calc = Alpha158Calculator()
    info = calc.get_factor_info("KMID")
    assert info["group"] == "kline"
    assert info["normalized_by"] == "close"

    info = calc.get_factor_info("ROC20")
    assert info["group"] == "trend"

    info = calc.get_factor_info("NONEXISTENT")
    assert info == {}


# ---------- IC 分析正确性 ----------

def test_ic_analysis_momentum_effective():
    """对已知有效的动量因子，IC 分析应判定为有效。

    构造一个强动量数据集：过去 20 日收益高的股票，未来 5 日收益也高。
    """
    rng = np.random.default_rng(42)
    n_codes = 30
    n_days = 200
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_codes + 1)]

    rows = []
    # 每只股票有不同的漂移率，构造持续动量
    drifts = rng.normal(0.001, 0.0005, n_codes)
    for ci, code in enumerate(codes):
        rets = rng.normal(drifts[ci], 0.01, n_days)
        prices = 10.0 * np.exp(np.cumsum(rets))
        for i, dt in enumerate(dates):
            rows.append({
                "code": code, "date": dt, "close": float(prices[i]),
                "open": float(prices[i]), "high": float(prices[i] * 1.005),
                "low": float(prices[i] * 0.995), "volume": 1e6,
            })
    data = pd.DataFrame(rows)

    # 计算 20 日动量因子
    data = data.sort_values(["code", "date"])
    data["MOM20"] = data.groupby("code")["close"].transform(
        lambda s: s.pct_change(20)
    )
    factor_df = data[["code", "date", "MOM20"]].dropna()

    analyzer = ICAnalyzer()
    report = analyzer.analyze(factor_df, data, "MOM20", periods=[1, 5, 10])

    assert "by_period" in report
    assert len(report["by_period"]) == 3
    # 动量因子在构造的数据集上应被判定为有效
    assert report["is_effective"], (
        f"动量因子应被判定有效: {report['recommendation']}"
    )
    # 最佳持有期的 RankIC 均值应为正
    best = report["by_period"][report["best_period"]]
    assert best["rank_ic_summary"]["ic_mean"] > 0


def test_ic_analysis_random_factor_not_effective():
    """对纯随机因子，IC 分析应判定为无效。"""
    rng = np.random.default_rng(99)
    data = make_synthetic_ohlcv(n_codes=20, n_days=150, seed=7)
    data = data.sort_values(["code", "date"])
    # 注入纯随机因子
    data["RANDOM"] = rng.normal(0, 1, len(data))

    analyzer = ICAnalyzer()
    report = analyzer.analyze(
        data[["code", "date", "RANDOM"]], data, "RANDOM", periods=[1, 5]
    )
    # 随机因子应被判定为无效
    assert not report["is_effective"], (
        f"随机因子不应被判定有效: {report['recommendation']}"
    )


def test_ic_summary_empty_series():
    """空 IC 序列应返回零值摘要，不崩溃。"""
    analyzer = ICAnalyzer()
    summary = analyzer.calc_ic_summary(pd.Series(dtype=float))
    assert summary["ic_mean"] == 0.0
    assert summary["n_days"] == 0


def test_forward_returns_correctness():
    """未来收益计算应正确。"""
    data = make_synthetic_ohlcv(n_codes=1, n_days=30, seed=11)
    analyzer = ICAnalyzer()
    result = analyzer.calc_forward_returns(data, periods=[1, 5])

    close = data.sort_values("date")["close"].values
    # 第 0 天的 forward_return_1 = close[1]/close[0] - 1
    expected = close[1] / close[0] - 1
    actual = result.sort_values("date")["forward_return_1"].values[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


# ---------- 性能测试 ----------

def test_factor_calculation_performance():
    """60+ 因子 × 10 股票 × 250 天应在 5 秒内完成。"""
    data = make_synthetic_ohlcv(n_codes=10, n_days=250, seed=33)
    calc = Alpha158Calculator()
    all_factors = calc.get_available_factors()

    t0 = time.perf_counter()
    calc.calculate(data, all_factors)
    elapsed = time.perf_counter() - t0
    print(f"\n[perf] {len(all_factors)} factors × 10 stocks × 250 days: "
          f"{elapsed*1000:.1f}ms")
    assert elapsed < 5.0, f"因子计算过慢: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
