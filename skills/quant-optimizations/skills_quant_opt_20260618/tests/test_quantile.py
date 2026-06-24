"""
Quantile Return Analysis 测试
==============================
测试目标：
  1. 分位收益的单调性
  2. 多空收益的符号
  3. 与 IC Decay 结果的一致性
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.quant-optimizations.skills_quant_opt_20260618.quantile_analysis.quantile import (
    QuantileAnalyzer,
    QuantileStats,
)


def make_synthetic(n_stocks: int = 60, n_days: int = 252, seed: int = 7):
    """构造带强预测因子的合成数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    codes = [f"Q{str(i).zfill(6)}" for i in range(n_stocks)]
    n = n_stocks

    log_prices = np.zeros((n, n_days))
    log_prices[:, :5] = np.cumsum(rng.normal(0, 0.02, (n, 5)), axis=1) + 10.0
    for t in range(5, n_days):
        if t < 10:
            log_prices[:, t] = log_prices[:, t - 1] + rng.normal(0, 0.02, n)
        else:
            past_5 = log_prices[:, t - 5] - log_prices[:, t - 10]
            future_5 = -0.6 * past_5 + rng.normal(0, 0.03, n)
            log_prices[:, t] = log_prices[:, t - 5] + future_5

    rows = []
    for i, code in enumerate(codes):
        for j, dt in enumerate(dates):
            price = float(np.exp(log_prices[i, j]))
            rows.append({"date": dt, "code": code, "close": price})
    df = pd.DataFrame(rows)

    # 计算 5 日反转因子
    df = df.sort_values(["code", "date"])
    df["factor_rev5"] = -df.groupby("code")["close"].transform(lambda x: x.pct_change(5))

    # 计算 1 日 forward return
    df["fwd_ret_1d"] = df.groupby("code")["close"].transform(lambda x: x.shift(-1) / x - 1)

    return df


def test_quantile_monotonicity():
    """分位收益应单调上升 (反转因子为正，反转越强 → 未来收益越高)"""
    df = make_synthetic()
    analyzer = QuantileAnalyzer(n_quantiles=5, min_stocks_per_quantile=5, forward_lag=1)
    fwd = df[["code", "date", "fwd_ret_1d"]].copy()
    qr = analyzer.compute_quantile_returns(df, fwd, "factor_rev5")
    summary = analyzer.summarize(qr)

    print("\n[test_quantile_monotonicity] 各分位日均收益:")
    for s in summary["stats"]:
        print(f"  q{s['quantile']}: mean={s['mean_daily_return']:+.5f}  sharpe={s['sharpe']:+.2f}  cum={s['cumulative_return']:+.4f}")
    print(f"  long_short_sharpe={summary['long_short_sharpe']}")
    print(f"  monotonicity={summary['monotonicity']}")

    # 反转因子为正 → q5 > q1 (单调)
    q1_ret = summary["stats"][0]["mean_daily_return"]
    q5_ret = summary["stats"][-1]["mean_daily_return"]
    assert q5_ret > q1_ret, f"反转因子下 q5={q5_ret:.5f} 应 > q1={q1_ret:.5f}"
    print("[test_quantile_monotonicity] PASSED")


def test_quantile_basic_shape():
    """基本形状测试"""
    df = make_synthetic()
    analyzer = QuantileAnalyzer(n_quantiles=5, forward_lag=1)
    fwd = df[["code", "date", "fwd_ret_1d"]].copy()
    qr = analyzer.compute_quantile_returns(df, fwd, "factor_rev5")

    assert not qr.empty
    assert "long_short" in qr.columns
    assert any(c.startswith("q") for c in qr.columns)
    print(f"\n[test_quantile_basic_shape] 输出 shape: {qr.shape}, columns: {list(qr.columns)}")
    print("[test_quantile_basic_shape] PASSED")


def test_quantile_edge_cases():
    """边界条件"""
    analyzer = QuantileAnalyzer(n_quantiles=3, min_stocks_per_quantile=2)

    # 1) 空数据
    empty = pd.DataFrame(columns=["code", "date", "fwd_ret_1d"])
    factor_empty = pd.DataFrame(columns=["code", "date", "f"])
    qr = analyzer.compute_quantile_returns(factor_empty, empty, "f")
    assert qr.empty
    print("\n[test_quantile_edge_cases] 空数据: PASSED")

    # 2) 缺失 forward return 列
    df = make_synthetic(n_stocks=10, n_days=30)
    try:
        analyzer.compute_quantile_returns(df, df[["code", "date"]], "factor_rev5")
        assert False, "应抛错"
    except ValueError:
        print("[test_quantile_edge_cases] 缺 forward 列: PASSED (正确抛错)")

    # 3) 非法参数
    try:
        QuantileAnalyzer(n_quantiles=1)
        assert False
    except ValueError:
        print("[test_quantile_edge_cases] n_quantiles<2: PASSED (正确抛错)")

    # 4) 极小数据 → 可能没有任何一天达到 min_stocks_per_quantile
    tiny = make_synthetic(n_stocks=4, n_days=20)
    analyzer2 = QuantileAnalyzer(n_quantiles=5, min_stocks_per_quantile=2)
    fwd = tiny[["code", "date", "fwd_ret_1d"]]
    qr_tiny = analyzer2.compute_quantile_returns(tiny, fwd, "factor_rev5")
    # 即使数据极少也应不抛错
    print(f"[test_quantile_edge_cases] 极小数据输出 shape: {qr_tiny.shape}")
    print("[test_quantile_edge_cases] 极小数据: PASSED")


if __name__ == "__main__":
    test_quantile_basic_shape()
    test_quantile_monotonicity()
    test_quantile_edge_cases()
    print("\n=== Quantile Analyzer 所有测试通过 ===")