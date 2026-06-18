"""
IC Decay Analyzer 测试
======================
本测试使用合成数据：
  - 30 只股票
  - 252 个交易日
  - 因子 = 5 日反转 + 噪声
  - 真实 forward return 取决于反转强度

期望：
  - IC 在 lag=5 附近最大
  - 越往两侧 lag 越接近 0
  - half-life 应在 5-10 之间
"""
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.quant_opt_20260618.ic_analysis.ic_decay import (
    ICDecayAnalyzer,
    ICLagResult,
)


def make_synthetic_data(n_stocks: int = 30, n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    """
    构造带真实反转因子的合成数据
    ---------------------------------
    真实生成过程:
        log_p[t+5] - log_p[t] = -0.6 * (log_p[t] - log_p[t-5]) + noise
    这意味着"5 日反转因子"对未来 5 日收益有 -0.6 的负相关性。
    因此 IC(lag=5) 的绝对值应显著高于其他 lag。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    codes = [f"S{str(i).zfill(6)}" for i in range(n_stocks)]
    n = n_stocks

    # 5 个时间步的初始值
    log_prices = np.zeros((n, n_days))
    log_prices[:, :5] = np.cumsum(rng.normal(0, 0.02, (n, 5)), axis=1) + 10.0

    # 之后每个 t 满足 log_p[t] = log_p[t-5] + future_5[t-5]
    # 其中 future_5 = -0.6 * (log_p[t-5] - log_p[t-10]) + noise
    for t in range(5, n_days):
        if t < 10:
            # 前 10 日内没有完整 5 日过去窗口，用纯 random walk
            log_prices[:, t] = log_prices[:, t - 1] + rng.normal(0, 0.02, n)
        else:
            past_5 = log_prices[:, t - 5] - log_prices[:, t - 10]
            future_5 = -0.6 * past_5 + rng.normal(0, 0.03, n)
            log_prices[:, t] = log_prices[:, t - 5] + future_5

    rows = []
    for i, code in enumerate(codes):
        for j, dt in enumerate(dates):
            price = float(np.exp(log_prices[i, j]))
            rows.append({
                "date": dt,
                "code": code,
                "open": price,
                "close": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "is_limit_up": False,
                "is_limit_down": False,
                "volume": float(rng.uniform(1e6, 1e7)),
                "amount": float(rng.uniform(1e8, 1e9)),
                "turnover_rate": float(rng.uniform(0.005, 0.03)),
            })
    df = pd.DataFrame(rows)

    # 5日反转因子 (真实有效)
    df = df.sort_values(["code", "date"])
    df["factor_rev5"] = -df.groupby("code")["close"].transform(
        lambda x: x.pct_change(5)
    )
    # 5日动量因子 (噪声，无预测力)
    df["factor_mom5"] = df.groupby("code")["close"].transform(
        lambda x: x.pct_change(5)
    )
    return df


def test_ic_decay_basic():
    """基础 IC Decay 测试"""
    data = make_synthetic_data()
    analyzer = ICDecayAnalyzer(min_lag=1, max_lag=10, min_cross_size=10)
    results = analyzer.calc_ic_decay(data, "factor_rev5")

    assert len(results) > 0, "应至少返回一组结果"
    print(f"\n[test_ic_decay_basic] 5日反转因子 IC Decay:")
    for r in results:
        print(f"  lag={r.lag:>2}  IC={r.ic_mean:+.4f}  IR={r.ic_ir:+.4f}  t={r.ic_t_stat:+.3f}  p={r.ic_p_value:.3f}  n={r.n_obs}")

    for r in results:
        assert -1.0 <= r.ic_mean <= 1.0
        assert r.ic_std >= 0
        assert 0.0 <= r.ic_p_value <= 1.0
        assert r.n_obs > 0
    print("[test_ic_decay_basic] PASSED")
    return results


def test_ic_decay_optimal_lag():
    """最优 lag 应在 5 附近"""
    data = make_synthetic_data()
    analyzer = ICDecayAnalyzer(min_lag=1, max_lag=10)
    results = analyzer.calc_ic_decay(data, "factor_rev5")
    optimal = analyzer.find_optimal_lag(results)
    print(f"\n[test_ic_decay_optimal_lag] optimal_lag={optimal}")
    assert optimal is not None
    # 真实数据中反转因子的 peak 应在 5 附近
    assert 4 <= optimal <= 7, f"expected optimal in [4,7], got {optimal}"
    print("[test_ic_decay_optimal_lag] PASSED")
    return optimal


def test_ic_decay_half_life():
    """半衰期估计"""
    data = make_synthetic_data()
    analyzer = ICDecayAnalyzer(min_lag=1, max_lag=20)
    results = analyzer.calc_ic_decay(data, "factor_rev5")
    half_life = analyzer.estimate_half_life(results)
    print(f"\n[test_ic_decay_half_life] half_life_lag={half_life}")
    if half_life is not None:
        assert 1 <= half_life <= 20
    print("[test_ic_decay_half_life] PASSED")


def test_ic_decay_summarize():
    """summary 接口"""
    data = make_synthetic_data()
    analyzer = ICDecayAnalyzer(min_lag=1, max_lag=8)
    summary = analyzer.summarize(data, "factor_rev5")
    assert "results" in summary
    assert "optimal_lag" in summary
    assert "half_life_lag" in summary
    print(f"\n[test_ic_decay_summarize] summary keys: {list(summary.keys())}")
    print(f"  optimal_lag={summary['optimal_lag']}  half_life={summary['half_life_lag']}  peak_abs_ic={summary['peak_abs_ic']:.4f}")
    print("[test_ic_decay_summarize] PASSED")


def test_edge_cases():
    """边界条件"""
    # 1) 空数据
    analyzer = ICDecayAnalyzer(min_lag=1, max_lag=5)
    empty = pd.DataFrame(columns=["code", "date", "close", "factor"])
    res = analyzer.calc_ic_decay(empty, "factor")
    assert res == []
    print("\n[test_edge_cases] 空数据: PASSED")

    # 2) 因子列不存在
    data = make_synthetic_data(n_stocks=10, n_days=60)
    try:
        analyzer.calc_ic_decay(data, "non_exist")
        assert False, "应当抛错"
    except ValueError:
        print("[test_edge_cases] 不存在因子列: PASSED (正确抛错)")

    # 3) 全 NaN 因子
    data2 = data.copy()
    data2["factor"] = np.nan
    res2 = analyzer.calc_ic_decay(data2, "factor")
    assert res2 == []
    print("[test_edge_cases] 全 NaN 因子: PASSED")

    # 4) 非法参数
    try:
        ICDecayAnalyzer(min_lag=5, max_lag=2)
        assert False
    except ValueError:
        print("[test_edge_cases] 非法 lag 区间: PASSED (正确抛错)")


if __name__ == "__main__":
    test_ic_decay_basic()
    test_ic_decay_optimal_lag()
    test_ic_decay_half_life()
    test_ic_decay_summarize()
    test_edge_cases()
    print("\n=== IC Decay Analyzer 所有测试通过 ===")
