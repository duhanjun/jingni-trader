"""
跨模块性能/正确性对比测试

验证目标：
1. 表达式引擎与 jingni-trader 现有硬编码因子在相同因子上结果完全一致
2. 表达式引擎在大数据量下的性能
3. 三套优化组合在一起（DSL 因子 + TopK 选股 + Walk-Forward 评估）的端到端流程
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "factor_expression_engine"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "topk_dropout_strategy"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "walk_forward_validation"))

from factor_expression_engine import FactorExpressionEngine
from topk_dropout_strategy import TopKDropoutStrategy
from walk_forward_validation import RollingSplit, WalkForwardRunner


# ---------------------------------------------------------------------------
# 合成大数据
# ---------------------------------------------------------------------------


def make_large_panel(n_dates: int = 1000, n_codes: int = 100) -> pd.DataFrame:
    """构造大面板数据：1000 日 × 100 股 = 10 万行。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    codes = [f"{i:06d}.SH" for i in range(1, n_codes + 1)]
    rows = []
    for code in codes:
        base = 10 + rng.normal(0, 1)
        rets = rng.normal(0, 0.02, size=n_dates)
        closes = base * np.exp(np.cumsum(rets))
        volumes = rng.integers(1_000_000, 5_000_000, size=n_dates).astype(float)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": closes[i] * (1 + rng.normal(0, 0.001)),
                "high": closes[i] * 1.005,
                "low": closes[i] * 0.995,
                "close": closes[i],
                "volume": volumes[i],
                "amount": volumes[i] * closes[i],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. 表达式引擎 vs 硬编码因子一致性测试
# ---------------------------------------------------------------------------


def test_expression_engine_matches_pandas_baseline():
    """对同样公式，表达式引擎结果与手写 pandas 结果应一致。"""
    df = make_large_panel(n_dates=300, n_codes=30)
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    eng.register("vol_10", "Ts_Std($close, 10) / Ts_Mean($close, 10)")
    eng.register("rev_20", "Sub(0, Delta($close, 20))")
    out = eng.compute_all(df)
    # 验证 mom_5
    expected_mom = (
        df["close"] / df.groupby("code")["close"].shift(5) - 1
    )
    np.testing.assert_allclose(
        out["mom_5"].values, expected_mom.values, equal_nan=True
    )
    # 验证 vol_10
    mean_10 = df.groupby("code")["close"].transform(
        lambda s: s.rolling(10, min_periods=2).mean()
    )
    std_10 = df.groupby("code")["close"].transform(
        lambda s: s.rolling(10, min_periods=2).std()
    )
    expected_vol = std_10 / mean_10
    np.testing.assert_allclose(
        out["vol_10"].values, expected_vol.values, equal_nan=True
    )


# ---------------------------------------------------------------------------
# 2. 性能测试
# ---------------------------------------------------------------------------


def test_performance_1_2x_factors_100k_rows():
    """大数据量下计算 2 个常见因子的耗时。"""
    df = make_large_panel(n_dates=1000, n_codes=100)
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    eng.register("rev_20", "Sub(0, Delta($close, 20))")

    t0 = time.perf_counter()
    for _ in range(3):
        out = eng.compute_all(df)
    elapsed = (time.perf_counter() - t0) / 3
    print(f"\n[perf] 100k rows × 2 factors 平均耗时: {elapsed*1000:.1f}ms")
    # 宽松阈值：< 5 秒
    assert elapsed < 5.0
    assert len(out) == 100_000
    assert out["mom_5"].notna().sum() > 0
    assert out["rev_20"].notna().sum() > 0


def test_performance_complex_alpha_101_style():
    """类 Alpha101 复合公式性能。"""
    df = make_large_panel(n_dates=500, n_codes=50)
    eng = FactorExpressionEngine()
    eng.register(
        "alpha_composite",
        "Rank(Mul(Sign(Delta($close, 5)), Abs(Ts_Mean($close, 20))))",
    )
    t0 = time.perf_counter()
    for _ in range(3):
        out = eng.compute_all(df)
    elapsed = (time.perf_counter() - t0) / 3
    print(f"\n[perf] 25k rows × 1 alpha-101 因子: {elapsed*1000:.1f}ms")
    assert elapsed < 3.0


# ---------------------------------------------------------------------------
# 3. 边界条件
# ---------------------------------------------------------------------------


def test_edge_case_empty_dataframe():
    """空 DataFrame 应优雅返回空 Series，不抛异常。"""
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    empty = pd.DataFrame({"code": [], "date": [], "close": []})
    out = eng.compute("mom_5", empty)
    assert isinstance(out, pd.Series)
    assert len(out) == 0


def test_edge_case_missing_column_raises():
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    bad = pd.DataFrame({"code": ["A"], "date": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(KeyError):
        eng.compute("mom_5", bad)


def test_edge_case_single_stock():
    df = pd.DataFrame({
        "code": ["A"] * 10,
        "date": pd.bdate_range("2024-01-01", periods=10),
        "close": np.arange(10, 20, dtype=float),
    })
    eng = FactorExpressionEngine()
    eng.register("mom_3", "$close / Delay($close, 3) - 1")
    out = eng.compute_all(df)
    assert out["mom_3"].iloc[:3].isna().all()  # 前 3 个无效
    assert not out["mom_3"].iloc[3:].isna().any()


def test_edge_case_constant_close():
    """价格不变时动量应为 0，波动率应为 0。"""
    df = pd.DataFrame({
        "code": ["A"] * 20,
        "date": pd.bdate_range("2024-01-01", periods=20),
        "close": [10.0] * 20,
    })
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    eng.register("vol_5", "Ts_Std($close, 5)")
    out = eng.compute_all(df)
    assert (out["mom_5"].dropna() == 0).all()
    assert (out["vol_5"].dropna() == 0).all()


# ---------------------------------------------------------------------------
# 4. 端到端集成：DSL 因子 → TopK 选股 → Walk-Forward 评估
# ---------------------------------------------------------------------------


def test_e2e_factor_to_walkforward_to_topk():
    """端到端：DSL 因子 → walk-forward 评估 IC → 调仓选股。"""
    df = make_large_panel(n_dates=600, n_codes=30)
    # 1) 用 DSL 计算未来 1 日收益
    eng = FactorExpressionEngine()
    eng.register("mom_5", "$close / Delay($close, 5) - 1")
    factor = eng.compute("mom_5", df)
    panel = df[["code", "date", "close"]].copy()
    panel["alpha_score"] = factor.values
    panel["fwd_ret_1d"] = (
        panel.groupby("code")["close"].pct_change().shift(-1)
    )
    panel = panel.dropna(subset=["alpha_score", "fwd_ret_1d"])

    # 2) Walk-Forward 评估 IC
    splitter = RollingSplit(
        train_period=200, valid_period=40, test_period=40, step=40
    )
    runner = WalkForwardRunner(splitter)
    from scipy.stats import spearmanr

    def fit_fn(train, valid):
        return {}

    def evaluate_fn(model, test):
        sub = test.dropna(subset=["alpha_score", "fwd_ret_1d"])
        if len(sub) < 5:
            return {"ic": 0.0, "n": len(sub)}
        ic, _ = spearmanr(sub["alpha_score"], sub["fwd_ret_1d"])
        return {"ic": float(ic), "n": len(sub)}

    ic_df = runner.run(panel, fit_fn, evaluate_fn, date_col="date")
    assert len(ic_df) >= 3
    print(f"\n[e2e] walk-forward IC folds: {ic_df['test_ic'].mean():.4f}")

    # 3) TopKDropout 选股（用最后一个测试日的分数）
    last_test_date = ic_df.iloc[-1]["test_end"]
    scores = panel[panel["date"] == last_test_date][["code", "alpha_score"]]
    if not scores.empty:
        strat = TopKDropoutStrategy(top_k=5, n_dropout=2, weight_method="score")
        holdings = strat.rebalance([], scores)
        assert len(holdings) == 5
        assert holdings["weight"].sum() == pytest.approx(1.0, abs=1e-9)
        # 得分最高者应获得最大权重
        top_code = scores.sort_values("alpha_score", ascending=False).iloc[0]["code"]
        assert holdings.set_index("code").loc[top_code, "weight"] == holdings["weight"].max()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))