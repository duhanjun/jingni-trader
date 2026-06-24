"""Factor DSL 单元测试"""
import numpy as np
import pandas as pd
import pytest

from quant_opt_20260618_r3.factor_dsl import FactorEngine, FactorError


@pytest.fixture
def sample_panel():
    """构造 3 只票 30 天的样本数据"""
    dates = pd.bdate_range("2024-01-02", periods=30)
    rows = []
    rng = np.random.default_rng(0)
    for c in ["AAA", "BBB", "CCC"]:
        close = 10 + np.cumsum(rng.normal(0, 0.5, 30))
        volume = rng.normal(1e6, 2e5, 30)
        for i, d in enumerate(dates):
            rows.append({
                "code": c, "date": d,
                "close": close[i], "volume": volume[i],
                "open": close[i] * (1 + rng.normal(0, 0.001)),
            })
    return pd.DataFrame(rows)


def test_register_and_compute_simple(sample_panel):
    engine = FactorEngine()
    engine.register("f1", "Ts_Mean(close, 5)")
    out = engine.compute(sample_panel, ["f1"])
    assert list(out.columns) == ["code", "date", "f1"]
    assert len(out) == 90
    # min_periods=2: 早期每只票第一个值 NaN, 第二个开始就有值
    for c in ["AAA", "BBB", "CCC"]:
        sub = out[out["code"] == c].sort_values("date")
        assert sub["f1"].iloc[0] != sub["f1"].iloc[0]  # NaN check
        assert sub["f1"].iloc[1] == sub["f1"].iloc[1]  # 非 NaN
    # 后期应无 NaN
    for c in ["AAA", "BBB", "CCC"]:
        sub = out[out["code"] == c].sort_values("date")
        assert sub["f1"].iloc[-5:].isna().sum() == 0


def test_rank_cross_sectional(sample_panel):
    engine = FactorEngine()
    engine.register("r", "Rank(close)")
    out = engine.compute(sample_panel, ["r"])
    # 每日 rank 应在 (0, 1] 区间
    for d, grp in out.groupby("date"):
        assert grp["r"].between(0, 1).all()
        # 3 只票应取 1/3, 2/3, 1
        assert set(grp["r"].round(4)) == {0.3333, 0.6667, 1.0}


def test_composite_alpha101_like(sample_panel):
    """模拟 Alpha101 风格公式: Rank(Delta(close, 5)) * -1"""
    engine = FactorEngine()
    engine.register("alpha", "-1 * Rank(Delta(close, 5))")
    out = engine.compute(sample_panel, ["alpha"])
    assert "alpha" in out.columns
    # 前 5 日应为 NaN
    for c in ["AAA", "BBB", "CCC"]:
        sub = out[out["code"] == c].sort_values("date")
        assert sub["alpha"].isna().sum() == 5


def test_decay_linear_vs_simple_mean(sample_panel):
    """Decay_Linear 应与简单均值不同 (权重不等)"""
    engine = FactorEngine()
    engine.register("dl", "Decay_Linear(close, 5)")
    engine.register("mean", "Ts_Mean(close, 5)")
    out = engine.compute(sample_panel, ["dl", "mean"])
    diff = (out["dl"] - out["mean"]).abs().sum()
    assert diff > 0  # 二者不应完全相等


def test_register_illegal_builtin():
    engine = FactorEngine()
    with pytest.raises(FactorError):
        engine.register("bad", "eval('1+1')")


def test_register_syntax_error():
    engine = FactorEngine()
    with pytest.raises(FactorError):
        engine.register("bad", "Ts_Mean(close, )")


def test_register_unknown_function():
    engine = FactorEngine()
    with pytest.raises(FactorError):
        engine.register("bad", "NonExist(close)")


def test_zscore_zero_mean(sample_panel):
    engine = FactorEngine()
    engine.register("z", "ZScore(close)")
    out = engine.compute(sample_panel, ["z"])
    for d, grp in out.groupby("date"):
        # 横截面 z-score 均值应接近 0
        assert abs(grp["z"].mean()) < 1e-6


def test_scale_formula_sign_only(sample_panel):
    """Sign(close - Delay(close, 1)) 应为 +1 / -1 / NaN"""
    engine = FactorEngine()
    engine.register("s", "Sign(close - Delay(close, 1))")
    out = engine.compute(sample_panel, ["s"])
    for c in ["AAA", "BBB", "CCC"]:
        sub = out[out["code"] == c].sort_values("date")
        valid = sub["s"].dropna()
        # 唯一取值: -1, 0, 1
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})


def test_compose_multiple_factors_one_pass(sample_panel):
    engine = FactorEngine()
    engine.register_many({
        "m5": "Ts_Mean(close, 5)",
        "v5": "Ts_Std(close, 5)",
        "delta": "Delta(close, 3) / Delay(close, 3)",
    })
    out = engine.compute(sample_panel)
    assert {"code", "date", "m5", "v5", "delta"}.issubset(out.columns)