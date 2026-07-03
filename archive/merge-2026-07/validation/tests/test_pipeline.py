"""
Pipeline 测试
"""
import numpy as np
import pandas as pd
import pytest

from validation.pipeline import (
    CrossSectionalScaler,
    IndustryNeutralizer,
    MissingValueFiller,
    Pipeline,
    Winsorizer,
)


@pytest.fixture
def factor_panel():
    dates = pd.bdate_range("2023-01-01", periods=30)
    codes = [f"{i:06d}.SZ" for i in range(1, 11)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                "date": d, "code": c,
                "industry": "Tech" if int(c.split(".")[0]) < 5 else "Finance",
                "factor": np.random.randn() + (5 if c < "000005.SZ" else 0),
            })
    return pd.DataFrame(rows)


def test_pipeline_creation():
    p = Pipeline([
        ("imputer", MissingValueFiller(strategy="median")),
        ("winsor", Winsorizer()),
    ])
    assert len(p.steps) == 2
    assert not p._fitted


def test_missing_value_filler(factor_panel):
    df = factor_panel.copy()
    df.loc[df.sample(5, random_state=1).index, "factor"] = np.nan
    imputer = MissingValueFiller(columns=["factor"], strategy="median")
    out = imputer.fit_transform(df)
    assert out["factor"].isna().sum() == 0
    # 填充值 = 训练集的中位数
    expected = float(df["factor"].median())
    assert abs(out["factor"].median() - expected) < 1e-8


def test_winsorizer_caps_extremes(factor_panel):
    w = Winsorizer(columns=["factor"], lower=0.05, upper=0.95)
    out = w.fit_transform(factor_panel)
    # 截尾后最大值/最小值应等于原分位数
    assert out["factor"].max() == pytest.approx(factor_panel["factor"].quantile(0.95), rel=1e-6)
    assert out["factor"].min() == pytest.approx(factor_panel["factor"].quantile(0.05), rel=1e-6)


def test_cross_sectional_scaler(factor_panel):
    cs = CrossSectionalScaler(columns=["factor"], by="date")
    out = cs.fit_transform(factor_panel)
    # 截面均值应接近 0
    grp_mean = out.groupby("date")["factor"].mean()
    assert np.allclose(grp_mean.abs().max(), 0, atol=1e-6)


def test_industry_neutralizer_residual_mean(factor_panel):
    ind = IndustryNeutralizer(factor_col="factor")
    out = ind.fit_transform(factor_panel)
    # 中性化后, 每个行业内均值应接近 0
    g = out.groupby(["date", "industry"])["factor"].mean().abs()
    assert g.max() < 1e-6


def test_pipeline_no_leakage():
    """测试 Pipeline 不会在 transform 阶段重新学习统计量"""
    train = pd.DataFrame({
        "date": pd.bdate_range("2023-01-01", periods=10).repeat(5),
        "code": [f"{i}.SZ" for i in range(5)] * 10,
        "factor": np.random.RandomState(1).normal(0, 1, 50),
    })
    test = pd.DataFrame({
        "date": pd.bdate_range("2023-01-15", periods=5).repeat(5),
        "code": [f"{i}.SZ" for i in range(5)] * 5,
        "factor": np.random.RandomState(2).normal(0, 1, 25),
    })
    p = Pipeline([
        ("imputer", MissingValueFiller(columns=["factor"])),
        ("winsor", Winsorizer(columns=["factor"])),
    ])
    p.fit(train)
    # transform 测试集不应报错, 且使用训练集统计量
    out = p.transform(test)
    assert not out.isna().any().any()
    # 训练集 winsor 上下界应来自训练分位数
    expected_hi = float(train["factor"].quantile(0.99))
    assert out["factor"].max() <= expected_hi + 1e-9


def test_pipeline_transform_without_fit_raises():
    p = Pipeline([("imputer", MissingValueFiller())])
    with pytest.raises(RuntimeError):
        p.transform(pd.DataFrame({"a": [1, 2, 3]}))


def test_pipeline_empty_steps_raises():
    with pytest.raises(ValueError):
        Pipeline([])
