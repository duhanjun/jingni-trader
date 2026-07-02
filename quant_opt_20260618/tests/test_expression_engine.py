"""
因子表达式引擎单元测试
"""
import math

import numpy as np
import pandas as pd
import pytest

from quant_opt_20260618.factor_engine import (
    calc_factor,
    calc_factors,
    parse_formula,
    list_operators,
    FormulaError,
)
from quant_opt_20260618.tests.fixtures import make_synthetic_ashare_data


# ─── 算子注册表 ───────────────────────────────────────────
def test_list_operators_contains_basics():
    ops = {op.name for op in list_operators()}
    for name in ["Ts_Mean", "Ts_Std", "Delay", "Delta", "Rank", "Demean", "Scale", "Abs", "Log", "Sign", "If"]:
        assert name in ops, f"missing operator {name}"


# ─── 公式解析 ────────────────────────────────────────────
def test_parse_simple_arith():
    tree = parse_formula("1 + 2 * 3")
    assert tree == ("binop", "+", ("num", 1.0), ("binop", "*", ("num", 2.0), ("num", 3.0)))


def test_parse_function_call():
    tree = parse_formula("Ts_Mean($close, 5)")
    assert tree == ("call", "Ts_Mean", [("col", "$close"), ("num", 5.0)])


def test_parse_nested():
    tree = parse_formula("Rank(Ts_Mean($close, 5))")
    assert tree == ("call", "Rank", [("call", "Ts_Mean", [("col", "$close"), ("num", 5.0)])])


def test_parse_unknown_op_raises_at_eval_time():
    # 解析阶段不会报错（未知算子保留为 token），但 eval 时报错
    tree = parse_formula("Unknown_Fn($close, 1)")
    df = make_synthetic_ashare_data(n_stocks=2, n_days=10)
    with pytest.raises(FormulaError):
        calc_factor(df, "Unknown_Fn($close, 1)")


# ─── 因子计算正确性 ───────────────────────────────────────
def test_calc_ts_mean_vs_pandas():
    df = make_synthetic_ashare_data(n_stocks=5, n_days=80)
    s = calc_factor(df, "Ts_Mean($close, 20)")
    # 与 pandas rolling 对照
    expected = df.sort_values(["code", "date"]).groupby("code")["close"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )
    s_sorted = s.loc[df.sort_values(["code", "date"]).index]
    pd.testing.assert_series_equal(
        s_sorted.fillna(-1).reset_index(drop=True),
        expected.fillna(-1).reset_index(drop=True),
        check_names=False,
    )


def test_calc_delta_matches_reference():
    df = make_synthetic_ashare_data(n_stocks=3, n_days=40)
    s = calc_factor(df, "Delta($close, 5)")
    expected = df.sort_values(["code", "date"]).groupby("code")["close"].diff(5)
    s_sorted = s.loc[df.sort_values(["code", "date"]).index]
    pd.testing.assert_series_equal(
        s_sorted.fillna(-1).reset_index(drop=True),
        expected.fillna(-1).reset_index(drop=True),
        check_names=False,
    )


def test_calc_rank_cross_section():
    df = make_synthetic_ashare_data(n_stocks=8, n_days=30)
    s = calc_factor(df, "Rank($close)")
    # 截面 rank pct 必须在 [0, 1]
    valid = s.dropna()
    assert valid.between(0, 1).all(), "截面 rank 必须在 [0, 1]"
    # 对于 N 只股票，rank pct 均值应为 (N+1) / (2N)
    # N=8 → 0.5625
    expected_mean = (8 + 1) / (2 * 8)
    daily_means = s.groupby(df["date"]).mean()
    assert abs(daily_means.mean() - expected_mean) < 0.02, (
        f"截面 rank 均值偏离 {expected_mean}: {daily_means.mean()}"
    )


def test_calc_demean_cross_section():
    df = make_synthetic_ashare_data(n_stocks=6, n_days=20)
    s = calc_factor(df, "Demean($close)")
    # 截面 demean 后，每天均值应该接近 0
    daily_means = s.groupby(df["date"]).mean()
    assert abs(daily_means.mean()) < 0.01, f"截面 demean 日均偏离 0: {daily_means.mean()}"


def test_calc_scale_cross_section():
    df = make_synthetic_ashare_data(n_stocks=4, n_days=20)
    s = calc_factor(df, "Scale($close)")
    # 截面 Scale 后，每天 abs 和应该接近 1（仅在 close > 0 时）
    daily_abs_sums = s.abs().groupby(df["date"]).sum()
    valid = daily_abs_sums.dropna()
    assert (abs(valid - 1.0) < 0.05).mean() > 0.8


def test_calc_if_logic():
    """If(cond, a, b) 算子的基础可用性测试"""
    df = make_synthetic_ashare_data(n_stocks=3, n_days=20)
    # 直接传 boolean 列：volume 在 fixture 中始终非 0
    s = calc_factor(df, "If($volume, 1, -1)")
    assert (s == 1).all()


def test_calc_factor_with_math():
    df = make_synthetic_ashare_data(n_stocks=3, n_days=20)
    s = calc_factor(df, "Log($close)")
    assert s.dropna().iloc[0] == pytest.approx(math.log(df["close"].iloc[0]))


def test_calc_batch_factors():
    df = make_synthetic_ashare_data(n_stocks=4, n_days=40)
    out = calc_factors(df, {
        "mom_5":   "Ts_Mean($close, 5)",
        "delta1":  "Delta($close, 1)",
        "rank_c":  "Rank($close)",
        "alpha1":  "Rank(Delta($close, 5))",
    })
    assert {"mom_5", "delta1", "rank_c", "alpha1"}.issubset(out.columns)
    assert out.shape[0] == df.shape[0]


def test_alpha101_like_formula():
    """
    Alpha101 风格的多算子组合：模拟一个简单动量反转因子
    Rank(Corr(Ts_Mean($close, 10), $volume, 5)) 在我们当前算子集
    中没有 Corr，所以用更接近的组合：
    Rank(Ts_Mean($close, 10)) - Rank(Ts_Mean($volume, 10))
    """
    df = make_synthetic_ashare_data(n_stocks=5, n_days=60)
    s = calc_factor(df, "Rank(Ts_Mean($close, 10)) - Rank(Ts_Mean($volume, 10))")
    valid = s.dropna()
    # 值域应在 [-1, 1]（两 rank 相减）
    assert valid.between(-1.01, 1.01).all(), "rank 差值应在 [-1, 1]"


# ─── 边界 / 异常 ────────────────────────────────────────
def test_missing_required_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(ValueError):
        calc_factor(df, "Ts_Mean($close, 2)")


def test_unknown_column():
    df = make_synthetic_ashare_data(n_stocks=2, n_days=10)
    # 解析阶段就会失败（不识别 $unknown）
    with pytest.raises(FormulaError):
        calc_factor(df, "Ts_Mean($unknown, 5)")


def test_duplicate_factor_name_overwrite():
    df = make_synthetic_ashare_data(n_stocks=2, n_days=30)
    out = calc_factors(df, {"x": "Ts_Mean($close, 5)", "x": "Delta($close, 1)"})
    # 字典后值覆盖前值
    expected = df.sort_values(["code", "date"]).groupby("code")["close"].diff(1)
    s = out["x"].loc[df.sort_values(["code", "date"]).index]
    pd.testing.assert_series_equal(
        s.fillna(-1).reset_index(drop=True),
        expected.fillna(-1).reset_index(drop=True),
        check_names=False,
    )
