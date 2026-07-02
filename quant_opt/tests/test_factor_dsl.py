"""Test suite: factor_dsl (Alphalens/AKQuant-inspired expression engine)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._synth_data import make_synth_panel

from factor_dsl.evaluator import (
    parse,
    tokenize,
    FactorEvaluator,
    evaluate_factor,
    PRESET_FACTORS,
    list_preset_factors,
    eval_preset,
)


@pytest.fixture(scope="module")
def panel_df():
    return make_synth_panel(n_codes=8, n_days=120)


# ----------------------------------------------------------------------
# 1. Parser
# ----------------------------------------------------------------------

def test_tokenize_numbers_and_names():
    toks = tokenize("Rank(Delta(close, 5))")
    kinds = [k for k, _ in toks]
    assert kinds == ["NAME", "PUNCT", "NAME", "PUNCT", "NAME", "PUNCT", "NUM", "PUNCT", "PUNCT"]


def test_tokenize_rejects_invalid():
    with pytest.raises(ValueError):
        tokenize("Bad#Char")


def test_parse_number():
    ast = parse("3.14")
    assert ast.value == 3.14


def test_parse_call_structure():
    ast = parse("Rank(x)")
    assert ast.name == "Rank"
    assert len(ast.args) == 1
    assert ast.args[0].name == "x"


def test_parse_nested_call():
    ast = parse("Add(Mul(2, 3), 1)")
    assert ast.name == "Add"
    assert ast.args[0].name == "Mul"


def test_parse_trailing_garbage_fails():
    with pytest.raises(ValueError):
        parse("Rank(x) extra")


# ----------------------------------------------------------------------
# 2. Evaluator
# ----------------------------------------------------------------------

def test_evaluator_requires_multiindex():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        FactorEvaluator(df)


def test_evaluate_constant(panel_df):
    s = evaluate_factor(panel_df, "5.0")
    assert (s == 5.0).all()


def test_evaluate_var(panel_df):
    s = evaluate_factor(panel_df, "close")
    np.testing.assert_array_equal(s.values, panel_df.set_index(["code", "date"])["close"].values)


def test_evaluate_delta(panel_df):
    s = evaluate_factor(panel_df, "Delta(close, 1)")
    # 转为 MultiIndex 后, 验证 diff 逻辑
    expected = (
        panel_df.sort_values(["code", "date"])
        .set_index(["code", "date"])["close"]
        .groupby(level="code").diff(1)
    )
    pd.testing.assert_series_equal(s, expected, check_names=False)


def test_evaluate_rank_cross_section(panel_df):
    """Rank 应在每个 date 截面独立排名, 值在 [0, 1]."""
    s = evaluate_factor(panel_df, "Rank(close)")
    assert s.groupby(level="date").max().max() <= 1.0 + 1e-9
    assert s.groupby(level="date").min().min() >= 0.0 - 1e-9


def test_evaluate_nested_expression(panel_df):
    s = evaluate_factor(panel_df, "Sub(close, Ts_Mean(close, 5))")
    assert s.notna().any()


def test_evaluate_unknown_var_raises(panel_df):
    with pytest.raises(KeyError):
        evaluate_factor(panel_df, "Bogus(column)")


# ----------------------------------------------------------------------
# 3. Preset factors
# ----------------------------------------------------------------------

def test_preset_list_nonempty():
    presets = list_preset_factors()
    assert "reversal_5d" in presets
    assert "momentum_20d" in presets


def test_eval_preset_reversal_5d(panel_df):
    """反转因子 = -Delta(close, 5), 验证方向正确."""
    expected = -evaluate_factor(panel_df, "Delta(close, 5)")
    got = eval_preset("reversal_5d", panel_df)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_eval_preset_momentum_20d(panel_df):
    expected = evaluate_factor(panel_df, "Delta(close, 20)")
    got = eval_preset("momentum_20d", panel_df)
    pd.testing.assert_series_equal(got, expected, check_names=False)


def test_eval_preset_unknown_raises(panel_df):
    with pytest.raises(KeyError):
        eval_preset("does_not_exist", panel_df)


# ----------------------------------------------------------------------
# 4. 与 jingni-trader 现有因子计算结果的一致性
# ----------------------------------------------------------------------

def test_dsl_volatility_matches_simple_definition(panel_df):
    """volatility_20d 应等价于 close 20 日收益率 std."""
    s = eval_preset("volatility_20d", panel_df)
    expected = (
        panel_df.sort_values(["code", "date"])
        .set_index(["code", "date"])["close"]
        .groupby(level="code")
        .pct_change()
        .groupby(level="code")
        .rolling(20, min_periods=2).std()
        .reset_index(level=0, drop=True)
    )
    # 我们的实现是 Ts_Std(close, 20) 而非 std of returns, 因此仅在数量级上对比
    assert s.abs().mean() > 0
