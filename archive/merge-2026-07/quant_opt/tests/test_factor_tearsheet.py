"""Test suite: factor_tearsheet (Alphalens-inspired)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._synth_data import make_synth_panel, make_synth_factor

from factor_tearsheet.tearsheet import (
    compute_forward_returns,
    get_clean_factor_and_forward_returns,
    mean_return_by_quantile,
    compute_mean_return_spread,
    factor_information_coefficient,
    ic_summary,
    factor_turnover,
    create_full_tear_sheet,
)


@pytest.fixture(scope="module")
def panel():
    return make_synth_panel(n_codes=15, n_days=120)


@pytest.fixture(scope="module")
def factor_df(panel):
    return make_synth_factor(panel, signal_strength=0.5)


# ----------------------------------------------------------------------
# 1. compute_forward_returns
# ----------------------------------------------------------------------

def test_compute_forward_returns_columns_and_shape(panel):
    out = compute_forward_returns(panel, periods=(1, 5, 10))
    assert "ret_forward_1D" in out.columns
    assert "ret_forward_5D" in out.columns
    assert "ret_forward_10D" in out.columns
    assert len(out) == len(panel)


def test_compute_forward_returns_last_values_are_nan(panel):
    """最后一根 bar 之后没有 future, 所以 ret_forward_* 末位应为 NaN."""
    out = compute_forward_returns(panel.sort_values(["code", "date"]), periods=(1, 5, 10))
    for code, sub in out.groupby("code"):
        assert np.isnan(sub.iloc[-1]["ret_forward_1D"])
        assert np.isnan(sub.iloc[-1]["ret_forward_5D"])


def test_compute_forward_returns_close_relationship(panel):
    """ret_forward_1D ≈ close_{t+1}/close_t - 1."""
    df = panel.sort_values(["code", "date"]).copy()
    out = compute_forward_returns(df, periods=(1,))
    df = df.merge(out[["code", "date", "ret_forward_1D"]], on=["code", "date"])
    df["expected"] = df.groupby("code")["close"].shift(-1) / df["close"] - 1
    diff = (df["ret_forward_1D"] - df["expected"]).dropna().abs()
    assert diff.max() < 1e-9


# ----------------------------------------------------------------------
# 2. get_clean_factor_and_forward_returns
# ----------------------------------------------------------------------

def test_clean_factor_output_index(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1, 5))
    assert isinstance(clean.index, pd.MultiIndex)
    assert clean.index.names == ["date", "code"]
    assert "factor" in clean.columns
    assert "factor_quantile" in clean.columns
    assert "ret_forward_1D" in clean.columns
    assert "ret_forward_5D" in clean.columns


def test_clean_factor_quantile_range(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1,))
    valid = clean["factor_quantile"][clean["factor_quantile"] >= 0]
    assert valid.min() >= 0
    assert valid.max() <= 4


def test_clean_factor_zscore_filter(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(
        factor_df, panel, quantiles=5, periods=(1,), filter_zscore=0.5
    )
    # 极端 z-score 被过滤后, 因子 std 应小于输入 std
    raw_std = factor_df["alpha_factor"].std()
    clean_std = clean["factor"].std()
    assert clean_std < raw_std * 0.9  # 至少过滤掉 10% 极端值


# ----------------------------------------------------------------------
# 3. 分位收益
# ----------------------------------------------------------------------

def test_mean_return_by_quantile_shape(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1,))
    mrq = mean_return_by_quantile(clean, by_date=True)
    assert "date" in mrq.columns
    assert "factor_quantile" in mrq.columns
    assert "ret_forward_1D" in mrq.columns
    assert mrq["factor_quantile"].nunique() <= 5


def test_long_short_spread_positive_for_predictive_factor(factor_df, panel):
    """高信号因子应产生正的多空收益 (long top - short bottom)."""
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1,))
    mrq = mean_return_by_quantile(clean, by_date=True)
    # quantiles=5 -> 桶 0..4, top=4, bottom=0
    spread = compute_mean_return_spread(mrq, upper_q=4, lower_q=0)
    # 我们的合成因子带 +0.5 信号强度, 期望 spread > 0
    assert spread["ret_forward_1D"] > 0


# ----------------------------------------------------------------------
# 4. IC
# ----------------------------------------------------------------------

def test_ic_summary_structure(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1, 5))
    ic_ts = factor_information_coefficient(clean, method="spearman")
    assert {"date", "period", "ic"}.issubset(ic_ts.columns)
    summ = ic_summary(ic_ts)
    assert "ret_forward_1D" in summ
    assert "ic_mean" in summ["ret_forward_1D"]
    assert "ic_ir" in summ["ret_forward_1D"]


def test_ic_mean_positive_for_predictive_factor(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1,))
    ic_ts = factor_information_coefficient(clean, method="spearman")
    summ = ic_summary(ic_ts)
    assert summ["ret_forward_1D"]["ic_mean"] > 0


# ----------------------------------------------------------------------
# 5. Turnover
# ----------------------------------------------------------------------

def test_turnover_in_unit_range(factor_df, panel):
    clean = get_clean_factor_and_forward_returns(factor_df, panel, quantiles=5, periods=(1,))
    to = factor_turnover(clean, quantile=4)
    if not to.empty:
        assert to["top_turnover"].dropna().between(0, 1).all()
        assert to["bottom_turnover"].dropna().between(0, 1).all()


# ----------------------------------------------------------------------
# 6. Full tear sheet
# ----------------------------------------------------------------------

def test_create_full_tear_sheet_keys(factor_df, panel):
    ts = create_full_tear_sheet(factor_df, panel, quantiles=5, periods=(1, 5, 10))
    assert "clean_factor_summary" in ts
    assert "mean_return_by_quantile" in ts
    assert "mean_return_spread" in ts
    assert "ic_summary" in ts
    assert "turnover" in ts
    assert ts["periods"] == [1, 5, 10]
    assert ts["quantiles"] == 5
    assert ts["clean_factor_summary"]["rows"] > 0
