"""Test suite: benchmarks.relative_metrics (QuantConnect/Pyfolio-inspired)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._synth_data import make_synth_equity

from benchmarks.relative_metrics import (
    alpha_beta,
    tracking_error,
    information_ratio,
    up_capture,
    down_capture,
    relative_metrics,
    augment_backtest_metrics,
)


@pytest.fixture(scope="module")
def equity_df():
    """策略 beta=0.9, alpha=0.0002 daily, 250+ 天."""
    return make_synth_equity(n_days=300, seed=2024)


@pytest.fixture(scope="module")
def strategy_eq(equity_df):
    return equity_df.set_index("date")["strategy_equity"]


@pytest.fixture(scope="module")
def benchmark_eq(equity_df):
    return equity_df.set_index("date")["benchmark_equity"]


# ----------------------------------------------------------------------
# 1. alpha_beta
# ----------------------------------------------------------------------

def test_alpha_beta_recovery():
    """合成数据 beta=0.9, alpha=0.0002/day -> 估计值应接近."""
    eq = make_synth_equity(n_days=500, seed=11)
    s = eq.set_index("date")["strategy_equity"]
    b = eq.set_index("date")["benchmark_equity"]
    s_r = s.pct_change().dropna()
    b_r = b.pct_change().dropna()
    ab = alpha_beta(s_r, b_r)
    # beta 应在 0.7-1.1 范围 (合成真实值 0.9)
    assert 0.7 < ab["beta"] < 1.1
    # alpha 年化 应显著 > 0
    assert ab["alpha"] > 0
    # R^2 应较高
    assert ab["r_squared"] > 0.5


def test_alpha_beta_zero_variance_bench():
    """基准无波动时, beta 应为 0."""
    s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    b = pd.Series(np.zeros(100))
    ab = alpha_beta(s, b)
    assert ab["beta"] == 0.0


# ----------------------------------------------------------------------
# 2. tracking_error
# ----------------------------------------------------------------------

def test_tracking_error_scales_with_idio_vol():
    eq_low = make_synth_equity(n_days=500, seed=11, strat_idio_vol=0.002)
    eq_high = make_synth_equity(n_days=500, seed=11, strat_idio_vol=0.01)
    s_low = eq_low.set_index("date")["strategy_equity"]
    b_low = eq_low.set_index("date")["benchmark_equity"]
    s_high = eq_high.set_index("date")["strategy_equity"]
    b_high = eq_high.set_index("date")["benchmark_equity"]
    te_low = tracking_error(s_low.pct_change().dropna(), b_low.pct_change().dropna())
    te_high = tracking_error(s_high.pct_change().dropna(), b_high.pct_change().dropna())
    assert te_high > te_low * 2


# ----------------------------------------------------------------------
# 3. information_ratio
# ----------------------------------------------------------------------

def test_ir_positive_for_strat_with_alpha():
    eq = make_synth_equity(n_days=500, seed=99)
    s = eq.set_index("date")["strategy_equity"]
    b = eq.set_index("date")["benchmark_equity"]
    ir = information_ratio(s.pct_change().dropna(), b.pct_change().dropna())
    assert ir > 0


# ----------------------------------------------------------------------
# 4. up_capture / down_capture
# ----------------------------------------------------------------------

def test_capture_ratios_in_unit_range():
    eq = make_synth_equity(n_days=500, seed=33)
    s = eq.set_index("date")["strategy_equity"]
    b = eq.set_index("date")["benchmark_equity"]
    s_r = s.pct_change().dropna()
    b_r = b.pct_change().dropna()
    uc = up_capture(s_r, b_r)
    dc = down_capture(s_r, b_r)
    # 捕获比率没有强约束, 但应在合理范围内
    assert -2.0 < uc < 3.0
    assert -2.0 < dc < 3.0


# ----------------------------------------------------------------------
# 5. relative_metrics 一站式
# ----------------------------------------------------------------------

def test_relative_metrics_schema(strategy_eq, benchmark_eq):
    rm = relative_metrics(strategy_eq, benchmark_eq)
    assert "alpha" in rm
    assert "beta" in rm
    assert "tracking_error" in rm
    assert "information_ratio" in rm
    assert "up_capture" in rm
    assert "down_capture" in rm
    assert "strategy" in rm and isinstance(rm["strategy"], dict)
    assert "benchmark" in rm and isinstance(rm["benchmark"], dict)
    # strategy 子字典包含基础指标
    for k in ("annual_return", "volatility", "sharpe_ratio", "max_drawdown"):
        assert k in rm["strategy"]


def test_relative_metrics_aligned_known_alpha(strategy_eq, benchmark_eq):
    rm = relative_metrics(strategy_eq, benchmark_eq)
    # 真实 alpha = 0.0002/day, 年化 ≈ 0.0002 * 252 = 0.0504
    # 估计值允许一定误差, 符号必须为正
    assert rm["alpha"] > 0
    assert rm["strategy"]["annual_return"] > rm["benchmark"]["annual_return"]


# ----------------------------------------------------------------------
# 6. augment_backtest_metrics
# ----------------------------------------------------------------------

def test_augment_with_benchmark(strategy_eq, benchmark_eq):
    base = {
        "total_return": 0.1,
        "annual_return": 0.05,
        "volatility": 0.15,
        "sharpe_ratio": 0.3,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
    }
    out = augment_backtest_metrics(base, strategy_eq, benchmark_eq)
    assert out["has_benchmark"] is True
    assert "alpha" in out
    assert "beta" in out
    assert "information_ratio" in out
    # 原有字段保留
    for k, v in base.items():
        assert out[k] == v


def test_augment_without_benchmark():
    base = {"total_return": 0.1, "sharpe_ratio": 0.5}
    out = augment_backtest_metrics(base, pd.Series(dtype=float), None)
    assert out["has_benchmark"] is False
    assert "alpha" not in out
