"""
综合指标库测试
"""
import numpy as np
import pandas as pd
import pytest

from validation.metrics import (
    calc_all_stats,
    factor_metrics,
    ratio_metrics,
    return_metrics,
    risk_metrics,
)
from validation.synth_data import make_synthetic_equity, make_synthetic_returns


def test_return_metrics_basic():
    eq = make_synthetic_equity(n_days=504, annual_return=0.10, annual_vol=0.18, seed=4)
    m = return_metrics(eq)
    assert "total_return" in m and "annual_return" in m
    # 期望年化收益接近 0.10, 允许一定随机误差
    assert 0.0 < m["annual_return"] < 0.2, f"unexpected annual_return: {m['annual_return']}"
    assert m["positive_days"] + m["negative_days"] == m["total_periods"]


def test_risk_metrics_drawdown():
    eq = make_synthetic_equity(seed=2)
    m = risk_metrics(eq)
    assert m["max_drawdown"] <= 0  # 回撤 <= 0
    assert m["volatility_annual"] > 0
    assert m["var_historical"] < 0  # 5% VaR 应为负
    assert m["cvar_historical"] <= m["var_historical"]  # CVaR 极值更糟


def test_risk_metrics_with_benchmark():
    eq = make_synthetic_equity(seed=3)
    bench = make_synthetic_equity(seed=4)
    m = risk_metrics(eq, benchmark=bench)
    assert "beta" in m
    assert "tracking_error_annual" in m


def test_ratio_metrics_sharpe_positive():
    eq = make_synthetic_equity(annual_return=0.10, annual_vol=0.18, seed=5)
    m = ratio_metrics(eq, risk_free=0.02)
    assert m["sharpe_ratio"] > 0


def test_ratio_metrics_sortino_calmar():
    eq = make_synthetic_equity(seed=6)
    m = ratio_metrics(eq, risk_free=0.02)
    assert "sortino_ratio" in m
    assert "calmar_ratio" in m
    assert m["omega_ratio"] >= 0


def test_factor_metrics_signal_detection():
    """构造强信号数据, 验证 IC/Rank IC 接近预期"""
    panel = make_synthetic_returns(n_stocks=100, n_days=200, signal_strength=0.05, seed=42)
    m = factor_metrics(panel[["date", "code", "factor"]], panel[["date", "code", "forward_return"]])
    assert m["ic_mean"] > 0  # 正向信号
    assert m["rank_ic_mean"] > 0
    assert m["n_dates"] == 200
    assert m["long_short_win_rate"] > 0.5


def test_factor_metrics_empty_panel():
    # 构造空 panel, 但保留所有必需列名
    panel = pd.DataFrame({
        "date": pd.Series([], dtype="datetime64[ns]"),
        "code": pd.Series([], dtype="object"),
        "factor": pd.Series([], dtype="float64"),
        "forward_return": pd.Series([], dtype="float64"),
    })
    m = factor_metrics(panel, panel)
    assert m == {}


def test_calc_all_stats_count():
    eq = make_synthetic_equity(seed=10)
    bench = make_synthetic_equity(seed=11)
    stats = calc_all_stats(eq, benchmark=bench, risk_free=0.02)
    # 一键综合输出至少 20 个指标
    assert len(stats) >= 20
    # 必须包含核心指标
    for k in ["annual_return", "sharpe_ratio", "max_drawdown", "volatility_annual",
              "sortino_ratio", "calmar_ratio", "var_historical", "cvar_historical",
              "skewness", "kurtosis", "omega_ratio"]:
        assert k in stats, f"缺少指标 {k}"


def test_sklearn_compat():
    """scipy 1.17 兼容测试: 验证 stats.skew / kurtosis API 可用"""
    eq = make_synthetic_equity(seed=99)
    m = risk_metrics(eq)
    assert np.isfinite(m["skewness"])
    assert np.isfinite(m["kurtosis"])
