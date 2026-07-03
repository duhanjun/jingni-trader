"""
Backtest Benchmark Comparison
==============================

为 jingni-trader 的 ``BacktestEngine._calc_metrics`` 提供**对比基准**的扩展能力.

当前 jingni-trader 的 ``_calc_metrics`` 仅输出绝对收益/夏普等指标,
**缺少与基准 (benchmark) 的相对指标**: alpha, beta, information ratio, tracking error.

本模块:
1. 接收 ``equity_curve`` 和 ``benchmark_curve`` (二者 index 对齐)
2. 计算 QuantStats 风格相对指标 (借鉴 Pyfolio/QuantConnect Lean)
3. 与现有 ``BaseBacktestMetrics`` 输出 schema 完全兼容
4. 100% 纯函数, 可直接被 reports-engine 复用

References
----------
- Pyfolio: timeseries.py -> perf_stats, common.py -> alpha_beta
- QuantConnect Lean: Report/ section  -> Alpha, Beta, InformationRatio, TrackingError
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# 1. 公共工具
# --------------------------------------------------------------------------------------

def _to_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def _annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).prod()
    n_years = len(returns) / trading_days
    if n_years <= 0 or cumulative <= 0:
        return 0.0
    return float(cumulative ** (1 / n_years) - 1)


def _annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days))


# --------------------------------------------------------------------------------------
# 2. Alpha / Beta / IR / Tracking Error
# --------------------------------------------------------------------------------------

def alpha_beta(strategy_ret: pd.Series, bench_ret: pd.Series) -> Dict[str, float]:
    """
    计算 CAPM alpha/beta (年化 alpha).

    Returns
    -------
    dict: {"alpha": float, "beta": float, "r_squared": float}
    """
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if len(aligned) < 5 or aligned["b"].var() == 0:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    cov = np.cov(aligned["s"], aligned["b"], ddof=1)
    var_b = float(cov[1, 1])
    if var_b == 0:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}
    beta = float(cov[0, 1] / var_b)
    alpha_daily = float((aligned["s"] - beta * aligned["b"]).mean())
    alpha_annual = alpha_daily * 252

    # R^2
    corr = float(aligned["s"].corr(aligned["b"]))
    r2 = corr ** 2
    return {"alpha": alpha_annual, "beta": beta, "r_squared": r2}


def tracking_error(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    年化跟踪误差 = std(strategy - bench) * sqrt(252).
    """
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if aligned.empty:
        return 0.0
    active = aligned["s"] - aligned["b"]
    return float(active.std() * np.sqrt(252))


def information_ratio(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    IR = mean(strategy - bench) * 252 / tracking_error.
    """
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    if aligned.empty:
        return 0.0
    te = tracking_error(aligned["s"], aligned["b"])
    if te == 0:
        return 0.0
    active_ann = float((aligned["s"] - aligned["b"]).mean()) * 252
    return active_ann / te


def up_capture(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    Up capture ratio: 上涨市中被动的策略收益 / 基准收益.
    """
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    up = aligned[aligned["b"] > 0]
    if up.empty or up["b"].sum() == 0:
        return 0.0
    return float(up["s"].sum() / up["b"].sum())


def down_capture(strategy_ret: pd.Series, bench_ret: pd.Series) -> float:
    """
    Down capture ratio: 下跌市中被动的策略收益 / 基准收益.
    """
    aligned = pd.concat([strategy_ret, bench_ret], axis=1, join="inner").dropna()
    aligned.columns = ["s", "b"]
    down = aligned[aligned["b"] < 0]
    if down.empty or down["b"].sum() == 0:
        return 0.0
    return float(down["s"].sum() / down["b"].sum())


# --------------------------------------------------------------------------------------
# 3. 一站式相对指标
# --------------------------------------------------------------------------------------

def relative_metrics(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    一站式计算对比基准的相对指标.

    Parameters
    ----------
    strategy_equity, benchmark_equity : pd.Series
        净值序列, index 为日期.
    risk_free : float
        年化无风险利率.
    trading_days : int
        年交易日.

    Returns
    -------
    dict:
        包含: excess_return, alpha, beta, r_squared, tracking_error,
              information_ratio, up_capture, down_capture,
              strategy_* (策略自身指标),
              benchmark_* (基准自身指标).
    """
    # 对齐
    s_eq = strategy_equity.dropna()
    b_eq = benchmark_equity.dropna()
    common_idx = s_eq.index.intersection(b_eq.index)
    s_eq = s_eq.loc[common_idx]
    b_eq = b_eq.loc[common_idx]

    s_ret = _to_returns(s_eq)
    b_ret = _to_returns(b_eq)

    ab = alpha_beta(s_ret, b_ret)
    te = tracking_error(s_ret, b_ret)
    ir = information_ratio(s_ret, b_ret)
    uc = up_capture(s_ret, b_ret)
    dc = down_capture(s_ret, b_ret)

    s_ann = _annualized_return(s_ret, trading_days)
    b_ann = _annualized_return(b_ret, trading_days)
    s_vol = _annualized_vol(s_ret, trading_days)
    b_vol = _annualized_vol(b_ret, trading_days)
    s_sharpe = (s_ann - risk_free) / s_vol if s_vol > 0 else 0.0
    b_sharpe = (b_ann - risk_free) / b_vol if b_vol > 0 else 0.0

    # 最大回撤
    def _mdd(eq: pd.Series) -> float:
        if eq.empty:
            return 0.0
        return float((eq / eq.cummax() - 1).min())

    return {
        "excess_annual_return": round(s_ann - b_ann, 6),
        "alpha": round(ab["alpha"], 6),
        "beta": round(ab["beta"], 6),
        "r_squared": round(ab["r_squared"], 6),
        "tracking_error": round(te, 6),
        "information_ratio": round(ir, 6),
        "up_capture": round(uc, 6),
        "down_capture": round(dc, 6),
        "strategy": {
            "annual_return": round(s_ann, 6),
            "volatility": round(s_vol, 6),
            "sharpe_ratio": round(s_sharpe, 6),
            "max_drawdown": round(_mdd(s_eq), 6),
        },
        "benchmark": {
            "annual_return": round(b_ann, 6),
            "volatility": round(b_vol, 6),
            "sharpe_ratio": round(b_sharpe, 6),
            "max_drawdown": round(_mdd(b_eq), 6),
        },
    }


# --------------------------------------------------------------------------------------
# 4. 与 jingni-trader 现有 _calc_metrics 对接
# --------------------------------------------------------------------------------------

def augment_backtest_metrics(
    base_metrics: Dict[str, Any],
    strategy_equity: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    risk_free: float = 0.03,
) -> Dict[str, Any]:
    """
    在 jingni-trader ``BacktestEngine._calc_metrics`` 的输出基础上,
    添加 benchmark 相对指标.  不修改 main 分支.

    Parameters
    ----------
    base_metrics : dict
        现有 _calc_metrics 的输出 (total_return, annual_return, ...).
    strategy_equity, benchmark_equity : pd.Series
    risk_free : float

    Returns
    -------
    dict:  base_metrics + (relative metrics if benchmark_equity provided)
    """
    out = dict(base_metrics)
    if benchmark_equity is None or benchmark_equity.empty:
        out["has_benchmark"] = False
        return out

    rm = relative_metrics(strategy_equity, benchmark_equity, risk_free=risk_free)
    out["has_benchmark"] = True
    out.update(rm)
    return out
