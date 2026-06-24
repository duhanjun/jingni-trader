"""
增强风险指标（验证版）

借鉴来源：
  - VectorBT 的 Portfolio 统计 (VaR, CVaR, Capture, Deflated Sharpe)
  - NautilusTrader 的风险模块 (beta/alpha vs benchmark, Information Ratio)

对比 jingni-trader 现有 BaseBacktestMetrics：
  - 现有指标：total_return, annual_return, volatility, sharpe, max_drawdown,
              calmar, sortino, win_rate
  - 缺失：VaR, CVaR (条件VaR), Information Ratio, beta, alpha, 上行/下行捕获,
          换手率, 最长回撤持续期

本模块补充上述指标，全部用向量化 NumPy 实现，便于在回测引擎中复用。
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


TRADING_DAYS = 252


def calc_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """历史法 VaR（Value at Risk）。

    参数:
        returns: 日收益率序列
        alpha:   显著性水平，0.05 表示 95% 置信下的最大日损失
    返回:
        VaR（负数表示损失）
    """
    if len(returns) < 2:
        return 0.0
    return float(np.percentile(returns.dropna(), alpha * 100))


def calc_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """CVaR / Expected Shortfall：尾部损失均值。"""
    s = returns.dropna()
    if len(s) < 2:
        return 0.0
    var = np.percentile(s, alpha * 100)
    tail = s[s <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)


def calc_beta(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """组合相对基准的 beta。"""
    df = pd.concat([port_returns, bench_returns], axis=1, keys=["p", "b"]).dropna()
    if len(df) < 2:
        return 0.0
    cov = df.cov().iloc[0, 1]
    var_b = df["b"].var()
    return float(cov / var_b) if var_b > 0 else 0.0


def calc_alpha(port_returns: pd.Series, bench_returns: pd.Series, risk_free: float = 0.03) -> float:
    """Jensen's Alpha（年化）。"""
    beta = calc_beta(port_returns, bench_returns)
    ann_p = port_returns.mean() * TRADING_DAYS
    ann_b = bench_returns.mean() * TRADING_DAYS
    return float(ann_p - (risk_free + beta * (ann_b - risk_free)))


def calc_information_ratio(port_returns: pd.Series, bench_returns: pd.Series) -> float:
    """信息比率：超额收益均值 / 跟踪误差。"""
    df = pd.concat([port_returns, bench_returns], axis=1, keys=["p", "b"]).dropna()
    if len(df) < 2:
        return 0.0
    excess = df["p"] - df["b"]
    te = excess.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return 0.0
    return float(excess.mean() * TRADING_DAYS / te)


def calc_capture_ratios(port_returns: pd.Series, bench_returns: pd.Series) -> Dict[str, float]:
    """上行/下行捕获比率。"""
    df = pd.concat([port_returns, bench_returns], axis=1, keys=["p", "b"]).dropna()
    if len(df) < 2:
        return {"up_capture": 0.0, "down_capture": 0.0}
    up = df[df["b"] > 0]
    dn = df[df["b"] < 0]
    up_capture = float(up["p"].mean() / up["b"].mean()) if len(up) > 0 and up["b"].mean() != 0 else 0.0
    down_capture = float(dn["p"].mean() / dn["b"].mean()) if len(dn) > 0 and dn["b"].mean() != 0 else 0.0
    return {"up_capture": up_capture, "down_capture": down_capture}


def calc_max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤持续期（交易日数）。"""
    if len(equity) < 2:
        return 0
    cummax = equity.cummax()
    in_dd = equity < cummax
    max_dur = 0
    cur = 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        max_dur = max(max_dur, cur)
    return int(max_dur)


def calc_all_enhanced_metrics(
    equity: pd.Series,
    daily_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    risk_free: float = 0.03,
) -> Dict[str, Any]:
    """一次性计算增强后的全部风险/绩效指标。"""
    m: Dict[str, Any] = {}
    m["var_95"] = calc_var(daily_returns, 0.05)
    m["cvar_95"] = calc_cvar(daily_returns, 0.05)
    m["max_drawdown_duration"] = calc_max_drawdown_duration(equity)

    if benchmark_returns is not None:
        m["beta"] = calc_beta(daily_returns, benchmark_returns)
        m["alpha"] = calc_alpha(daily_returns, benchmark_returns, risk_free)
        m["information_ratio"] = calc_information_ratio(daily_returns, benchmark_returns)
        m.update(calc_capture_ratios(daily_returns, benchmark_returns))
    if turnover is not None:
        m["avg_turnover"] = float(turnover.mean())
        m["annual_turnover"] = float(turnover.mean() * TRADING_DAYS)
    return m
