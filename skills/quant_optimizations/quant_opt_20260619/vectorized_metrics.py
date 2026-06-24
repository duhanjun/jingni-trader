"""
向量化绩效指标库
==================

借鉴项目:
    1. vectorbt (7k+ stars) - NumPy 向量化 + Numba JIT
    2. quantstats - 完整的 tear sheet 指标
    3. empyrical - 风险调整收益 (Sharpe / Sortino / Calmar / Alpha / Beta)

设计目标:
    - 用 NumPy 单次调用代替 base_backtest.py 中的逐次循环
    - 支持批量计算 (一次传入多条净值曲线)
    - 纯函数式, 无副作用, 易测试
    - 与现有 BaseBacktestMetrics 兼容

本文件仅作为 PoC 验证, 不直接修改 main 分支代码.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 核心工具
# ----------------------------------------------------------------------------
def to_2d(equity: np.ndarray) -> np.ndarray:
    """统一成 (T, N) 二维数组, N=策略数, T=时间步"""
    arr = np.asarray(equity, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    """净值 -> 简单收益率, 丢弃首行 NaN"""
    eq = to_2d(equity)
    return (eq[1:] - eq[:-1]) / eq[:-1]


# ----------------------------------------------------------------------------
# 单指标 (向量化实现)
# ----------------------------------------------------------------------------
def total_return(equity: np.ndarray) -> np.ndarray:
    eq = to_2d(equity)
    return eq[-1] / eq[0] - 1.0


def annual_return(equity: np.ndarray, trading_days: int = 252) -> np.ndarray:
    eq = to_2d(equity)
    n = len(eq)
    if n < 2:
        return np.zeros(eq.shape[1])
    years = (n - 1) / trading_days
    if years <= 0:
        return np.zeros(eq.shape[1])
    total = eq[-1] / eq[0]
    return np.power(total, 1.0 / years) - 1.0


def volatility(returns_2d: np.ndarray, trading_days: int = 252) -> np.ndarray:
    """年化波动率, 默认 ddof=1"""
    if returns_2d.shape[0] < 2:
        return np.zeros(returns_2d.shape[1])
    return returns_2d.std(axis=0, ddof=1) * np.sqrt(trading_days)


def sharpe(returns_2d: np.ndarray, risk_free: float = 0.03, trading_days: int = 252) -> np.ndarray:
    vol = volatility(returns_2d, trading_days)
    vol = np.where(vol == 0, np.nan, vol)
    ann_excess = returns_2d.mean(axis=0) * trading_days - risk_free
    return ann_excess / vol


def sortino(returns_2d: np.ndarray, risk_free: float = 0.03, trading_days: int = 252) -> np.ndarray:
    if returns_2d.shape[0] < 2:
        return np.zeros(returns_2d.shape[1])
    downside = np.where(returns_2d < 0, returns_2d, 0.0)
    downside_std = np.sqrt((downside ** 2).mean(axis=0)) * np.sqrt(trading_days)
    downside_std = np.where(downside_std == 0, np.nan, downside_std)
    ann_excess = returns_2d.mean(axis=0) * trading_days - risk_free
    return ann_excess / downside_std


def max_drawdown(equity: np.ndarray) -> np.ndarray:
    eq = to_2d(equity)
    running_max = np.maximum.accumulate(eq, axis=0)
    drawdown = (eq - running_max) / running_max
    return drawdown.min(axis=0)


def calmar(equity: np.ndarray, trading_days: int = 252) -> np.ndarray:
    ar = annual_return(equity, trading_days)
    mdd = max_drawdown(equity)
    safe = np.where(np.abs(mdd) < 1e-12, np.nan, np.abs(mdd))
    return ar / safe


def win_rate(returns_2d: np.ndarray) -> np.ndarray:
    if returns_2d.size == 0:
        return np.zeros(returns_2d.shape[1] if returns_2d.ndim > 1 else 1)
    return (returns_2d > 0).sum(axis=0) / returns_2d.shape[0]


def var_historic(returns_2d: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    if returns_2d.shape[0] < 2:
        return np.zeros(returns_2d.shape[1])
    return np.percentile(returns_2d, (1 - confidence) * 100, axis=0)


def cvar_historic(returns_2d: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    if returns_2d.shape[0] < 2:
        return np.zeros(returns_2d.shape[1])
    var = var_historic(returns_2d, confidence)
    out = np.empty(returns_2d.shape[1])
    for i in range(returns_2d.shape[1]):
        mask = returns_2d[:, i] <= var[i]
        if mask.sum() == 0:
            out[i] = var[i]
        else:
            out[i] = returns_2d[mask, i].mean()
    return out


def stability(returns_2d: np.ndarray) -> np.ndarray:
    """R² of cumulative log-returns vs time. 越大越稳定."""
    if returns_2d.shape[0] < 2:
        return np.zeros(returns_2d.shape[1])
    cum_log = np.cumsum(np.log1p(returns_2d), axis=0)
    t = np.arange(cum_log.shape[0])
    out = np.empty(cum_log.shape[1])
    for i in range(cum_log.shape[1]):
        y = cum_log[:, i]
        if np.std(y) < 1e-12:
            out[i] = 0.0
            continue
        corr = np.corrcoef(t, y)[0, 1]
        out[i] = corr ** 2
    return out


# ----------------------------------------------------------------------------
# 一键计算所有指标 (与 base_backtest.BaseBacktestMetrics.calc_all_metrics 兼容)
# ----------------------------------------------------------------------------
def calc_all(equity: np.ndarray,
             risk_free: float = 0.03,
             trading_days: int = 252) -> Dict[str, np.ndarray]:
    """
    输入: (T,) 或 (T, N) 净值曲线
    输出: dict[str -> (N,) ndarray]
    """
    eq = to_2d(equity)
    rets = equity_to_returns(equity)
    return {
        "total_return":  total_return(equity),
        "annual_return": annual_return(equity, trading_days),
        "volatility":    volatility(rets, trading_days),
        "sharpe_ratio":  sharpe(rets, risk_free, trading_days),
        "sortino_ratio": sortino(rets, risk_free, trading_days),
        "max_drawdown":  max_drawdown(equity),
        "calmar_ratio":  calmar(equity, trading_days),
        "win_rate":      win_rate(rets),
        "var_95":        var_historic(rets, 0.95),
        "cvar_95":       cvar_historic(rets, 0.95),
        "stability":     stability(rets),
    }


def to_dict(metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """把 (N,) 数组转成 list, 方便 JSON 序列化"""
    return {k: v.tolist() for k, v in metrics.items()}


if __name__ == "__main__":
    # 自测
    np.random.seed(0)
    eq1 = np.cumprod(1 + np.random.normal(0.0005, 0.01, 500)) * 1e6
    eq2 = np.cumprod(1 + np.random.normal(0.0010, 0.015, 500)) * 1e6
    eq = np.column_stack([eq1, eq2])

    res = calc_all(eq)
    print("批量计算结果 (2 条曲线):")
    for k, v in res.items():
        print(f"  {k:14s} = {v}")