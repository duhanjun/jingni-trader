"""
向量化绩效指标计算器

借鉴自 vectorbt (https://github.com/polakowo/vectorbt) 的纯 NumPy 性能指标实现，
目标是替代 jingni-trader 中基于 pandas pct_change() 的慢速指标计算，为参数
扫描和因子评估提供高速计算能力。

主要优化点：
1. 全部使用 numpy ndarray 输入，零 pandas 开销
2. 一次性计算多个指标，避免重复计算
3. 数值稳定的算法（避免除以零、log(0) 等）
4. 纯 Python/NumPy 实现，无外部重依赖（与 jingni-trader 主链路解耦）
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional


TRADING_DAYS_PER_YEAR = 252


def _to_ndarray(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr, dtype=np.float64)


def annualized_return(returns: np.ndarray, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化收益率：基于复利累计收益"""
    r = _to_ndarray(returns)
    if r.size == 0:
        return 0.0
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    cum = np.prod(1.0 + r)
    n_periods = r.size
    if cum <= 0:
        return -1.0
    return float(cum ** (periods / n_periods) - 1.0)


def annualized_volatility(returns: np.ndarray, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """年化波动率：日收益标准差 * sqrt(periods)"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: np.ndarray,
                 risk_free: float = 0.03,
                 periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """夏普比率：(年化收益 - 无风险利率) / 年化波动率"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods
    mean = np.mean(excess)
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(periods))


def sortino_ratio(returns: np.ndarray,
                  risk_free: float = 0.03,
                  periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """索提诺比率：仅用下行波动率作为分母"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods
    downside = r[r < 0] - risk_free / periods
    if downside.size < 2:
        return 0.0
    down_std = np.sqrt(np.mean(downside ** 2))
    if down_std < 1e-12:
        return 0.0
    return float(np.mean(excess) / down_std * np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    """最大回撤：peak-to-trough 最大跌幅（负数）"""
    eq = _to_ndarray(equity)
    if eq.size == 0:
        return 0.0
    eq = eq[~np.isnan(eq)]
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    drawdown = eq / running_max - 1.0
    return float(np.min(drawdown))


def max_drawdown_duration(equity: np.ndarray) -> int:
    """最大回撤持续期：peak 到 recovery 的最大间隔（交易日数）"""
    eq = _to_ndarray(equity)
    if eq.size == 0:
        return 0
    running_max = np.maximum.accumulate(eq)
    underwater = eq < running_max
    if not underwater.any():
        return 0
    max_dur = cur_dur = 0
    for v in underwater:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return int(max_dur)


def calmar_ratio(returns: np.ndarray,
                 periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calmar 比率：年化收益 / |最大回撤|"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return 0.0
    eq = np.cumprod(1.0 + r)
    mdd = max_drawdown(eq)
    if mdd >= 0:
        return 0.0
    ar = annualized_return(r, periods)
    return float(ar / abs(mdd))


def win_rate(returns: np.ndarray) -> float:
    """胜率：正收益日数占比"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    return float(np.mean(r > 0))


def information_ratio(returns: np.ndarray,
                      benchmark: np.ndarray,
                      periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """信息比率：(策略 - 基准) 主动收益的年化均值 / 跟踪误差"""
    r = _to_ndarray(returns)
    b = _to_ndarray(benchmark)
    n = min(r.size, b.size)
    if n < 2:
        return 0.0
    r = r[-n:]
    b = b[-n:]
    active = r - b
    active = active[~np.isnan(active)]
    if active.size < 2:
        return 0.0
    mean = np.mean(active)
    std = np.std(active, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(periods))


def omega_ratio(returns: np.ndarray,
                threshold: float = 0.0) -> float:
    """Omega 比率：收益超过阈值的累积 / 低于阈值的累积"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return 0.0
    excess = r - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_all_metrics(returns: np.ndarray,
                        benchmark: Optional[np.ndarray] = None,
                        equity: Optional[np.ndarray] = None,
                        risk_free: float = 0.03) -> Dict[str, float]:
    """一次性计算所有常用指标，避免重复计算中间量"""
    r = _to_ndarray(returns)
    r = r[~np.isnan(r)]
    if r.size == 0:
        return {}
    if equity is None:
        equity = np.cumprod(1.0 + r)
    metrics = {
        "annualized_return": annualized_return(r),
        "annualized_volatility": annualized_volatility(r),
        "sharpe_ratio": sharpe_ratio(r, risk_free),
        "sortino_ratio": sortino_ratio(r, risk_free),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_duration": max_drawdown_duration(equity),
        "calmar_ratio": calmar_ratio(r),
        "win_rate": win_rate(r),
        "omega_ratio": omega_ratio(r),
    }
    if benchmark is not None:
        b = _to_ndarray(benchmark)
        b = b[~np.isnan(b)]
        if b.size > 0:
            metrics["information_ratio"] = information_ratio(r, b)
    return metrics
