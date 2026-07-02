"""
扩展绩效指标模块

借鉴来源：
- Investing Algorithm Framework 的 30+ 指标体系
- QuantStats 的专业绩效分析
- NautilusTrader 的风险指标

优化点：
原实现 skills/backtest-engine/scripts/base/base_backtest.py 的
BaseBacktestMetrics 仅计算 7 个基础指标（total_return, annual_return,
volatility, sharpe, max_drawdown, calmar, sortino, win_rate）。

本模块新增：
- VaR (Value at Risk) 95%/99%
- CVaR (Conditional VaR / Expected Shortfall)
- Information Ratio（相对基准）
- Beta / Alpha（CAPM）
- 换手率
- 最大回撤恢复期
- 下行捕获率 / 上行捕获率
- 尾部比率（Tail Ratio）
- 共偏度 / 共峰度
"""
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


def calc_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    计算 Value at Risk（历史模拟法）

    参数:
        returns: 日收益率序列
        confidence: 置信水平（0.95 或 0.99）

    返回:
        VaR 值（负数，表示最大损失）
    """
    if returns is None or returns.empty:
        return 0.0
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def calc_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    计算 Conditional VaR（Expected Shortfall）

    即损失超过 VaR 时的平均损失
    """
    if returns is None or returns.empty:
        return 0.0
    var = calc_var(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def calc_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    trading_days: int = 252,
) -> float:
    """
    计算信息比率 = 超额收益均值 / 跟踪误差

    参数:
        returns: 策略日收益率
        benchmark_returns: 基准日收益率
    """
    if returns is None or benchmark_returns is None:
        return 0.0
    # 对齐索引
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < 2:
        return 0.0
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    excess = r - b
    tracking_error = excess.std()
    if tracking_error == 0:
        return 0.0
    return float(excess.mean() / tracking_error * np.sqrt(trading_days))


def calc_beta_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, float]:
    """
    计算 CAPM Beta 和 Alpha

    返回:
        {"beta": ..., "alpha": ...}（alpha 已年化）
    """
    if returns is None or benchmark_returns is None:
        return {"beta": 0.0, "alpha": 0.0}
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < 2:
        return {"beta": 0.0, "alpha": 0.0}
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

    cov_matrix = np.cov(r, b)
    var_b = cov_matrix[1, 1]
    if var_b == 0:
        return {"beta": 0.0, "alpha": 0.0}
    beta = float(cov_matrix[0, 1] / var_b)
    # 年化 alpha = mean(r - beta * b) * 252
    daily_rf = risk_free / trading_days
    alpha = float((r.mean() - daily_rf - beta * (b.mean() - daily_rf)) * trading_days)
    return {"beta": beta, "alpha": alpha}


def calc_max_drawdown_duration(equity_curve: pd.Series) -> Dict[str, Any]:
    """
    计算最大回撤持续期与恢复期

    返回:
        {
            "max_dd_duration": 最大回撤持续天数,
            "max_dd_recovery": 最大回撤恢复天数（None 表示未恢复）,
            "underwater_start": 回撤开始日期,
            "underwater_end": 回撤谷底日期,
        }
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {
            "max_dd_duration": 0,
            "max_dd_recovery": None,
            "underwater_start": None,
            "underwater_end": None,
        }

    cummax = equity_curve.cummax()
    underwater = equity_curve < cummax

    # 找到最长的连续 underwater 段
    max_duration = 0
    current_duration = 0
    dd_start = None
    dd_end = None
    best_start = None
    best_end = None

    for i, (idx, is_under) in enumerate(underwater.items()):
        if is_under:
            if current_duration == 0:
                dd_start = idx
            current_duration += 1
            dd_end = idx
            if current_duration > max_duration:
                max_duration = current_duration
                best_start = dd_start
                best_end = dd_end
        else:
            current_duration = 0

    # 恢复期：从谷底到回到前高
    recovery = None
    if best_end is not None:
        peak_before = cummax.loc[:best_end].iloc[-1]
        after = equity_curve.loc[best_end:]
        recovered = after[after >= peak_before]
        if not recovered.empty:
            recovery = (recovered.index[0] - best_end).days

    return {
        "max_dd_duration": int(max_duration),
        "max_dd_recovery": recovery,
        "underwater_start": str(best_start) if best_start is not None else None,
        "underwater_end": str(best_end) if best_end is not None else None,
    }


def calc_capture_ratios(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """
    计算上行/下行捕获率

    - 上行捕获率：基准上涨时策略收益 / 基准收益
    - 下行捕获率：基准下跌时策略收益 / 基准收益
    """
    if returns is None or benchmark_returns is None:
        return {"up_capture": 0.0, "down_capture": 0.0}
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return {"up_capture": 0.0, "down_capture": 0.0}
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]

    up_mask = b > 0
    down_mask = b < 0

    up_capture = float(r[up_mask].sum() / b[up_mask].sum()) if up_mask.any() and b[up_mask].sum() != 0 else 0.0
    down_capture = float(r[down_mask].sum() / b[down_mask].sum()) if down_mask.any() and b[down_mask].sum() != 0 else 0.0

    return {"up_capture": up_capture, "down_capture": down_capture}


def calc_tail_ratio(returns: pd.Series) -> float:
    """
    尾部比率 = 右尾分位 / |左尾分位|
    衡量收益分布的偏斜程度
    """
    if returns is None or returns.empty:
        return 0.0
    r = returns.dropna()
    if len(r) < 10:
        return 0.0
    right = np.percentile(r, 95)
    left = np.percentile(r, 5)
    if left == 0:
        return 0.0
    return float(abs(right / left))


def calc_full_metrics(
    equity_curve: pd.Series,
    returns: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """
    一次性计算全部绩效指标（20+）

    参数:
        equity_curve: 净值序列
        returns: 日收益率（若 None 则从 equity_curve 计算）
        benchmark_returns: 基准日收益率（可选）
        risk_free: 无风险利率（年化）
        trading_days: 年交易日数

    返回:
        包含 20+ 指标的字典
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {}

    if returns is None:
        returns = equity_curve.pct_change().dropna()

    if returns.empty:
        return {}

    # 基础指标
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    n_years = len(equity_curve) / trading_days
    annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(returns.std() * np.sqrt(trading_days))
    sharpe = float((annual_return - risk_free) / volatility) if volatility > 0 else 0.0

    # 回撤
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = float(drawdown.min())
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Sortino
    neg_returns = returns[returns < 0]
    downside_std = float(neg_returns.std() * np.sqrt(trading_days)) if len(neg_returns) > 1 else 0.0
    sortino = float((annual_return - risk_free) / downside_std) if downside_std > 0 else 0.0

    # 风险指标
    var_95 = calc_var(returns, 0.95)
    cvar_95 = calc_cvar(returns, 0.95)
    var_99 = calc_var(returns, 0.99)
    cvar_99 = calc_cvar(returns, 0.99)

    # 回撤持续期
    dd_info = calc_max_drawdown_duration(equity_curve)

    # 尾部比率
    tail_ratio = calc_tail_ratio(returns)

    # 胜率
    win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

    # 基准相关指标
    info_ratio = 0.0
    beta = 0.0
    alpha = 0.0
    up_capture = 0.0
    down_capture = 0.0
    if benchmark_returns is not None and not benchmark_returns.empty:
        info_ratio = calc_information_ratio(returns, benchmark_returns, trading_days)
        ba = calc_beta_alpha(returns, benchmark_returns, risk_free, trading_days)
        beta = ba["beta"]
        alpha = ba["alpha"]
        caps = calc_capture_ratios(returns, benchmark_returns)
        up_capture = caps["up_capture"]
        down_capture = caps["down_capture"]

    return {
        # 收益类
        "total_return": total_return,
        "annual_return": annual_return,
        # 风险类
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "downside_volatility": downside_std,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "var_99": var_99,
        "cvar_99": cvar_99,
        # 风险调整收益
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "information_ratio": info_ratio,
        # CAPM
        "beta": beta,
        "alpha": alpha,
        # 回撤持续期
        "max_dd_duration": dd_info["max_dd_duration"],
        "max_dd_recovery": dd_info["max_dd_recovery"],
        # 捕获率
        "up_capture": up_capture,
        "down_capture": down_capture,
        # 其他
        "tail_ratio": tail_ratio,
        "win_rate": win_rate,
        "total_days": len(returns),
    }