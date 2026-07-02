"""
增强版向量化回测绩效指标模块

借鉴来源：
- VectorBT: 向量化回测 + 全面的绩效指标体系
- NautilusTrader: 生产级风控指标
- quantstats: 专业绩效报告指标

优化点：
原 backtest-engine/engine.py 的 _calc_metrics 仅计算 7 个基础指标：
    total_return, annual_return, volatility, sharpe, max_drawdown,
    win_rate, calmar_ratio

本模块扩展为 20+ 个专业指标，全部向量化实现（numpy），包括：
- 风险调整收益：Sortino, Calmar, Omega
- 下行风险：VaR, CVaR, 最大连续亏损
- 交易质量：换手率、盈亏比、收益稳定性
- 尾部风险：偏度、峰度、尾部比率
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


TRADING_DAYS = 252
RISK_FREE_RATE = 0.03


def calc_enhanced_metrics(
    equity: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    init_capital: float = 1e6,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> Dict[str, float]:
    """
    计算增强版绩效指标（全向量化）

    参数:
        equity: 净值曲线（index=date, value=equity）
        trades: 交易记录 DataFrame（可选，含 date, action, amount 列）
        init_capital: 初始资金
        risk_free_rate: 无风险年利率
        trading_days: 年交易日数

    返回:
        包含 20+ 指标的字典
    """
    if equity is None or len(equity) < 2:
        return {}

    eq = equity.astype(float).dropna()
    if len(eq) < 2:
        return {}

    returns = eq.pct_change().dropna()
    n_days = len(returns)
    if n_days == 0:
        return {}

    daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = returns - daily_rf

    # ---- 收益类指标 ----
    cumulative = (1 + returns).prod()
    total_return = cumulative - 1
    annual_return = (1 + total_return) ** (trading_days / n_days) - 1 if n_days > 0 else 0.0
    best_day = float(returns.max())
    worst_day = float(returns.min())

    # ---- 风险类指标 ----
    volatility = float(returns.std() * np.sqrt(trading_days))
    downside_returns = returns[returns < 0]
    downside_dev = (
        float(np.sqrt((downside_returns ** 2).mean()) * np.sqrt(trading_days))
        if len(downside_returns) > 0
        else 0.0
    )

    # ---- 风险调整收益 ----
    sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    sortino = (annual_return - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

    # ---- 回撤类指标 ----
    cummax = eq.cummax()
    drawdown = eq / cummax - 1
    max_drawdown = float(drawdown.min())

    # 最大回撤持续期（恢复天数）
    in_drawdown = drawdown < 0
    max_dd_duration = 0
    current_dd = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_dd += 1
            max_dd_duration = max(max_dd_duration, current_dd)
        else:
            current_dd = 0

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # ---- 胜率与盈亏 ----
    win_rate = float((returns > 0).mean())
    avg_win = float(returns[returns > 0].mean()) if (returns > 0).any() else 0.0
    avg_loss = float(abs(returns[returns < 0].mean())) if (returns < 0).any() else 0.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 最大连续盈利/亏损天数
    max_consec_win = _max_consecutive((returns > 0).values)
    max_consec_loss = _max_consecutive((returns < 0).values)

    # ---- 尾部风险 ----
    skewness = float(returns.skew()) if n_days > 2 else 0.0
    kurtosis = float(returns.kurtosis()) if n_days > 3 else 0.0
    var_95 = float(np.percentile(returns, 5))  # 95% VaR
    cvar_95 = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95
    tail_ratio = (
        float(abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5)))
        if np.percentile(returns, 5) != 0
        else 0.0
    )

    # ---- Omega 比率 ----
    threshold = daily_rf
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else float("inf")

    # ---- 稳定性 ----
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    metrics = {
        # 收益类
        "total_return": round(float(total_return), 6),
        "annual_return": round(float(annual_return), 6),
        "best_day": round(best_day, 6),
        "worst_day": round(worst_day, 6),
        # 风险类
        "volatility": round(volatility, 6),
        "downside_deviation": round(downside_dev, 6),
        "max_drawdown": round(max_drawdown, 6),
        "max_dd_duration": int(max_dd_duration),
        "var_95": round(var_95, 6),
        "cvar_95": round(cvar_95, 6),
        # 风险调整收益
        "sharpe_ratio": round(float(sharpe), 4),
        "sortino_ratio": round(float(sortino), 4),
        "calmar_ratio": round(float(calmar), 4),
        "omega_ratio": round(float(omega), 4),
        # 胜率与盈亏
        "win_rate": round(win_rate, 4),
        "profit_loss_ratio": round(float(profit_loss_ratio), 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "max_consec_win_days": int(max_consec_win),
        "max_consec_loss_days": int(max_consec_loss),
        # 尾部风险
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
        "tail_ratio": round(tail_ratio, 4),
        # 稳定性
        "recovery_factor": round(float(recovery_factor), 4),
    }

    # ---- 换手率（需要交易记录）----
    if trades is not None and not trades.empty and "amount" in trades.columns:
        turnover = _calc_turnover(trades, eq, init_capital)
        metrics["annual_turnover"] = round(float(turnover), 4)
        metrics["n_trades"] = int(len(trades))

    return metrics


def _max_consecutive(bool_array: np.ndarray) -> int:
    """计算布尔数组的最大连续 True 长度（向量化）"""
    if len(bool_array) == 0:
        return 0
    # 向量化：找连续段
    arr = bool_array.astype(int)
    # 在 0-1 边界处标记
    diffs = np.diff(np.concatenate([[0], arr, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    lengths = ends - starts
    return int(lengths.max())


def _calc_turnover(trades: pd.DataFrame, equity: pd.Series, init_capital: float) -> float:
    """
    计算年化换手率

    换手率 = 总成交额 / (平均净值 * 年数)
    """
    if "date" not in trades.columns:
        total_amount = trades["amount"].sum()
        n_years = len(equity) / TRADING_DAYS
        avg_equity = equity.mean()
        if avg_equity > 0 and n_years > 0:
            return total_amount / (avg_equity * n_years)
        return 0.0

    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["date"])
    total_amount = trades["amount"].sum()
    n_years = len(equity) / TRADING_DAYS
    avg_equity = equity.mean()
    if avg_equity > 0 and n_years > 0:
        return total_amount / (avg_equity * n_years)
    return 0.0


def calc_metrics_from_equity_curve(
    equity_curve: pd.DataFrame,
    trades: Optional[pd.DataFrame] = None,
    init_capital: float = 1e6,
) -> Dict[str, float]:
    """
    从净值曲线 DataFrame 计算指标（兼容原 backtest-engine 接口）

    参数:
        equity_curve: 含 date, equity 列的 DataFrame
        trades: 交易记录
    """
    if equity_curve is None or equity_curve.empty or "equity" not in equity_curve.columns:
        return {}
    eq = equity_curve.set_index("date")["equity"]
    return calc_enhanced_metrics(eq, trades, init_capital)
