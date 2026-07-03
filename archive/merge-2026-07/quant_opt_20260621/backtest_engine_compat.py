"""
回测绩效指标计算（与 main 分支 skills/backtest-engine/scripts/base/base_backtest.py
中 BaseBacktestMetrics.calc_all_metrics 公式完全一致的自包含实现）

为何自包含：
- main 分支的 base_backtest.py 通过 `from scripts.base.base_backtest import ...`
  导入，依赖 sys.path 注入 scripts 包，在独立测试场景下不稳定。
- 本模块直接复刻同一套公式，确保向量化适配器产出的 metrics 与 native
  适配器在数学定义上一致，便于做 correctness 对比。
"""
from __future__ import annotations

from typing import Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd


def calc_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def calc_annual_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_years = len(equity_curve) / trading_days
    if n_years <= 0:
        return 0.0
    return float(total_return ** (1 / n_years) - 1)


def calc_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days))


def calc_sharpe(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
    vol = calc_volatility(returns, trading_days)
    if vol == 0:
        return 0.0
    ann_return = returns.mean() * trading_days
    return float((ann_return - risk_free) / vol)


def calc_max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return float(drawdown.min())


def calc_calmar(equity_curve: pd.Series, trading_days: int = 252) -> float:
    ann_return = calc_annual_return(equity_curve, trading_days)
    mdd = abs(calc_max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(ann_return / mdd)


def calc_win_rate(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    winning = (trades["pnl"] > 0).sum()
    total = len(trades)
    return float(winning / total) if total > 0 else 0.0


def calc_sortino(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
    negative_returns = returns[returns < 0]
    if len(negative_returns) < 2:
        return 0.0
    downside_std = negative_returns.std() * np.sqrt(trading_days)
    if downside_std == 0:
        return 0.0
    ann_return = returns.mean() * trading_days
    return float((ann_return - risk_free) / downside_std)


def calc_all_metrics_compat(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    risk_free: float = 0.03,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """与 BaseBacktestMetrics.calc_all_metrics 完全一致的指标集"""
    returns = equity_curve.pct_change().dropna()
    return {
        "total_return": calc_total_return(equity_curve),
        "annual_return": calc_annual_return(equity_curve, trading_days),
        "volatility": calc_volatility(returns, trading_days),
        "sharpe_ratio": calc_sharpe(returns, risk_free, trading_days),
        "max_drawdown": calc_max_drawdown(equity_curve),
        "calmar_ratio": calc_calmar(equity_curve, trading_days),
        "sortino_ratio": calc_sortino(returns, risk_free, trading_days),
        "win_rate": calc_win_rate(trades),
        "total_trades": len(trades),
        "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
