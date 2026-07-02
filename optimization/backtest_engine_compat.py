"""
兼容性模块：复用 main 分支的绩效计算逻辑，避免直接修改原代码
从 skills/backtest-engine/scripts/base/base_backtest.py 复制核心计算方法
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime


class BaseBacktestMetricsCompat:
    """绩效指标计算（与 main 分支 BaseBacktestMetrics 保持一致）"""

    @staticmethod
    def calc_total_return(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    @staticmethod
    def calc_annual_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
        if len(equity_curve) < 2:
            return 0.0
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = len(equity_curve) / trading_days
        if n_years <= 0:
            return 0.0
        return float(total_return ** (1 / n_years) - 1)

    @staticmethod
    def calc_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days))

    @staticmethod
    def calc_sharpe(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        vol = BaseBacktestMetricsCompat.calc_volatility(returns, trading_days)
        if vol == 0:
            return 0.0
        ann_return = returns.mean() * trading_days
        return float((ann_return - risk_free) / vol)

    @staticmethod
    def calc_max_drawdown(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return float(drawdown.min())

    @staticmethod
    def calc_calmar(equity_curve: pd.Series, trading_days: int = 252) -> float:
        ann_return = BaseBacktestMetricsCompat.calc_annual_return(equity_curve, trading_days)
        mdd = abs(BaseBacktestMetricsCompat.calc_max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        return float(ann_return / mdd)

    @staticmethod
    def calc_win_rate(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        winning = (trades["pnl"] > 0).sum()
        total = len(trades)
        return float(winning / total) if total > 0 else 0.0

    @staticmethod
    def calc_sortino(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        negative_returns = returns[returns < 0]
        if len(negative_returns) < 2:
            return 0.0
        downside_std = negative_returns.std() * np.sqrt(trading_days)
        if downside_std == 0:
            return 0.0
        ann_return = returns.mean() * trading_days
        return float((ann_return - risk_free) / downside_std)

    @staticmethod
    def calc_all_metrics(
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        risk_free: float = 0.03,
        trading_days: int = 252,
    ) -> Dict[str, Any]:
        returns = equity_curve.pct_change().dropna()
        return {
            "total_return": BaseBacktestMetricsCompat.calc_total_return(equity_curve),
            "annual_return": BaseBacktestMetricsCompat.calc_annual_return(equity_curve, trading_days),
            "volatility": BaseBacktestMetricsCompat.calc_volatility(returns, trading_days),
            "sharpe_ratio": BaseBacktestMetricsCompat.calc_sharpe(returns, risk_free, trading_days),
            "max_drawdown": BaseBacktestMetricsCompat.calc_max_drawdown(equity_curve),
            "calmar_ratio": BaseBacktestMetricsCompat.calc_calmar(equity_curve, trading_days),
            "sortino_ratio": BaseBacktestMetricsCompat.calc_sortino(returns, risk_free, trading_days),
            "win_rate": BaseBacktestMetricsCompat.calc_win_rate(trades),
            "total_trades": len(trades),
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
