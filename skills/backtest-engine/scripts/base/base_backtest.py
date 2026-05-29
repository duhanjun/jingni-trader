"""
回测引擎基类（轻量版）
提供绩效计算等通用方法，供各适配器复用
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime


class BaseBacktestMetrics:
    """
    回测绩效指标计算基类

    各后端适配器可继承此类以复用通用绩效计算方法。
    """

    @staticmethod
    def calc_total_return(equity_curve: pd.Series) -> float:
        """计算累计收益率"""
        if len(equity_curve) < 2:
            return 0.0
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    @staticmethod
    def calc_annual_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
        """计算年化收益率"""
        if len(equity_curve) < 2:
            return 0.0
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = len(equity_curve) / trading_days
        if n_years <= 0:
            return 0.0
        return float(total_return ** (1 / n_years) - 1)

    @staticmethod
    def calc_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """计算年化波动率"""
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days))

    @staticmethod
    def calc_sharpe(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        """计算夏普比率"""
        vol = BaseBacktestMetrics.calc_volatility(returns, trading_days)
        if vol == 0:
            return 0.0
        ann_return = returns.mean() * trading_days
        return float((ann_return - risk_free) / vol)

    @staticmethod
    def calc_max_drawdown(equity_curve: pd.Series) -> float:
        """计算最大回撤"""
        if len(equity_curve) < 2:
            return 0.0
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return float(drawdown.min())

    @staticmethod
    def calc_calmar(equity_curve: pd.Series, trading_days: int = 252) -> float:
        """计算Calmar比率"""
        ann_return = BaseBacktestMetrics.calc_annual_return(equity_curve, trading_days)
        mdd = abs(BaseBacktestMetrics.calc_max_drawdown(equity_curve))
        if mdd == 0:
            return 0.0
        return float(ann_return / mdd)

    @staticmethod
    def calc_win_rate(trades: pd.DataFrame) -> float:
        """
        计算胜率

        参数:
            trades: 含 pnl 列的成交记录
        """
        if trades.empty:
            return 0.0
        winning = (trades["pnl"] > 0).sum()
        total = len(trades)
        return float(winning / total) if total > 0 else 0.0

    @staticmethod
    def calc_sortino(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        """计算索提诺比率（下行风险调整）"""
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
        trading_days: int = 252
    ) -> Dict[str, Any]:
        """
        一次性计算所有常用绩效指标

        返回:
            包含所有绩效指标的字典
        """
        returns = equity_curve.pct_change().dropna()
        return {
            "total_return": BaseBacktestMetrics.calc_total_return(equity_curve),
            "annual_return": BaseBacktestMetrics.calc_annual_return(equity_curve, trading_days),
            "volatility": BaseBacktestMetrics.calc_volatility(returns, trading_days),
            "sharpe_ratio": BaseBacktestMetrics.calc_sharpe(returns, risk_free, trading_days),
            "max_drawdown": BaseBacktestMetrics.calc_max_drawdown(equity_curve),
            "calmar_ratio": BaseBacktestMetrics.calc_calmar(equity_curve, trading_days),
            "sortino_ratio": BaseBacktestMetrics.calc_sortino(returns, risk_free, trading_days),
            "win_rate": BaseBacktestMetrics.calc_win_rate(trades),
            "total_trades": len(trades),
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }