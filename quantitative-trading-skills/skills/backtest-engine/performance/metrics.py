from typing import Dict, Any
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class PerformanceMetrics:
    """
    绩效指标计算
    """

    @staticmethod
    def calculate(
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        config: Config,
    ) -> Dict[str, Any]:
        """
        计算绩效指标

        Args:
            equity_curve: 资金曲线
            trades_df: 交易记录
            config: 配置

        Returns:
            绩效指标 dict
        """
        if equity_curve.empty:
            return {}

        returns = equity_curve.pct_change().dropna()
        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]

        total_return = (final_value / initial_value) - 1

        n_days = len(equity_curve)
        annual_factor = config.ANNUALIZATION_FACTOR
        annual_return = (1 + total_return) ** (annual_factor / n_days) - 1

        volatility = returns.std() * np.sqrt(annual_factor)

        sharpe_ratio = (annual_return - config.RISK_FREE_RATE) / volatility if volatility > 0 else 0

        max_drawdown = PerformanceMetrics._calculate_max_drawdown(equity_curve)

        win_rate, profit_loss_ratio = PerformanceMetrics._calculate_trade_metrics(trades_df)

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "total_trades": len(trades_df) // 2,
            "initial_value": initial_value,
            "final_value": final_value,
        }

    @staticmethod
    def _calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        计算最大回撤

        Args:
            equity_curve: 资金曲线

        Returns:
            最大回撤
        """
        cumulative_max = equity_curve.cummax()
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return drawdown.min()

    @staticmethod
    def _calculate_trade_metrics(trades_df: pd.DataFrame) -> tuple[float, float]:
        """
        计算交易指标

        Args:
            trades_df: 交易记录

        Returns:
            (胜率, 盈亏比)
        """
        if trades_df.empty:
            return 0.0, 0.0

        profits = []
        buys = trades_df[trades_df["direction"] == "buy"].sort_values("date")
        sells = trades_df[trades_df["direction"] == "sell"].sort_values("date")

        min_len = min(len(buys), len(sells))
        for i in range(min_len):
            buy = buys.iloc[i]
            sell = sells.iloc[i]

            buy_cost = (
                buy["amount"]
                + buy["commission"]
                + buy["stamp_duty"]
                + buy["transfer_fee"]
            )
            sell_revenue = (
                sell["amount"]
                - sell["commission"]
                - sell["stamp_duty"]
                - sell["transfer_fee"]
            )

            profit = sell_revenue - buy_cost
            profits.append(profit)

        if not profits:
            return 0.0, 0.0

        profits_array = np.array(profits)
        winning_trades = profits_array[profits_array > 0]
        losing_trades = profits_array[profits_array <= 0]

        win_rate = len(winning_trades) / len(profits) if len(profits) > 0 else 0

        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 1

        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        return win_rate, profit_loss_ratio
