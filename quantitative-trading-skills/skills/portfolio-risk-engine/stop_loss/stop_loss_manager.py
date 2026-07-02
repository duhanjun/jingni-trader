import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class StopLossManager:
    """
    止损管理器
    """

    def __init__(
        self,
        daily_drawdown_threshold: float = -0.03,
        ma_period: int = 20,
        volume_ratio_threshold: float = 2.0
    ):
        self.daily_drawdown_threshold = daily_drawdown_threshold
        self.ma_period = ma_period
        self.volume_ratio_threshold = volume_ratio_threshold

    def check_daily_drawdown(self, returns: pd.Series) -> Dict:
        """
        检查单日回撤止损
        
        Args:
            returns: 收益率 Series
        
        Returns:
            检查结果字典
        """
        latest_return = returns.iloc[-1] if not returns.empty else 0
        
        return {
            "triggered": latest_return <= self.daily_drawdown_threshold,
            "latest_return": latest_return,
            "threshold": self.daily_drawdown_threshold
        }

    def check_ma_stop_loss(
        self,
        prices: pd.Series,
        volumes: Optional[pd.Series] = None
    ) -> Dict:
        """
        检查均线止损（放量跌破 MA）
        
        Args:
            prices: 价格 Series
            volumes: 成交量 Series（可选）
        
        Returns:
            检查结果字典
        """
        if len(prices) < self.ma_period:
            return {
                "triggered": False,
                "reason": "insufficient_data"
            }
        
        # 计算 MA
        ma = prices.rolling(window=self.ma_period).mean()
        latest_price = prices.iloc[-1]
        latest_ma = ma.iloc[-1]
        
        # 检查跌破 MA
        below_ma = latest_price < latest_ma
        
        # 检查放量
        volume_triggered = True
        if volumes is not None and len(volumes) >= self.ma_period:
            avg_volume = volumes.iloc[:-1].rolling(window=self.ma_period).mean().iloc[-1]
            latest_volume = volumes.iloc[-1]
            volume_triggered = latest_volume > avg_volume * self.volume_ratio_threshold
        
        return {
            "triggered": below_ma and volume_triggered,
            "latest_price": latest_price,
            "latest_ma": latest_ma,
            "below_ma": below_ma,
            "volume_triggered": volume_triggered
        }

    def check_portfolio_stop_loss(
        self,
        portfolio_returns: pd.Series,
        individual_prices: Optional[Dict[str, pd.Series]] = None,
        individual_volumes: Optional[Dict[str, pd.Series]] = None
    ) -> Dict:
        """
        检查组合止损
        
        Args:
            portfolio_returns: 组合收益率 Series
            individual_prices: 个股价格字典
            individual_volumes: 个股成交量字典
        
        Returns:
            检查结果字典
        """
        # 组合级别止损
        portfolio_drawdown = self.check_daily_drawdown(portfolio_returns)
        
        # 个股级别止损
        individual_stops = {}
        if individual_prices is not None:
            for stock, prices in individual_prices.items():
                volumes = individual_volumes.get(stock) if individual_volumes else None
                individual_stops[stock] = self.check_ma_stop_loss(prices, volumes)
        
        return {
            "portfolio": portfolio_drawdown,
            "individual": individual_stops,
            "any_triggered": (
                portfolio_drawdown["triggered"] or
                any(s["triggered"] for s in individual_stops.values())
            )
        }
