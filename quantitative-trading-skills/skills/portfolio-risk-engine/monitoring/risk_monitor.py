import numpy as np
import pandas as pd
from typing import Dict, Optional


class RiskMonitor:
    """
    实时风险监控器
    """

    def __init__(
        self,
        max_position_limit: float = 10000000.0,
        margin_ratio: float = 0.2,
        profit_warning_level: float = -0.05
    ):
        self.max_position_limit = max_position_limit
        self.margin_ratio = margin_ratio
        self.profit_warning_level = profit_warning_level

    def check_position_limit(self, positions: Dict[str, float]) -> Dict:
        """
        检查持仓限额
        
        Args:
            positions: 持仓市值字典
        
        Returns:
            检查结果字典
        """
        total_position = sum(positions.values())
        
        violations = []
        for stock, position in positions.items():
            if position > self.max_position_limit:
                violations.append({
                    "stock": stock,
                    "position": position,
                    "limit": self.max_position_limit
                })
        
        return {
            "total_position": total_position,
            "within_limit": total_position <= self.max_position_limit and len(violations) == 0,
            "violations": violations
        }

    def check_margin_requirement(
        self,
        positions: Dict[str, float],
        equity: float
    ) -> Dict:
        """
        检查保证金要求
        
        Args:
            positions: 持仓市值字典
            equity: 权益
        
        Returns:
            检查结果字典
        """
        total_position = sum(positions.values())
        required_margin = total_position * self.margin_ratio
        
        margin_ratio = equity / total_position if total_position > 0 else 1.0
        
        return {
            "total_position": total_position,
            "equity": equity,
            "required_margin": required_margin,
            "margin_ratio": margin_ratio,
            "sufficient": equity >= required_margin
        }

    def check_profit_warning(
        self,
        returns: pd.Series,
        initial_investment: float = 1.0
    ) -> Dict:
        """
        检查盈亏预警
        
        Args:
            returns: 收益率 Series
            initial_investment: 初始投资
        
        Returns:
            检查结果字典
        """
        cumulative_return = (1 + returns).prod() - 1
        current_value = initial_investment * (1 + cumulative_return)
        pnl = current_value - initial_investment
        
        return {
            "cumulative_return": cumulative_return,
            "current_value": current_value,
            "pnl": pnl,
            "warning_triggered": cumulative_return <= self.profit_warning_level,
            "warning_level": self.profit_warning_level
        }

    def monitor_all(
        self,
        positions: Dict[str, float],
        equity: float,
        returns: pd.Series,
        initial_investment: float = 1.0
    ) -> Dict:
        """
        全面风险监控
        
        Args:
            positions: 持仓市值字典
            equity: 权益
            returns: 收益率 Series
            initial_investment: 初始投资
        
        Returns:
            监控结果字典
        """
        position_check = self.check_position_limit(positions)
        margin_check = self.check_margin_requirement(positions, equity)
        profit_check = self.check_profit_warning(returns, initial_investment)
        
        all_ok = (
            position_check["within_limit"] and
            margin_check["sufficient"] and
            not profit_check["warning_triggered"]
        )
        
        return {
            "all_ok": all_ok,
            "position": position_check,
            "margin": margin_check,
            "profit": profit_check
        }
