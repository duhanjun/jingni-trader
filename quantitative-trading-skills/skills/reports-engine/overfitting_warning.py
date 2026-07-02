import pandas as pd
import numpy as np
from typing import Optional, Tuple


class OverfittingWarning:
    """
    过拟合风险警示器
    """

    def __init__(self, config):
        self.config = config

    def check_overfitting(
        self,
        is_returns: pd.Series,
        oos_returns: pd.Series
    ) -> dict:
        """
        检查过拟合风险
        
        Args:
            is_returns: 样本内收益率
            oos_returns: 样本外收益率
        
        Returns:
            检测结果字典
        """
        is_sharpe = self._calculate_sharpe(is_returns)
        oos_sharpe = self._calculate_sharpe(oos_returns)
        
        ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else float("inf")
        has_warning = ratio < self.config.OVERFITTING_THRESHOLD
        
        result = {
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "sharpe_ratio": ratio,
            "threshold": self.config.OVERFITTING_THRESHOLD,
            "has_warning": has_warning,
            "warning_message": self._generate_warning(has_warning, ratio)
        }
        
        return result

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """
        计算夏普比率
        """
        if len(returns) < 2:
            return np.nan
        
        excess_returns = returns - self.config.RISK_FREE_RATE / 252
        annualized_return = excess_returns.mean() * 252
        annualized_vol = excess_returns.std() * np.sqrt(252)
        
        if annualized_vol == 0:
            return np.nan
        
        return annualized_return / annualized_vol

    def _generate_warning(self, has_warning: bool, ratio: float) -> str:
        """
        生成警告信息
        """
        if has_warning:
            return f"警告：样本外夏普比率仅为样本内的 {ratio:.1%}，存在过拟合风险！"
        else:
            return f"样本外夏普比率为样本内的 {ratio:.1%}，过拟合风险较低。"
