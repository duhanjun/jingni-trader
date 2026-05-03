import os
from typing import Optional, Dict, Any


class Config:
    """
    投资组合风险引擎配置类
    """

    def __init__(self, **kwargs):
        # 协方差估计配置
        self.COVARIANCE_METHOD: str = kwargs.get("COVARIANCE_METHOD", "historical")
        self.EWMA_SPAN: int = kwargs.get("EWMA_SPAN", 60)

        # 组合优化配置
        self.OPTIMIZATION_METHOD: str = kwargs.get("OPTIMIZATION_METHOD", "max_sharpe")
        self.RISK_FREE_RATE: float = kwargs.get("RISK_FREE_RATE", 0.03)

        # A股约束配置
        self.MAX_SINGLE_WEIGHT: float = kwargs.get("MAX_SINGLE_WEIGHT", 0.10)
        self.MIN_SINGLE_WEIGHT: float = kwargs.get("MIN_SINGLE_WEIGHT", 0.0)
        self.MAX_INDUSTRY_DEVIATION: float = kwargs.get("MAX_INDUSTRY_DEVIATION", 0.05)

        # 止损配置
        self.DAILY_DRAWDOWN_THRESHOLD: float = kwargs.get("DAILY_DRAWDOWN_THRESHOLD", -0.03)
        self.MA_PERIOD: int = kwargs.get("MA_PERIOD", 20)
        self.VOLUME_RATIO_THRESHOLD: float = kwargs.get("VOLUME_RATIO_THRESHOLD", 2.0)

        # VaR 配置
        self.VAR_METHOD: str = kwargs.get("VAR_METHOD", "historical")
        self.CONFIDENCE_LEVEL: float = kwargs.get("CONFIDENCE_LEVEL", 0.95)
        self.VAR_HORIZON: int = kwargs.get("VAR_HORIZON", 1)

        # 实时监控配置
        self.MAX_POSITION_LIMIT: float = kwargs.get("MAX_POSITION_LIMIT", 10000000.0)
        self.MARGIN_RATIO: float = kwargs.get("MARGIN_RATIO", 0.2)
        self.PROFIT_WARNING_LEVEL: float = kwargs.get("PROFIT_WARNING_LEVEL", -0.05)


def get_config(**kwargs) -> Config:
    """
    获取配置实例
    
    Args:
        **kwargs: 可选的配置覆盖
    
    Returns:
        Config 实例
    """
    return Config(**kwargs)
