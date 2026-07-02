import os
from typing import Optional, Dict, Any


class Config:
    """
    配置类
    """

    def __init__(self, **kwargs):
        self.REPORT_DIR: str = kwargs.get("REPORT_DIR", os.environ.get("REPORT_DIR", "./reports"))
        self.BENCHMARK_CODE: str = kwargs.get("BENCHMARK_CODE", os.environ.get("BENCHMARK_CODE", "000300.SH"))
        self.RISK_FREE_RATE: float = kwargs.get("RISK_FREE_RATE", 0.03)
        
        self.ROLLING_WINDOW: int = kwargs.get("ROLLING_WINDOW", 252)
        self.DECAY_PERIODS: int = kwargs.get("DECAY_PERIODS", 20)
        
        self.OVERFITTING_THRESHOLD: float = kwargs.get("OVERFITTING_THRESHOLD", 0.6)
        
        self.PLOT_STYLE: str = kwargs.get("PLOT_STYLE", "seaborn-v0_8")
        self.FIGURE_SIZE: tuple = kwargs.get("FIGURE_SIZE", (12, 8))
        
        os.makedirs(self.REPORT_DIR, exist_ok=True)


def get_config(**kwargs) -> Config:
    """
    获取配置实例
    
    Args:
        **kwargs: 可选的配置覆盖
    
    Returns:
        Config 实例
    """
    return Config(**kwargs)
