import os
from typing import Optional, Dict, Any


class Config:
    """
    配置类
    """

    def __init__(self, **kwargs):
        self.FACTOR_DATA_DIR: str = kwargs.get(
            "FACTOR_DATA_DIR",
            os.environ.get("FACTOR_DATA_DIR", "./factor_data")
        )
        
        self.TA_LIBRARY: str = kwargs.get(
            "TA_LIBRARY",
            os.environ.get("TA_LIBRARY", "pandas-ta")
        )
        
        self.IC_LAG: int = kwargs.get("IC_LAG", 1)
        self.NEUTRALIZE_INDUSTRY: bool = kwargs.get("NEUTRALIZE_INDUSTRY", False)
        
        self.WINSORIZE_THRESHOLD: float = kwargs.get("WINSORIZE_THRESHOLD", 0.01)
        self.STANDARDIZE: bool = kwargs.get("STANDARDIZE", True)
        
        self.ENSEMBLE_METHOD: str = kwargs.get("ENSEMBLE_METHOD", "ic_weighted")


def get_config(**kwargs) -> Config:
    """
    获取配置实例
    
    Args:
        **kwargs: 可选的配置覆盖
    
    Returns:
        Config 实例
    """
    return Config(**kwargs)
