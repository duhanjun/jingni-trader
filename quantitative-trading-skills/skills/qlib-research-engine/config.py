import os
from typing import Optional, Dict, Any


class Config:
    """
    Qlib研究引擎配置类
    """

    def __init__(self, **kwargs):
        self.QLIB_DATA_DIR: str = kwargs.get(
            "QLIB_DATA_DIR", os.environ.get("QLIB_DATA_DIR", "./qlib_data")
        )
        self.QLIB_CACHE_DIR: str = kwargs.get(
            "QLIB_CACHE_DIR", os.environ.get("QLIB_CACHE_DIR", "./qlib_cache")
        )
        
        # 数据配置
        self.DATA_FREQUENCY: str = kwargs.get("DATA_FREQUENCY", "day")
        self.MARKET: str = kwargs.get("MARKET", "cn")
        
        # 模型配置
        self.DEFAULT_MODEL_TYPE: str = kwargs.get("DEFAULT_MODEL_TYPE", "lightgbm")
        self.N_EPOCHS: int = kwargs.get("N_EPOCHS", 100)
        self.EARLY_STOPPING_ROUNDS: int = kwargs.get("EARLY_STOPPING_ROUNDS", 20)
        
        # 因子配置
        self.FACTOR_LIBRARY: str = kwargs.get("FACTOR_LIBRARY", "alpha360")
        
        # 评估配置
        self.EVALUATION_METRICS: list = kwargs.get(
            "EVALUATION_METRICS", ["IC", "IR", "ACC", "F1"]
        )
        
        # 日志配置
        self.LOG_LEVEL: str = kwargs.get("LOG_LEVEL", "INFO")


def get_config(**kwargs) -> Config:
    """
    获取配置实例

    Args:
        **kwargs: 可选的配置覆盖

    Returns:
        Config 实例
    """
    return Config(**kwargs)