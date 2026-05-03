import os
from typing import Optional, Dict, Any
from pydantic import BaseModel


class Config(BaseModel):
    """
    配置类
    """
    EXECUTION_BACKEND: str = "sim"
    PAPER_TRADE: bool = True
    INITIAL_CAPITAL: float = 1000000.0
    LOG_DIR: str = "./logs"
    TRADE_LOG_DIR: str = "./trade_logs"
    
    XTQUANT_TOKEN: Optional[str] = None
    GM_TOKEN: Optional[str] = None
    VNPY_CONFIG: Optional[Dict[str, Any]] = None
    
    MAX_DAILY_LOSS_RATE: float = 0.02
    MAX_SINGLE_ORDER_AMOUNT: float = 100000.0
    MAX_ORDER_FREQUENCY_PER_MINUTE: int = 10
    MAX_POSITION_CONCENTRATION: float = 0.3
    
    COMMISSION_RATE: float = 0.00025
    MIN_COMMISSION: float = 5.0
    SLIPPAGE: float = 0.001


def get_config(**kwargs) -> Config:
    """
    获取配置实例
    
    Args:
        **kwargs: 可选的配置覆盖
    
    Returns:
        Config 实例
    """
    config_dict = {
        "EXECUTION_BACKEND": kwargs.get("EXECUTION_BACKEND", os.environ.get("EXECUTION_BACKEND", "sim")),
        "PAPER_TRADE": kwargs.get("PAPER_TRADE", os.environ.get("PAPER_TRADE", "true").lower() == "true"),
        "INITIAL_CAPITAL": float(kwargs.get("INITIAL_CAPITAL", os.environ.get("INITIAL_CAPITAL", "1000000"))),
        "LOG_DIR": kwargs.get("LOG_DIR", "./logs"),
        "TRADE_LOG_DIR": kwargs.get("TRADE_LOG_DIR", "./trade_logs"),
        "XTQUANT_TOKEN": kwargs.get("XTQUANT_TOKEN", os.environ.get("XTQUANT_TOKEN")),
        "GM_TOKEN": kwargs.get("GM_TOKEN", os.environ.get("GM_TOKEN")),
        "MAX_DAILY_LOSS_RATE": float(kwargs.get("MAX_DAILY_LOSS_RATE", 0.02)),
        "MAX_SINGLE_ORDER_AMOUNT": float(kwargs.get("MAX_SINGLE_ORDER_AMOUNT", 100000)),
        "MAX_ORDER_FREQUENCY_PER_MINUTE": int(kwargs.get("MAX_ORDER_FREQUENCY_PER_MINUTE", 10)),
        "MAX_POSITION_CONCENTRATION": float(kwargs.get("MAX_POSITION_CONCENTRATION", 0.3)),
        "COMMISSION_RATE": float(kwargs.get("COMMISSION_RATE", 0.00025)),
        "MIN_COMMISSION": float(kwargs.get("MIN_COMMISSION", 5.0)),
        "SLIPPAGE": float(kwargs.get("SLIPPAGE", 0.001)),
    }
    return Config(**config_dict)
