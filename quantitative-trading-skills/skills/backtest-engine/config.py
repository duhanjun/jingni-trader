import os
from typing import Optional, Dict, Any


class Config:
    """
    回测引擎配置类
    """

    def __init__(self, **kwargs):
        self.BACKTEST_ENGINE: str = kwargs.get(
            "BACKTEST_ENGINE", os.environ.get("BACKTEST_ENGINE", "rqalpha")
        )
        self.REPORT_DIR: str = kwargs.get(
            "REPORT_DIR", os.environ.get("REPORT_DIR", "./reports")
        )
        self.GM_TOKEN: Optional[str] = kwargs.get("GM_TOKEN", os.environ.get("GM_TOKEN"))

        # A股交易规则配置
        self.ENABLE_T1: bool = kwargs.get("ENABLE_T1", True)
        self.LIMIT_UP_DOWN_MODEL: str = kwargs.get(
            "LIMIT_UP_DOWN_MODEL", "strict"
        )
        self.COMMISSION_RATE: float = kwargs.get("COMMISSION_RATE", 0.00025)
        self.COMMISSION_MIN: float = kwargs.get("COMMISSION_MIN", 5.0)
        self.STAMP_DUTY_RATE: float = kwargs.get("STAMP_DUTY_RATE", 0.001)
        self.TRANSFER_FEE_RATE: float = kwargs.get("TRANSFER_FEE_RATE", 0.00002)
        self.ST_LIMIT_RATE: float = kwargs.get("ST_LIMIT_RATE", 0.05)
        self.NORMAL_LIMIT_RATE: float = kwargs.get("NORMAL_LIMIT_RATE", 0.10)

        # 绩效分析配置
        self.RISK_FREE_RATE: float = kwargs.get("RISK_FREE_RATE", 0.03)
        self.ANNUALIZATION_FACTOR: int = kwargs.get("ANNUALIZATION_FACTOR", 252)

        # Walk-Forward 配置
        self.WALK_FORWARD_TRAIN_WINDOW: int = kwargs.get(
            "WALK_FORWARD_TRAIN_WINDOW", 252
        )
        self.WALK_FORWARD_TEST_WINDOW: int = kwargs.get(
            "WALK_FORWARD_TEST_WINDOW", 63
        )

        # 可视化配置
        self.PLOT_BACKEND: str = kwargs.get("PLOT_BACKEND", "matplotlib")


def get_config(**kwargs) -> Config:
    """
    获取配置实例

    Args:
        **kwargs: 可选的配置覆盖

    Returns:
        Config 实例
    """
    return Config(**kwargs)
