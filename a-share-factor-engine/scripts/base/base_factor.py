"""
因子计算器抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd


class BaseFactorCalculator(ABC):
    """因子计算器基类"""

    @abstractmethod
    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """
        批量计算因子

        参数:
            data: 包含 OHLCV 等原始数据的 DataFrame
                  必须列: code, date, open, high, low, close, volume, amount
            factor_names: 需要计算的因子名称列表

        返回:
            因子 DataFrame，列为 code, date, [各因子列]
        """
        ...

    @abstractmethod
    def get_available_factors(self) -> List[str]:
        """返回支持的所有因子名称列表"""
        ...

    @abstractmethod
    def get_factor_info(self, factor_name: str) -> Dict:
        """返回因子元信息（方向、说明、参数等）"""
        ...