"""
数据源适配器基类（工厂模式）
用于统一创建不同数据源适配器实例
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDataSource(ABC):
    """
    数据源抽象基类

    与 BaseDataProvider 不同的是，本基类面向"数据源产品"层面，
    定义了数据源需要实现的完整接口，包括连接、断开、能力查询等。
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        建立与数据源的连接

        返回:
            是否连接成功
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """断开与数据源的连接"""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查是否已连接

        返回:
            是否处于连接状态
        """
        ...

    @abstractmethod
    def get_supported_features(self) -> List[str]:
        """
        查询该数据源支持的功能列表

        返回:
            功能名称列表，如 ['daily', 'minute', 'financial',
            'block_trade', 'north_flow']
        """
        ...

    @abstractmethod
    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取日线行情

        返回:
            DataFrame，标准列名
        """
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False