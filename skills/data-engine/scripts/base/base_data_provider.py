"""
数据提供者抽象基类
所有数据源适配器必须实现此接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """A股数据提供者基类"""

    @abstractmethod
    def get_daily(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        adjust: str = "hfq"
    ) -> pd.DataFrame:
        """
        获取日线行情数据

        参数:
            symbols: 股票代码列表，如 ['000001.SZ', '600000.SH']
            start_date: 开始日期 YYYYMMDD 或 YYYY-MM-DD
            end_date: 结束日期
            adjust: 复权方式 'hfq'(后复权), 'qfq'(前复权), ''(不复权)

        返回:
            DataFrame，必须包含列:
            code, date, open, high, low, close, volume, amount,
            pre_close, change_pct, turnover_rate,
            is_st, is_limit_up, is_limit_down, listed_days
        """
        ...

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全市场股票列表及其状态

        返回:
            DataFrame，包含列: code, name, industry, list_date, is_st
        """
        ...

    @abstractmethod
    def get_adj_factor(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取复权因子

        返回:
            DataFrame，包含列: code, date, adj_factor
        """
        ...

    @abstractmethod
    def get_financial(
        self,
        symbols: List[str],
        report_date: str,
        fields: List[str]
    ) -> pd.DataFrame:
        """
        获取财务数据

        返回:
            DataFrame，每行一只股票一个报告期
        """
        ...
