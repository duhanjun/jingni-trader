from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """
    A股数据提供者抽象基类
    """

    @abstractmethod
    def get_daily(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            codes: 股票代码列表，格式如 ["000001.SZ", "600000.SH"]
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD
            adj: 复权类型，"qfq"前复权，"hfq"后复权，"none"不复权
        
        Returns:
            DataFrame，包含以下字段：
            - code: 股票代码
            - date: 交易日期
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量（手）
            - amount: 成交额（元）
            - pre_close: 前收盘价
            - change_pct: 涨跌幅（%）
            - turnover_rate: 换手率（%）
            - is_st: 是否ST
            - is_limit_up: 是否涨停
            - is_limit_down: 是否跌停
        """
        pass

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        
        Returns:
            DataFrame，包含股票基本信息
        """
        pass

    @abstractmethod
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            交易日列表
        """
        pass

    def normalize_code(self, code: str) -> str:
        """
        规范化股票代码格式
        
        Args:
            code: 输入的股票代码
        
        Returns:
            规范化后的代码，如 000001.SZ
        """
        code = code.strip().upper()
        if "." in code:
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        return f"{code}.SZ"

    def normalize_codes(self, codes: List[str]) -> List[str]:
        """
        规范化股票代码列表
        """
        return [self.normalize_code(code) for code in codes]
