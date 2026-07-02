import pandas as pd
from typing import List, Optional
import sys
import os
# 确保模块可以正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from base import BaseDataProvider
from adapters import (
    TushareAdapter,
    BaoStockAdapter,
    AkShareAdapter,
    xtquantAdapter,
    gmAdapter,
)
from cleaning import DataCleaner


class AShareDataEngine:
    """
    A股数据引擎主类
    """

    def __init__(self, config: Config):
        self.config = config
        self.provider = self._create_provider()
        self.cleaner = DataCleaner(config)

    def _create_provider(self) -> BaseDataProvider:
        """
        根据配置创建数据源提供者
        """
        backend = self.config.DATA_BACKEND.lower()
        
        if backend == "tushare":
            return TushareAdapter(self.config.TUSHARE_TOKEN)
        elif backend == "baostock":
            return BaoStockAdapter()
        elif backend == "akshare":
            return AkShareAdapter()
        elif backend == "xtquant":
            return xtquantAdapter()
        elif backend == "gm":
            return gmAdapter(self.config.GM_TOKEN)
        else:
            raise ValueError(f"Unsupported data backend: {backend}")

    def get_daily(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adj: Optional[str] = None,
        clean: bool = True,
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adj: 复权类型，默认使用配置
            clean: 是否清洗数据
        
        Returns:
            数据 DataFrame
        """
        if adj is None:
            adj = self.config.ADJ_TYPE
        
        df = self.provider.get_daily(codes, start_date, end_date, adj)
        
        if clean and self.config.CLEANING_ENABLED and not df.empty:
            df = self.cleaner.clean(df)
        
        return df

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取股票列表
        """
        return self.provider.get_stock_list()

    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        """
        return self.provider.get_trading_calendar(start_date, end_date)

    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        检查数据质量
        
        Args:
            df: 数据 DataFrame
        
        Returns:
            质量报告
        """
        if df.empty:
            return {"error": "Empty DataFrame"}
        
        missing_rate = self.cleaner.calculate_missing_rate(df)
        return {
            "total_rows": len(df),
            "unique_codes": df["code"].nunique(),
            "date_range": (df["date"].min(), df["date"].max()),
            "missing_rate": missing_rate,
            "pass_missing_check": missing_rate < 0.02,
        }
