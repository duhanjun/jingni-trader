import pandas as pd
from typing import List
import warnings
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseDataProvider


class xtquantAdapter(BaseDataProvider):
    """
    xtquant 数据适配器（占位实现）
    """

    def __init__(self):
        warnings.warn("xtquant adapter is a placeholder. Full implementation requires xtquant library.")

    def get_daily(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        warnings.warn("xtquant adapter get_daily not implemented")
        return pd.DataFrame()

    def get_stock_list(self) -> pd.DataFrame:
        warnings.warn("xtquant adapter get_stock_list not implemented")
        return pd.DataFrame()

    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        warnings.warn("xtquant adapter get_trading_calendar not implemented")
        return []
