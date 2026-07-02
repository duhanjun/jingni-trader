"""
交易日历
借鉴: quant-stream 的 execution_to_signal 日期映射
"""
import pandas as pd
from typing import Dict, List, Optional


class TradingCalendar:
    """交易日历，处理信号日 → 执行日的 T+1 映射"""

    def __init__(self, trading_dates: Optional[pd.DatetimeIndex] = None):
        if trading_dates is not None:
            self.dates = pd.DatetimeIndex(sorted(set(trading_dates)))
        else:
            self.dates = pd.DatetimeIndex([])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, date_col: str = "date") -> "TradingCalendar":
        return cls(pd.to_datetime(df[date_col].unique()))

    def next_trading_day(self, date) -> Optional[pd.Timestamp]:
        date = pd.Timestamp(date)
        future = self.dates[self.dates > date]
        return future[0] if len(future) > 0 else None

    def get_execution_date(self, signal_date) -> Optional[pd.Timestamp]:
        return self.next_trading_day(signal_date)

    def build_execution_map(
        self, signal_dates: List[pd.Timestamp]
    ) -> Dict[pd.Timestamp, pd.Timestamp]:
        return {d: self.get_execution_date(d) for d in signal_dates if self.get_execution_date(d) is not None}