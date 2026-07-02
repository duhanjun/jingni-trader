import pandas as pd
from typing import List
import warnings
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseDataProvider

try:
    import tushare as ts
except ImportError:
    ts = None


class TushareAdapter(BaseDataProvider):
    """
    Tushare 数据适配器
    """

    def __init__(self, token: str = None):
        if ts is None:
            raise ImportError("tushare is not installed. Please install it with 'pip install tushare'")
        if token:
            ts.set_token(token)
        self.pro = ts.pro_api()

    def get_daily(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        codes = self.normalize_codes(codes)
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        all_data = []
        for code in codes:
            try:
                ts_code = code
                if adj == "qfq":
                    df = ts.pro_bar(ts_code=ts_code, adj="qfq", start_date=start_date, end_date=end_date)
                elif adj == "hfq":
                    df = ts.pro_bar(ts_code=ts_code, adj="hfq", start_date=start_date, end_date=end_date)
                else:
                    df = ts.pro_bar(ts_code=ts_code, adj=None, start_date=start_date, end_date=end_date)
                
                if df is not None and not df.empty:
                    df = self._normalize_data(df)
                    all_data.append(df)
            except Exception as e:
                warnings.warn(f"Failed to get data for {code}: {e}")

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result = self._add_derived_fields(result)
            return result
        return pd.DataFrame()

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["date"] = pd.to_datetime(df["trade_date"])
        df = df.rename(columns={
            "ts_code": "code",
            "vol": "volume",
        })
        cols = ["code", "date", "open", "high", "low", "close", "volume", "amount", "pre_close", "pct_chg"]
        existing_cols = [col for col in cols if col in df.columns]
        df = df[existing_cols]
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        return df

    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "pct_chg" in df.columns:
            df["change_pct"] = df["pct_chg"]
            df = df.drop("pct_chg", axis=1)
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = 0.0
        df["is_st"] = df["code"].apply(lambda x: "ST" in x)
        df["is_limit_up"] = False
        df["is_limit_down"] = False
        return df

    def get_stock_list(self) -> pd.DataFrame:
        df = self.pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,area,industry,list_date")
        df = df.rename(columns={"ts_code": "code"})
        return df

    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        df = self.pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date)
        trading_days = df[df["is_open"] == 1]["cal_date"].tolist()
        return [f"{day[:4]}-{day[4:6]}-{day[6:8]}" for day in trading_days]
