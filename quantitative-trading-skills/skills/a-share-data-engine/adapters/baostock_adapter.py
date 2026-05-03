import pandas as pd
from typing import List
import warnings
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseDataProvider

try:
    import baostock as bs
except ImportError:
    bs = None


class BaoStockAdapter(BaseDataProvider):
    """
    BaoStock 数据适配器
    """

    def __init__(self):
        if bs is None:
            raise ImportError("baostock is not installed. Please install it with 'pip install baostock'")
        lg = bs.login()
        if lg.error_code != "0":
            raise Exception(f"Baostock login failed: {lg.error_msg}")

    def __del__(self):
        try:
            bs.logout()
        except:
            pass

    def get_daily(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        codes = self.normalize_codes(codes)
        adjustflag = "3" if adj == "qfq" else "2" if adj == "hfq" else "1"
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

        all_data = []
        for code in codes:
            try:
                bs_code = code.lower().replace(".sz", ".sz").replace(".sh", ".sh")
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag=adjustflag,
                )
                data_list = []
                while (rs.error_code == "0") & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
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
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.rename(columns={
            "preclose": "pre_close",
            "turn": "turnover_rate",
            "pctChg": "change_pct",
        })
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        return df

    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_st"] = df["code"].apply(lambda x: "ST" in x)
        df["is_limit_up"] = False
        df["is_limit_down"] = False
        return df

    def get_stock_list(self) -> pd.DataFrame:
        rs = bs.query_all_stock(day="")
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        df["code"] = df["code"].str.upper()
        return df

    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        trading_days = df[df["is_trading_day"] == "1"]["calendar_date"].tolist()
        return trading_days
