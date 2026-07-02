import pandas as pd
from typing import List
import warnings
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import BaseDataProvider

try:
    import akshare as ak
except ImportError:
    ak = None


class AkShareAdapter(BaseDataProvider):
    """
    AkShare 数据适配器
    """

    def __init__(self):
        if ak is None:
            raise ImportError("akshare is not installed. Please install it with 'pip install akshare'")

    def get_daily(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        adj: str = "qfq",
    ) -> pd.DataFrame:
        codes = self.normalize_codes(codes)
        all_data = []

        for code in codes:
            try:
                symbol = code.split(".")[0]
                if adj == "qfq":
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""), adjust="qfq")
                elif adj == "hfq":
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""), adjust="hfq")
                else:
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""), adjust="")
                
                if df is not None and not df.empty:
                    df["code"] = code
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
        rename_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
            "涨跌幅": "change_pct",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        df["volume"] = df["volume"] / 100 if "volume" in df.columns else 0
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        return df

    def _add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "pre_close" not in df.columns:
            df["pre_close"] = df.groupby("code")["close"].shift(1)
        df["is_st"] = df["code"].apply(lambda x: "ST" in x)
        df["is_limit_up"] = False
        df["is_limit_down"] = False
        return df

    def get_stock_list(self) -> pd.DataFrame:
        df = ak.stock_info_a_code_name()
        df = df.rename(columns={"code": "symbol", "name": "name"})
        df["code"] = df["symbol"].apply(self.normalize_code)
        return df

    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        df = ak.tool_trade_date_hist_sina()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
        return df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
