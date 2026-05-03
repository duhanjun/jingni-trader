import pandas as pd
import numpy as np
from typing import Optional
import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class DataCleaner:
    """
    A股数据清洗器
    """

    def __init__(self, config: Config):
        self.config = config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的清洗流程
        
        Args:
            df: 原始数据
        
        Returns:
            清洗后的数据
        """
        df = df.copy()
        df = self._handle_missing_data(df)
        df = self._mark_suspension(df)
        df = self._mark_limit_up_down(df)
        df = self._filter_st(df)
        df = self._filter_new_stocks(df)
        df = self._fill_missing_dates(df)
        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失数据
        """
        numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pre_close", "change_pct", "turnover_rate"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["code", "date"])
        return df

    def _mark_suspension(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记停牌
        """
        if "volume" in df.columns:
            df["is_suspended"] = df["volume"] == 0
        else:
            df["is_suspended"] = False
        return df

    def _mark_limit_up_down(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标记涨跌停
        """
        if "change_pct" in df.columns and "pre_close" in df.columns:
            df["is_limit_up"] = (df["change_pct"] >= 9.8) & (df["change_pct"] <= 10.2)
            df["is_limit_down"] = (df["change_pct"] <= -9.8) & (df["change_pct"] >= -10.2)
        return df

    def _filter_st(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤ST股票
        """
        if self.config.FILTER_ST:
            df = df[~df["is_st"]]
        return df

    def _filter_new_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤新股（上市不足N天）
        """
        if self.config.FILTER_NEW_STOCK_DAYS > 0:
            if "list_date" not in df.columns:
                return df
            df["list_date"] = pd.to_datetime(df["list_date"])
            df["days_since_list"] = (df["date"] - df["list_date"]).dt.days
            df = df[df["days_since_list"] >= self.config.FILTER_NEW_STOCK_DAYS]
        return df

    def _fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失的交易日
        """
        if df.empty:
            return df

        dfs = []
        for code in df["code"].unique():
            code_df = df[df["code"] == code].sort_values("date").copy()
            if len(code_df) < 2:
                dfs.append(code_df)
                continue
            
            date_range = pd.date_range(start=code_df["date"].min(), end=code_df["date"].max(), freq="D")
            code_df = code_df.set_index("date")
            code_df = code_df.reindex(date_range)
            code_df["code"] = code
            code_df = code_df.reset_index().rename(columns={"index": "date"})
            dfs.append(code_df)
        
        result = pd.concat(dfs, ignore_index=True)
        return result

    def calculate_missing_rate(self, df: pd.DataFrame) -> float:
        """
        计算数据缺失率
        """
        if df.empty:
            return 1.0
        total_possible = len(df["code"].unique()) * len(df["date"].unique()) if "date" in df.columns else len(df)
        actual = len(df)
        return 1.0 - (actual / total_possible)
