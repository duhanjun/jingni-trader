import pandas as pd
import numpy as np
from typing import List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FactorBuilder:
    """
    因子构建器
    """

    def __init__(self, config: Config):
        self.config = config

    def build_factors(
        self,
        price_data: pd.DataFrame,
        factor_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        构建因子
        
        Args:
            price_data: 价格数据，包含 code, date, open, high, low, close, volume, amount, turnover_rate 等字段
            factor_list: 因子名称列表，默认为所有因子
        
        Returns:
            因子 DataFrame
        """
        if price_data.empty:
            return pd.DataFrame()
        
        df = price_data.copy()
        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        
        if factor_list is None:
            factor_list = self._get_all_factors()
        
        result = df[["code", "date"]].copy()
        
        for factor in factor_list:
            if hasattr(self, f"_compute_{factor}"):
                result[factor] = getattr(self, f"_compute_{factor}")(df)
        
        return result

    def _get_all_factors(self) -> List[str]:
        """
        获取所有因子名称
        """
        return [
            "momentum_1m",
            "momentum_3m",
            "reversal_1m",
            "lncap",
            "turnover",
            "volatility_20d",
            "rsi_14",
            "macd",
            "bollinger",
            "kdj_k",
            "kdj_d",
            "willr"
        ]

    def _compute_momentum_1m(self, df: pd.DataFrame) -> pd.Series:
        """
        1个月动量因子
        """
        return df.groupby("code")["close"].pct_change(20)

    def _compute_momentum_3m(self, df: pd.DataFrame) -> pd.Series:
        """
        3个月动量因子
        """
        return df.groupby("code")["close"].pct_change(60)

    def _compute_reversal_1m(self, df: pd.DataFrame) -> pd.Series:
        """
        1个月反转因子
        """
        return -df.groupby("code")["close"].pct_change(20)

    def _compute_lncap(self, df: pd.DataFrame) -> pd.Series:
        """
        市值因子
        """
        if "market_cap" in df.columns:
            return np.log(df["market_cap"])
        elif "amount" in df.columns:
            return np.log(df["close"] * df["volume"] * 100)
        else:
            return np.log(df["close"])

    def _compute_turnover(self, df: pd.DataFrame) -> pd.Series:
        """
        换手率因子
        """
        if "turnover_rate" in df.columns:
            return df["turnover_rate"]
        else:
            return df.groupby("code")["volume"].rolling(20).mean().reset_index(0, drop=True)

    def _compute_volatility_20d(self, df: pd.DataFrame) -> pd.Series:
        """
        20日波动率
        """
        df = df.copy()
        df["returns"] = df.groupby("code")["close"].pct_change()
        return df.groupby("code")["returns"].rolling(20).std().reset_index(0, drop=True)

    def _compute_rsi_14(self, df: pd.DataFrame) -> pd.Series:
        """
        RSI 14日
        """
        delta = df.groupby("code")["close"].apply(
            lambda x: x.diff()
        ).reset_index(0, drop=True)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.groupby(df["code"]).rolling(14).mean().reset_index(0, drop=True)
        avg_loss = loss.groupby(df["code"]).rolling(14).mean().reset_index(0, drop=True)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, df: pd.DataFrame) -> pd.Series:
        """
        MACD
        """
        def compute_macd_single(series):
            ema12 = series.ewm(span=12, adjust=False).mean()
            ema26 = series.ewm(span=26, adjust=False).mean()
            return ema12 - ema26
        return df.groupby("code")["close"].apply(compute_macd_single).reset_index(0, drop=True)

    def _compute_bollinger(self, df: pd.DataFrame) -> pd.Series:
        """
        布林带宽度因子
        """
        def compute_boll_single(series):
            sma20 = series.rolling(20).mean()
            std20 = series.rolling(20).std()
            return (series - sma20) / std20
        return df.groupby("code")["close"].apply(compute_boll_single).reset_index(0, drop=True)

    def _compute_kdj_k(self, df: pd.DataFrame) -> pd.Series:
        """
        KDJ K值
        """
        def compute_kdj_single(data):
            low = data["low"].rolling(9).min()
            high = data["high"].rolling(9).max()
            rsv = (data["close"] - low) / (high - low) * 100
            k = rsv.ewm(com=2, adjust=False).mean()
            return k
        return df.groupby("code").apply(compute_kdj_single).reset_index(0, drop=True)

    def _compute_kdj_d(self, df: pd.DataFrame) -> pd.Series:
        """
        KDJ D值
        """
        def compute_kdj_d_single(data):
            low = data["low"].rolling(9).min()
            high = data["high"].rolling(9).max()
            rsv = (data["close"] - low) / (high - low) * 100
            k = rsv.ewm(com=2, adjust=False).mean()
            d = k.ewm(com=2, adjust=False).mean()
            return d
        return df.groupby("code").apply(compute_kdj_d_single).reset_index(0, drop=True)

    def _compute_willr(self, df: pd.DataFrame) -> pd.Series:
        """
        威廉指标
        """
        def compute_willr_single(data):
            low = data["low"].rolling(14).min()
            high = data["high"].rolling(14).max()
            return (high - data["close"]) / (high - low) * (-100)
        return df.groupby("code").apply(compute_willr_single).reset_index(0, drop=True)

    def winsorize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        缩尾处理
        """
        result = factors.copy()
        factor_cols = [col for col in result.columns if col not in ["code", "date"]]
        for col in factor_cols:
            lower = result[col].quantile(self.config.WINSORIZE_THRESHOLD)
            upper = result[col].quantile(1 - self.config.WINSORIZE_THRESHOLD)
            result[col] = result[col].clip(lower, upper)
        return result

    def standardize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        标准化处理
        """
        result = factors.copy()
        factor_cols = [col for col in result.columns if col not in ["code", "date"]]
        for col in factor_cols:
            mean = result.groupby("date")[col].transform("mean")
            std = result.groupby("date")[col].transform("std")
            result[col] = (result[col] - mean) / std
        return result
