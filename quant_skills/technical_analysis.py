"""
技术分析模块
提供常用技术指标计算功能
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Tuple


class TechnicalAnalysis:
    """
    技术分析工具类
    提供各类技术指标的计算接口
    """
    
    def __init__(self):
        self._pandas_ta_available = False
        self._talib_available = False
        
        try:
            import pandas_ta as pta
            self._pta = pta
            self._pandas_ta_available = True
        except ImportError:
            pass
            
        try:
            import talib
            self._talib = talib
            self._talib_available = True
        except ImportError:
            pass
    
    def sma(
        self,
        data: pd.DataFrame,
        period: int = 20,
        price_col: str = "close"
    ) -> pd.Series:
        """
        简单移动平均线 (SMA)
        
        参数:
            data: 数据
            period: 周期
            price_col: 价格列
            
        返回:
            pd.Series: SMA
        """
        return data[price_col].rolling(window=period).mean()
    
    def ema(
        self,
        data: pd.DataFrame,
        period: int = 20,
        price_col: str = "close"
    ) -> pd.Series:
        """
        指数移动平均线 (EMA)
        
        参数:
            data: 数据
            period: 周期
            price_col: 价格列
            
        返回:
            pd.Series: EMA
        """
        return data[price_col].ewm(span=period, adjust=False).mean()
    
    def rsi(
        self,
        data: pd.DataFrame,
        period: int = 14,
        price_col: str = "close"
    ) -> pd.Series:
        """
        相对强弱指标 (RSI)
        
        参数:
            data: 数据
            period: 周期
            price_col: 价格列
            
        返回:
            pd.Series: RSI
        """
        if self._pandas_ta_available:
            rsi = self._pta.rsi(data[price_col], length=period)
        elif self._talib_available:
            rsi = self._talib.RSI(data[price_col].values, timeperiod=period)
            rsi = pd.Series(rsi, index=data.index)
        else:
            # 手动实现
            delta = data[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_col: str = "close"
    ) -> Dict[str, pd.Series]:
        """
        MACD指标
        
        参数:
            data: 数据
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            price_col: 价格列
            
        返回:
            Dict: 包含macd、signal、histogram的字典
        """
        if self._pandas_ta_available:
            macd_df = self._pta.macd(
                data[price_col],
                fast=fast_period,
                slow=slow_period,
                signal=signal_period
            )
            return {
                "macd": macd_df.iloc[:, 0],
                "signal": macd_df.iloc[:, 1],
                "histogram": macd_df.iloc[:, 2]
            }
        elif self._talib_available:
            macd, signal, hist = self._talib.MACD(
                data[price_col].values,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period
            )
            return {
                "macd": pd.Series(macd, index=data.index),
                "signal": pd.Series(signal, index=data.index),
                "histogram": pd.Series(hist, index=data.index)
            }
        else:
            # 手动实现
            ema_fast = self.ema(data, fast_period, price_col)
            ema_slow = self.ema(data, slow_period, price_col)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            histogram = macd_line - signal_line
            
            return {
                "macd": macd_line,
                "signal": signal_line,
                "histogram": histogram
            }
    
    def bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        price_col: str = "close"
    ) -> Dict[str, pd.Series]:
        """
        布林带
        
        参数:
            data: 数据
            period: 周期
            std_dev: 标准差倍数
            price_col: 价格列
            
        返回:
            Dict: 包含upper、middle、lower的字典
        """
        middle = self.sma(data, period, price_col)
        std = data[price_col].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower
        }
    
    def kdj(
        self,
        data: pd.DataFrame,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close"
    ) -> Dict[str, pd.Series]:
        """
        KDJ指标
        
        参数:
            data: 数据
            n: 周期
            m1: K值周期
            m2: D值周期
            high_col: 最高价列
            low_col: 最低价列
            close_col: 收盘价列
            
        返回:
            Dict: 包含k、d、j的字典
        """
        low_min = data[low_col].rolling(window=n).min()
        high_max = data[high_col].rolling(window=n).max()
        
        rsv = (data[close_col] - low_min) / (high_max - low_min) * 100
        k = rsv.ewm(com=m1 - 1, adjust=False).mean()
        d = k.ewm(com=m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return {"k": k, "d": d, "j": j}
    
    def atr(
        self,
        data: pd.DataFrame,
        period: int = 14,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close"
    ) -> pd.Series:
        """
        平均真实波幅 (ATR)
        
        参数:
            data: 数据
            period: 周期
            high_col: 最高价列
            low_col: 最低价列
            close_col: 收盘价列
            
        返回:
            pd.Series: ATR
        """
        high = data[high_col]
        low = data[low_col]
        close_prev = data[close_col].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def volatility(
        self,
        data: pd.DataFrame,
        period: int = 20,
        price_col: str = "close",
        annualize: bool = True
    ) -> pd.Series:
        """
        波动率计算
        
        参数:
            data: 数据
            period: 周期
            price_col: 价格列
            annualize: 是否年化
            
        返回:
            pd.Series: 波动率
        """
        returns = data[price_col].pct_change()
        vol = returns.rolling(window=period).std()
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    def add_all_indicators(
        self,
        data: pd.DataFrame,
        price_col: str = "close",
        high_col: str = "high",
        low_col: str = "low"
    ) -> pd.DataFrame:
        """
        添加常用技术指标到数据
        
        参数:
            data: 原始数据
            price_col: 收盘价列
            high_col: 最高价列
            low_col: 最低价列
            
        返回:
            pd.DataFrame: 包含技术指标的数据
        """
        df = data.copy()
        
        # 移动平均线
        df["sma_20"] = self.sma(df, 20, price_col)
        df["sma_60"] = self.sma(df, 60, price_col)
        df["ema_12"] = self.ema(df, 12, price_col)
        df["ema_26"] = self.ema(df, 26, price_col)
        
        # RSI
        df["rsi_14"] = self.rsi(df, 14, price_col)
        
        # MACD
        macd_data = self.macd(df, price_col=price_col)
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_hist"] = macd_data["histogram"]
        
        # 布林带
        bb_data = self.bollinger_bands(df, price_col=price_col)
        df["bb_upper"] = bb_data["upper"]
        df["bb_middle"] = bb_data["middle"]
        df["bb_lower"] = bb_data["lower"]
        
        # KDJ
        kdj_data = self.kdj(df, high_col=high_col, low_col=low_col, close_col=price_col)
        df["kdj_k"] = kdj_data["k"]
        df["kdj_d"] = kdj_data["d"]
        df["kdj_j"] = kdj_data["j"]
        
        # ATR
        df["atr_14"] = self.atr(df, high_col=high_col, low_col=low_col, close_col=price_col)
        
        # 波动率
        df["volatility"] = self.volatility(df, price_col=price_col)
        
        return df
