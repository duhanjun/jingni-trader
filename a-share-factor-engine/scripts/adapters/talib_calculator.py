"""
基于 TA-Lib 的因子计算器适配器
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from ..base.base_factor_calculator import BaseFactorCalculator


class TalibCalculator(BaseFactorCalculator):
    """TA-Lib 因子计算器"""

    def __init__(self):
        if not HAS_TALIB:
            raise ImportError("TA-Lib 未安装，请 pip install TA-Lib")

    def get_available_factors(self) -> List[str]:
        return [
            "rsi", "macd", "macd_signal", "macd_hist",
            "ma_5", "ma_20", "ma_60",
            "bollinger_upper", "bollinger_middle", "bollinger_lower",
            "atr", "adx", "cci",
            "obv", "mfi",
            "willr", "stoch_k", "stoch_d",
        ]

    def get_factor_info(self, factor_name: str) -> Dict:
        info_map = {
            "rsi": {"name": "RSI相对强弱", "direction": 0, "params": {"period": 14}},
            "macd": {"name": "MACD线", "direction": 0, "params": {"fast": 12, "slow": 26, "signal": 9}},
            "ma_20": {"name": "20日均线", "direction": 0, "params": {"period": 20}},
            "atr": {"name": "平均真实波幅", "direction": 0, "params": {"period": 14}},
        }
        return info_map.get(factor_name, {})

    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """批量计算因子，按股票分组"""
        if data.empty:
            return data

        result = data[['code', 'date']].copy()
        result = result.reset_index(drop=True)

        # 确保数据按股票和日期排序
        data = data.sort_values(['code', 'date']).reset_index(drop=True)

        for factor_name in factor_names:
            result[factor_name] = self._calc_single(data, factor_name)

        return result

    def _calc_single(self, data: pd.DataFrame, factor_name: str) -> pd.Series:
        """按股票分组计算单个因子"""
        series = pd.Series(index=data.index, dtype=float)

        for code in data['code'].unique():
            mask = data['code'] == code
            idx = data[mask].index

            try:
                values = self._calc_factor(data.loc[idx], factor_name)
                series.loc[idx] = values
            except Exception as e:
                print(f"计算 {code} 的 {factor_name} 失败: {e}")
                series.loc[idx] = np.nan

        return series

    def _calc_factor(self, df: pd.DataFrame, factor_name: str) -> np.ndarray:
        """计算单个股票的单个因子"""
        open_p = df['open'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        if factor_name == "rsi":
            return talib.RSI(close, timeperiod=14)
        elif factor_name == "macd":
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return macd
        elif factor_name == "macd_signal":
            _, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return signal
        elif factor_name == "macd_hist":
            _, _, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return hist
        elif factor_name == "ma_5":
            return talib.MA(close, timeperiod=5, matype=0)
        elif factor_name == "ma_20":
            return talib.MA(close, timeperiod=20, matype=0)
        elif factor_name == "ma_60":
            return talib.MA(close, timeperiod=60, matype=0)
        elif factor_name == "bollinger_upper":
            upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return upper
        elif factor_name == "bollinger_middle":
            _, middle, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return middle
        elif factor_name == "bollinger_lower":
            _, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return lower
        elif factor_name == "atr":
            return talib.ATR(high, low, close, timeperiod=14)
        elif factor_name == "adx":
            return talib.ADX(high, low, close, timeperiod=14)
        elif factor_name == "cci":
            return talib.CCI(high, low, close, timeperiod=14)
        elif factor_name == "obv":
            return talib.OBV(close, volume)
        elif factor_name == "mfi":
            return talib.MFI(high, low, close, volume, timeperiod=14)
        elif factor_name == "willr":
            return talib.WILLR(high, low, close, timeperiod=14)
        elif factor_name == "stoch_k":
            k, _ = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
            return k
        elif factor_name == "stoch_d":
            _, d = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
            return d
        else:
            raise ValueError(f"不支持的因子: {factor_name}")
