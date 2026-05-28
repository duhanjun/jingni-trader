"""
基于 pandas-ta 的因子计算器适配器（纯Python，无需编译）
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

try:
    import pandas_ta as pta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from ..base.base_factor_calculator import BaseFactorCalculator


class PandasTaCalculator(BaseFactorCalculator):
    """pandas-ta 因子计算器"""

    def __init__(self):
        if not HAS_PANDAS_TA:
            raise ImportError("pandas-ta 未安装，请 pip install pandas-ta")

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
        return {
            "rsi": {"name": "RSI相对强弱", "direction": 0, "params": {"length": 14}},
            "macd": {"name": "MACD线", "direction": 0, "params": {"fast": 12, "slow": 26, "signal": 9}},
            "ma_20": {"name": "20日均线", "direction": 0, "params": {"length": 20}},
            "atr": {"name": "平均真实波幅", "direction": 0, "params": {"length": 14}},
        }.get(factor_name, {})

    def calculate(self, data: pd.DataFrame, factor_names: List[str]) -> pd.DataFrame:
        """批量计算因子"""
        if data.empty:
            return data

        result = data[['code', 'date']].copy()
        result = result.reset_index(drop=True)

        # 按股票分组计算
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

    def _calc_factor(self, df: pd.DataFrame, factor_name: str) -> pd.Series:
        """计算单个股票的单个因子"""
        close = df['close']

        if factor_name == "rsi":
            return pta.rsi(close, length=14)
        elif factor_name == "macd":
            macd_df = pta.macd(close, fast=12, slow=26, signal=9)
            return macd_df['MACD_12_26_9']
        elif factor_name == "macd_signal":
            macd_df = pta.macd(close, fast=12, slow=26, signal=9)
            return macd_df['MACDs_12_26_9']
        elif factor_name == "macd_hist":
            macd_df = pta.macd(close, fast=12, slow=26, signal=9)
            return macd_df['MACDh_12_26_9']
        elif factor_name == "ma_5":
            return pta.sma(close, length=5)
        elif factor_name == "ma_20":
            return pta.sma(close, length=20)
        elif factor_name == "ma_60":
            return pta.sma(close, length=60)
        elif factor_name == "bollinger_upper":
            bb_df = pta.bbands(close, length=20, std=2)
            return bb_df['BBU_20_2.0']
        elif factor_name == "bollinger_middle":
            bb_df = pta.bbands(close, length=20, std=2)
            return bb_df['BBM_20_2.0']
        elif factor_name == "bollinger_lower":
            bb_df = pta.bbands(close, length=20, std=2)
            return bb_df['BBL_20_2.0']
        elif factor_name == "atr":
            return pta.atr(df['high'], df['low'], close, length=14)
        elif factor_name == "adx":
            adx_df = pta.adx(df['high'], df['low'], close, length=14)
            return adx_df['ADX_14']
        elif factor_name == "cci":
            return pta.cci(df['high'], df['low'], close, length=14)
        elif factor_name == "obv":
            return pta.obv(close, df['volume'])
        elif factor_name == "mfi":
            return pta.mfi(df['high'], df['low'], close, df['volume'], length=14)
        elif factor_name == "willr":
            return pta.willr(df['high'], df['low'], close, length=14)
        elif factor_name == "stoch_k":
            stoch_df = pta.stoch(df['high'], df['low'], close, k=5, d=3)
            return stoch_df['STOCHk_5_3_3']
        elif factor_name == "stoch_d":
            stoch_df = pta.stoch(df['high'], df['low'], close, k=5, d=3)
            return stoch_df['STOCHd_5_3_3']
        else:
            raise ValueError(f"不支持的因子: {factor_name}")
