"""
技术因子计算模块

从 price_data（OHLCV）计算技术指标因子，输出为因子列。
复用已有的 pandas_ta / multi_timeframe 模块，但输出为因子列而非信号。
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger("technical_factors")


def compute(price_data: pd.DataFrame, ctx=None) -> pd.DataFrame:
    """
    计算技术指标因子

    参数:
        price_data: OHLCV 日线数据（含 code, date, open, high, low, close, volume）
        ctx: Context 对象（可选）

    返回:
        DataFrame，列为 code, date, [各技术因子]
    """
    if price_data.empty:
        return price_data

    logger.info("开始计算技术指标因子...")
    df = price_data.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()

    for code in df['code'].unique():
        mask = df['code'] == code
        stock = df[mask].copy()
        close = stock['close']
        high = stock['high']
        low = stock['low']
        volume = stock['volume']

        idx = stock.index

        # MA 均线
        result.loc[idx, 'ma5'] = close.rolling(5, min_periods=1).mean().values
        result.loc[idx, 'ma10'] = close.rolling(10, min_periods=1).mean().values
        result.loc[idx, 'ma20'] = close.rolling(20, min_periods=5).mean().values
        result.loc[idx, 'ma60'] = close.rolling(60, min_periods=10).mean().values

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        result.loc[idx, 'macd_dif'] = dif.values
        result.loc[idx, 'macd_dea'] = dea.values
        result.loc[idx, 'macd_hist'] = (2 * (dif - dea)).values

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        result.loc[idx, 'rsi_14'] = (100 - (100 / (1 + rs))).values

        # KDJ
        low_9 = low.rolling(9, min_periods=1).min()
        high_9 = high.rolling(9, min_periods=1).max()
        rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        result.loc[idx, 'kdj_k'] = k.values
        result.loc[idx, 'kdj_d'] = d.values
        result.loc[idx, 'kdj_j'] = j.values

        # BOLL(20, 2)
        boll_mid = close.rolling(20, min_periods=5).mean()
        boll_std = close.rolling(20, min_periods=5).std()
        result.loc[idx, 'boll_mid'] = boll_mid.values
        result.loc[idx, 'boll_ub'] = (boll_mid + 2 * boll_std).values
        result.loc[idx, 'boll_lb'] = (boll_mid - 2 * boll_std).values

        # WR(14)
        high_14 = high.rolling(14, min_periods=1).max()
        low_14 = low.rolling(14, min_periods=1).min()
        result.loc[idx, 'wr'] = ((high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100).values

        # CCI(14)
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(14, min_periods=1).mean()
        tp_md = tp.rolling(14, min_periods=1).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        result.loc[idx, 'cci'] = ((tp - tp_sma) / (0.015 * tp_md.replace(0, np.nan))).values

        # OBV
        obv = (np.sign(close.diff()) * volume).cumsum()
        result.loc[idx, 'obv'] = obv.values

    logger.info(f"技术指标因子计算完成，共 {len(result.columns) - 2} 个因子")
    return result
