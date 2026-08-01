"""
斐波那契回撤位计算模块
====================

基于最近 N 日的最高价/最低价计算斐波那契回撤位，
用于识别价格潜在支撑/阻力区间。

回撤位:
  0.0   起点
  0.236 23.6% 回撤
  0.382 38.2% 回撤
  0.5   50%   回撤
  0.618 61.8% 回撤
  0.786 78.6% 回撤
  1.0   终点

趋势判断:
  对 lookback 窗口内的收盘价取前半段/后半段均值比较，
  后半段 >= 前半段视为上涨趋势，否则为下跌趋势。

  - 上涨趋势: 回撤自高点向低点度量（0.0=高点, 1.0=低点）
  - 下跌趋势: 回撤自低点向高点度量（0.0=低点, 1.0=高点）

支撑/阻力分类:
  始终以最新收盘价为准，价位高于现价者为阻力，低于现价者为支撑。
"""
from typing import List, Dict

import numpy as np
import pandas as pd


class FibonacciCalculator:
    """斐波那契回撤位计算"""

    LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    LEVEL_NAMES = ["起点", "23.6%回撤", "38.2%回撤", "50%回撤", "61.8%回撤", "78.6%回撤", "终点"]

    def calculate(self, df: pd.DataFrame, lookback: int = 120) -> List[Dict]:
        """
        基于最近 N 日的最高价和最低价计算斐波那契回撤位

        参数:
            df: 包含 high/low/close 列的 DataFrame，按日期升序排列
            lookback: 回看窗口长度（交易日）

        返回:
            [{"level": 0.382, "name": "38.2%回撤",
              "price": 14.85, "type": "support"/"resistance"}]
            在上涨趋势中：0.236/0.382 为阻力位，0.5/0.618/0.786 为支撑位
            在下跌趋势中：反过来
        """
        if df is None or len(df) < 2:
            return []

        window = df.tail(lookback)
        if len(window) < 2:
            return []

        required = {"high", "low", "close"}
        if not required.issubset(set(window.columns)):
            return []

        high = float(window["high"].max())
        low = float(window["low"].min())
        current_price = float(window["close"].iloc[-1])

        if high <= low:
            return []

        # 趋势判断：前半段 vs 后半段收盘均价
        closes = window["close"].values
        mid = len(closes) // 2
        if mid == 0:
            return []
        first_half_avg = float(np.mean(closes[:mid]))
        second_half_avg = float(np.mean(closes[mid:]))
        is_uptrend = second_half_avg >= first_half_avg

        results: List[Dict] = []
        for level, name in zip(self.LEVELS, self.LEVEL_NAMES):
            if is_uptrend:
                # 上涨趋势：回撤自高点向低点度量
                # 0.0 = 高点（起点），1.0 = 低点（终点）
                price = high - level * (high - low)
            else:
                # 下跌趋势：回撤自低点向高点度量
                # 0.0 = 低点（起点），1.0 = 高点（终点）
                price = low + level * (high - low)

            # 支撑/阻力分类：以最新收盘价为基准
            if price > current_price:
                level_type = "resistance"
            elif price < current_price:
                level_type = "support"
            else:
                level_type = "support"

            results.append({
                "level": level,
                "name": name,
                "price": round(float(price), 4),
                "type": level_type,
            })

        return results
