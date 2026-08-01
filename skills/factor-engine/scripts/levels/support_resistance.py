"""
多方法支撑阻力位计算模块
======================

综合前期高低点、均线、斐波那契回撤、布林带边界、整数关口、
密集成交区等多种方法，计算个股的支撑位与阻力位。

借鉴来源:
- 传统技术分析（道氏理论、波浪理论）的支撑/阻力识别方法
- 各方法独立计算后在 calculate_all 中汇总，按价位相对现价分类

依赖: pandas / numpy / scipy（均在 requirements.txt 中）
"""
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .fibonacci import FibonacciCalculator


class SupportResistanceCalculator:
    """多方法支撑阻力位计算"""

    # 均线周期 → 强度
    MA_PERIODS = [5, 10, 20, 60, 120, 250]
    MA_STRENGTH = {5: "弱", 10: "弱", 20: "中", 60: "强", 120: "强", 250: "很强"}

    # 前期高低点窗口 → 强度
    EXTREMA_WINDOWS = [20, 60, 120]
    EXTREMA_STRENGTH = {20: "中", 60: "强", 120: "很强"}

    # 强度排序（用于去重时保留更强级别）
    _STRENGTH_RANK = {"弱": 1, "中": 2, "强": 3, "很强": 4}

    def __init__(self):
        self.fib_calc = FibonacciCalculator()

    # ================================================================
    # 公共入口
    # ================================================================

    def calculate_all(self, df: pd.DataFrame, current_price: Optional[float] = None) -> Dict:
        """
        综合计算支撑阻力位，返回结构化结果

        参数:
            df: 包含 open/high/low/close/volume 列的 DataFrame，按日期升序
            current_price: 当前价格，None 时取最新收盘价

        返回:
            {
                "resistance": [{"price": 15.8, "type": "前高", "strength": "强", "method": "prior_high"}],
                "support":    [{"price": 13.2, "type": "前低", "strength": "强", "method": "prior_low"}],
                "current_price": 14.32,
                "nearest_resistance": 14.85,
                "nearest_support": 13.80,
            }
        """
        empty_result: Dict = {
            "resistance": [],
            "support": [],
            "current_price": None,
            "nearest_resistance": None,
            "nearest_support": None,
        }

        if df is None or df.empty:
            return empty_result

        # 确保按日期升序
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        # 汇总各方法结果
        all_levels: List[Dict] = []
        all_levels.extend(self._prior_highs_lows(df))
        all_levels.extend(self._ma_levels(df))
        all_levels.extend(self._fibonacci_levels(df, current_price))
        all_levels.extend(self._bollinger_levels(df))
        all_levels.extend(self._round_numbers(current_price))
        all_levels.extend(self._vwap_zones(df))

        # 去重：相近价位（0.1% 容差）合并，保留更强级别
        all_levels = self._deduplicate(all_levels, current_price)

        # 按相对现价分类
        tol = max(current_price * 0.001, 0.01)  # 0.1% 或至少 0.01
        resistance: List[Dict] = []
        support: List[Dict] = []
        for lv in all_levels:
            price = lv["price"]
            entry = {
                "price": price,
                "type": lv.get("type_label", ""),
                "strength": lv.get("strength", "中"),
                "method": lv.get("method", ""),
            }
            if price > current_price + tol:
                resistance.append(entry)
            elif price < current_price - tol:
                support.append(entry)
            # 过于接近现价的跳过（既非有效支撑也非有效阻力）

        # 阻力升序（最近者优先），支撑降序（最近者优先）
        resistance.sort(key=lambda x: x["price"])
        support.sort(key=lambda x: x["price"], reverse=True)

        nearest_resistance = resistance[0]["price"] if resistance else None
        nearest_support = support[0]["price"] if support else None

        return {
            "resistance": resistance,
            "support": support,
            "current_price": current_price,
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
        }

    # ================================================================
    # 方法一：前期高点/低点（滚动窗口局部极值）
    # ================================================================

    def _prior_highs_lows(self, df: pd.DataFrame, windows: Optional[List[int]] = None) -> List[Dict]:
        """
        前期高点/低点（滚动窗口局部极值）

        使用 scipy.signal.argrelextrema 识别局部极值:
          - 局部高点 → 阻力位（前高）
          - 局部低点 → 支撑位（前低）

        强度: 20 日=中, 60 日=强, 120 日=很强
        """
        if windows is None:
            windows = self.EXTREMA_WINDOWS

        results: List[Dict] = []
        if df is None or len(df) < 3:
            return results

        highs = df["high"].values
        lows = df["low"].values

        for w in windows:
            # argrelextrema 的 order = 每侧比较点数，取窗口一半
            order = max(2, w // 2)
            if len(highs) < order * 2 + 1:
                continue

            # 局部高点（前高 → 阻力）
            max_idx = argrelextrema(highs, np.greater, order=order)[0]
            # 局部低点（前低 → 支撑）
            min_idx = argrelextrema(lows, np.less, order=order)[0]

            strength = self.EXTREMA_STRENGTH.get(w, "中")

            for idx in max_idx:
                results.append({
                    "price": round(float(highs[idx]), 4),
                    "type_label": f"前高({w}日)",
                    "strength": strength,
                    "method": "prior_high",
                })
            for idx in min_idx:
                results.append({
                    "price": round(float(lows[idx]), 4),
                    "type_label": f"前低({w}日)",
                    "strength": strength,
                    "method": "prior_low",
                })

        return results

    # ================================================================
    # 方法二：均线支撑/阻力位
    # ================================================================

    def _ma_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        均线支撑/阻力位 (MA5/10/20/60/120/250)

        各均线最新值作为支撑/阻力参考位。
        强度: MA5/10=弱, MA20=中, MA60/120=强, MA250=很强
        """
        results: List[Dict] = []
        if df is None or df.empty:
            return results

        closes = df["close"]
        for period in self.MA_PERIODS:
            if len(closes) < period:
                continue
            ma = closes.rolling(window=period, min_periods=period).mean()
            ma_val = ma.iloc[-1]
            if pd.isna(ma_val):
                continue
            results.append({
                "price": round(float(ma_val), 4),
                "type_label": f"MA{period}",
                "strength": self.MA_STRENGTH.get(period, "中"),
                "method": f"ma_{period}",
            })
        return results

    # ================================================================
    # 方法三：斐波那契回撤位
    # ================================================================

    def _fibonacci_levels(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """
        斐波那契回撤位

        委托 FibonacciCalculator 计算，转换为统一格式。
        跳过起点(0.0)和终点(1.0)，仅保留中间回撤位
        （起终点即区间高低点，已由前期高低点方法覆盖）。
        """
        results: List[Dict] = []
        fib_levels = self.fib_calc.calculate(df)
        for lv in fib_levels:
            if lv["level"] in (0.0, 1.0):
                continue
            results.append({
                "price": lv["price"],
                "type_label": f"斐波那契{lv['name']}",
                "strength": "中",
                "method": "fibonacci",
            })
        return results

    # ================================================================
    # 方法四：布林带边界
    # ================================================================

    def _bollinger_levels(self, df: pd.DataFrame, window: int = 20,
                          num_std: float = 2.0) -> List[Dict]:
        """
        布林带边界

        上轨 = 阻力位，下轨 = 支撑位，中轨为中性参考。
        默认 20 周期、2 倍标准差。
        """
        results: List[Dict] = []
        if df is None or len(df) < window:
            return results

        closes = df["close"]
        ma = closes.rolling(window=window, min_periods=window).mean()
        std = closes.rolling(window=window, min_periods=window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std

        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        mid_val = ma.iloc[-1]

        if not pd.isna(upper_val):
            results.append({
                "price": round(float(upper_val), 4),
                "type_label": "布林带上轨",
                "strength": "中",
                "method": "bollinger_upper",
            })
        if not pd.isna(lower_val):
            results.append({
                "price": round(float(lower_val), 4),
                "type_label": "布林带下轨",
                "strength": "中",
                "method": "bollinger_lower",
            })
        if not pd.isna(mid_val):
            results.append({
                "price": round(float(mid_val), 4),
                "type_label": "布林带中轨",
                "strength": "弱",
                "method": "bollinger_middle",
            })
        return results

    # ================================================================
    # 方法五：整数关口
    # ================================================================

    def _round_numbers(self, current_price: float) -> List[Dict]:
        """
        整数关口

        寻找最接近现价的整数关口（1/5/10/50/100 的倍数），
        分别取上方和下方各一个。同一价位被多个步长命中时保留更强级别。
        强度: 1=弱, 5/10=中, 50=强, 100=很强
        """
        results: List[Dict] = []
        if current_price is None or current_price <= 0:
            return results

        steps = [
            (1, "弱"),
            (5, "中"),
            (10, "中"),
            (50, "强"),
            (100, "很强"),
        ]

        # 同一整数关口可能被多个步长命中，保留更强级别
        best: Dict[float, tuple] = {}
        for step, strength in steps:
            lower = (int(current_price // step)) * step
            upper = lower + step
            for rn in (lower, upper):
                if rn <= 0:
                    continue
                existing = best.get(rn)
                if existing is None or self._STRENGTH_RANK[strength] > self._STRENGTH_RANK[existing[0]]:
                    best[rn] = (strength, step)

        for rn, (strength, step) in best.items():
            results.append({
                "price": round(float(rn), 4),
                "type_label": f"整数关口({step})",
                "strength": strength,
                "method": "round_number",
            })
        return results

    # ================================================================
    # 方法六：密集成交区（成交量加权均价区间）
    # ================================================================

    def _vwap_zones(self, df: pd.DataFrame, lookback: int = 60,
                    num_bins: int = 20) -> List[Dict]:
        """
        密集成交区（成交量加权均价区间）

        将 lookback 窗口内的典型价格（H+L+C）/3 划分为 num_bins 个价格区间，
        统计各区间累计成交量。成交量显著高于均值（≥1.5 倍）的区间
        视为密集成交区，作为支撑/阻力参考。

        强度: 成交量比 ≥2.5=很强, ≥2.0=强, ≥1.5=中
        """
        results: List[Dict] = []
        if df is None or df.empty:
            return results

        window = df.tail(lookback)
        if len(window) < num_bins:
            return results
        if "volume" not in window.columns:
            return results

        high = window["high"].values
        low = window["low"].values
        close = window["close"].values
        volume = window["volume"].values.astype(float)

        total_vol = volume.sum()
        if total_vol <= 0:
            return results

        # 典型价格
        typical_price = (high + low + close) / 3.0

        price_min = float(typical_price.min())
        price_max = float(typical_price.max())
        if price_max <= price_min:
            return results

        # 分箱统计成交量
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_indices = np.digitize(typical_price, bin_edges[1:-1])

        bin_volumes = np.zeros(num_bins)
        for i, bi in enumerate(bin_indices):
            if 0 <= bi < num_bins:
                bin_volumes[bi] += volume[i]

        avg_volume = bin_volumes.mean()
        if avg_volume <= 0:
            return results

        # 筛选高成交量区间（≥1.5 倍均值）
        threshold = avg_volume * 1.5
        for bi in range(num_bins):
            if bin_volumes[bi] >= threshold:
                bin_center = (bin_edges[bi] + bin_edges[bi + 1]) / 2.0
                ratio = bin_volumes[bi] / avg_volume
                if ratio >= 2.5:
                    strength = "很强"
                elif ratio >= 2.0:
                    strength = "强"
                else:
                    strength = "中"
                results.append({
                    "price": round(float(bin_center), 4),
                    "type_label": "密集成交区",
                    "strength": strength,
                    "method": "vwap_zone",
                })
        return results

    # ================================================================
    # 辅助：去重
    # ================================================================

    def _deduplicate(self, levels: List[Dict], ref_price: float,
                     tol_pct: float = 0.001) -> List[Dict]:
        """
        去重：价位相近（容差 0.1%）的合并为一条，保留更强级别。
        若强度相同则保留先出现者。
        """
        if len(levels) <= 1:
            return levels

        levels_sorted = sorted(levels, key=lambda x: x["price"])
        result: List[Dict] = [levels_sorted[0]]
        for lv in levels_sorted[1:]:
            prev_price = result[-1]["price"]
            tol = max(abs(prev_price) * tol_pct, 0.01)
            if abs(lv["price"] - prev_price) <= tol:
                # 相近 → 合并，保留更强级别
                if (self._STRENGTH_RANK.get(lv.get("strength", "中"), 0)
                        > self._STRENGTH_RANK.get(result[-1].get("strength", "中"), 0)):
                    result[-1] = lv
            else:
                result.append(lv)
        return result
