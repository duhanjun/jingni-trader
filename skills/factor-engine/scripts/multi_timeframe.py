"""
多周期技术分析模块
支持日线、周线、月线三个级别的技术指标分析和共振信号检测
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("multi_timeframe")

# 尝试加载 TA-Lib，未安装时回退到纯 pandas 实现
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.debug("TA-Lib 未安装，将使用纯 pandas 实现指标计算")

# 兼容不同 pandas 版本的月线重采样规则
# pandas 2.1+ 使用 'ME'，旧版本使用 'M'
try:
    pd.date_range(start="2024-01-01", periods=3, freq="ME")
    _MONTHLY_RULE = "ME"
except (ValueError, TypeError):
    _MONTHLY_RULE = "M"


class MultiTimeframeAnalyzer:
    """多周期技术分析器"""

    TIMEFRAMES = {
        "daily": "日线",
        "weekly": "周线",
        "monthly": "月线",
    }

    # 重采样聚合规则：open=first, high=max, low=min, close=last, volume=sum
    _RESAMPLE_AGG = {
        "code": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        多周期综合分析

        参数:
            df: 日线OHLCV数据，必须包含 code, date, open, high, low, close, volume 列

        返回:
            {
                "timeframes": {
                    "daily": {"trend": "上涨", "strength": "强", "signals": [...]},
                    "weekly": {"trend": "震荡", "strength": "中", "signals": [...]},
                    "monthly": {"trend": "上涨", "strength": "弱", "signals": [...]},
                },
                "resonance": {
                    "bullish": True/False,  # 日线+月线同步看多
                    "bearish": True/False,
                    "description": "日线+月线同步看多，周线震荡"
                },
                "divergences": [
                    {"type": "顶背离", "timeframe": "daily", "description": "价格创新高但MACD未创新高"}
                ],
                "summary": "日线上涨趋势，周线震荡整理，月线上涨趋势放缓。多周期共振：偏多"
            }
        """
        if df is None or df.empty:
            logger.warning("输入数据为空")
            return self._empty_result()

        # 校验必要列
        required = {"code", "date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"输入数据缺少必要列: {missing}")

        # 仅分析单只股票，多只时取第一只并告警
        codes = df["code"].unique()
        if len(codes) > 1:
            logger.warning(f"输入包含多只股票({len(codes)}只)，仅分析第一只: {codes[0]}")
        code = codes[0]

        # 准备日线数据：按日期排序、去空、类型转换
        daily = df[df["code"] == code].copy()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date").reset_index(drop=True)
        daily = daily.dropna(subset=["open", "high", "low", "close", "volume"])

        if len(daily) < 60:
            logger.warning(f"数据量不足({len(daily)}行)，至少需要60个交易日")
            return self._empty_result()

        # 重采样为周线、月线
        weekly = self._resample(daily, "weekly")
        monthly = self._resample(daily, "monthly")

        # 分析各周期技术面
        tf_results: Dict = {}
        tf_results["daily"] = self._analyze_single_timeframe(daily, "daily")
        tf_results["weekly"] = self._analyze_single_timeframe(weekly, "weekly")
        tf_results["monthly"] = self._analyze_single_timeframe(monthly, "monthly")

        # 检测多周期共振与背离
        resonance = self._detect_resonance(tf_results)

        divergences: List[Dict] = []
        for tf_name, tf_df in [("daily", daily), ("weekly", weekly), ("monthly", monthly)]:
            divergences.extend(self._detect_divergence(tf_df, tf_name))

        summary = self._build_summary(tf_results, resonance, divergences)

        return {
            "code": str(code),
            "timeframes": tf_results,
            "resonance": resonance,
            "divergences": divergences,
            "summary": summary,
        }

    def _empty_result(self) -> Dict:
        """返回数据不足时的空结果"""
        return {
            "code": None,
            "timeframes": {
                tf: {"trend": "未知", "strength": "无", "signals": [], "indicators": {}}
                for tf in self.TIMEFRAMES
            },
            "resonance": {
                "bullish": False,
                "bearish": False,
                "all_bullish": False,
                "all_bearish": False,
                "description": "数据不足",
            },
            "divergences": [],
            "summary": "数据不足，无法分析",
        }

    def _resample(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        将日线数据重采样为周线/月线
        open=first, high=max, low=min, close=last, volume=sum
        """
        if freq == "daily":
            return df.copy()

        if freq == "weekly":
            rule = "W-FRI"  # 以周五为周末
        elif freq == "monthly":
            rule = _MONTHLY_RULE
        else:
            raise ValueError(f"不支持的时间周期: {freq}")

        indexed = df.set_index("date")
        agg = indexed.resample(rule).agg(self._RESAMPLE_AGG)
        # 丢弃没有交易的周期（如节假日产生的空行）
        agg = agg.dropna(subset=["open", "close"])

        result = agg.reset_index()
        # 确保 volume 为数值
        result["volume"] = pd.to_numeric(result["volume"], errors="coerce").fillna(0)
        return result

    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        分析单个周期的技术面
        计算: MA5/10/20/60, MACD, RSI, KDJ, 布林带位置
        判定趋势: 上涨/下跌/震荡
        判定强度: 强/中/弱
        生成信号: MACD金叉/死叉, RSI超买/超卖, 布林带位置, KDJ信号
        """
        if df is None or len(df) < 5:
            return {
                "trend": "未知",
                "strength": "无",
                "signals": [],
                "indicators": {},
            }

        try:
            ind = self._calc_indicators(df)
        except Exception as e:
            logger.warning(f"计算 {timeframe} 指标失败: {e}")
            return {
                "trend": "未知",
                "strength": "无",
                "signals": [],
                "indicators": {},
            }

        if ind.empty:
            return {
                "trend": "未知",
                "strength": "无",
                "signals": [],
                "indicators": {},
            }

        last = ind.iloc[-1]
        close = last["close"]
        ma5 = last.get("ma5")
        ma10 = last.get("ma10")
        ma20 = last.get("ma20")
        ma60 = last.get("ma60")
        macd_dif = last.get("macd_dif")
        macd_dea = last.get("macd_dea")
        macd_hist = last.get("macd_hist")
        rsi = last.get("rsi")
        k = last.get("kdj_k")
        d = last.get("kdj_d")
        j = last.get("kdj_j")
        boll_pos = last.get("boll_position")

        # ── 趋势判定 ──────────────────────────────
        # 上涨: MA5>MA20>MA60, MACD>0, close>MA20
        # 下跌: MA5<MA20<MA60, MACD<0, close<MA20
        # 震荡: 其他
        trend = "震荡"
        if all(pd.notna(v) for v in [ma5, ma20, ma60, macd_dif, close]):
            if ma5 > ma20 > ma60 and macd_dif > 0 and close > ma20:
                trend = "上涨"
            elif ma5 < ma20 < ma60 and macd_dif < 0 and close < ma20:
                trend = "下跌"

        # ── 强度判定 (基于价格偏离 MA20 的幅度，类 ADX 逻辑) ──
        strength = "中"
        if pd.notna(ma20) and ma20 > 0:
            distance = abs(close - ma20) / ma20
            if trend == "上涨":
                if distance > 0.05 and pd.notna(macd_hist) and macd_hist > 0:
                    strength = "强"
                elif distance < 0.02:
                    strength = "弱"
            elif trend == "下跌":
                if distance > 0.05 and pd.notna(macd_hist) and macd_hist < 0:
                    strength = "强"
                elif distance < 0.02:
                    strength = "弱"
            else:  # 震荡
                if distance < 0.015:
                    strength = "弱"
                elif distance > 0.04:
                    strength = "中"
                else:
                    strength = "中"

        # ── 信号生成 ──────────────────────────────
        signals: List[Dict] = []
        tf_label = self.TIMEFRAMES[timeframe]

        # MACD 金叉/死叉 (比较最后两根K线)
        if len(ind) >= 2:
            prev = ind.iloc[-2]
            prev_dif = prev.get("macd_dif")
            prev_dea = prev.get("macd_dea")
            if all(pd.notna(v) for v in [prev_dif, prev_dea, macd_dif, macd_dea]):
                if prev_dif <= prev_dea and macd_dif > macd_dea:
                    signals.append({
                        "type": "MACD金叉",
                        "description": f"{tf_label}MACD金叉信号",
                    })
                elif prev_dif >= prev_dea and macd_dif < macd_dea:
                    signals.append({
                        "type": "MACD死叉",
                        "description": f"{tf_label}MACD死叉信号",
                    })

        # MACD 柱状图红绿
        if pd.notna(macd_hist):
            if macd_hist > 0:
                signals.append({
                    "type": "MACD红柱",
                    "description": f"{tf_label}MACD柱状图为红",
                })
            else:
                signals.append({
                    "type": "MACD绿柱",
                    "description": f"{tf_label}MACD柱状图为绿",
                })

        # RSI 超买/超卖
        if pd.notna(rsi):
            if rsi > 70:
                signals.append({
                    "type": "RSI超买",
                    "description": f"{tf_label}RSI={rsi:.1f}超买",
                })
            elif rsi < 30:
                signals.append({
                    "type": "RSI超卖",
                    "description": f"{tf_label}RSI={rsi:.1f}超卖",
                })

        # 布林带位置
        if pd.notna(boll_pos):
            if boll_pos > 0.9:
                signals.append({
                    "type": "布林上轨",
                    "description": f"{tf_label}价格触及布林上轨",
                })
            elif boll_pos < 0.1:
                signals.append({
                    "type": "布林下轨",
                    "description": f"{tf_label}价格触及布林下轨",
                })

        # KDJ 信号
        if all(pd.notna(v) for v in [k, d, j]):
            if j > 100 or (k > 80 and d > 80):
                signals.append({
                    "type": "KDJ超买",
                    "description": f"{tf_label}KDJ处于超买区",
                })
            elif j < 0 or (k < 20 and d < 20):
                signals.append({
                    "type": "KDJ超卖",
                    "description": f"{tf_label}KDJ处于超卖区",
                })
            # KDJ 金叉/死叉
            if len(ind) >= 2:
                prev = ind.iloc[-2]
                prev_k = prev.get("kdj_k")
                prev_d = prev.get("kdj_d")
                if all(pd.notna(v) for v in [prev_k, prev_d]):
                    if prev_k <= prev_d and k > d:
                        signals.append({
                            "type": "KDJ金叉",
                            "description": f"{tf_label}KDJ金叉",
                        })
                    elif prev_k >= prev_d and k < d:
                        signals.append({
                            "type": "KDJ死叉",
                            "description": f"{tf_label}KDJ死叉",
                        })

        # 指标快照
        indicators = {}
        for name, val in [
            ("close", close), ("ma5", ma5), ("ma10", ma10),
            ("ma20", ma20), ("ma60", ma60),
            ("macd_dif", macd_dif), ("macd_dea", macd_dea), ("macd_hist", macd_hist),
            ("rsi", rsi),
            ("kdj_k", k), ("kdj_d", d), ("kdj_j", j),
            ("boll_position", boll_pos),
        ]:
            indicators[name] = float(val) if pd.notna(val) else None

        return {
            "trend": trend,
            "strength": strength,
            "signals": signals,
            "indicators": indicators,
        }

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标，返回带指标列的 DataFrame"""
        out = df.copy().reset_index(drop=True)

        close = out["close"].values.astype(float)
        high = out["high"].values.astype(float)
        low = out["low"].values.astype(float)

        # 移动平均线
        for period in [5, 10, 20, 60]:
            out[f"ma{period}"] = self._sma(close, period)

        # MACD
        dif, dea, hist = self._macd(close, fast=12, slow=26, signal=9)
        out["macd_dif"] = dif
        out["macd_dea"] = dea
        out["macd_hist"] = hist

        # RSI
        out["rsi"] = self._rsi(close, period=14)

        # KDJ
        k, d, j = self._kdj(high, low, close, n=9, m1=3, m2=3)
        out["kdj_k"] = k
        out["kdj_d"] = d
        out["kdj_j"] = j

        # 布林带
        upper, middle, lower = self._bollinger(close, period=20, nbdev=2)
        out["boll_upper"] = upper
        out["boll_middle"] = middle
        out["boll_lower"] = lower
        # 布林带位置: 0=下轨, 1=上轨, 0.5=中轨
        width = upper - lower
        out["boll_position"] = np.where(
            width > 0, (close - lower) / width, 0.5
        )

        return out

    def _detect_resonance(self, tf_results: Dict) -> Dict:
        """
        检测多周期共振
        看多共振: 日/周/月全部上涨，或 日线+月线 同步上涨
        看空共振: 日/周/月全部下跌，或 日线+月线 同步下跌
        """
        daily_trend = tf_results.get("daily", {}).get("trend", "未知")
        weekly_trend = tf_results.get("weekly", {}).get("trend", "未知")
        monthly_trend = tf_results.get("monthly", {}).get("trend", "未知")

        all_bullish = (
            daily_trend == "上涨"
            and weekly_trend == "上涨"
            and monthly_trend == "上涨"
        )
        all_bearish = (
            daily_trend == "下跌"
            and weekly_trend == "下跌"
            and monthly_trend == "下跌"
        )

        # 日线+月线同步（周线允许不同步）
        bull_partial = (daily_trend == "上涨" and monthly_trend == "上涨")
        bear_partial = (daily_trend == "下跌" and monthly_trend == "下跌")

        bullish = all_bullish or bull_partial
        bearish = all_bearish or bear_partial

        # 构建描述
        parts = [f"日线{daily_trend}", f"周线{weekly_trend}", f"月线{monthly_trend}"]
        base_desc = "；".join(parts)

        if all_bullish:
            resonance_desc = "三周期共振看多"
        elif all_bearish:
            resonance_desc = "三周期共振看空"
        elif bull_partial:
            resonance_desc = f"日线+月线同步看多，周线{weekly_trend}"
        elif bear_partial:
            resonance_desc = f"日线+月线同步看空，周线{weekly_trend}"
        else:
            resonance_desc = "无明确多空共振"

        description = f"{base_desc}。{resonance_desc}"

        return {
            "bullish": bullish,
            "bearish": bearish,
            "all_bullish": all_bullish,
            "all_bearish": all_bearish,
            "description": description,
        }

    def _detect_divergence(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """
        检测技术指标背离
        顶背离: 价格创新高但 MACD/RSI 未创新高
        底背离: 价格创新低但 MACD/RSI 未创新低
        回看 60 根K线进行比较
        """
        divergences: List[Dict] = []
        if df is None or len(df) < 30:
            return divergences

        try:
            ind = self._calc_indicators(df)
        except Exception as e:
            logger.warning(f"计算 {timeframe} 指标失败(背离检测): {e}")
            return divergences

        lookback = 60
        recent = ind.tail(lookback).reset_index(drop=True)
        if len(recent) < 20:
            return divergences

        # 最近 current_window 根K线作为"当前"窗口，避免同一峰值
        current_window = 5
        if len(recent) <= current_window:
            return divergences

        past = recent.iloc[:-current_window]
        if len(past) < 10:
            return divergences

        tf_label = self.TIMEFRAMES[timeframe]

        close_series = recent["close"]
        past_high = past["close"].max()
        past_high_idx = past["close"].idxmax()
        past_low = past["close"].min()
        past_low_idx = past["close"].idxmin()

        recent_high = close_series.iloc[-current_window:].max()
        recent_low = close_series.iloc[-current_window:].min()

        # ── 顶背离: 价格创新高但指标未创新高 ──
        if recent_high > past_high:
            current_macd = recent["macd_dif"].iloc[-1]
            past_macd = recent["macd_dif"].iloc[past_high_idx]
            if pd.notna(current_macd) and pd.notna(past_macd) and current_macd < past_macd:
                divergences.append({
                    "type": "顶背离",
                    "timeframe": timeframe,
                    "indicator": "MACD",
                    "description": f"{tf_label}价格创新高但MACD未创新高",
                })

            current_rsi = recent["rsi"].iloc[-1]
            past_rsi = recent["rsi"].iloc[past_high_idx]
            if pd.notna(current_rsi) and pd.notna(past_rsi) and current_rsi < past_rsi:
                divergences.append({
                    "type": "顶背离",
                    "timeframe": timeframe,
                    "indicator": "RSI",
                    "description": f"{tf_label}价格创新高但RSI未创新高",
                })

        # ── 底背离: 价格创新低但指标未创新低 ──
        if recent_low < past_low:
            current_macd = recent["macd_dif"].iloc[-1]
            past_macd = recent["macd_dif"].iloc[past_low_idx]
            if pd.notna(current_macd) and pd.notna(past_macd) and current_macd > past_macd:
                divergences.append({
                    "type": "底背离",
                    "timeframe": timeframe,
                    "indicator": "MACD",
                    "description": f"{tf_label}价格创新低但MACD未创新低",
                })

            current_rsi = recent["rsi"].iloc[-1]
            past_rsi = recent["rsi"].iloc[past_low_idx]
            if pd.notna(current_rsi) and pd.notna(past_rsi) and current_rsi > past_rsi:
                divergences.append({
                    "type": "底背离",
                    "timeframe": timeframe,
                    "indicator": "RSI",
                    "description": f"{tf_label}价格创新低但RSI未创新低",
                })

        return divergences

    def _build_summary(
        self, tf_results: Dict, resonance: Dict, divergences: List[Dict]
    ) -> str:
        """生成综合摘要文本"""
        daily = tf_results.get("daily", {})
        weekly = tf_results.get("weekly", {})
        monthly = tf_results.get("monthly", {})

        parts = [
            f"日线{daily.get('trend', '未知')}",
            f"周线{weekly.get('trend', '未知')}",
            f"月线{monthly.get('trend', '未知')}",
        ]
        summary = "、".join(parts) + "。"

        if resonance.get("all_bullish"):
            summary += "多周期共振：强多。"
        elif resonance.get("all_bearish"):
            summary += "多周期共振：强空。"
        elif resonance.get("bullish"):
            summary += "多周期共振：偏多。"
        elif resonance.get("bearish"):
            summary += "多周期共振：偏空。"
        else:
            summary += "多周期共振：无。"

        if divergences:
            div_types = [d["type"] for d in divergences]
            unique_types = list(dict.fromkeys(div_types))  # 去重保序
            summary += f"检测到{len(divergences)}个背离信号（{'、'.join(unique_types)}）。"

        return summary

    # ── 指标计算辅助方法 ─────────────────────────────

    def _sma(self, close: np.ndarray, period: int) -> np.ndarray:
        """简单移动平均"""
        if HAS_TALIB:
            try:
                return talib.MA(close, timeperiod=period, matype=0)
            except Exception as e:
                logger.debug(f"TA-Lib MA(period={period}) 失败，回退到 pandas: {e}")
        s = pd.Series(close)
        return s.rolling(window=period, min_periods=period).mean().values

    def _macd(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD 指标
        返回: (DIF, DEA, HIST)
        采用A股惯例: HIST = (DIF - DEA) * 2
        """
        if HAS_TALIB:
            try:
                dif, dea, _ = talib.MACD(
                    close,
                    fastperiod=fast,
                    slowperiod=slow,
                    signalperiod=signal,
                )
                hist = (dif - dea) * 2
                return dif, dea, hist
            except Exception as e:
                logger.debug(f"TA-Lib MACD 失败，回退到 pandas: {e}")
        # 回退: pandas EMA 实现
        s = pd.Series(close)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        hist = (dif - dea) * 2
        return dif.values, dea.values, hist.values

    def _rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI 相对强弱指标 (Wilder 平滑)"""
        if HAS_TALIB:
            try:
                return talib.RSI(close, timeperiod=period)
            except Exception as e:
                logger.debug(f"TA-Lib RSI 失败，回退到 pandas: {e}")
        # 回退: Wilder 平滑 RSI
        s = pd.Series(close)
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder 平滑等价于 ewm(alpha=1/period)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values

    def _kdj(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        KDJ 随机指标 (A股中式KDJ)
        RSV = (C - Ln) / (Hn - Ln) * 100
        K = SMA(RSV, m1)  [中式SMA即EMA, alpha=1/m1]
        D = SMA(K, m2)
        J = 3*K - 2*D

        注: TA-Lib STOCH 公式与中式KDJ不同(平滑方式差异)，故始终用 pandas 实现
        """
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        lowest_low = low_s.rolling(window=n, min_periods=1).min()
        highest_high = high_s.rolling(window=n, min_periods=1).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        rsv = (close_s - lowest_low) / denom * 100
        rsv = rsv.fillna(50)

        # 中式SMA = EMA(alpha=1/period)
        k = rsv.ewm(alpha=1.0 / m1, adjust=False).mean()
        d = k.ewm(alpha=1.0 / m2, adjust=False).mean()
        j = 3 * k - 2 * d
        return k.values, d.values, j.values

    def _bollinger(
        self, close: np.ndarray, period: int = 20, nbdev: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        布林带
        返回: (upper, middle, lower)
        """
        if HAS_TALIB:
            try:
                upper, middle, lower = talib.BBANDS(
                    close,
                    timeperiod=period,
                    nbdevup=nbdev,
                    nbdevdn=nbdev,
                    matype=0,
                )
                return upper, middle, lower
            except Exception as e:
                logger.debug(f"TA-Lib BBANDS 失败，回退到 pandas: {e}")
        # 回退: pandas 实现
        s = pd.Series(close)
        middle = s.rolling(window=period, min_periods=period).mean()
        std = s.rolling(window=period, min_periods=period).std(ddof=0)
        upper = middle + nbdev * std
        lower = middle - nbdev * std
        return upper.values, middle.values, lower.values
