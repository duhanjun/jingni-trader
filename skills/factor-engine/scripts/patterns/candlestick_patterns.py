"""
K线形态识别模块

基于 TA-Lib 的 61 种 CDL 蜡烛图形态识别，提供更高层级的接口，
包括批量识别、近期形态提取和信号汇总统计。

TA-Lib CDL 函数返回值约定：
    100  : 识别到看涨形态
    -100 : 识别到看跌形态
    0    : 无形态信号
"""
from typing import List, Dict
import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


# ---------------------------------------------------------------------------
# 61 种 CDL 形态定义：中文名 / 信号方向 / 可靠度 / 描述
# signal:    bullish 看涨 | bearish 看跌 | neutral 中性（含双向形态）
# reliability: high 高 | medium 中 | low 低
# ---------------------------------------------------------------------------
PATTERN_DEFINITIONS = {
    # ---- 看跌形态 ----
    "CDL2CROWS": {
        "name": "两只乌鸦",
        "signal": "bearish",
        "reliability": "medium",
        "description": "三根K线组合，长阳后高开收阴，再次高开收阴且收盘低于前日，预示下跌",
    },
    "CDL3BLACKCROWS": {
        "name": "三只乌鸦",
        "signal": "bearish",
        "reliability": "high",
        "description": "连续三根大阴线，预示上升趋势反转",
    },
    "CDLADVANCEBLOCK": {
        "name": "大敌当前",
        "signal": "bearish",
        "reliability": "medium",
        "description": "三根渐短阳线，上升动力衰减，预示见顶",
    },
    "CDLCONCEALBABYSWALL": {
        "name": "藏婴吞没",
        "signal": "bearish",
        "reliability": "medium",
        "description": "四日形态，下跌趋势中前两日无影线阴线，第三日跳空收阳后被大阴线吞没",
    },
    "CDLDARKCLOUDCOVER": {
        "name": "乌云盖顶",
        "signal": "bearish",
        "reliability": "high",
        "description": "阴线收盘价深入前日阳线实体过半，顶部反转信号",
    },
    "CDLEVENINGDOJISTAR": {
        "name": "黄昏十字星",
        "signal": "bearish",
        "reliability": "high",
        "description": "黄昏星变体，中间为十字星，顶部反转信号",
    },
    "CDLEVENINGSTAR": {
        "name": "黄昏星",
        "signal": "bearish",
        "reliability": "high",
        "description": "三根K线组合，阳线后跳空小实体再收大阴线，顶部反转信号",
    },
    "CDLGRAVESTONEDOJI": {
        "name": "墓碑十字",
        "signal": "bearish",
        "reliability": "medium",
        "description": "开盘价=收盘价=最低价的长上影十字星，顶部反转信号",
    },
    "CDLHANGINGMAN": {
        "name": "上吊线",
        "signal": "bearish",
        "reliability": "medium",
        "description": "下影线长、实体小且位于顶部的形态，顶部反转信号",
    },
    "CDLIDENTICAL3CROWS": {
        "name": "相同的三只乌鸦",
        "signal": "bearish",
        "reliability": "medium",
        "description": "三根大小相近的连续阴线，预示强烈下跌",
    },
    "CDLINNECK": {
        "name": "颈内线",
        "signal": "bearish",
        "reliability": "low",
        "description": "下跌趋势中阳线收盘价略高于前日阴线最低价，弱势反弹信号",
    },
    "CDLONNECK": {
        "name": "颈上线",
        "signal": "bearish",
        "reliability": "low",
        "description": "下跌趋势中小阳线收盘价等于前日阴线最低价，下跌延续信号",
    },
    "CDLSHOOTINGSTAR": {
        "name": "射击之星",
        "signal": "bearish",
        "reliability": "high",
        "description": "长上影线、小实体且位于顶部的形态，顶部反转信号",
    },
    "CDLSTALLEDPATTERN": {
        "name": "停滞形态",
        "signal": "bearish",
        "reliability": "low",
        "description": "上升趋势中三根渐短阳线，上升动力衰减，预示见顶",
    },
    "CDLTHRUSTING": {
        "name": "插入",
        "signal": "bearish",
        "reliability": "low",
        "description": "阳线收盘价略低于前日阴线中点，预示反弹失败、下跌延续",
    },
    "CDLUPSIDEGAP2CROWS": {
        "name": "向上跳空两只乌鸦",
        "signal": "bearish",
        "reliability": "medium",
        "description": "跳空高开后连续两根阴线，顶部反转信号",
    },
    # ---- 看涨形态 ----
    "CDL3STARSINSOUTH": {
        "name": "南方三星",
        "signal": "bullish",
        "reliability": "medium",
        "description": "三根逐渐缩小的阴线，下跌动力衰减，底部反转信号",
    },
    "CDL3WHITESOLDIERS": {
        "name": "三个白兵",
        "signal": "bullish",
        "reliability": "high",
        "description": "连续三根大阳线且收盘价依次抬高，下降趋势反转信号",
    },
    "CDLDRAGONFLYDOJI": {
        "name": "蜻蜓十字",
        "signal": "bullish",
        "reliability": "medium",
        "description": "开盘价=收盘价=最高价的长下影十字星，底部反转信号",
    },
    "CDLHAMMER": {
        "name": "锤子线",
        "signal": "bullish",
        "reliability": "high",
        "description": "下影线长、实体小且位于顶部的形态，底部反转信号",
    },
    "CDLHOMINGPIGEON": {
        "name": "家鸽",
        "signal": "bullish",
        "reliability": "medium",
        "description": "下跌趋势中前日大阴线被当日小阴线实体包含，底部反转信号",
    },
    "CDLINVERTEDHAMMER": {
        "name": "倒锤子线",
        "signal": "bullish",
        "reliability": "medium",
        "description": "上影线长、实体小且位于底部的形态，底部反转信号",
    },
    "CDLLADDERBOTTOM": {
        "name": "梯底",
        "signal": "bullish",
        "reliability": "medium",
        "description": "五根K线组合，前三根阴线梯次走低后两根阳线反转，底部反转信号",
    },
    "CDLMATCHINGLOW": {
        "name": "相同低价",
        "signal": "bullish",
        "reliability": "medium",
        "description": "两根阴线收盘价相同，下跌动力耗尽，底部反转信号",
    },
    "CDLMATHOLD": {
        "name": "垫形",
        "signal": "bullish",
        "reliability": "medium",
        "description": "上升趋势中跳空高开后小幅回落，随后恢复上涨，趋势延续信号",
    },
    "CDLMORNINGDOJISTAR": {
        "name": "早晨十字星",
        "signal": "bullish",
        "reliability": "high",
        "description": "早晨之星变体，中间为十字星，底部反转信号",
    },
    "CDLMORNINGSTAR": {
        "name": "早晨之星",
        "signal": "bullish",
        "reliability": "high",
        "description": "三根K线组合，大阴线后跳空小实体再收大阳线，底部反转信号",
    },
    "CDLPIERCING": {
        "name": "穿刺形态",
        "signal": "bullish",
        "reliability": "medium",
        "description": "阳线收盘价深入前日阴线实体过半，底部反转信号",
    },
    "CDLSTICKSANDWICH": {
        "name": "条形三明治",
        "signal": "bullish",
        "reliability": "medium",
        "description": "两根阴线夹一根阳线且收盘价相同，底部反转信号",
    },
    "CDLTAKURI": {
        "name": "探水竿",
        "signal": "bullish",
        "reliability": "medium",
        "description": "下影线极长的锤子线变体，底部反转信号",
    },
    "CDLUNIQUE3RIVER": {
        "name": "奇特三河床",
        "signal": "bullish",
        "reliability": "medium",
        "description": "三根K线组合，大阴线后小阴线再收阳线，底部反转信号",
    },
    # ---- 中性形态（含双向形态与无方向形态）----
    "CDL3INSIDE": {
        "name": "三内升/三内跌",
        "signal": "neutral",
        "reliability": "medium",
        "description": "三根K线组合，第二根被第一根包含，第三根确认方向，可涨可跌",
    },
    "CDL3LINESTRIKE": {
        "name": "三线打击",
        "signal": "neutral",
        "reliability": "low",
        "description": "四根K线组合，前三根延续趋势，第四根反向吞没，趋势中继或反转信号",
    },
    "CDL3OUTSIDE": {
        "name": "三外升/三外跌",
        "signal": "neutral",
        "reliability": "medium",
        "description": "三根K线组合，第二根吞没第一根，第三根确认方向，可涨可跌",
    },
    "CDLABANDONEDBABY": {
        "name": "弃婴",
        "signal": "neutral",
        "reliability": "high",
        "description": "三根K线组合，中间十字星跳空且两侧跳空，顶部或底部极端反转信号",
    },
    "CDLBELTHOLD": {
        "name": "捉腰带",
        "signal": "neutral",
        "reliability": "medium",
        "description": "长实体无影线或极短影线，可阳可阴，趋势延续或反转信号",
    },
    "CDLBREAKAWAY": {
        "name": "脱离",
        "signal": "neutral",
        "reliability": "low",
        "description": "五根K线组合，跳空后形成脱离形态，趋势反转确认信号",
    },
    "CDLCLOSINGMARUBOZU": {
        "name": "收盘光头光脚",
        "signal": "neutral",
        "reliability": "medium",
        "description": "收盘价等于最高或最低价的长实体K线，趋势延续信号",
    },
    "CDLCOUNTERATTACK": {
        "name": "反击",
        "signal": "neutral",
        "reliability": "medium",
        "description": "两根实体方向相反且收盘价相近的K线，趋势犹豫或反转信号",
    },
    "CDLDOJI": {
        "name": "十字星",
        "signal": "neutral",
        "reliability": "medium",
        "description": "开盘价与收盘价几乎相同，暗示市场犹豫不决",
    },
    "CDLDOJISTAR": {
        "name": "十字星",
        "signal": "neutral",
        "reliability": "low",
        "description": "十字星与前日K线形成跳空，强化反转信号",
    },
    "CDLENGULFING": {
        "name": "吞没形态",
        "signal": "neutral",
        "reliability": "high",
        "description": "阳线或阴线完全覆盖前日实体，标志性反转信号，方向取决于吞没方向",
    },
    "CDLGAPSIDESIDEWHITE": {
        "name": "向上/向下跳空并列阳线",
        "signal": "neutral",
        "reliability": "low",
        "description": "两根大小相近的阳线并列且与前日跳空，趋势延续信号",
    },
    "CDLHARAMI": {
        "name": "孕线",
        "signal": "neutral",
        "reliability": "medium",
        "description": "前日大实体包含当日小实体，趋势可能反转，方向取决于后续确认",
    },
    "CDLHARAMICROSS": {
        "name": "十字孕线",
        "signal": "neutral",
        "reliability": "medium",
        "description": "前日大实体包含当日十字星，强反转预警信号",
    },
    "CDLHIGHWAVE": {
        "name": "长腿十字",
        "signal": "neutral",
        "reliability": "low",
        "description": "上下影线均较长的十字星，市场多空分歧大",
    },
    "CDLHIKKAKE": {
        "name": "捉腰带",
        "signal": "neutral",
        "reliability": "low",
        "description": "三根K线组合，前两根高低点被包含后第三根突破，趋势延续或反转信号",
    },
    "CDLHIKKAKEMOD": {
        "name": "修正捉腰带",
        "signal": "neutral",
        "reliability": "low",
        "description": "捉腰带形态的修正版本，信号同 CDLHIKKAKE",
    },
    "CDLKICKING": {
        "name": "反冲",
        "signal": "neutral",
        "reliability": "medium",
        "description": "两根方向相反的秃头光脚K线跳空，强趋势反转或延续信号",
    },
    "CDLKICKINGBYLENGTH": {
        "name": "反冲(由长度区分)",
        "signal": "neutral",
        "reliability": "medium",
        "description": "反冲形态变体，按两根K线长度比区分强弱",
    },
    "CDLLONGLEGGEDDOJI": {
        "name": "长腿十字",
        "signal": "neutral",
        "reliability": "medium",
        "description": "上下影线均较长的十字星，市场犹豫不决",
    },
    "CDLLONGLINE": {
        "name": "长蜡烛",
        "signal": "neutral",
        "reliability": "medium",
        "description": "实体较长的K线，趋势延续信号",
    },
    "CDLMARUBOZU": {
        "name": "光头光脚",
        "signal": "neutral",
        "reliability": "medium",
        "description": "无上下影线的长实体K线，强趋势延续信号",
    },
    "CDLRICKSHAWMAN": {
        "name": "黄包车夫",
        "signal": "neutral",
        "reliability": "low",
        "description": "长上下影线、小实体的K线，市场方向犹豫信号",
    },
    "CDLRISEFALL3METHODS": {
        "name": "上升/下降三法",
        "signal": "neutral",
        "reliability": "medium",
        "description": "五根K线组合，大实体后三根小实体被包含再大实体延续，趋势中继信号",
    },
    "CDLSEPARATINGLINES": {
        "name": "分离线",
        "signal": "neutral",
        "reliability": "low",
        "description": "两根K线开盘价相同但方向相反，趋势延续信号",
    },
    "CDLSHORTLINE": {
        "name": "短蜡烛",
        "signal": "neutral",
        "reliability": "low",
        "description": "实体极短的K线，预示趋势犹豫",
    },
    "CDLSPINNINGTOP": {
        "name": "陀螺",
        "signal": "neutral",
        "reliability": "low",
        "description": "小实体、长上下影线的K线，市场多空平衡",
    },
    "CDLTASUKIGAP": {
        "name": "缺口",
        "signal": "neutral",
        "reliability": "medium",
        "description": "跳空后并列阴阳线，趋势延续信号",
    },
    "CDLTRISTAR": {
        "name": "三星",
        "signal": "neutral",
        "reliability": "low",
        "description": "三根十字星组合，趋势反转预警信号",
    },
    "CDLXSIDEGAP3METHODS": {
        "name": "向下跳空三法",
        "signal": "neutral",
        "reliability": "medium",
        "description": "五根K线组合，跳空后三根小实体被包含再延续，趋势中继信号",
    },
}


class CandlestickPatternRecognizer:
    """K线形态识别器

    对 TA-Lib 61 种 CDL 蜡烛图形态提供高层封装，支持批量识别、
    近期形态提取与信号汇总。

    典型用法::

        recognizer = CandlestickPatternRecognizer()
        patterns = recognizer.recognize(ohlc_df)
        recent = recognizer.recognize_recent(ohlc_df, lookback=20)
        summary = recognizer.summarize_signals(ohlc_df, lookback=60)
    """

    def __init__(self):
        if not HAS_TALIB:
            raise ImportError("TA-Lib 未安装，请 pip install TA-Lib")

    def get_all_patterns(self) -> List[str]:
        """返回全部 61 个 CDL 形态函数名"""
        return list(PATTERN_DEFINITIONS.keys())

    def recognize(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别单只股票的全部 K 线形态

        参数:
            df: 单只股票的 OHLC 数据，必须包含列 date/open/high/low/close

        返回:
            DataFrame，列: date, pattern_name, signal, chinese_name,
            signal_type, reliability。仅包含检测到形态的行。
        """
        if df.empty:
            return pd.DataFrame(
                columns=["date", "pattern_name", "signal",
                         "chinese_name", "signal_type", "reliability"]
            )

        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"输入数据缺少必要列: {col}")

        df_sorted = df.sort_values("date").reset_index(drop=True)
        open_p = df_sorted["open"].values.astype(float)
        high = df_sorted["high"].values.astype(float)
        low = df_sorted["low"].values.astype(float)
        close = df_sorted["close"].values.astype(float)
        dates = df_sorted["date"].values

        results = []
        for pattern_name in self.get_all_patterns():
            func = getattr(talib, pattern_name, None)
            if func is None:
                continue
            try:
                signals = func(open_p, high, low, close)
            except Exception as e:
                print(f"计算 {pattern_name} 失败: {e}")
                continue

            nonzero_idx = np.where(signals != 0)[0]
            if len(nonzero_idx) == 0:
                continue

            pattern_def = PATTERN_DEFINITIONS[pattern_name]
            for i in nonzero_idx:
                results.append({
                    "date": dates[i],
                    "pattern_name": pattern_name,
                    "signal": int(signals[i]),
                    "chinese_name": pattern_def["name"],
                    "signal_type": pattern_def["signal"],
                    "reliability": pattern_def["reliability"],
                })

        if not results:
            return pd.DataFrame(
                columns=["date", "pattern_name", "signal",
                         "chinese_name", "signal_type", "reliability"]
            )

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(
            ["date", "pattern_name"]
        ).reset_index(drop=True)
        return result_df

    def recognize_recent(self, df: pd.DataFrame,
                         lookback: int = 20) -> List[Dict]:
        """返回最近 N 个交易日检测到的形态

        参数:
            df: 单只股票的 OHLC 数据
            lookback: 回看交易日数，默认 20

        返回:
            形态列表，按日期降序排列
        """
        all_patterns = self.recognize(df)
        if all_patterns.empty:
            return []

        df_sorted = df.sort_values("date").reset_index(drop=True)
        n = min(lookback, len(df_sorted))
        recent_dates = set(df_sorted["date"].iloc[-n:].values)

        recent = all_patterns[all_patterns["date"].isin(recent_dates)]
        recent = recent.sort_values("date", ascending=False).reset_index(
            drop=True
        )
        return recent.to_dict("records")

    def summarize_signals(self, df: pd.DataFrame,
                          lookback: int = 60) -> Dict:
        """汇总最近 N 个交易日的形态信号统计

        参数:
            df: 单只股票的 OHLC 数据
            lookback: 回看交易日数，默认 60

        返回:
            {
                "bullish_count": 看涨形态数,
                "bearish_count": 看跌形态数,
                "recent_patterns": 形态列表,
                "dominant_signal": "bullish"/"bearish"/"neutral"
            }
        """
        all_patterns = self.recognize(df)
        if all_patterns.empty:
            return {
                "bullish_count": 0,
                "bearish_count": 0,
                "recent_patterns": [],
                "dominant_signal": "neutral",
            }

        df_sorted = df.sort_values("date").reset_index(drop=True)
        n = min(lookback, len(df_sorted))
        recent_dates = set(df_sorted["date"].iloc[-n:].values)

        recent = all_patterns[all_patterns["date"].isin(recent_dates)]
        recent = recent.sort_values("date", ascending=False).reset_index(
            drop=True
        )

        bullish_count = int((recent["signal"] > 0).sum())
        bearish_count = int((recent["signal"] < 0).sum())

        if bullish_count > bearish_count:
            dominant = "bullish"
        elif bearish_count > bullish_count:
            dominant = "bearish"
        else:
            dominant = "neutral"

        return {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "recent_patterns": recent.to_dict("records"),
            "dominant_signal": dominant,
        }
