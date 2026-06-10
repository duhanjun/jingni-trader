"""
算子注册表
借鉴: quant-stream 函数库分层设计 (Cross-sectional / Rolling / Technical / Math)
"""
import pandas as pd
import numpy as np
from typing import Callable, Dict


class OperatorRegistry:
    """算子注册表，按操作类型分类，支持动态扩展"""

    CATEGORY_LABELS = {
        "cross_sectional": "横截面",
        "rolling_window": "滚动窗口",
        "element_wise": "逐元素",
        "technical": "技术指标",
    }

    def __init__(self):
        self._operators: Dict[str, dict] = {}
        self._register_defaults()

    def _register_defaults(self):
        # ── 横截面算子 ──
        self.register("RANK", self._op_rank, "cross_sectional")
        self.register("SCALE", self._op_scale, "cross_sectional")
        self.register("ZSCORE", self._op_zscore, "cross_sectional")
        self.register("MEAN", self._op_cross_mean, "cross_sectional")
        self.register("STD", self._op_cross_std, "cross_sectional")
        self.register("MAX", self._op_cross_max, "cross_sectional")
        self.register("MIN", self._op_cross_min, "cross_sectional")

        # ── 滚动窗口算子 ──
        self.register("TS_MEAN", self._op_ts_mean, "rolling_window")
        self.register("TS_STD", self._op_ts_std, "rolling_window")
        self.register("TS_MAX", self._op_ts_max, "rolling_window")
        self.register("TS_MIN", self._op_ts_min, "rolling_window")
        self.register("TS_RANK", self._op_ts_rank, "rolling_window")
        self.register("TS_CORR", self._op_ts_corr, "rolling_window")

        # ── 逐元素算子 ──
        self.register("DELTA", self._op_delta, "element_wise")
        self.register("DELAY", self._op_delay, "element_wise")
        self.register("ABS", self._op_abs, "element_wise")
        self.register("LOG", self._op_log, "element_wise")
        self.register("EXP", self._op_exp, "element_wise")
        self.register("SQRT", self._op_sqrt, "element_wise")

        # ── 技术指标 ──
        self.register("SMA", self._op_sma, "technical")
        self.register("EMA", self._op_ema, "technical")
        self.register("RSI", self._op_rsi, "technical")
        self.register("MACD", self._op_macd, "technical")

    def register(self, name: str, fn: Callable, category: str = "custom"):
        self._operators[name.upper()] = {"fn": fn, "category": category}

    def get(self, name: str) -> Callable:
        name = name.upper()
        if name not in self._operators:
            raise ValueError(f"未知算子: {name}. 可用: {list(self._operators.keys())}")
        return self._operators[name]["fn"]

    def get_category(self, name: str) -> str:
        return self._operators.get(name.upper(), {}).get("category", "unknown")

    def list_all(self) -> Dict[str, str]:
        return {k: v["category"] for k, v in self._operators.items()}

    # ── 横截面算子实现 ──
    @staticmethod
    def _op_rank(series: pd.Series) -> pd.Series:
        return series.rank(pct=True)

    @staticmethod
    def _op_scale(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    @staticmethod
    def _op_zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std

    @staticmethod
    def _op_cross_mean(series: pd.Series) -> pd.Series:
        return pd.Series(series.mean(), index=series.index)

    @staticmethod
    def _op_cross_std(series: pd.Series) -> pd.Series:
        return pd.Series(series.std(), index=series.index)

    @staticmethod
    def _op_cross_max(series: pd.Series) -> pd.Series:
        return pd.Series(series.max(), index=series.index)

    @staticmethod
    def _op_cross_min(series: pd.Series) -> pd.Series:
        return pd.Series(series.min(), index=series.index)

    # ── 滚动窗口算子 ──
    def _op_ts_mean(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).mean()

    def _op_ts_std(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).std()

    def _op_ts_max(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).max()

    def _op_ts_min(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).min()

    def _op_ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(5, window // 2)).rank(pct=True)

    def _op_ts_corr(self, series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
        return series_a.rolling(window, min_periods=max(10, window // 2)).corr(series_b)

    # ── 逐元素算子 ──
    def _op_delta(self, series: pd.Series, periods: int = 1) -> pd.Series:
        return series.diff(periods)

    def _op_delay(self, series: pd.Series, periods: int = 1) -> pd.Series:
        return series.shift(periods)

    def _op_abs(self, series: pd.Series) -> pd.Series:
        return series.abs()

    def _op_log(self, series: pd.Series) -> pd.Series:
        return np.log(series.replace(0, np.nan))

    def _op_exp(self, series: pd.Series) -> pd.Series:
        return np.exp(series)

    def _op_sqrt(self, series: pd.Series) -> pd.Series:
        return np.sqrt(series.abs())

    # ── 技术指标 ──
    def _op_sma(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window, min_periods=max(3, window // 2)).mean()

    def _op_ema(self, series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, min_periods=max(3, window // 2)).mean()

    def _op_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _op_macd(self, series: pd.Series) -> pd.Series:
        ema12 = series.ewm(span=12, min_periods=12).mean()
        ema26 = series.ewm(span=26, min_periods=26).mean()
        return ema12 - ema26