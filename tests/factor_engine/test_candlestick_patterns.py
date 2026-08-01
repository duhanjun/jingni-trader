"""factor-engine L2 单元测试：CandlestickPatternRecognizer。

覆盖：
- PATTERN_DEFINITIONS 包含全部 61 种形态
- get_all_patterns() 返回 61 个名称
- recognize() 在 mocked talib 下识别形态
- recognize_recent() 按日期降序返回
- summarize_signals() 返回 bullish_count/bearish_count/dominant_signal
- 空 DataFrame 边界场景
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np

from synthetic_data import make_synthetic_daily


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")


def _load_candlestick_patterns():
    """加载 scripts.patterns.candlestick_patterns 为独立模块。

    candlestick_patterns.py 无相对导入（仅 `import talib`），
    可直接作为顶层模块加载。
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    for _m in ("talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    target_path = os.path.join(
        FACTOR_ENGINE_DIR, "scripts", "patterns", "candlestick_patterns.py"
    )
    spec = ilu.spec_from_file_location("_fe_candlestick_patterns", target_path)
    mod = ilu.module_from_spec(spec)
    sys.modules["_fe_candlestick_patterns"] = mod
    spec.loader.exec_module(mod)

    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v

    return mod


def _make_fake_talib_all_zeros(pattern_names, n):
    """构造一个 talib 替身，所有 CDL 函数返回长度为 n 的全 0 数组。"""
    fake = mock.MagicMock()

    def _zero(*args, **kwargs):
        return np.zeros(len(args[0]), dtype=int)

    for name in pattern_names:
        setattr(fake, name, mock.MagicMock(side_effect=_zero))
    return fake


def _make_fake_talib_with_signal(pattern_names, n, signals_map):
    """构造 talib 替身，signals_map 指定的形态返回非零信号。

    Args:
        signals_map: {pattern_name: int_array} 例如
            {"CDLHAMMER": arr_with_100_at_last_day}
    """
    fake = mock.MagicMock()

    def _zero(*args, **kwargs):
        return np.zeros(len(args[0]), dtype=int)

    for name in pattern_names:
        if name in signals_map:
            arr = signals_map[name]
            setattr(fake, name, mock.MagicMock(return_value=arr))
        else:
            setattr(fake, name, mock.MagicMock(side_effect=_zero))
    return fake


# =====================================================================
# 形态定义与枚举
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestPatternDefinitions:
    """PATTERN_DEFINITIONS 静态校验。"""

    def test_pattern_definitions_contains_61(self):
        """PATTERN_DEFINITIONS 包含 61 种形态"""
        mod = _load_candlestick_patterns()
        assert len(mod.PATTERN_DEFINITIONS) == 61

    def test_get_all_patterns_returns_61(self):
        """get_all_patterns() 返回 61 个名称"""
        mod = _load_candlestick_patterns()
        recognizer = mod.CandlestickPatternRecognizer()
        patterns = recognizer.get_all_patterns()
        assert len(patterns) == 61

    def test_each_pattern_has_required_fields(self):
        """每个形态定义含 name/signal/reliability/description 字段"""
        mod = _load_candlestick_patterns()
        for name, defn in mod.PATTERN_DEFINITIONS.items():
            assert "name" in defn, f"{name} 缺 name"
            assert "signal" in defn, f"{name} 缺 signal"
            assert "reliability" in defn, f"{name} 缺 reliability"
            assert "description" in defn, f"{name} 缺 description"
            assert defn["signal"] in ("bullish", "bearish", "neutral")


# =====================================================================
# recognize() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestRecognize:
    """recognize() 行为测试。"""

    def test_recognize_with_all_zeros_returns_empty(self):
        """所有 CDL 函数返回 0 → recognize() 返回空 DataFrame（带列名）"""
        mod = _load_candlestick_patterns()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-03-31"
        )
        n = len(df)

        fake_talib = _make_fake_talib_all_zeros(
            mod.PATTERN_DEFINITIONS.keys(), n
        )
        mod.talib = fake_talib

        recognizer = mod.CandlestickPatternRecognizer()
        result = recognizer.recognize(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = {
            "date", "pattern_name", "signal",
            "chinese_name", "signal_type", "reliability",
        }
        assert set(result.columns) == expected_cols

    def test_recognize_with_signal_returns_rows(self):
        """有形态命中 → recognize() 返回对应行"""
        mod = _load_candlestick_patterns()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-03-31"
        )
        n = len(df)

        # CDLHAMMER 在最后一天返回 100
        hammer_arr = np.zeros(n, dtype=int)
        hammer_arr[-1] = 100

        fake_talib = _make_fake_talib_with_signal(
            mod.PATTERN_DEFINITIONS.keys(), n,
            {"CDLHAMMER": hammer_arr},
        )
        mod.talib = fake_talib

        recognizer = mod.CandlestickPatternRecognizer()
        result = recognizer.recognize(df)
        assert not result.empty
        assert len(result) == 1
        row = result.iloc[0]
        assert row["pattern_name"] == "CDLHAMMER"
        assert row["signal"] == 100
        assert row["signal_type"] == "bullish"
        assert row["chinese_name"] == "锤子线"

    def test_recognize_with_empty_dataframe(self):
        """空 DataFrame → 返回带列名的空 DataFrame"""
        mod = _load_candlestick_patterns()
        recognizer = mod.CandlestickPatternRecognizer()
        result = recognizer.recognize(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = {
            "date", "pattern_name", "signal",
            "chinese_name", "signal_type", "reliability",
        }
        assert set(result.columns) == expected_cols


# =====================================================================
# recognize_recent() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestRecognizeRecent:
    """recognize_recent() 行为测试。"""

    def test_recognize_recent_sorted_descending(self):
        """recognize_recent() 按日期降序返回"""
        mod = _load_candlestick_patterns()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-03-31"
        )
        n = len(df)

        # CDLHAMMER 在多个日期返回信号
        hammer_arr = np.zeros(n, dtype=int)
        hammer_arr[-1] = 100
        hammer_arr[-5] = 100
        hammer_arr[-10] = 100

        fake_talib = _make_fake_talib_with_signal(
            mod.PATTERN_DEFINITIONS.keys(), n,
            {"CDLHAMMER": hammer_arr},
        )
        mod.talib = fake_talib

        recognizer = mod.CandlestickPatternRecognizer()
        recent = recognizer.recognize_recent(df, lookback=30)
        assert isinstance(recent, list)
        assert len(recent) == 3

        # 验证日期降序
        dates = [r["date"] for r in recent]
        assert dates == sorted(dates, reverse=True)

    def test_recognize_recent_empty_when_no_patterns(self):
        """无形态命中 → recognize_recent() 返回空列表"""
        mod = _load_candlestick_patterns()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-03-31"
        )
        n = len(df)
        fake_talib = _make_fake_talib_all_zeros(
            mod.PATTERN_DEFINITIONS.keys(), n
        )
        mod.talib = fake_talib

        recognizer = mod.CandlestickPatternRecognizer()
        recent = recognizer.recognize_recent(df, lookback=20)
        assert recent == []


# =====================================================================
# summarize_signals() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestSummarizeSignals:
    """summarize_signals() 行为测试。"""

    def test_summarize_returns_required_fields(self):
        """summarize_signals() 返回 bullish_count/bearish_count/dominant_signal 等字段"""
        mod = _load_candlestick_patterns()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-03-31"
        )
        n = len(df)

        # CDLHAMMER (bullish) + CDLSHOOTINGSTAR (bearish)
        hammer_arr = np.zeros(n, dtype=int)
        hammer_arr[-1] = 100
        star_arr = np.zeros(n, dtype=int)
        star_arr[-2] = -100

        fake_talib = _make_fake_talib_with_signal(
            mod.PATTERN_DEFINITIONS.keys(), n,
            {"CDLHAMMER": hammer_arr, "CDLSHOOTINGSTAR": star_arr},
        )
        mod.talib = fake_talib

        recognizer = mod.CandlestickPatternRecognizer()
        summary = recognizer.summarize_signals(df, lookback=60)
        assert "bullish_count" in summary
        assert "bearish_count" in summary
        assert "dominant_signal" in summary
        assert "recent_patterns" in summary
        assert summary["bullish_count"] == 1
        assert summary["bearish_count"] == 1
        # 相等 → neutral
        assert summary["dominant_signal"] == "neutral"

    def test_summarize_dominant_bullish(self):
        """看涨形态多于看跌 → dominant_signal='bullish'"""
        mod = _load_candlestick_patterns()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-03-31"
        )
        n = len(df)

        hammer_arr = np.zeros(n, dtype=int)
        hammer_arr[-1] = 100
        hammer_arr[-3] = 100

        fake_talib = _make_fake_talib_with_signal(
            mod.PATTERN_DEFINITIONS.keys(), n,
            {"CDLHAMMER": hammer_arr},
        )
        mod.talib = fake_talib

        recognizer = mod.CandlestickPatternRecognizer()
        summary = recognizer.summarize_signals(df, lookback=60)
        assert summary["bullish_count"] == 2
        assert summary["bearish_count"] == 0
        assert summary["dominant_signal"] == "bullish"

    def test_summarize_with_empty_dataframe(self):
        """空 DataFrame → 全部 0，dominant_signal='neutral'"""
        mod = _load_candlestick_patterns()
        recognizer = mod.CandlestickPatternRecognizer()
        summary = recognizer.summarize_signals(pd.DataFrame(), lookback=60)
        assert summary["bullish_count"] == 0
        assert summary["bearish_count"] == 0
        assert summary["dominant_signal"] == "neutral"
        assert summary["recent_patterns"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
