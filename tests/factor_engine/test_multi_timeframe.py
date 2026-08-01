"""factor-engine L2 单元测试：MultiTimeframeAnalyzer。

覆盖：
- analyze() 返回 timeframes 字典含 daily/weekly/monthly
- _resample() 正确聚合日线到周线
- 每个周期含 trend/strength/signals
- _detect_resonance() 看多/看空/中性场景
- _detect_divergence()
- 数据不足（< 60 行）
- 空 DataFrame
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


def _load_multi_timeframe():
    """加载 scripts.multi_timeframe 为独立模块。

    multi_timeframe.py 无相对导入（仅 `import talib`），
    可直接作为顶层模块加载。由于 conftest 已 mock talib，
    模块加载后 HAS_TALIB=True，但 talib 函数返回 MagicMock。
    测试中通过 monkeypatch HAS_TALIB=False 强制走纯 pandas 回退路径。
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
        FACTOR_ENGINE_DIR, "scripts", "multi_timeframe.py"
    )
    spec = ilu.spec_from_file_location("_fe_multi_timeframe", target_path)
    mod = ilu.module_from_spec(spec)
    sys.modules["_fe_multi_timeframe"] = mod
    spec.loader.exec_module(mod)

    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v

    return mod


def _make_single_stock_df(n_days=120, start="2023-06-01", seed=42):
    """构造单只股票的 OHLCV 日线数据，足够多周期分析。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, periods=n_days)
    base = 10.0
    closes = base * (1 + np.cumsum(rng.normal(0, 0.015, n_days)))
    opens = closes * (1 + rng.normal(0, 0.002, n_days))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    vol = rng.randint(1_000_000, 10_000_000, n_days)
    return pd.DataFrame({
        "code": "000001.SZ",
        "date": dates,
        "open": opens.round(2),
        "high": highs.round(2),
        "low": lows.round(2),
        "close": closes.round(2),
        "volume": vol,
    })


# =====================================================================
# analyze() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestMultiTimeframeAnalyze:
    """analyze() 端到端行为测试。"""

    def test_analyze_returns_timeframes_dict(self):
        """analyze() 返回含 daily/weekly/monthly 的 timeframes 字典"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False  # 强制纯 pandas 回退

        df = _make_single_stock_df(n_days=120)
        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)

        assert "timeframes" in result
        tfs = result["timeframes"]
        assert "daily" in tfs
        assert "weekly" in tfs
        assert "monthly" in tfs

    def test_each_timeframe_has_required_fields(self):
        """每个周期含 trend/strength/signals 字段"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        df = _make_single_stock_df(n_days=120)
        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)

        for tf_name in ("daily", "weekly", "monthly"):
            tf = result["timeframes"][tf_name]
            assert "trend" in tf
            assert "strength" in tf
            assert "signals" in tf
            assert "indicators" in tf
            assert tf["trend"] in ("上涨", "下跌", "震荡", "未知")
            assert tf["strength"] in ("强", "中", "弱", "无")

    def test_analyze_returns_resonance_and_summary(self):
        """analyze() 返回 resonance/divergences/summary"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        df = _make_single_stock_df(n_days=120)
        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)

        assert "resonance" in result
        assert "divergences" in result
        assert "summary" in result
        assert isinstance(result["divergences"], list)
        assert isinstance(result["summary"], str)

    def test_analyze_with_insufficient_data(self):
        """数据 < 60 行 → 返回空结果"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        df = _make_single_stock_df(n_days=30)
        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)

        # _empty_result 返回的 trend 应为 "未知"
        for tf_name in ("daily", "weekly", "monthly"):
            assert result["timeframes"][tf_name]["trend"] == "未知"
        assert result["resonance"]["bullish"] is False
        assert result["resonance"]["bearish"] is False
        assert "数据不足" in result["summary"]

    def test_analyze_with_empty_dataframe(self):
        """空 DataFrame → 返回空结果"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer.analyze(pd.DataFrame())

        for tf_name in ("daily", "weekly", "monthly"):
            assert result["timeframes"][tf_name]["trend"] == "未知"
        assert result["code"] is None

    def test_analyze_missing_columns_raises(self):
        """缺少必要列 → 抛 ValueError"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        analyzer = mod.MultiTimeframeAnalyzer()
        bad_df = pd.DataFrame({"code": ["000001.SZ"], "date": ["2024-01-01"]})
        with pytest.raises(ValueError, match="缺少必要列"):
            analyzer.analyze(bad_df)


# =====================================================================
# _resample() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestResample:
    """_resample() 重采样测试。"""

    def test_resample_weekly_aggregates_correctly(self):
        """周线重采样：open=first, high=max, low=min, close=last, volume=sum"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()

        df = _make_single_stock_df(n_days=30)
        weekly = analyzer._resample(df, "weekly")

        # 周线行数应远少于日线
        assert len(weekly) < len(df)
        assert len(weekly) > 0
        # 列齐全
        for col in ("open", "high", "low", "close", "volume"):
            assert col in weekly.columns

    def test_resample_monthly_aggregates_correctly(self):
        """月线重采样正确聚合"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()

        df = _make_single_stock_df(n_days=120)
        monthly = analyzer._resample(df, "monthly")

        assert len(monthly) < len(df)
        assert len(monthly) > 0
        for col in ("open", "high", "low", "close", "volume"):
            assert col in monthly.columns

    def test_resample_daily_returns_copy(self):
        """daily 频率 → 返回副本"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()

        df = _make_single_stock_df(n_days=10)
        daily = analyzer._resample(df, "daily")
        assert len(daily) == len(df)

    def test_resample_invalid_freq_raises(self):
        """不支持的时间周期 → 抛 ValueError"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()

        df = _make_single_stock_df(n_days=10)
        with pytest.raises(ValueError, match="不支持的时间周期"):
            analyzer._resample(df, "quarterly")

    def test_resample_weekly_ohlc_correctness(self):
        """周线 OHLC 聚合语义正确（取首日 open、最高 high、最低 low、末日 close）"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()

        # 构造 5 个交易日的数据（恰好一周）
        dates = pd.bdate_range("2024-01-01", periods=5)
        df = pd.DataFrame({
            "code": "000001.SZ",
            "date": dates,
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [10.5, 11.5, 12.5, 13.5, 14.5],
            "low": [9.5, 10.5, 11.5, 12.5, 13.5],
            "close": [11.0, 12.0, 13.0, 14.0, 15.0],
            "volume": [1000, 2000, 3000, 4000, 5000],
        })
        weekly = analyzer._resample(df, "weekly")
        assert len(weekly) == 1
        row = weekly.iloc[0]
        assert row["open"] == 10.0  # first
        assert row["high"] == 14.5  # max
        assert row["low"] == 9.5    # min
        assert row["close"] == 15.0  # last
        assert row["volume"] == 15000  # sum


# =====================================================================
# _detect_resonance() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestDetectResonance:
    """_detect_resonance() 多周期共振检测测试。"""

    def test_all_bullish_resonance(self):
        """三周期全部上涨 → all_bullish=True, bullish=True"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()
        tf_results = {
            "daily": {"trend": "上涨"},
            "weekly": {"trend": "上涨"},
            "monthly": {"trend": "上涨"},
        }
        resonance = analyzer._detect_resonance(tf_results)
        assert resonance["bullish"] is True
        assert resonance["all_bullish"] is True
        assert resonance["bearish"] is False
        assert "三周期共振看多" in resonance["description"]

    def test_all_bearish_resonance(self):
        """三周期全部下跌 → all_bearish=True, bearish=True"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()
        tf_results = {
            "daily": {"trend": "下跌"},
            "weekly": {"trend": "下跌"},
            "monthly": {"trend": "下跌"},
        }
        resonance = analyzer._detect_resonance(tf_results)
        assert resonance["bearish"] is True
        assert resonance["all_bearish"] is True
        assert resonance["bullish"] is False
        assert "三周期共振看空" in resonance["description"]

    def test_partial_bullish_resonance(self):
        """日线+月线同步上涨（周线不同步）→ bullish=True, all_bullish=False"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()
        tf_results = {
            "daily": {"trend": "上涨"},
            "weekly": {"trend": "震荡"},
            "monthly": {"trend": "上涨"},
        }
        resonance = analyzer._detect_resonance(tf_results)
        assert resonance["bullish"] is True
        assert resonance["all_bullish"] is False
        assert "日线+月线同步看多" in resonance["description"]

    def test_neutral_resonance(self):
        """混合趋势 → 无共振"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()
        tf_results = {
            "daily": {"trend": "上涨"},
            "weekly": {"trend": "下跌"},
            "monthly": {"trend": "震荡"},
        }
        resonance = analyzer._detect_resonance(tf_results)
        assert resonance["bullish"] is False
        assert resonance["bearish"] is False
        assert "无明确多空共振" in resonance["description"]


# =====================================================================
# _detect_divergence() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestDetectDivergence:
    """_detect_divergence() 背离检测测试。"""

    def test_detect_divergence_returns_list(self):
        """_detect_divergence() 返回 list（可能为空）"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        analyzer = mod.MultiTimeframeAnalyzer()
        df = _make_single_stock_df(n_days=120)
        divergences = analyzer._detect_divergence(df, "daily")
        assert isinstance(divergences, list)

    def test_detect_divergence_with_insufficient_data(self):
        """数据 < 30 行 → 返回空列表"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        analyzer = mod.MultiTimeframeAnalyzer()
        df = _make_single_stock_df(n_days=20)
        divergences = analyzer._detect_divergence(df, "daily")
        assert divergences == []

    def test_detect_divergence_each_entry_structure(self):
        """每个背离条目含 type/timeframe/indicator/description 字段（若有）"""
        mod = _load_multi_timeframe()
        mod.HAS_TALIB = False

        analyzer = mod.MultiTimeframeAnalyzer()
        df = _make_single_stock_df(n_days=120, seed=7)
        divergences = analyzer._detect_divergence(df, "daily")
        for d in divergences:
            assert "type" in d
            assert "timeframe" in d
            assert d["type"] in ("顶背离", "底背离")
            assert d["timeframe"] == "daily"


# =====================================================================
# _empty_result() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestEmptyResult:
    """空结果结构测试。"""

    def test_empty_result_has_all_timeframes(self):
        """_empty_result() 含全部三个周期"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer._empty_result()
        for tf in ("daily", "weekly", "monthly"):
            assert tf in result["timeframes"]
            assert result["timeframes"][tf]["trend"] == "未知"

    def test_empty_result_resonance_neutral(self):
        """_empty_result() 的 resonance 全部 False"""
        mod = _load_multi_timeframe()
        analyzer = mod.MultiTimeframeAnalyzer()
        result = analyzer._empty_result()
        assert result["resonance"]["bullish"] is False
        assert result["resonance"]["bearish"] is False
        assert result["resonance"]["all_bullish"] is False
        assert result["resonance"]["all_bearish"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
