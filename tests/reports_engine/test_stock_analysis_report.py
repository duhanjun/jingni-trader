"""reports-engine L2 单元测试：templates.stock_analysis_report.StockAnalysisReportGenerator。

覆盖：
- generate() 生成 HTML 文件到指定路径
- HTML 含关键章节（综合评分 / 多周期分析 / 技术指标 / K线形态 / 支撑阻力 / 风险提示 / 免责声明）
- _calc_comprehensive_score() 返回 0-100 范围评分
- _generate_risk_warnings() 返回字符串列表
- 最小数据（仅 OHLCV）/ 全量数据 / 空数据 均不抛异常
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
REPORTS_ENGINE_DIR = os.path.join(ROOT, "skills", "reports-engine")


@pytest.fixture
def ReportClass():
    """加载 StockAnalysisReportGenerator 类。

    临时把 scripts 包指向 reports-engine/scripts，使 report 模块内的
    相对导入 `from ..charts.kline_chart import KlineChartGenerator` 可解析。
    fixture teardown 时恢复原 scripts 缓存。
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(REPORTS_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    # 清掉可能已加载的 stock_analysis_report 模块缓存，确保在正确的 scripts 包下重新导入
    for key in list(sys.modules.keys()):
        if "stock_analysis_report" in key:
            sys.modules.pop(key, None)

    try:
        import scripts.templates.stock_analysis_report as report_mod  # noqa: F401
        yield report_mod.StockAnalysisReportGenerator
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


# ── 测试数据构造器 ──────────────────────────────────────────

def _make_ohlcv():
    """单只股票的 OHLCV 数据。"""
    return make_synthetic_daily(
        codes=["000001.SZ"], start="2024-01-01", end="2024-06-30"
    )


def _make_technical_indicators():
    return {
        "macd_dif": 0.15,
        "macd_dea": 0.08,
        "macd_hist": 0.14,
        "rsi": 55.0,
        "kdj_k": 65.0,
        "kdj_d": 60.0,
        "kdj_j": 75.0,
        "ma5": 10.5,
        "ma10": 10.3,
        "ma20": 10.2,
        "ma60": 9.8,
        "boll_position": 0.6,
    }


def _make_pattern_results():
    return {
        "bullish_count": 5,
        "bearish_count": 2,
        "dominant_signal": "bullish",
        "recent_patterns": [
            {
                "date": "2024-06-20",
                "pattern_name": "hammer",
                "chinese_name": "锤子线",
                "signal_type": "bullish",
                "reliability": "high",
            },
        ],
    }


def _make_support_resistance():
    return {
        "support": [
            {"price": 9.5, "type": "整数关口", "strength": "强", "method": "round"},
            {"price": 9.0, "type": "前期低点", "strength": "中", "method": "swing"},
        ],
        "resistance": [
            {"price": 11.0, "type": "前高", "strength": "中", "method": "swing"},
            {"price": 11.5, "type": "整数关口", "strength": "弱", "method": "round"},
        ],
        "current_price": 10.3,
        "nearest_support": 9.5,
        "nearest_resistance": 11.0,
    }


def _make_multi_timeframe():
    return {
        "timeframes": {
            "daily": {
                "trend": "上涨",
                "strength": "中",
                "indicators": {
                    "close": 10.3,
                    "macd_dif": 0.15,
                    "rsi": 55.0,
                    "kdj_k": 65.0,
                    "kdj_d": 60.0,
                    "kdj_j": 75.0,
                },
                "signals": [{"type": "金叉"}],
            },
            "weekly": {
                "trend": "上涨",
                "strength": "强",
                "indicators": {},
                "signals": [],
            },
            "monthly": {
                "trend": "震荡",
                "strength": "中",
                "indicators": {},
                "signals": [],
            },
        },
        "resonance": {
            "bullish": True,
            "all_bullish": False,
            "bearish": False,
            "all_bearish": False,
            "description": "日线和周线同步看多",
        },
        "divergences": [],
        "summary": "短期偏多，中期震荡",
    }


def _make_fundamental_data():
    return {
        "pe": 15.5,
        "pb": 2.1,
        "pe_percentile": 0.35,
        "pb_percentile": 0.45,
        "roe": 18.0,
        "gross_margin": 0.45,
        "revenue_growth": 0.20,
        "profit_growth": 0.25,
        "market_cap": 5e10,
    }


# ── 关键章节断言 ────────────────────────────────────────────

# 模板中实际章节标题：
#   "综合评分" / "多周期技术面分析" / "技术指标信号汇总"
#   "K线形态识别结果" / "支撑阻力位" / "风险提示" / "免责声明"
# 这里取各标题的子串作为存在性断言依据
_REQUIRED_SECTIONS = [
    "综合评分",
    "多周期技术面分析",
    "技术指标",
    "K线形态",
    "支撑阻力",
    "风险提示",
    "免责声明",
]


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestStockAnalysisReportGenerator:
    """StockAnalysisReportGenerator 单元测试。"""

    def test_generate_produces_html_file(self, ReportClass, tmp_path):
        """generate() 在指定路径生成 HTML 文件。"""
        gen = ReportClass()
        output_path = str(tmp_path / "report.html")
        result = gen.generate(
            stock_code="000001.SZ",
            stock_name="测试股票",
            ohlcv_data=_make_ohlcv(),
            technical_indicators=_make_technical_indicators(),
            pattern_results=_make_pattern_results(),
            support_resistance=_make_support_resistance(),
            multi_timeframe=_make_multi_timeframe(),
            fundamental_data=_make_fundamental_data(),
            output_path=output_path,
        )
        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_html_contains_key_sections(self, ReportClass, tmp_path):
        """HTML 含关键章节标题。"""
        gen = ReportClass()
        output_path = str(tmp_path / "report_sections.html")
        gen.generate(
            stock_code="000001.SZ",
            stock_name="测试股票",
            ohlcv_data=_make_ohlcv(),
            technical_indicators=_make_technical_indicators(),
            pattern_results=_make_pattern_results(),
            support_resistance=_make_support_resistance(),
            multi_timeframe=_make_multi_timeframe(),
            fundamental_data=_make_fundamental_data(),
            output_path=output_path,
        )
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()
        for section in _REQUIRED_SECTIONS:
            assert section in html, f"HTML 缺少关键章节: {section}"

    def test_calc_comprehensive_score_range(self, ReportClass):
        """_calc_comprehensive_score() 返回的评分在 0-100 范围内。"""
        gen = ReportClass()
        scores = gen._calc_comprehensive_score(
            _make_technical_indicators(),
            _make_pattern_results(),
            _make_multi_timeframe(),
            _make_fundamental_data(),
        )
        assert isinstance(scores, dict)
        # 必需字段
        for key in ("technical", "fundamental", "comprehensive", "rating",
                    "has_fundamental", "breakdown"):
            assert key in scores, f"scores 缺少字段: {key}"
        # 评分范围 0-100
        assert 0 <= scores["technical"] <= 100
        assert scores["fundamental"] is not None
        assert 0 <= scores["fundamental"] <= 100
        assert 0 <= scores["comprehensive"] <= 100
        # rating 为非空字符串
        assert isinstance(scores["rating"], str) and scores["rating"]
        # breakdown 子项也在范围内
        bd = scores["breakdown"]
        for k in ("trend", "indicator", "pattern", "volume"):
            assert k in bd
            assert 0 <= bd[k] <= 100
        for k in ("valuation", "profitability", "growth"):
            assert k in bd
            assert bd[k] is None or 0 <= bd[k] <= 100

    def test_calc_comprehensive_score_without_fundamental(self, ReportClass):
        """无基本面数据时 fundamental=None，comprehensive=technical。"""
        gen = ReportClass()
        scores = gen._calc_comprehensive_score(
            _make_technical_indicators(),
            _make_pattern_results(),
            _make_multi_timeframe(),
            fundamental_data=None,
        )
        assert scores["fundamental"] is None
        assert scores["has_fundamental"] is False
        assert scores["comprehensive"] == scores["technical"]

    def test_generate_risk_warnings_returns_list(self, ReportClass):
        """_generate_risk_warnings() 返回字符串列表。"""
        gen = ReportClass()
        warnings = gen._generate_risk_warnings(
            _make_technical_indicators(),
            _make_pattern_results(),
            _make_multi_timeframe(),
            _make_support_resistance(),
        )
        assert isinstance(warnings, list)
        assert len(warnings) > 0
        for w in warnings:
            assert isinstance(w, str) and w

    def test_generate_risk_warnings_empty_inputs(self, ReportClass):
        """空输入时仍返回非空字符串列表（兜底提示）。"""
        gen = ReportClass()
        warnings = gen._generate_risk_warnings({}, {}, {}, {})
        assert isinstance(warnings, list)
        assert len(warnings) > 0
        for w in warnings:
            assert isinstance(w, str)

    def test_with_minimal_data_only_ohlcv(self, ReportClass, tmp_path):
        """仅 OHLCV 数据（无基本面）也应生成报告。"""
        gen = ReportClass()
        output_path = str(tmp_path / "minimal_report.html")
        gen.generate(
            stock_code="000001.SZ",
            stock_name="最小数据测试",
            ohlcv_data=_make_ohlcv(),
            technical_indicators={},
            pattern_results={},
            support_resistance={},
            multi_timeframe={},
            fundamental_data=None,
            output_path=output_path,
        )
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()
        # 即使无基本面，关键章节仍应存在
        assert "综合评分" in html
        assert "风险提示" in html
        assert "免责声明" in html

    def test_with_full_data(self, ReportClass, tmp_path):
        """全量数据输入应正常生成报告。"""
        gen = ReportClass()
        output_path = str(tmp_path / "full_report.html")
        result = gen.generate(
            stock_code="600000.SH",
            stock_name="全量数据测试",
            ohlcv_data=_make_ohlcv(),
            technical_indicators=_make_technical_indicators(),
            pattern_results=_make_pattern_results(),
            support_resistance=_make_support_resistance(),
            multi_timeframe=_make_multi_timeframe(),
            fundamental_data=_make_fundamental_data(),
            output_path=output_path,
        )
        assert result == output_path
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()
        for section in _REQUIRED_SECTIONS:
            assert section in html

    def test_with_empty_data_does_not_crash(self, ReportClass, tmp_path):
        """空数据（空 DataFrame / 空 dict）不应导致崩溃。"""
        gen = ReportClass()
        output_path = str(tmp_path / "empty_report.html")
        gen.generate(
            stock_code="000001.SZ",
            stock_name="空数据测试",
            ohlcv_data=pd.DataFrame(),
            technical_indicators={},
            pattern_results={},
            support_resistance={},
            multi_timeframe={},
            fundamental_data=None,
            output_path=output_path,
        )
        assert os.path.exists(output_path)
        with open(output_path, "r", encoding="utf-8") as f:
            html = f.read()
        # 即使无数据，报告骨架仍应包含免责声明与综合评分章节
        assert "综合评分" in html
        assert "免责声明" in html

    def test_generate_returns_html_string_when_no_path(self, ReportClass):
        """未指定 output_path 时返回 HTML 字符串。"""
        gen = ReportClass()
        html = gen.generate(
            stock_code="000001.SZ",
            stock_name="无路径测试",
            ohlcv_data=_make_ohlcv(),
            technical_indicators=_make_technical_indicators(),
            pattern_results=_make_pattern_results(),
            support_resistance=_make_support_resistance(),
            multi_timeframe=_make_multi_timeframe(),
            fundamental_data=_make_fundamental_data(),
            output_path=None,
        )
        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "免责声明" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
