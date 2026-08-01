"""reports-engine L2 单元测试：charts.indicator_chart.IndicatorChartGenerator。

覆盖：
- generate_macd_panel() 返回 Figure
- generate_rsi_panel() 返回 Figure
- generate_kdj_panel() 返回 Figure
- generate_combined_panel() 返回含多个子图的 Figure
- 空 DataFrame 应返回空 figure（不抛异常）
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import plotly.graph_objects as go

from synthetic_data import make_synthetic_daily


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_ENGINE_DIR = os.path.join(ROOT, "skills", "reports-engine")


def _load_indicator_chart_module():
    """加载 reports-engine/scripts/charts/indicator_chart.py 为独立模块。"""
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

    try:
        target_path = os.path.join(
            REPORTS_ENGINE_DIR, "scripts", "charts", "indicator_chart.py"
        )
        spec = ilu.spec_from_file_location("_re_indicator_chart", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_re_indicator_chart"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestIndicatorChartGenerator:
    """IndicatorChartGenerator 单元测试。"""

    @pytest.fixture
    def generator(self):
        mod = _load_indicator_chart_module()
        return mod.IndicatorChartGenerator()

    @pytest.fixture
    def ohlcv_df(self):
        """单只股票的 OHLCV 数据。"""
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-06-30"
        )
        return df

    def test_generate_macd_panel_returns_figure(self, generator, ohlcv_df):
        """generate_macd_panel() 返回 plotly Figure 对象。"""
        fig = generator.generate_macd_panel(ohlcv_df)
        assert isinstance(fig, go.Figure)
        # 2 subplots: DIF/DEA scatters + MACD bar = 3 traces
        assert len(fig.data) == 3
        # 验证含 DIF/DEA scatter 与 MACD bar
        scatter_names = {t.name for t in fig.data if isinstance(t, go.Scatter)}
        assert {"DIF", "DEA"}.issubset(scatter_names)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1

    def test_generate_rsi_panel_returns_figure(self, generator, ohlcv_df):
        """generate_rsi_panel() 返回 plotly Figure 对象。"""
        fig = generator.generate_rsi_panel(ohlcv_df)
        assert isinstance(fig, go.Figure)
        # 1 RSI scatter trace
        rsi_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(rsi_traces) == 1
        assert rsi_traces[0].name.startswith("RSI")

    def test_generate_kdj_panel_returns_figure(self, generator, ohlcv_df):
        """generate_kdj_panel() 返回 plotly Figure 对象。"""
        fig = generator.generate_kdj_panel(ohlcv_df)
        assert isinstance(fig, go.Figure)
        # 3 scatter traces: K, D, J
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 3
        names = {t.name for t in scatter_traces}
        assert names == {"K", "D", "J"}

    def test_generate_combined_panel_returns_figure(self, generator, ohlcv_df):
        """generate_combined_panel() 返回含多个子图的 Figure。"""
        fig = generator.generate_combined_panel(ohlcv_df)
        assert isinstance(fig, go.Figure)
        # 3 subplots: MACD (2 scatter + 1 bar) + RSI (1 scatter) + KDJ (3 scatter) = 7
        assert len(fig.data) == 7
        # 验证子图布局：含 MACD / RSI / KDJ 子图标题
        # make_subplots 的 subplot 标题存储在 layout.annotations
        annotations_text = " ".join(
            str(a.text) for a in (fig.layout.annotations or [])
        )
        assert "MACD" in annotations_text
        assert "RSI" in annotations_text
        assert "KDJ" in annotations_text

    def test_empty_dataframe_returns_empty_figure(self, generator):
        """空 DataFrame 应返回空 figure，不抛异常。"""
        empty_df = pd.DataFrame()
        # 各方法都应能处理空数据
        fig_macd = generator.generate_macd_panel(empty_df)
        assert isinstance(fig_macd, go.Figure)
        assert len(fig_macd.data) == 0

        fig_rsi = generator.generate_rsi_panel(empty_df)
        assert isinstance(fig_rsi, go.Figure)
        assert len(fig_rsi.data) == 0

        fig_kdj = generator.generate_kdj_panel(empty_df)
        assert isinstance(fig_kdj, go.Figure)
        assert len(fig_kdj.data) == 0

        fig_combined = generator.generate_combined_panel(empty_df)
        assert isinstance(fig_combined, go.Figure)
        assert len(fig_combined.data) == 0

    def test_none_dataframe_returns_empty_figure(self, generator):
        """None DataFrame 应返回空 figure，不抛异常。"""
        fig = generator.generate_macd_panel(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
