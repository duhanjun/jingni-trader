"""reports-engine L2 单元测试：charts.kline_chart.KlineChartGenerator。

覆盖：
- generate() 返回 plotly Figure 对象
- 生成的图表包含 K 线 candlestick trace
- 生成的图表包含成交量 bar trace
- 指定 ma_periods 时添加对应均线
- 提供支撑阻力位时添加水平线
- 空 DataFrame 应返回空 figure（不抛异常）
- 默认参数应正常生成图表
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


def _load_kline_chart_module():
    """加载 reports-engine/scripts/charts/kline_chart.py 为独立模块。

    临时把 scripts 包指向 reports-engine/scripts，使 `from scripts.config import CHART_THEME`
    能正确解析；加载完毕后恢复原 scripts 缓存。
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

    try:
        target_path = os.path.join(REPORTS_ENGINE_DIR, "scripts", "charts", "kline_chart.py")
        spec = ilu.spec_from_file_location("_re_kline_chart", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_re_kline_chart"] = mod
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
class TestKlineChartGenerator:
    """KlineChartGenerator 单元测试。"""

    @pytest.fixture
    def generator(self):
        mod = _load_kline_chart_module()
        return mod.KlineChartGenerator()

    @pytest.fixture
    def ohlcv_df(self):
        """单只股票的 OHLCV 数据。"""
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-06-30"
        )
        return df

    def test_generate_returns_figure(self, generator, ohlcv_df):
        """generate() 返回 plotly Figure 对象。"""
        fig = generator.generate(ohlcv_df)
        assert isinstance(fig, go.Figure)

    def test_figure_has_candlestick_trace(self, generator, ohlcv_df):
        """生成的图表包含 K 线 candlestick trace。"""
        fig = generator.generate(ohlcv_df)
        candlestick_traces = [t for t in fig.data if isinstance(t, go.Candlestick)]
        assert len(candlestick_traces) == 1

    def test_figure_has_volume_trace(self, generator, ohlcv_df):
        """生成的图表包含成交量 bar trace。"""
        fig = generator.generate(ohlcv_df)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) >= 1

    def test_ma_lines_added_when_specified(self, generator, ohlcv_df):
        """指定 ma_periods 时添加对应均线 scatter trace。"""
        fig = generator.generate(ohlcv_df, ma_periods=[5, 10, 20])
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        assert len(scatter_traces) == 3
        ma_names = {t.name for t in scatter_traces}
        assert ma_names == {"MA5", "MA10", "MA20"}

    def test_support_resistance_lines_added(self, generator, ohlcv_df):
        """提供支撑阻力位时添加水平线（layout.shapes）。"""
        sr = {
            "support": [9.0, 9.5],
            "resistance": [11.0, 11.5],
        }
        fig = generator.generate(ohlcv_df, support_resistance=sr)
        n_shapes = len(fig.layout.shapes) if fig.layout.shapes else 0
        assert n_shapes >= 4  # 2 support + 2 resistance

    def test_empty_dataframe_returns_empty_figure(self, generator):
        """空 DataFrame 应返回空 figure，不抛异常。"""
        empty_df = pd.DataFrame()
        fig = generator.generate(empty_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_none_dataframe_returns_empty_figure(self, generator):
        """None DataFrame 应返回空 figure，不抛异常。"""
        fig = generator.generate(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_default_parameters(self, generator, ohlcv_df):
        """使用默认参数应正常生成图表（默认 ma_periods=[5,10,20,60]）。"""
        fig = generator.generate(ohlcv_df)
        assert isinstance(fig, go.Figure)
        # 1 candlestick + 4 MA scatters + 1 volume bar = 6 traces
        assert len(fig.data) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
