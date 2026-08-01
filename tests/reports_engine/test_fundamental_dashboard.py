"""reports-engine L2 单元测试：charts.fundamental_dashboard.FundamentalDashboardGenerator。

覆盖：
- generate_valuation_gauge() 返回 Figure
- generate_roe_trend() 返回 Figure
- generate_industry_radar() 返回 Figure
- generate_combined_dashboard() 返回 Figure
- 空 / None 数据应返回空 figure（不抛异常）
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


def _load_fundamental_dashboard_module():
    """加载 reports-engine/scripts/charts/fundamental_dashboard.py 为独立模块。"""
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
            REPORTS_ENGINE_DIR, "scripts", "charts", "fundamental_dashboard.py"
        )
        spec = ilu.spec_from_file_location("_re_fundamental_dashboard", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_re_fundamental_dashboard"] = mod
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
class TestFundamentalDashboardGenerator:
    """FundamentalDashboardGenerator 单元测试。"""

    @pytest.fixture
    def generator(self):
        mod = _load_fundamental_dashboard_module()
        return mod.FundamentalDashboardGenerator()

    @pytest.fixture
    def roe_df(self):
        """构造 ROE / 利润率趋势 DataFrame。"""
        return pd.DataFrame({
            "date": pd.bdate_range("2023Q1", periods=8, freq="QS"),
            "roe": [12.5, 13.0, 14.2, 15.0, 15.8, 16.5, 17.0, 18.2],
            "gross_margin": [38.0, 39.0, 40.5, 41.0, 42.0, 43.0, 44.0, 45.0],
            "net_margin": [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5],
        })

    @pytest.fixture
    def industry_data(self):
        return {
            "stock": {"PE": 15.0, "PB": 2.0, "ROE": 18.0, "gross_margin": 45.0},
            "industry_avg": {"PE": 20.0, "PB": 2.5, "ROE": 12.0, "gross_margin": 30.0},
        }

    @pytest.fixture
    def valuation_data(self):
        return {
            "PE": {"current": 15.5, "percentile": 0.35},
        }

    # ── generate_valuation_gauge ────────────────────────────

    def test_generate_valuation_gauge_returns_figure(self, generator):
        """generate_valuation_gauge() 返回 plotly Figure 对象。"""
        fig = generator.generate_valuation_gauge("PE", current_value=15.5, percentile=0.35)
        assert isinstance(fig, go.Figure)
        # 含一个 Indicator trace
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Indicator)

    def test_valuation_gauge_handles_percentile_above_1(self, generator):
        """percentile > 1 时按百分数处理（如 35.0 表示 35%）。"""
        fig = generator.generate_valuation_gauge("PB", current_value=2.1, percentile=45.0)
        assert isinstance(fig, go.Figure)

    # ── generate_roe_trend ───────────────────────────────────

    def test_generate_roe_trend_returns_figure(self, generator, roe_df):
        """generate_roe_trend() 返回 plotly Figure 对象。"""
        fig = generator.generate_roe_trend(roe_df)
        assert isinstance(fig, go.Figure)
        # 1 Bar (ROE) + 1 Scatter (gross_margin) + 1 Scatter (net_margin) = 3
        assert len(fig.data) == 3
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1
        assert bar_traces[0].name == "ROE(%)"

    # ── generate_industry_radar ──────────────────────────────

    def test_generate_industry_radar_returns_figure(self, generator, industry_data):
        """generate_industry_radar() 返回 plotly Figure 对象。"""
        fig = generator.generate_industry_radar(
            industry_data["stock"], industry_data["industry_avg"]
        )
        assert isinstance(fig, go.Figure)
        # 2 Scatterpolar traces: 个股 + 行业均值
        radar_traces = [t for t in fig.data if isinstance(t, go.Scatterpolar)]
        assert len(radar_traces) == 2

    # ── generate_combined_dashboard ──────────────────────────

    def test_generate_combined_dashboard_returns_figure(
        self, generator, valuation_data, roe_df, industry_data
    ):
        """generate_combined_dashboard() 返回 plotly Figure 对象。"""
        fig = generator.generate_combined_dashboard(
            valuation_data, roe_df, industry_data
        )
        assert isinstance(fig, go.Figure)
        # 应至少有 1 个 trace（valuation indicator + roe traces + radar traces）
        assert len(fig.data) >= 3

    def test_combined_dashboard_with_single_component(self, generator, roe_df):
        """仅 ROE 数据时应返回有效 figure（单组件回退路径）。"""
        fig = generator.generate_combined_dashboard({}, roe_df, {})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    # ── 空 / None 数据 ───────────────────────────────────────

    def test_roe_trend_empty_dataframe(self, generator):
        """空 DataFrame 应返回空 figure。"""
        fig = generator.generate_roe_trend(pd.DataFrame())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_roe_trend_none_dataframe(self, generator):
        """None DataFrame 应返回空 figure。"""
        fig = generator.generate_roe_trend(None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_industry_radar_empty_dicts(self, generator):
        """空字典应返回空 figure。"""
        fig = generator.generate_industry_radar({}, {})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_industry_radar_none_inputs(self, generator):
        """None 输入应返回空 figure。"""
        fig = generator.generate_industry_radar(None, None)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_combined_dashboard_all_empty(self, generator):
        """所有数据为空/None 应返回空 figure。"""
        fig = generator.generate_combined_dashboard({}, None, {})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_industry_radar_no_valid_metrics(self, generator):
        """所有指标值为 None 时应返回空 figure。"""
        import numpy as np
        stock = {"PE": np.nan, "PB": None}
        industry = {"PE": np.nan, "PB": None}
        fig = generator.generate_industry_radar(stock, industry)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
