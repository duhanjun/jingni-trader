"""reports-engine L2 单元测试：performance_metrics。

覆盖：
- total_return 总收益率
- annualized_return 年化收益率
- annualized_vol 年化波动率
- sharpe_ratio Sharpe 比率
- sortino_ratio Sortino 比率
- max_drawdown 最大回撤
- calmar_ratio Calmar 比率
- alpha_beta CAPM Alpha/Beta
- information_ratio 信息比率
- 边界：空 Series / 单元素 / 全零
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_ENGINE_DIR = os.path.join(ROOT, "skills", "reports-engine")


def _load_performance_metrics():
    """加载 reports-engine/scripts/optimizations/performance_metrics.py。"""
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
        target_path = os.path.join(REPORTS_ENGINE_DIR, "scripts/optimizations/performance_metrics.py")
        spec = ilu.spec_from_file_location("_re_performance_metrics", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_re_performance_metrics"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _make_equity(n=20, seed=42, growth=0.001):
    """构造净值序列（每天增长 growth）"""
    rng = np.random.RandomState(seed)
    returns = rng.normal(growth, 0.01, n)
    equity = pd.Series(100.0 * np.cumprod(1 + returns))
    return equity


def _make_returns(n=20, seed=42, growth=0.001):
    """构造日收益序列"""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(growth, 0.01, n))


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestTotalReturn:
    def test_positive_growth(self):
        mod = _load_performance_metrics()
        equity = pd.Series([100.0, 110.0, 105.0, 120.0])
        tr = mod.total_return(equity)
        assert abs(tr - 0.2) < 1e-6  # 120/100 - 1 = 0.2

    def test_empty_series(self):
        mod = _load_performance_metrics()
        assert mod.total_return(pd.Series([], dtype=float)) == 0.0

    def test_single_element(self):
        mod = _load_performance_metrics()
        assert mod.total_return(pd.Series([100.0])) == 0.0

    def test_zero_first_element(self):
        mod = _load_performance_metrics()
        assert mod.total_return(pd.Series([0.0, 100.0])) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestAnnualizedReturn:
    def test_positive_growth(self):
        mod = _load_performance_metrics()
        # 252 天涨 20% → 年化约 20%
        equity = pd.Series([100.0 * (1 + 0.2 * i / 252) for i in range(253)])
        ar = mod.annualized_return(equity)
        assert ar > 0

    def test_too_short_returns_zero(self):
        mod = _load_performance_metrics()
        assert mod.annualized_return(pd.Series([100.0])) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestAnnualizedVol:
    def test_positive_vol(self):
        mod = _load_performance_metrics()
        returns = _make_returns(30)
        av = mod.annualized_vol(returns)
        assert av > 0

    def test_empty_returns_zero(self):
        mod = _load_performance_metrics()
        assert mod.annualized_vol(pd.Series([], dtype=float)) == 0.0

    def test_single_returns_zero(self):
        mod = _load_performance_metrics()
        assert mod.annualized_vol(pd.Series([0.01])) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestSharpeRatio:
    def test_positive_returns(self):
        mod = _load_performance_metrics()
        returns = pd.Series([0.001] * 30)
        sr = mod.sharpe_ratio(returns)
        # 正收益 → Sharpe > 0（受 risk_free 影响）
        assert isinstance(sr, float)

    def test_zero_vol_returns_zero(self):
        """全零收益 → vol=0 → 0"""
        mod = _load_performance_metrics()
        returns = pd.Series([0.0] * 30)
        assert mod.sharpe_ratio(returns) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestSortinoRatio:
    def test_positive_returns(self):
        mod = _load_performance_metrics()
        returns = _make_returns(30, growth=0.002)
        sr = mod.sortino_ratio(returns)
        assert isinstance(sr, float)

    def test_all_positive_returns(self):
        """全正收益 → downside 为空 → 0"""
        mod = _load_performance_metrics()
        returns = pd.Series([0.01, 0.02, 0.015])
        assert mod.sortino_ratio(returns) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestMaxDrawdown:
    def test_known_drawdown(self):
        """100 → 120 → 90 → 110 → 最大回撤 = (90/120 - 1) = -0.25"""
        mod = _load_performance_metrics()
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        mdd = mod.max_drawdown(equity)
        assert abs(mdd - (-0.25)) < 1e-6

    def test_monotonic_increase(self):
        """单调递增 → 0 回撤"""
        mod = _load_performance_metrics()
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        assert mod.max_drawdown(equity) == 0.0

    def test_too_short(self):
        mod = _load_performance_metrics()
        assert mod.max_drawdown(pd.Series([100.0])) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestCalmarRatio:
    def test_positive_when_growth_and_drawdown(self):
        mod = _load_performance_metrics()
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        cr = mod.calmar_ratio(equity)
        # 年化收益 / |max_drawdown|
        assert isinstance(cr, float)

    def test_zero_drawdown_returns_zero(self):
        """无回撤 → 0"""
        mod = _load_performance_metrics()
        equity = pd.Series([100.0, 110.0, 120.0])
        assert mod.calmar_ratio(equity) == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestAlphaBeta:
    def test_returns_tuple(self):
        mod = _load_performance_metrics()
        returns = _make_returns(30)
        bench = _make_returns(30, seed=99)
        alpha, beta = mod.alpha_beta(returns, bench)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)

    def test_too_short_returns_zeros(self):
        """少于 20 样本 → 0, 0"""
        mod = _load_performance_metrics()
        returns = pd.Series([0.01, 0.02])
        bench = pd.Series([0.02, 0.01])
        alpha, beta = mod.alpha_beta(returns, bench)
        assert alpha == 0.0
        assert beta == 0.0


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestInformationRatio:
    def test_positive_when_outperform(self):
        mod = _load_performance_metrics()
        returns = pd.Series([0.02, 0.025, 0.022, 0.021, 0.023])
        bench = pd.Series([0.01, 0.012, 0.011, 0.013, 0.010])
        ir = mod.information_ratio(returns, bench)
        assert ir > 0

    def test_too_short_returns_zero(self):
        mod = _load_performance_metrics()
        returns = pd.Series([0.01])
        bench = pd.Series([0.02])
        assert mod.information_ratio(returns, bench) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
