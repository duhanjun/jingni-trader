"""backtest-engine L2 单元测试：ExtendedMetrics。

覆盖：
- calc_profit_factor 利润因子
- calc_payoff_ratio 平均盈亏比
- calc_max_consecutive_loss_days 最大连续亏损天数
- calc_downside_volatility 年化下行波动率
- calc_alpha_beta CAPM Alpha/Beta
- calc_information_ratio 信息比率
- calc_tracking_error 跟踪误差
- 边界：空 Series / 单元素 / 全正 / 全负
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
BACKTEST_ENGINE_DIR = os.path.join(ROOT, "skills", "backtest-engine")


def _load_extended_metrics():
    """加载 backtest-engine/scripts/optimizations/extended_metrics.py 为独立模块。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(BACKTEST_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    for _m in ("backtrader", "talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        target_path = os.path.join(BACKTEST_ENGINE_DIR, "scripts/optimizations/extended_metrics.py")
        spec = ilu.spec_from_file_location("_be_extended_metrics", target_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["_be_extended_metrics"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


def _make_returns(n=20, seed=42, positive_bias=False):
    """构造测试用收益序列。"""
    rng = np.random.RandomState(seed)
    if positive_bias:
        return pd.Series(rng.normal(0.001, 0.01, n))
    return pd.Series(rng.normal(0, 0.01, n))


@pytest.mark.skill_backtest_engine
@pytest.mark.unit
class TestCalcProfitFactor:
    """验证 calc_profit_factor 利润因子。"""

    def test_all_positive_returns(self):
        """全正收益 → 利润因子=inf"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, 0.02, 0.015])
        pf = metrics.ExtendedMetrics.calc_profit_factor(returns)
        assert pf == float("inf")

    def test_all_negative_returns(self):
        """全负收益 → 利润因子=0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([-0.01, -0.02])
        pf = metrics.ExtendedMetrics.calc_profit_factor(returns)
        assert pf == 0.0

    def test_mixed_returns(self):
        """正负混合 → 利润因子=总盈利/总亏损绝对值"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.03, -0.01, 0.02, -0.02])
        pf = metrics.ExtendedMetrics.calc_profit_factor(returns)
        # 总盈利 = 0.05, 总亏损 = 0.03
        assert abs(pf - 0.05 / 0.03) < 1e-6

    def test_empty_returns(self):
        """空 Series → 0"""
        metrics = _load_extended_metrics()
        assert metrics.ExtendedMetrics.calc_profit_factor(pd.Series([], dtype=float)) == 0.0


@pytest.mark.skill_backtest_engine
@pytest.mark.unit
class TestCalcPayoffRatio:
    """验证 calc_payoff_ratio 平均盈亏比。"""

    def test_mixed_returns(self):
        metrics = _load_extended_metrics()
        returns = pd.Series([0.04, -0.02, 0.06, -0.02])
        ratio = metrics.ExtendedMetrics.calc_payoff_ratio(returns)
        # avg_win = (0.04+0.06)/2 = 0.05, avg_loss = (-0.02 + -0.02)/2 = -0.02
        assert abs(ratio - 0.05 / 0.02) < 1e-6

    def test_all_positive_returns_zero(self):
        """全正（无亏损）→ 0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, 0.02])
        assert metrics.ExtendedMetrics.calc_payoff_ratio(returns) == 0.0

    def test_empty_returns_zero(self):
        metrics = _load_extended_metrics()
        assert metrics.ExtendedMetrics.calc_payoff_ratio(pd.Series([], dtype=float)) == 0.0


@pytest.mark.skill_backtest_engine
@pytest.mark.unit
class TestCalcMaxConsecutiveLossDays:
    """验证 calc_max_consecutive_loss_days。"""

    def test_no_losses(self):
        """全正收益 → 0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, 0.02, 0.03])
        assert metrics.ExtendedMetrics.calc_max_consecutive_loss_days(returns) == 0

    def test_continuous_losses(self):
        """连续亏损 3 天"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, -0.01, -0.02, -0.03, 0.01, -0.02])
        assert metrics.ExtendedMetrics.calc_max_consecutive_loss_days(returns) == 3

    def test_all_losses(self):
        """全负 → 长度"""
        metrics = _load_extended_metrics()
        returns = pd.Series([-0.01, -0.02, -0.03])
        assert metrics.ExtendedMetrics.calc_max_consecutive_loss_days(returns) == 3

    def test_empty(self):
        metrics = _load_extended_metrics()
        assert metrics.ExtendedMetrics.calc_max_consecutive_loss_days(pd.Series([], dtype=float)) == 0


@pytest.mark.skill_backtest_engine
@pytest.mark.unit
class TestCalcDownsideVolatility:
    """验证 calc_downside_volatility 年化下行波动率。"""

    def test_mixed_returns(self):
        """混合收益 → 仅用负收益计算年化波动率"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, -0.02, 0.015, -0.025, 0.005])
        dv = metrics.ExtendedMetrics.calc_downside_volatility(returns, trading_days=252)
        assert dv > 0
        # 仅 2 个负值，验证逻辑应使用 std * sqrt(252)
        neg = returns[returns < 0]
        expected = neg.std() * np.sqrt(252)
        assert abs(dv - expected) < 1e-6

    def test_all_positive_returns_zero(self):
        """全正（无负收益）→ 0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, 0.02, 0.03])
        assert metrics.ExtendedMetrics.calc_downside_volatility(returns) == 0.0

    def test_single_negative_returns_zero(self):
        """只有 1 个负值（<2 个）→ 0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, -0.02, 0.03])
        assert metrics.ExtendedMetrics.calc_downside_volatility(returns) == 0.0


@pytest.mark.skill_backtest_engine
@pytest.mark.unit
class TestCalcAlphaBeta:
    """验证 calc_alpha_beta CAPM Alpha/Beta。"""

    def test_returns_alpha_beta(self):
        """正常输入 → 返回 dict 含 alpha 和 beta"""
        metrics = _load_extended_metrics()
        returns = _make_returns(30)
        bench = _make_returns(30, seed=99)
        result = metrics.ExtendedMetrics.calc_alpha_beta(returns, bench)
        assert "alpha" in result
        assert "beta" in result
        assert isinstance(result["alpha"], float)
        assert isinstance(result["beta"], float)

    def test_too_short_returns_zeros(self):
        """少于 2 个样本 → alpha=0, beta=0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01])
        bench = pd.Series([0.02])
        result = metrics.ExtendedMetrics.calc_alpha_beta(returns, bench)
        assert result == {"alpha": 0.0, "beta": 0.0}

    def test_zero_variance_benchmark(self):
        """基准 0 方差 → alpha=0, beta=0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, 0.02, 0.015])
        bench = pd.Series([0.0, 0.0, 0.0])
        result = metrics.ExtendedMetrics.calc_alpha_beta(returns, bench)
        assert result == {"alpha": 0.0, "beta": 0.0}


@pytest.mark.skill_backtest_engine
@pytest.mark.unit
class TestCalcInformationRatioAndTrackingError:
    """验证 calc_information_ratio 与 calc_tracking_error。"""

    def test_information_ratio_positive_when_outperform(self):
        """策略持续跑赢基准 → IR > 0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.02, 0.025, 0.018, 0.022, 0.019])
        bench = pd.Series([0.01, 0.012, 0.008, 0.011, 0.010])
        ir = metrics.ExtendedMetrics.calc_information_ratio(returns, bench)
        assert ir > 0

    def test_information_ratio_zero_when_too_short(self):
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01])
        bench = pd.Series([0.02])
        assert metrics.ExtendedMetrics.calc_information_ratio(returns, bench) == 0.0

    def test_tracking_error_positive(self):
        """跟踪误差 > 0"""
        metrics = _load_extended_metrics()
        returns = _make_returns(20)
        bench = _make_returns(20, seed=99)
        te = metrics.ExtendedMetrics.calc_tracking_error(returns, bench)
        assert te > 0

    def test_tracking_error_zero_when_identical(self):
        """策略与基准相同 → 跟踪误差=0"""
        metrics = _load_extended_metrics()
        returns = pd.Series([0.01, 0.02, 0.015])
        bench = returns.copy()
        # 完全相同 → excess std = 0 → te = 0
        te = metrics.ExtendedMetrics.calc_tracking_error(returns, bench)
        assert te == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
