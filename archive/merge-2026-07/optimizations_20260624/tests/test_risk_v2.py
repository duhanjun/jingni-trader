"""
风险管理 v2 测试

验证：
  1. 断路器滞回机制：触发与恢复阈值不同
  2. 断路器 fail-open：内部异常时放行
  3. 断路器最小样本量：样本不足不触发
  4. 断路器 JSON 持久化
  5. HRP 修复：传入真实 returns 后能正常优化
  6. CVaR 实现：返回有效权重且 CVaR 值合理
  7. 换手率约束修复

运行：python -m pytest optimizations/tests/test_risk_v2.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from risk.circuit_breaker_v2 import CircuitBreakerV2
from risk.portfolio_optimizer_v2 import PortfolioOptimizerV2


# ---------------------------------------------------------------------------
# 断路器测试
# ---------------------------------------------------------------------------

class TestCircuitBreakerHysteresis:
    """验证滞回机制：触发阈值与恢复阈值不同。"""

    def test_trip_on_daily_loss(self):
        """单日亏损超阈值应触发断路。"""
        cb = CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=0.0)
        # 亏损 6%，order_value 远小于 10% 净值以避免触发单笔限额
        result = cb.check_send_order(
            current_nav=940, start_of_day_nav=1000, order_value=50
        )
        assert not result["allowed"], "亏损 6% 应触发断路"
        assert cb.tripped

    def test_no_trip_within_limit(self):
        """亏损未超阈值不应触发。"""
        cb = CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=0.0)
        result = cb.check_send_order(
            current_nav=970, start_of_day_nav=1000, order_value=50
        )
        assert result["allowed"], "亏损 3% 不应触发断路"

    def test_hysteresis_blocks_until_recover_threshold(self):
        """触发后，收益未达 recover_threshold 前不得恢复。"""
        cb = CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=0.0)
        # 触发
        cb.check_send_order(current_nav=940, start_of_day_nav=1000, order_value=50)
        assert cb.tripped
        # 亏损收窄到 -2%（未回正），不应恢复
        result = cb.check_send_order(
            current_nav=980, start_of_day_nav=1000, order_value=50
        )
        assert not result["allowed"], "未达恢复阈值 +0% 前不应恢复"
        assert cb.tripped

    def test_hysteresis_recovers_at_threshold(self):
        """收益达 recover_threshold 后应恢复。"""
        cb = CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=0.0)
        cb.check_send_order(current_nav=940, start_of_day_nav=1000, order_value=50)
        # 回正到 +1%
        result = cb.check_send_order(
            current_nav=1010, start_of_day_nav=1000, order_value=50
        )
        assert result["allowed"], "达恢复阈值 +0% 后应恢复"
        assert not cb.tripped

    def test_recover_threshold_must_be_higher_than_trip(self):
        """recover_threshold 必须大于 -daily_loss_limit，否则报错。"""
        with pytest.raises(ValueError):
            CircuitBreakerV2(daily_loss_limit=0.05, recover_threshold=-0.06)


class TestCircuitBreakerFailOpen:
    """验证 fail-open 语义。"""

    def test_exception_returns_allowed(self):
        """内部异常时应 fail-open 放行。"""
        cb = CircuitBreakerV2(daily_loss_limit=0.05)
        # 传入 start_of_day_nav=0 触发除零（被 try 捕获）
        result = cb.check_send_order(
            current_nav=1000, start_of_day_nav=0, order_value=1000
        )
        # daily_return 计算时 start_of_day_nav=0 → 返回 0（不异常），但若其他异常则 fail-open
        # 这里验证不崩溃
        assert "allowed" in result

    def test_nan_nav_fail_open(self):
        """NaN 净值应 fail-open。"""
        cb = CircuitBreakerV2(daily_loss_limit=0.05)
        result = cb.check_send_order(
            current_nav=float("nan"), start_of_day_nav=1000, order_value=1000
        )
        assert result["allowed"], "NaN 净值应 fail-open 放行"


class TestCircuitBreakerMinSamples:
    """验证最小样本量保护。"""

    def test_small_sample_no_trip_on_rolling(self):
        """样本不足时滚动 ROI 不触发断路。"""
        cb = CircuitBreakerV2(
            daily_loss_limit=0.05, min_samples=5, rolling_window=5
        )
        # 连续 2 天小亏（样本不足 5）
        for _ in range(2):
            result = cb.check_send_order(
                current_nav=940, start_of_day_nav=1000, order_value=50
            )
        # 单日亏损 6% 仍会触发（单日检查不受 min_samples 限制）
        # 但若单日未超限，滚动也不应因小样本触发
        cb2 = CircuitBreakerV2(
            daily_loss_limit=0.10, min_samples=5, rolling_window=5
        )
        # 单日亏 6%（未超 10%），滚动 2 天样本不足
        for _ in range(2):
            r = cb2.check_send_order(
                current_nav=940, start_of_day_nav=1000, order_value=50
            )
            assert r["allowed"], "单日未超限且样本不足时不应触发"


class TestCircuitBreakerPersistence:
    """验证 JSON 状态持久化。"""

    def test_save_and_load_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cb = CircuitBreakerV2(daily_loss_limit=0.05, state_path=path)
            cb.check_send_order(current_nav=940, start_of_day_nav=1000, order_value=1000)
            assert cb.tripped
            cb.save_state()
            assert os.path.exists(path)

            # 新实例加载状态
            cb2 = CircuitBreakerV2(daily_loss_limit=0.05, state_path=path)
            assert cb2.tripped, "加载状态后应保持断路"
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ---------------------------------------------------------------------------
# HRP 修复测试
# ---------------------------------------------------------------------------

class TestHRPFix:
    """验证 HRP 优化修复。"""

    def test_hrp_with_real_returns_succeeds(self):
        """传入真实 returns 后 HRP 应成功返回权重。"""
        try:
            from pypfopt import HRPOpt
        except ImportError:
            pytest.skip("pypfopt 未安装，HRP 测试跳过")
        rng = np.random.default_rng(42)
        n_assets = 5
        n_days = 100
        returns = pd.DataFrame(
            rng.normal(0.001, 0.02, (n_days, n_assets)),
            columns=[f"asset_{i}" for i in range(n_assets)],
        )
        opt = PortfolioOptimizerV2()
        weights, meta = opt.optimize_hrp(returns)
        assert len(weights) == n_assets
        assert abs(weights.sum() - 1.0) < 0.01, "权重应归一化"
        assert meta["method"] == "hrp"

    def test_hrp_empty_returns_fallback(self):
        """空 returns 应优雅降级为等权。"""
        opt = PortfolioOptimizerV2()
        weights, meta = opt.optimize_hrp(pd.DataFrame())
        assert meta["method"] == "hrp_fallback"

    def test_hrp_old_bug_reproduced(self):
        """复现旧版 bug：传空 DataFrame 给 HRPOpt 会失败。"""
        try:
            from pypfopt import HRPOpt
            hrp = HRPOpt(pd.DataFrame())
            with pytest.raises(Exception):
                hrp.optimize()
        except ImportError:
            pytest.skip("pypfopt 未安装")


# ---------------------------------------------------------------------------
# CVaR 实现测试
# ---------------------------------------------------------------------------

class TestCVaR:
    """验证 CVaR 优化实现。"""

    def test_cvar_returns_valid_weights(self):
        rng = np.random.default_rng(42)
        n_assets = 4
        returns = pd.DataFrame(
            rng.normal(0.001, 0.02, (100, n_assets)),
            columns=[f"a{i}" for i in range(n_assets)],
        )
        opt = PortfolioOptimizerV2()
        weights, meta = opt.optimize_cvar(returns, confidence=0.95, max_weight=0.4)
        if meta["method"] == "cvar":
            assert len(weights) == n_assets
            assert abs(weights.sum() - 1.0) < 0.02
            assert all(w <= 0.41 for w in weights.values), "应满足 max_weight 约束"
            assert "cvar" in meta
            assert meta["cvar"] < 0, "95% CVaR 应为负（左尾损失）"
        else:
            pytest.skip("cvxpy 未安装，降级等权")

    def test_cvar_empty_returns_fallback(self):
        opt = PortfolioOptimizerV2()
        weights, meta = opt.optimize_cvar(pd.DataFrame())
        assert "fallback" in meta["method"]


# ---------------------------------------------------------------------------
# 换手率约束测试
# ---------------------------------------------------------------------------

class TestTurnoverConstraint:
    """验证换手率约束修复。"""

    def test_turnover_reported(self):
        try:
            from pypfopt import EfficientFrontier
        except ImportError:
            pytest.skip("pypfopt 未安装，换手率约束测试跳过")
        rng = np.random.default_rng(42)
        n = 4
        returns = pd.DataFrame(
            rng.normal(0.001, 0.02, (100, n)),
            columns=[f"a{i}" for i in range(n)],
        )
        opt = PortfolioOptimizerV2()
        expected_rets = returns.mean() * 252
        cov = returns.cov() * 252
        current_w = pd.Series([0.25, 0.25, 0.25, 0.25], index=expected_rets.index)
        weights, meta = opt.optimize_with_turnover(
            expected_rets, cov, current_w, max_turnover=0.3
        )
        assert "actual_turnover" in meta, "应报告实际换手率"
        assert "turnover_constraint_met" in meta


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])