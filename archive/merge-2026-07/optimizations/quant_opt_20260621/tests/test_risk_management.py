"""
可组合风险管理模型测试

验证内容:
1. 单个模型正确性: 各风控模型在触发条件下正确调整持仓
2. 链式组合: 多个模型依次过滤, 行为符合预期
3. 边界条件: 空持仓、全清仓、冷却期、连续亏损
4. 与现有 RiskManager 对比: 功能覆盖度
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

_OPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _OPT_DIR not in sys.path:
    sys.path.insert(0, _OPT_DIR)

from risk_management_models import (
    PortfolioTarget, PortfolioState, RiskCheckResult,
    IRiskManagementModel, MaximumDrawdownRiskModel, TrailingStopRiskModel,
    MaxPositionRiskModel, PortfolioHeatRiskModel, CircuitBreakerRiskModel,
    VolatilityScalingModel, RiskManagerChain, default_risk_chain,
    build_portfolio_state,
)


def make_target(code, weight, pnl_pct=0.0, entry=10.0, current=10.0, holding=5):
    """构造测试用 PortfolioTarget"""
    return PortfolioTarget(
        code=code, target_weight=weight, current_weight=weight,
        entry_price=entry, current_price=current,
        unrealized_pnl_pct=pnl_pct, holding_days=holding,
    )


def make_state(drawdown=0.0, consecutive_losses=0, nav=1e6, peak=1e6,
               cash=1e5, daily_pnl=0.0, total_risk=0.03):
    """构造测试用 PortfolioState"""
    return PortfolioState(
        nav=nav, peak_nav=peak, cash=cash,
        drawdown=drawdown, daily_pnl=daily_pnl,
        consecutive_losses=consecutive_losses,
        total_risk_pct=total_risk, targets=[],
    )


class TestMaximumDrawdownRiskModel(unittest.TestCase):
    """最大回撤熔断模型测试"""

    def test_no_trigger_below_threshold(self):
        """回撤未达阈值, 不触发"""
        model = MaximumDrawdownRiskModel(max_drawdown=0.10)
        state = make_state(drawdown=-0.05)
        targets = [make_target("A", 0.5), make_target("B", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertFalse(result.triggered)
        self.assertEqual(len(result.adjusted_targets), 2)

    def test_trigger_at_threshold(self):
        """回撤达到阈值, 触发清仓"""
        model = MaximumDrawdownRiskModel(max_drawdown=0.10)
        state = make_state(drawdown=-0.10)
        targets = [make_target("A", 0.5), make_target("B", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertTrue(result.triggered)
        # 所有持仓应清零
        for t in result.adjusted_targets:
            self.assertEqual(t.target_weight, 0.0)

    def test_cooldown_period(self):
        """冷却期内保持空仓"""
        model = MaximumDrawdownRiskModel(max_drawdown=0.10, cooldown_days=3)
        state = make_state(drawdown=-0.15)
        targets = [make_target("A", 0.5), make_target("B", 0.5)]

        # 第一次触发
        r1 = model.manage_risk(state, targets)
        self.assertTrue(r1.triggered)

        # 冷却期第 1 天 (回撤已恢复)
        state2 = make_state(drawdown=-0.01)
        r2 = model.manage_risk(state2, [make_target("A", 0.5)])
        self.assertTrue(r2.triggered, "冷却期内应保持触发状态")
        for t in r2.adjusted_targets:
            self.assertEqual(t.target_weight, 0.0)

        # 冷却期第 2 天
        r3 = model.manage_risk(state2, [make_target("A", 0.5)])
        self.assertTrue(r3.triggered)

        # 冷却期第 3 天 (最后一天)
        r4 = model.manage_risk(state2, [make_target("A", 0.5)])
        self.assertTrue(r4.triggered)

        # 冷却期结束
        r5 = model.manage_risk(state2, [make_target("A", 0.5)])
        self.assertFalse(r5.triggered, "冷却期结束后应恢复正常")


class TestTrailingStopRiskModel(unittest.TestCase):
    """追踪止损模型测试"""

    def test_no_trigger_within_limit(self):
        """浮亏未达止损线, 不触发"""
        model = TrailingStopRiskModel(max_loss_pct=0.08)
        state = make_state()
        targets = [make_target("A", 0.5, pnl_pct=-0.03)]
        result = model.manage_risk(state, targets)
        self.assertFalse(result.triggered)

    def test_trigger_at_loss_threshold(self):
        """浮亏达到止损线, 触发清仓"""
        model = TrailingStopRiskModel(max_loss_pct=0.08)
        state = make_state()
        targets = [
            make_target("A", 0.5, pnl_pct=-0.10),  # 触发
            make_target("B", 0.5, pnl_pct=-0.03),  # 不触发
        ]
        result = model.manage_risk(state, targets)
        self.assertTrue(result.triggered)
        # A 应清仓, B 保持
        weights = {t.code: t.target_weight for t in result.adjusted_targets}
        self.assertEqual(weights["A"], 0.0)
        self.assertEqual(weights["B"], 0.5)

    def test_zero_weight_not_checked(self):
        """目标权重为 0 的不检查"""
        model = TrailingStopRiskModel(max_loss_pct=0.08)
        state = make_state()
        targets = [make_target("A", 0.0, pnl_pct=-0.50)]
        result = model.manage_risk(state, targets)
        self.assertFalse(result.triggered)


class TestMaxPositionRiskModel(unittest.TestCase):
    """单一持仓上限模型测试"""

    def test_no_trigger_within_limit(self):
        """权重在上限内, 不触发"""
        model = MaxPositionRiskModel(max_position_pct=0.05)
        state = make_state()
        targets = [make_target("A", 0.03), make_target("B", 0.04)]
        result = model.manage_risk(state, targets)
        self.assertFalse(result.triggered)

    def test_cap_excess_weight(self):
        """超限权重被截断"""
        model = MaxPositionRiskModel(max_position_pct=0.05)
        state = make_state()
        targets = [
            make_target("A", 0.10),  # 超限
            make_target("B", 0.03),  # 未超限
        ]
        result = model.manage_risk(state, targets)
        self.assertTrue(result.triggered)
        weights = {t.code: t.target_weight for t in result.adjusted_targets}
        self.assertEqual(weights["A"], 0.05)
        self.assertEqual(weights["B"], 0.03)


class TestPortfolioHeatRiskModel(unittest.TestCase):
    """组合总风险敞口模型测试"""

    def test_no_trigger_within_limit(self):
        """总风险在上限内, 不触发"""
        vols = {"A": 0.02, "B": 0.02}
        model = PortfolioHeatRiskModel(max_total_risk=0.06, volatilities=vols)
        state = make_state()
        # 总风险 = 0.5*0.02 + 0.5*0.02 = 0.02 < 0.06
        targets = [make_target("A", 0.5), make_target("B", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertFalse(result.triggered)

    def test_scale_down_when_exceeded(self):
        """总风险超限时等比缩减"""
        vols = {"A": 0.10, "B": 0.10}
        model = PortfolioHeatRiskModel(max_total_risk=0.06, volatilities=vols)
        state = make_state()
        # 总风险 = 0.5*0.10 + 0.5*0.10 = 0.10 > 0.06
        # 缩放 = 0.06/0.10 = 0.6
        targets = [make_target("A", 0.5), make_target("B", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertTrue(result.triggered)
        for t in result.adjusted_targets:
            self.assertAlmostEqual(t.target_weight, 0.5 * 0.6, places=6)


class TestCircuitBreakerRiskModel(unittest.TestCase):
    """连续亏损熔断模型测试"""

    def test_no_trigger_below_threshold(self):
        """连续亏损未达阈值, 不触发"""
        model = CircuitBreakerRiskModel(max_consecutive_losses=3)
        state = make_state(consecutive_losses=2)
        targets = [make_target("A", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertFalse(result.triggered)

    def test_trigger_at_threshold(self):
        """连续亏损达到阈值, 触发熔断"""
        model = CircuitBreakerRiskModel(max_consecutive_losses=3, cooldown_days=2)
        state = make_state(consecutive_losses=3)
        targets = [make_target("A", 0.5), make_target("B", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertTrue(result.triggered)
        for t in result.adjusted_targets:
            self.assertEqual(t.target_weight, 0.0)

    def test_cooldown_recovery(self):
        """冷却期后恢复"""
        model = CircuitBreakerRiskModel(max_consecutive_losses=2, cooldown_days=2)
        state = make_state(consecutive_losses=2)
        targets = [make_target("A", 0.5)]

        # 触发
        r1 = model.manage_risk(state, targets)
        self.assertTrue(r1.triggered)

        # 冷却第 1 天 (亏损已恢复)
        state2 = make_state(consecutive_losses=0)
        r2 = model.manage_risk(state2, targets)
        self.assertTrue(r2.triggered)

        # 冷却第 2 天
        r3 = model.manage_risk(state2, targets)
        self.assertTrue(r3.triggered)

        # 冷却结束
        r4 = model.manage_risk(state2, targets)
        self.assertFalse(r4.triggered)


class TestVolatilityScalingModel(unittest.TestCase):
    """波动率目标模型测试"""

    def test_scale_down_high_vol(self):
        """高波动时缩减"""
        model = VolatilityScalingModel(target_volatility=0.10, current_volatility=0.20)
        state = make_state()
        targets = [make_target("A", 0.5)]
        result = model.manage_risk(state, targets)
        self.assertTrue(result.triggered)
        # 缩放 = 0.10/0.20 = 0.5
        self.assertAlmostEqual(result.adjusted_targets[0].target_weight, 0.25, places=6)

    def test_scale_up_low_vol_capped(self):
        """低波动时加杠杆但有上限"""
        model = VolatilityScalingModel(target_volatility=0.20, current_volatility=0.10, max_leverage=1.5)
        state = make_state()
        targets = [make_target("A", 0.5)]
        result = model.manage_risk(state, targets)
        # 缩放 = 0.20/0.10 = 2.0, 但上限 1.5
        self.assertAlmostEqual(result.adjusted_targets[0].target_weight, 0.5 * 1.5, places=6)


class TestRiskManagerChain(unittest.TestCase):
    """风控模型链测试"""

    def test_chain_no_trigger(self):
        """所有模型都不触发"""
        chain = RiskManagerChain([
            MaximumDrawdownRiskModel(max_drawdown=0.10),
            TrailingStopRiskModel(max_loss_pct=0.08),
            MaxPositionRiskModel(max_position_pct=0.05),
        ])
        state = make_state(drawdown=-0.03)
        targets = [make_target("A", 0.03, pnl_pct=-0.02), make_target("B", 0.04, pnl_pct=0.01)]
        result = chain.manage(state, targets)
        self.assertFalse(result["any_triggered"])
        self.assertEqual(len(result["final_targets"]), 2)

    def test_chain_multiple_triggers(self):
        """多个模型依次触发"""
        chain = RiskManagerChain([
            MaxPositionRiskModel(max_position_pct=0.05),  # 先截断超限
            TrailingStopRiskModel(max_loss_pct=0.08),     # 再止损
        ])
        state = make_state()
        targets = [
            make_target("A", 0.10, pnl_pct=-0.15),  # 超限 + 止损
            make_target("B", 0.03, pnl_pct=0.01),   # 正常
        ]
        result = chain.manage(state, targets)
        self.assertTrue(result["any_triggered"])
        self.assertEqual(len(result["triggered_models"]), 2)
        # A 应被 MaxPosition 截断为 0.05, 再被 TrailingStop 清零
        weights = {t.code: t.target_weight for t in result["final_targets"]}
        self.assertEqual(weights["A"], 0.0)
        self.assertEqual(weights["B"], 0.03)

    def test_chain_drawdown_clears_all(self):
        """回撤熔断清空所有持仓, 后续模型基于空持仓"""
        chain = RiskManagerChain([
            MaximumDrawdownRiskModel(max_drawdown=0.10),
            MaxPositionRiskModel(max_position_pct=0.05),
        ])
        state = make_state(drawdown=-0.15)
        targets = [make_target("A", 0.10), make_target("B", 0.20)]
        result = chain.manage(state, targets)
        self.assertTrue(result["any_triggered"])
        self.assertTrue(result["all_cleared"])
        for t in result["final_targets"]:
            self.assertEqual(t.target_weight, 0.0)

    def test_default_chain(self):
        """默认风控链构建"""
        chain = default_risk_chain()
        self.assertEqual(len(chain.models), 5)
        self.assertIsInstance(chain.models[0], MaximumDrawdownRiskModel)
        self.assertIsInstance(chain.models[1], CircuitBreakerRiskModel)
        self.assertIsInstance(chain.models[2], TrailingStopRiskModel)
        self.assertIsInstance(chain.models[3], MaxPositionRiskModel)
        self.assertIsInstance(chain.models[4], PortfolioHeatRiskModel)


class TestBuildPortfolioState(unittest.TestCase):
    """测试从 DataFrame 构建状态"""

    def test_build_from_series(self):
        """从 Series 构建状态与目标"""
        weights = pd.Series({"A": 0.4, "B": 0.6})
        prices = pd.Series({"A": 11.0, "B": 9.0})
        entry_prices = pd.Series({"A": 10.0, "B": 10.0})
        holding_days = pd.Series({"A": 5, "B": 3})

        state, targets = build_portfolio_state(
            nav=1.1e6, peak_nav=1.2e6, cash=1e5,
            daily_pnl=-5000, consecutive_losses=2,
            weights=weights, prices=prices,
            entry_prices=entry_prices, holding_days=holding_days,
        )

        self.assertEqual(len(targets), 2)
        self.assertAlmostEqual(state.drawdown, (1.1e6 - 1.2e6) / 1.2e6)
        # A 的浮盈 = (11-10)/10 = 0.10
        self.assertAlmostEqual(targets[0].unrealized_pnl_pct, 0.10)
        # B 的浮亏 = (9-10)/10 = -0.10
        self.assertAlmostEqual(targets[1].unrealized_pnl_pct, -0.10)


class TestCoverageVsExistingRiskManager(unittest.TestCase):
    """与现有 portfolio-risk-engine.RiskManager 的功能覆盖度对比"""

    def test_existing_features_covered(self):
        """现有 RiskManager 的功能在新模型中都有对应"""
        # 现有: check_portfolio_stop (单日亏损) -> MaximumDrawdownRiskModel (回撤) + CircuitBreakerRiskModel (连续亏损)
        # 现有: check_individual_stop (个股止损) -> TrailingStopRiskModel
        # 现有: calc_var / calc_cvar -> 不在风控模型范围 (属于风险度量, 非风控动作)
        # 现有: 无单一持仓上限 -> MaxPositionRiskModel (新增能力)
        # 现有: 无组合总风险 -> PortfolioHeatRiskModel (新增能力)
        # 现有: 无波动率目标 -> VolatilityScalingModel (新增能力)

        chain = default_risk_chain()
        model_types = [type(m).__name__ for m in chain.models]

        # 现有功能对应
        self.assertIn("MaximumDrawdownRiskModel", model_types)
        self.assertIn("TrailingStopRiskModel", model_types)
        self.assertIn("CircuitBreakerRiskModel", model_types)

        # 新增能力
        self.assertIn("MaxPositionRiskModel", model_types)
        self.assertIn("PortfolioHeatRiskModel", model_types)

    def test_composability_advantage(self):
        """可组合性优势: 可任意增删模型"""
        # 只用单一模型
        chain1 = RiskManagerChain([MaxPositionRiskModel(0.05)])
        state = make_state()
        targets = [make_target("A", 0.10)]
        r1 = chain1.manage(state, targets)
        self.assertTrue(r1["any_triggered"])

        # 增加模型
        chain2 = RiskManagerChain([
            MaxPositionRiskModel(0.05),
            TrailingStopRiskModel(0.08),
        ])
        r2 = chain2.manage(state, targets)
        self.assertEqual(len(chain2.models), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
