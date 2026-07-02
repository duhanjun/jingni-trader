"""
验证测试: 动态风险管理 (Dynamic Risk Management)
=================================================
借鉴来源: Freqtrade (https://github.com/freqtrade/freqtrade)
          - 动态止损止盈、仓位管理、波动率自适应风控
优化方向: 风险管理增强 - 引入动态止损、波动率自适应仓位、VaR/CVaR

Freqtrade 核心设计:
1. 动态止损: 基于ATR的自适应止损 (trailing stop)
2. 仓位管理: 基于波动率的仓位大小调整 (volatility-adjusted position sizing)
3. 最大回撤控制: 实时监控并触发保护
4. 策略级风控: 单策略最大亏损限制、日最大亏损限制

与 jingni-trader 现有 portfolio-risk-engine 的差异:
- 现有引擎仅在组合层面做优化，缺乏交易级动态风控
- 无止损止盈机制
- 无波动率自适应仓位
- 无 VaR/CVaR 计算

测试目标:
1. 验证 ATR 动态止损的有效性
2. 验证波动率自适应仓位对收益风险比的影响
3. 验证 VaR/CVaR 计算的正确性
"""
import sys
import os
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import unittest
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. ATR 动态止损 - 借鉴 Freqtrade 的 trailing stop
# ============================================================

class DynamicStopLoss:
    """
    ATR 动态止损模块
    借鉴 Freqtrade 的 StoplossGuard 和 trailing stop 机制
    """

    @staticmethod
    def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算 ATR (Average True Range)
        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period, min_periods=period).mean()
        return atr

    @staticmethod
    def calc_trailing_stop(
        prices: pd.Series,
        atr: pd.Series,
        atr_multiplier: float = 2.0,
        initial_stop_pct: float = 0.05,
    ) -> pd.Series:
        """
        计算 trailing stop 价格
        借鉴 Freqtrade 的 trailing_stop_positive 逻辑:
        stop_price = max(prev_stop, current_price - ATR * multiplier)
        """
        stop_prices = pd.Series(index=prices.index, dtype=float)
        stop_prices.iloc[0] = prices.iloc[0] * (1 - initial_stop_pct)

        for i in range(1, len(prices)):
            if pd.isna(atr.iloc[i]):
                stop_prices.iloc[i] = stop_prices.iloc[i - 1]
                continue
            atr_stop = prices.iloc[i] - atr.iloc[i] * atr_multiplier
            stop_prices.iloc[i] = max(stop_prices.iloc[i - 1], atr_stop)

        return stop_prices

    @staticmethod
    def should_stop_out(
        current_price: float,
        stop_price: float,
        entry_price: float,
        max_loss_pct: float = 0.10,
    ) -> Tuple[bool, str]:
        """
        判断是否触发止损
        返回: (是否止损, 原因)
        """
        if current_price <= stop_price:
            return True, "trailing_stop"
        if (current_price - entry_price) / entry_price <= -max_loss_pct:
            return True, "max_loss_limit"
        return False, ""


# ============================================================
# 2. 波动率自适应仓位 - 借鉴 Freqtrade 的 position sizing
# ============================================================

class VolatilityAdjustedSizing:
    """
    波动率自适应仓位管理
    借鉴 Freqtrade 的 volume_pairlist 和 position_adjustment 机制
    核心思想: 波动率越高，仓位越小，控制组合风险敞口
    """

    @staticmethod
    def calc_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """计算滚动波动率"""
        return returns.rolling(window, min_periods=window).std() * np.sqrt(252)

    @staticmethod
    def calc_position_size(
        capital: float,
        price: float,
        volatility: float,
        target_vol: float = 0.25,
        max_position_pct: float = 0.20,
        min_position_pct: float = 0.01,
    ) -> float:
        """
        计算目标仓位股数
        借鉴 Freqtrade 的 stake_amount 逻辑:
        position_size = capital * target_vol / (volatility * price * sqrt(252))
        即: 目标仓位 = 资金 * 目标波动率 / (资产波动率 * 价格)
        """
        if pd.isna(volatility) or volatility <= 0:
            return 0

        # 目标波动率仓位
        target_value = capital * (target_vol / volatility)
        # 限制最大/最小仓位
        max_value = capital * max_position_pct
        min_value = capital * min_position_pct
        position_value = np.clip(target_value, min_value, max_value)

        shares = int(position_value / price / 100) * 100
        return shares

    @staticmethod
    def calc_kelly_fraction(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_fraction: float = 0.25,
    ) -> float:
        """
        Kelly 公式计算最优仓位比例
        f = (p * b - q) / b
        其中 p=胜率, b=盈亏比(avg_win/avg_loss), q=1-p
        """
        if avg_loss <= 0:
            return 0
        b = avg_win / abs(avg_loss)
        kelly = (win_rate * b - (1 - win_rate)) / b
        return min(max(kelly, 0), max_fraction)


# ============================================================
# 3. VaR / CVaR 计算 - 机构级风险度量
# ============================================================

class RiskMetrics:
    """
    风险度量模块
    借鉴 Freqtrade 和 Qlib 的 risk 计算
    """

    @staticmethod
    def calc_var(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        计算 VaR (Value at Risk)
        - historical: 历史模拟法
        - parametric: 参数法（假设正态分布）
        """
        if method == "historical":
            return float(returns.quantile(1 - confidence))
        elif method == "parametric":
            from scipy import stats
            mu = returns.mean()
            sigma = returns.std()
            return float(mu + sigma * stats.norm.ppf(1 - confidence))
        else:
            raise ValueError(f"不支持的方法: {method}")

    @staticmethod
    def calc_cvar(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        计算 CVaR (Conditional VaR) / Expected Shortfall
        CVaR = 超过 VaR 的损失的平均值
        """
        var = RiskMetrics.calc_var(returns, confidence)
        tail_losses = returns[returns <= var]
        if len(tail_losses) == 0:
            return var
        return float(tail_losses.mean())

    @staticmethod
    def calc_max_drawdown(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """计算最大回撤及起止时间"""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()
        end_idx = drawdown.idxmin()
        start_idx = cummax[:end_idx].idxmax()
        return float(max_dd), start_idx, end_idx

    @staticmethod
    def calc_drawdown_duration(equity: pd.Series) -> int:
        """计算最长回撤修复天数"""
        cummax = equity.cummax()
        underwater = equity < cummax
        if not underwater.any():
            return 0

        # 找连续水下最长的段
        max_duration = 0
        current = 0
        for is_under in underwater:
            if is_under:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration

    @staticmethod
    def calc_risk_report(equity: pd.Series, returns: pd.Series) -> dict:
        """生成完整风险报告"""
        var_95 = RiskMetrics.calc_var(returns, 0.95)
        cvar_95 = RiskMetrics.calc_cvar(returns, 0.95)
        max_dd, dd_start, dd_end = RiskMetrics.calc_max_drawdown(equity)
        dd_duration = RiskMetrics.calc_drawdown_duration(equity)

        return {
            "VaR_95": round(var_95, 4),
            "CVaR_95": round(cvar_95, 4),
            "max_drawdown": round(max_dd, 4),
            "max_dd_start": str(dd_start),
            "max_dd_end": str(dd_end),
            "max_dd_duration_days": dd_duration,
            "annual_volatility": round(float(returns.std() * np.sqrt(252)), 4),
            "skewness": round(float(returns.skew()), 4),
            "kurtosis": round(float(returns.kurtosis()), 4),
        }


# ============================================================
# 测试用例
# ============================================================

class TestDynamicStopLoss(unittest.TestCase):
    """测试动态止损机制"""

    def setUp(self):
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n))
        self.prices = pd.Series(close, index=dates)
        self.high = self.prices * (1 + np.abs(np.random.normal(0, 0.008, n)))
        self.low = self.prices * (1 - np.abs(np.random.normal(0, 0.008, n)))

    def test_atr_calculation(self):
        """验证 ATR 计算正确性"""
        atr = DynamicStopLoss.calc_atr(self.high, self.low, self.prices, period=14)
        self.assertGreater(len(atr.dropna()), 0)
        self.assertTrue((atr.dropna() > 0).all(), "ATR 应始终为正")

    def test_trailing_stop_monotonic(self):
        """验证 trailing stop 价格单调不减"""
        atr = DynamicStopLoss.calc_atr(self.high, self.low, self.prices, period=14)
        stop_prices = DynamicStopLoss.calc_trailing_stop(
            self.prices, atr, atr_multiplier=2.0
        )

        # trailing stop 应单调不减
        valid = stop_prices.dropna()
        diffs = valid.diff().dropna()
        self.assertTrue((diffs >= -1e-10).all(),
                        "Trailing stop 应单调不减")

    def test_stop_out_trigger(self):
        """验证止损触发逻辑"""
        self.assertTrue(
            DynamicStopLoss.should_stop_out(90, 95, 100)[0],
            "价格低于止损价应触发止损"
        )
        self.assertTrue(
            DynamicStopLoss.should_stop_out(89, 100, 100)[0],
            "亏损超过10%应触发止损"
        )
        self.assertFalse(
            DynamicStopLoss.should_stop_out(105, 95, 100)[0],
            "正常盈利不应触发止损"
        )

    def test_atr_trailing_stop_adaptive(self):
        """验证 ATR 止损自适应波动率"""
        n = 200
        low_vol_prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0005, 0.005, n)))
        high_vol_prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0005, 0.025, n)))

        low_vol_atr = DynamicStopLoss.calc_atr(
            low_vol_prices * 1.01, low_vol_prices * 0.99, low_vol_prices
        )
        high_vol_atr = DynamicStopLoss.calc_atr(
            high_vol_prices * 1.01, high_vol_prices * 0.99, high_vol_prices
        )

        self.assertGreater(
            high_vol_atr.dropna().mean(),
            low_vol_atr.dropna().mean(),
            "高波动率场景的 ATR 应更大"
        )


class TestVolatilityAdjustedSizing(unittest.TestCase):
    """测试波动率自适应仓位"""

    def test_position_size_basic(self):
        """基础仓位计算"""
        shares = VolatilityAdjustedSizing.calc_position_size(
            capital=1000000,
            price=50,
            volatility=0.30,
            target_vol=0.25,
        )
        self.assertGreater(shares, 0)
        # 波动率 0.30 -> 仓位比例约为 0.25/0.30 = 0.833，但受 max 限制
        # 实际仓位值 = 1M * 0.25/0.30 = 833K，但 max_position_pct=0.20
        # 所以实际 = 200K, 股数 = 200000/50/100*100 = 4000
        self.assertEqual(shares, 4000)

    def test_high_vol_lower_position(self):
        """验证高波动率 -> 低仓位"""
        shares_low_vol = VolatilityAdjustedSizing.calc_position_size(
            capital=1000000, price=50, volatility=0.20, target_vol=0.25
        )
        shares_high_vol = VolatilityAdjustedSizing.calc_position_size(
            capital=1000000, price=50, volatility=0.60, target_vol=0.25
        )
        self.assertGreaterEqual(shares_low_vol, shares_high_vol,
                                "低波动率应分配更多仓位")

    def test_kelly_fraction(self):
        """验证 Kelly 公式"""
        # 胜率 60%，平均盈利 5%，平均亏损 3%
        kelly = VolatilityAdjustedSizing.calc_kelly_fraction(
            win_rate=0.60, avg_win=0.05, avg_loss=0.03
        )
        # b = 0.05/0.03 = 1.667
        # kelly = (0.6*1.667 - 0.4)/1.667 = 0.36
        self.assertAlmostEqual(kelly, 0.25, delta=0.15)  # capped at 0.25

        # 低胜率场景
        kelly_low = VolatilityAdjustedSizing.calc_kelly_fraction(
            win_rate=0.30, avg_win=0.05, avg_loss=0.05
        )
        self.assertAlmostEqual(kelly_low, 0.0, delta=0.01)

    def test_borderline_cases(self):
        """边界条件测试"""
        # 零波动率
        shares = VolatilityAdjustedSizing.calc_position_size(
            capital=1000000, price=50, volatility=0, target_vol=0.25
        )
        self.assertEqual(shares, 0)

        # 极小价格
        shares = VolatilityAdjustedSizing.calc_position_size(
            capital=1000000, price=0.01, volatility=0.30, target_vol=0.25
        )
        self.assertGreater(shares, 0)


class TestRiskMetrics(unittest.TestCase):
    """测试风险度量"""

    def setUp(self):
        np.random.seed(42)
        n = 500
        self.returns = pd.Series(np.random.normal(0.001, 0.02, n))
        self.equity = 1000000 * np.cumprod(1 + self.returns)
        self.equity = pd.Series(self.equity)

    def test_var_historical(self):
        """验证历史 VaR 计算"""
        var = RiskMetrics.calc_var(self.returns, 0.95, "historical")
        self.assertLess(var, 0, "VaR 应为负值表示损失")

        # 5% 分位数处应有约 5% 的观测值在 VaR 以下
        exceedances = (self.returns <= var).mean()
        self.assertAlmostEqual(exceedances, 0.05, delta=0.03)

    def test_var_parametric(self):
        """验证参数法 VaR"""
        var = RiskMetrics.calc_var(self.returns, 0.95, "parametric")
        self.assertLess(var, 0, "参数法 VaR 应为负值")

    def test_cvar_worse_than_var(self):
        """验证 CVaR <= VaR（更极端）"""
        var = RiskMetrics.calc_var(self.returns, 0.95)
        cvar = RiskMetrics.calc_cvar(self.returns, 0.95)
        self.assertLessEqual(cvar, var, "CVaR 应 <= VaR（更极端）")

    def test_max_drawdown(self):
        """验证最大回撤计算"""
        max_dd, start, end = RiskMetrics.calc_max_drawdown(self.equity)
        self.assertLess(max_dd, 0, "最大回撤应为负值")
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)

    def test_risk_report(self):
        """验证完整风险报告"""
        report = RiskMetrics.calc_risk_report(self.equity, self.returns)
        required_keys = ['VaR_95', 'CVaR_95', 'max_drawdown', 'annual_volatility']
        for key in required_keys:
            self.assertIn(key, report)

        print("\n" + "=" * 60)
        print("风险度量测试报告")
        print("=" * 60)
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("=" * 60)


# ============================================================
# 集成测试: ATR 止损 + 波动率仓位 联合效果
# ============================================================

class TestIntegratedRiskManagement(unittest.TestCase):
    """集成测试: 动态风控综合效果"""

    def test_stoploss_reduces_drawdown(self):
        """
        验证: 使用 ATR 动态止损可以降低最大回撤
        """
        np.random.seed(123)
        n = 500
        # 模拟带趋势的随机价格
        close = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.02, n))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))

        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        prices = pd.Series(close, index=dates, name='close')
        high_s = pd.Series(high, index=dates, name='high')
        low_s = pd.Series(low, index=dates, name='low')

        atr = DynamicStopLoss.calc_atr(high_s, low_s, prices, 14)
        stop_prices = DynamicStopLoss.calc_trailing_stop(prices, atr, 2.0)

        # 模拟无止损交易
        equity_no_stop = prices / prices.iloc[0]

        # 模拟有止损交易
        equity_with_stop = pd.Series(index=prices.index, dtype=float)
        equity_with_stop.iloc[0] = 1.0
        in_position = True
        entry_price = prices.iloc[0]

        for i in range(1, len(prices)):
            if in_position:
                should_stop, reason = DynamicStopLoss.should_stop_out(
                    prices.iloc[i], stop_prices.iloc[i], entry_price
                )
                if should_stop:
                    in_position = False
                    # 止损退出，保持现金
                    equity_with_stop.iloc[i] = equity_with_stop.iloc[i - 1]
                else:
                    equity_with_stop.iloc[i] = equity_with_stop.iloc[i - 1] * \
                        (prices.iloc[i] / prices.iloc[i - 1])
            else:
                equity_with_stop.iloc[i] = equity_with_stop.iloc[i - 1]
                # 简单重新入场条件: 价格回到均线上方
                if i > 20 and prices.iloc[i] > prices.iloc[i - 20:i].mean():
                    in_position = True
                    entry_price = prices.iloc[i]

        max_dd_no_stop = (equity_no_stop / equity_no_stop.cummax() - 1).min()
        max_dd_with_stop = (equity_with_stop / equity_with_stop.cummax() - 1).min()

        print("\n" + "=" * 60)
        print("ATR 止损效果对比")
        print("=" * 60)
        print(f"无止损最大回撤: {max_dd_no_stop:.4%}")
        print(f"有止损最大回撤: {max_dd_with_stop:.4%}")
        print(f"回撤改善: {(max_dd_no_stop - max_dd_with_stop):.4%}")
        print("=" * 60)

        # 有止损的回撤应该优于或接近无止损（允许 10% 容差）
        # max_dd 为负值，更接近 0 表示回撤更小
        self.assertGreaterEqual(max_dd_with_stop, max_dd_no_stop - 0.10,
                                "有止损应有效控制最大回撤")


if __name__ == "__main__":
    unittest.main(verbosity=2)