"""
增强回测绩效指标测试

测试内容：
1. 正确性测试：已知净值曲线 → 已知指标
2. Alpha/Beta 测试：与 CAPM 理论值一致
3. 信息比率测试：与定义一致
4. 换手率测试：已知持仓变化 → 已知换手率
5. 最大回撤持续期测试：已知净值 → 已知持续期
6. 边界条件测试：空数据、单点、无基准
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine_opt.enhanced_metrics import (
    calc_turnover,
    calc_alpha_beta,
    calc_information_ratio,
    calc_max_drawdown_duration,
    calc_all_enhanced_metrics,
)


class TestAlphaBeta:
    def test_beta_one_for_identical_returns(self):
        """策略收益=基准收益时，beta 应为 1，alpha 应为 0"""
        np.random.seed(0)
        bench = pd.Series(np.random.normal(0.001, 0.01, 252))
        strat = bench.copy()  # 完全跟随基准
        alpha, beta = calc_alpha_beta(strat, bench, risk_free=0.0)
        assert abs(beta - 1.0) < 1e-6, f"beta 应为 1, 实际 {beta}"
        assert abs(alpha) < 1e-6, f"alpha 应为 0, 实际 {alpha}"

    def test_beta_zero_for_uncorrelated(self):
        """策略与基准不相关时，beta 应接近 0"""
        np.random.seed(1)
        bench = pd.Series(np.random.normal(0, 0.01, 1000))
        strat = pd.Series(np.random.normal(0, 0.01, 1000))
        alpha, beta = calc_alpha_beta(strat, bench, risk_free=0.0)
        assert abs(beta) < 0.1, f"beta 应接近 0, 实际 {beta}"

    def test_known_beta(self):
        """构造 beta=2 的策略，验证回归结果"""
        np.random.seed(2)
        bench = pd.Series(np.random.normal(0.001, 0.01, 500))
        noise = pd.Series(np.random.normal(0, 0.001, 500))
        strat = 2.0 * bench + 0.0005 + noise  # beta=2, alpha_daily=0.0005
        alpha, beta = calc_alpha_beta(strat, bench, risk_free=0.0)
        assert abs(beta - 2.0) < 0.05, f"beta 应为 2, 实际 {beta}"
        # alpha 年化 = 0.0005 * 252 ≈ 0.126，噪声导致估计有偏差，放宽容差
        assert abs(alpha - 0.126) < 0.03, f"alpha 应≈0.126, 实际 {alpha}"

    def test_empty_returns(self):
        """空数据应返回 0"""
        alpha, beta = calc_alpha_beta(pd.Series(dtype=float), pd.Series(dtype=float))
        assert alpha == 0.0
        assert beta == 0.0


class TestInformationRatio:
    def test_positive_ir_for_outperformer(self):
        """持续跑赢基准的策略 IR 应为正"""
        np.random.seed(3)
        bench = pd.Series(np.random.normal(0, 0.01, 252))
        strat = bench + 0.001  # 每日稳定超额 0.1%
        ir = calc_information_ratio(strat, bench)
        assert ir > 0, f"跑赢基准 IR 应为正, 实际 {ir}"

    def test_zero_ir_for_identical(self):
        """策略=基准时 IR 应为 0（跟踪误差=0）"""
        np.random.seed(4)
        r = pd.Series(np.random.normal(0.001, 0.01, 252))
        ir = calc_information_ratio(r, r)
        assert ir == 0.0, f"完全跟随 IR 应为 0, 实际 {ir}"

    def test_ir_definition(self):
        """IR = 年化超额均值 / 年化跟踪误差"""
        np.random.seed(5)
        bench = pd.Series(np.random.normal(0, 0.01, 252))
        strat = pd.Series(np.random.normal(0.001, 0.012, 252))
        ir = calc_information_ratio(strat, bench)
        excess = strat - bench
        expected_ir = (excess.mean() * 252) / (excess.std() * np.sqrt(252))
        assert abs(ir - expected_ir) < 1e-6


class TestTurnover:
    def test_known_turnover(self):
        """已知持仓变化 → 已知换手率"""
        # 构造 2 日：第1日全仓 A，第2日全仓 B（100% 换手）
        dates = pd.bdate_range("2023-01-02", periods=2)
        positions = pd.DataFrame({
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "code": ["A", "B", "A", "B"],
            "market_value": [1000000, 0, 0, 1000000],
        })
        equity_curve = pd.DataFrame({
            "date": dates,
            "equity": [1000000, 1000000],
        })
        # 单边换手 = (buy+sell)/2 / equity
        # 第2日：A 卖出 100万，B 买入 100万 → (100+100)/2/100 = 1.0
        # 日均 = 1.0（仅1个有效日），年化 = 1.0 * 252
        turnover = calc_turnover(positions, equity_curve, trading_days=252)
        assert abs(turnover - 252.0) < 1.0, f"换手率应为 252, 实际 {turnover}"

    def test_no_trading_zero_turnover(self):
        """持仓不变时换手率应为 0"""
        dates = pd.bdate_range("2023-01-02", periods=5)
        positions = pd.DataFrame({
            "date": dates.repeat(2),
            "code": ["A", "B"] * 5,
            "market_value": [500000, 500000] * 5,
        })
        equity_curve = pd.DataFrame({"date": dates, "equity": [1000000] * 5})
        turnover = calc_turnover(positions, equity_curve, trading_days=252)
        assert turnover == 0.0, f"无交易换手率应为 0, 实际 {turnover}"

    def test_empty_equity(self):
        """空净值曲线应返回 0"""
        turnover = calc_turnover(pd.DataFrame(), pd.DataFrame())
        assert turnover == 0.0


class TestMaxDrawdownDuration:
    def test_known_duration(self):
        """已知净值序列 → 已知回撤持续期"""
        # 净值：1.0, 0.9, 0.8, 0.9, 1.0, 1.1
        # 回撤持续：从第2日（<1.0）到第5日（=1.0），共 3 日 underwater
        equity = pd.Series([1.0, 0.9, 0.8, 0.9, 1.0, 1.1])
        dur = calc_max_drawdown_duration(equity)
        assert dur == 3, f"回撤持续期应为 3, 实际 {dur}"

    def test_no_drawdown(self):
        """单调递增无回撤，持续期应为 0"""
        equity = pd.Series(np.arange(1, 11, dtype=float))
        dur = calc_max_drawdown_duration(equity)
        assert dur == 0

    def test_full_drawdown(self):
        """全程回撤"""
        equity = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6])
        dur = calc_max_drawdown_duration(equity)
        assert dur == 4, f"全程回撤持续期应为 4, 实际 {dur}"


class TestAllEnhancedMetrics:
    def test_complete_metrics(self):
        """完整指标计算应返回所有字段"""
        np.random.seed(6)
        dates = pd.bdate_range("2023-01-02", periods=252)
        equity = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.01, 252)) * 1e6, index=dates)
        bench = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)

        metrics = calc_all_enhanced_metrics(
            equity, benchmark_returns=bench, risk_free=0.03
        )
        required = {"turnover", "alpha", "beta", "information_ratio", "max_drawdown_duration"}
        assert required.issubset(metrics.keys())
        assert isinstance(metrics["max_drawdown_duration"], int)

    def test_no_benchmark(self):
        """无基准时 alpha/beta/IR 应为 0"""
        equity = pd.Series([1e6, 1.01e6, 1.02e6, 1.015e6, 1.03e6])
        metrics = calc_all_enhanced_metrics(equity, benchmark_returns=None)
        assert metrics["alpha"] == 0.0
        assert metrics["beta"] == 0.0
        assert metrics["information_ratio"] == 0.0

    def test_empty_equity(self):
        """空净值应返回全 0"""
        metrics = calc_all_enhanced_metrics(pd.Series(dtype=float))
        assert metrics["turnover"] == 0.0
        assert metrics["max_drawdown_duration"] == 0
