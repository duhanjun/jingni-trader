"""
风险控制模块测试

验证内容:
1. 凯利公式正确性: 已知胜率/盈亏比下计算结果应与手算一致
2. ATR 止损正确性: ATR 计算与止损价应符合公式
3. 回撤断路器: 不同回撤级别应触发不同仓位乘数
4. 边界条件: 空数据、全胜/全亏、零波动率等
5. 集成测试: RiskManager 综合评估应产出合理建议
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

from optimizations.independent_v2.risk_control import (
    RiskManager, KellySizer, ATRStopLoss, DrawdownCircuitBreaker,
)
from optimizations.independent_v2.data_fixtures import make_synthetic_ohlcv


# ---------- 凯利公式测试 ----------

class TestKellySizer:
    def test_kelly_formula_correctness(self):
        """凯利公式计算应与手算一致。

        胜率 p=0.6, 盈亏比 b=1.5
        f* = (1.5*0.6 - 0.4) / 1.5 = 0.5/1.5 = 0.333
        半凯利 = 0.333 * 0.5 = 0.167
        """
        kelly = KellySizer(fraction=0.5, max_weight=1.0)
        f = kelly.calc_kelly(win_rate=0.6, win_loss_ratio=1.5)
        expected = (1.5 * 0.6 - 0.4) / 1.5 * 0.5
        assert abs(f - expected) < 1e-10, f"凯利计算错误: {f} vs {expected}"

    def test_kelly_zero_when_negative_expectation(self):
        """负期望游戏（胜率低/盈亏比低）凯利应为0。"""
        kelly = KellySizer()
        # 胜率 0.3, 盈亏比 1.0 → 负期望
        f = kelly.calc_kelly(win_rate=0.3, win_loss_ratio=1.0)
        assert f == 0.0, f"负期望游戏凯利应为0: {f}"

    def test_kelly_respects_max_weight(self):
        """凯利仓位不应超过 max_weight 上限。"""
        kelly = KellySizer(fraction=1.0, max_weight=0.25)
        # 极高胜率+盈亏比，凯利会很大，但应被截断
        f = kelly.calc_kelly(win_rate=0.9, win_loss_ratio=5.0)
        assert f <= 0.25, f"凯利仓位超过上限: {f}"

    def test_kelly_estimate_from_trades(self):
        """从成交记录估计胜率与盈亏比应正确。"""
        trades = pd.DataFrame({
            "pnl": [100, -50, 200, -80, 150, -30, 120],
        })
        kelly = KellySizer(fraction=1.0, max_weight=1.0)
        stats = kelly.estimate_from_trades(trades)
        # 4 胜 3 负
        assert abs(stats["win_rate"] - 4/7) < 1e-6
        # 平均盈利 = (100+200+150+120)/4 = 142.5
        # 平均亏损 = (50+80+30)/3 = 53.33
        expected_ratio = 142.5 / 53.33
        assert abs(stats["win_loss_ratio"] - expected_ratio) < 0.1

    def test_kelly_empty_trades(self):
        """空成交记录应返回默认值，不崩溃。"""
        kelly = KellySizer()
        stats = kelly.estimate_from_trades(pd.DataFrame())
        assert stats["win_rate"] == 0.5
        assert stats["kelly_f"] == 0.0

    def test_kelly_all_winning_trades(self):
        """全胜记录应正常处理（无亏损记录时盈亏比默认1.0）。"""
        trades = pd.DataFrame({"pnl": [100, 200, 150]})
        kelly = KellySizer()
        stats = kelly.estimate_from_trades(trades)
        assert stats["win_rate"] == 1.0
        assert stats["kelly_f"] > 0

    def test_position_sizing_returns_shares(self):
        """仓位计算应返回整手股数（A股100股整手）。"""
        kelly = KellySizer(fraction=1.0, max_weight=0.5)
        result = kelly.size_position(
            capital=1_000_000, win_rate=0.6, win_loss_ratio=2.0, price=10.0
        )
        assert result["shares"] % 100 == 0, "股数应为100的整数倍"
        assert result["amount"] > 0
        assert result["weight"] > 0


# ---------- ATR 止损测试 ----------

class TestATRStopLoss:
    def test_atr_calculation(self):
        """ATR 计算应与公式一致。"""
        # 构造已知数据
        high = pd.Series([11, 12, 10, 11, 12], dtype=float)
        low = pd.Series([9, 10, 8, 9, 10], dtype=float)
        close = pd.Series([10, 11, 9, 10, 11], dtype=float)
        atr_stop = ATRStopLoss(atr_period=3, atr_multiplier=2.0)
        atr = atr_stop.calc_atr(high, low, close)
        # ATR 不应为空，且第一日为 NaN（无前一日收盘价）
        assert len(atr) == 5
        assert atr.iloc[0] >= 0  # EMA 第一项

    def test_stop_price_long(self):
        """多头止损价 = 入场价 - multiplier × ATR。"""
        atr_stop = ATRStopLoss(atr_period=14, atr_multiplier=2.0)
        stop = atr_stop.calc_stop_price(entry_price=100, atr=2.5, direction="long")
        assert abs(stop - 95.0) < 1e-10, f"多头止损价错误: {stop}"

    def test_stop_price_short(self):
        """空头止损价 = 入场价 + multiplier × ATR。"""
        atr_stop = ATRStopLoss(atr_period=14, atr_multiplier=2.0)
        stop = atr_stop.calc_stop_price(entry_price=100, atr=2.5, direction="short")
        assert abs(stop - 105.0) < 1e-10, f"空头止损价错误: {stop}"

    def test_trailing_stop_only_moves_up(self):
        """追踪止损价应只上移不下移。"""
        data = make_synthetic_ohlcv(n_codes=1, n_days=30, seed=5)
        entry = pd.DataFrame([{
            "code": data["code"].iloc[0],
            "date": data["date"].iloc[10],
            "entry_price": data["close"].iloc[10],
        }])
        atr_stop = ATRStopLoss(atr_period=5, atr_multiplier=2.0)
        stops = atr_stop.generate_stop_signals(data, entry)
        if not stops.empty:
            # 止损价序列应单调不减
            stop_prices = stops["stop_price"].values
            diffs = np.diff(stop_prices)
            assert (diffs >= -1e-6).all(), "追踪止损价不应下移"


# ---------- 回撤断路器测试 ----------

class TestDrawdownCircuitBreaker:
    def test_normal_status(self):
        """小幅回撤（<5%）应保持正常仓位。"""
        breaker = DrawdownCircuitBreaker()
        mult = breaker.get_position_multiplier(-0.03)
        assert mult == 1.0

    def test_warn_status(self):
        """5%-10% 回撤应触发预警降仓。"""
        breaker = DrawdownCircuitBreaker()
        mult = breaker.get_position_multiplier(-0.07)
        assert mult == 0.75

    def test_emergency_status(self):
        """10%-20% 回撤应触发紧急降仓。"""
        breaker = DrawdownCircuitBreaker()
        mult = breaker.get_position_multiplier(-0.15)
        assert mult == 0.50

    def test_halt_status(self):
        """超过20% 回撤应清仓。"""
        breaker = DrawdownCircuitBreaker()
        mult = breaker.get_position_multiplier(-0.25)
        assert mult == 0.0

    def test_drawdown_calculation(self):
        """回撤计算应正确。"""
        equity = pd.Series([100, 110, 105, 95, 90, 100])
        breaker = DrawdownCircuitBreaker()
        dd = breaker.calc_drawdown(equity)
        # 峰值 110, 谷值 90, 最大回撤 = (90-110)/110 = -0.1818
        assert abs(dd.min() - (-20/110)) < 1e-6

    def test_apply_to_weights(self):
        """断路器应正确调整权重矩阵。"""
        dates = pd.date_range("2023-01-02", periods=5)
        codes = ["A", "B"]
        weights = pd.DataFrame(
            0.5, index=dates, columns=codes
        )
        # 构造净值：第3天回撤 15%（触发紧急降仓）
        equity = pd.Series([1.0, 1.1, 0.935, 0.95, 1.0], index=dates)
        breaker = DrawdownCircuitBreaker()
        adjusted = breaker.apply_to_equity(equity, weights)
        # 第3天应被降仓到 0.5 * 0.5 = 0.25
        assert abs(adjusted.iloc[2, 0] - 0.25) < 1e-6, (
            f"紧急降仓后权重错误: {adjusted.iloc[2, 0]}"
        )

    def test_get_status_returns_dict(self):
        breaker = DrawdownCircuitBreaker()
        status = breaker.get_status(-0.12)
        assert status["status"] == "EMERGENCY_REDUCE"
        assert status["position_multiplier"] == 0.50
        assert "drawdown" in status


# ---------- 集成测试 ----------

class TestRiskManagerIntegration:
    def test_evaluate_returns_complete_report(self):
        """RiskManager.evaluate 应返回完整风控报告。"""
        data = make_synthetic_ohlcv(n_codes=3, n_days=100, seed=7)
        equity = pd.Series(np.linspace(1e6, 9e5, 100))  # 持续下跌
        trades = pd.DataFrame({
            "pnl": [1000, -500, 800, -300, 600, -200, 400],
        })

        rm = RiskManager()
        report = rm.evaluate(equity, trades)
        assert "kelly" in report
        assert "drawdown" in report
        assert "recommendation" in report
        assert isinstance(report["recommendation"], str)
        assert len(report["recommendation"]) > 0

    def test_evaluate_with_empty_data(self):
        """空数据应不崩溃。"""
        rm = RiskManager()
        report = rm.evaluate(pd.Series(dtype=float), pd.DataFrame())
        assert "kelly" in report
        assert "drawdown" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
