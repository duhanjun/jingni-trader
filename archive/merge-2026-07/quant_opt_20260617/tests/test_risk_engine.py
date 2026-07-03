"""
多层风控引擎验证测试
==================

测试维度：
    1. 正确性：与 jingni-trader 现有 CircuitBreaker 的覆盖行为一致
    2. 增强能力：状态机 (ACTIVE/REDUCING/HALTED)、行业偏离、限速、reduce_only
    3. 边界：未配置行业数据、停牌无价格等异常场景

运行：
    PYTHONPATH=. python workspace/quant_opt_20260617/tests/test_risk_engine.py
"""
from __future__ import annotations

import os
import sys
import time
import unittest

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from quant_opt_20260617.risk_engine.multi_layer_risk import (
    AccountSnapshot,
    DenialReason,
    MultiLayerRiskEngine,
    TokenBucketThrottler,
    TradingState,
)


def make_account(
    nav: float = 1_000_000.0,
    cash: float = 1_000_000.0,
    positions: dict = None,
    industry_map: dict = None,
    bench_weights: dict = None,
    prices: dict = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        nav=nav,
        available_cash=cash,
        start_of_day_nav=nav,
        positions=positions or {
            "600000.SH": {"volume": 1000, "avg_cost": 12.0},
            "600001.SH": {"volume": 2000, "avg_cost": 8.0},
        },
        industry_map=industry_map or {
            "600000.SH": "银行",
            "600001.SH": "钢铁",
        },
        benchmark_industry_weights=bench_weights or {
            "银行": 0.10, "钢铁": 0.05, "科技": 0.30,
        },
        prices=prices or {"600000.SH": 13.0, "600001.SH": 7.5},
    )


class TestMultiLayerRisk(unittest.TestCase):

    def setUp(self):
        self.audit = os.path.join(HERE, "..", "reports", "risk_audit.jsonl")
        if os.path.exists(self.audit):
            os.remove(self.audit)
        self.engine = MultiLayerRiskEngine(
            max_daily_loss_ratio=0.03,
            max_single_order_ratio=0.02,
            max_single_stock_weight=0.10,
            max_industry_deviation=0.05,
            max_orders_per_sec=5.0,
            audit_log_path=self.audit,
        )
        self.account = make_account()

    # --- 状态机 ---

    def test_01_active_state_allows_normal(self):
        # 用不在 industry_map 里的代码，避免行业偏离误拒
        d = self.engine.check_order(
            "601999.SH", "buy", 100, 13.0, self.account
        )
        self.assertTrue(d.allowed, f"应允许: {d.reason_detail}")
        self.assertEqual(d.trading_state, TradingState.ACTIVE)

    def test_02_halt_blocks_all(self):
        self.engine.halt("测试")
        for side in ("buy", "sell"):
            d = self.engine.check_order("600000.SH", side, 100, 13.0, self.account)
            self.assertFalse(d.allowed)
            self.assertEqual(d.reason, DenialReason.STATE_HALTED)

    def test_03_reducing_blocks_buy_only(self):
        self.engine.reduce_only("测试")
        d_buy = self.engine.check_order("600000.SH", "buy", 100, 13.0, self.account)
        d_sell = self.engine.check_order("600000.SH", "sell", 100, 13.0, self.account)
        self.assertFalse(d_buy.allowed)
        self.assertEqual(d_buy.reason, DenialReason.STATE_REDUCE_ONLY)
        self.assertTrue(d_sell.allowed, "REDUCING 状态下卖出应被允许")

    def test_04_resume(self):
        # 用一个不在 industry_map 里的代码，避免行业偏离误拒
        self.engine.halt("测试")
        self.engine.resume()
        d = self.engine.check_order("601999.SH", "buy", 100, 5.0, self.account)
        self.assertTrue(d.allowed, f"恢复后应允许: {d.reason_detail}")

    # --- 单笔金额上限 ---

    def test_05_single_order_size(self):
        # 用不在 industry_map 里的代码避免误拒；订单金额 > 2% * 1M = 20K
        d = self.engine.check_order("601999.SH", "buy", 2000, 13.0, self.account)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, DenialReason.SINGLE_ORDER_SIZE)

    # --- 单标的最大持仓 ---

    def test_06_position_limit(self):
        # 用不在 industry_map 里的代码；构造一个高单价标的
        # 现持仓 4500 股 * 200 = 900K (90% of NAV)
        # 买 80 股 * 200 = 16K (刚好不超单笔 2% = 20K)
        # 未来 4580 * 200 = 916K / 1M = 91.6% > 10% → 触发 POSITION_LIMIT
        # 现金够 (16K << 1M)，所以不会先触发 INSUFFICIENT_CASH
        self.account.positions["601999.SH"] = {"volume": 4500, "avg_cost": 200.0}
        d = self.engine.check_order(
            "601999.SH", "buy", 80, 200.0, self.account
        )
        self.assertFalse(d.allowed, f"应被拒: {d.reason_detail}")
        self.assertEqual(d.reason, DenialReason.POSITION_LIMIT)

    # --- 日亏止损 ---

    def test_07_daily_loss(self):
        acc = make_account(nav=960_000.0, cash=1_000_000.0)
        # 当日 NAV 较 start_of_day 跌 4%
        acc.start_of_day_nav = 1_000_000.0
        d = self.engine.check_order("601999.SH", "buy", 100, 13.0, acc)
        self.assertFalse(d.allowed, f"应被拒: {d.reason_detail}")
        self.assertEqual(d.reason, DenialReason.DAILY_LOSS)

    # --- reduce_only / 减仓校验 ---

    def test_08_reduce_only_oversell(self):
        # 现持仓 1000 股，要卖 2000 股
        d = self.engine.check_order("600000.SH", "sell", 2000, 13.0, self.account)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, DenialReason.REDUCE_ONLY_FAIL)

    # --- 资金不足 ---

    def test_09_insufficient_cash(self):
        acc = make_account(cash=1000.0)
        d = self.engine.check_order("600000.SH", "buy", 1000, 13.0, acc)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, DenialReason.INSUFFICIENT_CASH)

    # --- 频率节流 ---

    def test_10_frequency_throttle(self):
        # 用不在 industry_map 里的代码，确保 100 股 * 5 元 = 500 元 < 各种阈值，仅频率是瓶颈
        decisions = []
        for i in range(10):
            d = self.engine.check_order("601999.SH", "buy", 100, 5.0, self.account)
            decisions.append(d)
        denied = [d for d in decisions if not d.allowed]
        self.assertGreater(len(denied), 0, "应至少有部分订单因限频被拒")
        self.assertTrue(all(d.reason == DenialReason.FREQUENCY for d in denied),
                        f"所有拒绝应都是 FREQUENCY，实际: {[d.reason for d in denied]}")

    # --- 行业偏离 ---

    def test_11_industry_deviation(self):
        # 用更宽松的 single_order_ratio 避免先触发 size 限制
        eng = MultiLayerRiskEngine(
            max_daily_loss_ratio=0.03,
            max_single_order_ratio=0.5,  # 50% 放宽
            max_single_stock_weight=0.95,
            max_industry_deviation=0.05,
            max_orders_per_sec=5.0,
        )
        acc = make_account(
            industry_map={"600999.SH": "科技"},
            bench_weights={"科技": 0.0},
        )
        # 买 10000 股 * 20 = 20万 → 行业从 0 升到 20%
        # 20% - 0 = 20% 超过 5% 阈值
        d = eng.check_order("600999.SH", "buy", 10_000, 20.0, acc)
        self.assertFalse(d.allowed, f"应被拒: {d.reason_detail}")
        self.assertEqual(d.reason, DenialReason.INDUSTRY_DEVIATION)

    def test_12_industry_deviation_ok(self):
        """小金额买同行，偏离可控，应允许"""
        # bench=0.05，买 100 * 13 = 1300 = 0.13% of NAV
        # 偏差 0.13% - 5% = -4.87% < 5% 阈值
        acc = make_account(
            industry_map={"600999.SH": "科技"},
            bench_weights={"科技": 0.05},
        )
        d = self.engine.check_order("600999.SH", "buy", 100, 13.0, acc)
        self.assertTrue(d.allowed, f"应允许: {d.reason_detail}")

    # --- 行业偏离：缺数据时不触发 ---

    def test_13_industry_skip_when_no_data(self):
        acc = make_account(industry_map={}, bench_weights={})
        d = self.engine.check_order("600999.SH", "buy", 100, 13.0, acc)
        self.assertTrue(d.allowed, "无行业数据时应跳过行业检查")

    # --- 审计日志 ---

    def test_14_audit_log_written(self):
        self.engine.check_order("600000.SH", "buy", 100, 13.0, self.account)
        self.engine.halt("测试")
        self.engine.check_order("600000.SH", "buy", 100, 13.0, self.account)
        self.assertTrue(os.path.exists(self.audit))
        with open(self.audit, "r") as f:
            lines = [line for line in f if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        # 最后一条应当是 DENY
        last = lines[-1]
        self.assertIn("DENY", last)
        self.assertIn("STATE_HALTED", last)

    # --- 统计接口 ---

    def test_15_stats(self):
        # 用不在 industry_map 里的代码，避免行业偏离误拒
        for i in range(5):
            self.engine.check_order("601999.SH", "buy", 100, 5.0, self.account)
        self.engine.halt("测试")
        for i in range(3):
            self.engine.check_order("601999.SH", "buy", 100, 5.0, self.account)
        stats = self.engine.stats()
        self.assertEqual(stats["trading_state"], "HALTED")
        self.assertEqual(stats["allowed"], 5)
        self.assertEqual(stats["denied"], 3)
        self.assertIn("STATE_HALTED", stats["denial_reasons"])

    # --- 与现有 CircuitBreaker 行为对齐 ---

    def test_16_alignment_with_existing_circuit_breaker(self):
        """
        与 skills/execution-monitor-engine 中的 CircuitBreaker 行为对齐：
        1) 日亏超限 → 拒
        2) 单笔超限 → 拒
        3) 频率超限 → 拒
        """
        # 由于 skills/execution-monitor-engine 是连字符目录名，
        # Python 不能直接以包名 import，这里通过 importlib 加载
        import importlib.util
        em_path = os.path.join(
            PROJECT_ROOT, "skills", "execution-monitor-engine", "engine.py"
        )
        spec = importlib.util.spec_from_file_location("execution_engine", em_path)
        if "execution_engine" in sys.modules:
            em_mod = sys.modules["execution_engine"]
        else:
            em_mod = importlib.util.module_from_spec(spec)
            sys.modules["execution_engine"] = em_mod
            spec.loader.exec_module(em_mod)
        CircuitBreaker = em_mod.CircuitBreaker
        Account = em_mod.Account

        # 场景 1: 日亏
        acc_a = Account(nav=1_000_000.0, available_cash=800_000.0,
                        start_of_day_nav=1_050_000.0)
        cb = CircuitBreaker()
        old = cb.check_send_order(acc_a, "X", 1000.0)
        self.assertFalse(old["allowed"])
        self.assertIn("单日亏损", old["reason"])

        new_acc = make_account(nav=950_000.0, cash=800_000.0)
        new_acc.start_of_day_nav = 1_000_000.0
        new = self.engine.check_order("600000.SH", "buy", 100, 5.0, new_acc)
        self.assertFalse(new.allowed)
        self.assertEqual(new.reason, DenialReason.DAILY_LOSS)

        # 场景 2: 单笔超限
        # 现有 CircuitBreaker 的 MAX_SINGLE_ORDER_RATIO = 0.10
        # 现有 CB 的 get_current_nav() 默认按 cash 计，所以 200K 订单 > 10% * 1M = 100K
        acc_a = Account(nav=1_000_000.0, available_cash=1_000_000.0,
                        start_of_day_nav=1_000_000.0)
        old = cb.check_send_order(acc_a, "X", 200_000.0)
        self.assertFalse(old["allowed"], f"应被拒: {old}")
        self.assertIn("单笔金额", old["reason"])

        # 新引擎：cash=nav=1M，订单 6000 股 * 13 = 78K > 2% (20K)
        new_acc = make_account(nav=1_000_000.0, cash=1_000_000.0)
        new = self.engine.check_order("601999.SH", "buy", 6000, 13.0, new_acc)
        self.assertFalse(new.allowed, f"应被拒: {new.reason_detail}")
        self.assertEqual(new.reason, DenialReason.SINGLE_ORDER_SIZE)


class TestTokenBucket(unittest.TestCase):

    def test_01_initial_capacity(self):
        """初始 capacity 个令牌可立即通过"""
        b = TokenBucketThrottler(rate_per_sec=5.0)
        self.assertTrue(b.allow())
        self.assertTrue(b.allow())

    def test_02_blocked_when_empty(self):
        """令牌耗尽后拒绝"""
        b = TokenBucketThrottler(rate_per_sec=5.0, capacity=2)
        self.assertTrue(b.allow())
        self.assertTrue(b.allow())
        # 第 3 个立即再来应该被拒
        self.assertFalse(b.allow())

    def test_03_refill(self):
        """等待一会后应有新令牌"""
        b = TokenBucketThrottler(rate_per_sec=100.0, capacity=1)
        self.assertTrue(b.allow())
        self.assertFalse(b.allow())
        time.sleep(0.05)  # 0.05s * 100/s = 5 个令牌
        self.assertTrue(b.allow())


def _standalone_main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _standalone_main()
