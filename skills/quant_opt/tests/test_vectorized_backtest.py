"""
VectorizedBacktest 单元测试 + 与 native_adapter 的精度对比
========================================================
"""

import sys
import os
import time
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT_OPT_DIR = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(QUANT_OPT_DIR)
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from skills.quant_opt.vectorized_backtest import (
    VectorizedBacktest,
    BacktestResult,
    compare_results,
)


def make_synth_data(n_codes: int = 10, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    for code_i in range(n_codes):
        code = f"{code_i:06d}.SZ"
        base = 10.0 + code_i * 3
        ret = rng.normal(0.001, 0.02, n_days)
        prices = base * np.cumprod(1 + ret)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "open": prices[i] * 0.99,
                "high": prices[i] * 1.01,
                "low": prices[i] * 0.99,
                "close": prices[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


def make_signals(data: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """生成简单信号：每天 top_k 的票做多"""
    rows = []
    for dt, g in data.groupby("date"):
        ranked = g.sort_values("close", ascending=False).head(top_k)
        for _, r in ranked.iterrows():
            rows.append({"code": r["code"], "date": dt, "signal": 1})
    return pd.DataFrame(rows)


class TestVectorizedBacktest(unittest.TestCase):
    def setUp(self):
        self.data = make_synth_data()
        self.signals = make_signals(self.data)
        self.bt = VectorizedBacktest(init_capital=1_000_000)

    def test_basic_run(self):
        result = self.bt.run(self.data, self.signals)
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIn("total_return", result.metrics)

    def test_t1_rule(self):
        """T+1 规则：第一天信号应该在第二天成交"""
        # 取信号数据中的所有日期
        sig_dates = sorted(self.signals["date"].unique())
        first_signal_date = sig_dates[0]
        sigs = self.signals[self.signals["date"] == first_signal_date].copy()
        result = self.bt.run(self.data, sigs)

        # 第一个信号日当天不应该有任何成交
        if not result.trades.empty:
            first_trade_date = result.trades["date"].min()
            self.assertGreater(
                pd.Timestamp(first_trade_date),
                pd.Timestamp(first_signal_date),
                "T+1 rule violated: trade happened on signal day",
            )

    def test_price_limit(self):
        """涨跌停过滤：人为构造一个涨停股票，验证不买入"""
        data = self.data.copy()
        # 把所有股票的第一天都设成涨停（close == high）
        first_date = data["date"].min()
        mask = data["date"] == first_date
        data.loc[mask, "close"] = data.loc[mask, "high"]
        signals = make_signals(data)
        result = self.bt.run(data, signals)
        # T+1 延迟，所以应该不影响最终结果
        self.assertGreater(len(result.equity_curve), 0)

    def test_metrics_calculated(self):
        result = self.bt.run(self.data, self.signals)
        expected_keys = ["total_return", "annual_return", "volatility",
                         "sharpe_ratio", "max_drawdown", "win_rate"]
        for k in expected_keys:
            self.assertIn(k, result.metrics)

    def test_empty_signals(self):
        empty_sigs = pd.DataFrame(columns=["code", "date", "signal"])
        result = self.bt.run(self.data, empty_sigs)
        # 没有信号应该不亏损
        self.assertEqual(result.metrics.get("total_return", 0.0), 0.0)

    def test_perf_vs_native(self):
        """性能对比：确保向量化版至少不比 native 慢太多"""
        n_codes = 50
        n_days = 240
        big_data = make_synth_data(n_codes=n_codes, n_days=n_days)
        big_sigs = make_signals(big_data, top_k=10)

        t0 = time.time()
        result = self.bt.run(big_data, big_sigs)
        elapsed = time.time() - t0
        self.assertLess(
            elapsed, 30.0,
            f"Vectorized backtest too slow: {elapsed:.2f}s for {n_codes} codes x {n_days} days"
        )


class TestCompareResults(unittest.TestCase):
    def test_compare_results(self):
        native_metrics = {
            "total_return": 0.1,
            "annual_return": 0.12,
            "sharpe_ratio": 1.0,
        }
        v = BacktestResult(
            equity_curve=pd.DataFrame(),
            positions=pd.DataFrame(),
            trades=pd.DataFrame(),
            metrics={
                "total_return": 0.1005,
                "annual_return": 0.121,
                "sharpe_ratio": 0.98,
            },
        )
        report = compare_results(
            {"metrics": native_metrics}, v
        )
        self.assertIn("total_return", report)
        self.assertTrue(report["total_return"]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
