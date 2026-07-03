"""vectorized_bt 单元测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
import pandas as pd

from quant_opt.vectorized_bt import run_vectorized_backtest


def make_data(n_stocks=10, n_days=200):
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rows = []
    for s in range(n_stocks):
        price = 10 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_days)))
        for d, p in zip(dates, price):
            rows.append({"date": d, "code": f"{s:06d}.SH", "close": p})
    return pd.DataFrame(rows)


def make_momentum_signals(data, top_k=3, lookback=5):
    pivot = data.pivot_table(index='date', columns='code', values='close')
    ret = pivot.pct_change(lookback)
    sig = (ret.rank(axis=1, ascending=False) <= top_k).astype(int)
    sig_long = sig.stack().reset_index()
    sig_long.columns = ['date', 'code', 'signal']
    return sig_long


class TestVectorizedBT(unittest.TestCase):

    def test_basic_run(self):
        data = make_data()
        signals = make_momentum_signals(data)
        out = run_vectorized_backtest(data, signals, top_k=3, use_vbt=False)
        self.assertIn("equity", out)
        self.assertIn("metrics", out)
        self.assertGreater(len(out["equity"]), 0)
        self.assertIn("total_return", out["metrics"])
        self.assertIn("sharpe_ratio", out["metrics"])
        self.assertIn("max_drawdown", out["metrics"])

    def test_empty_input(self):
        out = run_vectorized_backtest(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(out["metrics"], {})

    def test_no_signals(self):
        data = make_data(n_stocks=3, n_days=50)
        signals = data[['date', 'code']].copy()
        signals['signal'] = 0
        out = run_vectorized_backtest(data, signals, use_vbt=False)
        # 没有信号, 收益应接近 0
        self.assertIn("equity", out)
        self.assertGreater(len(out["equity"]), 0)

    def test_top_k_limit(self):
        data = make_data(n_stocks=5, n_days=100)
        signals = make_momentum_signals(data, top_k=2)
        out = run_vectorized_backtest(data, signals, top_k=2, use_vbt=False)
        weights = out.get("weights", pd.DataFrame())
        # 每天最多 2 个非零权重
        if not weights.empty:
            non_zero_per_day = (weights > 0).sum(axis=1)
            self.assertLessEqual(non_zero_per_day.max(), 2)

    def test_metrics_consistency(self):
        data = make_data(n_stocks=8, n_days=150)
        signals = make_momentum_signals(data, top_k=3)
        out = run_vectorized_backtest(data, signals, top_k=3, use_vbt=False)
        metrics = out["metrics"]
        # annual_vol > 0
        self.assertGreater(metrics["annual_vol"], 0)
        # max_drawdown <= 0
        self.assertLessEqual(metrics["max_drawdown"], 0)
        # total_return 应与 equity 末值/初值 - 1 一致
        eq = out["equity"]
        expected_tr = float(eq.iloc[-1] / eq.iloc[0] - 1)
        self.assertAlmostEqual(metrics["total_return"], expected_tr, places=4)


if __name__ == "__main__":
    unittest.main()
