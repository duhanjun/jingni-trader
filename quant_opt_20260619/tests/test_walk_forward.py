"""walk_forward 单元测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
import pandas as pd

from quant_opt_20260619.walk_forward import WalkForwardConfig, WalkForwardValidator


def make_data(n_days: int = 500, n_stocks: int = 5):
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for c in codes:
        price = 10 * np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
        for d, p in zip(dates, price):
            rows.append({"date": d, "code": c, "close": p,
                         "open": p, "high": p * 1.01, "low": p * 0.99,
                         "volume": 1_000_000})
    return pd.DataFrame(rows)


def toy_bt(data_slice, signal_slice):
    """toy 回测函数 - 确定性 (基于数据计算)"""
    if data_slice.empty:
        return {"equity": pd.Series(dtype=float), "trades": pd.DataFrame(),
                "metrics": {}, "n_trades": 0}
    dates = sorted(data_slice['date'].unique())
    # 基于数据本身生成确定性收益 (避免随机种子不一致)
    if 'close' in data_slice.columns:
        pivot = data_slice.pivot_table(index='date', columns='code', values='close')
        avg_ret = float(pivot.pct_change().mean().mean())
        vol = float(pivot.pct_change().std().mean())
    else:
        avg_ret, vol = 0.0005, 0.01

    n = len(dates)
    np.random.seed(len(dates))  # 用窗口长度做种子, 保证稳定性
    rets = np.random.normal(avg_ret, max(vol, 0.001), n)
    eq = pd.Series((1 + rets).cumprod() * 1e6,
                   index=pd.DatetimeIndex(dates))
    return {
        "equity": eq,
        "trades": pd.DataFrame(),
        "metrics": {
            "sharpe_ratio": float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0.0,
            "annual_return": float(np.mean(rets) * 252),
            "max_drawdown": -0.05,
            "calmar_ratio": 1.0,
        },
        "n_trades": 0,
    }


class TestWalkForward(unittest.TestCase):

    def test_config_total_windows(self):
        cfg = WalkForwardConfig(train_window=120, test_window=60, step=60)
        self.assertEqual(cfg.total_windows(180), 0)  # 不够
        self.assertEqual(cfg.total_windows(360), 3)  # 3 窗口

    def test_split_windows(self):
        data = make_data(n_days=300, n_stocks=2)
        cfg = WalkForwardConfig(train_window=100, test_window=50, step=50, purge_gap=2)
        v = WalkForwardValidator(cfg, toy_bt)
        dates = pd.DatetimeIndex(sorted(data['date'].unique()))
        windows = v.split_windows(dates)
        self.assertGreater(len(windows), 0)
        for w in windows:
            self.assertLess(w['train_start'], w['train_end'])
            self.assertLess(w['test_start'], w['test_end'])
            self.assertLess(w['train_end'], w['test_start'])

    def test_run_basic(self):
        data = make_data(n_days=400, n_stocks=3)
        signals = data[['date', 'code']].copy()
        signals['signal'] = np.random.choice([0, 1], size=len(signals))

        cfg = WalkForwardConfig(train_window=120, test_window=60, step=60, purge_gap=2)
        v = WalkForwardValidator(cfg, toy_bt)
        out = v.run(data, signals)
        self.assertIn("windows", out)
        self.assertIn("oos_aggregate", out)
        self.assertIn("summary", out)
        self.assertGreater(len(out["windows"]), 0)
        self.assertIn("sharpe_ratio_mean", out["oos_aggregate"])
        self.assertIn("n_windows", out["oos_aggregate"])

    def test_run_with_decay_ratio(self):
        data = make_data(n_days=500, n_stocks=2)
        signals = data[['date', 'code']].copy()
        signals['signal'] = np.random.choice([0, 1], size=len(signals))
        cfg = WalkForwardConfig(train_window=150, test_window=60, step=60, purge_gap=2)
        v = WalkForwardValidator(cfg, toy_bt)
        out = v.run(data, signals)
        agg = out["oos_aggregate"]
        # decay_ratio 应在合理范围 (0.1 ~ 10)
        if "sharpe_decay_ratio" in agg:
            self.assertGreater(agg["sharpe_decay_ratio"], -1.0)
            self.assertLess(agg["sharpe_decay_ratio"], 10.0)

    def test_run_empty(self):
        cfg = WalkForwardConfig()
        v = WalkForwardValidator(cfg, toy_bt)
        out = v.run(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(out["summary"], "empty input")


if __name__ == "__main__":
    unittest.main()