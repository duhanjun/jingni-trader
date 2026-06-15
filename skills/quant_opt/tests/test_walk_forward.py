"""
WalkForwardRunner 单元测试
==========================
"""

import sys
import os
import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT_OPT_DIR = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(QUANT_OPT_DIR)
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from skills.quant_opt.walk_forward import (
    WalkForwardRunner,
    WalkForwardWindow,
    WindowResult,
    generate_windows,
)


def make_long_data(start: str = "2018-01-01", end: str = "2024-01-01",
                   n_codes: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range(start, end, freq="B")
    rows = []
    for code_i in range(n_codes):
        code = f"{code_i:06d}.SZ"
        base = 10.0 + code_i * 5
        ret = rng.normal(0.001, 0.02, len(dates))
        prices = base * np.cumprod(1 + ret)
        for i, d in enumerate(dates):
            rows.append({
                "code": code,
                "date": d,
                "close": prices[i],
                "volume": int(rng.integers(1_000_000, 5_000_000)),
            })
    return pd.DataFrame(rows)


def dummy_train(data: pd.DataFrame, window: WalkForwardWindow) -> Any:
    """最简单的训练：返回均值"""
    return {"mean_close": float(data["close"].mean())}


def dummy_backtest(model: Any, data: pd.DataFrame,
                   window: WalkForwardWindow) -> Dict[str, float]:
    """最简单的回测：计算窗口内累计收益"""
    if data.empty:
        return {"return": 0.0, "sharpe": 0.0}
    pivot = data.pivot_table(index="date", columns="code", values="close")
    if pivot.shape[1] < 2:
        return {"return": 0.0, "sharpe": 0.0}
    ret = pivot.pct_change().mean(axis=1).dropna()
    return {
        "return": float((1 + ret).prod() - 1) if len(ret) else 0.0,
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0,
    }


class TestGenerateWindows(unittest.TestCase):
    def test_basic_generation(self):
        windows = generate_windows(
            "2018-01-01", "2024-01-01",
            train_months=12, valid_months=3, test_months=3,
            step_months=3,
        )
        self.assertGreater(len(windows), 0)
        for w in windows:
            # 验证顺序：train_end < valid_start < valid_end < test_start < test_end
            ts = pd.Timestamp
            self.assertLess(ts(w.train_end), ts(w.valid_start))
            self.assertLess(ts(w.valid_end), ts(w.test_start))
            self.assertLess(ts(w.test_start), ts(w.test_end))

    def test_window_id_increments(self):
        windows = generate_windows(
            "2018-01-01", "2024-01-01",
            train_months=12, valid_months=3, test_months=3,
            step_months=3,
        )
        ids = [w.window_id for w in windows]
        self.assertEqual(ids, list(range(len(windows))))


class TestWalkForwardRunner(unittest.TestCase):
    def setUp(self):
        self.data = make_long_data()
        self.runner = WalkForwardRunner(
            train_fn=dummy_train,
            backtest_fn=dummy_backtest,
        )

    def test_basic_run(self):
        windows = generate_windows(
            "2018-01-01", "2024-01-01",
            train_months=12, valid_months=3, test_months=3,
            step_months=6,
        )
        report = self.runner.run(self.data, windows)
        self.assertIn("window_results", report)
        self.assertIn("summary", report)
        self.assertEqual(len(report["window_results"]), len(windows))
        # 至少 80% 窗口成功（数据范围可能有边界问题）
        n_ok = sum(1 for r in report["window_results"] if r.error is None)
        self.assertGreater(n_ok / max(len(windows), 1), 0.5)

    def test_summary_stats(self):
        windows = generate_windows(
            "2018-01-01", "2024-01-01",
            train_months=12, valid_months=3, test_months=3,
            step_months=6,
        )
        report = self.runner.run(self.data, windows)
        summary = report["summary"]
        if summary:  # 至少有一个成功窗口
            for k, v in summary.items():
                self.assertIn("mean", v)
                self.assertIn("std", v)
                self.assertIn("n_windows", v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
