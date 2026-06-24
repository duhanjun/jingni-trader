"""
Recorders 模块单元测试
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from skills.quant-optimizations.quant_opt_20260619.recorders import (
    SignalRecorder, SigAnaRecorder, PortAnaRecorder, RecorderManager
)


class TestSignalRecorder(unittest.TestCase):
    def test_record_predictions_dataframe(self):
        rec = SignalRecorder("test_signal", output_dir="/tmp/_quant_opt_test")
        predictions = pd.DataFrame({
            "code": ["000001"] * 100,
            "date": pd.date_range("2024-01-01", periods=100),
            "pred": np.random.randn(100),
        })
        result = rec.record({"predictions": predictions})
        self.assertTrue(result["success"])
        self.assertIn("n_records", result["metrics"])
        self.assertEqual(result["metrics"]["n_records"], 100)

    def test_record_missing_predictions(self):
        rec = SignalRecorder("test_signal2", output_dir="/tmp/_quant_opt_test")
        result = rec.record({})
        self.assertFalse(result["success"])


class TestSigAnaRecorder(unittest.TestCase):
    def test_ic_analysis(self):
        np.random.seed(42)
        n = 1000
        predictions = pd.DataFrame({
            "code": (np.arange(n) // 10) % 10,
            "date": pd.date_range("2024-01-01", periods=10).repeat(100),
            "pred": np.random.randn(n),
        })
        forward_returns = pd.DataFrame({
            "code": (np.arange(n) // 10) % 10,
            "date": pd.date_range("2024-01-01", periods=10).repeat(100),
            "ret_forward_1d": np.random.randn(n) * 0.01,
            "ret_forward_5d": np.random.randn(n) * 0.02,
        })
        rec = SigAnaRecorder("test_siga", output_dir="/tmp/_quant_opt_test")
        result = rec.record({"predictions": predictions, "forward_returns": forward_returns})
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["n_periods"], 1)

    def test_missing_inputs(self):
        rec = SigAnaRecorder("test_siga2", output_dir="/tmp/_quant_opt_test")
        result = rec.record({})
        self.assertFalse(result["success"])


class TestPortAnaRecorder(unittest.TestCase):
    def test_basic_metrics(self):
        dates = pd.date_range("2024-01-01", periods=252)
        np.random.seed(0)
        eq_values = 1_000_000 * (1 + np.random.randn(252).cumsum() * 0.01)
        equity_curve = pd.DataFrame({"date": dates, "equity": eq_values})

        rec = PortAnaRecorder("test_port", output_dir="/tmp/_quant_opt_test")
        result = rec.record({"equity_curve": equity_curve})
        self.assertTrue(result["success"])
        m = result["metrics"]
        self.assertIn("sharpe_ratio", m)
        self.assertIn("max_drawdown", m)
        self.assertIn("annual_return", m)


class TestRecorderManager(unittest.TestCase):
    def test_register_and_run_all(self):
        mgr = RecorderManager(output_dir="/tmp/_quant_opt_test_mgr")
        mgr.register(SignalRecorder("sig", output_dir="/tmp/_quant_opt_test_mgr"))
        mgr.register(PortAnaRecorder("port", output_dir="/tmp/_quant_opt_test_mgr"))

        predictions = pd.DataFrame({
            "code": ["000001"] * 50,
            "date": pd.date_range("2024-01-01", periods=50),
            "pred": np.random.randn(50),
        })
        equity_curve = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=100),
            "equity": np.linspace(1_000_000, 1_050_000, 100),
        })
        ctx = {"predictions": predictions, "equity_curve": equity_curve}
        results = mgr.run_all(ctx)
        self.assertIn("sig", results)
        self.assertIn("port", results)
        self.assertTrue(results["sig"]["success"])
        self.assertTrue(results["port"]["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)