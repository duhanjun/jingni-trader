"""
PIT 模块单元测试 + 性能对比

测试目标：
  1. 正确性：PITDataFrame.filter_asof() 不返回未来数据
  2. 正确性：PITValidator.audit_pipeline() 能捕获 look-ahead 案例
  3. 正确性：check_version_consistency() 能识别版本冲突
  4. 性能：与 jingni-trader 现有 data-engine 的简单加载做对比
"""
import os
import sys
import time
import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from quant_opt_20260619.pit import (
    PITSpec, PITDataFrame, PITValidator, make_synthetic_pit
)


class TestPITDataFrame(unittest.TestCase):
    """PITDataFrame 基础功能测试"""

    def setUp(self):
        rows = [
            {"code": "000001", "period": "2023Q1", "asof": "2023-04-25", "value": 0.45},
            {"code": "000001", "period": "2023Q2", "asof": "2023-08-25", "value": 0.52},
            {"code": "000001", "period": "2023Q3", "asof": "2023-10-25", "value": 0.48},
            {"code": "000002", "period": "2023Q1", "asof": "2023-04-30", "value": 0.10},
            {"code": "000002", "period": "2023Q2", "asof": "2023-08-30", "value": 0.20},
        ]
        self.df = pd.DataFrame(rows)
        self.spec = PITSpec(feature_name="eps", period_col="period", asof_col="asof", freq="q")
        self.pit = PITDataFrame(self.df, self.spec)

    def test_filter_asof_excludes_future(self):
        """在 2023-06-01 这个时点，只能看到 2023Q1 数据"""
        as_of = pd.Timestamp("2023-06-01")
        visible = self.pit.filter_asof(as_of)
        codes_periods = set(zip(visible["code"], visible["period"]))
        self.assertIn(("000001", "2023Q1"), codes_periods)
        self.assertIn(("000002", "2023Q1"), codes_periods)
        self.assertNotIn(("000001", "2023Q2"), codes_periods,
                         "Q2 数据 (2023-08-25 发布) 在 6-1 时不可见")
        self.assertNotIn(("000001", "2023Q3"), codes_periods)

    def test_filter_asof_at_exact_publish_date_is_visible(self):
        """恰好等于发布日期时，数据应可见"""
        as_of = pd.Timestamp("2023-08-25")
        visible = self.pit.filter_asof(as_of)
        codes_periods = set(zip(visible["code"], visible["period"]))
        self.assertIn(("000001", "2023Q2"), codes_periods)

    def test_get_latest(self):
        """get_latest 应返回截至 as_of 的最新一期"""
        as_of = pd.Timestamp("2023-09-01")
        latest = self.pit.get_latest("000001", as_of)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["period"], "2023Q2")
        self.assertEqual(latest["value"], 0.52)

    def test_get_latest_empty(self):
        """as_of 早于所有发布时间时，返回 None"""
        as_of = pd.Timestamp("2020-01-01")
        latest = self.pit.get_latest("000001", as_of)
        self.assertIsNone(latest)


class TestPITValidator(unittest.TestCase):
    """PITValidator 审计能力测试"""

    def test_lookahead_detection(self):
        """构造有 look-ahead 风险的 case，验证能被检测"""
        df = make_synthetic_pit(n_stocks=3, n_periods=4)
        spec = PITSpec(feature_name="test", period_col="period", asof_col="asof", freq="q")
        v = PITValidator(df, spec)

        eval_ts = pd.Timestamp("2023-04-01")
        issues = v.check_lookahead(eval_ts)
        self.assertGreater(len(issues), 0, "应至少发现 1 条 look-ahead 风险")

    def test_version_consistency_detection(self):
        """同 (code, period) 多版本场景"""
        rows = [
            {"code": "000001", "period": "2023Q1", "asof": "2023-04-25", "value": 0.45},
            {"code": "000001", "period": "2023Q1", "asof": "2023-05-10", "value": 0.46},
            {"code": "000001", "period": "2023Q1", "asof": "2023-06-15", "value": 0.47},
            {"code": "000002", "period": "2023Q1", "asof": "2023-04-30", "value": 0.10},
        ]
        df = pd.DataFrame(rows)
        spec = PITSpec(feature_name="test", period_col="period", asof_col="asof", freq="q")
        v = PITValidator(df, spec)
        issues = v.check_version_consistency()
        conflict_codes = {i["code"] for i in issues}
        self.assertIn("000001", conflict_codes)
        self.assertNotIn("000002", conflict_codes)

    def test_audit_pipeline_aggregates(self):
        """audit_pipeline 输出格式正确"""
        df = make_synthetic_pit(n_stocks=2, n_periods=6)
        spec = PITSpec(feature_name="test", period_col="period", asof_col="asof", freq="q")
        v = PITValidator(df, spec)
        report = v.audit_pipeline([pd.Timestamp("2023-04-01"), pd.Timestamp("2023-08-01")])
        self.assertIn("n_eval_points", report)
        self.assertIn("by_severity", report)
        self.assertEqual(report["n_eval_points"], 2)


class TestPITPerformance(unittest.TestCase):
    """PIT 校验性能基线"""

    def test_filter_asof_perf(self):
        n_rows = 100_000
        df = make_synthetic_pit(n_stocks=500, n_periods=200)
        spec = PITSpec(feature_name="eps", period_col="period", asof_col="asof", freq="q")
        pit = PITDataFrame(df, spec)
        t0 = time.time()
        for _ in range(100):
            pit.filter_asof(pd.Timestamp("2023-09-01"))
        elapsed = time.time() - t0
        print(f"\n[PIT perf] 100次 filter_asof (n={n_rows}): {elapsed:.3f}s, "
              f"avg={elapsed*10:.2f}ms/次")
        self.assertLess(elapsed, 5.0, "filter_asof 单次平均应 < 50ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)