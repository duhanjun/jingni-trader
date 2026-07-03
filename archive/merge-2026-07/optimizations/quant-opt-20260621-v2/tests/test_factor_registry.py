"""
因子注册表测试
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from factor_registry import (
    FactorRegistry, FactorMeta, FactorDirection, FactorCategory,
    build_default_a_share_registry,
)


class TestFactorRegistry(unittest.TestCase):

    def test_register_and_get(self):
        reg = FactorRegistry()
        meta = FactorMeta(name="test_factor", direction=FactorDirection.POSITIVE,
                          category=FactorCategory.MOMENTUM, description="测试")
        reg.register(meta)
        self.assertEqual(reg.get("test_factor").name, "test_factor")
        self.assertIsNone(reg.get("nonexistent"))

    def test_list_by_category(self):
        reg = FactorRegistry()
        reg.register(FactorMeta(name="m1", category=FactorCategory.MOMENTUM))
        reg.register(FactorMeta(name="m2", category=FactorCategory.MOMENTUM))
        reg.register(FactorMeta(name="v1", category=FactorCategory.VOLATILITY))
        momentum = reg.list_by_category(FactorCategory.MOMENTUM)
        self.assertEqual(len(momentum), 2)

    def test_list_by_direction(self):
        reg = FactorRegistry()
        reg.register(FactorMeta(name="pos", direction=FactorDirection.POSITIVE))
        reg.register(FactorMeta(name="neg", direction=FactorDirection.NEGATIVE))
        pos = reg.list_by_direction(FactorDirection.POSITIVE)
        neg = reg.list_by_direction(FactorDirection.NEGATIVE)
        self.assertEqual(len(pos), 1)
        self.assertEqual(len(neg), 1)

    def test_remove(self):
        reg = FactorRegistry()
        reg.register(FactorMeta(name="x"))
        self.assertTrue(reg.remove("x"))
        self.assertIsNone(reg.get("x"))
        self.assertFalse(reg.remove("x"))

    def test_adjust_direction(self):
        """反向因子应被取负"""
        reg = FactorRegistry()
        reg.register(FactorMeta(name="pos_f", direction=FactorDirection.POSITIVE))
        reg.register(FactorMeta(name="neg_f", direction=FactorDirection.NEGATIVE))
        values = {
            "pos_f": pd.Series([1, 2, 3]),
            "neg_f": pd.Series([1, 2, 3]),
        }
        adjusted = reg.adjust_direction(values)
        self.assertTrue((adjusted["pos_f"] == values["pos_f"]).all())
        self.assertTrue((adjusted["neg_f"] == -values["neg_f"]).all())

    def test_validate_ic_sign(self):
        reg = FactorRegistry()
        reg.register(FactorMeta(name="pos_f", expected_ic_sign=1))
        reg.register(FactorMeta(name="neg_f", expected_ic_sign=-1))
        self.assertTrue(reg.validate_ic_sign("pos_f", 0.05))
        self.assertFalse(reg.validate_ic_sign("pos_f", -0.05))
        self.assertTrue(reg.validate_ic_sign("neg_f", -0.05))
        self.assertFalse(reg.validate_ic_sign("neg_f", 0.05))

    def test_default_registry_coverage(self):
        """默认注册表应覆盖 factor-engine 的核心因子"""
        reg = build_default_a_share_registry()
        names = reg.list_names()
        # 对照 factor-engine.compute_a_share_factors 的输出
        for expected in ["reversal_5d", "reversal_20d", "lncap",
                         "turnover_20d", "volatility_20d", "money_flow_20d"]:
            self.assertIn(expected, names, f"默认注册表缺少 {expected}")
        # reversal_20d 应为正向(已取负)
        self.assertEqual(reg.get("reversal_20d").direction, FactorDirection.POSITIVE)
        # lncap 应为负向(小盘溢价)
        self.assertEqual(reg.get("lncap").direction, FactorDirection.NEGATIVE)

    def test_to_dict_serializable(self):
        reg = build_default_a_share_registry()
        d = reg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertGreater(len(d), 0)
        # 应可序列化
        import json
        json.dumps(d, default=str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
