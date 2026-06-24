"""
CPCV 模块单元测试 + 与 jingni-trader 原 purged_group_ts_split 的对比

测试目标：
  1. 正确性：n_paths = C(n_splits, n_test_splits)
  2. 正确性：每条 path 的 train/test 无重叠
  3. 正确性：embargo 区间内样本不进入训练集
  4. 正确性：purge 区间内样本不进入训练集
  5. 对比：CPCV vs jingni 原 split 的差异
"""
import os
import sys
import unittest
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from skills.quant_optimizations.quant_opt_20260619.cpcv import CombinatorialPurgedCV, CPVCSplit


def make_synthetic_data(n_dates=200, n_codes=10):
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D")
    codes = [f"{600000 + i:06d}" for i in range(n_codes)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"code": c, "date": d, "value": np.random.randn()})
    return pd.DataFrame(rows)


class TestCPCVBasics(unittest.TestCase):
    def setUp(self):
        self.df = make_synthetic_data(200, 5)
        self.X = self.df.drop(columns=["value"])

    def test_n_paths_matches_combination(self):
        for n_splits, n_test in [(4, 2), (5, 2), (6, 2)]:
            cv = CombinatorialPurgedCV(n_splits=n_splits, n_test_splits=n_test,
                                        embargo_pct=0.0, purge_pct=0.0)
            expected = comb(n_splits, n_test)
            self.assertEqual(cv.n_paths(), expected)
            actual = sum(1 for _ in cv.split(self.X))
            self.assertEqual(actual, expected)

    def test_train_test_no_overlap(self):
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1,
                                    embargo_pct=0.0, purge_pct=0.0)
        for split in cv.split(self.X):
            overlap = set(split.train_idx.tolist()) & set(split.test_idx.tolist())
            self.assertEqual(len(overlap), 0, f"path {split.path_id} 有 train/test 重叠")

    def test_all_test_folds_covered_per_path(self):
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2,
                                    embargo_pct=0.0, purge_pct=0.0)
        for split in cv.split(self.X):
            n_test_expected = (len(self.X) // 4) * 2
            self.assertAlmostEqual(
                len(split.test_idx), n_test_expected,
                delta=5,
                msg=f"path {split.path_id} test 数量异常",
            )

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            CombinatorialPurgedCV(n_splits=3, n_test_splits=3)
        with self.assertRaises(ValueError):
            CombinatorialPurgedCV(n_splits=4, n_test_splits=2, embargo_pct=0.6)


class TestCPCVPurgeAndEmbargo(unittest.TestCase):
    """验证 purge + embargo 真的从 train 中删除了样本"""

    def setUp(self):
        np.random.seed(0)
        self.df = make_synthetic_data(200, 5)
        self.X = self.df.drop(columns=["value"])

    def test_embargo_removes_samples_after_test(self):
        """embargo 比例 0.1 时，test 后 10 个样本应被从 train 中剔除"""
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1,
                                    embargo_pct=0.05, purge_pct=0.0)
        for split in cv.split(self.X):
            test_max = split.test_idx.max()
            in_embargo = (split.train_idx > test_max) & (split.train_idx <= test_max + int(0.05 * len(self.X)))
            self.assertEqual(in_embargo.sum(), 0, "embargo 区间内样本泄漏到 train")

    def test_purge_removes_boundary_samples(self):
        """purge 比例 0.05 时，test 前后各 5% 样本应被剔除"""
        cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1,
                                    embargo_pct=0.0, purge_pct=0.05)
        for split in cv.split(self.X):
            test_min, test_max = split.test_idx.min(), split.test_idx.max()
            purge_window = int(0.05 * len(self.X))
            in_purge_pre = (split.train_idx >= test_min - purge_window) & (split.train_idx < test_min)
            in_purge_post = (split.train_idx > test_max) & (split.train_idx <= test_max + purge_window)
            self.assertEqual(in_purge_pre.sum(), 0, "test 前 purge 区间泄漏")
            self.assertEqual(in_purge_post.sum(), 0, "test 后 purge 区间泄漏")


class TestCPCVvsJingNi(unittest.TestCase):
    """对比 jingni-trader 原 purged_group_ts_split"""

    def test_cpcv_more_paths_than_legacy(self):
        try:
            from skills.strategy_model_engine.engine import ModelEngine
        except (ImportError, ModuleNotFoundError):
            self.skipTest("jingni-trader 子模块不可用，跳过此对比测试")
        dates = pd.date_range("2023-01-01", periods=240, freq="D")
        codes = [f"{600000 + i:06d}" for i in range(10)]
        rows = []
        for d in dates:
            for c in codes:
                rows.append({"code": c, "date": d, "value": np.random.randn()})
        df = pd.DataFrame(rows)

        me = ModelEngine()
        legacy = me.purged_group_ts_split(df["date"], n_splits=3)

        cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2,
                                      embargo_pct=0.01, purge_pct=0.01)
        new_paths = list(cpcv.split(df))

        self.assertGreater(len(new_paths), len(legacy),
                           "CPCV 应提供更多路径")

    def test_cpcv_smaller_train_size_due_to_embargo(self):
        """由于 embargo + purge，CPCV 单 path 的 train_size 应 < 无 purge 的版本"""
        n = 1000
        X = pd.DataFrame({"i": range(n)})

        cv_no_purge = CombinatorialPurgedCV(n_splits=4, n_test_splits=1,
                                            embargo_pct=0.0, purge_pct=0.0)
        cv_with = CombinatorialPurgedCV(n_splits=4, n_test_splits=1,
                                        embargo_pct=0.05, purge_pct=0.05)
        s_no = next(cv_no_purge.split(X))
        s_with = next(cv_with.split(X))
        self.assertLess(len(s_with.train_idx), len(s_no.train_idx),
                        "加 embargo+purge 后 train 应更小")


if __name__ == "__main__":
    unittest.main(verbosity=2)