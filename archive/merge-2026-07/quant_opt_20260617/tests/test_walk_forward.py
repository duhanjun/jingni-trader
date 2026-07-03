"""
Walk-Forward 验证测试
"""
import unittest
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from quant_opt_20260617.walk_forward import (
    WalkForwardCV, make_walk_forward_splits, WindowSplit
)
from quant_opt_20260617.tests._synthetic_data import generate_synthetic_a_share_data


class TestWalkForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_a_share_data(n_stocks=20, n_days=1000, seed=42)
        # 准备训练数据
        cls.data = cls.data.sort_values(['code', 'date']).reset_index(drop=True)
        cls.data['ret_5d'] = cls.data.groupby('code')['close'].transform(
            lambda x: x.shift(-5) / x - 1
        )
        cls.data = cls.data.dropna(subset=['ret_5d'])
        # 简单特征：5/10/20 日动量
        cls.data['mom_5'] = cls.data.groupby('code')['close'].transform(
            lambda x: x.pct_change(5)
        )
        cls.data['mom_10'] = cls.data.groupby('code')['close'].transform(
            lambda x: x.pct_change(10)
        )
        cls.data['mom_20'] = cls.data.groupby('code')['close'].transform(
            lambda x: x.pct_change(20)
        )
        cls.data = cls.data.dropna()
        cls.X = cls.data[['mom_5', 'mom_10', 'mom_20']].reset_index(drop=True)
        cls.y = cls.data['ret_5d'].reset_index(drop=True)
        cls.dates = cls.data['date'].reset_index(drop=True)

    def test_01_window_generation(self):
        """窗口生成器"""
        splits = make_walk_forward_splits(
            self.dates,
            train_window_days=300,
            test_window_days=30,
            purge_gap_days=10,
            min_train_samples=500,
        )
        self.assertGreater(len(splits), 0, "应至少生成一个窗口")
        for sp in splits:
            # 测试集必须严格在训练集之后
            self.assertGreater(sp.test_start, sp.train_end)
            # purge gap 应生效
            self.assertGreaterEqual(
                (sp.test_start - sp.train_end).days, 10
            )
            # 训练/测试集不重叠
            self.assertEqual(len(np.intersect1d(sp.train_idx, sp.test_idx)), 0)

    def test_02_window_count(self):
        """窗口数量合理性"""
        splits = make_walk_forward_splits(
            self.dates, train_window_days=300, test_window_days=30,
            step_days=30, purge_gap_days=10, min_train_samples=500,
        )
        # 1000 个交易日，step=30，理论可生成约 (1000-300-10)/30 ≈ 23 个
        self.assertGreaterEqual(len(splits), 5)

    def test_03_expanding_vs_rolling(self):
        """扩展窗口 vs 滚动窗口的训练样本量差异"""
        splits_expand = make_walk_forward_splits(
            self.dates, train_window_days=300, test_window_days=30,
            purge_gap_days=10, min_train_samples=500, expanding=True,
        )
        # 扩展窗口下，后面的窗口训练样本应 >= 前面的窗口
        for i in range(1, len(splits_expand)):
            self.assertGreaterEqual(
                len(splits_expand[i].train_idx),
                len(splits_expand[i - 1].train_idx)
            )

    def test_04_run_with_simple_model(self):
        """用 LinearRegression 跑通 walk-forward"""
        cv = WalkForwardCV(
            model_factory=lambda: LinearRegression(),
            scorer="auto",
            train_window_days=300,
            test_window_days=30,
            step_days=30,
            purge_gap_days=10,
            min_train_samples=500,
            verbose=False,
        )
        result = cv.run(self.X, self.y, self.dates)
        self.assertGreater(len(result.windows), 0)
        # 每个窗口都应有 oos_predictions
        for w in result.windows:
            self.assertIsNotNone(w.oos_predictions)
            self.assertGreater(len(w.oos_predictions), 0)
            self.assertIn("rank_ic", w.metrics)
            self.assertIn("rmse", w.metrics)
        # 整体 OOS 应有拼接预测
        self.assertIsNotNone(result.oos_predictions)
        self.assertGreater(len(result.oos_predictions), 0)
        # 整体 OOS 指标应包含 rank_ic
        self.assertIn("rank_ic", result.overall_metrics)

    def test_05_oos_no_overlap_with_own_train(self):
        """
        OOS 预测绝对不能与自身 fold 的训练集重叠（防 look-ahead）
        注意：OOS 与后续 fold 的训练集重叠是允许的（不同时间段）
        """
        cv = WalkForwardCV(
            model_factory=lambda: LinearRegression(),
            scorer="auto",
            train_window_days=300, test_window_days=30, step_days=30,
            purge_gap_days=10, min_train_samples=500, verbose=False,
        )
        result = cv.run(self.X, self.y, self.dates)
        # 重新生成 splits 以获得原始 train_idx
        splits = make_walk_forward_splits(
            self.dates, train_window_days=300, test_window_days=30,
            step_days=30, purge_gap_days=10, min_train_samples=500,
        )
        split_by_fold = {sp.fold_id: sp for sp in splits}
        for w in result.windows:
            sp = split_by_fold.get(w.fold_id)
            if sp is None:
                continue
            # OOS 索引 = w.oos_predictions.index
            oos_idx = set(w.oos_predictions.index.tolist())
            train_set = set(sp.train_idx.tolist())
            overlap = oos_idx & train_set
            self.assertEqual(len(overlap), 0,
                             f"fold {w.fold_id} OOS 与自身训练集重叠 {len(overlap)} 条")

    def test_06_to_dataframe(self):
        """to_dataframe 应能正常转表格"""
        cv = WalkForwardCV(
            model_factory=lambda: LinearRegression(),
            scorer="auto", train_window_days=300, test_window_days=30,
            step_days=30, purge_gap_days=10, min_train_samples=500,
            verbose=False,
        )
        result = cv.run(self.X, self.y, self.dates)
        df = WalkForwardCV.to_dataframe(result)
        self.assertIn("fold_id", df.columns)
        self.assertIn("rank_ic", df.columns)
        self.assertEqual(len(df), len(result.windows))

    def test_07_purge_gap_effect(self):
        """purge_gap 越大，能生成的窗口数越少"""
        splits_no_purge = make_walk_forward_splits(
            self.dates, train_window_days=300, test_window_days=30,
            step_days=30, purge_gap_days=1, min_train_samples=500,
        )
        splits_with_purge = make_walk_forward_splits(
            self.dates, train_window_days=300, test_window_days=30,
            step_days=30, purge_gap_days=30, min_train_samples=500,
        )
        self.assertGreaterEqual(len(splits_no_purge), len(splits_with_purge))

    def test_08_invalid_params(self):
        """无效日期范围应抛错"""
        dates_short = pd.Series(pd.to_datetime(['2024-01-01', '2024-01-02']))
        with self.assertRaises(ValueError):
            make_walk_forward_splits(dates_short)

    def test_09_perf_baseline(self):
        """性能：30 stocks × 1000 days，5 个 walk-forward 窗口，< 30s"""
        big = generate_synthetic_a_share_data(n_stocks=30, n_days=1000, seed=7)
        big = big.sort_values(['code', 'date']).reset_index(drop=True)
        big['ret_5d'] = big.groupby('code')['close'].transform(lambda x: x.shift(-5) / x - 1)
        big['mom_5'] = big.groupby('code')['close'].transform(lambda x: x.pct_change(5))
        big['mom_20'] = big.groupby('code')['close'].transform(lambda x: x.pct_change(20))
        big = big.dropna()
        X = big[['mom_5', 'mom_20']].reset_index(drop=True)
        y = big['ret_5d'].reset_index(drop=True)
        dates = big['date'].reset_index(drop=True)

        cv = WalkForwardCV(
            model_factory=lambda: LinearRegression(),
            scorer="rank_ic", train_window_days=300, test_window_days=30,
            step_days=60, purge_gap_days=10, min_train_samples=500, verbose=False,
        )
        t0 = time.time()
        result = cv.run(X, y, dates)
        elapsed = time.time() - t0
        self.assertGreater(len(result.windows), 0)
        self.assertLess(elapsed, 30.0, f"性能不达标: {elapsed:.2f}s > 30s")


if __name__ == "__main__":
    unittest.main()
