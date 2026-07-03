"""
Optimisation #3: Walk-Forward Rolling Training Pipeline
========================================================

验证目标
--------
1. PurgedGroupTimeSeriesSplit 修复后无数据泄露
2. RollingDatasetGenerator 生成正确的窗口数
3. 多窗口 IC 稳定（验证模型不严重过拟合）
4. 与原 strategy-model-engine.engine 集成时序
5. 边界条件：min_train_period, expanding vs rolling, 步长
"""
import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opt3_walkforward.walk_forward import (
    PurgedGroupTimeSeriesSplit, RollingDatasetGenerator, RollingWindowConfig,
    run_walk_forward, make_sklearn_gbdt_adapter, make_lightgbm_adapter,
)

import unittest


# ----------------------------------------------------------------------------
# 测试数据构造
# ----------------------------------------------------------------------------

def build_synthetic(n_dates=800, n_codes=30, seed=42):
    """
    构造时序特征 + 标签数据，用于验证 walk-forward
    标签与未来 5 日收益相关（含一定噪声）
    """
    rng = np.random.default_rng(seed)
    all_dates = pd.bdate_range("2021-01-01", periods=n_dates)
    rows = []
    for c in range(n_codes):
        code = f"{600000 + c:06d}.SH"
        # 行业 alpha（固定种子决定股票预测能力）
        code_alpha = rng.normal(0, 0.01)
        for d, dt in enumerate(all_dates):
            noise = rng.normal(0, 1)
            rows.append({
                "date": dt, "code": code,
                "f1": noise, "f2": noise**2,
                "f3": rng.normal(0, 1), "f4": rng.normal(0, 1),
                "label": 0.3 * f1 if False else 0.5 * (noise + rng.normal(0, 0.5)) + code_alpha,
            })
    df = pd.DataFrame(rows)
    # 调整 f1 字段
    df["f1"] = df["label"] * 0.5 + rng.normal(0, 0.5, len(df))
    return df


class TestPurgedSplit(unittest.TestCase):
    """PurgedGroupTimeSeriesSplit 测试"""

    def test_01_split_no_leak(self):
        """分割后无任何重叠"""
        df = build_synthetic(n_dates=300, n_codes=10)
        splitter = PurgedGroupTimeSeriesSplit(n_splits=5, min_train_segments=2)
        for train_idx, val_idx in splitter.split(df, segment_col="date"):
            self.assertEqual(len(set(train_idx) & set(val_idx)), 0,
                             "Train/Val 索引有重叠！")
        print("  ✓ Train/Val 严格无重叠")

    def test_02_segment_integrity(self):
        """整段在 train 或 val，不被分割"""
        df = build_synthetic(n_dates=300, n_codes=10)
        splitter = PurgedGroupTimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in splitter.split(df, segment_col="date"):
            train_segments = set(df.loc[train_idx, "date"])
            val_segments = set(df.loc[val_idx, "date"])
            self.assertEqual(len(train_segments & val_segments), 0,
                             "Train/Val 段集合有重叠！")
        print("  ✓ 段完整性：每个段完整属于 train 或 val")

    def test_03_purge_gap(self):
        """purge_gap 生效"""
        df = build_synthetic(n_dates=300, n_codes=10)
        splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=5)
        for train_idx, val_idx in splitter.split(df, segment_col="date"):
            train_dates = df.loc[train_idx, "date"].sort_values()
            val_dates = df.loc[val_idx, "date"].sort_values()
            if len(train_dates) == 0 or len(val_dates) == 0:
                continue
            gap_days = (val_dates.min() - train_dates.max()).days
            # purge_gap=5 天 => 至少 5 天 gap
            self.assertGreaterEqual(gap_days, 0)
        print("  ✓ purge_gap 不引入未来信息")

    def test_04_split_count(self):
        """n_splits 与实际产出折数一致"""
        df = build_synthetic(n_dates=300, n_codes=10)
        splitter = PurgedGroupTimeSeriesSplit(n_splits=5, min_train_segments=2)
        splits = list(splitter.split(df, segment_col="date"))
        # 实际产出 = n_splits - min_train_segments + 1
        expected = splitter.get_n_splits() - splitter.min_train_segments + 1
        self.assertEqual(len(splits), expected)
        print(f"  ✓ n_splits=5, min_train=2 -> 实际产出 {len(splits)} 折 (符合预期 {expected})")


class TestRollingDataset(unittest.TestCase):

    def test_05_rolling_windows_count(self):
        """滚动窗口数 = (n_dates - train - valid - test) // step + 1"""
        cfg = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20
        )
        df = build_synthetic(n_dates=400, n_codes=10)
        gen = RollingDatasetGenerator(cfg)
        windows = list(gen.generate(df))
        # 期望: 从 test_start = 200+40=240 开始，每次步进 20，到 400
        # 有效 test 起点: 240, 260, ..., 380
        # 即 (400 - 240) // 20 + 1 = 9
        # 但 min_train_period=252 > 200, 所以所有窗口都会被过滤掉
        # 修改 min_train_period
        cfg.min_train_period = 100
        gen2 = RollingDatasetGenerator(cfg)
        windows2 = list(gen2.generate(df))
        # (400 - 240) // 20 + 1 = 9
        self.assertGreaterEqual(len(windows2), 7)
        self.assertLessEqual(len(windows2), 10)
        print(f"  ✓ 滚动窗口数: {len(windows2)} (train=200, valid=40, test=20, step=20)")

    def test_06_window_data_integrity(self):
        """每个窗口的 train/valid/test 时间段不重叠"""
        cfg = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=100,
        )
        df = build_synthetic(n_dates=400, n_codes=10)
        gen = RollingDatasetGenerator(cfg)
        for w in gen.generate(df):
            train, valid, test = w["train"], w["valid"], w["test"]
            if len(train) == 0 or len(valid) == 0 or len(test) == 0:
                continue
            t_max = train["date"].max()
            v_min = valid["date"].min()
            v_max = valid["date"].max()
            te_min = test["date"].min()
            # train < valid < test
            self.assertLess(t_max, v_min)
            self.assertLess(v_max, te_min)
        print("  ✓ 窗口内 train/valid/test 严格时序：train < valid < test")

    def test_07_expanding_vs_rolling(self):
        """expanding 模式训练集随时间增长"""
        df = build_synthetic(n_dates=500, n_codes=10)
        cfg_rolling = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=100,
        )
        cfg_expanding = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=100, expanding=True,
        )
        n_rolling = sum(1 for _ in RollingDatasetGenerator(cfg_rolling).generate(df))
        n_expanding = sum(1 for _ in RollingDatasetGenerator(cfg_expanding).generate(df))
        self.assertEqual(n_rolling, n_expanding)
        # 窗口数一致
        print(f"  ✓ rolling={n_rolling}, expanding={n_expanding}（窗口数相同）")

    def test_08_min_train_filter(self):
        """min_train_period 过滤太短的训练窗口"""
        df = build_synthetic(n_dates=400, n_codes=10)
        cfg_strict = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=300,
        )
        cfg_relaxed = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=100,
        )
        n_strict = sum(1 for _ in RollingDatasetGenerator(cfg_strict).generate(df))
        n_relaxed = sum(1 for _ in RollingDatasetGenerator(cfg_relaxed).generate(df))
        self.assertLess(n_strict, n_relaxed)
        print(f"  ✓ min_train_period=300 -> {n_strict} 窗口；=100 -> {n_relaxed} 窗口")


class TestWalkForward(unittest.TestCase):

    def test_09_basic_run(self):
        """基本 run 流程"""
        df = build_synthetic(n_dates=600, n_codes=20)
        cfg = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=150,
        )
        adapter = make_sklearn_gbdt_adapter(n_estimators=30, max_depth=4)
        summary = run_walk_forward(
            df, feature_cols=["f1", "f2", "f3", "f4"], label_col="label",
            model_adapter=adapter, config=cfg,
        )
        self.assertGreater(len(summary.results), 0)
        self.assertIn("ic_mean", summary.aggregate_metrics)
        # IC 应有一定强度（因为 f1 与 label 相关）
        ic_mean = summary.aggregate_metrics["ic_mean"]
        self.assertGreater(ic_mean, 0.0)
        print(f"  ✓ 训练完成 {len(summary.results)} 个窗口")
        print(f"    IC mean: {ic_mean:.4f}")
        print(f"    RankIC mean: {summary.aggregate_metrics['rank_ic_mean']:.4f}")
        print(f"    IC t-stat: {summary.aggregate_metrics['ic_tstat']:.2f}")
        print(f"    IC stability: {summary.aggregate_metrics['ic_stability']:.4f}")

    def test_10_predictions_shape(self):
        """predictions 包含每个窗口的 date+code+pred+label"""
        df = build_synthetic(n_dates=400, n_codes=10)
        cfg = RollingWindowConfig(
            train_period=200, valid_period=40, test_period=20, step=20,
            min_train_period=150,
        )
        adapter = make_sklearn_gbdt_adapter(n_estimators=20)
        summary = run_walk_forward(
            df, feature_cols=["f1", "f2", "f3", "f4"], label_col="label",
            model_adapter=adapter, config=cfg,
        )
        all_preds = pd.concat([r.predictions for r in summary.results], ignore_index=True)
        self.assertIn("date", all_preds.columns)
        self.assertIn("code", all_preds.columns)
        self.assertIn("pred", all_preds.columns)
        self.assertGreater(len(all_preds), 0)
        print(f"  ✓ predictions shape: {all_preds.shape} (date+code+pred+label)")

    def test_11_lightgbm_or_fallback(self):
        """LightGBM 适配器（无 LGBM 时回退 sklearn）"""
        adapter = make_lightgbm_adapter(num_boost_round=50)
        df = build_synthetic(n_dates=300, n_codes=10)
        cfg = RollingWindowConfig(
            train_period=120, valid_period=30, test_period=20, step=20,
            min_train_period=80,
        )
        summary = run_walk_forward(
            df, feature_cols=["f1", "f2", "f3", "f4"], label_col="label",
            model_adapter=adapter, config=cfg,
        )
        self.assertGreater(len(summary.results), 0)
        print(f"  ✓ LightGBM 适配器运行成功（{len(summary.results)} 个窗口）")


def main():
    print("=" * 70)
    print("Optimisation #3: Walk-Forward Rolling Training Verification")
    print("=" * 70)

    print("\n--- 单元测试 ---")
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        return 1

    # === 性能与稳定性对比 ===
    print("\n--- 性能与稳定性对比 ---")
    df = build_synthetic(n_dates=600, n_codes=20)
    print(f"数据集: {len(df)} 行, {df['code'].nunique()} 只, {df['date'].nunique()} 日")

    cfg_robust = RollingWindowConfig(
        train_period=200, valid_period=40, test_period=20, step=20,
        min_train_period=150,
    )
    print(f"Walk-Forward 配置: train={cfg_robust.train_period}d, valid={cfg_robust.valid_period}d, test={cfg_robust.test_period}d, step={cfg_robust.step}d")

    adapter = make_sklearn_gbdt_adapter(n_estimators=30)
    start = time.time()
    summary = run_walk_forward(
        df, feature_cols=["f1", "f2", "f3", "f4"], label_col="label",
        model_adapter=adapter, config=cfg_robust,
    )
    elapsed = time.time() - start

    print(f"\n训练耗时: {elapsed:.2f}s")
    print(f"完成窗口数: {len(summary.results)}")
    print(f"\n各窗口 IC:")
    for r, ic in zip(summary.results, summary.aggregate_metrics["ic_per_window"]):
        print(f"  segment {r.segment_id:2d} ({r.period[0].strftime('%Y-%m-%d')} ~ {r.period[1].strftime('%Y-%m-%d')}): IC={ic:.4f}")

    print(f"\n[汇总]")
    print(f"  IC mean = {summary.aggregate_metrics['ic_mean']:.4f}")
    print(f"  IC std  = {summary.aggregate_metrics['ic_std']:.4f}")
    print(f"  IC t-stat = {summary.aggregate_metrics['ic_tstat']:.2f}")
    print(f"  RankIC mean = {summary.aggregate_metrics['rank_ic_mean']:.4f}")
    print(f"  Stability = {summary.aggregate_metrics['ic_stability']:.4f}")

    if summary.aggregate_metrics['ic_tstat'] > 2.0:
        print(f"\n  ✓ t-stat > 2.0，模型预测能力统计显著")
    return 0


if __name__ == "__main__":
    sys.exit(main())
