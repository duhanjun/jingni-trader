"""
测试 3: Point-in-Time 数据守卫 + Purged K-Fold CV

验证目标：
1. PointInTimeGuard 能正确注册和校验特征
2. PurgedKFoldTimeSeriesCV 严格按时间顺序切分，且包含 purge gap 和 embargo
3. WalkForwardValidator 输出合理的训练/验证窗口
4. PurgedSplit 的元数据正确
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import timedelta

from pit_guard import (
    PointInTimeGuard, PurgedKFoldTimeSeriesCV, PurgedSplit,
    WalkForwardValidator, LeakageDetector
)
from synthetic_data import generate_synthetic_data


def test_pit_guard_basic():
    """PointInTimeGuard 基础测试"""
    print("=" * 60)
    print("测试 3.1: PointInTimeGuard 基础")
    print("=" * 60)
    guard = PointInTimeGuard()
    guard.register_feature("ret_20d", max_lookback_days=20, description="20日收益")
    guard.register_feature("alpha_score", max_lookback_days=1, description="综合 alpha")

    data, factors = generate_synthetic_data(n_stocks=20, n_days=100, seed=1)

    # 校验正常特征
    r1 = guard.validate_data(factors, data, "ret_20d")
    print(f"  ret_20d 校验: valid={r1['valid']}, warnings={r1['warnings']}")
    assert r1["valid"], "合法特征应通过校验"
    print("  ✓ 合法特征通过校验")

    # 校验未注册的特征
    r2 = guard.validate_data(factors, data, "unknown_feature")
    assert r2["valid"] is True
    assert "未注册" in r2.get("note", "")
    print("  ✓ 未注册特征优雅跳过")
    return True


def test_pit_guard_detects_leakage():
    """PointInTimeGuard 检测泄露"""
    print("=" * 60)
    print("测试 3.2: PIT 检测泄露")
    print("=" * 60)
    guard = PointInTimeGuard()
    guard.register_feature("alpha_score", max_lookback_days=1)

    # 先生成 100 天数据，再额外补 30 天"未来"原始数据
    data, factors = generate_synthetic_data(n_stocks=5, n_days=100, seed=1)
    last_date = pd.to_datetime(factors["date"]).max()
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=30)
    future_rows = []
    for code in data["code"].unique()[:5]:
        for fd in future_dates:
            future_rows.append({
                "date": fd, "code": code,
                "alpha_score": 0.5,  # 模拟使用未来信息生成的 alpha
            })
    bad_factors = pd.concat([factors, pd.DataFrame(future_rows)], ignore_index=True)

    r = guard.validate_data(bad_factors, data, "alpha_score")
    print(f"  泄露特征校验: valid={r['valid']}, warnings数={len(r['warnings'])}")
    for w in r["warnings"][:3]:
        print(f"    - {w[:100]}")
    assert r["valid"] is False, "含未来数据的特征应被标记为 invalid"
    assert any("原始数据中找不到" in w for w in r["warnings"]), "应包含未来数据警告"
    print("  ✓ 泄露特征被正确检测")
    return True


def test_purged_kfold_split():
    """Purged K-Fold 切分测试"""
    print("=" * 60)
    print("测试 3.3: Purged K-Fold 切分")
    print("=" * 60)
    n_days = 400
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    samples = pd.Series(np.tile(dates, 50))  # 50 只股票 × 400 天

    cv = PurgedKFoldTimeSeriesCV(n_splits=5, purge_days=10, embargo_days=5, min_train_size=120)
    splits = cv.split(samples)
    print(f"  生成 {len(splits)} 个 purged splits")
    assert len(splits) >= 3, f"应至少 3 个 splits，实际 {len(splits)}"

    # 校验每个 split 的属性
    for i, sp in enumerate(splits):
        # 1. 时间顺序: train_end < val_start
        assert sp.train_end < sp.val_start, f"split {i}: train_end >= val_start"
        # 2. 训练集中不包含验证集（无重叠）
        assert sp.train_end < sp.val_start, f"split {i}: 训练和验证有重叠"
        # 3. purge gap 应存在
        gap_days = (sp.val_start - sp.train_end).days
        print(f"  split {i}: train=[{sp.train_start.date()},{sp.train_end.date()}], "
              f"val=[{sp.val_start.date()},{sp.val_end.date()}], "
              f"purge_gap={gap_days}天, embargo={sp.embargo_days}天")
        assert gap_days >= cv.purge_days, f"split {i}: purge gap 应 >= {cv.purge_days}天"

    print("  ✓ 所有 splits 时间顺序、purge gap 正确")
    return True


def test_walk_forward():
    """Walk-Forward 切分测试"""
    print("=" * 60)
    print("测试 3.4: Walk-Forward 切分")
    print("=" * 60)
    n_days = 500
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    samples = pd.Series(dates)

    wf = WalkForwardValidator(train_window=252, val_window=60, step=60)
    results = wf.split_with_test(samples)
    print(f"  生成 {len(results)} 个 walk-forward 窗口")
    assert len(results) >= 2

    for i, win in enumerate(results):
        tr_start, tr_end = win["train"]
        va_start, va_end = win["val"]
        train_days = (tr_end - tr_start).days
        val_days = (va_end - va_start).days
        print(f"  window {i}: train={train_days}天, val={val_days}天, "
              f"step={tr_start.date()}→{va_end.date()}")
        assert train_days >= 252 * 0.9, f"训练窗口约 1 年，实际 {train_days}天"
        assert tr_end < va_start, "训练结束 < 验证开始"

    print("  ✓ Walk-Forward 切分正确")
    return True


def test_purged_kfold_indices():
    """验证 split_indices 输出的索引正确"""
    print("=" * 60)
    print("测试 3.5: split_indices 索引正确性")
    print("=" * 60)
    n_days = 300
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    n_stocks = 10
    # 构造一个 (n_stocks, n_days) 的样本
    samples = pd.Series(np.tile(dates, n_stocks))
    samples.index = range(len(samples))

    cv = PurgedKFoldTimeSeriesCV(n_splits=3, purge_days=5, embargo_days=5, min_train_size=100)
    indices_list = cv.split_indices(samples)
    assert len(indices_list) >= 1, "应至少生成 1 个 split"

    for i, (train_idx, val_idx) in enumerate(indices_list):
        # 校验索引不重叠
        assert len(set(train_idx) & set(val_idx)) == 0, f"split {i}: train/val 索引重叠"
        # 校验 val_idx 中的日期都 > train_idx 中的最大日期
        train_dates = samples.iloc[train_idx]
        val_dates = samples.iloc[val_idx]
        assert val_dates.min() > train_dates.max(), (
            f"split {i}: 验证集最早日期 {val_dates.min()} 应晚于训练集最晚 {train_dates.max()}"
        )
        print(f"  split {i}: train={len(train_idx)} 样本, val={len(val_idx)} 样本, "
              f"val_first={val_dates.min().date()}, train_last={train_dates.max().date()}")

    print("  ✓ 索引切分无重叠且时间严格递增")
    return True


def test_leakage_detector():
    """泄露检测器"""
    print("=" * 60)
    print("测试 3.6: 泄露检测器")
    print("=" * 60)
    from sklearn.linear_model import LinearRegression

    n = 200
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
    })
    # 正常情况：y 与 X 有关
    y = 2 * X["f1"] + X["f2"] + np.random.randn(n) * 0.1
    model = LinearRegression().fit(X, y)

    def metric_fn(m, x, y_true):
        return m.score(x, y_true)

    r = LeakageDetector.shuffle_y_test(
        model, X, pd.Series(y), metric_fn, n_shuffles=3, random_state=42
    )
    print(f"  原始 R²: {r['original_metric']:.3f}")
    print(f"  打乱后 R²: {r['shuffled_mean']:.3f} ± {r['shuffled_std']:.3f}")
    print(f"  提升比例: {r['improvement_pct']:.1f}%")
    print(f"  泄露警告: {r['leakage_warning']}")
    # 正常模型：打乱后 R² 应大幅下降（接近 0）
    assert r["shuffled_mean"] < 0.5, "正常模型打乱后 R² 应大幅下降"
    assert r["leakage_warning"] is False, "正常模型不应触发泄露警告"
    print("  ✓ 正常模型未触发泄露警告")
    return True


def main():
    print("\n" + "=" * 60)
    print("【测试 3: PIT Guard + Purged CV 验证】")
    print("=" * 60 + "\n")

    results = {}
    try:
        test_pit_guard_basic()
        results["pit_basic"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["pit_basic"] = False

    try:
        test_pit_guard_detects_leakage()
        results["pit_leakage"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["pit_leakage"] = False

    try:
        test_purged_kfold_split()
        results["purged_split"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["purged_split"] = False

    try:
        test_walk_forward()
        results["walk_forward"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["walk_forward"] = False

    try:
        test_purged_kfold_indices()
        results["purged_indices"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["purged_indices"] = False

    try:
        test_leakage_detector()
        results["leakage_detector"] = True
    except AssertionError as e:
        print(f"  ✗ 失败: {e}")
        results["leakage_detector"] = False

    print("\n" + "=" * 60)
    print("测试 3 总结")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k}: {'通过' if v else '失败'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
