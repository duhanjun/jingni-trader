"""
Purged CV 测试
"""
import numpy as np
import pandas as pd
import pytest

from validation.purged_cv import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    WalkForwardSplitter,
    ic_time_series_split,
)


@pytest.fixture
def datetime_index():
    return pd.date_range("2020-01-01", periods=300, freq="B")


def test_purged_kfold_n_splits(datetime_index):
    X = pd.DataFrame({"x": range(300)}, index=datetime_index)
    splitter = PurgedKFold(n_splits=5)
    assert splitter.get_n_splits() == 5
    splits = list(splitter.split(X))
    assert len(splits) == 5
    for i, s in enumerate(splits):
        assert s.fold_id == i
        # 训练+测试 = 总数
        assert len(s.train_idx) + len(s.test_idx) == 300


def test_purged_kfold_no_overlap(datetime_index):
    X = pd.DataFrame({"x": range(300)}, index=datetime_index)
    splitter = PurgedKFold(n_splits=5)
    for s in splitter.split(X):
        train_set = set(s.train_idx.tolist())
        test_set = set(s.test_idx.tolist())
        assert train_set.isdisjoint(test_set)


def test_purged_kfold_with_purge_embargo(datetime_index):
    X = pd.DataFrame({"x": range(300)}, index=datetime_index)
    splitter = PurgedKFold(n_splits=5, purge_td="10D", embargo_td="10D")
    for s in splitter.split(X):
        # purge 之后训练样本 < 总样本
        assert len(s.train_idx) < 300 - len(s.test_idx)


def test_walk_forward_basic():
    splitter = WalkForwardSplitter(train_size=100, val_size=20, test_size=30, step_size=30)
    X = pd.DataFrame({"x": range(200)})
    n_splits = splitter.get_n_splits(200)
    # 1 + (200 - 150) / 30 = 1 + 1 = 2
    assert n_splits == 2
    splits = list(splitter.split(X))
    assert len(splits) == 2
    s = splits[0]
    assert len(s.train_idx) == 100
    assert len(s.val_idx) == 20
    assert len(s.test_idx) == 30
    # 训练在前, 验证居中, 测试在后
    assert s.train_idx[-1] < s.val_idx[0]
    assert s.val_idx[-1] < s.test_idx[0]


def test_walk_forward_rolling_vs_expanding():
    """rolling 训练窗口固定长度, expanding 训练窗口随 fold 增长"""
    X = pd.DataFrame({"x": range(500)})
    rolling = WalkForwardSplitter(train_size=200, test_size=50, step_size=50, expanding=False)
    expanding = WalkForwardSplitter(train_size=200, test_size=50, step_size=50, expanding=True)
    roll_splits = list(rolling.split(X))
    exp_splits = list(expanding.split(X))
    assert len(roll_splits) == len(exp_splits)
    for i, (r, e) in enumerate(zip(roll_splits, exp_splits)):
        # rolling 训练窗口固定 200 长度, expanding 训练窗口随 fold 增长
        assert len(r.train_idx) == 200
        assert len(e.train_idx) == 200 + i * 50  # expanding 累积
        # rolling 起点 = 当前 cursor, expanding 始终从 0
        assert r.train_idx[0] == i * 50
        assert e.train_idx[0] == 0
        # 两者测试窗口一致
        assert (r.test_idx == e.test_idx).all()


def test_walk_forward_too_short_raises():
    splitter = WalkForwardSplitter(train_size=100, test_size=50)
    X = pd.DataFrame({"x": range(50)})
    splits = list(splitter.split(X))
    assert len(splits) == 0


def test_combinatorial_purged_kfold():
    X = pd.DataFrame({"x": range(60)})
    splitter = CombinatorialPurgedKFold(n_groups=5, n_test_groups=2)
    splits = list(splitter.split(X))
    # C(5,2) = 10
    assert len(splits) == 10
    for s in splits:
        assert len(s.test_idx) == 24  # 5 groups × 12, 2 groups → 24
        assert len(s.train_idx) == 36


def test_ic_time_series_split():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04",
                                 "2020-01-05", "2020-01-06", "2020-01-07", "2020-01-08"]),
        "code": ["A"] * 8,
        "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    splits = list(ic_time_series_split(panel, n_splits=2, min_train_size=2, purge_days=1))
    assert len(splits) == 2
    train_df, _, test_df = splits[0]
    assert len(train_df) > 0 and len(test_df) > 0
    # 训练/测试时间不重叠
    assert not set(train_df["date"]).intersection(set(test_df["date"]))
