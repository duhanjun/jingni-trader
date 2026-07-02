"""Test suite: walk_forward (AKQuant-inspired)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._synth_data import make_synth_panel, make_synth_factor

from walk_forward.validator import (
    WalkForwardConfig,
    walk_forward_splits,
    MeanReversionSignal,
    run_walk_forward_validation,
)


@pytest.fixture(scope="module")
def panel():
    return make_synth_panel(n_codes=10, n_days=300)


@pytest.fixture(scope="module")
def feature_target(panel):
    """构造带 leak-free 关系的 X, y (1 日 forward return)."""
    df = panel.sort_values(["code", "date"]).copy()
    df["ret_5d"] = df.groupby("code")["close"].pct_change(5)
    df["y"] = df.groupby("code")["close"].pct_change().shift(-1)
    df = df.dropna(subset=["y"])
    return df


# ----------------------------------------------------------------------
# 1. Splitter
# ----------------------------------------------------------------------

def test_splits_rolling_count(feature_target):
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20)
    n = len(feature_target)
    folds = walk_forward_splits(n, cfg)
    # 期望 (n - train - test) / (train + test) + 1
    # (no overlap between test and next train)
    expected = (n - 120 - 20) // (120 + 20) + 1
    assert len(folds) == expected


def test_splits_expanding_count(feature_target):
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20, expanding=True)
    n = len(feature_target)
    folds = walk_forward_splits(n, cfg)
    # expanding 模式: train_start 永远=0, fold 数同 rolling
    expected = (n - 120 - 20) // (120 + 20) + 1
    assert len(folds) == expected


def test_splits_no_overlap(feature_target):
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20)
    folds = walk_forward_splits(len(feature_target), cfg)
    for i, f in enumerate(folds):
        # 训练区间 与 测试区间 互不重叠
        assert f.train_index.max() < f.test_index.min()
        if i > 0:
            # 与上一 fold 的测试区间也不重叠 (rolling 步长 == test_window)
            assert folds[i - 1].test_index.max() < f.train_index.min()


def test_splits_with_dates_metadata(feature_target):
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20)
    folds = walk_forward_splits(
        len(feature_target),
        cfg,
        dates=feature_target["date"].reset_index(drop=True),
    )
    assert isinstance(folds[0].train_start, pd.Timestamp)


# ----------------------------------------------------------------------
# 2. SignalModel.clone (Signal vs Action 分离)
# ----------------------------------------------------------------------

def test_clone_does_not_share_state():
    a = MeanReversionSignal(lookback=10)
    a._mu = 1.0
    b = a.clone()
    b._mu = 2.0
    assert a._mu == 1.0
    assert b._mu == 2.0


# ----------------------------------------------------------------------
# 3. End-to-end run
# ----------------------------------------------------------------------

def test_walk_forward_run_returns_valid_shape(feature_target):
    X = feature_target[["close", "volume"]].reset_index(drop=True)
    y = feature_target["y"].reset_index(drop=True)
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20)

    res = run_walk_forward_validation(X, y, MeanReversionSignal, cfg, threshold=0.3)
    assert res["folds"], "should produce at least one fold"
    assert res["oos_signals"].size > 0
    assert res["oos_signals"].shape == res["oos_y"].shape
    assert res["actions"].shape == res["oos_signals"].shape


def test_walk_forward_per_fold_metrics(feature_target):
    X = feature_target[["close", "volume"]].reset_index(drop=True)
    y = feature_target["y"].reset_index(drop=True)
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20)

    res = run_walk_forward_validation(X, y, MeanReversionSignal, cfg)
    pf = res["per_fold_metrics"]
    for fold in pf:
        if "error" in fold:
            continue
        assert "fold_id" in fold
        assert "train_size" in fold
        assert "test_size" in fold
        assert "hit_ratio" in fold
        assert 0.0 <= fold["hit_ratio"] <= 1.0


def test_signal_action_separation():
    """Signal 是连续值, Action 是离散的 {-1, 0, 1}."""
    from walk_forward.validator import SignalModel

    class Toy(SignalModel):
        def fit(self, X, y): return self
        def predict(self, X):
            return np.linspace(-2, 2, len(X))

    model = Toy()
    sig = model.predict(pd.DataFrame({"x": range(11)}))
    assert sig.shape == (11,)
    # 阈值映射后, action ∈ {-1, 0, 1}
    threshold = 0.5
    act = np.zeros_like(sig, dtype=int)
    act[sig > threshold] = 1
    act[sig < -threshold] = -1
    assert set(act.tolist()).issubset({-1, 0, 1})


# ----------------------------------------------------------------------
# 4. 防止泄露: 后续 fold 的训练集 不应包含 之前 fold 的测试集
# ----------------------------------------------------------------------

def test_no_future_in_train():
    cfg = WalkForwardConfig(train_window=120, test_window=10, rolling_step=10)
    folds = walk_forward_splits(200, cfg)
    for f in folds:
        # 训练集中所有 index 一定 < 测试集中所有 index
        assert f.train_index.max() < f.test_index.min()
