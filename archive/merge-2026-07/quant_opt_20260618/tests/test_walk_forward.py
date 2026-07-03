"""
Walk-Forward 滚动训练框架测试
"""
import numpy as np
import pandas as pd
import pytest

from quant_opt_20260618.walk_forward import (
    WalkForwardConfig,
    make_walk_forward_splits,
    walk_forward_train_predict,
    aggregate_wf_metrics,
)
from quant_opt_20260618.tests.fixtures import make_synthetic_ashare_data


def _make_panel(n_stocks=10, n_days=800, seed=0):
    """构造一个 (code, date, features..., label) 的面板数据"""
    df = make_synthetic_ashare_data(n_stocks=n_stocks, n_days=n_days, seed=seed)
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    # 简单 label：次日收益
    df["ret_1d"] = df.groupby("code")["close"].pct_change().shift(-1)
    df["feat_ma5"] = df.groupby("code")["close"].transform(lambda s: s.rolling(5).mean())
    df["feat_ma20"] = df.groupby("code")["close"].transform(lambda s: s.rolling(20).mean())
    df = df.dropna()
    return df


def test_make_splits_basic():
    dates = pd.Series(pd.bdate_range("2023-01-01", periods=1000))
    cfg = WalkForwardConfig(
        train_window_days=300, val_window_days=100, test_window_days=100,
        step_days=100, purge_days=5, embargo_days=5,
    )
    splits = make_walk_forward_splits(dates, cfg)
    assert len(splits) >= 3, f"expected >=3 folds, got {len(splits)}"
    for sp in splits:
        assert sp["train_start"] < sp["train_end"] < sp["val_start"] <= sp["val_end"] < sp["test_start"] <= sp["test_end"]
        # 窗口大小
        assert (sp["val_end"] - sp["val_start"]).days >= cfg.val_window_days - 2


def test_make_splits_too_short():
    dates = pd.Series(pd.bdate_range("2023-01-01", periods=200))
    cfg = WalkForwardConfig(
        train_window_days=300, val_window_days=100, test_window_days=100,
    )
    assert make_walk_forward_splits(dates, cfg) == []


def _fit_predict_linear(X_train, y_train, X_val, y_val, X_test):
    """最小线性回归实现，避免引入 sklearn 依赖"""
    Xt = np.hstack([np.ones((len(X_train), 1)), X_train.values])
    Xv = np.hstack([np.ones((len(X_test), 1)), X_test.values])
    coef, *_ = np.linalg.lstsq(Xt, y_train.values, rcond=None)
    return Xv @ coef


def test_walk_forward_smoke():
    df = _make_panel(n_stocks=8, n_days=900, seed=2)
    X = df[["feat_ma5", "feat_ma20"]]
    y = df["ret_1d"]
    dates = df["date"]
    cfg = WalkForwardConfig(
        train_window_days=300, val_window_days=100, test_window_days=100,
        step_days=100, purge_days=5, embargo_days=5,
    )
    oos, results = walk_forward_train_predict(X, y, dates, _fit_predict_linear, cfg)
    assert len(results) >= 3
    assert oos.shape[0] > 0
    assert "pred" in oos.columns
    # OOS 预测覆盖率应合理
    coverage = oos["pred"].notna().mean()
    assert 0.5 < coverage <= 1.0, f"OOS 覆盖率异常: {coverage}"


def test_aggregate_wf_metrics():
    df = _make_panel(n_stocks=6, n_days=900, seed=4)
    X = df[["feat_ma5", "feat_ma20"]]
    y = df["ret_1d"]
    dates = df["date"]
    cfg = WalkForwardConfig(
        train_window_days=300, val_window_days=100, test_window_days=100,
        step_days=100, purge_days=5, embargo_days=5,
    )
    _, results = walk_forward_train_predict(X, y, dates, _fit_predict_linear, cfg)
    summary = aggregate_wf_metrics(results)
    assert "n_folds" in summary
    assert "mean_ic" in summary
    assert summary["n_folds"] >= 1
    # ICIR 应当是有限数
    assert np.isfinite(summary["icir"])
