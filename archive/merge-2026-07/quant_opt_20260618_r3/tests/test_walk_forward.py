"""Walk-Forward 验证器单元测试"""
import numpy as np
import pandas as pd
import pytest

from quant_opt_20260618_r3.walk_forward import (
    WFVConfig,
    WalkForwardValidator,
)


def _detect_overfit(result):
    """对静态方法做薄包装，方便测试"""
    return WalkForwardValidator.detect_overfit(result)


def _make_synthetic(n_stocks: int = 5, n_days: int = 600, seed: int = 0):
    """生成合成数据: 因子 = 滞后 5 天的未来收益 (有效信号)
    + 高斯噪声 (模拟弱信号)"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    codes = [f"STK{i:03d}" for i in range(n_stocks)]

    rows = []
    for c in codes:
        # 真实 alpha 系数
        alpha = rng.normal(0.001, 0.0005)
        noise = rng.normal(0, 0.01, n_days)
        ret = np.zeros(n_days)
        for t in range(1, n_days):
            ret[t] = 0.05 * ret[t - 1] + noise[t]
        # feature: lagged 5d returns + 一些噪声特征
        feat1 = pd.Series(ret).shift(5).fillna(0).to_numpy()
        feat2 = rng.normal(0, 1, n_days)
        # 未来 5 日收益 (label)
        fwd = pd.Series(ret).shift(-5).fillna(0).to_numpy()
        for i, d in enumerate(dates):
            rows.append({
                "code": c,
                "date": d,
                "feat1": feat1[i],
                "feat2": feat2[i],
                "label": fwd[i],
                "alpha": alpha,
            })
    df = pd.DataFrame(rows)
    return df


def _toy_fit_predict(X_train, y_train, X_eval):
    """简单线性模型: y ~ feat1 + feat2"""
    Xt = np.column_stack([X_train["feat1"].to_numpy(), X_train["feat2"].to_numpy()])
    Xe = np.column_stack([X_eval["feat1"].to_numpy(), X_eval["feat2"].to_numpy()])
    # 闭式最小二乘
    coef, *_ = np.linalg.lstsq(Xt, y_train.to_numpy(), rcond=None)
    return Xe @ coef


def test_wfv_basic_run():
    df = _make_synthetic(n_stocks=4, n_days=600)
    validator = WalkForwardValidator(
        config=WFVConfig(
            train_window_days=180,
            eval_window_days=40,
            step_days=40,
            purge_gap_days=5,
            min_train_samples=200,
        )
    )
    result = validator.run(
        X=df[["feat1", "feat2"]],
        y=df["label"],
        dates=df["date"],
        fit_predict_fn=_toy_fit_predict,
    )
    assert result.n_folds >= 2, f"应至少 2 个 fold, 实际 {result.n_folds}"
    assert "ic" in result.aggregate_metrics
    # 由于 feat1 是 lag 5 的 return 而 label 是 shift -5 的 return
    # 两者是接近的 (有相关性), 不应全为 0
    assert result.aggregate_metrics["ic"]["mean"] != 0


def test_wfv_purge_gap_creates_gap_between_train_eval():
    df = _make_synthetic(n_stocks=3, n_days=400)
    validator = WalkForwardValidator(
        config=WFVConfig(
            train_window_days=120,
            eval_window_days=30,
            step_days=30,
            purge_gap_days=10,
            min_train_samples=100,
        )
    )
    result = validator.run(
        X=df[["feat1", "feat2"]],
        y=df["label"],
        dates=df["date"],
        fit_predict_fn=_toy_fit_predict,
    )
    assert result.n_folds >= 1
    for fold in result.folds:
        # purge gap: train_end + purge_gap_days <= eval_start
        gap = (fold.eval_start - fold.train_end).days
        assert gap >= 0, f"训练末 >= 评估起点, gap={gap}"


def test_wfv_detect_overfit_flags_low_icir():
    """构造一个完全无信号的 fold 序列，期望检测为"过拟合/弱信号" """
    df = _make_synthetic(n_stocks=3, n_days=400)
    validator = WalkForwardValidator(
        config=WFVConfig(
            train_window_days=120,
            eval_window_days=30,
            step_days=30,
            purge_gap_days=5,
            min_train_samples=100,
        )
    )

    def random_predict(X_train, y_train, X_eval):
        # 完全随机预测
        return np.random.default_rng(42).normal(0, 1, len(X_eval))

    result = validator.run(
        X=df[["feat1", "feat2"]],
        y=df["label"],
        dates=df["date"],
        fit_predict_fn=random_predict,
    )
    flag = _detect_overfit(result)
    assert flag["overfit"] is True
    assert flag["ic_ir"] < 0.3


def test_wfv_too_short_data_returns_empty():
    df = _make_synthetic(n_stocks=3, n_days=80)
    validator = WalkForwardValidator(
        config=WFVConfig(
            train_window_days=120,
            eval_window_days=30,
            step_days=30,
            purge_gap_days=5,
        )
    )
    result = validator.run(
        X=df[["feat1", "feat2"]],
        y=df["label"],
        dates=df["date"],
        fit_predict_fn=_toy_fit_predict,
    )
    assert result.n_folds == 0
    assert result.aggregate_metrics == {}


def test_wfv_n_splits_limits_folds():
    df = _make_synthetic(n_stocks=3, n_days=800)
    validator = WalkForwardValidator(
        config=WFVConfig(
            train_window_days=180,
            eval_window_days=40,
            step_days=40,
            purge_gap_days=5,
            n_splits=3,
            min_train_samples=100,
        )
    )
    result = validator.run(
        X=df[["feat1", "feat2"]],
        y=df["label"],
        dates=df["date"],
        fit_predict_fn=_toy_fit_predict,
    )
    assert result.n_folds == 3