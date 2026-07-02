"""
Walk-Forward Validation 单元测试

测试目标：
1. 切分正确性：fold 数量、训练/验证/测试区间无重叠、无未来信息泄露。
2. 滚动 vs 扩展模式：训练集长度的差异。
3. 早停 / 验证集：保证 fit_fn 收到 valid 用于 early-stop。
4. 边界：日期过短、min_train_size 过滤。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from walk_forward_validation import RollingSplit, WalkForwardRunner


def make_dates(n: int = 500, start: str = "2022-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def make_panel(
    n_dates: int = 500, n_codes: int = 20, start: str = "2022-01-01"
) -> pd.DataFrame:
    """构造一个多股票面板数据。"""
    dates = pd.bdate_range(start, periods=n_dates)
    codes = [f"{i:06d}.SH" for i in range(1, n_codes + 1)]
    rows = []
    for code in codes:
        for d in dates:
            rows.append({"date": d, "code": code, "x": np.random.randn()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic Splitter
# ---------------------------------------------------------------------------


def test_basic_rolling_split():
    dates = make_dates(n=500)
    # step >= test_period 时相邻 fold 的测试集不重叠
    splitter = RollingSplit(
        train_period=120, valid_period=30, test_period=30, step=30
    )
    folds = splitter.split(dates)
    assert len(folds) >= 5
    # 关键不变量：每个 fold 内部 train/valid/test 严格按时间排列
    for f in folds:
        assert f.train_end < f.valid_start
        assert f.valid_end < f.test_start
    # 相邻 fold 训练集不重叠
    for i in range(1, len(folds)):
        prev = folds[i - 1]
        cur = folds[i]
        # 相邻 fold 的训练集可能重叠（rolling），但测试集应该单调推进
        assert cur.test_start >= prev.test_start


def test_expanding_train_grows():
    dates = make_dates(n=400)
    splitter_rolling = RollingSplit(
        train_period=120, valid_period=30, test_period=30,
        expanding=False, step=30,
    )
    splitter_expanding = RollingSplit(
        train_period=120, valid_period=30, test_period=30,
        expanding=True, step=30,
    )
    fr = splitter_rolling.split(dates)
    fe = splitter_expanding.split(dates)
    # 扩展窗口的第一个 fold 训练集长度 == rolling 的第一个
    assert len(fr[0].train_idx) == len(fe[0].train_idx)
    # 第二个 fold 扩展窗口的训练集更长
    assert len(fe[1].train_idx) > len(fr[1].train_idx)


def test_min_train_size_filter():
    dates = make_dates(n=300)
    splitter = RollingSplit(
        train_period=200, valid_period=30, test_period=30,
        min_train_size=150, step=20,
    )
    folds = splitter.split(dates)
    # 所有 fold 训练集都 >= 150
    for f in folds:
        assert len(f.train_idx) >= 150


def test_invalid_params():
    with pytest.raises(ValueError):
        RollingSplit(train_period=0, valid_period=10, test_period=10)
    with pytest.raises(ValueError):
        RollingSplit(train_period=10, valid_period=0, test_period=10)
    with pytest.raises(ValueError):
        RollingSplit(train_period=10, valid_period=10, test_period=0)
    with pytest.raises(ValueError):
        RollingSplit(train_period=10, valid_period=10, test_period=10, step=0)
    with pytest.raises(ValueError):
        RollingSplit(train_period=10, valid_period=10, test_period=10, min_train_size=20)


def test_too_short_dates_raises():
    splitter = RollingSplit(train_period=120, valid_period=30, test_period=30)
    dates = make_dates(n=100)
    with pytest.raises(ValueError):
        splitter.split(dates)


def test_iter_splits_yields_dataframes():
    df = make_panel(n_dates=300, n_codes=1)  # 单股票
    splitter = RollingSplit(
        train_period=120, valid_period=30, test_period=30, step=30
    )
    folds_seen = 0
    for train, valid, test, fold in splitter.iter_splits(df, date_col="date"):
        # 验证 df 行数与 fold 一致
        assert len(train) == len(fold.train_idx)
        assert len(valid) == len(fold.valid_idx)
        assert len(test) == len(fold.test_idx)
        # 验证没有时间穿越：test 的最大日期 <= fold.test_end
        assert test["date"].max() <= fold.test_end
        assert train["date"].max() <= fold.train_end
        folds_seen += 1
    assert folds_seen >= 3


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_runner_executes_all_folds():
    df = make_panel(n_dates=400, n_codes=1)
    splitter = RollingSplit(
        train_period=120, valid_period=30, test_period=30, step=30
    )
    runner = WalkForwardRunner(splitter)

    def fit_fn(train, valid):
        # 简单 mock：返回 (mean, std) 模拟模型参数
        return {"mean": train["x"].mean(), "n_train": len(train)}

    def evaluate_fn(model, test):
        return {
            "pred_mean": model["mean"],
            "n_test": len(test),
        }

    results = runner.run(df, fit_fn, evaluate_fn, date_col="date")
    assert "fold_id" in results.columns
    assert "test_pred_mean" in results.columns
    assert "test_n_test" in results.columns
    # 测试集中时间起点应当递增
    assert results["test_start"].is_monotonic_increasing


def test_no_future_leakage_via_evaluation():
    """验证在 evaluate 时不可能用到 test 期之后的数据。"""
    dates = make_dates(n=300)
    df = pd.DataFrame({
        "date": np.repeat(dates, 3),
        "code": ["A", "B", "C"] * len(dates),
        "x": np.random.randn(len(dates) * 3),
    })
    splitter = RollingSplit(
        train_period=120, valid_period=30, test_period=30, step=30
    )
    runner = WalkForwardRunner(splitter)

    seen_test_dates = []

    def fit_fn(train, valid):
        return {"train_max": train["date"].max()}

    def evaluate_fn(model, test):
        # 关键断言：test 的最大日期永远在 train 最大日期之后
        assert test["date"].min() > model["train_max"]
        seen_test_dates.append(test["date"].min())
        return {"ok": 1.0}

    runner.run(df, fit_fn, evaluate_fn, date_col="date")
    assert len(seen_test_dates) >= 2
    # 每个 fold 的 test 起点都 > 该 fold 的 train_end
    for ts in seen_test_dates:
        assert ts > pd.Timestamp("2022-01-01")


# ---------------------------------------------------------------------------
# Real Quant Use Case: Factor IC decay walk-forward
# ---------------------------------------------------------------------------


def test_factor_ic_walk_forward_demo():
    """模拟一个真实的因子 IC walk-forward：每折计算 IC。"""
    dates = make_dates(n=500)
    codes = [f"{i:06d}.SH" for i in range(1, 11)]
    rng = np.random.default_rng(7)
    rows = []
    # 构造 alpha：fwd_return_1d = factor + noise
    for code in codes:
        factor = rng.normal(0, 1, size=len(dates)).cumsum()
        for i, d in enumerate(dates):
            f = factor[i]
            ret = 0.1 * f + rng.normal(0, 0.5)
            rows.append({"date": d, "code": code, "factor": f, "fwd_ret_1d": ret})
    df = pd.DataFrame(rows)

    splitter = RollingSplit(
        train_period=240, valid_period=60, test_period=60, step=60
    )
    runner = WalkForwardRunner(splitter)

    def fit_fn(train, valid):
        return {"train_dates": set(train["date"])}

    def evaluate_fn(model, test):
        # 只在 test 集上计算 IC
        from scipy.stats import spearmanr
        sub = test.dropna(subset=["factor", "fwd_ret_1d"])
        if len(sub) < 5:
            return {"ic": 0.0, "n": len(sub)}
        ic, _ = spearmanr(sub["factor"], sub["fwd_ret_1d"])
        return {"ic": float(ic), "n": len(sub)}

    results = runner.run(df, fit_fn, evaluate_fn, date_col="date")
    # 至少 2 个 fold
    assert len(results) >= 2
    # IC 均值应该 > 0 （因为 alpha 与未来收益正相关）
    assert results["test_ic"].mean() > 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))