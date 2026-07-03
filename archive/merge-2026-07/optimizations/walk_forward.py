"""
Walk-Forward 滚动验证 + Purged K-Fold 分割（优化验证版）

借鉴来源：
- AKQuant: 内置 Walk-forward Validation 框架
- Microsoft Qlib: RollingDataHandler + PurgedKFold
- Marcos López de Prado: 《Advances in Financial Machine Learning》第7章 Cross-Validation in Finance

针对 jingni-trader skills/strategy-model-engine/engine.py 的优化点：
1. 原实现 purged_group_ts_split 的 purge 逻辑用 timedelta(days=PURGE_GAP_DAYS)，
   但交易日与日历日不一致，应用交易日索引计算 gap
2. 原实现 train 方法中 train_mask = ~X.index.isin(test_dates.index) 有 bug：
   X.index 是行号而非日期，isin(test_dates.index) 永远为空，导致全部数据进训练集
3. 原实现无真正的 walk-forward（滚动训练 + 滚动测试），只有单次 train/test split
4. 原实现无 embargo（测试集后的隔离期），易造成标签泄漏

本模块仅用于性能/正确性对比验证，不修改 main 分支代码。
"""
from __future__ import annotations

from typing import List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 优化版：基于交易日的 Purged Time Series Split
# ----------------------------------------------------------------------

def purged_ts_split(
    dates: pd.Series,
    n_splits: int = 5,
    purge_bars: int = 5,
    embargo_bars: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    基于交易日的 Purged Time Series Split。

    相比原实现 purged_group_ts_split 的改进：
    1. 用交易日位置（而非 timedelta 日历日）计算 purge/embargo gap，
       避免周末/假期造成的 gap 不准确
    2. 增加 embargo：每个测试集后额外隔离 embargo_bars 个交易日，
       防止前瞻收益标签泄漏到下一折训练集
    3. 返回的是基于 dates 的位置索引，与 X/y 的行索引对齐

    参数:
        dates: 每个样本对应的日期（pd.Series，与 X 同长度同索引）
        n_splits: 折数
        purge_bars: 训练集尾部与测试集之间的隔离交易日数（消除标签泄漏）
        embargo_bars: 测试集之后的隔离交易日数（消除下一折的标签泄漏）

    返回:
        [(train_idx, test_idx), ...]，索引为 dates 的行位置
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    if n_dates < n_splits + 1:
        return []

    # 每个日期映射到其样本行索引
    date_to_pos: dict = {}
    for pos, d in enumerate(dates.values):
        date_to_pos.setdefault(d, []).append(pos)

    test_size = n_dates // (n_splits + 1)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_splits):
        # 测试集日期区间
        test_start = (i + 1) * test_size
        test_end = min(test_start + test_size, n_dates)
        if test_start >= n_dates:
            break

        test_dates = unique_dates[test_start:test_end]

        # 训练集日期区间：test_start 之前，但要 purge 掉尾部 purge_bars 个交易日
        train_end = max(test_start - purge_bars, 0)
        train_dates = unique_dates[:train_end]

        # embargo：下一折训练集要跳过 test_end 后 embargo_bars 个交易日
        # 当前折训练集不受影响，但若 purge_bars 不足覆盖前瞻标签，embargo 在下一折生效

        train_idx = np.concatenate([date_to_pos[d] for d in train_dates if d in date_to_pos])
        test_idx = np.concatenate([date_to_pos[d] for d in test_dates if d in date_to_pos])

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def walk_forward_splits(
    dates: pd.Series,
    train_window: int = 504,    # 约 2 年交易日
    test_window: int = 63,      # 约 3 个月交易日
    step: Optional[int] = None, # 滚动步长，默认 = test_window
    purge_bars: int = 5,
    embargo_bars: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-Forward 滚动分割（借鉴 AKQuant Walk-forward Validation）。

    相比原实现单次 train/test split 的改进：
    - 真正的滚动训练：每次用过去 train_window 个交易日训练，预测未来 test_window 个交易日
    - 滚动 step 步长前进，生成多个 (train, test) 折
    - 每折之间有 purge + embargo 隔离，防止标签泄漏

    参数:
        dates: 样本日期序列
        train_window: 训练集交易日数
        test_window: 测试集交易日数
        step: 滚动步长（默认等于 test_window，即不重叠）
        purge_bars: 训练尾与测试头之间的隔离交易日数
        embargo_bars: 测试尾与下一折训练头之间的隔离交易日数
    """
    if step is None:
        step = test_window

    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    if n_dates < train_window + test_window + purge_bars:
        return []

    date_to_pos: dict = {}
    for pos, d in enumerate(dates.values):
        date_to_pos.setdefault(d, []).append(pos)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while start + train_window + purge_bars + test_window <= n_dates:
        train_dates = unique_dates[start : start + train_window]
        test_start = start + train_window + purge_bars
        test_dates = unique_dates[test_start : test_start + test_window]

        train_idx = np.concatenate([date_to_pos[d] for d in train_dates if d in date_to_pos])
        test_idx = np.concatenate([date_to_pos[d] for d in test_dates if d in date_to_pos])

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        start += step
        # embargo：下一折训练集起始要跳过 embargo
        # 通过 step >= test_window + embargo_bars 保证
        if step < test_window + embargo_bars:
            start += embargo_bars

    return splits


# ----------------------------------------------------------------------
# Walk-Forward 训练器（验证用，使用 sklearn dummy 模型）
# ----------------------------------------------------------------------

def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    model_factory,
    train_window: int = 252,
    test_window: int = 63,
    purge_bars: int = 5,
) -> Tuple[pd.Series, List[Dict]]:
    """
    Walk-Forward 滚动训练 + 预测。

    每折用 train_window 个交易日训练，预测后续 test_window 个交易日，
    将所有折的 out-of-sample 预测拼接为完整序列。

    参数:
        X: 特征矩阵
        y: 标签
        dates: 每个样本的日期
        model_factory: 无参 callable，每次调用返回新模型实例
        train_window: 训练集交易日数
        test_window: 测试集交易日数
        purge_bars: 隔离交易日数

    返回:
        (predictions, fold_info)
        predictions: 与 X 同索引的 out-of-sample 预测序列
        fold_info: 每折的训练/测试日期范围、样本数
    """
    splits = walk_forward_splits(
        dates, train_window=train_window, test_window=test_window,
        purge_bars=purge_bars,
    )

    if not splits:
        return pd.Series(dtype=float, index=X.index), []

    predictions = pd.Series(np.nan, index=X.index, dtype=float)
    fold_info = []

    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        model = model_factory()
        model.fit(X_train.values, y_train.values)
        pred = model.predict(X_test.values)

        predictions.iloc[test_idx] = pred
        fold_info.append({
            "fold": i,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_date_range": f"{dates.iloc[train_idx].min()} ~ {dates.iloc[train_idx].max()}",
            "test_date_range": f"{dates.iloc[test_idx].min()} ~ {dates.iloc[test_idx].max()}",
        })

    return predictions, fold_info


# ----------------------------------------------------------------------
# 基准实现（复刻原 strategy-model-engine/engine.py 的逻辑，含 bug）
# ----------------------------------------------------------------------

def purged_group_ts_split_baseline(
    dates: pd.Series,
    n_splits: int = 5,
    purge_gap_days: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """复刻原 purged_group_ts_split 逻辑（含 timedelta bug，用于对比）。"""
    from datetime import timedelta

    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)

    splits = []
    test_size = n_dates // (n_splits + 1)

    for i in range(n_splits):
        train_end_idx = n_dates - (n_splits - i) * test_size
        val_start_idx = train_end_idx + 1
        val_end_idx = min(val_start_idx + test_size, n_dates)

        if val_start_idx >= n_dates:
            break

        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]

        if purge_gap_days > 0:
            purge_date = unique_dates[train_end_idx] - timedelta(days=purge_gap_days)
            train_dates = [d for d in train_dates if d <= purge_date]

        train_idx = dates[dates.isin(train_dates)].index.values
        val_idx = dates[dates.isin(val_dates)].index.values

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


def train_baseline_bug(
    X: pd.DataFrame,
    y: pd.Series,
    test_dates: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    复刻原 train 方法中的索引 bug：
        train_mask = ~X.index.isin(test_dates.index)
    X.index 是行号（0,1,2,...），test_dates.index 也是行号，
    isin 会把行号匹配上，导致训练集/测试集划分错误。

    本函数仅用于演示 bug，返回划分后的 X_train, X_test, y_train, y_test。
    """
    train_mask = ~X.index.isin(test_dates.index)
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_test = X.loc[~train_mask]
    y_test = y.loc[~train_mask]
    return X_train, X_test, y_train, y_test
