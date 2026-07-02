"""
Walk-Forward 滚动验证框架

借鉴自：
- Microsoft Qlib: qlib.contrib.rolling.base.RollingGen / qlib.contrib.rolling.model.RollingModel
- AKQuant: akquant.backtest.WalkForward

设计目标：
- 在 A 股回测中常用「训练 → 验证 → 测试」滑动窗口，避免前视偏差。
- 提供可重用的 RollingSplit：按时间窗口切分 train/valid/test，支持「扩展窗口」与「滚动窗口」两种模式。
- 与 jingni-trader 的 strategy-model-engine / backtest-engine 解耦：仅产出 (train_idx, valid_idx, test_idx) 列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["RollingSplit", "WalkForwardFold", "WalkForwardRunner"]


# ---------------------------------------------------------------------------
# Fold 数据结构
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """单次滚动窗口。"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp          # 训练集右开区间
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray = field(default=None)
    valid_idx: np.ndarray = field(default=None)
    test_idx: np.ndarray = field(default=None)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class RollingSplit:
    """滚动窗口切分器。

    Parameters
    ----------
    train_period : int
        训练集长度（交易日）。
    valid_period : int
        验证集长度（交易日）。
    test_period : int
        测试集长度（交易日）。
    expanding : bool
        - False (rolling):  训练集固定长度 train_period，窗口滑动。
        - True  (expanding): 训练集从 0 到当前 fold 的开始，长度递增。
    min_train_size : int
        训练集最小长度（少于则跳过）。
    step : int
        每次窗口向前推进多少个交易日。
    """

    def __init__(
        self,
        train_period: int = 240,
        valid_period: int = 60,
        test_period: int = 60,
        expanding: bool = False,
        min_train_size: int = 120,
        step: int = 20,
    ) -> None:
        if train_period <= 0 or valid_period <= 0 or test_period <= 0:
            raise ValueError("train/valid/test period 必须为正整数")
        if step <= 0:
            raise ValueError("step 必须为正整数")
        if min_train_size > train_period:
            raise ValueError("min_train_size 不能超过 train_period")
        self.train_period = train_period
        self.valid_period = valid_period
        self.test_period = test_period
        self.expanding = expanding
        self.min_train_size = min_train_size
        self.step = step

    # ------------------------------------------------------------------
    def split(
        self, dates: pd.DatetimeIndex | pd.Series
    ) -> List[WalkForwardFold]:
        """生成所有 fold。

        Parameters
        ----------
        dates : pd.DatetimeIndex | pd.Series
            按时间排序的交易日序列。
        """
        if isinstance(dates, pd.Series):
            dates = pd.DatetimeIndex(pd.unique(dates))
        else:
            dates = pd.DatetimeIndex(dates).sort_values()
        if dates.has_duplicates:
            dates = dates.drop_duplicates().sort_values()

        n = len(dates)
        window = self.train_period + self.valid_period + self.test_period
        if n < window:
            raise ValueError(
                f"日期序列长度 {n} 小于最小窗口 {window}，无法生成 fold"
            )

        folds: List[WalkForwardFold] = []
        fold_id = 0
        # 滑窗起点
        i = 0
        while i + window <= n:
            train_start_idx = 0 if self.expanding else i
            train_end_idx = i + self.train_period
            valid_start_idx = train_end_idx
            valid_end_idx = valid_start_idx + self.valid_period
            test_start_idx = valid_end_idx
            test_end_idx = test_start_idx + self.test_period

            if (train_end_idx - train_start_idx) < self.min_train_size:
                i += self.step
                continue

            train_idx = np.arange(train_start_idx, train_end_idx)
            valid_idx = np.arange(valid_start_idx, valid_end_idx)
            test_idx = np.arange(test_start_idx, test_end_idx)

            folds.append(
                WalkForwardFold(
                    fold_id=fold_id,
                    train_start=dates[train_start_idx],
                    train_end=dates[train_end_idx - 1],
                    valid_start=dates[valid_start_idx],
                    valid_end=dates[valid_end_idx - 1],
                    test_start=dates[test_start_idx],
                    test_end=dates[test_end_idx - 1],
                    train_idx=train_idx,
                    valid_idx=valid_idx,
                    test_idx=test_idx,
                )
            )
            fold_id += 1
            i += self.step

        return folds

    # ------------------------------------------------------------------
    def iter_splits(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WalkForwardFold]]:
        """直接对 DataFrame 进行切分，yield (train, valid, test, fold_meta)。"""
        dates = pd.DatetimeIndex(df[date_col].unique()).sort_values()
        for fold in self.split(dates):
            train = df.iloc[fold.train_idx]
            valid = df.iloc[fold.valid_idx]
            test = df.iloc[fold.test_idx]
            yield train, valid, test, fold


# ---------------------------------------------------------------------------
# Generic Runner
# ---------------------------------------------------------------------------


class WalkForwardRunner:
    """Walk-Forward 执行器。

    接收一个 `fit_fn(train) -> fitted_model` 和 `evaluate_fn(model, data) -> dict`，
    对每个 fold 执行训练-验证-测试并收集结果。
    """

    def __init__(self, splitter: RollingSplit) -> None:
        self.splitter = splitter

    def run(
        self,
        df: pd.DataFrame,
        fit_fn,
        evaluate_fn,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """运行 walk-forward。

        Parameters
        ----------
        df : pd.DataFrame
            完整数据，按 date 排序。
        fit_fn : callable
            接受 (train_df, valid_df) -> model  （提供验证集便于 early-stop）
        evaluate_fn : callable
            接受 (model, test_df) -> dict
        date_col : str
            时间列名。

        Returns
        -------
        pd.DataFrame
            每个 fold 的评估结果。
        """
        records = []
        for train, valid, test, fold in self.splitter.iter_splits(df, date_col):
            model = fit_fn(train, valid)
            test_metrics = evaluate_fn(model, test)
            test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
            records.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "valid_start": fold.valid_start,
                    "valid_end": fold.valid_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    **test_metrics,
                }
            )
        return pd.DataFrame(records)
