"""
Purged K-Fold + Walk-Forward 时序交叉验证（借鉴来源: VectorBT / AKQuant / Advances in Financial Machine Learning）

设计动机
========
jingni-trader 当前状态:
    - config.py 中声明了 `PURGE_GAP_DAYS = 5` / `TRAIN_WINDOW_MONTHS = 36`
      等时序 CV 参数,但实际模型训练阶段无对应实现
    - skills/strategy-model-engine/SKILL.md 提到 "Purged Group Time Series Split",
      但 scripts/base/base_model.py 仅有 train/predict/save/load,无 CV 工具
    - 因子 IC 分析中无法避免 label 与 feature 之间的时序污染

借鉴要点
========
- Marcos López de Prado 《Advances in Financial Machine Learning》(AFML) 第 7 章
  Purged K-Fold + Embargo:  解决金融时序中的 label leakage
- VectorBT PRO 的 `from_purged_kfold` 与 `Splitter` API:
  https://vectorbt.pro/features/optimization/#purged-cv
- AKQuant 的 Walk-forward Validation (内置滚动训练框架):
  https://akquant.akfamily.xyz/en/advanced/ml/

本模块提供
==========
1. PurgedKFold: K-Fold + 训练/测试之间 purge gap + 后续 embargo
2. CombinatorialPurgedKFold: AFML 第 12 章路径式多 backtest
3. WalkForwardSplitter: 滚动训练/验证/测试三段切分
4. ic_time_series_split: 因子 IC 分析专用的扩展 walk-forward
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------
@dataclass
class CVSplit:
    """一次切分的结果"""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: int

    def summary(self) -> dict:
        return {
            "fold": self.fold_id,
            "train_size": int(len(self.train_idx)),
            "val_size": int(len(self.val_idx)) if self.val_idx is not None else 0,
            "test_size": int(len(self.test_idx)),
        }


# ---------------------------------------------------------------------------
# Purged K-Fold
# ---------------------------------------------------------------------------
class PurgedKFold:
    """
    Purged K-Fold 交叉验证 (AFML 第 7 章)

    步骤:
        1. 将样本按时间排序后 K 折
        2. 训练集 -> 测试集 之间剔除 `purge_td` 时间窗内的样本 (避免 label leakage)
        3. 测试集后 `embargo_td` 时间窗内的样本也禁用 (避免前视偏差)

    参数
    -----
    n_splits : 折数
    purge_td : 训练/测试边界两侧的剔除时长, 单位与 `times` 一致 (默认: "5D" 5 天)
    embargo_td : 测试集之后的禁用时长
    times : 与样本等长的 DatetimeIndex / Series, 默认为整数索引

    用法
    -----
    >>> splitter = PurgedKFold(n_splits=5, purge_td="5D", embargo_td="5D")
    >>> for split in splitter.split(X):
    ...     train, test = split.train_idx, split.test_idx
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_td: Optional[str] = None,
        embargo_td: Optional[str] = None,
        times: Optional[pd.Series] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.purge_td = purge_td
        self.embargo_td = embargo_td
        self.times = times

    def _to_timedelta(self, td_str: Optional[str]) -> Optional[pd.Timedelta]:
        if td_str is None:
            return None
        return pd.Timedelta(td_str)

    def get_n_splits(self) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[CVSplit]:
        n = len(X)
        if n < self.n_splits:
            raise ValueError(f"样本数 {n} 小于折数 {self.n_splits}")

        # 时间索引: 默认按行号
        if self.times is not None:
            times = pd.Series(self.times).reset_index(drop=True)
        elif isinstance(X.index, pd.DatetimeIndex):
            times = pd.Series(X.index)
        else:
            times = pd.Series(np.arange(n))

        purge = self._to_timedelta(self.purge_td)
        embargo = self._to_timedelta(self.embargo_td)
        is_dt = pd.api.types.is_datetime64_any_dtype(times)

        # K 折边界
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        indices = np.arange(n)
        current = 0
        for k in range(self.n_splits):
            start, stop = current, current + fold_sizes[k]
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            current = stop

            if is_dt and (purge is not None or embargo is not None):
                # purge: 测试集起点 - purge 之后, 测试集终点 + purge 之前的训练样本剔除
                t_test_start = times.iloc[test_idx].min()
                t_test_end = times.iloc[test_idx].max()
                t_train = times.iloc[train_idx].reset_index(drop=True)

                keep_mask = np.ones(len(train_idx), dtype=bool)
                if purge is not None:
                    purge_after = t_test_start - purge
                    purge_before = t_test_end + purge
                    keep_mask &= ~((t_train > purge_after) & (t_train < purge_before)).to_numpy()
                if embargo is not None:
                    embargo_until = t_test_end + embargo
                    keep_mask &= (t_train > embargo_until).to_numpy() | (t_train < t_test_start).to_numpy()
                train_idx = train_idx[keep_mask]

            yield CVSplit(
                train_idx=train_idx,
                val_idx=np.array([], dtype=int),
                test_idx=test_idx,
                fold_id=k,
            )


# ---------------------------------------------------------------------------
# Walk-Forward Splitter
# ---------------------------------------------------------------------------
class WalkForwardSplitter:
    """
    滚动训练/验证/测试三段切分 (借鉴 AKQuant / AFML)

    时间序列从前往后推进,每次切分包含:
        - train : 过去 train_size 长度用于训练
        - val   : 中间 val_size 长度用于超参 (可省略)
        - test  : 最后 test_size 长度用于评估

    参数
    -----
    train_size : 训练窗口长度
    val_size   : 验证窗口长度 (None 表示无验证)
    test_size  : 测试窗口长度
    step_size  : 每次窗口前进步长, 默认与 test_size 相同
    expanding  : True 表示 train 是 expanding (从开始累积), False 表示 rolling (固定窗口)

    示例
    ----
    >>> splitter = WalkForwardSplitter(train_size=252, val_size=63, test_size=63, step_size=63)
    >>> list(splitter.split(daily_index))  # 滚动 4 年训练 + 1 季度验证 + 1 季度测试
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        val_size: Optional[int] = None,
        step_size: Optional[int] = None,
        expanding: bool = True,
    ) -> None:
        if train_size < 1 or test_size < 1:
            raise ValueError("train_size / test_size 必须 >= 1")
        if val_size is not None and val_size < 1:
            raise ValueError("val_size 必须 >= 1")
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.expanding = expanding

    def get_n_splits(self, n_samples: int) -> int:
        span = self.train_size + (self.val_size or 0) + self.test_size
        if n_samples < span:
            return 0
        return 1 + (n_samples - span) // self.step_size

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[CVSplit]:
        n = len(X)
        span = self.train_size + (self.val_size or 0) + self.test_size
        if n < span:
            return

        cursor = 0
        fold = 0
        while cursor + span <= n:
            if self.expanding:
                train_start = 0
            else:
                train_start = cursor
            train_end = cursor + self.train_size
            val_start = train_end
            val_end = val_start + (self.val_size or 0)
            test_start = val_end
            test_end = test_start + self.test_size

            yield CVSplit(
                train_idx=np.arange(train_start, train_end),
                val_idx=np.arange(val_start, val_end) if self.val_size else np.array([], dtype=int),
                test_idx=np.arange(test_start, test_end),
                fold_id=fold,
            )
            fold += 1
            cursor += self.step_size


# ---------------------------------------------------------------------------
# Combinatorial Purged K-Fold (AFML 第 12 章)
# ---------------------------------------------------------------------------
class CombinatorialPurgedKFold:
    """
    组合式 purged K-Fold (AFML §12)

    将数据分为 n_groups 组,每次选 n_test_groups 组作为测试集,
    其余作为训练集,产生 C(n_groups, n_test_groups) 条回测路径。
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_td: Optional[str] = None,
        embargo_td: Optional[str] = None,
    ) -> None:
        if n_test_groups >= n_groups:
            raise ValueError("n_test_groups 必须小于 n_groups")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_td = purge_td
        self.embargo_td = embargo_td

    def _combinations(self, n: int, k: int) -> Iterator[Tuple[int, ...]]:
        from itertools import combinations

        return combinations(range(n), k)

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[CVSplit]:
        n = len(X)
        group_size = n // self.n_groups
        group_indices = [
            np.arange(i * group_size, (i + 1) * group_size) for i in range(self.n_groups)
        ]
        # 余数放入最后一组
        group_indices[-1] = np.concatenate(
            [group_indices[-1], np.arange(self.n_groups * group_size, n)]
        )

        purge = pd.Timedelta(self.purge_td) if self.purge_td else None
        embargo = pd.Timedelta(self.embargo_td) if self.embargo_td else None

        is_dt = isinstance(X.index, pd.DatetimeIndex)
        times = X.index if is_dt else pd.Series(np.arange(n), index=X.index)

        fold = 0
        for combo in self._combinations(self.n_groups, self.n_test_groups):
            test_idx = np.concatenate([group_indices[i] for i in combo])
            test_set = set(test_idx.tolist())
            train_idx = np.array(
                [i for i in range(n) if i not in test_set], dtype=int
            )

            if purge is not None or embargo is not None:
                t_test_start = times[test_idx].min()
                t_test_end = times[test_idx].max()
                keep = np.ones(len(train_idx), dtype=bool)
                if purge is not None:
                    purge_after = t_test_start - purge
                    purge_before = t_test_end + purge
                    keep &= ~((times[train_idx] > purge_after) & (times[train_idx] < purge_before)).to_numpy()
                if embargo is not None:
                    embargo_until = t_test_end + embargo
                    keep &= (times[train_idx] > embargo_until).to_numpy() | (times[train_idx] < t_test_start).to_numpy()
                train_idx = train_idx[keep]

            yield CVSplit(
                train_idx=train_idx,
                val_idx=np.array([], dtype=int),
                test_idx=test_idx,
                fold_id=fold,
            )
            fold += 1


# ---------------------------------------------------------------------------
# 因子 IC 分析专用切分
# ---------------------------------------------------------------------------
def ic_time_series_split(
    factor_panel: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 60,
    purge_days: int = 5,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    因子 IC 分析的时序切分 (按截面日划分)。

    参数
    -----
    factor_panel : MultiIndex [date, code] 排序后的 DataFrame, 含 factor 列
    n_splits : 折数
    min_train_size : 最小训练天数
    purge_days : 训练/测试边界之间跳过的天数

    返回
    ----
    (train_df, val_df, test_df) 迭代器
    """
    if "date" not in factor_panel.columns:
        raise ValueError("factor_panel 必须包含 'date' 列")
    if "code" not in factor_panel.columns:
        raise ValueError("factor_panel 必须包含 'code' 列")

    dates = np.sort(factor_panel["date"].unique())
    n_dates = len(dates)
    if n_dates < min_train_size + n_splits:
        raise ValueError("日期数不足以进行切分")

    fold_size = (n_dates - min_train_size) // n_splits
    if fold_size < 1:
        raise ValueError("折大小为 0, 请增大数据或减少折数")

    for k in range(n_splits):
        test_start = min_train_size + k * fold_size
        test_end = test_start + fold_size
        purge_start = max(0, test_start - purge_days)
        train_dates = dates[:purge_start]
        test_dates = dates[test_start:test_end]
        if len(test_dates) == 0:
            break
        train_df = factor_panel[factor_panel["date"].isin(train_dates)]
        test_df = factor_panel[factor_panel["date"].isin(test_dates)]
        val_df = test_df.iloc[0:0]  # empty
        yield train_df, val_df, test_df
