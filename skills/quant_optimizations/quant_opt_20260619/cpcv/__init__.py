"""
Combinatorial Purged Cross-Validation (CPCV)
+ Embargo 隔离期

借鉴自：
  1. Marcos López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 7
  2. Qlib 的 TimeSeriesSplit 思路 (qlib.model.cross_validation)
  3. jingni-trader 现存的 purged_group_ts_split (skills/strategy-model-engine/engine.py:109)

设计动机：
  jingni-trader 当前的 purged_group_ts_split 仅做单序列时间切分 + 简单 purge，
  缺少：
    - embargo：训练集尾部与验证集头部的样本"冷却期"（防止标签泄漏）
    - combinatorial：单一 train/test 切分过拟合风险高
    - 多路径 backtest：CPCV 给出 N 个独立 (train, test) 路径，
      可生成 N 条独立 equity 曲线，更稳健地估计策略真实表现

参考实现：
  https://github.com/sylvaincom/sylvaincom.github.io/blob/master/code/cpcv.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from itertools import combinations
from typing import Generator, List, Optional, Tuple


@dataclass
class CPVCSplit:
    """
    单条 CPCV 切分

    Attributes:
        train_idx: 训练集下标
        test_idx: 测试集下标
        path_id: 路径编号 (0..n_paths-1)
        fold_id: 测试集所在的 fold 编号
    """
    train_idx: np.ndarray
    test_idx: np.ndarray
    path_id: int
    fold_id: int

    def __repr__(self):
        return (f"CPVCSplit(path={self.path_id}, fold={self.fold_id}, "
                f"n_train={len(self.train_idx)}, n_test={len(self.test_idx)})")


class CombinatorialPurgedCV:
    """
    组合式 purged K 折 + embargo 交叉验证

    关键参数：
        n_splits:  时间序列分成多少个连续 fold (K)
        n_test_splits: 每个 path 中作为测试集的 fold 数量 (k)
        embargo_pct: 训练集尾部隔离比例 (相对样本总量)
        purge_pct: 训练/测试边界处的 purge 比例

    生成路径数 = C(n_splits, n_test_splits) (组合数)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
    ):
        if n_test_splits >= n_splits:
            raise ValueError(f"n_test_splits ({n_test_splits}) 必须 < n_splits ({n_splits})")
        if not 0 <= embargo_pct < 0.5:
            raise ValueError(f"embargo_pct 必须在 [0, 0.5) 之间")
        if not 0 <= purge_pct < 0.5:
            raise ValueError(f"purge_pct 必须在 [0, 0.5) 之间")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Generator[CPVCSplit, None, None]:
        """
        生成所有 (train, test) 组合路径

        与 sklearn KFold 不同：
        - 时间序列必须按"原始顺序"切分
        - 训练 = N 个 fold 中除去被选为 test 的 k 个
        - embargo 在训练集尾部（紧邻 test 起点）实施
        - purge 在训练/测试边界实施
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        fold_bounds = np.cumsum(np.concatenate([[0], fold_sizes]))

        embargo_size = int(self.embargo_pct * n_samples)
        purge_size = int(self.purge_pct * n_samples)

        all_combos = combinations(range(self.n_splits), self.n_test_splits)
        for path_id, test_folds in enumerate(all_combos):
            test_fold_set = set(test_folds)
            test_idx_parts = []
            train_idx_parts = []
            for fold_id in range(self.n_splits):
                fold_start, fold_end = fold_bounds[fold_id], fold_bounds[fold_id + 1]
                fold_idx = indices[fold_start:fold_end]
                if fold_id in test_fold_set:
                    test_idx_parts.append(fold_idx)
                else:
                    train_idx_parts.append(fold_idx)

            train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)
            test_idx = np.concatenate(test_idx_parts) if test_idx_parts else np.array([], dtype=int)

            train_idx, test_idx = self._apply_purge_and_embargo(
                train_idx, test_idx, test_fold_set, fold_bounds, embargo_size, purge_size
            )

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield CPVCSplit(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    path_id=path_id,
                    fold_id=tuple(sorted(test_folds)),
                )

    def _apply_purge_and_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        test_fold_set: set,
        fold_bounds: np.ndarray,
        embargo_size: int,
        purge_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        实施 purge + embargo
        """
        if len(train_idx) == 0 or len(test_idx) == 0:
            return train_idx, test_idx

        test_start_pos = fold_bounds[min(test_fold_set)]
        test_end_pos = fold_bounds[max(test_fold_set) + 1]

        if purge_size > 0:
            purge_start = max(0, test_start_pos - purge_size)
            purge_end = min(len(train_idx) + len(test_idx), test_end_pos + purge_size)
            in_purge_zone = (train_idx >= purge_start) & (train_idx < purge_end)
            train_idx = train_idx[~in_purge_zone]

        if embargo_size > 0:
            emb_start = test_end_pos
            emb_end = min(len(train_idx) + len(test_idx), test_end_pos + embargo_size)
            in_embargo = (train_idx >= emb_start) & (train_idx < emb_end)
            train_idx = train_idx[~in_embargo]

        return train_idx, test_idx

    def n_paths(self) -> int:
        """组合路径数 = C(n_splits, n_test_splits)"""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


def compare_with_jingni_split():
    """
    对比 jingni-trader 当前 purged_group_ts_split 与新 CPCV 的差异
    """
    from skills.strategy_model_engine.engine import ModelEngine
    dates = pd.date_range("2023-01-01", periods=240, freq="D")
    codes = [f"{600000 + i:06d}" for i in range(10)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"code": c, "date": d, "value": np.random.randn()})
    df = pd.DataFrame(rows)

    me = ModelEngine()
    legacy = me.purged_group_ts_split(df["date"], n_splits=3)

    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, embargo_pct=0.01, purge_pct=0.01)
    new_paths = list(cpcv.split(df))

    return {
        "legacy_splits": len(legacy),
        "legacy_purge_gap_used": "仅 PURGE_GAP_DAYS 单点切除",
        "cpcv_paths": len(new_paths),
        "cpcv_embargo_pct": cpcv.embargo_pct,
        "cpcv_purge_pct": cpcv.purge_pct,
        "cpcv_test_combination_per_path": cpcv.n_test_splits,
    }