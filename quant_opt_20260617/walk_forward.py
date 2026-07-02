"""
Walk-Forward 滚动训练验证框架
==================================

借鉴来源
--------
- akquant ML Guide 的 Walk-Forward Validation（rolling retrain + OOS test）
- qlib 的 RollingGen（在 RollingGen 中，Trainer 负责 fit，RollingGen 负责滑窗）
- hudson-and-thames/mlfinlab 的 CombinatorialPurgedCV

设计目标
--------
解决 jingni-trader strategy-model-engine/engine.py 的两个核心痛点：
1. 当前 train() 是"一次性全量训练"，没有样本外验证，没有滚动重训
2. 缺乏对 look-ahead bias 的显式防御

核心思想
--------
按时间顺序把样本切分为 N 个窗口，每个窗口内：
  - Train:  [T_i, T_i + train_window)
  - Test:   [T_i + train_window + purge_gap, T_i + train_window + purge_gap + test_window)
逐窗口训练 → 样本外预测 → 拼接所有 OOS 预测 → 评估整体 OOS 表现

应用
----
- 任何 sklearn 兼容的 estimator（fit/predict）
- 支持 custom scorer：IC, rank IC, mse, accuracy
- 输出每个窗口的 (model, metrics) 列表 + 整体 OOS 指标
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from scipy.stats import spearmanr


# ============================================================
# 数据结构
# ============================================================

@dataclass
class WindowSplit:
    """单个滚动窗口的切分"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray
    test_idx: np.ndarray
    purged_train_idx: np.ndarray  # 去除 purge_gap 后的训练集


@dataclass
class WindowResult:
    """单个窗口的结果"""
    fold_id: int
    train_size: int
    test_size: int
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]
    metrics: Dict[str, float]
    oos_predictions: Optional[pd.Series] = None  # index 对应 test_idx
    model: Optional[Any] = None


@dataclass
class WalkForwardResult:
    """整体结果"""
    model_factory: Callable
    windows: List[WindowResult]
    oos_predictions: Optional[pd.Series] = None  # 跨所有窗口拼接
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"WalkForwardResult  (n_windows={len(self.windows)})",
            f"  config: {self.config}",
            f"  overall: {self.overall_metrics}",
        ]
        for w in self.windows:
            lines.append(
                f"  fold {w.fold_id}: train={w.train_period[0]}~{w.train_period[1]} "
                f"({w.train_size}), test={w.test_period[0]}~{w.test_period[1]} "
                f"({w.test_size}), metrics={w.metrics}"
            )
        return "\n".join(lines)


# ============================================================
# 窗口生成器
# ============================================================

def make_walk_forward_splits(
    dates: pd.Series,
    train_window_days: int = 504,
    test_window_days: int = 63,
    step_days: Optional[int] = None,
    purge_gap_days: int = 20,
    min_train_samples: int = 1000,
    expanding: bool = True,
) -> List[WindowSplit]:
    """
    按时间顺序生成滚动窗口

    参数
    ----
    dates: 训练样本的 date 列（与 X.index 对齐）
    train_window_days: 训练窗口长度（交易日；504 ≈ 2 年）
    test_window_days: 测试窗口长度（63 ≈ 3 个月）
    step_days: 滑动步长（默认 = test_window_days，不重叠）
    purge_gap_days: 训练/测试之间的 purge gap（防御 look-ahead）
    min_train_samples: 训练样本量下限
    expanding: True = 扩展窗口（训练从最早开始），False = 滚动窗口（固定长度）

    返回
    ----
    WindowSplit 列表
    """
    if step_days is None:
        step_days = test_window_days

    dates = pd.to_datetime(dates)
    unique_dates = np.sort(dates.unique())
    # 转成 pd.Timestamp 以便后续使用 .date() 等方法
    unique_dates = pd.to_datetime(unique_dates)
    n_days = len(unique_dates)
    if n_days < 30:
        raise ValueError("unique dates 数量过少 (<30)，无法生成有效切分")

    splits: List[WindowSplit] = []
    fold_id = 0
    cursor = 0

    while True:
        # 训练窗口的结束位置
        if expanding:
            train_end_pos = cursor + train_window_days
        else:
            train_end_pos = cursor + train_window_days
        test_start_pos = train_end_pos + purge_gap_days
        test_end_pos = test_start_pos + test_window_days

        if test_end_pos > n_days:
            break
        if train_end_pos > n_days:
            break

        train_start_date = unique_dates[cursor]
        train_end_date = unique_dates[train_end_pos - 1]
        test_start_date = unique_dates[test_start_pos]
        test_end_date = unique_dates[test_end_pos - 1]

        train_idx = np.where(
            (dates >= train_start_date) & (dates <= train_end_date)
        )[0]
        test_idx = np.where(
            (dates >= test_start_date) & (dates <= test_end_date)
        )[0]

        if len(train_idx) >= min_train_samples and len(test_idx) > 0:
            splits.append(WindowSplit(
                fold_id=fold_id,
                train_start=train_start_date,
                train_end=train_end_date,
                test_start=test_start_date,
                test_end=test_end_date,
                train_idx=train_idx,
                test_idx=test_idx,
                purged_train_idx=train_idx,
            ))
            fold_id += 1

        cursor += step_days

    return splits


# ============================================================
# 主类
# ============================================================

class WalkForwardCV:
    """
    Walk-Forward 验证主类

    用法
    ----
    >>> cv = WalkForwardCV(
    ...     model_factory=lambda: LGBMRegressor(n_estimators=100),
    ...     scorer="ic",  # or custom callable
    ...     train_window_days=504,
    ...     test_window_days=63,
    ...     purge_gap_days=20,
    ... )
    >>> result = cv.run(X, y, dates)
    >>> print(result.summary())
    >>> result.oos_predictions  # 全部 OOS 预测
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        scorer: Union[str, Callable] = "ic",
        train_window_days: int = 504,
        test_window_days: int = 63,
        step_days: Optional[int] = None,
        purge_gap_days: int = 20,
        expanding: bool = True,
        min_train_samples: int = 1000,
        verbose: bool = True,
    ):
        self.model_factory = model_factory
        self.scorer = scorer
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
        self.purge_gap_days = purge_gap_days
        self.expanding = expanding
        self.min_train_samples = min_train_samples
        self.verbose = verbose

    def _score(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        scores: Dict[str, float] = {}
        if self.scorer == "ic" or self.scorer == "rank_ic":
            if len(y_true) >= 2:
                ic, _ = spearmanr(y_true, y_pred, nan_policy="omit")
                scores["rank_ic"] = float(ic) if not np.isnan(ic) else 0.0
        if self.scorer == "mse" or self.scorer == "auto":
            scores["mse"] = float(mean_squared_error(y_true, y_pred))
            scores["rmse"] = float(np.sqrt(scores["mse"]))
        if self.scorer == "accuracy":
            scores["accuracy"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
            scores["f1"] = float(f1_score(y_true, (y_pred > 0.5).astype(int), zero_division=0))
        if self.scorer == "auto":
            # 始终附带 rank_ic 作为辅助指标
            if "rank_ic" not in scores and len(y_true) >= 2:
                ic, _ = spearmanr(y_true, y_pred, nan_policy="omit")
                scores["rank_ic"] = float(ic) if not np.isnan(ic) else 0.0
        if callable(self.scorer):
            scores["custom"] = float(self.scorer(y_true, y_pred))
        return scores

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        refit: bool = True,
    ) -> WalkForwardResult:
        """
        执行 walk-forward 滚动训练/预测

        参数
        ----
        X: 特征矩阵（index 可以是任意，但需与 dates 对齐）
        y: 标签
        dates: 与 X 行对应的 date 列
        refit: 是否每个窗口重新训练（True = 重新训练；False = 用全量数据一次性训练）
        """
        # 对齐
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        dates = pd.Series(dates).reset_index(drop=True)

        splits = make_walk_forward_splits(
            dates=dates,
            train_window_days=self.train_window_days,
            test_window_days=self.test_window_days,
            step_days=self.step_days,
            purge_gap_days=self.purge_gap_days,
            min_train_samples=self.min_train_samples,
            expanding=self.expanding,
        )

        if not splits:
            raise ValueError("无法生成任何 walk-forward 窗口，请检查日期范围或调整参数")

        if self.verbose:
            print(f"WalkForwardCV: 生成 {len(splits)} 个窗口 "
                  f"(train={self.train_window_days}d, test={self.test_window_days}d, "
                  f"purge={self.purge_gap_days}d)")

        windows: List[WindowResult] = []
        oos_pred_list: List[pd.Series] = []

        for sp in splits:
            if refit:
                model = self.model_factory()
            else:
                # 第一次训练后复用模型（不推荐用于严格 OOS）
                if not windows:
                    model = self.model_factory()
                else:
                    model = windows[-1].model

            X_train = X.iloc[sp.purged_train_idx]
            y_train = y.iloc[sp.purged_train_idx]
            X_test = X.iloc[sp.test_idx]
            y_test = y.iloc[sp.test_idx]

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception as e:
                if self.verbose:
                    print(f"  fold {sp.fold_id} 训练/预测失败: {e}")
                continue

            # OOS 预测（保留原 index）
            oos_pred = pd.Series(preds, index=X_test.index, name="pred")
            oos_pred_list.append(oos_pred)

            scores = self._score(y_test.values, preds)

            windows.append(WindowResult(
                fold_id=sp.fold_id,
                train_size=len(X_train),
                test_size=len(X_test),
                train_period=(str(sp.train_start.date()), str(sp.train_end.date())),
                test_period=(str(sp.test_start.date()), str(sp.test_end.date())),
                metrics=scores,
                oos_predictions=oos_pred,
                model=model,
            ))

            if self.verbose:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
                print(f"  fold {sp.fold_id}: {metric_str} "
                      f"(train={sp.train_start.date()}~{sp.train_end.date()}, "
                      f"test={sp.test_start.date()}~{sp.test_end.date()})")

        # 拼接 OOS 预测
        all_oos = pd.concat(oos_pred_list) if oos_pred_list else pd.Series(dtype=float)
        # 整体 OOS 评估
        overall_metrics: Dict[str, float] = {}
        if not all_oos.empty:
            # 用 y 在 all_oos.index 处取值
            oos_y = y.iloc[all_oos.index]
            overall_metrics = self._score(oos_y.values, all_oos.values)
            # 额外：OOS 覆盖率
            overall_metrics["oos_coverage"] = float(len(all_oos)) / float(len(y))
            if self.verbose:
                print(f"\nOverall OOS metrics: {overall_metrics}")

        return WalkForwardResult(
            model_factory=self.model_factory,
            windows=windows,
            oos_predictions=all_oos,
            overall_metrics=overall_metrics,
            config={
                "train_window_days": self.train_window_days,
                "test_window_days": self.test_window_days,
                "step_days": self.step_days or self.test_window_days,
                "purge_gap_days": self.purge_gap_days,
                "expanding": self.expanding,
                "scorer": self.scorer if isinstance(self.scorer, str) else "custom",
                "n_windows": len(windows),
            },
        )

    @staticmethod
    def to_dataframe(result: WalkForwardResult) -> pd.DataFrame:
        """把结果转成 DataFrame（每个窗口一行）"""
        rows = []
        for w in result.windows:
            row = {
                "fold_id": w.fold_id,
                "train_size": w.train_size,
                "test_size": w.test_size,
                "train_start": w.train_period[0],
                "train_end": w.train_period[1],
                "test_start": w.test_period[0],
                "test_end": w.test_period[1],
            }
            row.update(w.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
