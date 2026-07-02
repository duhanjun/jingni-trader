"""
Walk-Forward Rolling Training Pipeline
======================================

借鉴来源
--------
- Qlib (microsoft/qlib) 的 `qlib.contrib.data.rolling.RollingGen`
  核心思想：固定大小的滚动窗口 + 严格的 train/test split + segment_id 隔离
- AKQuant 的 walk-forward validation
- 业界标准的时序交叉验证（避免未来信息泄露）

原项目痛点（基于 /workspace/skills/strategy-model-engine/engine.py）
--------------------------------------------------------
1. `purged_group_ts_split` 实现有 bug：
   - 把 group 的最后一行归到 train_idx、最后一行归到 val_idx
   - 应当整组二选一
2. 缺少滚动窗口训练管道
3. 缺少多窗口的 IC 聚合（验证模型的稳定性）
4. 缺少分段独立评估（segment_id 隔离）
5. 实验跟踪未提供子运行（MLflow nested runs）

设计目标
--------
- 修复 PurgedGroupTimeSeriesSplit
- 提供 RollingDatasetGenerator，产出 N 个 (train_df, valid_df, test_df)
- 通用模型适配器（LightGBM / sklearn / 任意 callable）
- 多窗口 IC/RankIC 聚合（mean, std, t-stat）
- 与原 strategy-model-engine 兼容（不破坏现有 API）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from collections import defaultdict
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger("walkforward")


# ============================================================================
# 1. 修复后的 Purged Group Time-Series Split
# ============================================================================

class PurgedGroupTimeSeriesSplit:
    """
    修复后的 Purged Group Time-Series Split
    ---------------------------------------
    原 bug (model-engine/engine.py:62-89)：
        train_idx = list(df.index[df["date"] <= split_date])
        val_idx = list(df.index[df["date"] > split_date])
    这样会把同一个 code 的"末日"分到 train，"新日"分到 val，
    形成数据泄露。修复后按"date segment"为单位划分。

    正确实现：使用 segment_id 隔离，每个 segment（训练段）的全部数据
    都在同一折（train 或 val），避免同一交易日数据被分割。
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_segments: int = 2,
        purge_gap: int = 0,
    ):
        self.n_splits = n_splits
        self.min_train_segments = min_train_segments
        self.purge_gap = purge_gap

    def split(
        self, df: pd.DataFrame, segment_col: str = "date"
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        segments = sorted(df[segment_col].unique())
        n_segments = len(segments)
        if n_segments < self.min_train_segments + 1:
            raise ValueError(
                f"段数 {n_segments} 不足，至少需要 {self.min_train_segments + 1}"
            )

        # 按段均分 n_splits
        segment_folds = np.array_split(segments, self.n_splits)
        for i, val_segments in enumerate(segment_folds):
            if i < self.min_train_segments - 1:
                continue
            train_segments = []
            for j, fold in enumerate(segment_folds):
                if j < i:
                    train_segments.extend(fold)
            # purge：移除与 val 段相邻的训练样本
            if self.purge_gap > 0:
                val_set = set(pd.to_datetime(val_segments))
                purge_cut = min(val_set) - pd.Timedelta(days=self.purge_gap)
                train_segments = [s for s in train_segments
                                  if pd.to_datetime(s) < purge_cut]
            train_idx = df.index[df[segment_col].isin(train_segments)].values
            val_idx = df.index[df[segment_col].isin(val_segments)].values
            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ============================================================================
# 2. 滚动窗口数据生成器
# ============================================================================

@dataclass
class RollingWindowConfig:
    train_period: int = 252 * 3       # 3 年训练
    valid_period: int = 63            # 3 个月验证
    test_period: int = 21             # 1 个月测试
    step: int = 21                    # 1 个月滚动一次
    min_train_period: int = 252       # 至少 1 年训练
    expanding: bool = False           # True=扩展窗口, False=滚动窗口


class RollingDatasetGenerator:
    """
    滚动数据集生成器
    ----------------
    输入：(date, code, feature_1, ..., label) 的 DataFrame
    输出：每个窗口的 (train_df, valid_df, test_df, segment_id)
    """

    def __init__(self, config: RollingWindowConfig):
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Iterator[Dict[str, Any]]:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        dates = sorted(df[date_col].unique())
        n_dates = len(dates)
        if n_dates < self.config.train_period + self.config.valid_period + self.config.test_period:
            raise ValueError(
                f"日期数 {n_dates} 不足，需要至少 "
                f"{self.config.train_period + self.config.valid_period + self.config.test_period}"
            )

        # 计算所有窗口
        windows = []
        test_start_idx = (
            self.config.train_period + self.config.valid_period
        )
        while test_start_idx < n_dates:
            test_end_idx = min(test_start_idx + self.config.test_period, n_dates)
            valid_end_idx = test_start_idx
            valid_start_idx = max(0, valid_end_idx - self.config.valid_period)
            if self.config.expanding:
                train_start_idx = 0
            else:
                train_start_idx = max(
                    0, valid_start_idx - self.config.train_period
                )
            if valid_start_idx - train_start_idx < self.config.min_train_period:
                test_start_idx += self.config.step
                continue

            windows.append({
                "segment_id": len(windows) + 1,
                "train_start": dates[train_start_idx],
                "train_end": dates[valid_start_idx - 1],
                "valid_start": dates[valid_start_idx],
                "valid_end": dates[valid_end_idx - 1],
                "test_start": dates[test_start_idx],
                "test_end": dates[test_end_idx - 1],
            })
            test_start_idx += self.config.step

        for w in windows:
            train_mask = (df[date_col] >= w["train_start"]) & (df[date_col] <= w["train_end"])
            valid_mask = (df[date_col] >= w["valid_start"]) & (df[date_col] <= w["valid_end"])
            test_mask = (df[date_col] >= w["test_start"]) & (df[date_col] <= w["test_end"])
            yield {
                "segment_id": w["segment_id"],
                "train": df[train_mask].copy(),
                "valid": df[valid_mask].copy(),
                "test": df[test_mask].copy(),
                "train_period": (w["train_start"], w["train_end"]),
                "valid_period": (w["valid_start"], w["valid_end"]),
                "test_period": (w["test_start"], w["test_end"]),
            }


# ============================================================================
# 3. 模型适配器（统一接口）
# ============================================================================

ModelAdapter = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame], Tuple[Any, Dict[str, float]]]
"""
模型适配器签名：
    train(X_train, y_train, X_valid, y_valid, X_test) -> (model, info)
- X_train, y_train: 训练特征与标签
- X_valid, y_valid: 验证特征与标签（可为空）
- X_test: 测试特征（用于预测）
- 返回: (model, info)  其中 model 有 .predict(X) 方法
"""


def make_lightgbm_adapter(
    params: Optional[Dict] = None,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
) -> ModelAdapter:
    """构造一个 LightGBM 适配器（如不可用则回退到 sklearn）"""
    try:
        import lightgbm as lgb
        default = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        if params:
            default.update(params)

        def adapter(X_train, y_train, X_valid, y_valid, X_test):
            train_set = lgb.Dataset(X_train, label=y_train)
            valid_sets = [train_set]
            valid_names = ["train"]
            if X_valid is not None and len(X_valid) > 0:
                valid_sets.append(lgb.Dataset(X_valid, label=y_valid, reference=train_set))
                valid_names.append("valid")
            callbacks = [lgb.log_evaluation(period=0)]
            if X_valid is not None:
                callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            model = lgb.train(
                default, train_set,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            info = {"best_iter": model.best_iteration if hasattr(model, "best_iteration") else None}
            return model, info
        return adapter
    except ImportError:
        logger.warning("lightgbm 未安装，回退到 sklearn GBDT")
        return make_sklearn_gbdt_adapter()


def make_sklearn_gbdt_adapter(
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.05,
) -> ModelAdapter:
    """sklearn GBDT 适配器（回退方案）"""
    from sklearn.ensemble import GradientBoostingRegressor

    def adapter(X_train, y_train, X_valid, y_valid, X_test):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model, {}
    return adapter


# ============================================================================
# 4. Walk-Forward 训练主流程
# ============================================================================

@dataclass
class WalkForwardResult:
    segment_id: int
    period: Tuple[pd.Timestamp, pd.Timestamp]
    predictions: pd.DataFrame         # [date, code, pred]
    metrics: Dict[str, float]          # IC, RankIC, MAE, etc.
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]


@dataclass
class WalkForwardSummary:
    results: List[WalkForwardResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """汇总多窗口结果"""
        if not self.results:
            return {}
        ic_list = [r.metrics.get("ic", np.nan) for r in self.results]
        rankic_list = [r.metrics.get("rank_ic", np.nan) for r in self.results]
        ic_arr = np.array([x for x in ic_list if not np.isnan(x)])
        rankic_arr = np.array([x for x in rankic_list if not np.isnan(x)])
        out = {
            "n_windows": len(self.results),
            "ic_mean": float(ic_arr.mean()) if len(ic_arr) > 0 else None,
            "ic_std": float(ic_arr.std()) if len(ic_arr) > 0 else None,
            "ic_tstat": float(ic_arr.mean() / (ic_arr.std() / np.sqrt(len(ic_arr))))
                if len(ic_arr) > 1 else None,
            "rank_ic_mean": float(rankic_arr.mean()) if len(rankic_arr) > 0 else None,
            "rank_ic_std": float(rankic_arr.std()) if len(rankic_arr) > 0 else None,
            "rank_ic_tstat": float(rankic_arr.mean() / (rankic_arr.std() / np.sqrt(len(rankic_arr))))
                if len(rankic_arr) > 1 else None,
            "ic_per_window": ic_list,
            "rank_ic_per_window": rankic_list,
        }
        # 稳定性指标：相邻窗口的 IC 变化
        if len(ic_arr) > 1:
            ic_diff = np.diff(ic_arr)
            out["ic_stability"] = float(1 / (1 + np.std(ic_diff)))
        else:
            out["ic_stability"] = None
        return out


def _calc_ic(pred_df: pd.DataFrame, label_col: str = "label", pred_col: str = "pred") -> Dict[str, float]:
    """逐日 IC + RankIC"""
    if pred_df.empty or label_col not in pred_df.columns:
        return {"ic": 0.0, "rank_ic": 0.0, "n_days": 0}
    daily_ic = []
    daily_rankic = []
    for dt, g in pred_df.groupby("date"):
        g = g[[label_col, pred_col]].dropna()
        if len(g) < 5:
            continue
        if g[label_col].std() == 0 or g[pred_col].std() == 0:
            continue
        daily_ic.append(np.corrcoef(g[label_col], g[pred_col])[0, 1])
        daily_rankic.append(pd.Series(g[label_col]).corr(pd.Series(g[pred_col]), method="spearman"))
    if not daily_ic:
        return {"ic": 0.0, "rank_ic": 0.0, "n_days": 0}
    return {
        "ic": float(np.mean(daily_ic)),
        "rank_ic": float(np.mean(daily_rankic)),
        "n_days": len(daily_ic),
        "ic_std": float(np.std(daily_ic)),
    }


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_adapter: ModelAdapter,
    config: Optional[RollingWindowConfig] = None,
    retrain_each_window: bool = True,
) -> WalkForwardSummary:
    """
    执行 Walk-Forward 滚动训练
    --------------------------
    返回: WalkForwardSummary 包含每窗口的预测、IC、汇总指标
    """
    if config is None:
        config = RollingWindowConfig()

    generator = RollingDatasetGenerator(config)
    summary = WalkForwardSummary()

    for window in generator.generate(df):
        segment_id = window["segment_id"]
        train = window["train"]
        valid = window["valid"]
        test = window["test"]

        if len(train) < 100 or len(test) == 0:
            logger.warning(f"segment {segment_id}: 训练/测试数据不足，跳过")
            continue

        X_train = train[feature_cols]
        y_train = train[label_col]
        X_valid = valid[feature_cols] if len(valid) > 0 else None
        y_valid = valid[label_col] if len(valid) > 0 else None
        X_test = test[feature_cols]

        # 训练
        model, info = model_adapter(
            X_train, y_train,
            X_valid, y_valid,
            X_test,
        )

        # 预测
        try:
            preds = model.predict(X_test)
        except Exception as e:
            logger.error(f"segment {segment_id} 预测失败: {e}")
            continue

        test_pred = test[["date", "code"]].copy()
        if label_col in test.columns:
            test_pred[label_col] = test[label_col].values
        test_pred["pred"] = np.asarray(preds)
        test_pred["segment_id"] = segment_id

        # 计算指标
        train_metrics = _calc_ic(
            train.assign(pred=model.predict(X_train))
        ) if retrain_each_window else {}
        val_metrics = _calc_ic(
            valid.assign(pred=model.predict(X_valid)) if X_valid is not None and len(X_valid) > 0 else pd.DataFrame()
        )
        test_metrics = _calc_ic(test_pred, label_col=label_col, pred_col="pred")

        summary.results.append(WalkForwardResult(
            segment_id=segment_id,
            period=window["test_period"],
            predictions=test_pred,
            metrics=test_metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        ))

    summary.aggregate_metrics = summary.summary()
    return summary
