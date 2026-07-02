"""
滚动训练 / 验证框架 (Walk-Forward Validation)

借鉴自：
  - AKQuant ml 子模块 (内置 walk-forward framework)
  - Qlib 论文中 rolling re-training 实践
  - Advances in Financial ML (Marcos López de Prado) 第 7 / 11 / 12 章

设计目标：
  在 jingni-trader 现有的 purged_group_ts_split 基础上，
  提供一个更完整的、用于"在线模拟"的滚动训练框架：
    1. 训练窗口按 (train, val, test) 三段滚动
    2. 每段训练一个模型，对应测试集做预测
    3. 把所有测试段拼成完整的 out-of-sample 预测
    4. 报告 IC / decile 收益 / 换手 等指标

本模块不直接修改 jingni-trader 的 strategy-model-engine，
仅作为独立的验证模块，与现有实现并跑对比。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WalkForwardConfig:
    train_window_days: int = 504     # 训练窗口（交易日，约 2 年）
    val_window_days: int = 126       # 验证窗口（约半年）
    test_window_days: int = 126      # 测试窗口（约半年）
    step_days: int = 126             # 滚动步长
    purge_days: int = 5              # 训练/测试 之间的清洗期
    embargo_days: int = 5            # 测试集后延 embargo


@dataclass
class WalkForwardResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def make_walk_forward_splits(
    dates: pd.Series,
    cfg: WalkForwardConfig,
) -> List[Dict[str, pd.Timestamp]]:
    """
    生成滚动训练 / 验证 / 测试三段式时间窗口。

    Parameters
    ----------
    dates : pd.Series
        每行对应的"观察日"（如某只股票某一天的 date）
    cfg : WalkForwardConfig

    Returns
    -------
    list of dict
        每个 dict 包含 train_start, train_end, val_start, val_end,
        test_start, test_end
    """
    sorted_dates = pd.Series(sorted(pd.unique(pd.to_datetime(dates)))).reset_index(drop=True)
    if len(sorted_dates) < cfg.train_window_days + cfg.val_window_days + cfg.test_window_days:
        return []

    splits: List[Dict[str, pd.Timestamp]] = []
    n = len(sorted_dates)
    cursor = cfg.train_window_days
    while True:
        train_start = sorted_dates[0]
        train_end = sorted_dates[cursor - 1]
        val_start = sorted_dates[cursor]
        val_end = sorted_dates[min(cursor + cfg.val_window_days - 1, n - 1)]
        test_start_idx = min(cursor + cfg.val_window_days + cfg.purge_days, n - 1)
        test_end_idx = min(test_start_idx + cfg.test_window_days - 1, n - 1)
        test_start = sorted_dates[test_start_idx]
        test_end = sorted_dates[test_end_idx]
        splits.append({
            "train_start": train_start,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        next_cursor = cursor + cfg.step_days
        if test_end_idx + cfg.embargo_days >= n - 1:
            break
        cursor = next_cursor
    return splits


def walk_forward_train_predict(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    fit_predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame], np.ndarray],
    cfg: Optional[WalkForwardConfig] = None,
) -> Tuple[pd.DataFrame, List[WalkForwardResult]]:
    """
    滚动训练 + 拼接 out-of-sample 预测。

    Parameters
    ----------
    X : pd.DataFrame
        特征
    y : pd.Series
        标签（与 X 同索引）
    dates : pd.Series
        观察日（与 X 同索引）
    fit_predict_fn : callable
        (X_train, y_train, X_val, y_val, X_test) -> np.ndarray 的预测函数
    cfg : WalkForwardConfig

    Returns
    -------
    oos_predictions : pd.DataFrame
        out-of-sample 预测结果，列: date/pred
    results : list of WalkForwardResult
        每折的训练 / 验证 / 测试 元信息
    """
    cfg = cfg or WalkForwardConfig()
    df = X.copy()
    df["__y__"] = y.values
    df["__date__"] = pd.to_datetime(dates.values)
    df["_orig_idx"] = np.arange(len(df))

    splits = make_walk_forward_splits(df["__date__"], cfg)
    results: List[WalkForwardResult] = []
    oos_parts: List[pd.DataFrame] = []

    for fold_id, sp in enumerate(splits, 1):
        train_mask = (df["__date__"] >= sp["train_start"]) & (df["__date__"] <= sp["train_end"])
        val_mask = (df["__date__"] >= sp["val_start"]) & (df["__date__"] <= sp["val_end"])
        test_mask = (df["__date__"] >= sp["test_start"]) & (df["__date__"] <= sp["test_end"])
        if train_mask.sum() < 100 or val_mask.sum() < 30 or test_mask.sum() < 30:
            continue

        X_train = df.loc[train_mask].drop(columns=["__y__", "__date__", "_orig_idx"])
        y_train = df.loc[train_mask, "__y__"]
        X_val = df.loc[val_mask].drop(columns=["__y__", "__date__", "_orig_idx"])
        y_val = df.loc[val_mask, "__y__"]
        X_test = df.loc[test_mask].drop(columns=["__y__", "__date__", "_orig_idx"])
        y_test = df.loc[test_mask, "__y__"]

        try:
            pred = np.asarray(fit_predict_fn(X_train, y_train, X_val, y_val, X_test), dtype=float)
        except Exception as exc:  # noqa: BLE001
            metrics = {"error": str(exc)}
            pred = np.full(len(X_test), np.nan)

        # 防止用户返回的预测长度与测试集不符
        if pred.shape[0] != X_test.shape[0]:
            pred = np.full(X_test.shape[0], np.nan)
        metrics = _eval_predictions(pred, y_test.values)

        pred_df = df.loc[test_mask, ["_orig_idx", "__date__"]].copy()
        pred_df["pred"] = pred
        oos_parts.append(pred_df)

        results.append(WalkForwardResult(
            fold_id=fold_id,
            train_start=sp["train_start"], train_end=sp["train_end"],
            val_start=sp["val_start"], val_end=sp["val_end"],
            test_start=sp["test_start"], test_end=sp["test_end"],
            metrics=metrics,
            predictions=pred_df,
        ))

    if not oos_parts:
        return pd.DataFrame(columns=["_orig_idx", "__date__", "pred"]), results
    oos = pd.concat(oos_parts, axis=0).sort_values("_orig_idx").reset_index(drop=True)
    oos = oos.rename(columns={"__date__": "date"})
    return oos, results


def _eval_predictions(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if np.isnan(pred).all():
        return {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "pred_nan_ratio": 1.0}
    valid = ~np.isnan(pred) & ~np.isnan(y)
    if valid.sum() < 30:
        return {"ic": 0.0, "rank_ic": 0.0, "icir": 0.0, "valid_n": int(valid.sum())}
    p, yy = pred[valid], y[valid]
    ic = float(np.corrcoef(p, yy)[0, 1]) if p.std() > 0 and yy.std() > 0 else 0.0
    rank_ic = float(stats.spearmanr(p, yy).correlation)
    return {
        "ic": round(ic, 6),
        "rank_ic": round(rank_ic, 6),
        "pred_nan_ratio": float(np.isnan(pred).mean()),
        "valid_n": int(valid.sum()),
    }


def aggregate_wf_metrics(results: List[WalkForwardResult]) -> Dict[str, float]:
    """把多折的 IC / Rank IC 汇总为平均和 ICIR"""
    ics = [r.metrics.get("ic", 0.0) for r in results if "ic" in r.metrics]
    rank_ics = [r.metrics.get("rank_ic", 0.0) for r in results if "rank_ic" in r.metrics]
    if not ics:
        return {"n_folds": 0}
    return {
        "n_folds": len(ics),
        "mean_ic": round(float(np.mean(ics)), 6),
        "std_ic": round(float(np.std(ics)), 6),
        "icir": round(float(np.mean(ics) / (np.std(ics) + 1e-9)), 4),
        "mean_rank_ic": round(float(np.mean(rank_ics)), 6),
        "positive_ic_ratio": round(float(np.mean([ic > 0 for ic in ics])), 4),
    }


__all__ = [
    "WalkForwardConfig",
    "WalkForwardResult",
    "make_walk_forward_splits",
    "walk_forward_train_predict",
    "aggregate_wf_metrics",
]
