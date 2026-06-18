"""
================================================================================
借鉴项目:
  - Microsoft Qlib (qlib.workflow.record_temp.RollingGen, qlib.contrib.evaluate)
  - AKQuant (akquant.ml.WalkForwardValidation 滚动训练)
借鉴要点:
  - Qlib RollingGen: 用 yaml 配置 N 折滚动 (train_days / test_days / stride),
    自动产出"训练-测试"日期对, 然后在每折上训练 + 评估 + 聚合。
  - AKQuant 强调 "fit -> predict -> backtest" 闭环, on_train_signal 事件。
================================================================================
优化点: jingni-trader 当前 backtest-engine 是一次性 in-sample 评估, 没有
       样本外验证, 极易过拟合。本模块提供轻量级 Walk-Forward 评估:
         1) 滚动切分: 训练窗口 N 天, 测试窗口 M 天, 步长 K 天
         2) 在每折上: 训练 -> 截面打分 -> 多空分层 -> 计算 IC + 多空收益
         3) 跨折聚合: 给出 OOS (样本外) 的 IC 均值, 收益, 回撤
       验证内容:
         a) 正确性: 与单期评估结果对照
         b) 关键: 验证 "训练期学到" vs "测试期直接用" 两种预处理对 OOS IC 的影响
         c) 边界: 折数 = 0 / 训练期太短
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("quant_opt.walk_forward")


# ----------------------------------------------------------------------------
# 借鉴 Qlib RollingGen 的分窗器
# ----------------------------------------------------------------------------
@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # inclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp        # inclusive
    fold_id: int


def generate_folds(dates: pd.DatetimeIndex, train_days: int = 504,
                   test_days: int = 63, stride: int = 63) -> List[Fold]:
    """
    滚动切分 (仿 Qlib RollingGen + AKQuant walk-forward):
      - train_days: 训练窗口交易日数 (默认 504 ≈ 2 年)
      - test_days:  测试窗口交易日数 (默认 63 ≈ 1 季度)
      - stride:     滚动步长 (默认 63, 避免相邻折大量重叠)
    """
    dates = pd.DatetimeIndex(sorted(set(dates))).sort_values()
    if len(dates) < train_days + test_days:
        raise ValueError(
            f"dates length {len(dates)} < train_days {train_days} + test_days {test_days}"
        )
    folds: List[Fold] = []
    fid = 0
    # 第一个 train_start 必须早于 dates[0] + train_days
    start_cursor = dates[0]
    while True:
        train_start = start_cursor
        train_end_idx = dates.get_loc(train_start) + train_days - 1
        if train_end_idx + 1 >= len(dates):
            break
        train_end = dates[train_end_idx]
        test_start_idx = train_end_idx + 1
        test_start = dates[test_start_idx]
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= len(dates):
            test_end = dates[-1]
        else:
            test_end = dates[test_end_idx]
        folds.append(Fold(train_start, train_end, test_start, test_end, fid))
        fid += 1
        # 步进
        next_start_idx = dates.get_loc(train_start) + stride
        if next_start_idx + train_days >= len(dates):
            break
        start_cursor = dates[next_start_idx]
    return folds


# ----------------------------------------------------------------------------
# 借鉴 Qlib Alpha158 思路: 用可解释线性合成 (rolling beta) 给每日截面打分
# ----------------------------------------------------------------------------
def _train_predict_one_fold(fold: Fold, data: pd.DataFrame,
                            factor_cols: List[str],
                            forward_col: str = "ret_forward_1d") -> Dict:
    """
    在一折上:
      训练期: 截面回归 factor -> forward, 得到日度权重 (rolling mean of coefs)
      测试期: 用训练期平均权重打分 -> 形成 alpha_score
      再按日截面计算 IC 与多空 5 分位收益
    """
    train_mask = (data["date"] >= fold.train_start) & (data["date"] <= fold.train_end)
    test_mask = (data["date"] >= fold.test_start) & (data["date"] <= fold.test_end)
    train = data.loc[train_mask].dropna(subset=factor_cols + [forward_col])
    test = data.loc[test_mask].dropna(subset=factor_cols)

    if len(train) < 100 or test.empty:
        return {"fold_id": fold.fold_id, "skipped": True,
                "ic": np.nan, "ls_spread": np.nan}

    X_tr = train[factor_cols].values
    y_tr = train[forward_col].values
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    coefs = model.coef_

    # 训练期 IC (in-sample)
    train_pred = X_tr @ coefs
    from scipy.stats import spearmanr
    train_ic, _ = spearmanr(train_pred, y_tr)

    # 测试期打分 & IC
    X_te = test[factor_cols].values
    test_pred = X_te @ coefs
    test_score = test[["code", "date"]].copy()
    test_score["alpha_score"] = test_pred
    test_score["forward"] = test.loc[test_score.index, forward_col].values \
        if forward_col in test.columns else np.nan

    # 截面 IC
    valid = test_score.dropna(subset=["alpha_score", "forward"])
    if len(valid) < 30:
        return {"fold_id": fold.fold_id, "skipped": True,
                "ic": np.nan, "ls_spread": np.nan, "in_sample_ic": train_ic}
    daily_ic = valid.groupby("date").apply(
        lambda x: spearmanr(x["alpha_score"], x["forward"]).correlation
        if len(x) >= 10 else np.nan,
        include_groups=False,
    )
    oos_ic = float(daily_ic.dropna().mean()) if daily_ic.notna().any() else np.nan

    # 多空 5 分位 spread
    valid["quantile"] = valid.groupby("date")["alpha_score"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
    )
    long_short = valid[valid["quantile"].isin([0, 4])].groupby(
        ["date", "quantile"]
    )["forward"].mean().unstack("quantile")
    ls_spread = (long_short[4] - long_short[0]).mean() if 4 in long_short.columns \
        and 0 in long_short.columns else np.nan

    return {
        "fold_id": fold.fold_id,
        "skipped": False,
        "in_sample_ic": float(train_ic) if not np.isnan(train_ic) else np.nan,
        "ic": oos_ic,
        "ls_spread": float(ls_spread) if not np.isnan(ls_spread) else np.nan,
        "n_test_obs": int(len(valid)),
        "train_range": (str(fold.train_start.date()), str(fold.train_end.date())),
        "test_range": (str(fold.test_start.date()), str(fold.test_end.date())),
    }


# ----------------------------------------------------------------------------
# 借鉴 Qlib / AKQuant: Walk-Forward 主入口
# ----------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    folds: List[Dict] = field(default_factory=list)
    oos_ic_mean: float = 0.0
    oos_ic_std: float = 0.0
    oos_ir: float = 0.0
    oos_ls_spread_mean: float = 0.0
    in_sample_ic_mean: float = 0.0
    overfit_gap: float = 0.0       # = in_sample_ic - oos_ic
    n_folds: int = 0
    n_skipped: int = 0


def run_walk_forward(data: pd.DataFrame, factor_cols: List[str],
                     forward_col: str = "ret_forward_1d",
                     train_days: int = 504, test_days: int = 63,
                     stride: int = 63) -> WalkForwardResult:
    """主入口"""
    folds = generate_folds(data["date"], train_days, test_days, stride)
    fold_results: List[Dict] = []
    for fold in folds:
        r = _train_predict_one_fold(fold, data, factor_cols, forward_col)
        fold_results.append(r)
    valid = [r for r in fold_results if not r.get("skipped", False)]
    is_ic = np.array([r["in_sample_ic"] for r in valid if r.get("in_sample_ic") is not None
                      and not np.isnan(r["in_sample_ic"])])
    oos_ic = np.array([r["ic"] for r in valid if r.get("ic") is not None
                       and not np.isnan(r["ic"])])
    oos_ls = np.array([r["ls_spread"] for r in valid if r.get("ls_spread") is not None
                       and not np.isnan(r["ls_spread"])])

    res = WalkForwardResult(folds=fold_results, n_folds=len(folds),
                            n_skipped=len(folds) - len(valid))
    if len(oos_ic) > 0:
        res.oos_ic_mean = float(oos_ic.mean())
        res.oos_ic_std = float(oos_ic.std(ddof=1)) if len(oos_ic) > 1 else 0.0
        res.oos_ir = (res.oos_ic_mean / res.oos_ic_std) if res.oos_ic_std > 0 else 0.0
    if len(oos_ls) > 0:
        res.oos_ls_spread_mean = float(oos_ls.mean())
    if len(is_ic) > 0:
        res.in_sample_ic_mean = float(is_ic.mean())
        res.overfit_gap = res.in_sample_ic_mean - res.oos_ic_mean
    return res


__all__ = [
    "Fold", "generate_folds", "run_walk_forward", "WalkForwardResult",
]
