"""
Walk-forward 验证框架
=====================

借鉴来源：
- Microsoft RD-Agent(Q) (NeurIPS 2025) 的因子-模型联合优化中的滚动训练循环
- AkQuant 内置 Walk-forward Validation 框架
- Qlib 的 `qrun` 中 train/valid/test 时间切分

设计目标：
- 把 jingni-trader 中缺失的"样本外验证"机制补齐
- 滚动训练 (rolling) / 锚定扩展 (anchored) 两种模式
- 输出：每折的 IC、绩效指标、跨折均值±标准差
- 与 strategy-model-engine / factor-engine 解耦，可独立调用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WFold:
    """单折信息"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WFConfig:
    train_window_days: int = 252 * 2      # 训练窗口
    test_window_days: int = 63            # 测试窗口（约一个季度）
    step_days: int = 63                   # 滑动步长
    anchored: bool = False                # True: 锚定起点；False: 滚动
    min_train_days: int = 252             # 最小训练期
    purge_gap_days: int = 5               # purge gap，避免标签泄漏


class WalkForwardSplitter:
    """时间序列 walk-forward 切分器"""

    def __init__(self, config: Optional[WFConfig] = None):
        self.config = config or WFConfig()

    def split(self, dates: pd.DatetimeIndex) -> List[WFold]:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
        if len(dates) < self.config.min_train_days + self.config.test_window_days:
            raise ValueError(
                f"数据不足：需要至少 {self.config.min_train_days + self.config.test_window_days} 天，实际 {len(dates)}"
            )

        folds: List[WFold] = []
        cfg = self.config
        anchor_start = dates[0]  # 锚定模式从最早日期开始
        first_test_start = anchor_start + pd.Timedelta(days=cfg.min_train_days)
        first_test_end = first_test_start + pd.Timedelta(days=cfg.test_window_days)

        cursor = first_test_start
        fold_id = 0
        while cursor + pd.Timedelta(days=cfg.test_window_days) <= dates[-1]:
            test_start = cursor
            test_end = cursor + pd.Timedelta(days=cfg.test_window_days)
            if cfg.anchored:
                train_start = anchor_start
            else:
                train_start = cursor - pd.Timedelta(days=cfg.train_window_days)
            train_end = test_start - pd.Timedelta(days=cfg.purge_gap_days + 1)

            if train_end <= train_start:
                cursor += pd.Timedelta(days=cfg.step_days)
                continue
            if train_end < anchor_start:
                cursor += pd.Timedelta(days=cfg.step_days)
                continue

            folds.append(WFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
            fold_id += 1
            cursor += pd.Timedelta(days=cfg.step_days)
        return folds


# ──────────────────────────────────────────────────────────────────
# 评估器
# ──────────────────────────────────────────────────────────────────

def evaluate_factor_oos(
    factor_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    在样本外区间内评估因子的 IC。

    参数:
        factor_df:          long-format, [code, date, <factor_col>]
        forward_returns_df: long-format, [code, date, ret_forward_Nd]
        ic_type:            "spearman" | "pearson"

    返回:
        {
            "ic_mean": float, "ic_std": float, "ic_ir": float,
            "ic_positive_ratio": float, "n_dates": int
        }
    """
    from scipy import stats

    if factor_df.empty or forward_returns_df.empty:
        return {}

    factor_col = [c for c in factor_df.columns if c not in ("code", "date")]
    if not factor_col:
        return {}
    factor_col = factor_col[0]

    fwd_col = [c for c in forward_returns_df.columns if c not in ("code", "date")]
    if not fwd_col:
        return {}
    fwd_col = fwd_col[0]

    merged = factor_df[["code", "date", factor_col]].merge(
        forward_returns_df[["code", "date", fwd_col]], on=["code", "date"], how="inner"
    ).dropna(subset=[factor_col, fwd_col])

    if merged.empty:
        return {}

    ics: List[float] = []
    for dt, g in merged.groupby("date"):
        if len(g) < 10:
            continue
        if ic_type == "spearman":
            r, _ = stats.spearmanr(g[factor_col], g[fwd_col])
        else:
            r, _ = stats.pearsonr(g[factor_col], g[fwd_col])
        if not np.isnan(r):
            ics.append(float(r))

    if not ics:
        return {}
    arr = np.array(ics)
    return {
        "ic_mean": float(arr.mean()),
        "ic_std": float(arr.std()),
        "ic_ir": float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0,
        "ic_positive_ratio": float((arr > 0).mean()),
        "n_dates": len(arr),
    }


def walk_forward_evaluate(
    factor_formula: str,
    data: pd.DataFrame,
    forward_period: int = 5,
    config: Optional[WFConfig] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Walk-forward 评估一条因子公式

    参数:
        factor_formula:   Alpha101 风格公式字符串
        data:             long-format 行情 (code, date, open, high, low, close, volume, amount)
        forward_period:   前视期（天）
        config:           WFConfig
        extra:            注入表达式引擎的额外字段

    返回:
        {
            "n_folds": int,
            "per_fold": List[Dict],
            "aggregate": {ic_mean, ic_std, ic_ir, ...}
        }
    """
    from quant_opt.factor_expr import compile_factor

    cfg = config or WFConfig()
    dates = pd.to_datetime(data["date"].unique())
    splitter = WalkForwardSplitter(cfg)
    folds = splitter.split(dates)

    # 预计算 forward returns（整段，避免逐折重复 shift）
    fwd = data.sort_values(["code", "date"]).copy()
    fwd["fwd_ret"] = fwd.groupby("code")["close"].shift(-forward_period) / fwd["close"] - 1

    per_fold: List[Dict[str, Any]] = []

    for fold in folds:
        test_mask = (fwd["date"] >= fold.test_start) & (fwd["date"] <= fold.test_end)
        test_data = fwd[test_mask].copy()
        if test_data.empty:
            continue

        try:
            factor_values = compile_factor(factor_formula, test_data, extra=extra)
        except Exception as e:
            per_fold.append({
                "fold_id": fold.fold_id,
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "error": str(e),
            })
            continue

        # 把 (code, date) MultiIndex 还原
        if isinstance(factor_values.index, pd.MultiIndex):
            fv = factor_values.reset_index()
            fv.columns = ["code", "date", "factor_value"]
        else:
            fv = pd.DataFrame({
                "code": test_data["code"].values,
                "date": test_data["date"].values,
                "factor_value": factor_values.values,
            })

        merged = fv.merge(test_data[["code", "date", "fwd_ret"]], on=["code", "date"], how="inner").dropna()
        eval_res = evaluate_factor_oos(
            merged.rename(columns={"factor_value": "factor"})[["code", "date", "factor"]],
            merged.rename(columns={"fwd_ret": "ret_forward"})[["code", "date", "ret_forward"]],
            ic_type="spearman",
        )
        per_fold.append({
            "fold_id": fold.fold_id,
            "train_start": str(fold.train_start.date()),
            "train_end": str(fold.train_end.date()),
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
            **eval_res,
        })

    # 聚合
    valid = [f for f in per_fold if "ic_mean" in f]
    if valid:
        ic_means = np.array([f["ic_mean"] for f in valid])
        ic_stds = np.array([f["ic_std"] for f in valid])
        ic_irs = np.array([f["ic_ir"] for f in valid])
        aggregate = {
            "ic_mean_avg": float(ic_means.mean()),
            "ic_mean_std": float(ic_means.std()),
            "ic_std_avg": float(ic_stds.mean()),
            "ic_ir_avg": float(ic_irs.mean()),
            "ic_ir_std": float(ic_irs.std()),
            "n_folds_valid": len(valid),
            "ic_positive_fold_ratio": float((ic_means > 0).mean()),
        }
    else:
        aggregate = {"n_folds_valid": 0}

    return {
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "aggregate": aggregate,
    }
