"""
Walk-forward 滚动验证框架

借鉴来源：
- Microsoft Qlib: walk-forward rolling train/test split，避免前视偏差
- AKQuant: 内置 Walk-forward Validation 框架

核心改进点：
jingni-trader 当前 strategy-model-engine 一次性 train/test split，无法评估
模型在不同市场环境下的稳定性。Walk-forward 用滚动窗口模拟真实投研流程：
在每个时点只用过去 N 个月数据训练，预测未来 M 个月，循环推进。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    """Walk-forward 配置"""
    train_months: int = 36       # 训练窗口（月）
    test_months: int = 3         # 测试窗口（月）
    step_months: int = 3         # 滚动步长（月）
    min_train_samples: int = 100  # 最小训练样本数


@dataclass
class WalkForwardFold:
    """单个 walk-forward 折"""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int


def generate_walk_forward_folds(
    dates: pd.Series,
    config: WalkForwardConfig,
) -> List[WalkForwardFold]:
    """生成 walk-forward 滚动折"""
    if dates.empty:
        return []

    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    start = dates.iloc[0]
    end = dates.iloc[-1]

    folds: List[WalkForwardFold] = []
    cursor = start + pd.DateOffset(months=config.train_months) - pd.Timedelta(days=1)
    fold_id = 0

    while cursor < end:
        train_start = start if fold_id == 0 else cursor - pd.DateOffset(months=config.train_months)
        train_end = cursor
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=config.test_months) - pd.Timedelta(days=1)
        if test_start > end:
            break

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        train_size = int(train_mask.sum())
        test_size = int(test_mask.sum())

        if train_size >= config.min_train_samples and test_size > 0:
            folds.append(WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=min(test_end, end),
                train_size=train_size,
                test_size=test_size,
            ))

        fold_id += 1
        cursor = cursor + pd.DateOffset(months=config.step_months)

    return folds


def walk_forward_split(
    df: pd.DataFrame,
    date_col: str = "date",
    config: Optional[WalkForwardConfig] = None,
) -> Iterator[Tuple[WalkForwardFold, pd.DataFrame, pd.DataFrame]]:
    """生成 walk-forward 训练/测试集迭代器"""
    if config is None:
        config = WalkForwardConfig()

    if date_col not in df.columns:
        raise ValueError(f"DataFrame 缺少日期列: {date_col}")

    dates = pd.to_datetime(df[date_col])
    folds = generate_walk_forward_folds(dates, config)

    for fold in folds:
        train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
        test_mask = (dates >= fold.test_start) & (dates <= fold.test_end)
        yield fold, df.loc[train_mask].copy(), df.loc[test_mask].copy()


def walk_forward_evaluate(
    df: pd.DataFrame,
    factor_names: List[str],
    forward_col: str = "ret_forward_5d",
    top_k: int = 20,
    config: Optional[WalkForwardConfig] = None,
) -> pd.DataFrame:
    """
    Walk-forward 评估：每折用 IC 加权选 top_k 股票，计算 OOS 收益

    这是一个轻量验证框架，演示 walk-forward 的价值。
    实际生产中应替换为真正的 ML 模型训练 + 预测。

    返回:
        DataFrame[fold_id, train_start, train_end, test_start, test_end,
                  oos_return, oos_sharpe, n_factors_used]
    """
    if config is None:
        config = WalkForwardConfig()

    records = []
    for fold, train_df, test_df in walk_forward_split(df, "date", config):
        if train_df.empty or test_df.empty:
            continue

        # 在训练集上计算每个因子的 IC，选 IC_IR 最高的因子
        ic_weights = {}
        for f in factor_names:
            if f not in train_df.columns or forward_col not in train_df.columns:
                continue
            valid = train_df[["date", f, forward_col]].dropna()
            if len(valid) < 30:
                continue
            ic = valid[f].corr(valid[forward_col])
            ic_series = valid.groupby("date").apply(
                lambda x: x[f].corr(x[forward_col]) if len(x) > 5 else np.nan,
                include_groups=False,
            )
            ic_std = float(ic_series.std()) if not ic_series.empty else 0.0
            ic_ir = ic / ic_std if ic_std > 0 else 0
            ic_weights[f] = ic_ir

        if not ic_weights:
            continue

        # 归一化权重
        total_w = sum(abs(v) for v in ic_weights.values())
        if total_w == 0:
            continue
        ic_weights = {k: v / total_w for k, v in ic_weights.items()}

        # 在测试集上算 alpha_score，选 top_k
        test_df = test_df.copy()
        test_df["alpha_score"] = 0.0
        for f, w in ic_weights.items():
            if f in test_df.columns:
                test_df["alpha_score"] += w * test_df.groupby("date")[f].rank(pct=True)

        # 每日选 top_k，等权持有，计算次日收益
        def _pick(x):
            if x.empty or forward_col not in x.columns:
                return np.nan
            return x.nlargest(top_k, "alpha_score")[forward_col].mean()

        daily_picks = test_df.groupby("date").apply(_pick, include_groups=False).dropna()
        if daily_picks.empty:
            continue

        oos_return = float((1 + daily_picks).prod() - 1)
        std = float(daily_picks.std())
        oos_sharpe = float(daily_picks.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        records.append({
            "fold_id": fold.fold_id,
            "train_start": str(fold.train_start.date()),
            "train_end": str(fold.train_end.date()),
            "test_start": str(fold.test_start.date()),
            "test_end": str(fold.test_end.date()),
            "oos_return": round(oos_return, 4),
            "oos_sharpe": round(oos_sharpe, 4),
            "n_factors_used": len(ic_weights),
            "train_size": fold.train_size,
            "test_size": fold.test_size,
        })

    return pd.DataFrame(records)
