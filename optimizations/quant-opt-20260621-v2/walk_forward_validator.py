"""
Walk-Forward 验证模块

借鉴来源:
- Microsoft Qlib: workflow 中内置 walk-forward 训练/测试切分
- QuantConnect Lean: Rolling Window Walk-Forward 分析
- VectorBT: walk-forward optimization 支持

优化点:
jingni-trader backtest-engine 的 SKILL.md 声称支持 "Walk-Forward 分析检测过拟合"，
config.py 中也定义了 WF_TRAIN_MONTHS / WF_TEST_MONTHS 参数，
但实际代码中并未实现。这是文档与实现的不一致。

本模块实现滚动窗口 Walk-Forward 验证:
1. 将历史数据按时间切分为多个 (train, test) 窗口
2. 在 train 窗口训练/调参，在 test 窗口验证
3. 拼接所有 test 窗口的收益，得到样本外真实表现
4. 对比 in-sample 与 out-of-sample 表现，检测过拟合
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger("walk-forward")


@dataclass
class WalkForwardWindow:
    """单个 walk-forward 窗口"""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_months: int = 36,
    test_months: int = 12,
    step_months: Optional[int] = None,
) -> List[WalkForwardWindow]:
    """
    生成 walk-forward 滚动窗口

    参数:
        dates: 所有日期序列
        train_months: 训练窗口月数
        test_months: 测试窗口月数
        step_months: 滚动步长(默认等于 test_months，即无重叠)

    返回:
        WalkForwardWindow 列表
    """
    if step_months is None:
        step_months = test_months

    dates = pd.DatetimeIndex(sorted(dates.unique()))
    if len(dates) == 0:
        return []

    start = dates[0]
    end = dates[-1]

    windows = []
    win_id = 0
    cur_train_start = start

    while True:
        train_end = cur_train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        if test_start > end:
            break
        if test_end > end:
            test_end = end

        # 确保窗口内有实际数据
        train_dates = dates[(dates >= cur_train_start) & (dates <= train_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]
        if len(train_dates) < 20 or len(test_dates) < 5:
            break

        windows.append(WalkForwardWindow(
            window_id=win_id,
            train_start=cur_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        win_id += 1
        cur_train_start = cur_train_start + pd.DateOffset(months=step_months)

    return windows


def walk_forward_validate(
    data: pd.DataFrame,
    factor_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: List[str],
    train_months: int = 36,
    test_months: int = 12,
    ic_type: str = "spearman",
) -> Dict[str, Any]:
    """
    Walk-Forward 因子稳定性验证

    在每个 train 窗口计算因子 IC，检查该 IC 在后续 test 窗口是否保持。

    参数:
        data: 原始行情数据(含 date 列)
        factor_df: 因子数据(含 code, date, [因子列])
        forward_returns: 前瞻收益(含 code, date, ret_forward_5d)
        factor_names: 待验证因子名
        train_months: 训练窗口月数
        test_months: 测试窗口月数

    返回:
        {
            "windows": [...],
            "in_sample_ic": {因子: [各窗口train期IC]},
            "out_of_sample_ic": {因子: [各窗口test期IC]},
            "stability": {因子: {ic_decay_ratio, consistent_ratio}},
            "overfitting_warning": bool,
        }
    """
    try:
        from .vectorized_ic_analysis import calc_ic_vectorized
    except ImportError:
        from vectorized_ic_analysis import calc_ic_vectorized

    data["date"] = pd.to_datetime(data["date"])
    factor_df = factor_df.copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    forward_returns = forward_returns.copy()
    forward_returns["date"] = pd.to_datetime(forward_returns["date"])

    dates = data["date"]
    windows = generate_walk_forward_windows(dates, train_months, test_months)
    logger.info(f"生成 {len(windows)} 个 walk-forward 窗口")

    in_sample_ic: Dict[str, List[float]] = {f: [] for f in factor_names}
    out_sample_ic: Dict[str, List[float]] = {f: [] for f in factor_names}

    for win in windows:
        # 训练期 IC
        train_factor = factor_df[(factor_df["date"] >= win.train_start) & (factor_df["date"] <= win.train_end)]
        train_fwd = forward_returns[(forward_returns["date"] >= win.train_start) & (forward_returns["date"] <= win.train_end)]
        if not train_factor.empty and not train_fwd.empty:
            train_ic = calc_ic_vectorized(train_factor, train_fwd, factor_names,
                                          "ret_forward_5d", ic_type)
            for f in factor_names:
                in_sample_ic[f].append(float(train_ic[f].mean()) if f in train_ic else 0.0)

        # 测试期 IC
        test_factor = factor_df[(factor_df["date"] >= win.test_start) & (factor_df["date"] <= win.test_end)]
        test_fwd = forward_returns[(forward_returns["date"] >= win.test_start) & (forward_returns["date"] <= win.test_end)]
        if not test_factor.empty and not test_fwd.empty:
            test_ic = calc_ic_vectorized(test_factor, test_fwd, factor_names,
                                         "ret_forward_5d", ic_type)
            for f in factor_names:
                out_sample_ic[f].append(float(test_ic[f].mean()) if f in test_ic else 0.0)

    # 稳定性分析
    stability: Dict[str, Dict[str, float]] = {}
    overfitting_factors = []
    for f in factor_names:
        is_ic = np.array(in_sample_ic[f])
        oos_ic = np.array(out_sample_ic[f])
        if len(is_ic) == 0 or len(oos_ic) == 0:
            stability[f] = {"ic_decay_ratio": 0.0, "consistent_ratio": 0.0, "is_mean": 0.0, "oos_mean": 0.0}
            continue

        is_mean = float(np.mean(is_ic))
        oos_mean = float(np.mean(oos_ic))
        decay_ratio = (is_mean - oos_mean) / abs(is_mean) if is_mean != 0 else 0.0
        # 符号一致比例
        min_len = min(len(is_ic), len(oos_ic))
        consistent = float(np.mean(np.sign(is_ic[:min_len]) == np.sign(oos_ic[:min_len])))
        stability[f] = {
            "ic_decay_ratio": round(decay_ratio, 4),
            "consistent_ratio": round(consistent, 4),
            "is_mean": round(is_mean, 6),
            "oos_mean": round(oos_mean, 6),
        }
        # 过拟合判定: 样本外 IC 衰减超过 50% 或符号不一致率 > 50%
        if decay_ratio > 0.5 or consistent < 0.5:
            overfitting_factors.append(f)

    return {
        "n_windows": len(windows),
        "windows": [
            {"id": w.window_id,
             "train": [str(w.train_start.date()), str(w.train_end.date())],
             "test": [str(w.test_start.date()), str(w.test_end.date())]}
            for w in windows
        ],
        "in_sample_ic": in_sample_ic,
        "out_of_sample_ic": out_sample_ic,
        "stability": stability,
        "overfitting_warning": len(overfitting_factors) > 0,
        "overfitting_factors": overfitting_factors,
    }
