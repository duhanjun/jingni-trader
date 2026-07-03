"""
Walk-Forward 滚动验证模块

借鉴来源：
- VectorBT PRO 的 walk-forward 优化与交叉验证
- QuantConnect LEAN 的滚动回测
- 论文 "The Probability of Backtest Overfitting" (Bailey et al.)

优化点：
原 jingni-trader 的回测引擎只做单次全样本回测，容易过拟合。
config.py 中已定义 WF_TRAIN_MONTHS / WF_TEST_MONTHS 但未实现。

本模块实现滚动窗口验证：
1. 将历史数据按时间切分为多个 train + test 窗口
2. 在 train 窗口优化参数
3. 在 test 窗口验证（样本外）
4. 拼接所有 test 窗口的收益，得到样本外综合表现

支持：
- 滚动窗口（固定训练期 + 固定测试期，逐步前进）
- 扩展窗口（训练期逐步扩大）
- Purged 交叉验证（训练与测试间留出 embargo 期，防止数据泄漏）
- CSCV（Combinatorial Symmetric Cross Validation）过拟合检测
"""
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd


def generate_walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_months: int = 36,
    test_months: int = 12,
    step_months: Optional[int] = None,
    expand: bool = False,
    embargo_days: int = 5,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    生成 Walk-Forward 窗口

    参数:
        dates: 全部日期索引
        train_months: 训练期月数
        test_months: 测试期月数
        step_months: 步进月数（默认 = test_months）
        expand: True=扩展窗口，False=滚动窗口
        embargo_days: 训练与测试间的隔离天数（防止数据泄漏）

    返回:
        [(train_dates, test_dates), ...]
    """
    if len(dates) == 0:
        return []

    if step_months is None:
        step_months = test_months

    dates = pd.DatetimeIndex(sorted(dates))
    start = dates[0]
    end = dates[-1]

    # 按月生成边界
    month_starts = pd.date_range(start=start, end=end, freq="MS")
    if len(month_starts) < train_months + test_months:
        return []

    windows = []
    i = 0
    while i + train_months + test_months <= len(month_starts):
        if expand:
            train_start = month_starts[0]
        else:
            train_start = month_starts[i]
        train_end = month_starts[i + train_months] - pd.Timedelta(days=1)
        test_start = month_starts[i + train_months] + pd.Timedelta(days=embargo_days)

        # test_end: 最后一个测试月的月末
        test_end_idx = i + train_months + test_months
        if test_end_idx < len(month_starts):
            test_end = month_starts[test_end_idx] - pd.Timedelta(days=1)
        else:
            # 到达数据末尾，用最后一个日期
            test_end = end

        if test_end > end:
            test_end = end

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)

        train_dates = dates[train_mask]
        test_dates = dates[test_mask]

        if len(train_dates) > 0 and len(test_dates) > 0:
            windows.append((train_dates, test_dates))

        i += step_months

    return windows


def walk_forward_validate(
    data: pd.DataFrame,
    factor_df: pd.DataFrame,
    param_grid: Dict[str, list],
    train_months: int = 36,
    test_months: int = 12,
    backtest_fn: Optional[Callable] = None,
    metric_key: str = "sharpe_ratio",
    expand: bool = False,
    embargo_days: int = 5,
) -> Dict[str, Any]:
    """
    Walk-Forward 滚动验证

    参数:
        data: 行情数据
        factor_df: 因子数据
        param_grid: 参数网格
        train_months: 训练期月数
        test_months: 测试期月数
        backtest_fn: 自定义回测函数（默认用向量化回测）
        metric_key: 用于选择最优参数的指标
        expand: 扩展窗口 vs 滚动窗口
        embargo_days: 隔离期

    返回:
        {
            "windows": 每个窗口的详情,
            "oos_returns": 样本外日收益率拼接,
            "oos_metrics": 样本外综合指标,
            "is_oos_consistency": 样本内外一致性,
            "overfitting_probability": 过拟合概率,
        }
    """
    from .vectorized_backtest import vectorized_backtest, _generate_signals
    from .enhanced_metrics import calc_full_metrics
    from itertools import product

    if backtest_fn is None:
        backtest_fn = vectorized_backtest

    # 获取全部日期
    all_dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    windows = generate_walk_forward_windows(
        all_dates, train_months, test_months, expand=expand, embargo_days=embargo_days
    )

    if not windows:
        return {
            "windows": [],
            "oos_returns": pd.Series(dtype=float),
            "oos_metrics": {},
            "is_oos_consistency": {},
            "overfitting_probability": None,
            "error": "数据不足以生成 walk-forward 窗口",
        }

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    window_results = []
    oos_returns_list = []
    is_metrics_list = []  # 样本内
    oos_metrics_list = []  # 样本外

    for w_idx, (train_dates, test_dates) in enumerate(windows):
        # 切分数据
        train_data = data[data["date"].isin(train_dates)]
        test_data = data[data["date"].isin(test_dates)]
        train_factor = factor_df[factor_df["date"].isin(train_dates)]
        test_factor = factor_df[factor_df["date"].isin(test_dates)]

        if train_data.empty or test_data.empty:
            continue

        # 在训练集上网格搜索最优参数
        best_score = -np.inf
        best_params = None
        best_is_metrics = None

        for combo in product(*values):
            params = dict(zip(keys, combo))
            signals = _generate_signals(train_factor, params)
            if signals.empty:
                continue
            try:
                bt = backtest_fn(train_data, signals)
                if bt["equity_curve"].empty:
                    continue
                eq = bt["equity_curve"].set_index("date")["equity"]
                metrics = calc_full_metrics(eq, bt["returns"])
                score = metrics.get(metric_key, -np.inf)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_is_metrics = metrics
            except Exception:
                continue

        if best_params is None:
            continue

        # 在测试集上验证
        test_signals = _generate_signals(test_factor, best_params)
        if test_signals.empty:
            continue

        try:
            oos_bt = backtest_fn(test_data, test_signals)
            if oos_bt["equity_curve"].empty:
                continue
            oos_eq = oos_bt["equity_curve"].set_index("date")["equity"]
            oos_metrics = calc_full_metrics(oos_eq, oos_bt["returns"])
            oos_returns_list.append(oos_bt["returns"])
            oos_metrics_list.append(oos_metrics)
            is_metrics_list.append(best_is_metrics)
        except Exception:
            continue

        window_results.append({
            "window_idx": w_idx,
            "train_start": str(train_dates[0].date()),
            "train_end": str(train_dates[-1].date()),
            "test_start": str(test_dates[0].date()),
            "test_end": str(test_dates[-1].date()),
            "best_params": best_params,
            "is_sharpe": best_is_metrics.get("sharpe_ratio", 0),
            "oos_sharpe": oos_metrics.get("sharpe_ratio", 0),
            "is_return": best_is_metrics.get("annual_return", 0),
            "oos_return": oos_metrics.get("annual_return", 0),
        })

    # 拼接样本外收益
    if oos_returns_list:
        oos_returns = pd.concat(oos_returns_list).sort_index()
        oos_equity = (1 + oos_returns).cumprod()
        oos_metrics = calc_full_metrics(oos_equity, oos_returns)
    else:
        oos_returns = pd.Series(dtype=float)
        oos_metrics = {}

    # 一致性分析：样本内 vs 样本外
    consistency = _calc_is_oos_consistency(is_metrics_list, oos_metrics_list)

    # 过拟合概率（CSCV 简化版）
    overfit_prob = _calc_overfitting_probability(is_metrics_list, oos_metrics_list)

    return {
        "windows": window_results,
        "oos_returns": oos_returns,
        "oos_metrics": oos_metrics,
        "is_oos_consistency": consistency,
        "overfitting_probability": overfit_prob,
    }


def _calc_is_oos_consistency(
    is_metrics: List[Dict],
    oos_metrics: List[Dict],
) -> Dict[str, Any]:
    """计算样本内外的表现一致性"""
    if not is_metrics or not oos_metrics:
        return {}

    is_sharpe = [m.get("sharpe_ratio", 0) for m in is_metrics]
    oos_sharpe = [m.get("sharpe_ratio", 0) for m in oos_metrics]

    is_mean = float(np.mean(is_sharpe))
    oos_mean = float(np.mean(oos_sharpe))

    # 一致性：样本外 / 样本内（越接近 1 越好）
    ratio = oos_mean / is_mean if is_mean != 0 else 0.0

    # 相关性
    if len(is_sharpe) > 1:
        corr = float(np.corrcoef(is_sharpe, oos_sharpe)[0, 1])
    else:
        corr = 0.0

    return {
        "is_sharpe_mean": round(is_mean, 4),
        "oos_sharpe_mean": round(oos_mean, 4),
        "degradation_ratio": round(float(oos_mean - is_mean), 4),
        "oos_is_ratio": round(ratio, 4),
        "is_oos_correlation": round(corr, 4),
    }


def _calc_overfitting_probability(
    is_metrics: List[Dict],
    oos_metrics: List[Dict],
) -> Optional[float]:
    """
    简化版过拟合概率估计

    借鉴 CSCV（Combinatorial Symmetric Cross Validation）思想：
    如果样本内排名与样本外排名相关性低，说明过拟合严重。

    返回 0-1 之间的概率值，越高越可能过拟合
    """
    if len(is_metrics) < 3 or len(oos_metrics) < 3:
        return None

    is_sharpe = np.array([m.get("sharpe_ratio", 0) for m in is_metrics])
    oos_sharpe = np.array([m.get("sharpe_ratio", 0) for m in oos_metrics])

    # 标准化
    is_std = is_sharpe.std()
    oos_std = oos_sharpe.std()
    if is_std == 0 or oos_std == 0:
        return 0.0

    is_z = (is_sharpe - is_sharpe.mean()) / is_std
    oos_z = (oos_sharpe - oos_sharpe.mean()) / oos_std

    # PBO（Probability of Backtest Overfitting）简化：
    # 样本外收益为负的比例
    neg_oos_ratio = float((oos_sharpe < 0).mean())

    # 样本内外排名不一致的比例
    is_rank = pd.Series(is_sharpe).rank().values
    oos_rank = pd.Series(oos_sharpe).rank().values
    rank_mismatch = float((is_rank != oos_rank).mean())

    # 综合过拟合概率
    pbo = 0.5 * neg_oos_ratio + 0.5 * rank_mismatch
    return round(min(max(pbo, 0.0), 1.0), 4)
