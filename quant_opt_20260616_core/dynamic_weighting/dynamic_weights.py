"""
动态因子加权具体实现
====================

所有函数都接受 ``ic_history: pd.DataFrame`` 作为输入：

    ic_history.index = pd.DatetimeIndex
    ic_history.columns = factor names

返回当前期的因子权重。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


def icir_decay_weights(
    ic_history: pd.DataFrame,
    halflife: int = 60,
    min_periods: int = 20,
    floor: float = 0.0,
) -> Dict[str, float]:
    """IC-IR 指数衰减加权

    parameters
    ----------
    ic_history:
        因子历史 IC 序列 (index=date, columns=factor)
    halflife:
        指数衰减半衰期（交易日），halflife=60 表示 60 个交易日前的 IC
        权重衰减为 1/2。
    min_periods:
        至少需要的样本量，少于该值则视为该因子无效
    floor:
        最小权重地板（防止某因子权重为 0 导致跟踪误差累积）

    references
    ----------
    AlphaForge (Shi et al., 2024) 中"temporal performance for selection"
    思想。
    """
    if ic_history.empty:
        return {}

    # 只保留满足最小样本量的列
    valid = ic_history.dropna(axis=1, thresh=min_periods)
    if valid.empty:
        return {}

    # 计算每列的指数衰减均值与标准差
    weights: Dict[str, float] = {}
    decay = 0.5 ** (1.0 / halflife)
    n = len(valid)
    # 越近权重越大
    exp_weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    # 转为 pandas.Series, 与 valid 的 index 对齐
    exp_w_series = pd.Series(exp_weights, index=valid.index)

    for col in valid.columns:
        series = valid[col].dropna()
        if len(series) < min_periods:
            continue
        aligned_w = exp_w_series.loc[series.index]
        w_sum = aligned_w.sum()
        if w_sum <= 0:
            continue
        mean = (series * aligned_w).sum() / w_sum
        # 衰减方差
        var = (aligned_w * (series - mean) ** 2).sum() / w_sum
        std = float(np.sqrt(max(var, 1e-12)))
        icir = mean / std
        weights[col] = float(icir)

    if not weights:
        return {}

    # 取绝对值作为非负权重 (因子的方向通过排序解决)
    abs_w = {k: abs(v) for k, v in weights.items()}
    total = sum(abs_w.values())
    if total <= 0:
        n_factors = len(abs_w)
        return {k: 1.0 / n_factors for k in abs_w}

    norm = {k: v / total for k, v in abs_w.items()}
    if floor > 0:
        # 应用地板再归一化
        floored = {k: max(v, floor) for k, v in norm.items()}
        s = sum(floored.values())
        norm = {k: v / s for k, v in floored.items()}
    return norm


def softmax_ic_weights(
    ic_history: pd.DataFrame,
    lookback: int = 60,
    temperature: float = 0.01,
) -> Dict[str, float]:
    """softmax 形式的 IC 加权

    使用最近 ``lookback`` 期 IC 均值，经 softmax 转为权重。
    ``temperature`` 越小, 越接近 argmax (winner-takes-all)。

    论文使用类似"temporal performance of factors for selection"的方式
    动态调整权重, softmax 是其中一种最简单的实现。
    """
    if ic_history.empty:
        return {}
    recent = ic_history.tail(lookback)
    means = recent.mean()
    means = means.dropna()
    if means.empty:
        return {}
    scaled = means / max(temperature, 1e-6)
    e = np.exp(scaled - scaled.max())
    p = e / e.sum()
    return {factor: float(p.loc[factor]) for factor in means.index}


class DynamicFactorWeighting:
    """动态加权器

    使用示例
    --------
    >>> weighting = DynamicFactorWeighting(method="icir_decay", halflife=60)
    >>> weights = weighting.compute(ic_history)
    """

    METHODS: Dict[str, Callable[..., Dict[str, float]]] = {
        "icir_decay": icir_decay_weights,
        "softmax_ic": softmax_ic_weights,
    }

    def __init__(
        self,
        method: str = "icir_decay",
        halflife: int = 60,
        lookback: int = 60,
        min_periods: int = 20,
        temperature: float = 0.01,
        floor: float = 0.0,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(
                f"unknown method: {method}, choices={list(self.METHODS)}"
            )
        self.method = method
        self.halflife = halflife
        self.lookback = lookback
        self.min_periods = min_periods
        self.temperature = temperature
        self.floor = floor

    def compute(self, ic_history: pd.DataFrame) -> Dict[str, float]:
        if self.method == "icir_decay":
            return icir_decay_weights(
                ic_history,
                halflife=self.halflife,
                min_periods=self.min_periods,
                floor=self.floor,
            )
        if self.method == "softmax_ic":
            return softmax_ic_weights(
                ic_history,
                lookback=self.lookback,
                temperature=self.temperature,
            )
        raise ValueError(f"unknown method: {self.method}")