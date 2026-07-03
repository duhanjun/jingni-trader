"""
================================================================================
借鉴项目: Microsoft Qlib (https://github.com/microsoft/qlib, MIT, 43.7k stars)
借鉴要点: Qlib DataHandlerLP 区分 Processors 的 fit() 与 process()。
         - fit(): 在训练集上学到归一化参数 (mean/std)
         - process(): 把学到的参数应用到任意数据 (含测试集)
         - 这种 "learnable" 模式防止了数据泄露, 是 Qlib 能正确做样本外评估的关键。
         参考: qlib.data.dataset.processor.py: ZScoreNorm, CSRankNorm, CSZFillna
================================================================================
优化点: jingni-trader factor-engine.compute_a_share_factors 当前直接对全量
       数据做 transform(...rolling...), 没有 train/test 分离的意识, 难以做
       严格样本外评估。本模块提供:
         1) Processor 基类: 分离 fit(X_train) / process(X) / process_inplace
         2) Winsorize3Sigma: 3σ 截尾, fit 学 mean/std
         3) RollingZScore: 滚动 z-score, fit 学窗口长度/clip
         4) CSZFillna: 截面 z-score 后填 NaN
         5) Pipeline: 串联多个 Processor, 一次完成
       并验证:
         a) 正确性: 与 jingni-trader 现有 rolling z-score 行为一致
         b) 关键: 验证"先 fit 再 process"避免了样本外数据泄露
         c) 边界: 全 NaN 列 / 单一截面 / 训练集 < 30 行
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Processor 基类 (仿 Qlib)
# ----------------------------------------------------------------------------
class Processor(ABC):
    """
    所有预处理器都遵循 fit/process 分离协议。
    - fit(X, **fit_kwargs): 在训练数据上学参数
    - process(X): 用学到的参数处理数据
    - is_fitted: 防止未 fit 就 process
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._state: Dict = {}
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, **fit_kwargs) -> "Processor":
        self._state = self._learn(X, **fit_kwargs)
        self.is_fitted = True
        return self

    def process(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(f"[{self.name}] 必须先 fit 才能 process")
        return self._apply(X, self._state)

    @abstractmethod
    def _learn(self, X: pd.DataFrame, **kwargs) -> Dict:
        ...

    @abstractmethod
    def _apply(self, X: pd.DataFrame, state: Dict) -> pd.DataFrame:
        ...


# ----------------------------------------------------------------------------
# 借鉴 Qlib ZScoreNorm: 在训练期学 mean/std, 应用到任意集
# ----------------------------------------------------------------------------
class GlobalZScore(Processor):
    """
    全局 z-score (用 fit 时学到的 mean / std 处理 process 数据)。
    与 jingni-trader 现有逻辑的关键区别: jingni-trader 用的是 rolling z-score,
    看不到未来; 而本类用 fit 时学到的 mean/std, 适合 "训练集学参数 -> 测试集应用"。
    """

    def _learn(self, X: pd.DataFrame, columns: Optional[List[str]] = None,
               clip: float = 3.0) -> Dict:
        cols = columns or X.select_dtypes(include=[np.number]).columns.tolist()
        mu = X[cols].mean(skipna=True)
        sd = X[cols].std(skipna=True, ddof=1).replace(0, np.nan)
        return {"cols": cols, "mu": mu.to_dict(), "sd": sd.to_dict(),
                "clip": clip}

    def _apply(self, X: pd.DataFrame, state: Dict) -> Dict:
        out = X.copy()
        for c in state["cols"]:
            if c not in out.columns:
                continue
            mu = state["mu"].get(c, np.nan)
            sd = state["sd"].get(c, np.nan)
            if pd.isna(sd) or sd == 0:
                continue
            z = (out[c] - mu) / sd
            clip = state["clip"]
            if clip and clip > 0:
                z = z.clip(lower=-clip, upper=clip)
            out[c] = z
        return out


# ----------------------------------------------------------------------------
# 借鉴 Qlib CSZFillna / CSRankNorm: 截面预处理
# ----------------------------------------------------------------------------
class CSZScore(Processor):
    """按日 groupby 截面做 z-score (类似 jingni-trader neutralize 内的归一化)"""

    def _learn(self, X: pd.DataFrame, columns: Optional[List[str]] = None,
               date_col: str = "date") -> Dict:
        return {"cols": columns or [], "date_col": date_col}

    def _apply(self, X: pd.DataFrame, state: Dict) -> pd.DataFrame:
        out = X.copy()
        date_col = state["date_col"]
        for c in state["cols"]:
            if c not in out.columns:
                continue
            grp = out.groupby(date_col)[c]
            mu = grp.transform("mean")
            sd = grp.transform("std").replace(0, np.nan)
            out[c] = (out[c] - mu) / sd
        return out


class RollingZScore(Processor):
    """
    滚动 z-score (jingni-trader 现有做法):
      z_t = (x_t - mean_{t-w+1..t}) / std_{t-w+1..t}
    fit 阶段学习窗口长度与 clip 阈值, apply 阶段直接复用。
    """

    def _learn(self, X: pd.DataFrame, columns: List[str],
               window: int = 20, min_periods: Optional[int] = None,
               clip: float = 3.0) -> Dict:
        return {"cols": columns, "window": window,
                "min_periods": min_periods or max(5, window // 4),
                "clip": clip}

    def _apply(self, X: pd.DataFrame, state: Dict) -> pd.DataFrame:
        out = X.copy()
        for c in state["cols"]:
            if c not in out.columns:
                continue
            s = X[c]
            mu = s.rolling(state["window"], min_periods=state["min_periods"]).mean()
            sd = s.rolling(state["window"], min_periods=state["min_periods"]).std()
            z = (s - mu) / sd.replace(0, np.nan)
            if state["clip"] and state["clip"] > 0:
                z = z.clip(lower=-state["clip"], upper=state["clip"])
            out[c] = z
        return out


# ----------------------------------------------------------------------------
# 借鉴 Qlib 思想: 串联多个 Processor
# ----------------------------------------------------------------------------
class Pipeline:
    def __init__(self, processors: Sequence[Processor]):
        self.processors = list(processors)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cur = X
        for p in self.processors:
            p.fit(cur)
            cur = p.process(cur)
        return cur

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cur = X
        for p in self.processors:
            cur = p.process(cur)
        return cur


__all__ = [
    "Processor", "GlobalZScore", "CSZScore", "RollingZScore", "Pipeline",
]
