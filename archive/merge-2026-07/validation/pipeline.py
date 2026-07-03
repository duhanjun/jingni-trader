"""
数据预处理 Pipeline（借鉴来源: AKQuant Pipeline / sklearn Pipeline）

设计动机
========
当前 jingni-trader 的模型训练入口在
    skills/strategy-model-engine/scripts/base/base_model.py
仅暴露 train/predict/save/load 接口,缺少"训练-测试严格隔离"的预处理。

潜在问题:
    1. 标准化 / 缺失值填补在 train+test 联合计算统计量 -> 前视偏差
    2. 行业中性化、市值中性化如果用全样本回归 -> 信息泄露
    3. 没有"transform-only"接口保证测试集只使用训练集统计量

借鉴 AKQuant ML Guide §5 "Preventing Data Leakage: Using Pipeline":
    https://akquant.akfamily.xyz/en/advanced/ml/#5-preventing-data-league-using-pipeline

借鉴 sklearn Pipeline 设计模式:
    - fit_transform 在训练集
    - transform 在测试集
    - 禁止在 transform 阶段再学习任何统计量

本模块提供
==========
1. TransformerStep:  抽象 step, fit / transform 接口分离
2. Pipeline:         多 step 串联, 保证训练/测试隔离
3. StandardScalerByCode: 按股票分组的横截面 Z-Score
4. IndustryNeutralizer: 行业中性化 (OLS 残差)
5. MissingValueFiller: 缺失值填充 (中位数 / 均值 / 前向填充)
6. WINSORIZE 工具: 极端值截尾
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------
class TransformerStep(ABC):
    """Pipeline 中的一个变换步骤"""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TransformerStep":
        """在训练数据上学习统计量"""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用学到的变换,严禁再学习新统计量"""

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# 具体实现
# ---------------------------------------------------------------------------
class MissingValueFiller(TransformerStep):
    """缺失值填充器,只在 fit 时计算中位数/均值"""

    def __init__(self, columns: Optional[Sequence[str]] = None, strategy: str = "median") -> None:
        if strategy not in ("median", "mean", "ffill"):
            raise ValueError(f"不支持的 strategy: {strategy}")
        self.columns = columns
        self.strategy = strategy
        self._fill_values: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "MissingValueFiller":
        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c not in X.columns:
                continue
            if self.strategy == "median":
                self._fill_values[c] = float(X[c].median())
            elif self.strategy == "mean":
                self._fill_values[c] = float(X[c].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.strategy == "ffill":
            for c in (self.columns or out.columns):
                if c in out.columns:
                    out[c] = out[c].ffill().bfill()
        else:
            for c, v in self._fill_values.items():
                if c in out.columns:
                    out[c] = out[c].fillna(v)
        return out


class Winsorizer(TransformerStep):
    """分位数截尾,按列在 fit 时记录分位"""

    def __init__(self, columns: Optional[Sequence[str]] = None,
                 lower: float = 0.01, upper: float = 0.99) -> None:
        self.columns = columns
        self.lower = lower
        self.upper = upper
        self._bounds: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Winsorizer":
        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c not in X.columns:
                continue
            lo = float(X[c].quantile(self.lower))
            hi = float(X[c].quantile(self.upper))
            self._bounds[c] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (lo, hi) in self._bounds.items():
            if c in out.columns:
                out[c] = out[c].clip(lower=lo, upper=hi)
        return out


class CrossSectionalScaler(TransformerStep):
    """
    横截面标准化:  按 date 分组,组内 Z-Score。
    借鉴 qlib 的 `CSZScoreNorm` / VectorBT 的横截面处理。
    """

    def __init__(self, columns: Optional[Sequence[str]] = None,
                 by: str = "date") -> None:
        self.columns = columns
        self.by = by
        # 训练时的全局 fallback (组内 std==0 时使用)
        self._global_std: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CrossSectionalScaler":
        cols = self.columns or X.select_dtypes(include=[np.number]).columns.tolist()
        for c in cols:
            if c in X.columns:
                self._global_std[c] = float(X[c].std(ddof=0)) or 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        cols = self.columns or [c for c in out.columns if c not in (self.by, "code", "date")]
        for c in cols:
            if c not in out.columns or self.by not in out.columns:
                continue
            g = out.groupby(self.by)[c]
            mean = g.transform("mean")
            std = g.transform("std").fillna(0.0)
            # 训练集的全局 std 作为兜底
            std = std.where(std > 1e-12, self._global_std.get(c, 1.0))
            out[c] = (out[c] - mean) / std
        return out


class IndustryNeutralizer(TransformerStep):
    """
    行业中性化: 对每行用线性回归剔除行业虚拟变量的影响。
    训练时 fit 回归系数, transform 时应用。
    """

    def __init__(self, factor_col: str, industry_col: str = "industry",
                 date_col: str = "date") -> None:
        self.factor_col = factor_col
        self.industry_col = industry_col
        self.date_col = date_col
        self._models: Dict[Any, LinearRegression] = {}
        self._model_columns: Dict[Any, List[str]] = {}
        self._industries_seen: List[Any] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IndustryNeutralizer":
        if self.factor_col not in X.columns or self.industry_col not in X.columns:
            return self
        self._industries_seen = sorted(X[self.industry_col].dropna().unique().tolist())
        for date, g in X.groupby(self.date_col):
            g = g.dropna(subset=[self.factor_col, self.industry_col])
            if len(g) < 3 or g[self.industry_col].nunique() < 2:
                continue
            dummies = pd.get_dummies(g[self.industry_col], drop_first=True)
            if dummies.shape[1] == 0:
                continue
            model = LinearRegression()
            model.fit(dummies.values, g[self.factor_col].values)
            self._models[date] = model
            self._model_columns[date] = dummies.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.factor_col not in out.columns:
            return out
        residuals = np.full(len(out), np.nan)
        for date, idx in out.groupby(self.date_col).indices.items():
            g = out.iloc[idx]
            if date in self._models:
                # 用 fit 时记录的列名构造 dummies, 保证 transform 阶段
                # 的列序与 fit 完全一致
                dummies = pd.get_dummies(g[self.industry_col], drop_first=True)
                # 补齐 fit 阶段出现但 transform 缺失的行业
                for col in self._model_columns[date]:
                    if col not in dummies.columns:
                        dummies[col] = 0
                # 剔除 transform 出现但 fit 未见的行业
                extra_cols = [c for c in dummies.columns
                              if c not in self._model_columns[date]]
                if extra_cols:
                    dummies = dummies.drop(columns=extra_cols)
                # 按 fit 阶段列序排列
                dummies = dummies[self._model_columns[date]]
                pred = self._models[date].predict(dummies.values)
                residuals[idx] = g[self.factor_col].values - pred
            else:
                # 测试集出现训练集未见过日期 -> 仅做去均值
                residuals[idx] = g[self.factor_col].values - g[self.factor_col].mean()
        out[self.factor_col] = residuals
        return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Pipeline:
    """
    多步骤 Pipeline, 强制 fit_transform / transform 分离

    用法
    ----
    >>> pipe = Pipeline([
    ...     ("imputer", MissingValueFiller(strategy="median")),
    ...     ("winsor", Winsorizer(lower=0.01, upper=0.99)),
    ...     ("csz", CrossSectionalScaler()),
    ... ])
    >>> train_X = pipe.fit_transform(train_X)
    >>> test_X = pipe.transform(test_X)   # 严禁再次 fit
    """

    def __init__(self, steps: List[Tuple[str, TransformerStep]]) -> None:
        if not steps:
            raise ValueError("Pipeline 至少包含一个 step")
        self.steps = steps
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Pipeline":
        Xc = X
        for _, step in self.steps:
            step.fit(Xc, y)
            Xc = step.transform(Xc)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Pipeline 必须先 fit 再 transform")
        out = X
        for _, step in self.steps:
            out = step.transform(out)
        return out

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
