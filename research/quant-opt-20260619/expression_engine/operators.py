"""
因子表达式操作符实现

参考 Qlib ops.py 设计：
- ElemOperator：一元运算符（Log/Abs/...）
- PairOperator：二元运算符（Add/Sub/...）
- Rolling：滚动窗口运算符（Ref/Mean/...）

所有运算符面向"宽表"数据（index=date, columns=multi-index (code, field)），
输入/输出均为 pandas.DataFrame 以便链式组合。
"""
from __future__ import annotations
from typing import Callable, Dict, List, Optional
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 基类
# -----------------------------------------------------------------------------
class ExpressionOps:
    """所有操作符的基类"""

    def __init__(self, *args):
        self.args = args

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._load(data)

    def _load(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.args))})"


class ElemOperator(ExpressionOps):
    """一元运算符：作用于每个元素"""

    def __init__(self, feature: ExpressionOps):
        super().__init__(feature)
        self.feature = feature

    def _load(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(self.feature(data))

    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PairOperator(ExpressionOps):
    """二元运算符：元素级运算"""

    def __init__(self, left: ExpressionOps, right: ExpressionOps):
        super().__init__(left, right)
        self.left = left
        self.right = right

    def _load(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(self.left(data), self.right(data))

    def _apply(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class Rolling(ExpressionOps):
    """滚动窗口运算符：跨时间窗口"""

    def __init__(self, feature: ExpressionOps, window: int):
        super().__init__(feature, window)
        self.feature = feature
        self.window = int(window)

    def _load(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._apply(self.feature(data), self.window)

    def _apply(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        raise NotImplementedError


# -----------------------------------------------------------------------------
# 一元运算符
# -----------------------------------------------------------------------------
class Abs(ElemOperator):
    def _apply(self, df):
        return df.abs()


class Log(ElemOperator):
    def _apply(self, df):
        return np.log(df.replace(0, np.nan))


class Sign(ElemOperator):
    def _apply(self, df):
        return np.sign(df)


class Sqrt(ElemOperator):
    def _apply(self, df):
        return np.sqrt(df.abs())


class Power(ElemOperator):
    """Power(feature, n)"""
    def __init__(self, feature: ExpressionOps, n: float):
        super().__init__(feature)
        self.feature = feature
        self.n = float(n)

    def _load(self, data):
        return self.feature(data) ** self.n


class Rank(ElemOperator):
    """截面排名：每个时间点内各股票排 % 排名"""
    def _apply(self, df):
        return df.rank(axis=1, pct=True)


# -----------------------------------------------------------------------------
# 二元运算符
# -----------------------------------------------------------------------------
class Add(PairOperator):
    def _apply(self, left, right):
        return left + right


class Sub(PairOperator):
    def _apply(self, left, right):
        return left - right


class Mul(PairOperator):
    def _apply(self, left, right):
        return left * right


class Div(PairOperator):
    def _apply(self, left, right):
        return left / right.replace(0, np.nan)


class Greater(PairOperator):
    def _apply(self, left, right):
        return (left > right).astype(float)


class Less(PairOperator):
    def _apply(self, left, right):
        return (left < right).astype(float)


class Equal(PairOperator):
    def _apply(self, left, right):
        return (left == right).astype(float)


class And(PairOperator):
    def _apply(self, left, right):
        return ((left != 0) & (right != 0)).astype(float)


class Or(PairOperator):
    def _apply(self, left, right):
        return ((left != 0) | (right != 0)).astype(float)


class Not(ElemOperator):
    def _apply(self, df):
        return (df == 0).astype(float)


class If(PairOperator):
    """If(condition, true_branch, false_branch)"""
    def __init__(self, condition: ExpressionOps, true_branch: ExpressionOps, false_branch: ExpressionOps):
        super().__init__(true_branch, false_branch)
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def _load(self, data):
        cond = self.condition(data).astype(float)
        t = self.true_branch(data)
        f = self.false_branch(data)
        return t.where(cond > 0, f)


# -----------------------------------------------------------------------------
# 滚动窗口运算符
# -----------------------------------------------------------------------------
class Ref(Rolling):
    """Ref(feature, n): n 期前的值"""
    def _apply(self, df, window):
        return df.shift(window)


class Delta(Rolling):
    """Delta(feature, n): feature - Ref(feature, n)"""
    def _apply(self, df, window):
        return df - df.shift(window)


class Mean(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).mean()


class Std(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).std()


class Sum(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).sum()


class Max(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).max()


class Min(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).min()


class Med(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).median()


class Mad(Rolling):
    """Mean absolute deviation"""
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )


class Quantile(Rolling):
    def _apply(self, df, window):
        return df.rolling(window, min_periods=max(2, window // 2)).quantile(0.5)


class Slope(Rolling):
    """线性回归斜率"""
    def _apply(self, df, window):
        def _slope(x):
            if len(x) < 2 or np.all(np.isnan(x)):
                return np.nan
            y = x - np.nanmean(x)
            t = np.arange(len(x)) - np.nanmean(np.arange(len(x)))
            denom = np.sum(t * t)
            if denom == 0:
                return np.nan
            return np.sum(t * y) / denom
        return df.rolling(window, min_periods=max(2, window // 2)).apply(_slope, raw=True)


class Rsquare(Rolling):
    """线性回归 R²"""
    def _apply(self, df, window):
        def _r2(x):
            if len(x) < 2 or np.all(np.isnan(x)):
                return np.nan
            y = x - np.nanmean(x)
            t = np.arange(len(x)) - np.nanmean(np.arange(len(x)))
            denom_t = np.sum(t * t)
            if denom_t == 0:
                return np.nan
            cov = np.sum(t * y)
            ssr = (cov ** 2) / denom_t
            sst = np.sum(y * y)
            if sst == 0:
                return 0.0
            return ssr / sst
        return df.rolling(window, min_periods=max(2, window // 2)).apply(_r2, raw=True)


class Resi(Rolling):
    """回归残差"""
    def _apply(self, df, window):
        def _resi(x):
            if len(x) < 2 or np.all(np.isnan(x)):
                return np.nan
            y = x - np.nanmean(x)
            t = np.arange(len(x)) - np.nanmean(np.arange(len(x)))
            denom_t = np.sum(t * t)
            if denom_t == 0:
                return np.nan
            slope = np.sum(t * y) / denom_t
            intercept = np.nanmean(x) - slope * np.nanmean(np.arange(len(x)))
            last = x[-1]
            return last - (intercept + slope * (len(x) - 1))
        return df.rolling(window, min_periods=max(2, window // 2)).apply(_resi, raw=True)


class Corr(Rolling):
    """Corr(feature_a, feature_b, window)"""
    def __init__(self, left, right, window):
        super().__init__(left, window)
        self.left = left
        self.right = right
        self.window = int(window)

    def _load(self, data):
        a = self.left(data)
        b = self.right(data)
        return a.rolling(self.window, min_periods=max(2, self.window // 2)).corr(b)


class Cov(Rolling):
    def __init__(self, left, right, window):
        super().__init__(left, window)
        self.left = left
        self.right = right
        self.window = int(window)

    def _load(self, data):
        a = self.left(data)
        b = self.right(data)
        return a.rolling(self.window, min_periods=max(2, self.window // 2)).cov(b)


class SumIf(Rolling):
    """SumIf(condition, feature, window): 满足条件时在窗口内求和"""
    def __init__(self, condition, feature, window):
        super().__init__(condition, window)
        self.condition = condition
        self.feature = feature
        self.window = int(window)

    def _load(self, data):
        cond = self.condition(data)
        feat = self.feature(data)
        masked = feat.where(cond > 0)
        return masked.rolling(self.window, min_periods=1).sum()


# -----------------------------------------------------------------------------
# 常量包装
# -----------------------------------------------------------------------------
class Constant(ExpressionOps):
    """数值常量"""
    def __init__(self, value: float):
        super().__init__(value)
        self.value = float(value)

    def _load(self, data):
        # 使用 data 的扁平列：若 data 是 MultiIndex 列，取第二级（code）作为列
        if isinstance(data.columns, pd.MultiIndex):
            # 取第一层的第一列对应的第二层值
            codes = data.columns.get_level_values(1).unique()
            return pd.DataFrame(
                self.value, index=data.index, columns=codes
            )
        return pd.DataFrame(self.value, index=data.index, columns=data.columns)


class Feature(ExpressionOps):
    """特征引用：$close, $open, $high, $low, $volume, $amount, $vwap, $returns"""
    _ALIAS = {
        "$close": "close", "$open": "open", "$high": "high", "$low": "low",
        "$volume": "volume", "$amount": "amount", "$vwap": "vwap",
        "$returns": "returns", "$change_pct": "change_pct",
    }

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        if name not in self._ALIAS:
            raise KeyError(f"未知特征: {name}; 可用: {list(self._ALIAS)}")

    def _load(self, data):
        col = self._ALIAS[self.name]
        if col not in data.columns.get_level_values(0):
            raise KeyError(f"数据中缺少特征列: {col}")
        return data[col]


# 字段名 -> 内部表达
FEATURE_ALIAS = Feature._ALIAS
