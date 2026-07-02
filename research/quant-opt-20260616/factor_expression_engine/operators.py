"""
因子表达式算子注册表

所有 DSL 算子在此注册。算子签名统一为：
    op_func(df: pd.DataFrame, *args) -> pd.Series

DSL 语法（参考 Qlib / AKQuant）：
    变量       : $close, $open, $high, $low, $volume, $amount, $vwap, $change_pct
    时间序列   : Ts_Mean(x, d) / Ts_Std(x, d) / Ts_Sum(x, d) /
                 Ts_Max(x, d) / Ts_Min(x, d) / Ts_Rank(x, d) /
                 Delay(x, d) / Delta(x, d) / Ts_ArgMax(x, d) / Ts_ArgMin(x, d)
    横截面     : Rank(x) / Scale(x) / ZScore(x)
    数学/逻辑  : Abs / Log / Sign / Sqrt / Power(x, p) /
                 Add / Sub / Mul / Div / Greater / Less / Equal / And / Or / If
    别名       : Ref = Delay; Mean = Ts_Mean; Std = Ts_Std
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 列名到 DataFrame 列名的映射。$close -> close
VAR_MAP = {
    "$close": "close",
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$volume": "volume",
    "$amount": "amount",
    "$vwap": "vwap",
    "$change_pct": "change_pct",
    "$turnover_rate": "turnover_rate",
}


def _resolve_var(name: str, df: pd.DataFrame) -> pd.Series:
    """把 DSL 变量 $xxx 解析为 DataFrame 中对应列。"""
    if name not in VAR_MAP:
        raise KeyError(f"未知变量: {name}，支持的变量: {list(VAR_MAP.keys())}")
    col = VAR_MAP[name]
    if col not in df.columns:
        raise KeyError(f"DataFrame 缺少列: {col}")
    return df[col]


# ---------------------------------------------------------------------------
# 时间序列算子
# ---------------------------------------------------------------------------

def ts_mean(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=2).mean()
    )


def ts_std(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=2).std()
    )


def ts_sum(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=1).sum()
    )


def ts_max(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=1).max()
    )


def ts_min(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=1).min()
    )


def ts_rank(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    """当前值在过去 d 天内的横排（百分位）。"""
    d = int(d)

    def _rank_last(s: pd.Series) -> pd.Series:
        ranks = s.rank(method="average", pct=True)
        return ranks.iloc[-1]  # 取最新一天的横排

    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=2).apply(_rank_last, raw=False)
    )


def ts_argmax(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=2).apply(
            np.argmax, raw=True
        )
    )


def ts_argmin(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].transform(
        lambda s: s.rolling(d, min_periods=2).apply(
            np.argmin, raw=True
        )
    )


def delay(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    """Ref/Delay：d 天前的值。"""
    d = int(d)
    return df.groupby("code", group_keys=False)[x.name].shift(d)


def delta(df: pd.DataFrame, x: pd.Series, d: int) -> pd.Series:
    d = int(d)
    return x - delay(df, x, d)


# ---------------------------------------------------------------------------
# 横截面算子（按 date 分组）
# ---------------------------------------------------------------------------

def cs_rank(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return df.groupby("date", group_keys=False)[x.name].rank(pct=True)


def cs_scale(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    """横截面缩放到绝对值之和为 1。"""
    return df.groupby("date", group_keys=False)[x.name].transform(
        lambda s: s / s.abs().sum() if s.abs().sum() != 0 else 0.0
    )


def cs_zscore(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return df.groupby("date", group_keys=False)[x.name].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() != 0 else 0.0
    )


# ---------------------------------------------------------------------------
# 数学 / 逻辑算子
# ---------------------------------------------------------------------------

def op_abs(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return x.abs()


def op_log(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def op_sign(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return np.sign(x)


def op_sqrt(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    return np.sqrt(x)


def op_add(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return x + y


def op_sub(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return x - y


def op_mul(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return x * y


def op_div(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return x / y.replace(0, np.nan)


def op_greater(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return (x > y).astype(float)


def op_less(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return (x < y).astype(float)


def op_equal(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return (x == y).astype(float)


def op_and(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return ((x != 0) & (y != 0)).astype(float)


def op_or(df: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
    return ((x != 0) | (y != 0)).astype(float)


def op_if(df: pd.DataFrame, cond: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where(cond != 0, a, b)


def op_power(df: pd.DataFrame, x: pd.Series, p) -> pd.Series:
    return x ** float(p)


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------

# 算子元数据：(name, arity, func)，
# arity = -1 表示变参，>=0 表示固定参数个数。
# 自动从元数据生成 call 接口。
OPERATORS: Dict[str, Callable] = {
    # 时间序列
    "Ts_Mean": ts_mean,
    "Ts_Std": ts_std,
    "Ts_Sum": ts_sum,
    "Ts_Max": ts_max,
    "Ts_Min": ts_min,
    "Ts_Rank": ts_rank,
    "Ts_ArgMax": ts_argmax,
    "Ts_ArgMin": ts_argmin,
    "Delay": delay,
    "Delta": delta,
    # 别名
    "Ref": delay,
    "Mean": ts_mean,
    "Std": ts_std,
    # 横截面
    "Rank": cs_rank,
    "Scale": cs_scale,
    "ZScore": cs_zscore,
    # 数学
    "Abs": op_abs,
    "Log": op_log,
    "Sign": op_sign,
    "Sqrt": op_sqrt,
    "Add": op_add,
    "Sub": op_sub,
    "Mul": op_mul,
    "Div": op_div,
    "Power": op_power,
    # 逻辑
    "Greater": op_greater,
    "Less": op_less,
    "Equal": op_equal,
    "And": op_and,
    "Or": op_or,
    "If": op_if,
}

# 算子 arity（参数数量，不含 df）
ARITY = {
    "Ts_Mean": 2, "Ts_Std": 2, "Ts_Sum": 2, "Ts_Max": 2, "Ts_Min": 2,
    "Ts_Rank": 2, "Ts_ArgMax": 2, "Ts_ArgMin": 2,
    "Delay": 2, "Delta": 2,
    "Ref": 2, "Mean": 2, "Std": 2,
    "Rank": 1, "Scale": 1, "ZScore": 1,
    "Abs": 1, "Log": 1, "Sign": 1, "Sqrt": 1,
    "Add": 2, "Sub": 2, "Mul": 2, "Div": 2, "Power": 2,
    "Greater": 2, "Less": 2, "Equal": 2, "And": 2, "Or": 2,
    "If": 3,
}


def register_operator(name: str, func: Callable, arity: int) -> None:
    """注册自定义算子。"""
    if name in OPERATORS:
        raise ValueError(f"算子 {name} 已存在")
    OPERATORS[name] = func
    ARITY[name] = arity
