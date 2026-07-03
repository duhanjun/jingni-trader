"""
因子注册机制（验证版）

借鉴来源：Microsoft Qlib 的声明式因子框架
          (https://github.com/microsoft/qlib)

对比 jingni-trader 现有实现：
  - pandas_ta_calculator.py 用 if/elif 硬编码因子分发，新增因子需修改分发逻辑；
  - factor-engine/engine.py 的 compute_a_share_factors 把所有因子写死在一个方法里。

本模块提供装饰器注册机制：
  - 用 @register_factor 注册因子，自动收集到全局注册表；
  - 计算时按名称批量调度，无需修改分发代码；
  - 支持因子元信息（方向、参数、依赖列），便于 IC 分析与文档化。

设计目标：可扩展性、可维护性，新增因子零侵入。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Any, Optional
import functools

import numpy as np
import pandas as pd


# 全局因子注册表：name -> FactorInfo
_FACTOR_REGISTRY: Dict[str, "FactorInfo"] = {}


class FactorInfo:
    """因子元信息容器。"""

    def __init__(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
        direction: int = 0,
        params: Optional[Dict[str, Any]] = None,
        requires: Optional[List[str]] = None,
        description: str = "",
    ):
        self.name = name
        self.func = func
        self.direction = direction          # 1=正向(越大越买), -1=反向, 0=中性
        self.params = params or {}
        self.requires = requires or ["close"]  # 依赖的原始列
        self.description = description

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算单个因子（按 code 分组应用 func）。"""
        return data.groupby("code", group_keys=False).apply(self.func, include_groups=False)


def register_factor(
    name: str,
    direction: int = 0,
    params: Optional[Dict[str, Any]] = None,
    requires: Optional[List[str]] = None,
    description: str = "",
) -> Callable:
    """装饰器：注册一个因子计算函数。

    被装饰函数签名: func(df: DataFrame) -> Series
    其中 df 为单只股票按 date 排序的 OHLCV 子表。
    """
    def decorator(func: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
        info = FactorInfo(name, func, direction, params, requires, description)
        _FACTOR_REGISTRY[name] = info

        @functools.wraps(func)
        def wrapper(df: pd.DataFrame) -> pd.Series:
            return func(df)
        wrapper.factor_info = info  # 便于反射访问
        return wrapper

    return decorator


def list_factors() -> List[str]:
    """返回已注册的全部因子名。"""
    return sorted(_FACTOR_REGISTRY.keys())


def get_factor_info(name: str) -> FactorInfo:
    """获取因子元信息。"""
    if name not in _FACTOR_REGISTRY:
        raise KeyError(f"因子未注册: {name}，已注册: {list_factors()}")
    return _FACTOR_REGISTRY[name]


def compute_factors(data: pd.DataFrame, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
    """批量计算已注册因子。

    参数:
        data:         长表 OHLCV (code, date, open, high, low, close, volume, ...)
        factor_names: 需计算的因子列表，None 表示全部

    返回:
        DataFrame: (code, date, [各因子列])
    """
    if data.empty:
        return data[["code", "date"]].copy()

    if factor_names is None:
        factor_names = list_factors()

    result = data[["code", "date"]].copy()
    df = data.sort_values(["code", "date"]).copy()

    for name in factor_names:
        info = _FACTOR_REGISTRY.get(name)
        if info is None:
            raise KeyError(f"因子未注册: {name}")
        try:
            result[name] = info.compute(df).values
        except Exception as e:  # 单因子失败不影响其它因子
            result[name] = np.nan
            print(f"[factor_registry] 计算 {name} 失败: {e}")
    return result


# ---------------------------------------------------------------------------
# 内置因子：用注册机制声明（演示零侵入扩展）
# ---------------------------------------------------------------------------

@register_factor(
    "reversal_20d",
    direction=-1,
    requires=["close"],
    description="20日反转因子：过去20日收益取负",
)
def _reversal_20d(df: pd.DataFrame) -> pd.Series:
    return -df["close"].pct_change(20)


@register_factor(
    "volatility_20d",
    direction=-1,
    requires=["close"],
    description="20日波动率因子",
)
def _volatility_20d(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rolling(20, min_periods=10).std()


@register_factor(
    "turnover_change",
    direction=0,
    requires=["turnover_rate"],
    description="换手率动量：5日均换手 / 20日均换手 - 1",
)
def _turnover_change(df: pd.DataFrame) -> pd.Series:
    if "turnover_rate" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tr = df["turnover_rate"]
    m5 = tr.rolling(5, min_periods=3).mean()
    m20 = tr.rolling(20, min_periods=5).mean()
    return m5 / m20.replace(0, np.nan) - 1


@register_factor(
    "volume_ratio",
    direction=0,
    requires=["volume"],
    description="量比：当日量 / 20日均量",
)
def _volume_ratio(df: pd.DataFrame) -> pd.Series:
    m20 = df["volume"].rolling(20, min_periods=5).mean()
    return df["volume"] / m20.replace(0, np.nan)


@register_factor(
    "ma_bias_20",
    direction=0,
    requires=["close"],
    description="20日均线乖离：(close - MA20) / MA20",
)
def _ma_bias_20(df: pd.DataFrame) -> pd.Series:
    ma20 = df["close"].rolling(20, min_periods=10).mean()
    return (df["close"] - ma20) / ma20.replace(0, np.nan)
