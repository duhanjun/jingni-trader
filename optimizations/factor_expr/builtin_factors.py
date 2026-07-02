"""
内置因子库 (Builtin Factors)
============================
借鉴来源: Qlib Alpha158 因子分类 + jingni-trader 现有因子定义

将 jingni-trader factor-engine/engine.py:48-117 中硬编码的因子,
改造为通过 @register_factor 装饰器注册的可扩展因子库。

因子分类 (参考 Qlib Alpha158):
- momentum: 动量类 (ret_5d, ret_20d)
- reversal: 反转类 (reversal_5d, reversal_20d)
- volume: 量价类 (volume_ratio, turnover_change)
- volatility: 波动类 (volatility_20d, amplitude)
- liquidity: 流动性类 (turnover_5d, turnover_20d)
- size: 市值类 (lncap)
- technical: 技术指标类 (rsi_14, ma_diff)

每个因子通过表达式引擎计算, 实现声明式定义 + 注册表管理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .expression_engine import ExpressionEngine
from .factor_registry import FactorRegistry


def register_builtins(registry: FactorRegistry | None = None) -> FactorRegistry:
    """注册所有内置因子到注册表, 返回注册表实例。

    幂等: 重复调用安全。
    """
    registry = registry or FactorRegistry.instance()
    engine = ExpressionEngine()

    # --- 动量类 ---
    if "ret_5d" not in registry:
        registry.register(
            name="ret_5d",
            category="momentum",
            direction=1,
            fields=("close",),
            window=5,
            description="5日收益率",
        )(lambda df: engine.compute("ROC($close, 5)", df))

    if "ret_20d" not in registry:
        registry.register(
            name="ret_20d",
            category="momentum",
            direction=1,
            fields=("close",),
            window=20,
            description="20日收益率",
        )(lambda df: engine.compute("ROC($close, 20)", df))

    # --- 反转类 (方向为 -1) ---
    if "reversal_5d" not in registry:
        registry.register(
            name="reversal_5d",
            category="reversal",
            direction=-1,
            fields=("close",),
            window=5,
            description="5日反转 (负收益率)",
        )(lambda df: -engine.compute("ROC($close, 5)", df))

    if "reversal_20d" not in registry:
        registry.register(
            name="reversal_20d",
            category="reversal",
            direction=-1,
            fields=("close",),
            window=20,
            description="20日反转 (负收益率)",
        )(lambda df: -engine.compute("ROC($close, 20)", df))

    # --- 波动类 ---
    if "volatility_20d" not in registry:
        registry.register(
            name="volatility_20d",
            category="volatility",
            direction=-1,
            fields=("close",),
            window=20,
            description="20日收益率波动率",
        )(lambda df: engine.compute("STD(ROC($close, 1), 20)", df))

    if "amplitude_20d" not in registry:
        registry.register(
            name="amplitude_20d",
            category="volatility",
            direction=-1,
            fields=("high", "low", "close"),
            window=20,
            description="20日振幅均值",
        )(lambda df: engine.compute("MA(($high - $low) / $close, 20)", df))

    # --- 量价类 ---
    if "volume_ratio" not in registry:
        registry.register(
            name="volume_ratio",
            category="volume",
            direction=1,
            fields=("volume",),
            window=20,
            description="成交量比 (当日/MA20)",
        )(lambda df: engine.compute("$volume / MA($volume, 20)", df))

    if "turnover_change" not in registry:
        registry.register(
            name="turnover_change",
            category="volume",
            direction=1,
            fields=("turnover_rate",),
            window=5,
            description="换手率变化 (5日/20日)",
        )(lambda df: engine.compute("MA($turnover_rate, 5) / MA($turnover_rate, 20)", df))

    # --- 流动性类 ---
    if "turnover_5d" not in registry:
        registry.register(
            name="turnover_5d",
            category="liquidity",
            direction=-1,
            fields=("turnover_rate",),
            window=5,
            description="5日平均换手率",
        )(lambda df: engine.compute("MA($turnover_rate, 5)", df))

    if "turnover_20d" not in registry:
        registry.register(
            name="turnover_20d",
            category="liquidity",
            direction=-1,
            fields=("turnover_rate",),
            window=20,
            description="20日平均换手率",
        )(lambda df: engine.compute("MA($turnover_rate, 20)", df))

    # --- 市值类 ---
    if "lncap" not in registry:
        registry.register(
            name="lncap",
            category="size",
            direction=-1,
            fields=("total_mv",),
            window=0,
            description="对数总市值",
        )(lambda df: np.log(df["total_mv"].where(df["total_mv"] > 0, np.nan)))

    # --- 技术指标类 ---
    if "ma_diff" not in registry:
        registry.register(
            name="ma_diff",
            category="technical",
            direction=1,
            fields=("close",),
            window=20,
            description="MA20 - MA5 趋势因子",
        )(lambda df: engine.compute("MA($close, 20) - MA($close, 5)", df))

    if "rsi_14" not in registry:
        registry.register(
            name="rsi_14",
            category="technical",
            direction=-1,
            fields=("close",),
            window=14,
            description="14日 RSI",
        )(lambda df: engine.compute("RSI($close, 14)", df))

    if "price_momentum" not in registry:
        registry.register(
            name="price_momentum",
            category="technical",
            direction=1,
            fields=("close",),
            window=20,
            description="价格动量 (截面排名)",
        )(lambda df: engine.compute("CSRank(ROC($close, 20))", df))

    return registry
