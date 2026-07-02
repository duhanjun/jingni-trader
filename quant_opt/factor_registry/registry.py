"""
因子注册表（Factor Registry）—— 装饰器驱动的因子库

借鉴自：
- Microsoft Qlib (qlib/data/ops.py)：通过模块加载时注册算子
- AlphaAgent (RndmVariableQ/AlphaAgent)：用 AST 描述因子，并提供统一接口

设计目标：
1. 第三方用户可通过 @register_factor("xxx") 装饰器一行注册自定义因子
2. 统一因子元信息：方向、参数、文档
3. 与 Expression Engine 互通（因子实现可以是函数或 Expression 树）
4. 支持按名称 / 标签 / 类别批量查询
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class FactorDirection(IntEnum):
    """因子方向：1 = 正向（值越大越好），-1 = 反向，0 = 中性"""
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0


@dataclass
class FactorSpec:
    """单个因子的元信息"""
    name: str
    func: Callable
    description: str = ""
    direction: FactorDirection = FactorDirection.NEUTRAL
    category: str = "custom"
    tags: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    is_expression: bool = False  # func 返回 Expression 对象？

    def __repr__(self) -> str:  # pragma: no cover
        return f"<FactorSpec {self.name!r} dir={self.direction.name} cat={self.category}>"


class FactorRegistry:
    """全局因子注册表，单例"""

    def __init__(self):
        self._specs: Dict[str, FactorSpec] = {}

    # ------------------------------------------------------------------
    # 注册 API
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        *,
        description: str = "",
        direction: FactorDirection = FactorDirection.NEUTRAL,
        category: str = "custom",
        tags: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """装饰器：注册一个因子函数。

        被装饰函数签名约定：
            def my_factor(data: pd.DataFrame, **params) -> pd.Series

        或：
            def my_factor_expr(**params) -> Expression
        """
        def decorator(func: Callable) -> Callable:
            sig = inspect.signature(func)
            declared_params = {k: v.default for k, v in sig.parameters.items()
                               if v.default is not inspect.Parameter.empty}
            merged_params = {**declared_params, **(params or {})}
            is_expr = func.__name__.endswith("_expr") or "Expression" in str(
                sig.return_annotation
            )
            spec = FactorSpec(
                name=name,
                func=func,
                description=description or func.__doc__ or "",
                direction=direction,
                category=category,
                tags=tags or [],
                params=merged_params,
                is_expression=is_expr,
            )
            if name in self._specs:
                raise ValueError(f"因子 {name!r} 已被注册，请使用不同名称或先 unregister")
            self._specs[name] = spec

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def add(self, spec: FactorSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"因子 {spec.name!r} 已被注册")
        self._specs[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._specs.pop(name, None)

    # ------------------------------------------------------------------
    # 查询 API
    # ------------------------------------------------------------------
    def get(self, name: str) -> FactorSpec:
        if name not in self._specs:
            raise KeyError(
                f"因子 {name!r} 不在注册表中；可用因子: {sorted(self._specs)}"
            )
        return self._specs[name]

    def list(self) -> List[str]:
        return sorted(self._specs)

    def list_by_category(self, category: str) -> List[str]:
        return [n for n, s in self._specs.items() if s.category == category]

    def list_by_tag(self, tag: str) -> List[str]:
        return [n for n, s in self._specs.items() if tag in s.tags]

    def filter(self, *, direction: Optional[FactorDirection] = None,
               category: Optional[str] = None,
               tag: Optional[str] = None) -> List[FactorSpec]:
        result = list(self._specs.values())
        if direction is not None:
            result = [s for s in result if s.direction == direction]
        if category is not None:
            result = [s for s in result if s.category == category]
        if tag is not None:
            result = [s for s in result if tag in s.tags]
        return result

    def summary(self) -> pd.DataFrame:
        rows = []
        for s in self._specs.values():
            rows.append({
                "name": s.name,
                "category": s.category,
                "direction": s.direction.name,
                "tags": ",".join(s.tags),
                "is_expression": s.is_expression,
                "description": s.description[:60],
            })
        return pd.DataFrame(rows)

    def compute(self, name: str, data: pd.DataFrame, **override_params) -> pd.Series:
        """计算单个因子"""
        spec = self.get(name)
        params = {**spec.params, **override_params}
        if spec.is_expression:
            from quant_opt.expression_engine import Evaluator
            expr = spec.func(**params)
            return Evaluator(data).eval(expr)
        return spec.func(data, **params)

    def compute_many(
        self,
        names: List[str],
        data: pd.DataFrame,
        shared_evaluator: bool = True,
    ) -> pd.DataFrame:
        """批量计算多个因子。shared_evaluator=True 时，Expression 之间共享子表达式。"""
        if shared_evaluator and any(self.get(n).is_expression for n in names):
            from quant_opt.expression_engine import Evaluator
            ev = Evaluator(data)
            out = {}
            for n in names:
                spec = self.get(n)
                if spec.is_expression:
                    out[n] = ev.eval(spec.func(**spec.params))
                else:
                    out[n] = spec.func(data, **spec.params)
            return pd.DataFrame(out, index=data.index)
        return pd.DataFrame({n: self.compute(n, data) for n in names}, index=data.index)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)


# 全局注册表单例
REGISTRY = FactorRegistry()


# 便捷装饰器
def register_factor(
    name: str,
    *,
    description: str = "",
    direction: FactorDirection = FactorDirection.NEUTRAL,
    category: str = "custom",
    tags: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Callable:
    return REGISTRY.register(
        name=name,
        description=description,
        direction=direction,
        category=category,
        tags=tags,
        params=params,
    )


# ---------------------------------------------------------------------------
# 内置 A 股因子库（对照 jingni-trader.compute_a_share_factors 覆盖）
# ---------------------------------------------------------------------------
def _safe_pct(s: pd.Series, periods: int) -> pd.Series:
    return s.groupby(s.index.get_level_values(0) if isinstance(s.index, pd.MultiIndex) else s.index).pct_change(periods) if False else s.pct_change(periods)


@register_factor(
    "ret_1d",
    description="1日收益率",
    direction=FactorDirection.POSITIVE,
    category="momentum",
    tags=["basic", "turnover"],
)
def ret_1d(data: pd.DataFrame) -> pd.Series:
    return data.groupby("code")["close"].pct_change()


@register_factor(
    "ret_5d",
    description="5日收益率",
    direction=FactorDirection.POSITIVE,
    category="momentum",
    tags=["basic"],
)
def ret_5d(data: pd.DataFrame) -> pd.Series:
    return data.groupby("code")["close"].pct_change(5)


@register_factor(
    "ret_20d",
    description="20日收益率",
    direction=FactorDirection.POSITIVE,
    category="momentum",
    tags=["basic"],
)
def ret_20d(data: pd.DataFrame) -> pd.Series:
    return data.groupby("code")["close"].pct_change(20)


@register_factor(
    "reversal_5d",
    description="5日反转因子（负5日收益）",
    direction=FactorDirection.POSITIVE,
    category="reversal",
    tags=["classic"],
)
def reversal_5d(data: pd.DataFrame) -> pd.Series:
    return -ret_5d(data)


@register_factor(
    "reversal_20d",
    description="20日反转因子",
    direction=FactorDirection.POSITIVE,
    category="reversal",
    tags=["classic"],
)
def reversal_20d(data: pd.DataFrame) -> pd.Series:
    return -ret_20d(data)


@register_factor(
    "volatility_20d",
    description="20日波动率",
    direction=FactorDirection.NEUTRAL,
    category="risk",
    tags=["risk"],
)
def volatility_20d(data: pd.DataFrame) -> pd.Series:
    return data.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )


@register_factor(
    "volume_ma_20",
    description="20日均量",
    direction=FactorDirection.NEUTRAL,
    category="volume",
    tags=["liquidity"],
)
def volume_ma_20(data: pd.DataFrame) -> pd.Series:
    return data.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )


@register_factor(
    "volume_ratio",
    description="量比（当日量/20日均量）",
    direction=FactorDirection.POSITIVE,
    category="volume",
    tags=["liquidity", "momentum"],
)
def volume_ratio(data: pd.DataFrame) -> pd.Series:
    return data["volume"] / volume_ma_20(data).replace(0, np.nan)


@register_factor(
    "hl_range",
    description="日内振幅 (high-low)/close",
    direction=FactorDirection.NEUTRAL,
    category="volatility",
    tags=["intraday"],
)
def hl_range(data: pd.DataFrame) -> pd.Series:
    return (data["high"] - data["low"]) / data["close"]


@register_factor(
    "ma_deviation_20",
    description="20日均线偏离度 = close/MA20 - 1",
    direction=FactorDirection.NEUTRAL,
    category="technical",
    tags=["trend"],
    params={"window": 20},
)
def ma_deviation_20(data: pd.DataFrame, window: int = 20) -> pd.Series:
    ma = data.groupby("code")["close"].transform(
        lambda x: x.rolling(window, min_periods=3).mean()
    )
    return data["close"] / ma - 1


# Expression 形式的因子（演示 Expression Engine 互通）
@register_factor(
    "expr_reversal_20d",
    description="用表达式引擎构建的 20 日反转因子",
    direction=FactorDirection.POSITIVE,
    category="reversal",
    tags=["expression", "classic"],
)
def expr_reversal_20d_expr():
    from quant_opt.expression_engine import F, Ref
    return -(F("close") / Ref(F("close"), 20) - 1)


__all__ = [
    "FactorDirection",
    "FactorSpec",
    "FactorRegistry",
    "REGISTRY",
    "register_factor",
]
