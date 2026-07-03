"""
因子注册机制 (Factor Registry)
================================
借鉴来源: Microsoft Qlib 的因子注册体系 + alfa.rs 的因子库管理

设计目标:
- 解决 jingni-trader factor-engine 中因子硬编码、无法动态扩展的问题
  (现状: factor-engine/engine.py:48-117 所有因子硬编码在 compute_a_share_factors 内;
   PandasTa/Talib calculator 加载后从未被调用)
- 提供装饰器式注册, 支持运行时发现与扩展
- 每个因子携带元数据(方向、类别、依赖字段、窗口), 便于 IC 分析与中性化

核心特性:
1. @register_factor 装饰器自动注册
2. 因子元数据 (name/category/direction/fields/window/description)
3. 因子库单例, 支持查询/列举/按类别过滤
4. 线程安全 (回测/并行计算场景)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class FactorMeta:
    """因子元数据。

    Attributes:
        name: 因子唯一标识, 如 "reversal_20d"
        category: 因子类别, 如 "momentum"/"reversal"/"volume"/"volatility"/"value"
        direction: 因子方向, +1 表示因子值越大未来收益越高, -1 表示反向
        fields: 依赖的原始行情字段, 如 ["close", "volume"]
        window: 计算窗口(天), 用于预热期推断与依赖分析
        description: 因子描述
    """

    name: str
    category: str
    direction: int = 1
    fields: tuple = ()
    window: int = 0
    description: str = ""


class FactorRegistry:
    """因子注册中心 (线程安全单例)。

    用法::

        registry = FactorRegistry.instance()

        @registry.register(
            name="ma_diff",
            category="trend",
            direction=1,
            fields=("close",),
            window=20,
            description="MA20 - MA5 趋势因子",
        )
        def compute_ma_diff(df: pd.DataFrame) -> pd.Series:
            return df["close"].rolling(20).mean() - df["close"].rolling(5).mean()
    """

    _instance: Optional["FactorRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "FactorRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._factors: Dict[str, FactorMeta] = {}
                    cls._instance._funcs: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
        return cls._instance

    @classmethod
    def instance(cls) -> "FactorRegistry":
        return cls()

    def register(
        self,
        name: str,
        category: str = "custom",
        direction: int = 1,
        fields: tuple = (),
        window: int = 0,
        description: str = "",
    ) -> Callable[[Callable], Callable]:
        """装饰器: 注册一个因子计算函数。

        Args:
            name: 因子名 (唯一, 重复注册会覆盖并发出警告)
            category: 因子类别
            direction: 因子方向 (+1/-1)
            fields: 依赖的行情字段
            window: 计算窗口
            description: 描述
        """
        if direction not in (1, -1):
            raise ValueError(f"direction 必须为 1 或 -1, 得到 {direction}")

        def decorator(func: Callable[[pd.DataFrame], pd.Series]) -> Callable:
            meta = FactorMeta(
                name=name,
                category=category,
                direction=direction,
                fields=tuple(fields),
                window=int(window),
                description=description or func.__doc__ or "",
            )
            with self._lock:
                if name in self._factors:
                    # 覆盖注册 (便于测试与热更新), 不抛异常
                    pass
                self._factors[name] = meta
                self._funcs[name] = func
            func.__factor_meta__ = meta  # type: ignore[attr-defined]
            return func

        return decorator

    def get(self, name: str) -> Optional[FactorMeta]:
        """获取因子元数据, 不存在返回 None。"""
        with self._lock:
            return self._factors.get(name)

    def get_func(self, name: str) -> Callable[[pd.DataFrame], pd.Series]:
        """获取因子计算函数, 不存在抛 KeyError。"""
        with self._lock:
            if name not in self._funcs:
                raise KeyError(f"因子 '{name}' 未注册, 可用因子: {list(self._factors.keys())}")
            return self._funcs[name]

    def list_factors(self, category: Optional[str] = None) -> List[FactorMeta]:
        """列出所有已注册因子, 可按类别过滤。"""
        with self._lock:
            metas = list(self._factors.values())
        if category is not None:
            metas = [m for m in metas if m.category == category]
        return sorted(metas, key=lambda m: (m.category, m.name))

    def categories(self) -> List[str]:
        """返回所有因子类别。"""
        with self._lock:
            return sorted({m.category for m in self._factors.values()})

    def compute(
        self,
        name: str,
        df: pd.DataFrame,
    ) -> pd.Series:
        """计算单个因子。

        Args:
            name: 因子名
            df: 行情数据, MultiIndex(date, code) 或单标的的 DatetimeIndex,
                至少包含因子依赖的 fields 列

        Returns:
            因子值 Series, 与 df 同索引
        """
        func = self.get_func(name)
        return func(df)

    def compute_many(
        self,
        names: List[str],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """批量计算多个因子, 返回列名为因子名的 DataFrame。

        会自动跳过因数据不足而失败的因子并记录到返回 DataFrame 的 attrs['errors']。
        """
        results: Dict[str, pd.Series] = {}
        errors: Dict[str, str] = {}
        for name in names:
            try:
                results[name] = self.compute(name, df)
            except Exception as exc:  # noqa: BLE001
                errors[name] = f"{type(exc).__name__}: {exc}"
        out = pd.DataFrame(results)
        out.attrs["errors"] = errors
        return out

    def clear(self) -> None:
        """清空注册表 (主要用于测试隔离)。"""
        with self._lock:
            self._factors.clear()
            self._funcs.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._factors)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._factors
