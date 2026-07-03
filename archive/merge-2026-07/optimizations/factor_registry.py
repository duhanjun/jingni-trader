"""
因子注册表框架（Qlib Alpha158 思想）

借鉴来源：Microsoft Qlib —— 表达式引擎 + 算子注册器模式。
参考：
    - https://qlib.readthedocs.io/en/stable/component/data.html
    - Qlib 论文: https://arxiv.org/abs/2009.11189

核心思想：
    jingni-trader 现有 factor-engine/engine.py 的 compute_a_share_factors()
    将所有因子硬编码在一个方法内，新增因子必须修改源码，扩展性差、难维护。

    Qlib 的做法：每个因子/算子是一个自包含对象，通过注册器（Registry）自动
    发现，支持依赖声明、拓扑排序计算、元信息查询。本实现采用装饰器注册模式，
    让用户用一行 @register_factor 即可新增因子。

设计要点：
    1. @register_factor 装饰器：声明 name / direction / category / deps
    2. FactorRegistry 单例：管理所有已注册因子
    3. 依赖解析：根据 deps 做拓扑排序，自动按序计算
    4. 元信息查询：list_factors() / get_factor_info() 便于文档生成
    5. 与现有 engine 兼容：compute_all() 输出与原 compute_a_share_factors
       相同的 (code, date, [因子列]) 结构
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class FactorInfo:
    """因子元信息"""
    name: str
    func: Callable
    direction: int = 1          # 1 正向, -1 反向
    category: str = "custom"
    description: str = ""
    deps: List[str] = field(default_factory=list)  # 依赖的其他因子名

    def __call__(self, data: pd.DataFrame, deps_values: Dict[str, pd.Series]) -> pd.Series:
        return self.func(data, deps_values)


class FactorRegistry:
    """因子注册表（单例）"""

    _instance: Optional["FactorRegistry"] = None
    _registry: Dict[str, FactorInfo] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(
        cls,
        name: str,
        direction: int = 1,
        category: str = "custom",
        description: str = "",
        deps: Optional[List[str]] = None,
    ):
        """装饰器：注册一个因子

        用法:
            @FactorRegistry.register("reversal_20d", direction=-1,
                                     category="reversal", deps=["ret_20d"])
            def reversal_20d(data, deps):
                return -deps["ret_20d"]
        """
        def decorator(func: Callable) -> Callable:
            cls._registry[name] = FactorInfo(
                name=name, func=func, direction=direction,
                category=category, description=description,
                deps=deps or [],
            )
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[FactorInfo]:
        return cls._registry.get(name)

    @classmethod
    def list_factors(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def list_by_category(cls) -> Dict[str, List[str]]:
        cats: Dict[str, List[str]] = {}
        for name, info in cls._registry.items():
            cats.setdefault(info.category, []).append(name)
        return cats

    @classmethod
    def get_factor_info(cls, name: str) -> Optional[Dict]:
        info = cls._registry.get(name)
        if info is None:
            return None
        return {
            "name": info.name,
            "direction": info.direction,
            "category": info.category,
            "description": info.description,
            "deps": info.deps,
        }

    @classmethod
    def _topo_sort(cls, factor_names: List[str]) -> List[str]:
        """按依赖关系拓扑排序"""
        visited = set()
        result = []
        visiting = set()  # 检测循环依赖

        def visit(name: str):
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"检测到循环依赖: {name}")
            visiting.add(name)
            info = cls._registry.get(name)
            if info:
                for dep in info.deps:
                    if dep in cls._registry:
                        visit(dep)
            visiting.discard(name)
            visited.add(name)
            result.append(name)

        for name in factor_names:
            visit(name)
        return result

    @classmethod
    def compute(
        cls,
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """批量计算因子（自动按依赖排序）

        参数:
            data: 原始 OHLCV 数据，含 code, date, open, high, low, close, volume...
            factor_names: 要计算的因子列表；None 表示计算全部

        返回:
            DataFrame: code, date, [各因子列]
        """
        if factor_names is None:
            factor_names = cls.list_factors()

        ordered = cls._topo_sort(factor_names)
        result = data[["code", "date"]].copy()
        computed: Dict[str, pd.Series] = {}

        for name in ordered:
            info = cls._registry.get(name)
            if info is None:
                continue
            deps_values = {d: computed[d] for d in info.deps if d in computed}
            try:
                values = info.func(data, deps_values)
                result[name] = values
                computed[name] = values
            except Exception as e:
                # 单因子失败不影响其他因子
                result[name] = np.nan
        return result


# ════════════════════════════════════════════════════════════
#  内置因子库（对应 jingni-trader 原 compute_a_share_factors）
# ════════════════════════════════════════════════════════════

# ── 收益类 ──────────────────────────────────────────────────
@FactorRegistry.register(
    "ret_1d", direction=1, category="return",
    description="1日收益率", deps=[],
)
def _ret_1d(data, deps):
    return data.groupby("code")["close"].pct_change()


@FactorRegistry.register(
    "ret_5d", direction=1, category="return",
    description="5日收益率", deps=[],
)
def _ret_5d(data, deps):
    return data.groupby("code")["close"].pct_change(5)


@FactorRegistry.register(
    "ret_20d", direction=1, category="return",
    description="20日收益率", deps=[],
)
def _ret_20d(data, deps):
    return data.groupby("code")["close"].pct_change(20)


@FactorRegistry.register(
    "ret_60d", direction=1, category="return",
    description="60日收益率", deps=[],
)
def _ret_60d(data, deps):
    return data.groupby("code")["close"].pct_change(60)


# ── 反转类 ──────────────────────────────────────────────────
@FactorRegistry.register(
    "reversal_5d", direction=-1, category="reversal",
    description="5日反转（负 5日收益）", deps=["ret_5d"],
)
def _reversal_5d(data, deps):
    return -deps["ret_5d"]


@FactorRegistry.register(
    "reversal_20d", direction=-1, category="reversal",
    description="20日反转（负 20日收益）", deps=["ret_20d"],
)
def _reversal_20d(data, deps):
    return -deps["ret_20d"]


# ── 波动率类 ────────────────────────────────────────────────
@FactorRegistry.register(
    "volatility_20d", direction=-1, category="volatility",
    description="20日收益波动率", deps=[],
)
def _volatility_20d(data, deps):
    return data.groupby("code")["close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=10).std()
    )


# ── 换手率类 ────────────────────────────────────────────────
@FactorRegistry.register(
    "turnover_20d", direction=-1, category="turnover",
    description="20日平均换手率", deps=[],
)
def _turnover_20d(data, deps):
    if "turnover_rate" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return data.groupby("code")["turnover_rate"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )


@FactorRegistry.register(
    "turnover_5d", direction=-1, category="turnover",
    description="5日平均换手率", deps=[],
)
def _turnover_5d(data, deps):
    if "turnover_rate" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    return data.groupby("code")["turnover_rate"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )


@FactorRegistry.register(
    "turnover_change", direction=1, category="turnover",
    description="换手率变化（5日/20日-1）", deps=["turnover_5d", "turnover_20d"],
)
def _turnover_change(data, deps):
    return deps["turnover_5d"] / deps["turnover_20d"].replace(0, np.nan) - 1


# ── 量价类 ──────────────────────────────────────────────────
@FactorRegistry.register(
    "volume_20d", direction=1, category="volume",
    description="20日平均成交量", deps=[],
)
def _volume_20d(data, deps):
    return data.groupby("code")["volume"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )


@FactorRegistry.register(
    "volume_ratio", direction=1, category="volume",
    description="量比（当日量/20日均量）", deps=["volume_20d"],
)
def _volume_ratio(data, deps):
    return data["volume"] / deps["volume_20d"].replace(0, np.nan)


# ── 规模类 ──────────────────────────────────────────────────
@FactorRegistry.register(
    "lncap", direction=-1, category="size",
    description="对数市值（用成交额/换手率估算）", deps=[],
)
def _lncap(data, deps):
    if "amount" not in data.columns or "turnover_rate" not in data.columns:
        return pd.Series(np.nan, index=data.index)
    mv = data["amount"] / data["turnover_rate"].replace(0, np.nan) * 100
    return mv.replace(0, np.nan).apply(lambda x: np.log(x) if x > 0 else np.nan)
