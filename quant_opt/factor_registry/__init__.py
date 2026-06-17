"""
因子注册表（Factor Registry）

通过装饰器或 FactorSpec.add() 扩展 A 股因子库。
"""
from .registry import (
    FactorDirection,
    FactorRegistry,
    FactorSpec,
    REGISTRY,
    register_factor,
)

__all__ = [
    "FactorDirection",
    "FactorRegistry",
    "FactorSpec",
    "REGISTRY",
    "register_factor",
]
