"""
算子注册表
===========

算子按两类组织：
- ``TimeOp``  : 沿时间轴对单一标的计算 (Ref, Mean, Std, Delta, Sum, ...)
- ``CrossOp`` : 横截面对当日所有标的计算 (Rank, Scale, ...)

新算子通过 ``@register_op`` 装饰器接入。
"""

from typing import Callable, Dict
from .engine import TimeOp, CrossOp

OP_REGISTRY: Dict[str, Callable] = {}


def register_op(name: str, op_type: type, doc: str = ""):
    """装饰器：把算子实例挂到全局注册表。"""
    def decorator(func: Callable) -> Callable:
        if name in OP_REGISTRY:
            raise ValueError(f"duplicate operator: {name}")
        OP_REGISTRY[name] = op_type(name, func, doc)
        return func
    return decorator