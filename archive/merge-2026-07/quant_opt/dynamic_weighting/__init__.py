"""
动态因子加权模块
================

借鉴自 arXiv 2406.18394 "AlphaForge: A Framework to Mine and Dynamically
Combine Formulaic Alpha Factors"。

原论文核心论点：

- 固定权重 / 固定因子集合难以适应市场风格切换
- 不同因子在不同时段的有效性差异显著
- 应当基于因子在**最近一段时间的 IC 表现**动态调整融合权重

本模块实现两种动态加权策略：

1. ``icir_decay``     : IC-IR 衰减加权（推荐）。近期表现好的因子权重大。
2. ``softmax_ic``     : softmax(IC) 加权，强调最优因子。

对比 jingni-trader 现有实现
--------------------------
- 现有 ``_get_ic_weights`` 仅使用 5 日 IC-IR 单一时间点，没有衰减
- 现有实现在 IC 全为 0 时退化为等权，没有平滑过渡
- 现有实现没有时间衰减窗口与换手惩罚

参考：
- 论文：https://arxiv.org/abs/2406.18394
- vnpy.alpha：https://github.com/vnpy/vnpy/tree/main/vnpy/alpha
"""

from .dynamic_weights import (
    DynamicFactorWeighting,
    icir_decay_weights,
    softmax_ic_weights,
)

__all__ = [
    "DynamicFactorWeighting",
    "icir_decay_weights",
    "softmax_ic_weights",
]
