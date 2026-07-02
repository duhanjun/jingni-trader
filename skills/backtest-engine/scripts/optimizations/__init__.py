"""
jingni-trader 回测引擎优化模块

本目录包含基于开源量化项目（VectorBT / Qlib / vn.py）学习成果的优化实现，
位于 feat/quant-opt-20260624 分支，不修改 main 分支原有代码。

模块：
- vectorized_adapter.py : 向量化回测适配器（借鉴 VectorBT 矩阵运算思路）
- extended_metrics.py   : 扩展绩效指标（信息比率、盈亏比、利润因子等）
- vectorized_ic.py      : 向量化因子 IC 分析（借鉴 Qlib groupby 批量计算）
"""
