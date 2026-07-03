"""
quant_opt_20260621 - 量化优化验证模块

基于对 VectorBT / AKQuant / Qlib 等开源项目的学习，
对 jingni-trader 现有实现进行的优化验证原型。

包含：
- vectorized_backtest: 向量化回测引擎（借鉴 VectorBT）
- factor_expression_engine: 因子表达式引擎（借鉴 AKQuant / Qlib Alpha101）
- vectorized_ic: 向量化 IC 分析（消除 per-date 循环）
"""
