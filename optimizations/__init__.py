"""optimizations 包: jingni-trader 量化优化验证代码。

基于对开源量化项目 (Microsoft Qlib / AKQuant / alfa.rs / vectorbt) 的学习,
对 jingni-trader 现有架构的优化验证实现。

子模块:
- factor_expr: 因子表达式引擎 + 注册机制 (借鉴 Qlib Expression Engine)
- vectorized_backtest: 向量化回测引擎 + T+1 (借鉴 AKQuant / vectorbt)
- reports: 验证报告生成
"""
