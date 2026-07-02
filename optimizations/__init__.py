"""optimizations 包: jingni-trader 量化优化验证代码。

基于对开源量化项目 (Microsoft Qlib / AKQuant / alfa.rs / vectorbt / FactorEngine) 的学习,
对 jingni-trader 现有架构的优化验证实现。独立于 main 分支原有代码，未合并前不修改原代码。

子模块:
- factor_expr: 因子表达式引擎 + 注册机制 (借鉴 Qlib Expression Engine)
- vectorized_backtest: 向量化回测引擎 + T+1 (借鉴 AKQuant / vectorbt)
- polars_ic_analysis: Polars 向量化 IC 分析 (借鉴 AKQuant / Qlib)
- polars_neutralize: Polars 向量化因子中性化, FWL 定理 (借鉴 Qlib)
- vectorized_metrics: 增强版绩效指标 24 个 (借鉴 VectorBT)
- reports: 验证报告生成
"""
