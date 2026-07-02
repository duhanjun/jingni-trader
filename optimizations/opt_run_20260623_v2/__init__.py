"""
jingni-trader 优化验证模块（2026-06-23 第二轮）

基于对开源量化项目（Qlib / Backtrader / vn.py）的学习，
针对 jingni-trader 现有架构实现 3 个高价值优化方向的验证代码：

1. ExpressionEngine  - 因子表达式引擎（借鉴 Qlib DSL）
2. PITProvider       - Point-in-Time 数据提供者（借鉴 Qlib PIT）
3. VectorizedBacktest - 向量化回测引擎（借鉴 Backtrader/Qlib 性能设计）

所有代码独立存放于本子目录，不修改 main 分支原有代码，也不影响
feat/quant-opt-20260623 分支上既有优化工作。
"""
