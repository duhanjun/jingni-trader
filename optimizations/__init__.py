"""
jingni-trader 优化验证模块

本目录包含基于开源量化项目（Qlib / NautilusTrader / VectorBT）学习成果
对 jingni-trader 现有模块进行的优化验证代码。

所有代码独立存放，不修改 main 分支的任何文件。

优化方向：
1. vectorized_backtest.py  - 向量化回测引擎（借鉴 VectorBT）
2. enhanced_metrics.py     - 增强绩效指标（借鉴 VectorBT / Qlib）
3. vectorized_ic.py        - 向量化 IC 分析（借鉴 Qlib）
4. test_optimizations.py   - 验证测试套件
"""
