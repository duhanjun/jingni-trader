"""
jingni-trader 优化验证实验目录

本目录下的所有代码均为基于开源项目学习成果的优化验证实现，
不修改 main 分支的任何现有代码。

借鉴来源：
- Microsoft Qlib: Point-in-time 数据设计、Alpha158 因子库、向量化回测、Walk-forward 验证
- AKQuant: Polars 驱动的因子表达式引擎、Zero-Copy 数据架构
- Backtrader: 事件驱动回测架构、Analyzer 系统

模块清单：
- factor_engine_polars.py : Polars 向量化因子计算引擎 + 因子表达式 DSL
- backtest_vectorized.py  : NumPy 向量化回测引擎（替代 Python for-loop）
- walk_forward.py         : Walk-forward 滚动验证框架
- tests/                  : 正确性、性能、边界测试
"""
