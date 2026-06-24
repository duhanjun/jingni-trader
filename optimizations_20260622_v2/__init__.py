"""
jingni-trader 量化优化模块

本模块集合了基于开源量化项目学习成果的优化实现，包括：
- vectorized_ic: 向量化IC分析（借鉴 Qlib 的高效因子评估）
- vectorized_neutralize: 向量化因子中性化（替代逐日 Python 循环）
- vectorized_backtest: 向量化回测引擎（借鉴 VectorBT 的向量化设计）
- enhanced_metrics: 扩展绩效指标（借鉴 Investing Algorithm Framework 的 30+ 指标体系）
- walk_forward: Walk-Forward 滚动验证（借鉴 VectorBT PRO 的防过拟合方法）

所有优化代码独立于 main 分支原有代码，通过对比测试验证其正确性与性能提升。
"""

__version__ = "0.1.0"
__author__ = "jingni-trader optimization team"