"""quant_opt - 量化交易优化验证包

借鉴 Qlib / VectorBT / Riskfolio-Lib 等开源项目的设计思路，
对 jingni-trader 进行优化验证。所有代码独立于 main 分支，
不修改原有模块。

子模块:
- core.vectorized_backtest: 向量化回测引擎 (借鉴 VectorBT)
- core.factor_expression:   因子表达式引擎 (借鉴 Qlib)
- core.vectorized_ic:       向量化 IC 分析与中性化
"""
__version__ = "0.1.0"
