"""
quant_opt_20260621 - 基于 2026-06-21 联网学习成果的 jingni-trader 优化验证包

本包独立于 main 分支既有代码，仅用于验证以下优化方向的可行性与收益：
1. 向量化回测引擎（借鉴 VectorBT 的多维数组 + Numba 思路，纯 numpy 实现）
2. 因子表达式引擎（借鉴 Microsoft Qlib 的 Expression Engine + Alpha158）
3. 向量化 IC 分析（借鉴 Qlib/VectorBT 的 cross-sectional groupby 思路）
4. 扩展风险指标（借鉴 VectorBT portfolio stats：VaR / CVaR / Information Ratio / Turnover）

设计原则：
- 不修改 main 分支任何既有文件
- 接口与既有 BaseBacktestEngine / FactorEngine 保持兼容，便于后续合并
- 所有模块均提供 correctness / performance / boundary 三类自验证测试
"""
