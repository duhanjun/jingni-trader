"""
quant_opt 包：jingni-trader 量化优化验证模块
==========================================

本目录为独立验证模块（位于 feat/quant-opt-20260617 分支），
不修改 main 分支代码，仅作为优化方向的可行性验证。

包含以下子模块（每个子模块均可独立 import）：

- factor_expr       : 因子表达式引擎（借鉴 AkQuant 表达式风格）
- vectorized_bt     : 向量化回测引擎（借鉴 Qlib 向量化思路）
- walk_forward      : Walk-forward 验证（借鉴 RD-Agent(Q) 滚动训练）
- metrics           : 增强绩效指标（借鉴 AkQuant 报告的 IR/Alpha/Beta）
- reports           : HTML 报告生成（基准对比 + 因子热力图）
- benchmarks        : 端到端测试用例

设计原则：
- 不依赖 jingni-trader 的 workspace 目录
- 仅使用 pandas / numpy / scipy / sklearn（基线依赖）
- 所有 API 与主仓的 Context 解耦，便于独立运行
"""
__version__ = "0.1.0"
