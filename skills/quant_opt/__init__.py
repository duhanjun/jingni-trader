"""
jingni-trader / quant_opt
============================
实验性优化模块（Experimental Optimization Modules）

本目录包含借鉴自以下开源项目的优化方向（仅在新分支中使用，不合入 main）：

- AKQuant (akfamily/akquant, MIT): 因子表达式引擎、TA-Lib 集成、向量化回测思路
- FinRL-X (AI4Finance-Foundation/FinRL-Trading): 模块化、滚动训练、回测/执行统一接口
- simtradelab (kay-ou/SimTradeLab): 轻量回测框架 PTrade API 兼容
- 国内多因子选股系统 (henrylin99/quantitative_analysis): 白名单因子表达式引擎

包含的子模块：

1. ``factor_expression_engine`` - 基于 AST 白名单的安全因子表达式引擎
2. ``vectorized_backtest`` - 向量化回测引擎（性能对比）
3. ``walk_forward`` - Walk-forward 滚动训练框架

设计原则：
- 100% 兼容现有 API，不修改 main 分支任何代码
- 仅作为可插拔的扩展模块提供
- 每个子模块自带单元测试

Copyright (C) 2026 jingni-trader
"""
__version__ = "0.1.0-experimental"
