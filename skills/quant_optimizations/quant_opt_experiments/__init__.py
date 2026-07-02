"""
jingni-trader 量化优化实验包

本目录包含基于开源量化项目研究后提出的 3 个候选优化方向的验证实现。

参考项目：
- Microsoft qlib:     因子表达式引擎 (Ref, Mean, $close)  + Alpha158/360
- vectorbt:           矢量化回测引擎 (Vectorized Portfolio)
- RiceQuant rqfactor: 因子依赖图、生命周期管理
- RD-Agent:           LLM 驱动的因子研究闭环

优化方向：
1. factor_expression_engine/   声明式因子表达式引擎（Qlib 风格）
2. vectorized_backtest/        矢量化回测引擎（vectorbt 风格）
3. walk_forward_eval/          因子 IC 稳定性 + 滚动评估（米筐 / Qlib 风格）

本包仅作为【可借鉴思路的最小可运行验证】，尚未合并到主分支。
"""
__version__ = "0.1.0"