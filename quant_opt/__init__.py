"""quant_opt - 量化交易优化验证模块

包含借鉴自开源社区的最佳实践, 用于 jingni-trader 的优化方向验证。

模块清单:
    - alpha_expression_engine: 声明式因子表达式引擎 (借鉴 qlib)
    - metrics: 扩展绩效指标库 (借鉴 vectorbt)
    - walk_forward: 滚动前向验证 (借鉴行业实践 / qlib RollingGen)
    - risk_engine: 事前风控引擎 (借鉴 NautilusTrader)
    - vectorized_bt: 向量化回测器 (借鉴 vectorbt)
    - intent_parser: 增强意图解析器 (借鉴 qlib workflow / TradingAgents)

本目录独立于 main 分支代码, 所有修改仅限 feat/quant-opt-* 分支。
"""
__version__ = "0.1.0"
__all__ = [
    "alpha_expression_engine",
    "metrics",
    "walk_forward",
    "risk_engine",
    "vectorized_bt",
    "intent_parser",
]
