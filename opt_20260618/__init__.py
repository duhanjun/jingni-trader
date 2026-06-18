"""
jingni-trader 量化交易框架优化验证模块 (2026-06-18)

本目录是 feat/quant-opt-20260618 分支下的优化探索代码，
仅作验证用途，尚未合并到 main 分支。

包含四个独立的优化模块：
    1. vectorized_backtest  - 向量化回测引擎
    2. strategy_api         - Strategy 抽象基类 + SignalStrategy 组合模式
    3. pit_guard            - Point-in-Time 数据守卫 + Walk-Forward CV
    4. stability_test       - 多源并行稳健性测试
"""

__all__ = [
    "vectorized_backtest",
    "strategy_api",
    "pit_guard",
    "stability_test",
]
