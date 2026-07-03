"""
quant_opt - jingni-trader 量化交易优化验证包

借鉴自开源社区前沿项目的优化思路验证：
  - AKQuant:  Walk-forward Validation 框架 / 因子表达式引擎 / Signal-Action 分离
  - Qlib:      Point-in-Time 数据治理 / 数据处理器链
  - VectorBT:  向量化回测与参数扫描
  - Backtrader: 事件驱动回测范式

本包仅作为离线验证代码，不直接修改主仓 `skills/*` 内的代码。
"""
__version__ = "0.1.0"