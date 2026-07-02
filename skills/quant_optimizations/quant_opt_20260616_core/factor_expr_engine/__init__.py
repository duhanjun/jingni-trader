"""
因子表达式引擎 (Factor Expression Engine)
=====================================

借鉴自 Microsoft Qlib 的表达式引擎设计：
https://qlib.readthedocs.io/en/latest/component/data.html
https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py

Qlib 允许使用形如 ``Ref($close, 5) / $close - 1`` 的字符串公式来定义因子，
极大提升因子可扩展性。本模块为 jingni-trader 提供一个轻量级、可独立运行的
表达式引擎子集，包含：

- ``$open``/``$high``/``$low``/``$close``/``$volume``/``$amount`` 等行情字段
- 时间算子 ``Ref``、``Delta``、``Mean``、``Std``、``Sum``
- 截面算子 ``Rank``、``Scale``
- 基础算术 ``+ - * /``

设计目标
--------
1. **零外部依赖** —— 仅依赖 numpy/pandas
2. **安全求值** —— 不使用 ``eval``/``exec``，基于 AST 白名单
3. **可扩展** —— 算子以注册表形式提供，便于新增
4. **可测试** —— 每个算子都有单元测试覆盖

使用示例
--------
>>> from skills.quant_optimizations.quant_opt_20260616_core.factor_expr_engine.engine import FactorExprEngine
>>> engine = FactorExprEngine()
>>> df = engine.compute(
...     data=df_with_ohlcv,
...     expr="Rank(($close - Ref($close, 20)) / Ref($close, 20))",
...     name="momentum_20d_rank",
... )
"""

from .engine import FactorExprEngine, FactorField, TimeOp, CrossOp
from .registry import OP_REGISTRY, register_op

__all__ = [
    "FactorExprEngine",
    "FactorField",
    "TimeOp",
    "CrossOp",
    "OP_REGISTRY",
    "register_op",
]