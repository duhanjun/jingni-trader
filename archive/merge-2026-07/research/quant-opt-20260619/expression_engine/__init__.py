"""
表达式引擎 - jingni-trader 因子表达式 DSL

借鉴自 Microsoft Qlib (https://github.com/microsoft/qlib) 的 Expression Engine 核心设计，
在不引入外部依赖的前提下，以独立、可插拔的模块形式实现。

支持语法示例：
    Ref($close, 5) / Ref($close, 1) - 1         # 5日收益率
    Mean($close, 20) / $close                   # 20日均价/最新价
    ($close - Mean($close, 20)) / Std($close, 20)  # 标准化偏离
    Log($close)                                 # 对数价格
    Abs(Ref($close, 1) / $close - 1)            # 1日收益率绝对值
    Rank($volume)                               # 截面排名
    Delta($close, 5) / $close                   # 5日价格变动率
    Corr($close, $volume, 20)                   # 20日滚动相关系数
    SumIf($volume > Mean($volume, 20), $volume, 5)  # 条件滚动求和
"""
from .engine import ExpressionEngine
from .parser import ExpressionParser
from .operators import (
    ElemOperator,
    PairOperator,
    Rolling,
    Ref,
    Mean,
    Std,
    Sum,
    Delta,
    Corr,
    Cov,
    Max,
    Min,
    Med,
    Mad,
    Quantile,
    Slope,
    Rsquare,
    Resi,
    Rank,
    Abs,
    Log,
    Sign,
    Sqrt,
    Power,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    Equal,
    And,
    Or,
    Not,
    If,
    SumIf,
)

__all__ = [
    "ExpressionEngine",
    "ExpressionParser",
    "ElemOperator",
    "PairOperator",
    "Rolling",
    "Ref", "Mean", "Std", "Sum", "Delta", "Corr", "Cov",
    "Max", "Min", "Med", "Mad", "Quantile", "Slope", "Rsquare", "Resi",
    "Rank", "Abs", "Log", "Sign", "Sqrt", "Power",
    "Add", "Sub", "Mul", "Div",
    "Greater", "Less", "Equal",
    "And", "Or", "Not", "If", "SumIf",
]
