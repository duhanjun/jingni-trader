"""
factor_library.py
=================

借鉴 Qlib ``Alpha158DL`` / ``Alpha360DL`` (https://github.com/microsoft/qlib)
为 jingni-trader 提供一个开箱即用的因子库, 每个因子以 ``ExpressionEngine``
支持的表达式定义, 通过 ``direction`` 标注方向 (1 越大越好 / -1 越小越好),
便于后续 IC 自动评估与因子筛选。

设计目标
--------
1. **可扩展**: 用户通过 ``register_factor`` 即可新增自定义因子
2. **分类**: 6 大类 - 动量、反转、波动、成交量、价值、质量
3. **可序列化**: 每个因子都能 ``to_dict()`` 写入元数据, 方便报告输出
4. **与 jingni-trader 兼容**: 输出 ``(name, expression, direction, category, description)`` 结构

完整表达式会由 ``expression_engine.evaluate_expression`` 计算, 保证与
Qlib 的声明式因子设计思想一致。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pandas as pd

from .expression_engine import evaluate_expression, Field

logger = logging.getLogger("quant_opt_20260616.factor_library")


@dataclass(frozen=True)
class FactorDef:
    """单个因子定义"""
    name: str           # 因子英文名 (主键)
    expression: str     # 表达式字符串
    direction: int      # 1: 越大越好, -1: 越小越好
    category: str       # momentum / reversal / volatility / volume / value / quality
    description: str    # 中文说明

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 预定义因子库 (Qlib Alpha158 风格, A 股适配)
# ============================================================================

DEFAULT_FACTORS: List[FactorDef] = [
    # ── 动量类 (Momentum) ────────────────────────────────────
    FactorDef("mom_5",       "Mean($close / Ref($close, 1) - 1, 5)",  1, "momentum", "5日动量 (累计收益)"),
    FactorDef("mom_10",      "Mean($close / Ref($close, 1) - 1, 10)", 1, "momentum", "10日动量"),
    FactorDef("mom_20",      "Mean($close / Ref($close, 1) - 1, 20)", 1, "momentum", "20日动量"),
    FactorDef("mom_60",      "Mean($close / Ref($close, 1) - 1, 60)", 1, "momentum", "60日动量"),
    FactorDef("accel_5_20",  "Mean($close / Ref($close, 1) - 1, 5) - Mean($close / Ref($close, 1) - 1, 20)",
              1, "momentum", "5日-20日 动量差 (加速度)"),

    # ── 反转类 (Reversal) ────────────────────────────────────
    FactorDef("rev_1",       "$close / Ref($close, 1) - 1",          -1, "reversal", "1日反转 (昨日涨, 今日易跌)"),
    FactorDef("rev_5",       "Mean($close / Ref($close, 1) - 1, 5)", -1, "reversal", "5日反转"),
    FactorDef("rev_10",      "Mean($close / Ref($close, 1) - 1, 10)", -1, "reversal", "10日反转"),
    FactorDef("rev_60_neg",  "Mean($close / Ref($close, 1) - 1, 60)", -1, "reversal", "60日反转 (长期超跌反弹)"),

    # ── 波动类 (Volatility) ───────────────────────────────────
    FactorDef("vol_5",       "Std($close / Ref($close, 1) - 1, 5)",  -1, "volatility", "5日波动率"),
    FactorDef("vol_20",      "Std($close / Ref($close, 1) - 1, 20)", -1, "volatility", "20日波动率"),
    FactorDef("vol_60",      "Std($close / Ref($close, 1) - 1, 60)", -1, "volatility", "60日波动率"),
    FactorDef("range_20",    "Mean(($high - $low) / $close, 20)",     -1, "volatility", "20日振幅 (日均高低差/收盘价)"),
    FactorDef("hl_range_5",  "Mean(($high - $low) / $close, 5)",      -1, "volatility", "5日振幅"),

    # ── 成交量类 (Volume) ────────────────────────────────────
    FactorDef("vol_ratio_5_20", "Mean($volume, 5) / Mean($volume, 20)", 1, "volume", "5日均量/20日均量 (放量)"),
    FactorDef("vol_ratio_1_5",  "$volume / Mean($volume, 5)",            1, "volume", "当日量/5日均量"),
    FactorDef("amount_5",       "Mean($amount, 5)",                      1, "volume", "5日均成交额"),
    FactorDef("amount_20",      "Mean($amount, 20)",                     1, "volume", "20日均成交额"),
    FactorDef("turnover_5",     "Mean($volume / ($close * 100), 5)",     1, "volume", "5日均换手率近似"),
    FactorDef("price_corr_vol_20",
              "Mean($close * $volume, 20) - Mean($close, 20) * Mean($volume, 20)",
              1, "volume", "20日价量相关 (放量上涨信号)"),

    # ── 价值类 (Value) ────────────────────────────────────────
    # 注: 完整价值因子需要基本面, 这里用价格代理
    FactorDef("ep_proxy",     "1 / $close",                            1, "value", "市盈率代理 (1/价格)"),
    FactorDef("bp_proxy",     "1 / $close",                            1, "value", "市净率代理"),
    FactorDef("log_price",    "Log($close)",                          -1, "value", "对数价格 (规模代理)"),

    # ── 质量类 (Quality) ──────────────────────────────────────
    FactorDef("trend_60",     "$close / Mean($close, 60) - 1",         1, "quality", "60日趋势强度"),
    FactorDef("ma_cross_5_20", "Mean($close, 5) / Mean($close, 20) - 1", 1, "quality", "5日上穿20日均线信号"),
    FactorDef("high_60",      "$close / Max($close, 60) - 1",          1, "quality", "60日新高 (动量强度)"),
    FactorDef("low_60",       "$close / Min($close, 60) - 1",          1, "quality", "60日新低 (超跌反弹信号)"),
    FactorDef("skew_20",      "($close - Mean($close, 20)) / Std($close, 20)", 1, "quality", "20日价格z-score"),
]


# ============================================================================
# 因子注册表
# ============================================================================

class FactorLibrary:
    """可扩展的因子库"""

    def __init__(self, factors: Optional[List[FactorDef]] = None):
        self._factors: Dict[str, FactorDef] = {}
        for f in (factors or DEFAULT_FACTORS):
            self._factors[f.name] = f

    def register(self, factor: FactorDef) -> None:
        if factor.name in self._factors:
            logger.warning(f"因子 {factor.name} 已存在, 将被覆盖")
        self._factors[factor.name] = factor

    def get(self, name: str) -> Optional[FactorDef]:
        return self._factors.get(name)

    def list(self, category: Optional[str] = None) -> List[FactorDef]:
        if category is None:
            return list(self._factors.values())
        return [f for f in self._factors.values() if f.category == category]

    def categories(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in self._factors.values():
            out[f.category] = out.get(f.category, 0) + 1
        return out

    def compute(self, name: str, data: pd.DataFrame) -> pd.Series:
        """计算单个因子"""
        f = self._factors.get(name)
        if f is None:
            raise KeyError(f"未注册因子: {name}")
        return evaluate_expression(data, f.expression)

    def compute_batch(self, data: pd.DataFrame, names: Optional[List[str]] = None) -> pd.DataFrame:
        """批量计算多个因子"""
        names = names or list(self._factors.keys())
        results: Dict[str, pd.Series] = {}
        for n in names:
            try:
                results[n] = self.compute(n, data)
            except Exception as e:
                logger.warning(f"因子 {n} 计算失败: {e}")
        df = pd.DataFrame(results)
        df.insert(0, "code", data["code"].values)
        df.insert(1, "date", data["date"].values)
        return df

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self._factors.values()])


DEFAULT_LIBRARY = FactorLibrary()


__all__ = [
    "FactorDef", "FactorLibrary", "DEFAULT_FACTORS", "DEFAULT_LIBRARY",
]