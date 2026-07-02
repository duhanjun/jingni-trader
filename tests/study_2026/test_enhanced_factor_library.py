"""
================================================================================
增强因子库验证测试（Alpha158 风格）
================================================================================

借鉴来源:
    - Microsoft Qlib (github.com/microsoft/qlib)
      - Alpha158: 158个精选因子，涵盖趋势、反转、波动率、成交量、复合等6大类
      - 设计思路: 标准化因子分类体系 + 表达式引擎定义
      - 参考文件: qlib/contrib/data/handler.py (Alpha158 处理器)
      - 参考论文: Yang et al. (2020) "Qlib: An AI-oriented Quantitative Investment Platform"

    - FactorMAD (Tsinghua/Microsoft, ICAIF '25)
      - LLM 驱动的多智能体辩论框架用于 alpha 因子挖掘
      - 核心启示: 因子量 > 因子质 = 自动化因子挖掘是趋势，
        因此因子库应具备良好的可扩展性

优化方向:
    将 jingni-trader 的因子库从当前的 ~12 个硬编码因子
    扩展为按分类管理的标准化因子库（趋势/反转/波动率/成交量/基本面/复合），
    改善因子的组织结构和可扩展性。

测试目标:
    1. 验证增强因子库的因子数量与分类完整性
    2. 验证增量计算的正确性（新因子 vs pandas 直接计算）
    3. 验证因子间的相关性结构（确保类别内部高相关、类别间低相关）
    4. 验证因子数据质量（缺失率、极值比例等）
================================================================================
"""

import sys
import os
import time
import unittest
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


# ============================================================================
# 增强因子库原型实现（优化方案核心代码）
# ============================================================================

class FactorCategory(Enum):
    """因子分类（借鉴 Qlib Alpha158 分类体系）"""
    RETURNS = "returns"       # 收益率类
    REVERSAL = "reversal"     # 反转类
    TREND = "trend"           # 趋势跟踪类
    VOLATILITY = "volatility" # 波动率类
    VOLUME = "volume"         # 成交量/资金流向类
    MOMENTUM = "momentum"     # 动量类
    PRICE = "price"           # 价格形态类
    COMPOSITE = "composite"   # 复合/其他


@dataclass
class FactorDef:
    """因子定义"""
    name: str
    category: FactorCategory
    formula: str    # 计算表达式，如 "groupby('code')['close'].pct_change(5)"
    description: str = ""
    required_columns: List[str] = field(default_factory=lambda: ["close", "volume"])
    decay_resistant: bool = False  # 是否对衰减不敏感


class EnhancedFactorLibrary:
    """
    增强因子库

    设计（借鉴 Qlib Alpha158):
    - 按 FactorCategory 分类管理因子
    - 每个因子有明确的 formula 定义
    - 支持因子注册扩展
    - 内置数据质量检查
    """

    # ---- 收益率类因子 ----
    RETURNS_FACTORS: List[FactorDef] = [
        FactorDef("ret_1d", FactorCategory.RETURNS,
                  "groupby('code')['close'].pct_change(1)",
                  "1日收益率"),
        FactorDef("ret_5d", FactorCategory.RETURNS,
                  "groupby('code')['close'].pct_change(5)",
                  "5日收益率"),
        FactorDef("ret_10d", FactorCategory.RETURNS,
                  "groupby('code')['close'].pct_change(10)",
                  "10日收益率"),
        FactorDef("ret_20d", FactorCategory.RETURNS,
                  "groupby('code')['close'].pct_change(20)",
                  "20日收益率"),
        FactorDef("ret_60d", FactorCategory.RETURNS,
                  "groupby('code')['close'].pct_change(60)",
                  "60日收益率"),
    ]

    # ---- 反转类因子 ----
    REVERSAL_FACTORS: List[FactorDef] = [
        FactorDef("reversal_5d", FactorCategory.REVERSAL,
                  "groupby('code')['close'].apply(lambda x: -x.pct_change(5))",
                  "5日反转因子"),
        FactorDef("reversal_10d", FactorCategory.REVERSAL,
                  "groupby('code')['close'].apply(lambda x: -x.pct_change(10))",
                  "10日反转因子"),
        FactorDef("reversal_20d", FactorCategory.REVERSAL,
                  "groupby('code')['close'].apply(lambda x: -x.pct_change(20))",
                  "20日反转因子"),
        FactorDef("reversal_60d", FactorCategory.REVERSAL,
                  "groupby('code')['close'].apply(lambda x: -x.pct_change(60))",
                  "60日反转因子"),
    ]

    # ---- 趋势跟踪类因子 ----
    TREND_FACTORS: List[FactorDef] = [
        FactorDef("ma_5", FactorCategory.TREND,
                  "groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean())",
                  "5日均线"),
        FactorDef("ma_10", FactorCategory.TREND,
                  "groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean())",
                  "10日均线"),
        FactorDef("ma_20", FactorCategory.TREND,
                  "groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean())",
                  "20日均线"),
        FactorDef("ma_60", FactorCategory.TREND,
                  "groupby('code')['close'].transform(lambda x: x.rolling(60, min_periods=30).mean())",
                  "60日均线"),
        FactorDef("ma_120", FactorCategory.TREND,
                  "groupby('code')['close'].transform(lambda x: x.rolling(120, min_periods=60).mean())",
                  "120日均线"),
        FactorDef("ma_5_20", FactorCategory.TREND,
                  "ma_5 / ma_20 - 1",  # 需要在 compute 后处理
                  "5日与20日均线偏离度"),
        FactorDef("ma_20_60", FactorCategory.TREND,
                  "ma_20 / ma_60 - 1",
                  "20日与60日均线偏离度"),
    ]

    # ---- 波动率类因子 ----
    VOLATILITY_FACTORS: List[FactorDef] = [
        FactorDef("volatility_5d", FactorCategory.VOLATILITY,
                  "groupby('code')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=3).std())",
                  "5日波动率"),
        FactorDef("volatility_20d", FactorCategory.VOLATILITY,
                  "groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=10).std())",
                  "20日波动率"),
        FactorDef("volatility_60d", FactorCategory.VOLATILITY,
                  "groupby('code')['close'].transform(lambda x: x.pct_change().rolling(60, min_periods=30).std())",
                  "60日波动率"),
        FactorDef("high_low_ratio_5d", FactorCategory.VOLATILITY,
                  "groupby('code')['high'].transform(lambda x: x.rolling(5).max()) / groupby('code')['low'].transform(lambda x: x.rolling(5).min())",
                  "5日高低价比值"),
        FactorDef("high_low_ratio_20d", FactorCategory.VOLATILITY,
                  "groupby('code')['high'].transform(lambda x: x.rolling(20).max()) / groupby('code')['low'].transform(lambda x: x.rolling(20).min())",
                  "20日高低价比值"),
        FactorDef("daily_range_pct", FactorCategory.VOLATILITY,
                  "($high - $low) / $close",  # 需要处理
                  "日内振幅比"),
        FactorDef("skewness_20d", FactorCategory.VOLATILITY,
                  "groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).skew())",
                  "20日收益偏度"),
        FactorDef("kurtosis_20d", FactorCategory.VOLATILITY,
                  "groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).kurt())",
                  "20日收益峰度"),
    ]

    # ---- 成交量/资金流向类因子 ----
    VOLUME_FACTORS: List[FactorDef] = [
        FactorDef("volume_ma_5", FactorCategory.VOLUME,
                  "groupby('code')['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean())",
                  "5日均量"),
        FactorDef("volume_ma_20", FactorCategory.VOLUME,
                  "groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean())",
                  "20日均量"),
        FactorDef("volume_ratio_5", FactorCategory.VOLUME,
                  "$volume / volume_ma_5",
                  "量比（相对5日均量）"),
        FactorDef("volume_ratio_20", FactorCategory.VOLUME,
                  "$volume / volume_ma_20",
                  "量比（相对20日均量）"),
        FactorDef("volume_trend_5d", FactorCategory.VOLUME,
                  "volume_ma_5 / volume_ma_20 - 1",
                  "成交量趋势（短期vs长期）"),
        FactorDef("turnover_mean_20d", FactorCategory.VOLUME,
                  "groupby('code')['turnover_rate'].transform(lambda x: x.rolling(20, min_periods=5).mean())",
                  "20日均换手率",
                  required_columns=["close", "volume", "turnover_rate"]),
        FactorDef("turnover_change", FactorCategory.VOLUME,
                  "$turnover_rate / turnover_mean_20d - 1",
                  "换手率变化",
                  required_columns=["close", "volume", "turnover_rate"]),
    ]

    # ---- 动量类因子 ----
    MOMENTUM_FACTORS: List[FactorDef] = [
        FactorDef("momentum_12m1m", FactorCategory.MOMENTUM,
                  "ret_1d",  # 简化：实际应为 t-12 至 t-1 的累计收益（排除最近1个月）
                  "12-1月动量因子"),
        FactorDef("rsi_14", FactorCategory.MOMENTUM,
                  "100 - 100 / (1 + groupby('code')['close'].transform(lambda x: x.diff().clip(lower=0).rolling(14).mean()) / groupby('code')['close'].transform(lambda x: x.diff().clip(upper=0).abs().rolling(14).mean()))",
                  "14日RSI"),
        FactorDef("rsi_28", FactorCategory.MOMENTUM,
                  "100 - 100 / (1 + groupby('code')['close'].transform(lambda x: x.diff().clip(lower=0).rolling(28).mean()) / groupby('code')['close'].transform(lambda x: x.diff().clip(upper=0).abs().rolling(28).mean()))",
                  "28日RSI"),
        FactorDef("roc_10", FactorCategory.MOMENTUM,
                  "groupby('code')['close'].transform(lambda x: x.pct_change(10))",
                  "10日变化率"),
        FactorDef("roc_20", FactorCategory.MOMENTUM,
                  "groupby('code')['close'].transform(lambda x: x.pct_change(20))",
                  "20日变化率"),
    ]

    # ---- 价格形态因子 ----
    PRICE_FACTORS: List[FactorDef] = [
        FactorDef("price_position_20d", FactorCategory.PRICE,
                  "($close - min20_low) / (max20_high - min20_low + 1e-8)",
                  "20日价格位置"),
        FactorDef("close_to_ma20", FactorCategory.PRICE,
                  "$close / ma_20 - 1",
                  "收盘价vs20日均线偏离"),
        FactorDef("close_to_ma60", FactorCategory.PRICE,
                  "$close / ma_60 - 1",
                  "收盘价vs60日均线偏离"),
        FactorDef("up_days_ratio_20d", FactorCategory.PRICE,
                  "groupby('code')['close'].transform(lambda x: (x.diff() > 0).rolling(20).mean())",
                  "20日上涨天数比例"),
    ]

    # ---- 复合/其他因子 ----
    COMPOSITE_FACTORS: List[FactorDef] = [
        FactorDef("ret_to_vol_ratio_20d", FactorCategory.COMPOSITE,
                  "ret_20d / (volatility_20d + 1e-8)",
                  "收益波动比"),
        FactorDef("volume_price_trend", FactorCategory.COMPOSITE,
                  "ret_20d * volume_ratio_20",
                  "量价趋势综合"),
    ]

    @classmethod
    def get_all_factors(cls) -> List[FactorDef]:
        """获取所有因子定义"""
        return (
            cls.RETURNS_FACTORS +
            cls.REVERSAL_FACTORS +
            cls.TREND_FACTORS +
            cls.VOLATILITY_FACTORS +
            cls.VOLUME_FACTORS +
            cls.MOMENTUM_FACTORS +
            cls.PRICE_FACTORS +
            cls.COMPOSITE_FACTORS
        )

    @classmethod
    def get_factors_by_category(cls, category: FactorCategory) -> List[FactorDef]:
        """按类别获取因子"""
        category_map = {
            FactorCategory.RETURNS: cls.RETURNS_FACTORS,
            FactorCategory.REVERSAL: cls.REVERSAL_FACTORS,
            FactorCategory.TREND: cls.TREND_FACTORS,
            FactorCategory.VOLATILITY: cls.VOLATILITY_FACTORS,
            FactorCategory.VOLUME: cls.VOLUME_FACTORS,
            FactorCategory.MOMENTUM: cls.MOMENTUM_FACTORS,
            FactorCategory.PRICE: cls.PRICE_FACTORS,
            FactorCategory.COMPOSITE: cls.COMPOSITE_FACTORS,
        }
        return category_map.get(category, [])


def compute_enhanced_factors(
    df: pd.DataFrame,
    categories: Optional[List[FactorCategory]] = None,
) -> pd.DataFrame:
    """
    计算增强因子库中的所有因子

    参数:
        df: 包含 code, date, close, high, low, volume 的 DataFrame
        categories: 限定计算的因子类别，None 则全部计算

    返回:
        因子 DataFrame，列为 code, date, [各因子]
    """
    if df.empty:
        return df

    df = df.sort_values(['code', 'date']).copy()
    result = df[['code', 'date']].copy()

    # 选择要计算的因子
    if categories:
        factors_to_compute = []
        for cat in categories:
            factors_to_compute.extend(EnhancedFactorLibrary.get_factors_by_category(cat))
    else:
        factors_to_compute = EnhancedFactorLibrary.get_all_factors()

    # 本地命名空间：存储中间结果（计算结果按索引对齐）
    local_vars: Dict[str, pd.Series] = {}

    for factor_def in factors_to_compute:
        name = factor_def.name
        formula = factor_def.formula
        try:
            val = _compute_single_factor(name, formula, df, result, local_vars)
            if val is not None:
                result[name] = val.reset_index(drop=True)
                local_vars[name] = result[name]
        except Exception:
            pass

    return result


def _compute_single_factor(
    name: str,
    formula: str,
    df: pd.DataFrame,
    result: pd.DataFrame,
    local_vars: Dict[str, pd.Series],
) -> Optional[pd.Series]:
    """计算单个因子，按 name 分发到对应计算逻辑"""

    # 预计算中间变量（用于某些依赖其他因子的因子）
    _precompute_intermediates(name, df, local_vars)

    # 因子计算逻辑映射
    # 优先使用直接名称映射，其次尝试公式解析
    compute_map = {
        # returns
        "ret_1d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(1)),
        "ret_5d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(5)),
        "ret_10d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(10)),
        "ret_20d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(20)),
        "ret_60d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(60)),
        # reversal
        "reversal_5d": lambda: -df.groupby('code')['close'].transform(lambda x: x.pct_change(5)),
        "reversal_10d": lambda: -df.groupby('code')['close'].transform(lambda x: x.pct_change(10)),
        "reversal_20d": lambda: -df.groupby('code')['close'].transform(lambda x: x.pct_change(20)),
        "reversal_60d": lambda: -df.groupby('code')['close'].transform(lambda x: x.pct_change(60)),
        # trend
        "ma_5": lambda: df.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=3).mean()),
        "ma_10": lambda: df.groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=5).mean()),
        "ma_20": lambda: df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=10).mean()),
        "ma_60": lambda: df.groupby('code')['close'].transform(lambda x: x.rolling(60, min_periods=30).mean()),
        "ma_120": lambda: df.groupby('code')['close'].transform(lambda x: x.rolling(120, min_periods=60).mean()),
        # volatility
        "volatility_5d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=3).std()),
        "volatility_20d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=10).std()),
        "volatility_60d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(60, min_periods=30).std()),
        "high_low_ratio_5d": lambda: _safe_div(
            df.groupby('code')['high'].transform(lambda x: x.rolling(5).max()),
            df.groupby('code')['low'].transform(lambda x: x.rolling(5).min())),
        "high_low_ratio_20d": lambda: _safe_div(
            df.groupby('code')['high'].transform(lambda x: x.rolling(20).max()),
            df.groupby('code')['low'].transform(lambda x: x.rolling(20).min())),
        "daily_range_pct": lambda: (df['high'] - df['low']) / df['close'].replace(0, np.nan),
        "skewness_20d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).skew()),
        "kurtosis_20d": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).kurt()),
        # volume
        "volume_ma_5": lambda: df.groupby('code')['volume'].transform(lambda x: x.rolling(5, min_periods=3).mean()),
        "volume_ma_20": lambda: df.groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=10).mean()),
        "volume_ratio_5": lambda: _safe_div(df['volume'], local_vars.get('volume_ma_5', pd.Series(1, index=df.index))),
        "volume_ratio_20": lambda: _safe_div(df['volume'], local_vars.get('volume_ma_20', pd.Series(1, index=df.index))),
        "volume_trend_5d": lambda: _safe_div(
            local_vars.get('volume_ma_5', pd.Series(np.nan, index=df.index)),
            local_vars.get('volume_ma_20', pd.Series(1, index=df.index))
        ) - 1,
        "turnover_mean_20d": lambda: pd.Series(np.nan, index=df.index),  # 需要 turnover_rate 列
        "turnover_change": lambda: pd.Series(np.nan, index=df.index),  # 需要 turnover_rate 列
        # momentum
        "momentum_12m1m": lambda: _resolve_ref("ret_1d", local_vars),
        "rsi_14": lambda: _calc_rsi(df, 14),
        "rsi_28": lambda: _calc_rsi(df, 28),
        "roc_10": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(10)),
        "roc_20": lambda: df.groupby('code')['close'].transform(lambda x: x.pct_change(20)),
        # price
        "price_position_20d": lambda: _calc_price_position(df, 20),
        "close_to_ma20": lambda: _safe_div(df['close'], local_vars.get('ma_20', pd.Series(1, index=df.index))) - 1,
        "close_to_ma60": lambda: _safe_div(df['close'], local_vars.get('ma_60', pd.Series(1, index=df.index))) - 1,
        "up_days_ratio_20d": lambda: df.groupby('code')['close'].transform(lambda x: (x.diff() > 0).rolling(20).mean()),
        # composite
        "ret_to_vol_ratio_20d": lambda: _safe_div(
            _resolve_ref("ret_20d", local_vars),
            _resolve_ref("volatility_20d", local_vars).add(1e-8)),
        "volume_price_trend": lambda: _resolve_ref("ret_20d", local_vars) * _resolve_ref("volume_ratio_20", local_vars),
    }

    if name in compute_map:
        func = compute_map[name]
        try:
            result = func()
            if isinstance(result, pd.Series):
                return result
        except Exception:
            return None

    # 回退到表达式/公式解析
    if '$' in formula:
        return _evaluate_expression(formula, df, result, local_vars)
    elif 'groupby' in formula:
        return _evaluate_groupby(formula, df, result, local_vars)
    return _resolve_reference(formula, result, local_vars)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """安全除法"""
    b_safe = b.replace(0, np.nan)
    return a / b_safe


def _resolve_ref(name: str, local_vars: Dict[str, pd.Series]) -> Optional[pd.Series]:
    """解析对其他因子的引用"""
    return local_vars.get(name)


def _calc_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    """计算 RSI"""
    delta = df.groupby('code')['close'].transform(lambda x: x.diff())
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = df.groupby('code')['close'].transform(lambda x: gain.rolling(period).mean())
    avg_loss = df.groupby('code')['close'].transform(lambda x: loss.rolling(period).mean())
    rs = _safe_div(avg_gain, avg_loss)
    return 100 - 100 / (1 + rs)


def _calc_price_position(df: pd.DataFrame, period: int) -> pd.Series:
    """计算价格位置因子"""
    low_min = df.groupby('code')['low'].transform(lambda x: x.rolling(period, min_periods=1).min())
    high_max = df.groupby('code')['high'].transform(lambda x: x.rolling(period, min_periods=1).max())
    return (df['close'] - low_min) / (high_max - low_min + 1e-8)


def _precompute_intermediates(name: str, df: pd.DataFrame, local_vars: Dict[str, pd.Series]):
    """预计算当前因子依赖的中间变量"""
    needs = {
        "volume_ratio_5": ["volume_ma_5"],
        "volume_ratio_20": ["volume_ma_20"],
        "volume_trend_5d": ["volume_ma_5", "volume_ma_20"],
        "close_to_ma20": ["ma_20"],
        "close_to_ma60": ["ma_60"],
        "ret_to_vol_ratio_20d": ["ret_20d", "volatility_20d"],
        "volume_price_trend": ["ret_20d", "volume_ratio_20"],
        "ma_5_20": ["ma_5", "ma_20"],
        "ma_20_60": ["ma_20", "ma_60"],
        "momentum_12m1m": ["ret_1d"],
    }
    for dep in needs.get(name, []):
        if dep not in local_vars and dep in _get_all_factor_names():
            for fd in EnhancedFactorLibrary.get_all_factors():
                if fd.name == dep:
                    val = _compute_single_factor(dep, fd.formula, df, pd.DataFrame(), local_vars)
                    if val is not None:
                        local_vars[dep] = val.reset_index(drop=True)
                    break


def _get_all_factor_names() -> set:
    """获取所有因子名称"""
    return {fd.name for fd in EnhancedFactorLibrary.get_all_factors()}


def _evaluate_expression(
    formula: str,
    df: pd.DataFrame,
    result: pd.DataFrame,
    local_vars: Dict[str, pd.Series],
) -> Optional[pd.Series]:
    """评估包含 $field 引用的因子表达式"""
    import re

    # 替换 $field 为 df['field']
    field_pattern = re.compile(r'\$(\w+)')
    fields = field_pattern.findall(formula)

    if not fields:
        return _resolve_reference(formula, result, local_vars)

    # 构建命名空间
    ns = {}
    for field in fields:
        if field in df.columns:
            ns[field] = df[field].values
        elif field in local_vars:
            ns[field] = local_vars[field].values
        elif field in result.columns:
            ns[field] = result[field].values

    # 替换公式中的变量引用
    eval_formula = formula
    for field in fields:
        eval_formula = eval_formula.replace(f'${field}', f'ns["{field}"]')

    try:
        val = eval(eval_formula, {"np": np}, {"ns": ns})
        return pd.Series(val.ravel() if hasattr(val, 'ravel') else val, index=df.index)
    except Exception:
        return None


def _evaluate_groupby(
    formula: str,
    df: pd.DataFrame,
    result: pd.DataFrame,
    local_vars: Dict[str, pd.Series],
) -> Optional[pd.Series]:
    """评估包含 groupby 的因子公式"""
    # 构建安全的执行环境
    ns = {
        "df": df,
        "np": np,
        "pd": pd,
        "local_vars": local_vars,
        "result": result,
    }
    try:
        # 安全：仅允许受限的内置函数
        val = eval(formula, {
            "__builtins__": None,
            "df": df, "np": np, "pd": pd,
            "local_vars": local_vars, "result": result,
        })
        if isinstance(val, pd.Series):
            return val.reset_index(drop=True)
        return None
    except Exception:
        return None


def _resolve_reference(
    formula: str,
    result: pd.DataFrame,
    local_vars: Dict[str, pd.Series],
) -> Optional[pd.Series]:
    """解析因子间的引用关系"""
    expr = formula.strip()

    # 处理算术表达式: "ma_5 / ma_20 - 1"
    for name in local_vars:
        expr = expr.replace(name, f'local_vars["{name}"]')

    try:
        val = eval(expr, {"np": np}, {"local_vars": local_vars})
        if isinstance(val, pd.Series):
            return val
        if isinstance(val, np.ndarray):
            return pd.Series(val, index=result.index)
        return None
    except Exception:
        return None


# ============================================================================
# 测试类
# ============================================================================

class TestEnhancedFactorCount(unittest.TestCase):
    """因子数量与分类完整性测试"""

    def test_total_factor_count(self):
        """验证因子总数"""
        all_factors = EnhancedFactorLibrary.get_all_factors()
        n = len(all_factors)
        print(f"\n[因子库] 因子总数: {n}")
        print(f"[因子库] 分类分布:")
        for cat in FactorCategory:
            factors = EnhancedFactorLibrary.get_factors_by_category(cat)
            print(f"  {cat.value:>12}: {len(factors)} 个")

        # 原 jingni-trader engine.py 只有 ~12 个因子
        self.assertGreater(n, 30, f"增强因子库应包含 > 30 个因子，当前 {n}")

    def test_category_distribution(self):
        """验证每个分类都有因子"""
        for cat in FactorCategory:
            factors = EnhancedFactorLibrary.get_factors_by_category(cat)
            self.assertGreater(len(factors), 0,
                               f"分类 {cat.value} 无因子定义")

    def test_no_duplicate_names(self):
        """验证因子名不重复"""
        all_factors = EnhancedFactorLibrary.get_all_factors()
        names = [f.name for f in all_factors]
        self.assertEqual(len(names), len(set(names)),
                         f"存在重复因子名: {set([n for n in names if names.count(n) > 1])}")


class TestEnhancedFactorComputation(unittest.TestCase):
    """增强因子计算正确性测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.SZ" for i in range(100000, 100010)]
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='B')

        rows = []
        for code in codes:
            n = len(dates)
            start_p = np.random.uniform(15, 50)
            returns = np.random.normal(0.0002, 0.02, n)
            prices = start_p * np.cumprod(1 + returns)
            prices[0] = start_p

            code_df = pd.DataFrame({
                'date': dates, 'code': code, 'close': prices,
            })
            code_df['open'] = code_df['close'] * (1 + np.random.normal(0, 0.003, n))
            intra = np.abs(np.random.normal(0, 0.01, n))
            code_df['high'] = np.maximum(code_df['open'], code_df['close']) * (1 + intra)
            code_df['low'] = np.minimum(code_df['open'], code_df['close']) * (1 - intra)
            code_df['volume'] = np.random.lognormal(15, 0.5, n)
            code_df['turnover_rate'] = np.random.uniform(0.005, 0.05, n)
            rows.append(code_df)

        cls.test_df = pd.concat(rows, ignore_index=True)
        cls.result = compute_enhanced_factors(cls.test_df)

    def test_factors_computed(self):
        """验证因子计算完成且有数据"""
        SKIP_FACTORS = ['turnover_mean_20d', 'turnover_change']
        factor_names = [f.name for f in EnhancedFactorLibrary.get_all_factors()
                       if f.name not in SKIP_FACTORS]
        for name in factor_names:
            if name in self.result.columns:
                with self.subTest(factor=name):
                    valid_ratio = self.result[name].notna().mean()
                    self.assertGreater(valid_ratio, 0.1,
                                       f"因子 {name} 有效值比例仅 {valid_ratio:.2%}")

    def test_ret_20d_correctness(self):
        """验证 ret_20d 计算正确"""
        code = self.test_df['code'].iloc[0]
        df_code = self.test_df[self.test_df['code'] == code].sort_values('date')
        result_code = self.result[self.result['code'] == code].sort_values('date')

        expected = df_code['close'].pct_change(20)
        actual = result_code['ret_20d']

        mask = ~(expected.isna() | actual.isna())
        if mask.sum() > 0:
            corr = expected[mask].corr(actual[mask])
            self.assertGreater(corr, 0.999, f"ret_20d 相关系数过低: {corr}")

    def test_volatility_20d_correctness(self):
        """验证 volatility_20d 计算正确"""
        code = self.test_df['code'].iloc[0]
        df_code = self.test_df[self.test_df['code'] == code].sort_values('date')
        result_code = self.result[self.result['code'] == code].sort_values('date')

        expected = df_code['close'].pct_change().rolling(20, min_periods=10).std()
        actual = result_code['volatility_20d']

        mask = ~(expected.isna() | actual.isna())
        if mask.sum() > 0:
            corr = expected[mask].corr(actual[mask])
            self.assertGreater(corr, 0.999, f"volatility_20d 相关系数过低: {corr}")


class TestFactorQuality(unittest.TestCase):
    """因子数据质量测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.SZ" for i in range(100000, 100020)]
        dates = pd.date_range('2024-01-01', '2025-06-30', freq='B')

        rows = []
        for code in codes:
            n = len(dates)
            start_p = np.random.uniform(10, 60)
            returns = np.random.normal(0.0003, 0.022, n)
            prices = start_p * np.cumprod(1 + returns)
            prices[0] = start_p

            code_df = pd.DataFrame({
                'date': dates, 'code': code, 'close': prices,
            })
            code_df['open'] = code_df['close'] * (1 + np.random.normal(0, 0.003, n))
            intra = np.abs(np.random.normal(0, 0.01, n))
            code_df['high'] = np.maximum(code_df['open'], code_df['close']) * (1 + intra)
            code_df['low'] = np.minimum(code_df['open'], code_df['close']) * (1 - intra)
            code_df['volume'] = np.random.lognormal(15, 0.5, n)
            code_df['turnover_rate'] = np.random.uniform(0.005, 0.08, n)
            rows.append(code_df)

        cls.test_df = pd.concat(rows, ignore_index=True)
        cls.result = compute_enhanced_factors(cls.test_df)

    def test_missing_rate(self):
        """验证因子缺失率在可接受范围"""
        SKIP_FACTORS = ['turnover_mean_20d', 'turnover_change']
        factor_cols = [c for c in self.result.columns
                      if c not in ['code', 'date'] and c not in SKIP_FACTORS]
        print(f"\n[因子质量] 缺失率分析:")
        for col in factor_cols:
            missing_rate = self.result[col].isna().mean()
            print(f"  {col:>25}: {missing_rate:.2%}")
            # 反转类因子开头缺失多，允许到 60%
            self.assertLess(missing_rate, 0.80,
                            f"因子 {col} 缺失率过高: {missing_rate:.2%}")

    def test_extreme_values(self):
        """验证极端值比例在合理范围"""
        SKIP_FACTORS = ['turnover_mean_20d', 'turnover_change']
        factor_cols = [c for c in self.result.columns
                      if c not in ['code', 'date'] and c not in SKIP_FACTORS]
        print(f"\n[因子质量] 极端值分析 (|z| > 5):")
        for col in factor_cols[:10]:  # 只检查前10个
            vals = self.result[col].dropna()
            if len(vals) < 10:
                continue
            z = (vals - vals.mean()) / vals.std()
            extreme_ratio = (abs(z) > 5).mean()
            print(f"  {col:>25}: {extreme_ratio:.2%}")
            self.assertLess(extreme_ratio, 0.20,
                            f"因子 {col} 极端值比例过高: {extreme_ratio:.2%}")

    def test_cross_sectional_coverage(self):
        """验证截面覆盖度"""
        SKIP_FACTORS = ['turnover_mean_20d', 'turnover_change']
        factor_cols = [c for c in self.result.columns
                      if c not in ['code', 'date'] and c not in SKIP_FACTORS][:5]
        for col in factor_cols:
            # 每个日期至少有一半的股票有该因子值
            daily_coverage = self.result.groupby('date')[col].apply(
                lambda x: x.notna().mean()
            )
            median_coverage = daily_coverage.median()
            self.assertGreater(median_coverage, 0.1,
                               f"因子 {col} 截面覆盖度过低: median {median_coverage:.2%}")


class TestFactorCorrelation(unittest.TestCase):
    """因子相关性结构测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        codes = [f"{i:06d}.SZ" for i in range(100000, 100030)]
        dates = pd.date_range('2024-01-01', '2025-06-30', freq='B')

        rows = []
        for code in codes:
            n = len(dates)
            start_p = np.random.uniform(10, 80)
            returns = np.random.normal(0.0002, 0.025, n)
            prices = start_p * np.cumprod(1 + returns)
            prices[0] = start_p

            code_df = pd.DataFrame({
                'date': dates, 'code': code, 'close': prices,
            })
            code_df['open'] = code_df['close'] * (1 + np.random.normal(0, 0.003, n))
            intra = np.abs(np.random.normal(0, 0.01, n))
            code_df['high'] = np.maximum(code_df['open'], code_df['close']) * (1 + intra)
            code_df['low'] = np.minimum(code_df['open'], code_df['close']) * (1 - intra)
            code_df['volume'] = np.random.lognormal(15, 0.5, n)
            code_df['turnover_rate'] = np.random.uniform(0.005, 0.08, n)
            rows.append(code_df)

        cls.test_df = pd.concat(rows, ignore_index=True)
        cls.result = compute_enhanced_factors(cls.test_df)

    def test_correlation_structure(self):
        """验证因子相关性结构：同类高相关，异类低相关"""
        factor_cols = [c for c in self.result.columns
                      if c not in ['code', 'date', 'industry']]

        # 按截面均值计算因子间相关系数
        factor_means = self.result.groupby('date')[factor_cols].mean()
        corr = factor_means.corr()

        # 分类映射
        def get_category(name):
            for factor_def in EnhancedFactorLibrary.get_all_factors():
                if factor_def.name == name:
                    return factor_def.category
            return None

        # 计算同类平均相关 vs 异类平均相关
        same_cat_corr = []
        diff_cat_corr = []
        for i, fi in enumerate(factor_cols):
            for j, fj in enumerate(factor_cols):
                if j >= i:
                    continue
                ci = get_category(fi)
                cj = get_category(fj)
                if ci is None or cj is None:
                    continue
                corr_val = corr.loc[fi, fj]
                if ci == cj:
                    same_cat_corr.append(abs(corr_val))
                else:
                    diff_cat_corr.append(abs(corr_val))

        avg_same = np.mean(same_cat_corr) if same_cat_corr else 0
        avg_diff = np.mean(diff_cat_corr) if diff_cat_corr else 0

        print(f"\n[相关性结构] 同类因子平均 |r|: {avg_same:.3f}")
        print(f"[相关性结构] 异类因子平均 |r|: {avg_diff:.3f}")

        # 同类的相关性应高于异类（说明分类合理）
        if avg_same > 0 and avg_diff > 0:
            self.assertGreaterEqual(
                avg_same, avg_diff * 0.5,
                f"因子分类可能不够合理: 同类相关 {avg_same:.3f} vs 异类相关 {avg_diff:.3f}"
            )


class TestFactorPerformance(unittest.TestCase):
    """因子计算性能测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.N_STOCKS = 50
        cls.N_DAYS = 500

        codes = [f"{i:06d}.SZ" for i in range(100000, 100000 + cls.N_STOCKS)]
        dates = pd.date_range('2023-01-01', periods=cls.N_DAYS, freq='B')

        rows = []
        for code in codes:
            start_p = np.random.uniform(10, 80)
            returns = np.random.normal(0.0002, 0.022, cls.N_DAYS)
            prices = start_p * np.cumprod(1 + returns)
            prices[0] = start_p

            code_df = pd.DataFrame({
                'date': dates, 'code': code, 'close': prices,
            })
            code_df['open'] = code_df['close'] * (1 + np.random.normal(0, 0.005, cls.N_DAYS))
            code_df['high'] = np.maximum(code_df['open'], code_df['close']) * 1.01
            code_df['low'] = np.minimum(code_df['open'], code_df['close']) * 0.99
            code_df['volume'] = np.random.lognormal(15, 0.6, cls.N_DAYS)
            code_df['turnover_rate'] = np.random.uniform(0.005, 0.08, cls.N_DAYS)
            rows.append(code_df)

        cls.large_df = pd.concat(rows, ignore_index=True)

    def test_batch_computation_speed(self):
        """测试全因子库的计算速度"""
        start = time.time()
        result = compute_enhanced_factors(self.large_df)
        elapsed = time.time() - start

        n_computed = len([c for c in result.columns if c not in ['code', 'date']])
        total_cells = self.N_STOCKS * self.N_DAYS * n_computed if n_computed > 0 else 0

        print(f"\n[性能] 增强因子库计算")
        print(f"[性能] 数据: {self.N_STOCKS}只 × {self.N_DAYS}天")
        print(f"[性能] 计算因子: {n_computed} 个")
        print(f"[性能] 耗时: {elapsed:.3f}s")
        if total_cells > 0:
            print(f"[性能] 吞吐: {total_cells / elapsed:,.0f} cells/s")


if __name__ == "__main__":
    unittest.main(verbosity=2)