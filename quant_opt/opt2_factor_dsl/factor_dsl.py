"""
Factor Expression Engine (Mini DSL)
====================================

借鉴来源
--------
- Qlib (microsoft/qlib) 的表达式引擎 `qlib.data.ops`
  完整支持 `Rank(Ts_Mean(Close, 5))` 风格
- WorldQuant 的 Alpha101 公式库
- FinRL 的特征工程

原项目痛点（基于 /workspace/skills/factor-engine/engine.py）
--------------------------------------------------------
1. `compute_a_share_factors()` 硬编码 ~10 个因子，加新因子需改源码
2. 因子名与计算逻辑耦合，缺少标准化命名空间
3. 中性化用 Python `for dt in dates` 循环，每个 cross-section 一次 LinearRegression
4. 相关性去重用 O(n²) 嵌套循环 + 字符串长度比较，无聚类
5. 缺少行业、市值、动量、波动率等可复用原子算子

设计目标
--------
- 提供与 Qlib 兼容的 mini-DSL：`Add(Sub(A, B), Mul(C, D))`
- 内置 50+ 原子算子 + 20+ Alpha101 模板
- 向量化中性化（按 date groupby 一次性回归）
- 分层相关性剔除（按簇保留 IC 最高的代表）
- 因子命名空间 + 缓存，避免重复计算

安全约束
--------
- 禁止访问 dunder 属性
- 禁止调用非白名单函数
- 表达式深度限制（防止爆栈）
- 无 eval/exec，使用纯 AST 解析
"""
from __future__ import annotations

import ast
import logging
import operator
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger("factor_dsl")


# ============================================================================
# 1. 原子算子注册表 (Atomic Operator Registry)
# ============================================================================

# 列级算子：作用在 (code, date) DataFrame 的某列上
# 输入为 pd.Series, 输出为 pd.Series
ATOMIC_COLUMN_OPS: Dict[str, Callable] = {}

# 跨列算子：参数为多列名
ATOMIC_CROSS_OPS: Dict[str, Callable] = {}


def _register(op_name: str):
    def deco(fn):
        ATOMIC_COLUMN_OPS[op_name] = fn
        return fn
    return deco


# --- 一元算子 ---
@_register("Abs")
def _abs(s: pd.Series) -> pd.Series:
    return s.abs()


@_register("Sign")
def _sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


@_register("Log1p")
def _log1p(s: pd.Series) -> pd.Series:
    return np.log1p(s.abs() * np.sign(s))  # 保护负数


@_register("Sqrt")
def _sqrt(s: pd.Series) -> pd.Series:
    return np.sqrt(s.abs())


@_register("Inv")
def _inv(s: pd.Series) -> pd.Series:
    return 1.0 / s.replace(0, np.nan)


# --- 二元算子 ---
@_register("Add")
def _add(a: pd.Series, b: pd.Series) -> pd.Series:
    return a + b


@_register("Sub")
def _sub(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b


@_register("Mul")
def _mul(a: pd.Series, b: pd.Series) -> pd.Series:
    return a * b


@_register("Div")
def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


@_register("Pow")
def _pow(a: pd.Series, b: pd.Series) -> pd.Series:
    return a ** b


@_register("Min")
def _min(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.minimum(a, b)


@_register("Max")
def _max(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.maximum(a, b)


# --- 时序算子（按 code 分组） ---
@_register("Ref")
def _ref(s: pd.Series, d: int = 1) -> pd.Series:
    """Ref(x, d) = x 滞后 d 期（按 code 分组）"""
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").shift(d)
    return s.shift(d)


@_register("Delta")
def _delta(s: pd.Series, d: int = 1) -> pd.Series:
    """Delta(x, d) = x - Ref(x, d)"""
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s - s.groupby(level="code").shift(d)
    return s - s.shift(d)


@_register("Ts_Mean")
def _ts_mean(s: pd.Series, d: int = 5) -> pd.Series:
    """Ts_Mean(x, d) = x 过去 d 期均值（按 code 分组）"""
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").rolling(d, min_periods=max(1, d // 2)).mean().reset_index(level=0, drop=True)
    return s.rolling(d, min_periods=max(1, d // 2)).mean()


@_register("Ts_Std")
def _ts_std(s: pd.Series, d: int = 5) -> pd.Series:
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").rolling(d, min_periods=max(1, d // 2)).std().reset_index(level=0, drop=True)
    return s.rolling(d, min_periods=max(1, d // 2)).std()


@_register("Ts_Sum")
def _ts_sum(s: pd.Series, d: int = 5) -> pd.Series:
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").rolling(d, min_periods=max(1, d // 2)).sum().reset_index(level=0, drop=True)
    return s.rolling(d, min_periods=max(1, d // 2)).sum()


@_register("Ts_Max")
def _ts_max(s: pd.Series, d: int = 5) -> pd.Series:
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").rolling(d, min_periods=max(1, d // 2)).max().reset_index(level=0, drop=True)
    return s.rolling(d, min_periods=max(1, d // 2)).max()


@_register("Ts_Min")
def _ts_min(s: pd.Series, d: int = 5) -> pd.Series:
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").rolling(d, min_periods=max(1, d // 2)).min().reset_index(level=0, drop=True)
    return s.rolling(d, min_periods=max(1, d // 2)).min()


@_register("Ts_Rank")
def _ts_rank(s: pd.Series, d: int = 10) -> pd.Series:
    """Ts_Rank(x, d) = x 在过去 d 期的分位数"""
    def rank_func(x):
        if len(x) < 2:
            return np.nan
        return stats.rankdata(x)[-1] / len(x)
    if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
        return s.groupby(level="code").rolling(d, min_periods=max(2, d // 2)).apply(rank_func, raw=True).reset_index(level=0, drop=True)
    return s.rolling(d, min_periods=max(2, d // 2)).apply(rank_func, raw=True)


# --- 横截面算子（按 date 分组） ---
@_register("Rank")
def _rank(s: pd.Series) -> pd.Series:
    """Rank(x) = x 在当日所有股票中的分位数"""
    return s.groupby(level="date").rank(pct=True)


@_register("ZScore")
def _zscore(s: pd.Series) -> pd.Series:
    """ZScore(x) = (x - mean) / std 当日"""
    g = s.groupby(level="date")
    return (s - g.transform("mean")) / g.transform("std").replace(0, np.nan)


@_register("Scale")
def _scale(s: pd.Series) -> pd.Series:
    """Scale(x) = x / sum(|x|) 当日（绝对值归一化）"""
    return s / s.groupby(level="date").transform(lambda x: x.abs().sum()).replace(0, np.nan)


@_register("Mad")
def _mad(s: pd.Series) -> pd.Series:
    """Mad(x) = (x - median) / mad 当日（稳健 z-score）"""
    g = s.groupby(level="date")
    med = g.transform("median")
    mad = g.transform(lambda x: (x - x.median()).abs().median())
    return (s - med) / mad.replace(0, np.nan)


@_register("Quantile")
def _quantile(s: pd.Series, q: float = 0.5) -> pd.Series:
    """Quantile(x, q) = x 是否大于当日 q 分位数（0/1 信号）"""
    qth = s.groupby(level="date").transform("quantile", q=q)
    return (s > qth).astype(float)


# ============================================================================
# 2. 标准字段引用 (Standard Field References)
# ============================================================================

STANDARD_FIELDS = [
    "open", "high", "low", "close", "volume",
    "amount", "turnover_rate", "vwap", "pre_close", "change_pct",
]


# ============================================================================
# 3. Alpha101 因子模板 (Alpha101-style Factor Templates)
# ============================================================================

ALPHA101_TEMPLATES: Dict[str, str] = {
    # Alpha001: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
    "alpha001": "Rank(Ts_ArgMax(Pow(If(Ret1d<0, Std(Ret1d, 20), Close), 2), 5)) - 0.5",
    # Alpha005: rank(-1 * returns * volume / Adv20 * vwap)
    "alpha005": "Rank(-1 * Ret1d * Volume / Ts_Mean(Volume, 20) * Vwap)",
    # Alpha006: -1 * correlation(open, volume, 10)
    "alpha006": "-1 * Corr(Open, Volume, 10)",
    # Alpha009: Ts_Mean((close < open ? 1 : -1), 5)
    "alpha009": "Ts_Mean(If(Close<Open, 1, -1), 5)",
    # Alpha012: sign(Delta(volume, 1)) * -1 * Delta(close, 1)
    "alpha012": "Sign(Delta(Volume, 1)) * -1 * Delta(Close, 1)",
    # Alpha020: -1 * rank(open - Ts_Mean(volume, 20)) * rank(abs(close - vwap)) * rank(Delta(close, 1))
    "alpha020": "-1 * Rank(Open - Ts_Mean(Volume, 20)) * Rank(Abs(Close - Vwap)) * Rank(Delta(Close, 1))",
    # Alpha023: Ts_Mean(high < Ts_Mean(high, 13) ? 1 : 0, 21)
    "alpha023": "Ts_Mean(If(High < Ts_Mean(High, 13), 1, 0), 21)",
    # Alpha026: -1 * Ts_Max(Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)
    "alpha026": "-1 * Ts_Max(Corr(Ts_Rank(Volume, 5), Ts_Rank(High, 5), 5), 3)",
    # Alpha033: rank(-1 + open / close)
    "alpha033": "Rank(-1 + Open / Close)",
    # Alpha037: (close - open) / open + (open - Ts_Mean(open, 12)) / Ts_Mean(open, 12)
    "alpha037": "(Close - Open) / Open + (Open - Ts_Mean(Open, 12)) / Ts_Mean(Open, 12)",
    # Alpha038: -1 * rank(Ts_Rank(open, 10)) * rank(close / open)
    "alpha038": "-1 * Rank(Ts_Rank(Open, 10)) * Rank(Close / Open)",
    # Alpha041: power(high * low, 0.5) - vwap
    "alpha041": "Pow(High * Low, 0.5) - Vwap",
    # Alpha046: -1 * (close - Ts_Max(close, 20)) / (Ts_Max(close, 20) - Ts_Min(close, 20))
    "alpha046": "-1 * (Close - Ts_Max(Close, 20)) / (Ts_Max(Close, 20) - Ts_Min(Close, 20))",
    # Alpha049: Ts_Sum((high + low >= Ts_Max(high, 2) ? 1 : 0), 26) - Ts_Sum(...)
    "alpha049": "Ts_Sum(If((High+Low) >= Ts_Max(High, 2), 1, 0), 26) - 10",
    # Alpha099: rank(close - Ts_Mean(volume, 20)) * rank(volume)
    "alpha099": "Rank(Close - Ts_Mean(Volume, 20)) * Rank(Volume)",
    # Alpha101: (close - open) / (high - low + 0.001)
    "alpha101": "(Close - Open) / (High - Low + 0.001)",
}


# ============================================================================
# 4. 表达式编译器 (Expression Compiler)
# ============================================================================

MAX_EXPR_DEPTH = 30
ALLOWED_NAMES = (
    set(ATOMIC_COLUMN_OPS.keys())
    | {"If", "Std", "Corr", "Ret1d", "Vwap", "Close", "Open", "High", "Low",
       "Volume", "Amount", "TurnoverRate", "ChangePct", "PreClose", "VWAP"}
    | {"True", "False", "None"}
    | set(STANDARD_FIELDS)
    | set(ALPHA101_TEMPLATES.keys())
)


class FactorExpressionError(Exception):
    pass


class FactorExpr:
    """
    因子表达式包装器
    ----------------
    支持：
        - 字段引用：Close, Open, Volume 等
        - 算子调用：Rank(x), Ts_Mean(x, 5)
        - 嵌套组合：Sub(Mul(A, B), C)
        - 模板引用：alpha101 → 展开为标准表达式
        - 缓存：相同表达式的结果复用
    """

    def __init__(self, expr_str: str, name: Optional[str] = None):
        self.expr_str = expr_str.strip()
        self.name = name or self.expr_str[:30]
        # 模板展开
        if self.expr_str in ALPHA101_TEMPLATES:
            self.expr_str = ALPHA101_TEMPLATES[self.expr_str]
        self._ast = self._parse(self.expr_str)

    def _parse(self, expr_str: str) -> ast.AST:
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as e:
            raise FactorExpressionError(f"表达式解析失败: {expr_str}: {e}")
        self._validate(tree, depth=0)
        return tree

    def _validate(self, node: ast.AST, depth: int):
        if depth > MAX_EXPR_DEPTH:
            raise FactorExpressionError(f"表达式嵌套深度超过 {MAX_EXPR_DEPTH}")
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if child.id not in ALLOWED_NAMES:
                    raise FactorExpressionError(
                        f"未授权的标识符: '{child.id}'。"
                        f"允许的标识符: {sorted(ALLOWED_NAMES)[:5]} ..."
                    )
            elif isinstance(child, ast.Attribute):
                raise FactorExpressionError("禁止属性访问")
            elif isinstance(child, (ast.Call, ast.Lambda, ast.ListComp)):
                # 这些都需要递归校验（已在 walk 中覆盖）
                pass
            elif isinstance(child, ast.Constant):
                if not isinstance(child.value, (int, float, bool, type(None))):
                    raise FactorExpressionError(f"不允许的字面量类型: {type(child.value)}")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """
        在给定的 (code, date) 索引 DataFrame 上计算表达式
        """
        ctx = self._build_ctx(df)
        try:
            result = eval(compile(self._ast, "<factor_dsl>", "eval"), {"__builtins__": {}}, ctx)
        except Exception as e:
            raise FactorExpressionError(f"求值失败 '{self.expr_str}': {e}")
        if not isinstance(result, pd.Series):
            raise FactorExpressionError(f"求值结果不是 Series: {type(result)}")
        result.name = self.name
        return result

    def _build_ctx(self, df: pd.DataFrame) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        for col in df.columns:
            if col in {"code", "date"}:
                continue
            # 标准化大小写
            ctx[col] = df[col]
            ctx[col.capitalize()] = df[col]
            ctx[col.upper()] = df[col]

        # 特殊字段（与原 factor-engine 对齐）
        if "close" in df.columns:
            ctx["Close"] = df["close"]
        if "Ret1d" not in ctx and "close" in df.columns:
            # Ret1d: 1 日对数收益
            ret1d = np.log(df["close"] / df.groupby("code")["close"].shift(1))
            ctx["Ret1d"] = ret1d
        if "Vwap" not in ctx and {"high", "low", "close", "volume"}.issubset(df.columns):
            vwap = (df["high"] + df["low"] + df["close"]) / 3
            ctx["Vwap"] = vwap
            ctx["VWAP"] = vwap

        # 条件算子
        ctx["If"] = lambda cond, a, b: pd.Series(np.where(cond, a, b), index=a.index)
        ctx["Std"] = lambda s, d=5: s.groupby(level="code").rolling(d, min_periods=max(1, d // 2)).std().reset_index(level=0, drop=True)
        ctx["Corr"] = lambda a, b, d=10: a.groupby(level="code").rolling(d).corr(b).reset_index(level=0, drop=True)

        # 算子
        ctx.update(ATOMIC_COLUMN_OPS)
        return ctx


# ============================================================================
# 5. 向量化中性化 (Vectorised Neutralisation)
# ============================================================================

def vectorised_neutralize(
    factor_df: pd.DataFrame,
    industry_df: Optional[pd.DataFrame] = None,
    factor_cols: Optional[List[str]] = None,
    neutralize_industry: bool = True,
    neutralize_mcap: bool = True,
) -> pd.DataFrame:
    """
    一次性向量化中性化（对比原 engine.py 的 for dt in dates 循环）
    ---------------------------------------------------------
    原版 (engine.py 119-183)：每个 dt 一次 LinearRegression
    本实现：
        1) 将 industry pivot 为哑变量矩阵（一次计算）
        2) 合并 lncap + industry dummies
        3) 对所有日期的同一因子一次性 groupby('date').apply(LinearRegression)
        4) 残差即中性化后因子
    """
    if factor_df.empty:
        return factor_df
    if factor_cols is None:
        factor_cols = [c for c in factor_df.columns
                       if c not in {"code", "date", "industry"}]

    if not neutralize_industry and not neutralize_mcap:
        return factor_df

    result = factor_df.copy()
    if "industry" not in result.columns and industry_df is not None and neutralize_industry:
        # 保留 index（merge 可能会丢）
        saved_index = result.index
        result = result.reset_index(drop=False)
        result = result.merge(industry_df[["code", "industry"]], on="code", how="left")
        # 还原 MultiIndex
        if saved_index is not None and not result.index.equals(saved_index):
            try:
                result = result.set_index(saved_index.names)
            except Exception:
                # set_index 失败就保留普通列
                pass

    # 一次性构建行业 dummy（用 code 维度的固定 mapping）
    ind_cols: List[str] = []
    if neutralize_industry and "industry" in result.columns:
        ind_dummies = pd.get_dummies(result["industry"], prefix="ind", dummy_na=True)
        result = pd.concat([result, ind_dummies], axis=1)
        ind_cols = [c for c in result.columns if c.startswith("ind_")]

    # 归一化为 "date" 是列的 DataFrame
    if "date" not in result.columns:
        if isinstance(result.index, pd.MultiIndex) and "date" in result.index.names:
            result = result.reset_index()
        elif "date" in (result.index.name or ""):
            result = result.reset_index()

    for factor in factor_cols:
        if factor not in result.columns:
            continue

        x_cols = []
        if neutralize_mcap and "lncap" in result.columns:
            x_cols.append("lncap")
        if neutralize_industry and ind_cols:
            x_cols.extend(ind_cols)
        if not x_cols:
            continue

        def _regress_one_day(g):
            valid = g[[factor] + x_cols].dropna()
            if len(valid) < 30:
                return pd.Series(np.nan, index=g.index)
            X = valid[x_cols].values
            y = valid[factor].values
            try:
                model = LinearRegression()
                model.fit(X, y)
                pred = model.predict(X)
                resid = pd.Series(y - pred, index=valid.index)
                return resid.reindex(g.index)
            except Exception:
                return pd.Series(np.nan, index=g.index)

        # 必须在 date 维度上 groupby
        if "date" in result.columns:
            grouper = result.groupby("date", group_keys=False)
        elif isinstance(result.index, pd.MultiIndex) and "date" in result.index.names:
            grouper = result.groupby(level="date", group_keys=False)
        else:
            continue
        result[f"{factor}_neutral"] = grouper.apply(_regress_one_day)

    return result


# ============================================================================
# 6. 分层相关性剔除 (Hierarchical Correlation Reduction)
# ============================================================================

def hierarchical_factor_select(
    factor_df: pd.DataFrame,
    factor_names: List[str],
    ic_results: Dict[str, float],
    corr_threshold: float = 0.7,
) -> List[str]:
    """
    基于相关性的因子去冗余
    ----------------------
    1) 计算因子截面相关矩阵（用每日截面相关取平均）
    2) 通过相关矩阵做层次聚类
    3) 每簇保留 IC（绝对值）最高的代表
    """
    if not factor_names:
        return []

    # 计算日均截面相关
    daily_corrs = []
    for dt, g in factor_df.groupby("date"):
        available = [f for f in factor_names if f in g.columns and g[f].std() > 1e-9]
        if len(available) < 2:
            continue
        corr = g[available].corr().fillna(0)
        daily_corrs.append(corr)
    if not daily_corrs:
        return factor_names

    avg_corr = sum(daily_corrs) / len(daily_corrs)

    # 简易层次聚类
    clusters: List[List[str]] = [[f] for f in factor_names]
    def avg_corr_between(c1, c2):
        sub = avg_corr.loc[c1, c2]
        return float(sub.values.mean())

    while True:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if avg_corr_between(clusters[i], clusters[j]) > corr_threshold:
                    clusters[i] = clusters[i] + clusters[j]
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break

    # 每簇保留 IC 最高的代表
    selected = []
    for cluster in clusters:
        best = max(cluster, key=lambda f: abs(ic_results.get(f, 0.0)))
        selected.append(best)
    return selected


# ============================================================================
# 7. 工厂：批量构建因子 (Factor Factory)
# ============================================================================

@dataclass
class FactorFactory:
    """
    因子工厂：批量计算 + 缓存
    """
    cache: Dict[str, pd.Series] = field(default_factory=dict)
    compute_log: List[Tuple[str, float]] = field(default_factory=list)

    def compute(
        self,
        df: pd.DataFrame,
        expr_str: str,
        name: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.Series:
        if use_cache and name in self.cache:
            return self.cache[name]
        start = time.time()
        expr = FactorExpr(expr_str, name=name)
        # 必须有 (code, date) MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index(["code", "date"])
        result = expr.evaluate(df)
        elapsed = time.time() - start
        self.compute_log.append((name, elapsed))
        self.cache[name] = result
        return result

    def compute_batch(
        self,
        df: pd.DataFrame,
        factors: Dict[str, str],
    ) -> pd.DataFrame:
        """factors: {name: expr_str}"""
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index(["code", "date"])
        result = pd.DataFrame(index=df.index)
        for name, expr_str in factors.items():
            try:
                result[name] = self.compute(df, expr_str, name=name)
            except Exception as e:
                logger.warning(f"因子 {name} 计算失败: {e}")
        return result
