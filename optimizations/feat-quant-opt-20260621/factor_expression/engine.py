"""
因子表达式引擎 (Factor Expression Engine)

借鉴来源:
- Microsoft Qlib: 表达式 DSL，支持 $close, Mean($close, 20), Ref($close, 1) 等
- WorldQuant Alpha101: 因子公式语法
- alphalens: 因子分析接口

针对 jingni-trader factor-engine 的优化点:
原版 compute_a_share_factors() 把所有因子硬编码在 Python 函数中，新增因子需要改源码、
无法运行时配置。本引擎提供:
1. 字符串表达式定义因子，运行时解析
2. 支持字段引用: $close, $open, $high, $low, $volume, $amount, $turnover_rate
3. 支持算术运算: + - * / ** ( )
4. 支持时序滚动: Mean, Std, Max, Min, Sum, Var, Skew, Kurt, Med
5. 支持时序引用: Ref(expr, n)  (n>0 未来, n<0 过去)
6. 支持横截面运算: CSRank (cross-sectional rank), CSZScore
7. 安全沙箱: 白名单运算符与函数，禁止 import/eval/exec
8. 自动处理停牌缺失值 (前向填充 + 跳过 NaN)

示例:
    engine = FactorExpressionEngine()
    df = engine.compute(data, {
        "momentum_20d": "$close / Ref($close, 20) - 1",
        "ma5": "Mean($close, 5)",
        "vol_20d": "Std($close / Ref($close,1) - 1, 20) * Sqrt(252)",
        "rsi_like": "CSRank(Mean($close, 5) - Mean($close, 20))",
        "reversal_5d": "-1 * ($close / Ref($close, 5) - 1)",
    })
"""
import ast
import operator
import logging
import math
from typing import Dict, List, Any, Callable, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("factor-expression-engine")


# ---------------------------------------------------------------------
# 安全的运算符与函数白名单
# ---------------------------------------------------------------------
_SAFE_BINOPS: Dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_SAFE_UNARYOPS: Dict[type, Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 滚动窗口函数 (按 code 分组)
def _rolling_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(1, n // 2)).mean()

def _rolling_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).std()

def _rolling_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).max()

def _rolling_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).min()

def _rolling_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).sum()

def _rolling_var(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).var()

def _rolling_skew(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(3, n // 2)).skew()

def _rolling_kurt(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(4, n // 2)).kurt()

def _rolling_median(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).median()

def _ref(s: pd.Series, n: int) -> pd.Series:
    """Ref($close, n): n>0 取过去第n期, n<0 取未来第|n|期"""
    if n >= 0:
        return s.shift(n)
    else:
        return s.shift(n)  # shift 负数 = 未来

def _rank_ts(s: pd.Series, n: int) -> pd.Series:
    """时序排名 (在窗口 n 内的百分位)"""
    return s.rolling(n, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

def _ewm_mean(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, min_periods=1).mean()

def _ewm_std(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, min_periods=1).std()

def _delta(s: pd.Series, n: int = 1) -> pd.Series:
    return s.diff(n)

def _abs(s: pd.Series) -> pd.Series:
    return s.abs()

def _log(s: pd.Series) -> pd.Series:
    return np.log(s.replace(0, np.nan))

def _sqrt(s: pd.Series) -> pd.Series:
    return np.sqrt(s.clip(lower=0))

def _sign(s: pd.Series) -> pd.Series:
    return np.sign(s)

# 函数注册表: name -> (callable, is_windowed)
# is_windowed=True 表示该函数第二个参数是窗口长度
_FUNCTION_REGISTRY: Dict[str, tuple] = {
    "Mean": (_rolling_mean, True),
    "Std": (_rolling_std, True),
    "Max": (_rolling_max, True),
    "Min": (_rolling_min, True),
    "Sum": (_rolling_sum, True),
    "Var": (_rolling_var, True),
    "Skew": (_rolling_skew, True),
    "Kurt": (_rolling_kurt, True),
    "Med": (_rolling_median, True),
    "Median": (_rolling_median, True),
    "Ref": (_ref, True),
    "Rank": (_rank_ts, True),
    "Ema": (_ewm_mean, True),
    "EwmMean": (_ewm_mean, True),
    "EwmStd": (_ewm_std, True),
    "Delta": (_delta, True),
    "Abs": (_abs, False),
    "Log": (_log, False),
    "Sqrt": (_sqrt, False),
    "Sign": (_sign, False),
}

# 横截面函数 (按 date 分组)
def _cs_rank(s: pd.Series) -> pd.Series:
    """横截面排名 (当日所有股票中的百分位)"""
    return s.groupby(level="date").rank(pct=True)

def _cs_zscore(s: pd.Series) -> pd.Series:
    """横截面 Z-Score"""
    grp = s.groupby(level="date")
    return (s - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)

def _cs_demean(s: pd.Series) -> pd.Series:
    """横截面去均值"""
    return s - s.groupby(level="date").transform("mean")

def _cs_quantile(s: pd.Series, n: int = 5) -> pd.Series:
    """横截面分层 (1~n)"""
    def q(x):
        try:
            return pd.qcut(x, n, labels=False, duplicates="drop") + 1
        except Exception:
            return pd.Series(1, index=x.index)
    return s.groupby(level="date").transform(q)

_CROSS_SECTIONAL_REGISTRY: Dict[str, tuple] = {
    "CSRank": (_cs_rank, False),
    "CSZScore": (_cs_zscore, False),
    "CSDemean": (_cs_demean, False),
    "CSQuantile": (_cs_quantile, True),
}

# 字段别名
_FIELD_ALIASES = {
    "$close": "close",
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$volume": "volume",
    "$vol": "volume",
    "$amount": "amount",
    "$turnover": "turnover_rate",
    "$turnover_rate": "turnover_rate",
    "$vwap": "vwap",
    "$pre_close": "pre_close",
    "$change_pct": "change_pct",
}


class ExpressionParser:
    """基于 Python ast 的安全表达式解析器"""

    # 预处理: 将 $field 转为合法标识符 _FIELD_field
    _PREFIX = "_F_"

    def __init__(self, expr: str):
        self.expr = expr
        # 预处理: $close -> _F_close (合法 Python 标识符)
        self._preprocessed = self._preprocess(expr)
        self.tree = ast.parse(self._preprocessed, mode="eval").body
        # 收集所有字段依赖
        self._field_deps = self._collect_field_deps(self.tree)

    @classmethod
    def _preprocess(cls, expr: str) -> str:
        """将 $field 转为合法标识符"""
        import re
        # 匹配 $word 或 $word_word
        return re.sub(r'\$(\w+)', lambda m: f"{cls._PREFIX}{m.group(1)}", expr)

    def _collect_field_deps(self, node: ast.AST) -> List[str]:
        """收集表达式依赖的字段 (返回 $xxx 形式)"""
        deps = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id.startswith(self._PREFIX):
                field = n.id[len(self._PREFIX):]
                deps.add(f"${field}")
        return sorted(deps)

    def evaluate(self, df: pd.DataFrame, group_col: str = "code") -> pd.Series:
        """在 DataFrame 上求值

        参数:
            df: 长表 DataFrame，必须含 group_col, date, 以及被引用的字段
            group_col: 分组列 (一般为 code)
        返回:
            pd.Series, 索引与 df 对齐
        """
        # 确保有 date 列用于横截面运算
        if "date" not in df.columns:
            raise ValueError("DataFrame 必须包含 date 列")
        # 设置多重索引便于横截面运算
        idx_df = df.set_index([group_col, "date"])
        result = self._eval_node(self.tree, df, idx_df, group_col)
        # 对齐回原 df
        if isinstance(result, pd.Series):
            # result 的索引可能是 MultiIndex (code, date)
            # 需要重新对齐到原 df 的顺序
            target_idx = df.set_index([group_col, "date"]).index
            return result.reindex(target_idx).values
        return result

    def _eval_node(
        self,
        node: ast.AST,
        df: pd.DataFrame,
        idx_df: pd.DataFrame,
        group_col: str,
    ) -> Any:
        # 常量
        if isinstance(node, ast.Constant):
            return node.value

        # 字段引用 ($xxx 预处理后变为 _F_xxx)
        if isinstance(node, ast.Name):
            return self._resolve_field(node.id, df, idx_df, group_col)

        # 二元运算
        if isinstance(node, ast.BinOp):
            op_func = _SAFE_BINOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            left = self._eval_node(node.left, df, idx_df, group_col)
            right = self._eval_node(node.right, df, idx_df, group_col)
            return self._safe_binop(op_func, left, right)

        # 一元运算
        if isinstance(node, ast.UnaryOp):
            op_func = _SAFE_UNARYOPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            operand = self._eval_node(node.operand, df, idx_df, group_col)
            return op_func(operand)

        # 函数调用
        if isinstance(node, ast.Call):
            return self._eval_call(node, df, idx_df, group_col)

        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    def _resolve_field(
        self,
        name: str,
        df: pd.DataFrame,
        idx_df: pd.DataFrame,
        group_col: str,
    ) -> Any:
        """解析字段引用"""
        # _F_close -> close (原始 $close)
        if name.startswith(self._PREFIX):
            field_name = name[len(self._PREFIX):]
            # 查找别名映射
            original = f"${field_name}"
            real_field = _FIELD_ALIASES.get(original, field_name)
            if real_field not in idx_df.columns:
                raise ValueError(f"字段 {original} 不存在于数据中。可用列: {list(idx_df.columns)}")
            return idx_df[real_field].copy()

        # 常量
        if name in ("pi", "e"):
            return math.pi if name == "pi" else math.e

        raise ValueError(f"未知标识符: {name}")

    def _safe_binop(self, op: Callable, left: Any, right: Any) -> Any:
        """安全的二元运算，处理 Series 与标量"""
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = op(left, right)
            if isinstance(result, pd.Series):
                result = result.replace([np.inf, -np.inf], np.nan)
            return result
        except Exception as e:
            raise ValueError(f"运算失败: {e}")

    def _eval_call(
        self,
        node: ast.Call,
        df: pd.DataFrame,
        idx_df: pd.DataFrame,
        group_col: str,
    ) -> Any:
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name is None:
            raise ValueError("只支持简单函数调用")

        # 横截面函数
        if func_name in _CROSS_SECTIONAL_REGISTRY:
            func, has_window = _CROSS_SECTIONAL_REGISTRY[func_name]
            args = [self._eval_node(a, df, idx_df, group_col) for a in node.args]
            if has_window and len(args) >= 2:
                # CSQuantile(series, n)
                series, n = args[0], int(args[1])
                return func(series, n)
            return func(args[0])

        # 时序函数
        if func_name in _FUNCTION_REGISTRY:
            func, is_windowed = _FUNCTION_REGISTRY[func_name]
            args = [self._eval_node(a, df, idx_df, group_col) for a in node.args]
            if is_windowed:
                if len(args) < 2:
                    raise ValueError(f"{func_name} 需要两个参数: (series, window)")
                series, window = args[0], int(args[1])
                # 按 group_col 分组应用滚动函数
                if isinstance(series, pd.Series):
                    return series.groupby(level=group_col).apply(
                        lambda x: func(x.droplevel(group_col) if isinstance(x.index, pd.MultiIndex) else x, window)
                    )
                else:
                    raise ValueError(f"{func_name} 第一个参数必须是 Series")
            else:
                # 非窗口函数 (Abs, Log, Sqrt, Sign)
                series = args[0]
                if isinstance(series, pd.Series):
                    return series.groupby(level=group_col).apply(
                        lambda x: func(x.droplevel(group_col) if isinstance(x.index, pd.MultiIndex) else x)
                    )
                return func(series)

        raise ValueError(f"未知函数: {func_name}")


class FactorExpressionEngine:
    """因子表达式引擎

    用法:
        engine = FactorExpressionEngine()
        factors_df = engine.compute(data, {
            "momentum_20d": "$close / Ref($close, 20) - 1",
            "vol_20d": "Std($close / Ref($close,1) - 1, 20) * Sqrt(252)",
        })
    """

    def __init__(self):
        self.parser_cache: Dict[str, ExpressionParser] = {}

    def compute(
        self,
        data: pd.DataFrame,
        factor_definitions: Dict[str, str],
        group_col: str = "code",
    ) -> pd.DataFrame:
        """
        批量计算因子

        参数:
            data: 长表 DataFrame，至少含 group_col, date, 以及被引用的字段
            factor_definitions: {因子名: 表达式字符串}
            group_col: 分组列

        返回:
            DataFrame，含 group_col, date, [各因子列]
        """
        if data.empty:
            return pd.DataFrame(columns=[group_col, "date"])

        # 预处理: 排序、设置索引
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values([group_col, "date"]).reset_index(drop=True)

        # 结果容器
        result = df[[group_col, "date"]].copy()

        for factor_name, expr in factor_definitions.items():
            try:
                parser = self.parser_cache.get(expr)
                if parser is None:
                    parser = ExpressionParser(expr)
                    self.parser_cache[expr] = parser
                values = parser.evaluate(df, group_col=group_col)
                # values 可能是 numpy array 或 Series
                if isinstance(values, pd.Series):
                    values = values.values
                result[factor_name] = values
                logger.info(f"因子 {factor_name} 计算完成: {expr}")
            except Exception as e:
                logger.error(f"因子 {factor_name} 计算失败 ({expr}): {e}")
                result[factor_name] = np.nan

        return result

    def list_available_functions(self) -> Dict[str, str]:
        """列出所有可用函数及说明"""
        funcs = {}
        for name, (fn, is_windowed) in _FUNCTION_REGISTRY.items():
            sig = f"{name}(series, window)" if is_windowed else f"{name}(series)"
            funcs[name] = sig
        for name, (fn, is_windowed) in _CROSS_SECTIONAL_REGISTRY.items():
            sig = f"{name}(series, n)" if is_windowed else f"{name}(series)"
            funcs[name] = sig
        return funcs

    def list_available_fields(self) -> List[str]:
        """列出所有可用字段"""
        return list(_FIELD_ALIASES.keys())

    def validate_expression(self, expr: str) -> Dict[str, Any]:
        """校验表达式语法与安全性 (不求值)"""
        try:
            parser = ExpressionParser(expr)
            deps = parser._field_deps

            # 安全检查: 遍历 AST，确保所有函数调用都在白名单内
            unsafe_calls = []
            for n in ast.walk(parser.tree):
                if isinstance(n, ast.Call):
                    func_name = n.func.id if isinstance(n.func, ast.Name) else None
                    if func_name is None:
                        unsafe_calls.append("非简单函数调用")
                    elif func_name not in _FUNCTION_REGISTRY and func_name not in _CROSS_SECTIONAL_REGISTRY:
                        unsafe_calls.append(func_name)
                # 检查属性访问 (如 __import__)
                if isinstance(n, ast.Attribute):
                    unsafe_calls.append(f"属性访问: {n.attr}")

            if unsafe_calls:
                return {
                    "valid": False,
                    "expression": expr,
                    "dependencies": deps,
                    "error": f"不安全的调用: {unsafe_calls}",
                }

            return {
                "valid": True,
                "expression": expr,
                "dependencies": deps,
                "error": None,
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "expression": expr,
                "dependencies": [],
                "error": f"语法错误: {e}",
            }
        except Exception as e:
            return {
                "valid": False,
                "expression": expr,
                "dependencies": [],
                "error": str(e),
            }

    def _collect_dependencies(self, node: ast.AST) -> List[str]:
        """收集表达式依赖的字段 (已弃用，保留兼容)"""
        deps = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id.startswith(ExpressionParser._PREFIX):
                field = n.id[len(ExpressionParser._PREFIX):]
                deps.add(f"${field}")
        return sorted(deps)
