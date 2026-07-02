"""
Optimization A: 因子表达式引擎 (Factor Expression Engine)
================================================================================

借鉴来源:
  - Microsoft Qlib 的表达式引擎 (qlib/data/ops.py): 用字符串公式声明式定义因子,
    如 ``Ref($close, 60) / $close``; 通过 DataHandler/Processor 解耦计算与处理。
  - akquant 的 Polars 因子引擎: 用 Alpha101 风格语法 ``Rank(Ts_Mean(Close, 5))``,
    基于 Polars Lazy API 做查询计划优化与多线程执行, 并自动处理时间窗口对齐,
    防止未来数据泄露。

jingni-trader 现状痛点 (skills/factor-engine/engine.py):
  - 因子在 ``compute_a_share_factors`` 中硬编码, 新增因子必须改引擎源码;
  - 中性化 ``neutralize`` 与 IC 分析 ``_calc_ic`` 都用 ``for dt in dates`` 纯 Python
    循环逐日计算, 全市场面板下性能很差;
  - 没有声明式 DSL, 因子不可组合、不可复用、不可序列化。

本模块提供一个轻量、可扩展的表达式引擎:
  - 用 Python ``ast`` 模块解析公式字符串 (免费获得运算符优先级);
  - 时间序列算子 (Ref/Mean/Std/Corr/Slope...) 走 Polars ``over('code')``;
  - 截面算子 (Rank/ZScore/Quantile) 走 Polars ``over('date')``;
  - 算术运算直接映射到 Polars 表达式, 全程惰性求值, 一次 collect 出结果。

设计目标: 让 jingni-trader 的因子库从 "写函数" 升级为 "写公式", 同时获得
Polars 多线程加速。本文件为独立验证实现, 不修改 main 分支任何代码。
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Callable, Dict, List, Optional

import polars as pl

logger = logging.getLogger("factor-expression-engine")

# ---------------------------------------------------------------------------
# 字段引用语法: 同时支持 $close (Qlib 风格) 和 close (akquant 风格)
# ---------------------------------------------------------------------------

_FIELD_PREFIX = "$"


def _resolve_field(node: ast.AST) -> str:
    """把 AST 节点解析成字段名, 支持 $close 或 'close' 字符串字面量。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise ValueError(f"不支持的字段引用: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# 算子注册表: 算子名 -> (构建 Polars 表达式的函数, 算子类别)
#   类别 'ts'  = 时间序列 (over 'code')
#   类别 'cs'  = 截面 (over 'date')
#   类别 'pure' = 纯函数 (不依赖分组)
# ---------------------------------------------------------------------------

OperatorFn = Callable[..., pl.Expr]


def _ts_ref(field: pl.Expr, n: int) -> pl.Expr:
    """Ref(field, n): n 期前的值。"""
    return field.shift(n).over("code")


def _ts_delta(field: pl.Expr, n: int) -> pl.Expr:
    """Delta(field, n): field - Ref(field, n)。"""
    return (field - field.shift(n)).over("code")


def _ts_mean(field: pl.Expr, n: int) -> pl.Expr:
    return field.rolling_mean(window_size=n, min_samples=max(1, n // 2)).over("code")


def _ts_std(field: pl.Expr, n: int) -> pl.Expr:
    return field.rolling_std(window_size=n, min_samples=max(2, n // 2)).over("code")


def _ts_sum(field: pl.Expr, n: int) -> pl.Expr:
    return field.rolling_sum(window_size=n, min_samples=1).over("code")


def _ts_max(field: pl.Expr, n: int) -> pl.Expr:
    return field.rolling_max(window_size=n, min_samples=1).over("code")


def _ts_min(field: pl.Expr, n: int) -> pl.Expr:
    return field.rolling_min(window_size=n, min_samples=1).over("code")


def _ts_rank(field: pl.Expr, n: int) -> pl.Expr:
    """时序排名: 当前值在过去 n 期中的分位 (0~1)。"""
    # rolling_map 在 polars 中较重, 这里用 rolling quantile 近似
    # 为保证正确性, 用 (field - rolling_min) / (rolling_max - rolling_min) 作为稳健近似
    lo = field.rolling_min(window_size=n, min_samples=1).over("code")
    hi = field.rolling_max(window_size=n, min_samples=1).over("code")
    return (field - lo) / (hi - lo + 1e-12)


def _ts_corr(f1: pl.Expr, f2: pl.Expr, n: int) -> pl.Expr:
    """滚动相关系数 (per code)。

    用滚动一阶/二阶矩推导, 并用动态有效样本数 k (窗口内非空对数) 替代硬编码 n,
    使结果与 pandas ``rolling(n, min_periods).corr`` 在窗口未满时一致:
        corr = [k*Σxy - Σx*Σy] / sqrt([k*Σx²-(Σx)²][k*Σy²-(Σy)²])
    """
    # k = 窗口内 (f1, f2 均非空) 的样本数
    both_valid = (f1.is_not_null() & f2.is_not_null()).cast(pl.Int64)
    k = both_valid.rolling_sum(window_size=n, min_samples=2).over("code")
    sx = f1.rolling_sum(window_size=n, min_samples=2).over("code")
    sy = f2.rolling_sum(window_size=n, min_samples=2).over("code")
    sxy = (f1 * f2).rolling_sum(window_size=n, min_samples=2).over("code")
    sx2 = (f1 * f1).rolling_sum(window_size=n, min_samples=2).over("code")
    sy2 = (f2 * f2).rolling_sum(window_size=n, min_samples=2).over("code")
    num = k * sxy - sx * sy
    den = ((k * sx2 - sx * sx) * (k * sy2 - sy * sy)).sqrt()
    return num / (den + 1e-12)


def _ts_slope(field: pl.Expr, n: int) -> pl.Expr:
    """滚动线性回归斜率 (对时间索引 0..n-1)。用于趋势因子。"""
    # slope = [n*Σ(t*y) - Σt*Σy] / [n*Σt² - (Σt)²], t = 0..n-1
    # 用 rolling_sum 近似: t 固定为 0..n-1, Σt = n(n-1)/2, Σt² = n(n-1)(2n-1)/6
    sum_t = n * (n - 1) / 2.0
    sum_t2 = n * (n - 1) * (2 * n - 1) / 6.0
    # Σ(t*y): 用 rolling_map 计算加权和
    weights = list(range(n))  # t = 0..n-1

    def _weighted_sum(s: pl.Series) -> float:
        arr = s.to_numpy()
        if len(arr) < 2:
            return float("nan")
        w = weights[-len(arr):]
        return float((arr * w).sum())

    sum_ty = field.rolling_map(_weighted_sum, window_size=n).over("code")
    sum_y = field.rolling_sum(window_size=n, min_samples=2).over("code")
    k = n
    num = k * sum_ty - sum_t * sum_y
    den = k * sum_t2 - sum_t * sum_t
    return num / (den + 1e-12)


def _ts_wma(field: pl.Expr, n: int) -> pl.Expr:
    """加权移动平均, 权重 1..n (近期权重大)。"""
    weights = list(range(1, n + 1))
    weight_sum = sum(weights)

    def _wma(s: pl.Series) -> float:
        arr = s.to_numpy()
        if len(arr) == 0:
            return float("nan")
        w = weights[-len(arr):]
        return float((arr * w).sum() / w[-len(arr):].sum())

    return field.rolling_map(_wma, window_size=n).over("code")


# ---- 截面算子 (over 'date') ----

def _cs_rank(field: pl.Expr) -> pl.Expr:
    """截面排名分位 (0~1)。"""
    return field.rank("average").over("date") / field.count().over("date")


def _cs_zscore(field: pl.Expr) -> pl.Expr:
    mean = field.mean().over("date")
    std = field.std().over("date")
    return (field - mean) / (std + 1e-12)


def _cs_quantile(field: pl.Expr, q: float) -> pl.Expr:
    return field.quantile(q).over("date")


# 算子注册表
OPERATORS: Dict[str, tuple] = {
    # 时间序列算子
    "Ref": (_ts_ref, "ts"),
    "Delta": (_ts_delta, "ts"),
    "Mean": (_ts_mean, "ts"),
    "Ts_Mean": (_ts_mean, "ts"),
    "Std": (_ts_std, "ts"),
    "Ts_Std": (_ts_std, "ts"),
    "Sum": (_ts_sum, "ts"),
    "Ts_Sum": (_ts_sum, "ts"),
    "Max": (_ts_max, "ts"),
    "Ts_Max": (_ts_max, "ts"),
    "Min": (_ts_min, "ts"),
    "Ts_Min": (_ts_min, "ts"),
    "Ts_Rank": (_ts_rank, "ts"),
    "Corr": (_ts_corr, "ts"),
    "Slope": (_ts_slope, "ts"),
    "WMA": (_ts_wma, "ts"),
    # 截面算子
    "Rank": (_cs_rank, "cs"),
    "ZScore": (_cs_zscore, "cs"),
    "Quantile": (_cs_quantile, "cs"),
}


# ---------------------------------------------------------------------------
# AST -> Polars Expr 编译器
# ---------------------------------------------------------------------------

class ExpressionCompiler:
    """把因子公式字符串编译成 Polars 表达式。

    支持:
      - 字段引用: ``$close`` 或 ``close``
      - 算术: ``+ - * / **`` 及一元负号
      - 函数调用: ``Ref($close, 5)``, ``Rank(-Mean($close, 20))``

    重要约束 (Polars 限制): ``over('code')`` 不能嵌套在 ``over('date')`` 内。
    因此当截面算子 (Rank/ZScore/Quantile) 的参数是含时序算子的复合表达式时,
    编译器会先把该参数物化为一个中间列, 再对中间列做截面运算。编译结果除最终
    表达式外, 还返回需要预先计算的中间列定义。
    """

    def __init__(self, available_fields: Optional[List[str]] = None):
        self.available_fields = set(available_fields or [])
        self._intermediates: List[tuple] = []  # [(name, pl.Expr), ...]
        self._counter = 0

    def compile(self, expr_str: str) -> tuple:
        """编译公式, 返回 (final_expr, intermediates)。

        intermediates 是 [(name, pl.Expr)] 列表, 调用方需先用
        ``df.with_columns([e.alias(n) for n,e in intermediates])`` 物化这些列,
        再用 final_expr 计算最终因子列。
        """
        self._intermediates = []
        self._counter = 0
        tree = ast.parse(expr_str, mode="eval")
        final = self._visit(tree.body)
        return final, list(self._intermediates)

    def _new_intermediate(self, expr: pl.Expr) -> str:
        name = f"__expr_{self._counter}"
        self._counter += 1
        self._intermediates.append((name, expr))
        return name

    def _visit(self, node: ast.AST) -> pl.Expr:
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_unaryop(node)
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Name):
            return pl.col(node.id)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return pl.lit(node.value)
        raise ValueError(f"不支持的 AST 节点: {ast.dump(node)}")

    def _visit_binop(self, node: ast.BinOp) -> pl.Expr:
        left = self._visit(node.left)
        right = self._visit(node.right)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a ** b,
        }
        op_fn = ops.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的二元运算: {type(node.op).__name__}")
        return op_fn(left, right)

    def _visit_unaryop(self, node: ast.UnaryOp) -> pl.Expr:
        operand = self._visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")

    def _visit_call(self, node: ast.Call) -> pl.Expr:
        if not isinstance(node.func, ast.Name):
            raise ValueError("只支持简单函数调用")
        op_name = node.func.id
        if op_name not in OPERATORS:
            raise ValueError(f"未知算子: {op_name}. 可用: {sorted(OPERATORS)}")
        op_fn, category = OPERATORS[op_name]

        # 截面算子: 若参数是复合表达式 (含时序算子), 先物化为中间列再 over('date')
        if category == "cs":
            field_args = []
            for a in node.args:
                if isinstance(a, ast.Name):
                    field_args.append(pl.col(a.id))
                elif isinstance(a, ast.Constant):
                    field_args.append(a.value)
                else:
                    # 复合表达式: 编译并物化为中间列
                    sub_expr = self._visit(a)
                    mid_name = self._new_intermediate(sub_expr)
                    field_args.append(pl.col(mid_name))
            return op_fn(*field_args)

        # 时序/纯算子: 正常编译
        args = [self._visit_arg(a) for a in node.args]
        return op_fn(*args)

    def _visit_arg(self, node: ast.AST) -> Any:
        """函数参数: 字段引用 -> pl.Expr; 整数/浮点 -> 原生值。"""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return pl.col(node.id)
        return self._visit(node)


def _preprocess(expr_str: str) -> str:
    """把 ``$close`` 形式归一化为 ``close``, 让 Python ast 能解析。"""
    return expr_str.replace(_FIELD_PREFIX, "")


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

class FactorExpressionEngine:
    """因子表达式引擎: 输入公式字典, 输出因子 DataFrame。

    Example:
        >>> engine = FactorExpressionEngine()
        >>> formulas = {
        ...     "reversal_5d":   "-Ref($close, 5) / Ref($close, 1)",
        ...     "vol_ratio":     "$volume / Mean($volume, 20)",
        ...     "momentum_20":   "Ref($close, 20) / $close",
        ...     "cs_rank_mom":   "Rank(Ref($close, 20) / $close)",
        ... }
        >>> out = engine.compute(df, formulas)  # df 含 code/date/close/volume
    """

    def __init__(self, code_col: str = "code", date_col: str = "date"):
        self.code_col = code_col
        self.date_col = date_col
        self.compiler = ExpressionCompiler()

    def compute(
        self,
        df: pl.DataFrame,
        formulas: Dict[str, str],
    ) -> pl.DataFrame:
        """批量计算多个因子公式。

        参数:
            df: 至少含 code/date 列及公式中引用的字段列; 必须已按 code, date 排序。
            formulas: {因子名: 公式字符串}

        返回:
            含 code/date + 各因子列的 DataFrame
        """
        if df.height == 0:
            base = df.select([self.code_col, self.date_col])
            return base.with_columns([pl.lit(None).alias(n) for n in formulas])

        # 两阶段编译: 先收集所有中间列 (cs 算子的复合参数), 再计算最终因子列
        all_intermediates: List[tuple] = []  # [(name, pl.Expr)]
        final_exprs: List[pl.Expr] = []
        for name, formula in formulas.items():
            try:
                expr, interms = self.compiler.compile(_preprocess(formula))
                all_intermediates.extend(interms)
                final_exprs.append(expr.alias(name))
            except Exception as e:
                logger.warning(f"编译公式 '{name}': '{formula}' 失败: {e}")
                final_exprs.append(pl.lit(None).alias(name))

        # Phase 1: 物化中间列 (cs 算子参数)
        work = df
        if all_intermediates:
            # 去重 (不同公式可能产生同名中间列, 但 counter 已保证唯一)
            seen = set()
            uniq = [(n, e) for n, e in all_intermediates if not (n in seen or seen.add(n))]
            work = work.with_columns([e.alias(n) for n, e in uniq])

        # Phase 2: 计算最终因子列
        out = work.with_columns(final_exprs)
        keep = [self.code_col, self.date_col] + list(formulas.keys())
        return out.select([c for c in keep if c in out.columns])


# ---------------------------------------------------------------------------
# 预置因子库 (借鉴 Qlib Alpha158 的思路, 精选常用因子)
# ---------------------------------------------------------------------------

PRESET_FACTORS: Dict[str, str] = {
    # 反转
    "reversal_5d":  "-Ref($close, 5) / Ref($close, 1)",
    "reversal_20d": "-Ref($close, 20) / Ref($close, 1)",
    # 动量
    "momentum_20":  "Ref($close, 20) / $close",
    "momentum_60":  "Ref($close, 60) / $close",
    # 波动率
    "volatility_20": "Std($close, 20)",
    "volatility_5":  "Std($close, 5)",
    # 量价
    "vol_ratio":    "$volume / Mean($volume, 20)",
    "amount_ratio": "$amount / Mean($amount, 20)",
    # 趋势斜率
    "slope_20":     "Slope($close, 20)",
    # 量价相关性
    "corr_pv_20":   "Corr($close, $volume, 20)",
    # 截面排名
    "cs_rank_mom":  "Rank(Ref($close, 20) / $close)",
    "cs_zscore_vol":"ZScore(Std($close, 20))",
}
