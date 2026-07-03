"""
Factor Expression Engine (借鉴 AKQuant / Qlib 设计)

核心思路: 借鉴 AKQuant 的 akquant.factor.FactorEngine 与 微软 Qlib 的
Data Layer Expression Engine, 允许用户用公式字符串描述 Alpha 因子。

设计要点:
1. 算子分四类: 时序算子 (Ts_*)、截面算子 (Rank/Scale)、数学/逻辑算子、基本运算
2. 输入: DataFrame[code, date, open, high, low, close, volume, ...]
3. 输出: 因子值 DataFrame[code, date, <factor_name>]
4. 支持嵌套:  Rank(Ts_Mean($close, 5)) / Std($close, 20)
5. 内部用 Python AST 解析, 避免 eval 注入风险
"""
from __future__ import annotations
import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 字段引用: $open, $close, $high, $low, $volume, $amount, $vwap
# ---------------------------------------------------------------------------
FIELD_RE = re.compile(r"^\$([a-zA-Z_][a-zA-Z0-9_]*)$")
_FIELD_TOKEN_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")

VALID_FIELDS = {
    "open", "high", "low", "close", "volume", "amount", "vwap",
    "turnover", "returns", "pct_chg", "is_limit_up", "is_limit_down",
}


def _normalize_formula(formula: str) -> str:
    """把 $close 形式转换成 Field("close") 让 Python AST 可解析.
    不用正则替换避免与字符串字面量冲突——本公式语法不出现字符串字面量.
    """
    return _FIELD_TOKEN_RE.sub(r'Field("\1")', formula)


# ---------------------------------------------------------------------------
# 算子注册表
# ---------------------------------------------------------------------------
@dataclass
class OperatorSpec:
    """算子元信息"""
    name: str
    category: str           # 'ts' | 'cs' | 'math' | 'logic' | 'field'
    arity: int              # 参数个数 (算子调用形式下)
    description: str = ""
    example: str = ""


OPERATOR_REGISTRY: Dict[str, OperatorSpec] = {}


def _register(spec: OperatorSpec) -> None:
    OPERATOR_REGISTRY[spec.name] = spec


# 时序算子
_register(OperatorSpec("Ts_Mean",  "ts",   2, "滚动时序均值",       "Ts_Mean($close, 5)"))
_register(OperatorSpec("Ts_Sum",   "ts",   2, "滚动时序求和",       "Ts_Sum($volume, 10)"))
_register(OperatorSpec("Ts_Std",   "ts",   2, "滚动时序标准差",     "Ts_Std($close, 20)"))
_register(OperatorSpec("Ts_Min",   "ts",   2, "滚动时序最小值",     "Ts_Min($low, 20)"))
_register(OperatorSpec("Ts_Max",   "ts",   2, "滚动时序最大值",     "Ts_Max($high, 20)"))
_register(OperatorSpec("Ts_Median","ts",   2, "滚动时序中位数",     "Ts_Median($close, 20)"))
_register(OperatorSpec("Ts_Rank",  "ts",   2, "滚动时序百分位",     "Ts_Rank($close, 20)"))
_register(OperatorSpec("Delay",    "ts",   2, "时序滞后",           "Delay($close, 1)"))
_register(OperatorSpec("Delta",    "ts",   2, "一阶差分",           "Delta($close, 1)"))
# Rank 视为时序百分位排名(默认用法, 同 Ts_Rank)  -- 完整版可作为 CS 算子 meta
_register(OperatorSpec("Rank",     "ts",   2, "滚动时序百分位",     "Rank($close, 20)"))

# 截面算子
_register(OperatorSpec("Cs_Rank",  "cs",   1, "截面百分位排名",     "Cs_Rank($close)"))
_register(OperatorSpec("Cs_Scale", "cs",   1, "截面归一化(零和)",   "Cs_Scale($close)"))
_register(OperatorSpec("Cs_Zscore","cs",   1, "截面ZScore标准化",   "Cs_Zscore($close)"))

# 数学算子
for name in ("Abs", "Sign", "Log", "Sqrt"):
    _register(OperatorSpec(name, "math", 1, f"数学函数 {name}", f"{name}($close)"))
_register(OperatorSpec("Power",    "math", 2, "幂函数",             "Power($close, 2)"))

# 逻辑算子
for name in ("If", "And", "Or"):
    _register(OperatorSpec(name, "logic", 3 if name == "If" else 2, f"逻辑 {name}", f"{name}(a, b, c)"))


# ---------------------------------------------------------------------------
# 数据上下文
# ---------------------------------------------------------------------------
@dataclass
class FactorContext:
    """
    单只股票的时间序列上下文.
    所有时序算子都接收该上下文来计算.
    """
    df: pd.DataFrame      # 必须有 date 列, 其它列由 fields 提供

    def get(self, field: str) -> pd.Series:
        """读取某一列, 自动处理 $ 前缀"""
        name = field.lstrip("$")
        if name not in VALID_FIELDS and name not in self.df.columns:
            raise KeyError(f"未识别的字段: {field}")
        if name in self.df.columns:
            return self.df[name]
        # 找不到就按字段名生成(可能没有 is_limit_up 等标记)
        if name in ("is_limit_up", "is_limit_down"):
            return pd.Series(0, index=self.df.index)
        raise KeyError(f"字段 {name} 不在数据中: {self.df.columns.tolist()}")


# ---------------------------------------------------------------------------
# 算子实现
# ---------------------------------------------------------------------------
def _to_array(x: Any) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=False)
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    return np.asarray(x, dtype=float)


def _ts_window(ctx: FactorContext, fn: Callable[[np.ndarray, int], np.ndarray],
               field: str, window: int) -> pd.Series:
    arr = _to_array(ctx.get(field))
    out = fn(arr, int(window))
    return pd.Series(out, index=ctx.df.index)


def ts_mean(ctx: FactorContext, field: str, window: int) -> pd.Series:
    return _ts_window(ctx, lambda a, w: pd.Series(a).rolling(w).mean().to_numpy(), field, window)


def ts_sum(ctx: FactorContext, field: str, window: int) -> pd.Series:
    return _ts_window(ctx, lambda a, w: pd.Series(a).rolling(w).sum().to_numpy(), field, window)


def ts_std(ctx: FactorContext, field: str, window: int) -> pd.Series:
    return _ts_window(ctx, lambda a, w: pd.Series(a).rolling(w).std().to_numpy(), field, window)


def ts_min(ctx: FactorContext, field: str, window: int) -> pd.Series:
    return _ts_window(ctx, lambda a, w: pd.Series(a).rolling(w).min().to_numpy(), field, window)


def ts_max(ctx: FactorContext, field: str, window: int) -> pd.Series:
    return _ts_window(ctx, lambda a, w: pd.Series(a).rolling(w).max().to_numpy(), field, window)


def ts_median(ctx: FactorContext, field: str, window: int) -> pd.Series:
    return _ts_window(ctx, lambda a, w: pd.Series(a).rolling(w).median().to_numpy(), field, window)


def delay(ctx: FactorContext, field: str, n: int) -> pd.Series:
    return ctx.get(field).shift(int(n))


def delta(ctx: FactorContext, field: str, n: int) -> pd.Series:
    return ctx.get(field).diff(int(n))


def ts_rank(ctx: FactorContext, field: str, window: int) -> pd.Series:
    """滚动百分位排名"""
    arr = _to_array(ctx.get(field))
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(int(window) - 1, n):
        win = arr[i - int(window) + 1: i + 1]
        out[i] = (pd.Series(win).rank(pct=True).iloc[-1])
    return pd.Series(out, index=ctx.df.index)


# ---------------------------------------------------------------------------
# 截面算子 (输入: MultiIndex Series 或者 wide DF)
# ---------------------------------------------------------------------------
def cs_rank(group: pd.Series) -> pd.Series:
    """截面百分位排名: 输入 index=(date, code) 的 Series"""
    return group.groupby(level=0).rank(pct=True)


def cs_scale(group: pd.Series) -> pd.Series:
    """截面归一化: 让每日 sum(|x|)=1"""
    abs_sum = group.abs().groupby(level=0).sum()
    return group / abs_sum.reindex(group.index.get_level_values(0)).values


def cs_zscore(group: pd.Series) -> pd.Series:
    """截面 ZScore 标准化"""
    mean = group.groupby(level=0).transform("mean")
    std = group.groupby(level=0).transform("std")
    out = (group - mean) / std.replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# 数学/逻辑算子
# ---------------------------------------------------------------------------
def _apply_math(fn_name: str, x: Any) -> pd.Series:
    arr = _to_array(x)
    if fn_name == "Abs":
        return pd.Series(np.abs(arr), index=x.index if isinstance(x, pd.Series) else None)
    if fn_name == "Sign":
        return pd.Series(np.sign(arr), index=x.index if isinstance(x, pd.Series) else None)
    if fn_name == "Log":
        return pd.Series(np.log(np.where(arr > 0, arr, np.nan)), index=x.index if isinstance(x, pd.Series) else None)
    if fn_name == "Sqrt":
        return pd.Series(np.sqrt(np.where(arr >= 0, arr, np.nan)), index=x.index if isinstance(x, pd.Series) else None)
    raise ValueError(f"未知数学算子: {fn_name}")


def _if_op(cond: Any, a: Any, b: Any) -> pd.Series:
    """三目 If 算子: 条件为真返回 a, 否则返回 b"""
    c = _to_array(cond)
    a_arr = _to_array(a)
    b_arr = _to_array(b)
    # 条件 >0 视为真 (0/NaN 视为假)
    mask = (c > 0)
    out = np.where(mask, a_arr, b_arr)
    idx = None
    if isinstance(cond, pd.Series):
        idx = cond.index
    elif isinstance(a, pd.Series):
        idx = a.index
    elif isinstance(b, pd.Series):
        idx = b.index
    return pd.Series(out, index=idx)


def _and_op(a: Any, b: Any) -> pd.Series:
    ca = _to_array(a) > 0
    cb = _to_array(b) > 0
    out = np.where(ca & cb, 1, 0)
    idx = a.index if isinstance(a, pd.Series) else None
    return pd.Series(out, index=idx)


def _or_op(a: Any, b: Any) -> pd.Series:
    ca = _to_array(a) > 0
    cb = _to_array(b) > 0
    out = np.where(ca | cb, 1, 0)
    idx = a.index if isinstance(a, pd.Series) else None
    return pd.Series(out, index=idx)


# ---------------------------------------------------------------------------
# AST 解析与执行
# ---------------------------------------------------------------------------
class _ParseError(Exception):
    pass


def _parse_field(node: ast.Name) -> str:
    name = node.id
    if not name.startswith("$"):
        raise _ParseError(f"字段必须以 $ 开头: {name} (可用: {sorted(VALID_FIELDS)})")
    return name


def _eval_field_call(node: ast.Call, ctx: FactorContext) -> pd.Series:
    """处理 Field("name") 节点: 返回对应列"""
    if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
        raise _ParseError("Field() 只能接受 1 个字符串常量")
    name = node.args[0].value
    if not isinstance(name, str):
        raise _ParseError("Field() 参数必须是字符串")
    return ctx.get(name)


def _eval_ts_op(op_name: str, args: List[ast.AST], ctx: FactorContext) -> pd.Series:
    """时序算子求值: 前一参数是字段或子表达式, 第二个是窗口"""
    if len(args) != OPERATOR_REGISTRY[op_name].arity:
        raise _ParseError(f"{op_name} 需要 {OPERATOR_REGISTRY[op_name].arity} 个参数")
    field_arg = args[0]
    # 字段可能是 $xxx(已经转换成 Field("xxx")) 或 子表达式
    if isinstance(field_arg, ast.Call):
        if isinstance(field_arg.func, ast.Name) and field_arg.func.id == "Field":
            field_str = field_arg.args[0].value if isinstance(field_arg.args[0], ast.Constant) else None
            if not isinstance(field_str, str):
                raise _ParseError("Field() 参数必须为字符串常量")
            field = "$" + field_str
        else:
            field = _eval_call(field_arg, ctx)
            win = _eval_const(args[1])
            fn = {
                "Ts_Mean": ts_mean, "Ts_Sum": ts_sum, "Ts_Std": ts_std,
                "Ts_Min": ts_min, "Ts_Max": ts_max, "Ts_Median": ts_median,
                "Delay": delay, "Delta": delta, "Rank": ts_rank,
            }[op_name]
            return fn(ctx, field, win)
    elif isinstance(field_arg, ast.Name) and field_arg.id.startswith("$"):
        # 仍然支持直接 $xxx (兼容旧写法)
        field = "$" + field_arg.id[1:]
    else:
        raise _ParseError(f"{op_name} 第一个参数必须为字段或子表达式; 实际 {ast.dump(field_arg)}")
    win = _eval_const(args[1])

    fn = {
        "Ts_Mean": ts_mean, "Ts_Sum": ts_sum, "Ts_Std": ts_std,
        "Ts_Min": ts_min, "Ts_Max": ts_max, "Ts_Median": ts_median,
        "Delay": delay, "Delta": delta, "Rank": ts_rank, "Ts_Rank": ts_rank,
    }[op_name]
    return fn(ctx, field, win)


def _eval_const(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise _ParseError(f"常量必须是数字: {node.value!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_const(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_const(node.operand)
    raise _ParseError(f"不支持的常量节点: {ast.dump(node)}")


def _eval_call(node: ast.Call, ctx: FactorContext) -> pd.Series:
    if not isinstance(node.func, ast.Name):
        raise _ParseError(f"不支持的调用: {ast.dump(node.func)}")
    op_name = node.func.id
    # 特殊处理: Field("name") 是字段访问
    if op_name == "Field":
        return _eval_field_call(node, ctx)
    if op_name not in OPERATOR_REGISTRY:
        raise _ParseError(f"未知算子: {op_name}; 已注册: {sorted(OPERATOR_REGISTRY)}")
    spec = OPERATOR_REGISTRY[op_name]
    if spec.category == "ts":
        return _eval_ts_op(op_name, node.args, ctx)
    if spec.category == "math":
        x = _eval_expr(node.args[0], ctx) if node.args else None
        if op_name == "Power":
            exp = _eval_const(node.args[1])
            return pd.Series(_to_array(x) ** exp, index=x.index if isinstance(x, pd.Series) else None)
        return _apply_math(op_name, x)
    if spec.category == "logic":
        if op_name == "If":
            cond = _eval_expr(node.args[0], ctx)
            a = _eval_expr(node.args[1], ctx)
            b = _eval_expr(node.args[2], ctx)
            return _if_op(cond, a, b)
        if op_name == "And":
            return _and_op(_eval_expr(node.args[0], ctx), _eval_expr(node.args[1], ctx))
        if op_name == "Or":
            return _or_op(_eval_expr(node.args[0], ctx), _eval_expr(node.args[1], ctx))
    raise _ParseError(f"未实现的算子类别: {op_name}")


def _eval_expr(node: ast.AST, ctx: FactorContext) -> pd.Series:
    if isinstance(node, ast.Name):
        if not node.id.startswith("$"):
            raise _ParseError(f"非 $ 字段名: {node.id}")
        return ctx.get(node.id)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return pd.Series([node.value] * len(ctx.df), index=ctx.df.index)
        raise _ParseError(f"常量必须为数字: {node.value!r}")
    if isinstance(node, ast.UnaryOp):
        v = _eval_expr(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise _ParseError(f"不支持的一元运算: {ast.dump(node.op)}")
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, ctx)
        right = _eval_expr(node.right, ctx)
        l, r = _to_array(left), _to_array(right)
        if isinstance(node.op, ast.Add):
            return pd.Series(l + r, index=left.index)
        if isinstance(node.op, ast.Sub):
            return pd.Series(l - r, index=left.index)
        if isinstance(node.op, ast.Mult):
            return pd.Series(l * r, index=left.index)
        if isinstance(node.op, ast.Div):
            with np.errstate(divide="ignore", invalid="ignore"):
                return pd.Series(np.where(r != 0, l / r, np.nan), index=left.index)
        raise _ParseError(f"不支持的二元运算: {ast.dump(node.op)}")
    if isinstance(node, ast.Compare):
        # 支持 >, <, >=, <=, ==, !=
        left = _eval_expr(node.left, ctx)
        l = _to_array(left)
        out = np.ones(len(l), dtype=float)
        for op, comp_node in zip(node.ops, node.comparators):
            r = _to_array(_eval_expr(comp_node, ctx))
            if isinstance(op, ast.Gt):
                out = out * (l > r)
            elif isinstance(op, ast.Lt):
                out = out * (l < r)
            elif isinstance(op, ast.GtE):
                out = out * (l >= r)
            elif isinstance(op, ast.LtE):
                out = out * (l <= r)
            elif isinstance(op, ast.Eq):
                out = out * (l == r)
            elif isinstance(op, ast.NotEq):
                out = out * (l != r)
            else:
                raise _ParseError(f"不支持的比较: {ast.dump(op)}")
            l = r
        return pd.Series(out, index=left.index)
    if isinstance(node, ast.Call):
        return _eval_call(node, ctx)
    raise _ParseError(f"无法解析节点: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# 因子引擎主类
# ---------------------------------------------------------------------------
class FactorEngine:
    """
    因子表达式引擎 (借鉴 AKQuant.akquant.factor.FactorEngine)

    用法:
        engine = FactorEngine()
        engine.register_formula("alpha_001", "Rank(Ts_Mean($close, 5))")
        result = engine.compute(data, formulas=["alpha_001", "alpha_002"])
        # result 是 DataFrame[code, date, alpha_001, alpha_002, ...]
    """

    def __init__(self):
        self._formulas: Dict[str, str] = {}
        self._parsed: Dict[str, ast.Expression] = {}

    def register_formula(self, name: str, formula: str) -> None:
        """注册一个因子公式, 可重复注册覆盖.
        字段语法: $close, $open, $high, $low, $volume, $amount, $vwap
        算子语法: Ts_Mean($close, 5), Rank(Ts_Mean($close, 5)), If(cond, a, b) 等
        """
        # 前处理: 把 $xxx 转成 Field("xxx") 让 Python AST 能解析
        normalized = _normalize_formula(formula)
        tree = ast.parse(normalized, mode="eval")
        self._formulas[name] = formula
        self._parsed[name] = tree

    def available_operators(self) -> List[str]:
        return sorted(OPERATOR_REGISTRY.keys())

    def operator_info(self, name: str) -> Optional[OperatorSpec]:
        return OPERATOR_REGISTRY.get(name)

    def compute(
        self,
        data: pd.DataFrame,
        formulas: Optional[List[str]] = None,
        apply_cross_section: bool = True,
    ) -> pd.DataFrame:
        """
        批量计算因子

        参数:
            data: 行情数据, 必须包含 [code, date, open, high, low, close, volume]
                  以及可选的 amount, vwap, turnover, pct_chg
            formulas: 因子名列表; 为 None 时计算所有已注册公式
            apply_cross_section: 是否对 Cs_Rank / Cs_Scale / Cs_Zscore 等截面算子
                                 的输出做截面归一化 (推荐 True)

        返回:
            DataFrame[code, date, <各因子列>]
        """
        if "code" not in data.columns or "date" not in data.columns:
            raise ValueError("data 必须包含 code, date 列")
        if formulas is None:
            formulas = list(self._formulas.keys())
        for f in formulas:
            if f not in self._parsed:
                raise KeyError(f"未注册因子: {f}; 请先调用 register_formula")

        # 收集所有用到的字段
        referenced_fields: set = set()
        for fname in formulas:
            for m in _FIELD_TOKEN_RE.finditer(self._formulas[fname]):
                referenced_fields.add(m.group(1))

        # 输出容器: factor_name -> {(code, date): value}
        records: Dict[str, Dict[Any, Any]] = {f: {} for f in formulas}

        # 按股票分组求值
        groups = list(data.groupby("code", sort=False))
        for code, g in groups:
            g = g.sort_values("date").reset_index(drop=True)
            ctx = FactorContext(df=g)
            for fname in formulas:
                try:
                    val = _eval_expr(self._parsed[fname].body, ctx)
                    val.index = g["date"].values
                    for d, v in val.items():
                        records[fname][(code, d)] = v
                except Exception:
                    for d in g["date"].values:
                        records[fname][(code, d)] = np.nan

        # 拼装: 直接构造 MultiIndex Series, 然后 unstack 得到 wide
        result = data[["code", "date"]].copy().reset_index(drop=True)
        for fname in formulas:
            if not records[fname]:
                result[fname] = np.nan
                continue
            s = pd.Series(records[fname], name=fname)
            s.index = pd.MultiIndex.from_tuples(s.index, names=["code", "date"])
            if apply_cross_section:
                s = cs_zscore(s)
            wide = s.unstack("code")  # date x code

            # 用 stack 回 long 格式, 索引 (date, code), 再按 (code, date) 排序
            long_df = wide.stack(future_stack=True).reset_index()
            long_df.columns = ["date", "code", fname]
            long_df = long_df[["code", "date", fname]]

            # 对齐到 result 顺序
            lookup = long_df.set_index(["code", "date"])[fname]
            result_idx = list(zip(result["code"], result["date"]))
            result[fname] = [lookup.get(k, np.nan) for k in result_idx]
        return result

    # ----- 截面算子便捷入口 -----
    @staticmethod
    def cs_rank(series: pd.Series) -> pd.Series:
        return cs_rank(series)

    @staticmethod
    def cs_scale(series: pd.Series) -> pd.Series:
        return cs_scale(series)

    @staticmethod
    def cs_zscore(series: pd.Series) -> pd.Series:
        return cs_zscore(series)


__all__ = [
    "FactorEngine", "OPERATOR_REGISTRY", "OperatorSpec",
    "VALID_FIELDS", "cs_rank", "cs_scale", "cs_zscore",
]