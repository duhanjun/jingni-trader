"""
方向 1：声明式因子表达式引擎（Qlib 风格）

借鉴：Microsoft qlib 的 Expression Engine
      - 因子用表达式声明：$close, Ref($close, 5), Mean($close, 20), $high - $low
      - 编译为 AST → 在面板数据上批量执行
      - 自动构建因子依赖图
      - 复用 panel 计算 (Ref/Mean) 而不重复算

目标：
- 让研究员无需写 Python 代码即可定义 / 组合因子
- 支持 Qlib 风格语法并保持 pandas 兼容
- 内置 Alpha158 子集（量价类核心因子）作为开箱即用的库
- 记录每个因子的依赖图，便于追溯和缓存
"""
from __future__ import annotations
import ast
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data_utils import ensure_panel, pivot_field


def _to_matrix(value, panel: pd.DataFrame):
    """将标量提升为 (date x code) 矩阵，便于和面板数据一起 concat/算术运算"""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value
    if np.isscalar(value):
        # 构造与面板同形状的常量矩阵（用 pivot 模板对齐 index/columns）
        template = pivot_field(panel, "close")
        return pd.DataFrame(value, index=template.index, columns=template.columns)
    raise TypeError(f"无法提升为矩阵: {type(value).__name__}")


# ---------------------------------------------------------------------------
# 1) AST 节点：把表达式编译为可执行的操作树
# ---------------------------------------------------------------------------
class ExprNode:
    """所有表达式节点的基类"""

    def evaluate(self, ctx: Dict[str, pd.DataFrame], cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

    def dependencies(self) -> List[str]:
        return []

    def __repr__(self) -> str:
        return self.__class__.__name__


class FieldNode(ExprNode):
    """$close / $open / $high / $low / $volume / $amount 等原始字段"""

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, ctx, cache):
        key = f"field:{self.name}"
        if key not in cache:
            cache[key] = pivot_field(ctx["panel"], self.name.lstrip("$"))
        return cache[key]

    def dependencies(self):
        return [self.name]

    def __repr__(self):
        return self.name


class RefNode(ExprNode):
    """Ref(field, n) → 字段在 n 期前的值（panel shift）"""

    def __init__(self, child: ExprNode, n: int):
        self.child = child
        self.n = n

    def evaluate(self, ctx, cache):
        key = f"ref:{self.n}:{id(self.child)}:{repr(self.child)}"
        if key not in cache:
            cache[key] = self.child.evaluate(ctx, cache).shift(self.n)
        return cache[key]

    def dependencies(self):
        return self.child.dependencies()

    def __repr__(self):
        return f"Ref({self.child}, {self.n})"


class RollingNode(ExprNode):
    """Rolling{Mean/Std/Sum/Min/Max/Median}(field, window) → 按时间窗聚合"""

    SUPPORTED = {"Mean", "Std", "Sum", "Min", "Max", "Median", "EMA"}

    def __init__(self, op: str, child: ExprNode, window: int, min_periods: Optional[int] = None):
        assert op in self.SUPPORTED, f"不支持的滚动算子: {op}"
        self.op = op
        self.child = child
        self.window = window
        self.min_periods = min_periods or max(1, window // 2)

    def evaluate(self, ctx, cache):
        key = f"roll:{self.op}:{self.window}:{id(self.child)}:{repr(self.child)}"
        if key not in cache:
            data = self.child.evaluate(ctx, cache)
            agg = {
                "Mean": lambda x: x.rolling(self.window, min_periods=self.min_periods).mean(),
                "Std": lambda x: x.rolling(self.window, min_periods=self.min_periods).std(),
                "Sum": lambda x: x.rolling(self.window, min_periods=self.min_periods).sum(),
                "Min": lambda x: x.rolling(self.window, min_periods=self.min_periods).min(),
                "Max": lambda x: x.rolling(self.window, min_periods=self.min_periods).max(),
                "Median": lambda x: x.rolling(self.window, min_periods=self.min_periods).median(),
                "EMA": lambda x: x.ewm(span=self.window, min_periods=self.min_periods, adjust=False).mean(),
            }[self.op]
            # 关闭时按 code 单独 rolling，但因是 pivot 表，rolling 默认按行（time）
            cache[key] = agg(data)
        return cache[key]

    def dependencies(self):
        return self.child.dependencies()

    def __repr__(self):
        return f"{self.op}({self.child}, {self.window})"


class DeltaNode(ExprNode):
    """Delta(field, n) = field - Ref(field, n)"""

    def __init__(self, child: ExprNode, n: int):
        self.child = child
        self.n = n

    def evaluate(self, ctx, cache):
        key = f"delta:{self.n}:{id(self.child)}:{repr(self.child)}"
        if key not in cache:
            cur = self.child.evaluate(ctx, cache)
            ref = RefNode(self.child, self.n).evaluate(ctx, cache)
            cache[key] = cur - ref
        return cache[key]

    def dependencies(self):
        return self.child.dependencies()

    def __repr__(self):
        return f"Delta({self.child}, {self.n})"


class BinaryOpNode(ExprNode):
    SUPPORTED = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/"}

    def __init__(self, op: str, left: ExprNode, right: ExprNode):
        assert op in self.SUPPORTED
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, ctx, cache):
        key = f"bin:{self.op}:{id(self.left)}:{id(self.right)}"
        if key not in cache:
            l = self.left.evaluate(ctx, cache)
            r = self.right.evaluate(ctx, cache)
            if self.op == "Add":
                cache[key] = l + r
            elif self.op == "Sub":
                cache[key] = l - r
            elif self.op == "Mult":
                cache[key] = l * r
            elif self.op == "Div":
                cache[key] = l / r.replace(0, np.nan)
        return cache[key]

    def dependencies(self):
        return self.left.dependencies() + self.right.dependencies()

    def __repr__(self):
        sym = self.SUPPORTED[self.op]
        return f"({self.left} {sym} {self.right})"


class AbsNode(ExprNode):
    def __init__(self, child: ExprNode):
        self.child = child

    def evaluate(self, ctx, cache):
        return self.child.evaluate(ctx, cache).abs()

    def dependencies(self):
        return self.child.dependencies()

    def __repr__(self):
        return f"Abs({self.child})"


class FuncNode(ExprNode):
    """通用数值函数节点：Max(a, b), Min(a, b), Sign(x), Log(x), 等"""

    SUPPORTED = {"Max", "Min", "Sign", "Log", "Sqrt", "Pow"}

    def __init__(self, op: str, args: List[ExprNode]):
        assert op in self.SUPPORTED, f"不支持的函数: {op}"
        self.op = op
        self.args = args

    def evaluate(self, ctx, cache):
        key = f"fn:{self.op}:{':'.join(id(a).__repr__() for a in self.args)}"
        if key not in cache:
            evals = [a.evaluate(ctx, cache) for a in self.args]
            # 处理包含标量的场景：将 scalar 提升为 DataFrame
            evals = [_to_matrix(e, ctx["panel"]) for e in evals]
            if self.op == "Max":
                cache[key] = pd.concat(evals, axis=1).max(axis=1)
            elif self.op == "Min":
                cache[key] = pd.concat(evals, axis=1).min(axis=1)
            elif self.op == "Sign":
                cache[key] = np.sign(evals[0])
            elif self.op == "Log":
                cache[key] = np.log(np.abs(evals[0]) + 1e-9) * np.sign(evals[0])
            elif self.op == "Sqrt":
                cache[key] = np.sqrt(np.abs(evals[0]))
            elif self.op == "Pow":
                cache[key] = evals[0] ** evals[1]
        return cache[key]

    def dependencies(self):
        deps = []
        for a in self.args:
            deps.extend(a.dependencies())
        return deps

    def __repr__(self):
        return f"{self.op}({', '.join(repr(a) for a in self.args)})"


# ---------------------------------------------------------------------------
# 2) 表达式解析器：把字符串编译成 AST
# ---------------------------------------------------------------------------
_FIELD_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z_0-9]*)")


def _tokenize(expr: str) -> List[str]:
    """把 "$close + Mean($volume, 5)" 切成 [$close, +, Mean, (, $volume, ,, 5, )]"""
    tokens: List[str] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-*/(),":
            tokens.append(c)
            i += 1
            continue
        m = _FIELD_RE.match(expr, i)
        if m:
            tokens.append(f"${m.group(1)}")
            i = m.end()
            continue
        # 函数名 / 数字
        j = i
        while j < len(expr) and (expr[j].isalnum() or expr[j] == "_" or expr[j] == "."):
            j += 1
        if j == i:
            raise ValueError(f"无法解析: {expr[i:]}")
        tokens.append(expr[i:j])
        i = j
    return tokens


_PY_OP_MAP = {
    ast.Add: "Add",
    ast.Sub: "Sub",
    ast.Mult: "Mult",
    ast.Div: "Div",
}


def _build_node(tree: ast.AST) -> ExprNode:
    if isinstance(tree, ast.Expression):
        return _build_node(tree.body)

    if isinstance(tree, ast.BinOp):
        op = _PY_OP_MAP.get(type(tree.op))
        if op is None:
            raise ValueError(f"不支持的二元运算符: {type(tree.op).__name__}")
        return BinaryOpNode(op, _build_node(tree.left), _build_node(tree.right))

    if isinstance(tree, ast.UnaryOp) and isinstance(tree.op, ast.USub):
        # -x 等价于 (0 - x)
        zero_node = _ConstNode(0.0)
        return BinaryOpNode("Sub", zero_node, _build_node(tree.operand))

    if isinstance(tree, ast.Name):
        name = tree.id
        if not name.startswith("$"):
            name = f"${name}"
        return FieldNode(name)

    if isinstance(tree, ast.Constant):
        return _ConstNode(tree.value)

    if isinstance(tree, ast.Call):
        if not isinstance(tree.func, ast.Name):
            raise ValueError("仅支持简单函数调用")
        fn = tree.func.id
        return _build_call(fn, tree.args)

    raise ValueError(f"无法解析的语法树节点: {type(tree).__name__}")


def _build_call(fn: str, args) -> "ExprNode":
    import ast as _ast
    def _is_number(arg):
        if isinstance(arg, _ast.Constant) and isinstance(arg.value, (int, float)):
            return True
        if hasattr(arg, "n") and isinstance(getattr(arg, "n"), (int, float)):
            return True
        return False

    def _num(arg):
        if isinstance(arg, _ast.Constant):
            return arg.value
        if hasattr(arg, "n"):
            return arg.n
        raise ValueError(f"参数类型不支持: {type(arg).__name__}")

    # 1) Ref / Delta: 第二参必须是数字
    if fn in ("Ref", "Delta"):
        if len(args) != 2:
            raise ValueError(f"{fn} 需要 2 个参数")
        child = _build_node(args[0])
        n = int(_num(args[1]))
        return RefNode(child, n) if fn == "Ref" else DeltaNode(child, n)

    # 2) Rolling family (Mean/Std/Sum/Median/EMA): 第二参必须是数字窗口
    if fn in RollingNode.SUPPORTED and fn not in {"Max", "Min"}:
        if len(args) != 2:
            raise ValueError(f"{fn} 需要 2 个参数")
        child = _build_node(args[0])
        w = int(_num(args[1]))
        return RollingNode(fn, child, w)

    # 3) Abs
    if fn == "Abs":
        if len(args) != 1:
            raise ValueError("Abs 需要 1 个参数")
        return AbsNode(_build_node(args[0]))

    # 4) Max / Min: 第二参为正数 → 窗口；为 0/表达式 → 元素级 (元素级更常见)
    if fn in {"Max", "Min"}:
        if len(args) != 2:
            raise ValueError(f"{fn} 需要 2 个参数")
        if _is_number(args[1]) and int(_num(args[1])) >= 1:
            # 窗口 Max/Min (窗口必须 >= 1)
            child = _build_node(args[0])
            w = int(_num(args[1]))
            return RollingNode(fn, child, w)
        # 元素级 Max/Min (默认分支，包括 0 / 表达式 第二参)
        return FuncNode(fn, [_build_node(a) for a in args])

    # 5) 其它通用函数 (Sign/Log/Sqrt/Pow)
    if fn in FuncNode.SUPPORTED:
        return FuncNode(fn, [_build_node(a) for a in args])

    raise ValueError(f"不支持的函数: {fn}")


class _ConstNode(ExprNode):
    """标量常数（不会出现在缓存键中）"""

    def __init__(self, value: float):
        self.value = float(value)

    def evaluate(self, ctx, cache):
        return self.value

    def dependencies(self):
        return []

    def __repr__(self):
        return f"{self.value:g}"


def parse(expr: str) -> ExprNode:
    """把 Qlib 风格表达式解析为 ExprNode"""
    tokens_str = " ".join(_tokenize(expr))
    # 把 $close 转成 Python 标识符 close，以便 ast.parse
    py_str = _FIELD_RE.sub(lambda m: m.group(1), tokens_str)
    try:
        tree = ast.parse(py_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误: {expr}\n  →  {e}")
    return _build_node(tree)


# ---------------------------------------------------------------------------
# 3) 因子计算引擎
# ---------------------------------------------------------------------------
@dataclass
class FactorSpec:
    name: str
    expr: str
    description: str = ""
    node: Optional[ExprNode] = None
    dependencies: List[str] = field(default_factory=list)


class FactorEngine:
    """声明式因子计算引擎"""

    def __init__(self, panel: pd.DataFrame):
        self.panel = ensure_panel(panel)
        self._specs: Dict[str, FactorSpec] = {}

    def register(self, name: str, expr: str, description: str = "") -> FactorSpec:
        node = parse(expr)
        deps = sorted(set(node.dependencies()))
        spec = FactorSpec(name=name, expr=expr, description=description,
                          node=node, dependencies=deps)
        self._specs[name] = spec
        return spec

    def compute(self, name: str) -> pd.DataFrame:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"因子 {name} 未注册")
        ctx = {"panel": self.panel}
        cache: Dict[str, pd.DataFrame] = {}
        result = spec.node.evaluate(ctx, cache)
        return result  # (date x code) matrix

    def compute_all(self) -> pd.DataFrame:
        """把所有注册的因子拼成长表 DataFrame[date, code, factor_name...]
        使用 outer join 保持所有 (date, code) 组合，并对齐到面板索引
        """
        if not self._specs:
            return pd.DataFrame(columns=["date", "code"])

        # 1) 直接从 panel 构造完整 (date, code) MultiIndex
        full_index = pd.MultiIndex.from_frame(
            self.panel[["date", "code"]], names=["date", "code"]
        )

        # panel 允许的 codes (用于截断)
        valid_codes = list(self.panel["code"].unique())

        # 2) 每个因子 → 对齐到 full_index 的 Series
        aligned: Dict[str, pd.Series] = {}
        for name in self._specs:
            mat = self.compute(name)
            if isinstance(mat, pd.DataFrame):
                # 只保留 panel 中出现过的 code（避免某些因子在 Max/Min 中引入多余列）
                cols = [c for c in mat.columns if c in valid_codes]
                mat = mat[cols] if cols else mat.iloc[:, :0]
                ser = mat.stack(future_stack=True)
                if len(ser) == len(full_index):
                    ser.index = full_index
                else:
                    # 长度不匹配时重设索引（reindex 引入 NaN）
                    ser = ser.reset_index(drop=True)
                    ser.index = full_index[:len(ser)] if len(ser) <= len(full_index) else full_index
                aligned[name] = ser.rename(name)
            else:
                aligned[name] = mat.rename(name).reindex(full_index)

        # 3) 拼成 DataFrame
        out = pd.concat(aligned.values(), axis=1)
        out.columns = list(aligned.keys())
        return out.reset_index()

    def dependency_graph(self) -> Dict[str, List[str]]:
        return {n: s.dependencies for n, s in self._specs.items()}


# ---------------------------------------------------------------------------
# 4) 内置 Alpha158 子集（量价类核心因子，可直接 register）
#    来自 qlib.contrib.data.handler.Alpha158
# ---------------------------------------------------------------------------
ALPHA158_PV_SUBSET: Dict[str, str] = {
    # 基础动量 / 反转
    "KMID":      "($close - $open) / $open",
    "KLEN":      "($high - $low) / $open",
    "KUP":       "($high - Max($open, Ref($close,1))) / $open",
    "KLOW":      "(Min($open, Ref($close,1)) - $low) / $open",
    "OPEN0":     "$open / $close",
    "HIGH0":     "$high / $close",
    "LOW0":      "$low / $close",

    # 简单价格 / 成交量 rolling
    "MA5":       "Mean($close, 5) / $close",
    "MA10":      "Mean($close, 10) / $close",
    "MA20":      "Mean($close, 20) / $close",
    "MA60":      "Mean($close, 60) / $close",
    "STD5":      "Std($close, 5) / $close",
    "STD20":     "Std($close, 20) / $close",
    "VOL_MA5":   "Mean($volume, 5)",
    "VOL_MA20":  "Mean($volume, 20)",

    # 动量 / 收益率
    "ROC5":      "$close / Ref($close, 5) - 1",
    "ROC10":     "$close / Ref($close, 10) - 1",
    "ROC20":     "$close / Ref($close, 20) - 1",
    "ROC60":     "$close / Ref($close, 60) - 1",

    # 量价配合
    "VROC5":     "$volume / Mean($volume, 5) - 1",
    "PV_CORR5":  "Mean(($close - Ref($close,1)) * ($volume - Ref($volume,1)), 5)",  # 简化版

    # 波动率
    "ATR14":     "Mean(Max($high - $low, Max(Abs($high - Ref($close,1)), Abs($low - Ref($close,1)))), 14)",
    "RSI14_RISE":"Mean(Max($close - Ref($close,1), 0), 14) / (Mean(Abs($close - Ref($close,1)), 14) + 0.000000001)",
}


def register_alpha158_pv(engine: FactorEngine) -> List[FactorSpec]:
    """把 Alpha158 的量价子集注册到引擎里"""
    specs = []
    for name, expr in ALPHA158_PV_SUBSET.items():
        try:
            specs.append(engine.register(name, expr, description=f"Alpha158 子集: {name}"))
        except Exception as e:
            print(f"[warn] Alpha158 因子 {name} 注册失败: {e}")
    return specs
