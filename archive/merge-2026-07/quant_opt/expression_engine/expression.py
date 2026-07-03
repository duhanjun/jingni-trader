"""
借鉴来源: Microsoft Qlib Expression Engine
- 官方仓库: https://github.com/microsoft/qlib
- 核心模块: qlib/data/ops.py
- 设计文档: https://qlib.readthedocs.io/en/latest/component/data.html

jingni-trader 现状:
  factor-engine/engine.py 中 compute_a_share_factors() 把 14 个 A 股
  Alpha 因子硬编码在 if/else 与 groupby().rolling() 中,无法在不修改
  源代码的前提下增加 / 组合新因子。

借鉴方案:
  提供一个 Qlib 风格的"表达式字符串"接口,例如
      "Mean($close, 5) / $close"
      "Rank(($close - Ref($close, 1)) / Ref($close, 1))"
  用户无需改任何 Python 代码,只需传入表达式即可注册新因子。

本文件实现一个最小可运行的版本,覆盖 Qlib 80% 常用场景:
  - ElemOperator (一元算子): Abs, Log, Sign, Rank
  - PairOperator (二元算子): Add, Sub, Mul, Div
  - Rolling    (滚动窗口算子): Mean, Sum, Std, Max, Min, Ref, Slope
  - Alpha158 技术面因子子集 (23 个, 一行代码注册)

设计要点:
  1) 解析阶段: 字符串 → AST → Operator 树 (FieldRef/Literal 叶子 + 内部节点)
  2) 求值阶段: ExpressionEngine 把 Operator 树递归解析为 Series
  3) Rolling 算子的"输入"可以是字段名 (string) 或任意子表达式
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ===========================================================================
# 1. Operator 基类
# ===========================================================================
class ExpressionOps:
    """所有表达式的根类。Qlib 中定义在 qlib.data.ops.ExpressionOps。"""

    def load(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ===========================================================================
# 2. 叶子节点: FieldRef / Literal
# ===========================================================================
@dataclass
class FieldRef(ExpressionOps):
    name: str

    def load(self, *args, **kwargs):
        raise RuntimeError("FieldRef 应当由 ExpressionEngine 求值,而不是直接 load")

    def __repr__(self):
        return f"${self.name}"


@dataclass
class Literal(ExpressionOps):
    value: float

    def load(self, *args, **kwargs):
        if args and isinstance(args[0], pd.Series):
            return pd.Series(self.value, index=args[0].index)
        return self.value

    def __repr__(self):
        return str(self.value)


# ===========================================================================
# 3. 一元 / 二元 / 滚动 三类基础算子
# ===========================================================================
class ElemOperator(ExpressionOps):
    """逐元素一元算子,Qlib 中所有 _ElementOperator 的父类。"""

    def __init__(self, feature: Union[str, ExpressionOps]):
        self.feature = feature

    def load(self, *args, **kwargs) -> pd.Series:
        if len(args) != 1:
            raise ValueError(f"{type(self).__name__} 需要 1 个参数,实际 {len(args)}")
        return self._apply(args[0])

    def _apply(self, s: pd.Series) -> pd.Series:
        raise NotImplementedError


class PairOperator(ExpressionOps):
    """二元算子,Qlib 中所有 _PairOperator 的父类。"""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def load(self, *args, **kwargs) -> pd.Series:
        if len(args) != 2:
            raise ValueError(f"{type(self).__name__} 需要 2 个参数,实际 {len(args)}")
        return self._apply(args[0], args[1])

    def _apply(self, a: pd.Series, b) -> pd.Series:
        raise NotImplementedError


class Rolling(ExpressionOps):
    """滚动窗口算子,Qlib 中所有 _Rolling 结尾的类都继承此类。"""

    def __init__(self, feature, window: int):
        # feature 可以是 str (字段名) 或 ExpressionOps (子表达式)
        self.feature = feature
        self.window = int(window)

    def load(self, *args, **kwargs) -> pd.Series:
        s = args[0]
        if not isinstance(s, pd.Series):
            raise TypeError("Rolling 算子的输入必须是 Series")
        if isinstance(s.index, pd.MultiIndex) and "code" in s.index.names:
            grouped = s.groupby(level="code")
        else:
            grouped = s
        out = self._apply(grouped)
        # pandas 的 groupby+rolling 会在结果索引中保留一个额外的 group key 层级
        # 我们要 drop 掉这个 group key,重新对齐到原始的 (date, code) 索引
        if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > s.index.nlevels:
            # 用 droplevel 去掉 group key 那一层
            out = out.droplevel(0)
        # 重新对齐到 s.index (处理缺失值)
        if isinstance(s.index, pd.MultiIndex):
            out = out.reorder_levels(s.index.names).reindex(s.index)
        return out

    def _apply(self, grouped) -> pd.Series:
        raise NotImplementedError


# ===========================================================================
# 4. 具体算子实现
# ===========================================================================
class Abs(ElemOperator):
    def _apply(self, s): return s.abs()


class Log(ElemOperator):
    def _apply(self, s): return np.log(s.replace(0, np.nan))


class Sign(ElemOperator):
    def _apply(self, s): return np.sign(s)


class Rank(ElemOperator):
    """横截面 rank,按日期做 percent rank。Qlib: qlib.data.ops.Rank。"""

    def _apply(self, s):
        if isinstance(s.index, pd.MultiIndex) and "date" in s.index.names:
            return s.groupby(level="date").rank(pct=True)
        return s.rank(pct=True)


class Add(PairOperator):
    def _apply(self, a, b): return a + b


class Sub(PairOperator):
    def _apply(self, a, b): return a - b


class Mul(PairOperator):
    def _apply(self, a, b): return a * b


class Div(PairOperator):
    def _apply(self, a, b): return a / b.replace(0, np.nan)


class MaxE(PairOperator):
    """MaxE($a, $b) = 逐元素 max,对应 Qlib 公式中的 Max(open, close) 用法。"""
    def _apply(self, a, b):
        a_vals = a.values if isinstance(a, pd.Series) else np.asarray(a)
        b_vals = b.values if isinstance(b, pd.Series) else np.asarray(b)
        out = np.fmax(a_vals, b_vals)
        return pd.Series(out, index=a.index)


class MinE(PairOperator):
    """MinE($a, $b) = 逐元素 min,对应 Qlib 公式中的 Min(open, close) 用法。"""
    def _apply(self, a, b):
        a_vals = a.values if isinstance(a, pd.Series) else np.asarray(a)
        b_vals = b.values if isinstance(b, pd.Series) else np.asarray(b)
        out = np.fmin(a_vals, b_vals)
        return pd.Series(out, index=a.index)


class Ref(Rolling):
    """Ref($close, 1) = 前 1 期 close,Qlib 语义完全一致。"""

    def _apply(self, grouped): return grouped.shift(self.window)


class Mean(Rolling):
    """Mean($close, 5) = 5 日均值。"""

    def _apply(self, grouped):
        return grouped.rolling(self.window, min_periods=max(2, self.window // 2)).mean()


class Sum(Rolling):
    def _apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).sum()


class Std(Rolling):
    def _apply(self, grouped):
        return grouped.rolling(self.window, min_periods=max(2, self.window // 2)).std()


class Max(Rolling):
    def _apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).max()


class Min(Rolling):
    def _apply(self, grouped):
        return grouped.rolling(self.window, min_periods=1).min()


class Slope(Rolling):
    """
    Slope($close, N) = 过去 N 期 close 对时间的 OLS 斜率,Qlib Alpha158 核心。
    用滑动窗口 + 向量化协方差公式实现,O(N) 而非 O(N*W)。
    """

    def _apply(self, grouped):
        w = self.window
        x = np.arange(w, dtype=float)
        x_mean = x.mean()
        x_dev2 = ((x - x_mean) ** 2).sum()
        out_values = np.full(len(grouped.obj), np.nan, dtype=float)
        # grouped.obj 是按原顺序的 Series,grouped.groups 是 {key: positions}
        # 但更稳的做法: 遍历每个 key + 实际位置
        for key, sub in grouped:
            arr = sub.values.astype(float)
            n = len(arr)
            if n < w:
                continue
            kernel = np.lib.stride_tricks.sliding_window_view(arr, w)
            y_mean = kernel.mean(axis=1)
            cov = ((kernel - y_mean[:, None]) * (x - x_mean)).sum(axis=1) / x_dev2
            # 找到 sub 在 grouped.obj 中的位置
            positions = grouped.indices[key]
            out_values[positions[w - 1:positions[-1] + 1]] = cov
        out = pd.Series(out_values, index=grouped.obj.index)
        return out


# ===========================================================================
# 5. 表达式解析器 (支持 $field 与函数调用)
# ===========================================================================
class ExpressionParser:
    """
    支持语法 (Qlib 风格子集):
        $field            -> FieldRef 叶子
        Func($field, N)   -> 一元函数 / 滚动函数
        Func($a, $b)      -> 二元函数
        a + b / a - b     -> 二元运算
    """

    FUNC_MAP: Dict[str, type] = {
        "Abs": Abs, "Log": Log, "Sign": Sign, "Rank": Rank,
        "Add": Add, "Sub": Sub, "Mul": Mul, "Div": Div,
        "MaxE": MaxE, "MinE": MinE,
        "Ref": Ref, "Mean": Mean, "Sum": Sum, "Std": Std,
        "Max": Max, "Min": Min, "Slope": Slope,
    }

    @classmethod
    def parse(cls, expr: str) -> ExpressionOps:
        field_pattern = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")
        fields: List[str] = field_pattern.findall(expr)
        placeholder_expr = field_pattern.sub(r"__fld_\1__", expr)
        tree = ast.parse(placeholder_expr, mode="eval")
        op = cls._build(tree.body)
        op._expr_string = expr  # type: ignore[attr-defined]
        op._fields = fields      # type: ignore[attr-defined]
        return op

    @classmethod
    def _build(cls, node) -> ExpressionOps:
        if isinstance(node, ast.Name):
            if node.id.startswith("__fld_") and node.id.endswith("__"):
                return FieldRef(node.id[6:-2])
            raise ValueError(f"未识别标识符: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("只支持具名函数调用")
            func_name = node.func.id
            if func_name not in cls.FUNC_MAP:
                raise ValueError(f"不支持的算子: {func_name}")
            cls_ = cls.FUNC_MAP[func_name]
            args = [cls._build(a) for a in node.args]

            if issubclass(cls_, ElemOperator):
                # 一元算子可以接受字段名 (string) 或任意子表达式
                if len(args) != 1:
                    raise ValueError(f"{func_name} 需要 1 个参数")
                if isinstance(args[0], FieldRef):
                    return cls_(args[0].name)
                # 子表达式: 直接传 Operator 树,稍后由 ExpressionEngine 求值
                return cls_(args[0])

            if issubclass(cls_, PairOperator):
                if len(args) != 2:
                    raise ValueError(f"{func_name} 需要 2 个参数")
                return cls_(args[0], args[1])

            if issubclass(cls_, Rolling):
                if len(args) != 2:
                    raise ValueError(f"{func_name} 需要 (feature, window)")
                # 第二个参数必须是整数常量
                window_val = cls._extract_const(args[1])
                if window_val is None:
                    raise ValueError(f"{func_name} 的第二个参数必须是整数常量")
                # 第一个参数可以是字段名 (string) 或任意子表达式
                if isinstance(args[0], FieldRef):
                    return cls_(args[0].name, window_val)
                # 子表达式: 直接传 Operator 树,稍后由 ExpressionEngine 求值
                return cls_(args[0], window_val)
        if isinstance(node, ast.BinOp):
            op_map = {ast.Add: Add, ast.Sub: Sub, ast.Mult: Mul, ast.Div: Div}
            cls_ = op_map.get(type(node.op))
            if cls_ is None:
                raise ValueError(f"不支持二元运算: {type(node.op).__name__}")
            return cls_(cls._build(node.left), cls._build(node.right))
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(f"不支持字面量: {node.value!r}")
            return Literal(node.value)
        raise ValueError(f"无法解析的节点: {ast.dump(node)}")

    @staticmethod
    def _extract_const(op: ExpressionOps) -> Optional[int]:
        if isinstance(op, Literal):
            return int(op.value)
        if isinstance(op, ast.Constant) and isinstance(op.value, (int, float)):
            return int(op.value)
        return None


# ===========================================================================
# 6. 表达式求值器
# ===========================================================================
class ExpressionEngine:
    """
    表达式求值器。把解析后的 Operator 树应用到 DataFrame 上。

    使用示例::

        engine = ExpressionEngine(data_df)   # data_df 含 close, volume ...
        result = engine.eval("Mean($close, 5) / Ref($close, 1) - 1")
    """

    def __init__(self, data: pd.DataFrame):
        if "date" not in data.columns or "code" not in data.columns:
            raise ValueError("data 必须包含 date, code 列")
        self.data = data
        self._indexed = data.set_index(["date", "code"]).sort_index()

    def eval(self, expr: str) -> pd.Series:
        op = ExpressionParser.parse(expr)
        return self._eval_op(op)

    def _eval_op(self, op: ExpressionOps) -> pd.Series:
        # 1) 叶子节点直接求值
        if isinstance(op, FieldRef):
            if op.name not in self._indexed.columns:
                raise KeyError(f"字段 ${op.name} 不在数据中,可选: {list(self._indexed.columns)}")
            return self._indexed[op.name]
        if isinstance(op, Literal):
            return pd.Series(op.value, index=self._indexed.index)

        # 2) 一元算子: 先解析内部,再调用 load
        if isinstance(op, ElemOperator):
            if isinstance(op.feature, ExpressionOps):
                s = self._eval_op(op.feature)
            else:
                # str 字段名
                s = self._indexed[op.feature]
            return op.load(s)

        # 3) 二元算子: 解析两个子项
        if isinstance(op, PairOperator):
            sa = self._resolve(op.a)
            sb = self._resolve(op.b)
            return op.load(sa, sb)

        # 4) 滚动算子: 先解析 feature (可能字段名也可能子表达式)
        if isinstance(op, Rolling):
            if isinstance(op.feature, ExpressionOps):
                s = self._eval_op(op.feature)
            else:
                s = self._indexed[op.feature]
            return op.load(s)

        raise ValueError(f"未知算子类型: {type(op)}")

    def _resolve(self, op):
        if isinstance(op, ExpressionOps):
            return self._eval_op(op)
        return op


# ===========================================================================
# 7. 工厂函数 —— 一行代码注册因子
# ===========================================================================
def register_factor(data: pd.DataFrame, expr: str, name: Optional[str] = None) -> pd.Series:
    """
    一行代码注册 + 计算 A 股因子 (Qlib 风格)。

    参数:
        data:  行情数据,需要包含 date, code 列
        expr:  Qlib 风格表达式字符串
        name:  因子名称 (默认使用表达式)

    返回:
        MultiIndex (date, code) 上的因子值 Series
    """
    engine = ExpressionEngine(data)
    result = engine.eval(expr)
    result.name = name or expr
    return result


# ===========================================================================
# 8. Alpha158 技术面因子子集 (23 个, 来源: Qlib Alpha158 论文)
# ===========================================================================
ALPHA158_TECHNICAL_SUBSET = {
    "KMID":    "($close - $open) / $open",
    "KLEN":    "($high - $low) / $open",
    "KUP":     "($high - MaxE($open, $close)) / $open",
    "KLOW":    "($close - MinE($open, $close)) / $open",
    "ROC5":    "Ref($close, 5) / $close - 1",
    "ROC10":   "Ref($close, 10) / $close - 1",
    "ROC20":   "Ref($close, 20) / $close - 1",
    "ROC60":   "Ref($close, 60) / $close - 1",
    "MA5":     "Mean($close, 5) / $close",
    "MA10":    "Mean($close, 10) / $close",
    "MA20":    "Mean($close, 20) / $close",
    "MA60":    "Mean($close, 60) / $close",
    "STD5":    "Std($close, 5) / $close",
    "STD20":   "Std($close, 20) / $close",
    "STD60":   "Std($close, 60) / $close",
    "BETA20":  "Slope($close, 20)",
    "BETA60":  "Slope($close, 60)",
    "TSMAX20": "Max($close, 20) / $close - 1",
    "TSMIN20": "$close / Min($close, 20) - 1",
    "VMA5":    "Mean($volume, 5) / ($volume + 1e-12)",
    "VSTD20":  "Std($volume, 20) / (Mean($volume, 20) + 1e-12)",
    "VSUMP":   "Sum(MaxE($volume - Ref($volume, 1), 0), 10) / (Sum(Abs($volume - Ref($volume, 1)), 10) + 1e-12)",
    "WVMA20":  "Std(Abs($close / Ref($close, 1) - 1) * $volume, 20)",
}


def compute_alpha158_subset(data: pd.DataFrame) -> pd.DataFrame:
    """批量计算 Alpha158 技术面子集,返回 (date, code, factor_name) 透视结果。"""
    engine = ExpressionEngine(data)
    results = {}
    for name, expr in ALPHA158_TECHNICAL_SUBSET.items():
        s = engine.eval(expr)
        s.name = name
        results[name] = s
    df = pd.concat(results.values(), axis=1)
    df.index.set_names(["date", "code"], inplace=True)
    return df.reset_index()
