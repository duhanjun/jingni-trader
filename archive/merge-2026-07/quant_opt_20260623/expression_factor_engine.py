"""
表达式因子引擎验证模块
借鉴 Microsoft Qlib 的表达式引擎 + 多级缓存机制

核心改进点（对照 jingni-trader 现有 pandas_ta_calculator.py）：
1. 表达式引擎：支持 `Ref($close, -1) / $close - 1` 风格的字符串公式定义因子
   - 现有实现：if/elif 硬编码 18 个因子，新增因子需改源码
   - 新实现：表达式解析 + 算子注册，用户一行字符串即可定义因子
2. 因子缓存：按 (表达式, 数据指纹) 缓存计算结果，避免重复计算
   - 现有实现：每次运行全量重算
   - 新实现：LRU + 指纹缓存，二次计算 O(1)
3. 算子注册表：可扩展的自定义算子（Ref, Mean, Std, Rank, Corr 等）
4. PIT 安全：Ref/Shift 严格使用历史数据，杜绝前视偏差

借鉴来源：
- Qlib Expression Engine: https://qlib.readthedocs.io/en/latest/component/data.html
  - D.features(fields=["Ref($close, 60) / $close"]) 表达式 API
  - 多级缓存：DatasetCache / DiskCache / MemCache
  - Alpha158 / Alpha360 因子库
- Qlib ops.py: https://github.com/microsoft/qlib/blob/main/qlib/data/ops.py
"""
from __future__ import annotations
import hashlib
import re
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ============================================================
# 算子注册表：借鉴 Qlib ops.py 的 Operator 体系
# ============================================================

class OperatorRegistry:
    """算子注册表，支持扩展自定义算子"""

    def __init__(self):
        self._ops: Dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(func: Callable):
            self._ops[name] = func
            return func
        return decorator

    def get(self, name: str) -> Optional[Callable]:
        return self._ops.get(name)

    def list_ops(self) -> List[str]:
        return sorted(self._ops.keys())


# 全局默认算子注册表
DEFAULT_REGISTRY = OperatorRegistry()


# ---- 内置算子（借鉴 Qlib）----
@DEFAULT_REGISTRY.register("Ref")
def op_ref(series: pd.Series, n: int) -> pd.Series:
    """引用 n 周期前的值（PIT 安全：n>0 取历史，n<0 取未来仅用于标签）"""
    return series.shift(int(n))


@DEFAULT_REGISTRY.register("Mean")
def op_mean(series: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return series.rolling(n, min_periods=max(1, n // 2)).mean()


@DEFAULT_REGISTRY.register("Std")
def op_std(series: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return series.rolling(n, min_periods=max(1, n // 2)).std()


@DEFAULT_REGISTRY.register("Sum")
def op_sum(series: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return series.rolling(n, min_periods=max(1, n // 2)).sum()


@DEFAULT_REGISTRY.register("Max")
def op_max(series: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return series.rolling(n, min_periods=max(1, n // 2)).max()


@DEFAULT_REGISTRY.register("Min")
def op_min(series: pd.Series, n: int) -> pd.Series:
    n = int(n)
    return series.rolling(n, min_periods=max(1, n // 2)).min()


@DEFAULT_REGISTRY.register("Rank")
def op_rank(series: pd.Series) -> pd.Series:
    """横截面排名（百分比），需在 groupby date 后调用"""
    return series.rank(pct=True)


@DEFAULT_REGISTRY.register("Delta")
def op_delta(series: pd.Series, n: int) -> pd.Series:
    return series - series.shift(int(n))


@DEFAULT_REGISTRY.register("RSI")
def op_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """RSI 指标（纯 pandas 实现，不依赖 pandas_ta）"""
    n = int(n)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


@DEFAULT_REGISTRY.register("WMA")
def op_wma(series: pd.Series, n: int) -> pd.Series:
    """加权移动平均"""
    n = int(n)
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(n).apply(lambda x: np.dot(x, weights), raw=True)


# ============================================================
# 表达式解析器：将字符串公式转为可执行计算图
# ============================================================

class ExpressionParser:
    """
    解析 Qlib 风格表达式

    支持:
      $close, $open, $high, $low, $volume, $amount  # 字段引用
      Ref($close, 5)                                  # 函数调用
      Ref($close, 5) / $close - 1                     # 四则运算
      ($close - Mean($close, 20)) / Std($close, 20)   # 嵌套
    """

    FIELD_PATTERN = re.compile(r"\$(\w+)")
    FUNC_PATTERN = re.compile(r"(\w+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)")

    def __init__(self, registry: OperatorRegistry = DEFAULT_REGISTRY):
        self.registry = registry

    def parse(self, expr: str) -> Callable[[pd.DataFrame], pd.Series]:
        """将表达式字符串编译为接受 DataFrame 返回 Series 的函数"""
        tokens = self._tokenize(expr)
        ast = self._build_ast(tokens)
        return lambda df: self._eval_ast(ast, df)

    def _tokenize(self, expr: str) -> List[str]:
        """简单分词：识别字段、函数、运算符、数字、括号、逗号"""
        token_pattern = re.compile(
            r"(\$\w+|[\w]+(?=\())|(\d+\.?\d*)|([+\-*/(),])|(\s+)"
        )
        tokens = []
        pos = 0
        while pos < len(expr):
            m = token_pattern.match(expr, pos)
            if not m:
                raise ValueError(f"无法解析表达式位置 {pos}: {expr[pos:]}")
            if m.group(1) or m.group(2) or m.group(3):
                tokens.append(m.group(0).strip())
            pos = m.end()
        return tokens

    def _build_ast(self, tokens: List[str]) -> Any:
        """构建简单 AST（递归下降解析四则运算 + 函数调用）"""
        self._tokens = tokens
        self._pos = 0
        return self._parse_expr()

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _parse_expr(self) -> Any:
        node = self._parse_term()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_term()
            node = ("binop", op, node, right)
        return node

    def _parse_term(self) -> Any:
        node = self._parse_factor()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_factor()
            node = ("binop", op, node, right)
        return node

    def _parse_factor(self) -> Any:
        tok = self._peek()
        if tok == "(":
            self._consume()
            node = self._parse_expr()
            if self._peek() != ")":
                raise ValueError("缺少右括号")
            self._consume()
            return node
        if tok == "-":
            self._consume()
            return ("neg", self._parse_factor())
        if tok and tok.startswith("$"):
            self._consume()
            return ("field", tok[1:])
        if tok and tok.replace(".", "", 1).isdigit():
            self._consume()
            return ("num", float(tok))
        if tok and self.registry.get(tok) is not None:
            self._consume()
            if self._peek() != "(":
                raise ValueError(f"函数 {tok} 后需跟括号")
            self._consume()
            args = [self._parse_expr()]
            while self._peek() == ",":
                self._consume()
                args.append(self._parse_expr())
            if self._peek() != ")":
                raise ValueError("函数缺少右括号")
            self._consume()
            return ("func", tok, args)
        raise ValueError(f"无法解析 token: {tok}")

    def _eval_ast(self, node: Any, df: pd.DataFrame) -> pd.Series:
        kind = node[0]
        if kind == "field":
            col = node[1]
            if col not in df.columns:
                raise KeyError(f"字段 ${col} 不存在于数据列: {list(df.columns)}")
            return df[col]
        if kind == "num":
            return pd.Series(node[1], index=df.index)
        if kind == "neg":
            return -self._eval_ast(node[1], df)
        if kind == "binop":
            _, op, left, right = node
            l = self._eval_ast(left, df)
            r = self._eval_ast(right, df)
            if op == "+":
                return l + r
            if op == "-":
                return l - r
            if op == "*":
                return l * r
            if op == "/":
                return l / r.replace(0, np.nan)
        if kind == "func":
            _, name, args = node
            func = self.registry.get(name)
            if func is None:
                raise KeyError(f"未注册算子: {name}")
            evaluated = [self._eval_ast(a, df) for a in args]
            # 数字参数保持原值
            final_args = []
            for a, ev in zip(args, evaluated):
                if a[0] == "num":
                    final_args.append(a[1])
                else:
                    final_args.append(ev)
            return func(*final_args)
        raise ValueError(f"未知 AST 节点: {node}")


# ============================================================
# 多级因子缓存（借鉴 Qlib DatasetCache）
# ============================================================

class FactorCache:
    """LRU + 指纹缓存，避免重复计算"""

    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._cache: OrderedDict[str, pd.Series] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _fingerprint(expr: str, data: pd.DataFrame) -> str:
        """计算 (表达式, 数据) 指纹"""
        cols = sorted(data.columns)
        shape = data.shape
        # 用列名 + shape + 首尾行 hash 作为数据指纹（轻量）
        try:
            head = data.iloc[0].to_dict() if len(data) > 0 else {}
            tail = data.iloc[-1].to_dict() if len(data) > 0 else {}
            sig = f"{cols}|{shape}|{head}|{tail}"
        except Exception:
            sig = f"{cols}|{shape}"
        return hashlib.md5(f"{expr}::{sig}".encode()).hexdigest()

    def get(self, expr: str, data: pd.DataFrame) -> Optional[pd.Series]:
        key = self._fingerprint(expr, data)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, expr: str, data: pd.DataFrame, value: pd.Series):
        key = self._fingerprint(expr, data)
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# ============================================================
# 表达式因子引擎
# ============================================================

class ExpressionFactorEngine:
    """
    基于 Qlib 风格表达式的因子引擎

    用法:
        engine = ExpressionFactorEngine()
        engine.add_factor("momentum_20", "Ref($close, 20) / $close - 1")
        engine.add_factor("rsi_14", "RSI($close, 14)")
        result = engine.compute(data)  # data: code, date, open, high, low, close, volume
    """

    def __init__(self, registry: OperatorRegistry = DEFAULT_REGISTRY):
        self.parser = ExpressionParser(registry)
        self.cache = FactorCache(max_size=256)
        self._factors: Dict[str, str] = {}  # name -> expression
        self._compiled: Dict[str, Callable] = {}

    def add_factor(self, name: str, expression: str):
        """注册因子（编译表达式）"""
        compiled = self.parser.parse(expression)
        self._factors[name] = expression
        self._compiled[name] = compiled

    def list_factors(self) -> Dict[str, str]:
        return dict(self._factors)

    def compute(self, data: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
        """
        批量计算所有已注册因子

        参数:
            data: 必须含 code, date 列 + OHLCV
            use_cache: 是否启用缓存

        返回:
            DataFrame: code, date, [各因子列]
        """
        if data.empty:
            return data

        result = data[["code", "date"]].copy()
        df_sorted = data.sort_values(["code", "date"]).reset_index(drop=True)

        for name, compiled in self._compiled.items():
            expr_str = self._factors[name]
            series = None
            if use_cache:
                cached = self.cache.get(expr_str, df_sorted)
                if cached is not None:
                    series = cached.copy()
                    series.index = df_sorted.index

            if series is None:
                series = pd.Series(np.nan, index=df_sorted.index, dtype=float)
                # 按股票分组计算（保证不跨股票泄漏）
                for code, idx in df_sorted.groupby("code").groups.items():
                    sub = df_sorted.loc[idx]
                    try:
                        vals = compiled(sub)
                        series.loc[idx] = vals.values if isinstance(vals, pd.Series) else vals
                    except Exception as e:
                        print(f"计算 {code} 的 {name}({expr_str}) 失败: {e}")
                if use_cache:
                    self.cache.put(expr_str, df_sorted, series.copy())

            result[name] = series.values

        return result

    def compute_single(self, data: pd.DataFrame, expression: str, use_cache: bool = True) -> pd.Series:
        """计算单个表达式因子（不注册）"""
        compiled = self.parser.parse(expression)
        # 预校验字段存在性，让 KeyError 提前抛出
        fields = re.findall(r"\$(\w+)", expression)
        missing = [f for f in fields if f not in data.columns]
        if missing:
            raise KeyError(f"字段 ${missing[0]} 不存在于数据列: {list(data.columns)}")

        series = pd.Series(np.nan, index=data.index, dtype=float)
        for code, idx in data.groupby("code").groups.items():
            sub = data.loc[idx]
            try:
                vals = compiled(sub)
                series.loc[idx] = vals.values if isinstance(vals, pd.Series) else vals
            except Exception:
                series.loc[idx] = np.nan
        return series

    def cache_stats(self) -> Dict[str, int]:
        return self.cache.stats()


# ============================================================
# Alpha158 风格因子库（借鉴 Qlib Alpha158）
# ============================================================

def build_alpha158_factors() -> Dict[str, str]:
    """
    构建类 Alpha158 因子库（精选 20 个代表性因子）

    借鉴 Qlib Alpha158: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
    """
    factors = {}
    # ---- 价格动量 ----
    factors["mom_5"] = "Ref($close, 5) / $close - 1"
    factors["mom_20"] = "Ref($close, 20) / $close - 1"
    factors["mom_60"] = "Ref($close, 60) / $close - 1"
    # ---- 反转（短周期）----
    factors["rev_5"] = "-(Ref($close, 5) / $close - 1)"
    factors["rev_20"] = "-(Ref($close, 20) / $close - 1)"
    # ---- 均线偏离 ----
    factors["ma_5_bias"] = "$close / Mean($close, 5) - 1"
    factors["ma_20_bias"] = "$close / Mean($close, 20) - 1"
    factors["ma_60_bias"] = "$close / Mean($close, 60) - 1"
    # ---- 波动率 ----
    factors["vol_20"] = "Std(Ref($close, 1) / $close - 1, 20)"
    factors["vol_60"] = "Std(Ref($close, 1) / $close - 1, 60)"
    # ---- 成交量因子 ----
    factors["vol_ratio_5_20"] = "Mean($volume, 5) / Mean($volume, 20)"
    factors["vol_delta_5"] = "Delta($volume, 5) / Ref($volume, 5)"
    # ---- 振幅 ----
    factors["range_20"] = "(Max($high, 20) - Min($low, 20)) / $close"
    # ---- RSI ----
    factors["rsi_14"] = "RSI($close, 14)"
    # ---- 加权均线 ----
    factors["wma_20_bias"] = "$close / WMA($close, 20) - 1"
    # ---- 高低价相对位置 ----
    factors["high_loc_20"] = "($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20))"
    # ---- 量价相关 ----
    factors["amount_ma_5"] = "Mean($amount, 5)"
    factors["amount_ma_20"] = "Mean($amount, 20)"
    # ---- 短期收益 ----
    factors["ret_1d"] = "Ref($close, 1) / $close - 1"
    factors["ret_10d"] = "Ref($close, 10) / $close - 1"
    return factors
