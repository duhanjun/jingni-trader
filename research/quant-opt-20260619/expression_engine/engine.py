"""
ExpressionEngine: 整合 parser + operators + cache 的对外门面

设计要点（参考 Qlib + vectorbt 优势）：
- 统一宽表接口：data 是 (index=date, columns=MultiIndex(code, field))
- 表达式解析 + 懒求值
- 表达式 hash -> 缓存结果，避免同一表达式重复计算
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
import hashlib
import time
import numpy as np
import pandas as pd

from .parser import ExpressionParser
from .operators import ExpressionOps, Feature


class ExpressionEngine:
    """表达式计算引擎"""

    def __init__(self, data: Optional[pd.DataFrame] = None, enable_cache: bool = True):
        self.data = data
        self.parser = ExpressionParser()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._hit_count = 0
        self._miss_count = 0
        self.enable_cache = enable_cache
        self._stats: Dict[str, Dict[str, Any]] = {}

    # -------- 数据 --------
    def set_data(self, data: pd.DataFrame):
        self.data = data
        self.clear_cache()

    def clear_cache(self):
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    # -------- 解析 --------
    def parse(self, expr: str) -> ExpressionOps:
        return self.parser.parse(expr)

    # -------- 计算 --------
    def evaluate(self, expr: Union[str, ExpressionOps]) -> pd.DataFrame:
        """求值一个表达式，结果为 DataFrame(index=date, columns=code)"""
        if self.data is None:
            raise RuntimeError("请先 set_data")

        if isinstance(expr, str):
            cache_key = expr
            if self.enable_cache and cache_key in self._cache:
                self._hit_count += 1
                return self._cache[cache_key]
            node = self.parse(expr)
        else:
            cache_key = repr(expr)
            if self.enable_cache and cache_key in self._cache:
                self._hit_count += 1
                return self._cache[cache_key]
            node = expr

        start = time.perf_counter()
        result = self._evaluate_node(node)
        elapsed = time.perf_counter() - start

        self._stats[cache_key] = {
            "elapsed_s": elapsed,
            "shape": result.shape,
            "nan_ratio": float(result.isna().mean().mean()) if result.size else 0.0,
        }
        if self.enable_cache:
            self._cache[cache_key] = result
        self._miss_count += 1
        return result

    def _evaluate_node(self, node: ExpressionOps) -> pd.DataFrame:
        """递归求值：所有 operator 类都实现 __call__"""
        if isinstance(node, Feature):
            col = node._ALIAS[node.name]
            if col not in self.data.columns.get_level_values(0):
                raise KeyError(f"数据缺少特征 {col}; 可用: {self.data.columns.get_level_values(0).unique().tolist()}")
            return self.data[col]
        if hasattr(node, "_load"):
            return node._load(self.data)
        return node(self.data)

    # -------- 批量 --------
    def evaluate_many(self, exprs: List[str]) -> pd.DataFrame:
        """批量评估多个表达式，返回拼接好的 DataFrame"""
        results = {}
        for expr in exprs:
            res = self.evaluate(expr)
            results[expr] = res
        return pd.concat(results, axis=1)

    # -------- 统计 --------
    def stats(self) -> Dict[str, Any]:
        return {
            "cache_hit": self._hit_count,
            "cache_miss": self._miss_count,
            "cache_hit_ratio": (
                self._hit_count / max(1, self._hit_count + self._miss_count)
            ),
            "expression_stats": self._stats,
        }

    # -------- 上下文管理器 --------
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.clear_cache()
        return False


# -----------------------------------------------------------------------------
# 便捷工具
# -----------------------------------------------------------------------------
def build_pivot_panel(
    long_df: pd.DataFrame,
    code_col: str = "code",
    date_col: str = "date",
    feature_cols: List[str] = None,
) -> pd.DataFrame:
    """
    把长表（code, date, open, high, low, close, volume, amount）转成宽表
    index=date, columns=MultiIndex(code, field)
    """
    feature_cols = feature_cols or ["open", "high", "low", "close", "volume", "amount"]
    frames = []
    for f in feature_cols:
        if f not in long_df.columns:
            continue
        pivot = long_df.pivot(index=date_col, columns=code_col, values=f)
        pivot.columns = pd.MultiIndex.from_product([[f], pivot.columns])
        frames.append(pivot)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, axis=1)
    result.sort_index(axis=1, level=[0, 1], inplace=True)
    return result
