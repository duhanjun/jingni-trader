"""
参数化扫描装饰器（借鉴来源: VectorBT @parameterized / @chunked）

设计动机
========
jingni-trader 当前因子计算与回测都是单组参数,优化场景下
需要扫描多组参数 (例如窗口 5/10/20/60), 原始实现需要 for 循环重复
调用,无法直接构造"参数维度"概念,也不支持向量化广播。

借鉴 VectorBT PRO 的核心 API:
    @vbt.parameterized
    @vbt.chunked(chunk_len=1000)
    def pipeline(data, fast, slow, signal): ...
    https://vectorbt.pro/features/optimization/#parameterized-decorator

本模块提供
==========
- @parameterized : 把参数化函数转成"参数网格 -> 结果矩阵"API
- @chunked       : 大网格分块,避免内存爆炸
- sweep: 顶层函数,接受参数网格直接扫描
"""
from __future__ import annotations

import functools
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SweepResult:
    """一次扫描的结果"""
    grid: List[Dict[str, Any]]
    values: np.ndarray  # 形状: (n_params, *result_shape)
    elapsed_sec: float
    func_name: str

    def to_dataframe(self, value_name: str = "value") -> pd.DataFrame:
        flat = self.values.reshape(len(self.grid), -1)
        cols = [f"{value_name}_{i}" for i in range(flat.shape[1])] \
            if flat.ndim > 1 else [value_name]
        df = pd.DataFrame(flat, columns=cols)
        for k in self.grid[0].keys():
            df[k] = [g[k] for g in self.grid]
        return df

    def best(self, higher_is_better: bool = True, value_name: str = "value") -> Dict[str, Any]:
        arr = np.asarray(self.values)
        if higher_is_better:
            idx = int(np.nanargmax(arr))
        else:
            idx = int(np.nanargmin(arr))
        return {
            "params": self.grid[idx],
            "value": float(arr[idx]) if np.isscalar(arr[idx]) else arr[idx],
        }


def _to_grid(param_space: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_space.keys())
    if not keys:
        raise ValueError("param_space 不能为空")
    values = [list(param_space[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def sweep(
    func: Callable[..., Any],
    param_space: Dict[str, Iterable[Any]],
    common_kwargs: Optional[Dict[str, Any]] = None,
    chunk_size: Optional[int] = None,
    n_jobs: int = 1,
) -> SweepResult:
    """
    对函数 `func` 进行参数网格扫描。

    参数
    -----
    func : 接受 **params 关键字的函数
    param_space : {"param_name": [v1, v2, ...], ...}
    common_kwargs : 每次调用都相同的额外参数
    chunk_size : 大网格分块, 配合 n_jobs 调度
    n_jobs : 并行进程数 (1 表示串行)
    """
    common = dict(common_kwargs or {})
    grid = _to_grid(param_space)
    if not grid:
        raise ValueError("param_space 为空")

    t0 = time.perf_counter()
    if n_jobs == 1 or chunk_size is None or len(grid) <= chunk_size:
        results = [func(**common, **params) for params in grid]
    else:
        # 简化版分块,单进程顺序执行
        results = []
        for i in range(0, len(grid), chunk_size):
            chunk = grid[i : i + chunk_size]
            results.extend(func(**common, **params) for params in chunk)
    elapsed = time.perf_counter() - t0

    return SweepResult(
        grid=grid,
        values=np.array(results, dtype=object),
        elapsed_sec=elapsed,
        func_name=getattr(func, "__name__", "func"),
    )


def parameterized(
    param_space: Optional[Dict[str, Iterable[Any]]] = None,
):
    """
    装饰器: 把普通函数转成"可参数化"函数。

    用法 (函数必须以关键字形式接收参数):
        >>> @parameterized({"window": [5, 10, 20]})
        ... def rolling_mean(series, window):
        ...     return float(pd.Series(series).rolling(window).mean().iloc[-1])
        >>> rolling_mean(series, common_kwargs={"series": data})
    """
    def deco(func: Callable[..., Any]) -> Callable[..., SweepResult]:
        @functools.wraps(func)
        def wrapper(**kwargs) -> SweepResult:
            sp = param_space or {}
            common_kwargs = dict(kwargs)
            return sweep(func, sp, common_kwargs=common_kwargs)

        return wrapper
    return deco


def chunked(chunk_len: int = 1000):
    """装饰器: 标记函数支持分块, 与 sweep 配合使用"""
    def deco(func: Callable[..., Any]) -> Callable[..., Any]:
        func._vbt_chunk_len = chunk_len
        return func
    return deco
