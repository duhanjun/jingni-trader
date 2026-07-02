"""
内容指纹缓存 (Content-Fingerprint Cache)

借鉴自：
  - Qlib 的 ExpressionCache / DatasetCache
  - AKQuant 的运行时热启动 (warm start) 机制
  - 通用"懒计算 + 哈希指纹"模式

问题背景：
  jingni-trader 当前在 `execute_stage` 中已经做了
  "如果产物文件存在则跳过" 的判断，但当上游参数微调而产物未变时，
  会用错缓存；当上游参数变了，又可能重复计算已有结果。
  通过给 (输入数据指纹, 公式 / 参数) 计算 hash，可以更细粒度地复用
  中间结果。

本模块提供：
  1. fingerprint(obj)：对 DataFrame / dict / 标量统一计算 hash
  2. DiskCache：以 (key, fingerprint) 为键的 pickle 缓存
  3. get_or_compute(cache, key, fingerprint, fn)：惰性求值
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


def fingerprint(obj: Any, max_rows: int = 5000) -> str:
    """
    计算输入对象的指纹 hash。

    - DataFrame: 取列名 + 前 max_rows 行 + shape
    - dict:    JSON 序列化后 hash
    - 其它:    pickle 后 hash
    """
    h = hashlib.sha256()
    if isinstance(obj, pd.DataFrame):
        h.update(f"DataFrame;shape={obj.shape};cols={list(obj.columns)};".encode())
        try:
            sample = obj.head(max_rows)
            h.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
        except Exception:
            h.update(pickle.dumps(sample))
    elif isinstance(obj, pd.Series):
        h.update(f"Series;len={len(obj)};name={obj.name};".encode())
        try:
            h.update(pd.util.hash_pandas_object(obj.head(max_rows), index=True).values.tobytes())
        except Exception:
            h.update(pickle.dumps(obj.head(max_rows)))
    elif isinstance(obj, (dict, list, tuple)):
        h.update(json.dumps(obj, sort_keys=True, default=str).encode())
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        h.update(repr(obj).encode())
    else:
        h.update(pickle.dumps(obj))
    return h.hexdigest()[:16]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_compute_sec: float = 0.0
    total_reused_sec: float = 0.0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class DiskCache:
    """基于 fingerprint 的磁盘 pickle 缓存"""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.stats = CacheStats()

    def _path(self, key: str, fp: str) -> str:
        safe_key = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:80]
        return os.path.join(self.root, f"{safe_key}__{fp}.pkl")

    def has(self, key: str, fp: str) -> bool:
        return os.path.exists(self._path(key, fp))

    def get(self, key: str, fp: str) -> Optional[Any]:
        p = self._path(key, fp)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, fp: str, value: Any) -> None:
        with open(self._path(key, fp), "wb") as f:
            pickle.dump(value, f)

    def clear(self) -> None:
        for fn in os.listdir(self.root):
            if fn.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.root, fn))
                except OSError:
                    pass
        self.stats = CacheStats()

    def get_or_compute(
        self,
        key: str,
        fingerprint_value: str,
        compute_fn: Callable[[], Any],
    ) -> Any:
        cached = self.get(key, fingerprint_value)
        if cached is not None:
            self.stats.hits += 1
            return cached
        t0 = time.perf_counter()
        value = compute_fn()
        self.stats.total_compute_sec += time.perf_counter() - t0
        self.set(key, fingerprint_value, value)
        self.stats.misses += 1
        return value


__all__ = [
    "fingerprint",
    "CacheStats",
    "DiskCache",
]
