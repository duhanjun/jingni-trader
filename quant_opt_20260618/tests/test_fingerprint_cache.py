"""
内容指纹缓存单元测试
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from quant_opt_20260618.cache import fingerprint, DiskCache, CacheStats
from quant_opt_20260618.tests.fixtures import make_synthetic_ashare_data


def test_fingerprint_dataframe_changes_with_data():
    df1 = make_synthetic_ashare_data(n_stocks=3, n_days=20, seed=1)
    df2 = make_synthetic_ashare_data(n_stocks=3, n_days=20, seed=2)  # 不同种子 → 不同数据
    df3 = make_synthetic_ashare_data(n_stocks=3, n_days=20, seed=1)  # 同种子 → 同数据
    fp1 = fingerprint(df1)
    fp2 = fingerprint(df2)
    fp3 = fingerprint(df3)
    assert fp1 != fp2
    assert fp1 == fp3
    assert isinstance(fp1, str) and len(fp1) == 16


def test_fingerprint_dict():
    a = {"x": 1, "y": [1, 2, 3]}
    b = {"y": [1, 2, 3], "x": 1}
    c = {"x": 1, "y": [1, 2, 4]}
    assert fingerprint(a) == fingerprint(b)  # 顺序无关
    assert fingerprint(a) != fingerprint(c)


def test_fingerprint_scalar_types():
    assert fingerprint(1) != fingerprint("1")
    assert fingerprint(1.0) != fingerprint(1)
    assert fingerprint(None) != fingerprint(0)


def test_disk_cache_hit_miss(tmp_path):
    cache = DiskCache(root=str(tmp_path / "cache"))
    counter = {"calls": 0}

    def compute():
        counter["calls"] += 1
        return {"result": 42}

    fp = "abc123"
    # 第一次：miss
    v1 = cache.get_or_compute("k", fp, compute)
    assert v1["result"] == 42
    assert counter["calls"] == 1
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0

    # 第二次：hit
    v2 = cache.get_or_compute("k", fp, compute)
    assert v2["result"] == 42
    assert counter["calls"] == 1
    assert cache.stats.hits == 1


def test_disk_cache_different_fp_invalidates(tmp_path):
    cache = DiskCache(root=str(tmp_path / "cache"))
    counter = {"calls": 0}

    def compute():
        counter["calls"] += 1
        return counter["calls"]

    cache.get_or_compute("k", "fp1", compute)
    cache.get_or_compute("k", "fp2", compute)  # fp 不同，触发重算
    assert counter["calls"] == 2


def test_disk_cache_clear(tmp_path):
    cache = DiskCache(root=str(tmp_path / "cache"))
    cache.get_or_compute("k", "fp", lambda: 1)
    assert cache.has("k", "fp")
    cache.clear()
    assert not cache.has("k", "fp")
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0
