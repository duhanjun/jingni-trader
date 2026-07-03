"""
jingni-trader 兼容包装层

将 jingni-trader 仓库内的原生回测适配器与本次新增的向量化适配器
统一暴露为可独立测试的接口，避免相对导入问题。
"""
from __future__ import annotations

import os
import sys
import importlib.util
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------- 模块级基类（替代原 base_backtest_engine.BaseBacktestEngine）----------

class _StubBase(ABC):
    """轻量基类，替代原 BaseBacktestEngine，避免修改原代码。"""

    @abstractmethod
    def run_backtest(self, *args, **kwargs):
        ...


def _load_base_metrics():
    """动态加载原 BaseBacktestMetrics（绩效计算工具类）。"""
    base_dir = os.path.join(ROOT, "skills", "backtest-engine", "scripts")
    spec = importlib.util.spec_from_file_location(
        "jingni_backtest_base",
        os.path.join(base_dir, "base", "base_backtest.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BaseBacktestMetrics


# 模块级单例
_BaseBacktestMetrics = _load_base_metrics()


# ---------- 兼容包装类 ----------

class NativeAdapterCompat:
    """对 jingni-trader 原生回测适配器的兼容包装。

    直接复用原 native_adapter.py 的逻辑，仅修复相对导入问题，
    不修改 main 分支的任何代码。
    """

    def __init__(self):
        base_dir = os.path.join(ROOT, "skills", "backtest-engine", "scripts")
        native_path = os.path.join(base_dir, "adapters", "native_adapter.py")
        with open(native_path, "r", encoding="utf-8") as f:
            src = f.read()
        # 替换相对导入为模块级引用
        src = src.replace(
            "from ..base.base_backtest_engine import BaseBacktestEngine",
            "from optimizations.independent_v2.jingni_compat import _StubBase as BaseBacktestEngine",
        )
        src = src.replace(
            "from ..base.base_backtest import BaseBacktestMetrics",
            "from optimizations.independent_v2.jingni_compat import _BaseBacktestMetrics as BaseBacktestMetrics",
        )
        ns: Dict[str, Any] = {"__name__": "native_adapter_compat"}
        exec(compile(src, native_path, "exec"), ns)
        self._NativeAdapter = ns["NativeAdapter"]

    def run_backtest(self, data, signals, **kwargs):
        return self._NativeAdapter().run_backtest(data, signals, **kwargs)


class VectorizedAdapterCompat:
    """对本次新增的向量化回测适配器的兼容包装。"""

    def __init__(self):
        from .vectorized_backtest.vectorized_adapter import VectorizedAdapter
        self._cls = VectorizedAdapter

    def run_backtest(self, data, signals, **kwargs):
        return self._cls().run_backtest(data, signals, **kwargs)
