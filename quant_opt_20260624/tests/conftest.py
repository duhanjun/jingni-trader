"""
pytest 共享 fixtures 与路径配置

处理：
1. 将 quant_opt_20260624 目录加入 sys.path
2. 通过文件路径加载主仓库的 NativeAdapter（hyphen 目录无法常规 import）
"""
import os
import sys
import importlib.util
import pytest

# 将本目录加入 sys.path，使测试可直接 import 优化模块
_OPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../quant_opt_20260624/tests
sys.path.insert(0, _OPT_DIR)
# 优化模块所在目录（synthetic_data.py 等）
_OPT_ROOT = os.path.abspath(os.path.join(_OPT_DIR, ".."))
sys.path.insert(0, _OPT_ROOT)

_REPO_ROOT = os.path.abspath(os.path.join(_OPT_DIR, "..", ".."))  # /workspace


def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import types

# 主仓库 skills 目录用 hyphen 命名，无法作为合法 Python 包名。
# 这里构造一个合成的包层次 jt_bt.scripts.{base,adapters}，并把各目录的 __path__ 指过去，
# 使 native_adapter.py 中的相对导入 `from ..base.X import Y` 能正确解析。
_BT_ROOT = os.path.join(_REPO_ROOT, "skills", "backtest-engine", "scripts")

def _make_pkg(name: str, path: str):
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg
    return pkg

_make_pkg("jt_bt", _BT_ROOT)
_make_pkg("jt_bt.scripts", _BT_ROOT)
_make_pkg("jt_bt.scripts.base", os.path.join(_BT_ROOT, "base"))
_make_pkg("jt_bt.scripts.adapters", os.path.join(_BT_ROOT, "adapters"))

# 先加载 base 模块（无相对导入）
_load_module_from_path(
    "jt_bt.scripts.base.base_backtest_engine",
    os.path.join(_BT_ROOT, "base", "base_backtest_engine.py"),
)
_load_module_from_path(
    "jt_bt.scripts.base.base_backtest",
    os.path.join(_BT_ROOT, "base", "base_backtest.py"),
)

# 再加载 native_adapter —— 其 `from ..base.X` 现在可解析到 jt_bt.scripts.base.X
_native_mod = _load_module_from_path(
    "jt_bt.scripts.adapters.native_adapter",
    os.path.join(_BT_ROOT, "adapters", "native_adapter.py"),
)
NativeAdapter = _native_mod.NativeAdapter


@pytest.fixture
def native_adapter():
    return NativeAdapter()


@pytest.fixture
def vectorized_adapter():
    from vectorized_backtest import VectorizedAdapter
    return VectorizedAdapter()


@pytest.fixture(scope="session")
def small_dataset():
    """小数据集：20 股票 × 120 日，用于正确性测试"""
    from synthetic_data import generate_synthetic_ohlcv, generate_signals
    data = generate_synthetic_ohlcv(n_codes=20, n_days=120, seed=42)
    signals = generate_signals(data, strategy="ma_cross", seed=42)
    return data, signals


@pytest.fixture(scope="session")
def large_dataset():
    """大数据集：200 股票 × 500 日，用于性能测试"""
    from synthetic_data import generate_synthetic_ohlcv, generate_signals
    data = generate_synthetic_ohlcv(n_codes=200, n_days=500, seed=123)
    signals = generate_signals(data, strategy="ma_cross", seed=123)
    return data, signals
