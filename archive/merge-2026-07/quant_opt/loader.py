"""
模块加载工具

skills 目录下的子技能文件夹使用连字符 (如 backtest-engine, factor-engine),
无法作为标准 Python 模块名直接 import。本工具用 importlib.util 按文件路径
加载这些模块, 供优化验证代码复用原生实现做对比测试。

注意: 仅用于新分支的验证测试, 不修改 main 分支任何代码。
"""
import os
import sys
import importlib.util
from types import ModuleType

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_module_from_path(module_name: str, file_path: str) -> ModuleType:
    """按文件路径加载 Python 模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_native_adapter():
    """加载原生回测适配器 (skills/backtest-engine/scripts/adapters/native_adapter.py)"""
    base_dir = os.path.join(_WORKSPACE_ROOT, "skills", "backtest-engine", "scripts")
    # 先加载依赖的 base 模块 (相对导入需要包上下文)
    pkg_name = "_jt_backtest"
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [base_dir]
        sys.modules[pkg_name] = pkg

    for sub in ["base", "base.base_backtest", "base.base_backtest_engine", "adapters"]:
        full = f"{pkg_name}.{sub}"
        if full not in sys.modules:
            mod = ModuleType(full)
            if sub == "base" or sub == "adapters":
                mod.__path__ = [os.path.join(base_dir, sub.split(".")[-1])]
                mod.__package__ = full
            else:
                mod.__package__ = pkg_name + "." + ".".join(sub.split(".")[:-1])
            sys.modules[full] = mod

    # 加载 base.base_backtest (无相对导入)
    _load_module_from_path(
        f"{pkg_name}.base.base_backtest",
        os.path.join(base_dir, "base", "base_backtest.py"),
    )
    # base.base_backtest_engine 依赖 base.base_backtest? 实际不依赖
    _load_module_from_path(
        f"{pkg_name}.base.base_backtest_engine",
        os.path.join(base_dir, "base", "base_backtest_engine.py"),
    )
    # native_adapter 用相对导入 ..base.base_backtest_engine / ..base.base_backtest
    adapter_mod = _load_module_from_path(
        f"{pkg_name}.adapters.native_adapter",
        os.path.join(base_dir, "adapters", "native_adapter.py"),
    )
    return adapter_mod.NativeAdapter


def load_factor_engine():
    """加载原生因子引擎 (skills/factor-engine/engine.py)"""
    base_dir = os.path.join(_WORKSPACE_ROOT, "skills", "factor-engine", "scripts")
    pkg_name = "_jt_factor"
    if pkg_name not in sys.modules:
        pkg = ModuleType(pkg_name)
        pkg.__path__ = [base_dir]
        sys.modules[pkg_name] = pkg

    # factor-engine/engine.py 用 sys.path.insert 加载 scripts.config 等
    # 直接把它当作独立脚本加载, 并预先把 scripts 目录加入 path
    sys.path.insert(0, base_dir)
    # config 模块
    config_path = os.path.join(base_dir, "config.py")
    if os.path.exists(config_path) and "config" not in sys.modules:
        _load_module_from_path("config", config_path)

    engine_path = os.path.join(_WORKSPACE_ROOT, "skills", "factor-engine", "engine.py")
    mod = _load_module_from_path("_jt_factor_engine", engine_path)
    return mod.FactorEngine


def load_backtest_base_classes():
    """加载 BaseBacktestEngine 与 BaseBacktestMetrics (供向量化适配器继承)"""
    base_dir = os.path.join(_WORKSPACE_ROOT, "skills", "backtest-engine", "scripts", "base")
    metrics_mod = _load_module_from_path(
        "_jt_backtest_base_metrics",
        os.path.join(base_dir, "base_backtest.py"),
    )
    engine_mod = _load_module_from_path(
        "_jt_backtest_base_engine",
        os.path.join(base_dir, "base_backtest_engine.py"),
    )
    return engine_mod.BaseBacktestEngine, metrics_mod.BaseBacktestMetrics
