"""优化模块集合 - 从 quant_optimizations 整合而来

T2-2: Polars 后端统一入口
=========================

环境变量 ``QUANT_FACTOR_BACKEND`` 控制 IC 计算 / 中性化等热路径的
DataFrame 后端：

- ``pandas``（默认）：行为与 v1.x 完全一致
- ``polars``：使用 polars 多线程 Rust 引擎，5000 股 × 1000 日场景
  下 IC 计算提速 5-15×
- ``auto``：检测 polars 可用性自动选择

> 注意：与 ``FACTOR_BACKEND``（pandas_ta / talib 技术指标后端）语义
> 不同，本后端作用于 optimizations/ 目录下的向量化热路径。
"""
import logging
import os
from typing import Optional

logger = logging.getLogger("optimizations")


# ---------------------------------------------------------------------------
# Polars 可用性与统一后端选择
# ---------------------------------------------------------------------------


def _polars_available() -> bool:
    """检查 polars 是否可导入。"""
    try:
        import polars  # noqa: F401
        return True
    except ImportError:
        return False


# 模块加载时检查并日志提示（仅提示一次）
if not _polars_available():
    logger.info(
        "polars 未安装，使用 pandas 后端。pip install polars>=0.20.0 启用加速"
    )


# 从环境变量读取默认后端（模块加载时固化，遵循项目硬约束）
_DEFAULT_BACKEND = os.environ.get("QUANT_FACTOR_BACKEND", "pandas")


def get_backend(backend: str = "auto") -> str:
    """统一后端选择逻辑。

    参数
    ----
    backend: ``"pandas"`` / ``"polars"`` / ``"auto"``

    返回
    ----
    实际使用的后端字符串（``"pandas"`` 或 ``"polars"``）。

    - ``auto``：polars 可用时返回 ``"polars"``，否则 ``"pandas"``
    - ``polars``：polars 不可用时回退 ``"pandas"`` 并 warning
    - ``pandas``：直接返回
    """
    if backend == "auto":
        return "polars" if _polars_available() else "pandas"
    if backend == "polars" and not _polars_available():
        logger.warning("polars 未安装，自动回退 pandas 后端")
        return "pandas"
    return backend


def resolve_backend(backend: Optional[str]) -> str:
    """解析调用方传入的 backend 参数。

    - ``None``：使用模块级默认（环境变量 ``QUANT_FACTOR_BACKEND``）
    - 其他：走 ``get_backend``
    """
    if backend is None:
        return get_backend(_DEFAULT_BACKEND)
    return get_backend(backend)
