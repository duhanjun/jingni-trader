"""factor-engine 调用 data-engine pit 模块的薄包装。

避免直接 import data-engine 的模块（会污染 sys.path），
改为动态加载并缓存模块引用。

参考 PRD 第十章 10.7 跨 Skill 依赖处理方案。
"""
from __future__ import annotations

import importlib.util as _ilu
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("factor_engine.pit_guard")

_CACHED_MODULE = None
_PATH_POLICY_PATH: Optional[Path] = None


def _resolve_pit_path() -> Path:
    """解析 data-engine/scripts/pit.py 的绝对路径。

    支持两种场景：
    1. 从项目根目录运行（cwd 或父目录含 skills/data-engine）
    2. 从 factor-engine 目录运行（向上查找 skills/data-engine）
    """
    global _PATH_POLICY_PATH
    if _PATH_POLICY_PATH is not None and _PATH_POLICY_PATH.exists():
        return _PATH_POLICY_PATH

    # 候选路径：从本文件位置向上查找
    candidates = []
    current = Path(__file__).resolve().parent
    for _ in range(6):  # 最多向上 6 层
        candidates.append(current / "skills" / "data-engine" / "scripts" / "pit.py")
        candidates.append(current / "data-engine" / "scripts" / "pit.py")
        current = current.parent

    # 也支持从项目根目录的 skills/factor-engine/scripts/ 向上找
    for c in candidates:
        if c.exists():
            _PATH_POLICY_PATH = c
            return c

    # 兜底：尝试 QUANT_WORK_DIR 环境变量
    work_dir = os.environ.get("QUANT_WORK_DIR", "")
    if work_dir:
        p = Path(work_dir).parent / "skills" / "data-engine" / "scripts" / "pit.py"
        if p.exists():
            _PATH_POLICY_PATH = p
            return p

    raise FileNotFoundError(
        "无法定位 data-engine/scripts/pit.py，请确认从项目根目录运行"
    )


def _load_pit_module():
    """动态加载 data-engine 的 pit 模块（缓存）。"""
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE
    pit_path = _resolve_pit_path()
    spec = _ilu.spec_from_file_location("_data_engine_pit", str(pit_path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHED_MODULE = mod
    logger.debug(f"已动态加载 pit 模块: {pit_path}")
    return mod


def pit_filter(df, asof: str, caller: str = ""):
    """PIT 过滤的薄包装，委托给 data-engine 的 pit_filter。"""
    mod = _load_pit_module()
    if caller:
        return mod.ensure_pit_filtered(df, asof, caller=caller)
    return mod.pit_filter(df, asof)


def is_strict_mode() -> bool:
    """是否严格模式。"""
    mod = _load_pit_module()
    return mod._is_strict_mode()
