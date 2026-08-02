"""P0-4 master 调用 path_policy 的薄包装。

避免直接 import execution-monitor-engine 的模块（会污染 sys.path），
改为动态加载并缓存模块引用。

设计同 factor-engine/scripts/pit_guard.py 的模式。
"""
import importlib.util as _ilu
import logging
import os
from pathlib import Path

logger = logging.getLogger("path-policy-loader")

_CACHED_MODULE = None
_TRACKER_INSTANCE = None


def _resolve_path_policy_path() -> Path:
    """解析 execution-monitor-engine/scripts/path_policy.py 的绝对路径。"""
    # master/scripts/path_policy_loader.py → master/ → skills/execution-monitor-engine/scripts/
    master_root = Path(__file__).parent.parent
    return (
        master_root / "skills" / "execution-monitor-engine" / "scripts" / "path_policy.py"
    )


def _load_path_policy_module():
    """动态加载 execution-monitor-engine 的 path_policy 模块（缓存）。"""
    global _CACHED_MODULE
    if _CACHED_MODULE is not None:
        return _CACHED_MODULE
    policy_path = _resolve_path_policy_path()
    if not policy_path.exists():
        raise FileNotFoundError(f"path_policy.py 不存在: {policy_path}")
    spec = _ilu.spec_from_file_location("_path_policy", str(policy_path))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHED_MODULE = mod
    logger.debug(f"已动态加载 path_policy 模块: {policy_path}")
    return mod


def get_git_tracker():
    """获取全局 GitChangeTracker 实例（单例）。

    返回 None 如果模块不可用。
    """
    global _TRACKER_INSTANCE
    if _TRACKER_INSTANCE is not None:
        return _TRACKER_INSTANCE
    try:
        mod = _load_path_policy_module()
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _TRACKER_INSTANCE = mod.GitChangeTracker(repo_root=repo_root)
        return _TRACKER_INSTANCE
    except Exception as e:
        logger.debug(f"GitChangeTracker 初始化跳过: {e}")
        return None


def post_run_guard(tracker, violator: str = "master-engine") -> None:
    """退出兜底校验的薄包装（委托给 path_policy.post_run_guard）。"""
    if tracker is None:
        return
    try:
        mod = _load_path_policy_module()
        mod.post_run_guard(tracker, violator=violator)
    except Exception as e:
        logger.debug(f"post_run_guard 跳过: {e}")


def guard_llm_write(changed_paths: list, violator: str = "reports-engine") -> list:
    """LLM 写文件前的前置校验薄包装（委托给 path_policy.guard_llm_write）。"""
    try:
        mod = _load_path_policy_module()
        return mod.guard_llm_write(changed_paths, violator=violator)
    except Exception as e:
        logger.warning(f"guard_llm_write 跳过: {e}")
        return []
