"""skill 版本检查与自动部署工具。

设计原则：
- 安全第一：版本检查只查 GitHub 最新 commit 并与本地版本比对，**永不修改用户文件**
- 自动部署：若子 skill 目录不存在（全新安装），自动从 GitHub 克隆（greenfield 操作，无用户数据风险）
- 检测到落后时输出提示信息（INFO 日志），告知用户手动执行 git pull
- 失败静默降级（debug 日志），不阻断主流程
- 24h 内只查一次（靠 skill-sync.yml 的 last_check 缓存）

触发点：
- jingni-trader：engine.py 的 MasterEngine.__init__
- jingni-datafeed：jingni_client.py 模块加载时

用户手动同步命令（agent/用户均可执行）：
- git 仓库：cd <skill_root> && git pull
- 非 git 仓库：重新从 GitHub 下载 zip 替换

skill-sync.yml 字段：
    repo: duhanjun/jingni-trader       # GitHub 仓库（owner/repo）
    branch: main                        # 跟踪分支
    local_commit: a1b2c3d4...           # 上次同步到的 commit SHA
    last_check: 2026-07-24T17:00:00Z    # 上次检查时间（用于 24h 节流）
    update_policy: auto                 # auto | disabled
    check_interval_hours: 24             # 检查间隔
"""
from __future__ import annotations

import os
import sys
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logger = logging.getLogger("skill_sync")

_GITHUB_API = "https://api.github.com/repos/{repo}/commits/{ref}"
_DEFAULT_INTERVAL_HOURS = 24

# 进程内标志：避免同进程多次 import 触发重复检查
_SYNC_DONE: bool = False


# ============================================================================
# 元数据读写
# ============================================================================

def _read_meta(skill_root: str) -> Optional[Dict[str, Any]]:
    """读取 skill-sync.yml。"""
    meta_path = os.path.join(skill_root, "skill-sync.yml")
    if not os.path.exists(meta_path):
        return None
    if yaml is None:
        logger.debug("PyYAML 未安装，跳过 skill 版本检查")
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        if not meta.get("repo"):
            logger.debug("skill-sync.yml 缺少 repo 字段，跳过版本检查")
            return None
        return meta
    except Exception as e:
        logger.debug(f"读取 skill-sync.yml 失败: {e}")
        return None


def _write_meta(skill_root: str, meta: Dict[str, Any]) -> None:
    """更新 skill-sync.yml 的 last_check 字段（仅此一字段，不修改其他）。"""
    if yaml is None:
        return
    meta_path = os.path.join(skill_root, "skill-sync.yml")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.debug(f"写入 skill-sync.yml 失败: {e}")


# ============================================================================
# GitHub API
# ============================================================================

def _http_get(url: str, timeout: int = 15) -> Optional[bytes]:
    """简单 HTTP GET，失败返回 None。"""
    req = urllib.request.Request(url, headers={
        "User-Agent": "skill-sync-check/1.0",
        "Accept": "application/vnd.github+json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        # 403 = 限流，404 = 仓库不存在
        logger.debug(f"GitHub API HTTP {e.code}: {url}")
        return None
    except Exception as e:
        logger.debug(f"GitHub API 请求失败: {e}")
        return None


def _check_remote_latest(repo: str, branch: str) -> Optional[Tuple[str, str]]:
    """查 GitHub 远程最新 commit。

    Returns:
        (sha, commit_date_iso) 或 None（失败）
    """
    url = _GITHUB_API.format(repo=repo, ref=branch)
    raw = _http_get(url)
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
        sha = data.get("sha", "")
        date = data.get("commit", {}).get("committer", {}).get("date", "")
        if not sha:
            return None
        return sha, date
    except Exception as e:
        logger.debug(f"解析 GitHub API 响应失败: {e}")
        return None


# ============================================================================
# 是否跳过检查
# ============================================================================

def _should_skip(meta: Dict[str, Any], force: bool = False) -> bool:
    """是否跳过本次版本检查。"""
    if force:
        return False

    # policy=disabled
    if meta.get("update_policy", "auto") == "disabled":
        logger.debug("update_policy=disabled，跳过版本检查")
        return True

    # 检查间隔节流
    last_check = meta.get("last_check", "")
    interval = int(meta.get("check_interval_hours", _DEFAULT_INTERVAL_HOURS))
    if last_check:
        try:
            last_dt = datetime.fromisoformat(last_check.replace("Z", "+00:00"))
            now_dt = datetime.now(timezone.utc)
            elapsed_hours = (now_dt - last_dt).total_seconds() / 3600
            if elapsed_hours < interval:
                logger.debug(
                    f"距上次检查仅 {elapsed_hours:.1f}h（< {interval}h），跳过"
                )
                return True
        except Exception:
            pass  # 时间格式异常 → 不跳过，继续检查

    return False


# ============================================================================
# 主入口：只检测、不修改用户文件
# ============================================================================

def _check_one(skill_root: str, force: bool = False) -> Dict[str, Any]:
    """检查单个 skill 目录的版本状态。

    返回值：
        {
            "status": "up_to_date" | "behind" | "skipped" | "failed",
            "local_commit": str,
            "remote_commit": str,
            "message": str,           # 给用户/agent 看的提示信息
        }

    本函数 **不会修改任何用户文件**，只会更新 skill-sync.yml 的 last_check 字段。
    """
    meta = _read_meta(skill_root)
    if not meta:
        return {
            "status": "skipped",
            "message": "no skill-sync.yml or invalid metadata",
        }

    if _should_skip(meta, force=force):
        return {
            "status": "skipped",
            "message": "skip policy or recent check",
        }

    repo = meta["repo"]
    branch = meta.get("branch", "main")
    local_commit = meta.get("local_commit", "")

    # 查远程最新 commit
    remote = _check_remote_latest(repo, branch)
    if not remote:
        # 网络失败：更新 last_check 避免短时间反复重试
        meta["last_check"] = datetime.now(timezone.utc).isoformat()
        _write_meta(skill_root, meta)
        return {
            "status": "failed",
            "message": f"remote check failed for {repo}",
        }

    remote_sha, remote_date = remote

    # 更新 last_check（仅此一字段）
    meta["last_check"] = datetime.now(timezone.utc).isoformat()
    _write_meta(skill_root, meta)

    if remote_sha == local_commit:
        return {
            "status": "up_to_date",
            "local_commit": local_commit,
            "remote_commit": remote_sha,
            "message": f"{repo} 已是最新版本 ({remote_sha[:7]})",
        }

    # 检测到落后：仅输出提示，不自动更新
    msg = (
        f"[skill 版本更新提示] {repo} 有新版本可用："
        f"{local_commit[:7] if local_commit else '(未同步过)'} → {remote_sha[:7]} "
        f"(commit date: {remote_date})。"
        f"请手动执行同步："
    )
    # 判断 skill_root 是不是 git 仓库，给不同的同步命令
    if os.path.exists(os.path.join(skill_root, ".git")):
        msg += f" cd {skill_root} && git pull"
    else:
        archive_url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
        msg += f" 从 {archive_url} 下载最新版本替换当前目录（注意备份 .env、workspace 等用户数据）"

    logger.info(msg)
    return {
        "status": "behind",
        "local_commit": local_commit,
        "remote_commit": remote_sha,
        "remote_date": remote_date,
        "message": msg,
    }


def sync_all(skill_root: str, force: bool = False) -> Dict[str, Any]:
    """主入口：检查 skill 版本状态（只检测、不修改文件）。

    Args:
        skill_root: skill 根目录（含 skill-sync.yml）
        force: True 时强制检查（忽略 24h 缓存）

    Returns:
        {"status": "up_to_date" | "behind" | "skipped" | "failed", "message": str}
    """
    global _SYNC_DONE
    if _SYNC_DONE and not force:
        # 进程内已检查过，不再重复
        return {"status": "skipped", "message": "already checked in this process"}
    _SYNC_DONE = True

    try:
        return _check_one(skill_root, force=force)
    except Exception as e:
        # 任何异常都静默降级，不阻断主流程
        logger.debug(f"sync_all 异常: {e}")
        return {"status": "failed", "message": str(e)}


# ============================================================================
# 子 skill 自动部署：缺失时从 GitHub 克隆
# ============================================================================

# 子 skill 依赖声明（skill 目录 → GitHub 仓库）
_SUBSKILL_DEPENDENCIES = {
    "jingni-datafeed": "duhanjun/jingni-datafeed",
}


def ensure_skill(project_root: str, skill_name: str) -> Dict[str, Any]:
    """确保子 skill 目录存在，不存在则从 GitHub 自动克隆。

    与版本检查不同：版本检查是对已有代码做"只检测不修改"，
    而本函数是 greenfield 操作——目录不存在时从零创建，无用户数据风险。

    Args:
        project_root: jingni-trader 项目根目录
        skill_name: 子 skill 名（如 "jingni-datafeed"），
                    对应 skills/{skill_name}/ 目录

    Returns:
        {"status": "ok" | "cloned" | "skipped" | "failed", "message": str}
    """
    repo = _SUBSKILL_DEPENDENCIES.get(skill_name)
    if not repo:
        return {"status": "skipped", "message": f"未知子 skill: {skill_name}"}

    skill_dir = os.path.join(project_root, "skills", skill_name)
    branch = "main"

    # 目录已存在 → 运行正常的版本检查（只检测、不修改）
    if os.path.exists(skill_dir):
        if os.path.exists(os.path.join(skill_dir, "skill-sync.yml")):
            try:
                result = _check_one(skill_dir)
                return result
            except Exception as e:
                logger.debug(f"版本检查失败: {e}")
                return {"status": "ok", "message": f"{skill_name} 已存在（版本检查失败）"}
        return {"status": "ok", "message": f"{skill_name} 已存在"}

    # 目录不存在 → 自动从 GitHub 克隆
    clone_url = f"https://github.com/{repo}.git"
    logger.info(f"检测到 {skill_name} 未安装，正在从 GitHub 自动克隆: {clone_url}")

    try:
        import subprocess
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, clone_url, skill_dir],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.join(project_root, "skills"),
        )
        if result.returncode != 0:
            err = result.stderr.strip()[:300]
            logger.warning(f"自动克隆 {skill_name} 失败: {err}")
            return {
                "status": "failed",
                "message": f"克隆失败，请手动执行: git clone {clone_url} skills/{skill_name}",
            }

        # 克隆成功后写入 local_commit 到 skill-sync.yml
        _update_local_commit_after_clone(skill_dir, repo, branch)
        logger.info(f"{skill_name} 自动克隆成功: {skill_dir}")
        return {
            "status": "cloned",
            "message": f"{skill_name} 已从 {repo} 自动克隆到 skills/{skill_name}",
        }
    except FileNotFoundError:
        logger.debug("git 命令不可用，跳过自动克隆")
        return {
            "status": "failed",
            "message": f"git 不可用，请手动克隆: git clone {clone_url} skills/{skill_name}",
        }
    except Exception as e:
        logger.debug(f"自动克隆异常: {e}")
        return {
            "status": "failed",
            "message": f"自动克隆失败，请手动执行: git clone {clone_url} skills/{skill_name}",
        }


def _update_local_commit_after_clone(skill_dir: str, repo: str, branch: str) -> None:
    """克隆成功后，更新 skill-sync.yml 的 local_commit 为当前 HEAD SHA。"""
    meta_path = os.path.join(skill_dir, "skill-sync.yml")
    if not os.path.exists(meta_path):
        return
    remote = _check_remote_latest(repo, branch)
    if not remote:
        return
    sha, date = remote
    try:
        if yaml is None:
            return
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        meta["local_commit"] = sha
        meta["local_commit_date"] = date
        meta["last_check"] = datetime.now(timezone.utc).isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.debug(f"写入 local_commit 失败: {e}")


# ============================================================================
# CLI 入口
# ============================================================================

def _main() -> int:
    """命令行入口：python scripts/skill_sync.py [--force] [skill_root]

    仅检查版本状态，输出提示信息。不修改任何文件（除了 skill-sync.yml 的 last_check 字段）。
    """
    import argparse
    parser = argparse.ArgumentParser(description="skill 版本检查工具（只检测、不修改）")
    parser.add_argument("skill_root", nargs="?", default=None,
                        help="skill 根目录（默认为脚本所在目录的父目录）")
    parser.add_argument("--force", action="store_true",
                        help="强制检查（忽略 24h 缓存）")
    args = parser.parse_args()

    if args.skill_root:
        skill_root = os.path.abspath(args.skill_root)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        skill_root = os.path.dirname(script_dir)

    if not os.path.exists(os.path.join(skill_root, "skill-sync.yml")):
        print(f"错误: {skill_root} 下未找到 skill-sync.yml", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # CLI 调用时重置进程内标志，允许显式强制检查
    global _SYNC_DONE
    _SYNC_DONE = False

    result = sync_all(skill_root, force=args.force)
    print(f"状态: {result['status']} - {result['message']}")
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(_main())
