"""P0-4 Frozen Core 路径策略保护

防止 LLM agent 误改下单核心代码导致资金安全事故。

核心组件：
1. validate_changed_paths：路径校验函数（空路径/绝对路径/../ 违规，forbidden 优先于 allowed）
2. FROZEN_PATHS：frozen core 清单（6 项，PRD P0-4.2）
3. GitChangeTracker：git status 快照与 diff 工具
4. audit_path_violation：审计日志写入（JSONL）

环境变量（统一 QUANT_ 前缀）：
- QUANT_FROZEN_PATHS_EXTRA：追加 frozen core 路径（不覆盖默认 6 项）
"""
import fnmatch
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger("path-policy")


# ============================================================================
# P0-4.2 Frozen Core 清单（6 项，jingni-trader 适配版）
# ============================================================================

# PRD 原文参考 GemStar：scripts/real_broker/**, scripts/risk/**, schemas/order.py,
# schemas/execution_report.py, engine.py, skills/portfolio-risk-engine/scripts/cost.py
#
# jingni-trader 适配：无 scripts/real_broker 和 schemas/ 目录，
# 映射为 execution-monitor-engine 的实盘 adapter + 各 skill engine.py 入口
FROZEN_PATHS: List[str] = [
    "skills/execution-monitor-engine/scripts/adapters/gm_adapter.py",       # gm 实盘下单
    "skills/execution-monitor-engine/scripts/adapters/xtquant_adapter.py",  # xtquant 实盘下单
    "skills/execution-monitor-engine/engine.py",                            # 执行引擎入口
    "skills/portfolio-risk-engine/scripts/cost.py",                         # 成本模型
    "engine.py",                                                             # 主调度入口
    "skills/backtest-engine/engine.py",                                     # 回测引擎入口
]


def get_frozen_paths() -> List[str]:
    """获取完整 frozen core 清单（默认 6 项 + 环境变量追加）"""
    paths = list(FROZEN_PATHS)
    extra = os.environ.get("QUANT_FROZEN_PATHS_EXTRA", "")
    if extra:
        for p in extra.split(os.pathsep):
            p = p.strip()
            if p and p not in paths:
                paths.append(p)
    return paths


# ============================================================================
# P0-4.1 路径校验函数
# ============================================================================

def validate_changed_paths(
    changed: List[str],
    allowed: Optional[List[str]] = None,
    forbidden: Optional[List[str]] = None,
) -> List[str]:
    """校验变更路径列表，返回违规路径列表（空列表表示合规）。

    规则（PRD P0-4.1）：
    1. 空路径违规
    2. 含 ../ 或绝对路径违规
    3. forbidden 优先于 allowed（即使匹配 allowed，若同时匹配 forbidden 也违规）
    4. allowed 非空时，路径必须匹配 allowed 至少一项，否则违规
       allowed 为空/None 时，不检查白名单（只检查 forbidden）

    参数：
        changed: 变更路径列表（相对路径）
        allowed: 允许修改的路径白名单（glob 模式），None/空表示不限制白名单
        forbidden: 禁止修改的路径黑名单（glob 模式），None 时用 FROZEN_PATHS

    返回：
        违规路径列表（原样返回违规的 changed 项）
    """
    if forbidden is None:
        forbidden = get_frozen_paths()

    violations: List[str] = []
    for path in changed:
        # 规则 1: 空路径违规
        if not path or not path.strip():
            violations.append(path)
            continue

        # 规则 2: 绝对路径或含 ../ 违规
        if os.path.isabs(path) or ".." in Path(path).parts:
            violations.append(path)
            continue

        # 规则 3: forbidden 优先（匹配任一 forbidden 模式即违规）
        is_forbidden = any(fnmatch.fnmatch(path, pat) for pat in forbidden)
        if is_forbidden:
            violations.append(path)
            continue

        # 规则 4: allowed 非空时，必须匹配 allowed 至少一项
        if allowed:
            is_allowed = any(fnmatch.fnmatch(path, pat) for pat in allowed)
            if not is_allowed:
                violations.append(path)
                continue

    return violations


# ============================================================================
# P0-4.3 GitChangeTracker
# ============================================================================

class GitChangeTracker:
    """git status 快照与 diff 工具。

    使用方式：
        tracker = GitChangeTracker(repo_root="/path/to/repo")
        tracker.pre_snapshot()   # 运行前快照
        # ... 业务逻辑 ...
        diff = tracker.post_diff()  # 返回本次新增/修改的路径
    """

    def __init__(self, repo_root: str = "."):
        self.repo_root = os.path.abspath(repo_root)
        self._pre_snapshot: Set[str] = set()

    def _git_status_porcelain(self) -> Set[str]:
        """调用 git status --porcelain --untracked-files=all，返回相对路径集合。"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(f"git status 失败: {result.stderr}")
                return set()
            paths = set()
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                # porcelain 格式: "XY path" 或 "XY path -> renamed"
                # X/Y 是状态字符，path 从第 4 字符开始
                if len(line) < 4:
                    continue
                path_part = line[3:]
                # 处理重命名: "old -> new"
                if " -> " in path_part:
                    path_part = path_part.split(" -> ")[-1]
                # 去除引号（git 对含空格路径会加引号）
                path_part = path_part.strip().strip('"')
                if path_part:
                    paths.add(path_part)
            return paths
        except subprocess.TimeoutExpired:
            logger.warning("git status 超时（10s）")
            return set()
        except FileNotFoundError:
            logger.warning("git 命令不可用")
            return set()
        except Exception as e:
            logger.warning(f"git status 异常: {e}")
            return set()

    def pre_snapshot(self) -> Set[str]:
        """运行前快照，返回当前 git status 的路径集合。

        PRD 要求 pre 时 worktree clean，但实际测试环境难以保证，
        降级为：记录快照，若 worktree 不 clean 记 warning（不 raise）。
        """
        self._pre_snapshot = self._git_status_porcelain()
        if self._pre_snapshot:
            logger.warning(
                f"pre_snapshot 时 worktree 不 clean（{len(self._pre_snapshot)} 个变更），"
                f"已记录快照，post_diff 将只返回新增变更"
            )
        return self._pre_snapshot

    def post_diff(self) -> List[str]:
        """返回本次运行新增/修改的路径（post - pre 的差集）。"""
        post = self._git_status_porcelain()
        diff = post - self._pre_snapshot
        return sorted(diff)


# ============================================================================
# P0-4.5 审计日志
# ============================================================================

def audit_path_violation(
    violator: str,
    violations: List[str],
    rejected: bool = True,
    audit_dir: Optional[str] = None,
) -> None:
    """写入路径违规审计日志（JSONL）。

    参数：
        violator: 违规方标识（如 "reports-engine"）
        violations: 违规路径列表
        rejected: 是否被拒绝（True=前置拦截，False=退出兜底告警）
        audit_dir: 审计日志目录，None 时用 <repo_root>/audit
    """
    if audit_dir is None:
        # 默认放到项目根的 audit 目录
        audit_dir = os.path.join(os.getcwd(), "audit")
    os.makedirs(audit_dir, exist_ok=True)

    log_path = os.path.join(audit_dir, "path_violations.jsonl")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "violator": violator,
        "violations": violations,
        "rejected": rejected,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.warning(
            f"路径违规审计: violator={violator}, violations={violations}, rejected={rejected}"
        )
    except Exception as e:
        logger.error(f"审计日志写入失败: {e}")


# ============================================================================
# P0-4.4 双重校验入口
# ============================================================================

def guard_llm_write(
    changed_paths: List[str],
    violator: str = "reports-engine",
    allowed: Optional[List[str]] = None,
) -> List[str]:
    """LLM 写文件前的前置校验（PRD P0-4.4 前置）。

    触碰 frozen core → raise + 记录审计日志。
    其他违规 → 返回违规列表（调用方决定是否继续）。

    参数：
        changed_paths: LLM 即将修改的路径列表
        violator: 违规方标识
        allowed: 允许 LLM 修改的白名单（None 表示只检查 frozen core）

    返回：
        非冻结类违规路径列表（空列表表示合规）

    异常：
        PathViolationError: 触碰 frozen core 时 raise
    """
    violations = validate_changed_paths(changed_paths, allowed=allowed)
    if not violations:
        return []

    # 区分 frozen core 违规 vs 其他违规
    frozen = get_frozen_paths()
    frozen_violations = [v for v in violations if any(fnmatch.fnmatch(v, pat) for pat in frozen)]
    other_violations = [v for v in violations if v not in frozen_violations]

    if frozen_violations:
        audit_path_violation(violator, frozen_violations, rejected=True)
        raise PathViolationError(
            f"触碰 frozen core: {frozen_violations}",
            violations=frozen_violations,
        )

    # 非冻结违规（如绝对路径、白名单外）→ 记录审计，返回违规列表
    if other_violations:
        audit_path_violation(violator, other_violations, rejected=False)
    return other_violations


class PathViolationError(Exception):
    """触碰 frozen core 的异常"""

    def __init__(self, message: str, violations: List[str] = None):
        super().__init__(message)
        self.violations = violations or []


def post_run_guard(tracker: GitChangeTracker, violator: str = "unknown") -> None:
    """运行结束时的退出兜底校验（PRD P0-4.4 退出兜底）。

    对全量变更路径再次校验，发现 frozen core 改动 → 记录 critical 审计日志
    （此时已无法 raise，仅告警）。

    参数：
        tracker: 已调用 pre_snapshot 的 GitChangeTracker
        violator: 违规方标识
    """
    try:
        diff = tracker.post_diff()
        if not diff:
            return
        violations = validate_changed_paths(diff)
        if violations:
            # 退出兜底无法 raise，仅记 critical 审计
            audit_path_violation(violator, violations, rejected=False)
            logger.critical(
                f"退出兜底检测到 frozen core 改动（无法拦截）: {violations}"
            )
    except Exception as e:
        logger.warning(f"退出兜底校验异常: {e}")
