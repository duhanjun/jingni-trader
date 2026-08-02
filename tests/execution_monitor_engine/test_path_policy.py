"""P0-4 Frozen Core 路径策略保护 L2 单元测试

覆盖：
1. validate_changed_paths 合规/违规/forbidden 优先级/绝对路径
2. GitChangeTracker pre_snapshot/post_diff
3. guard_llm_write 触碰 frozen core raise
4. post_run_guard 退出兜底
5. audit_path_violation 审计日志

测试用例（PRD P0-4.6 要求 5 个）：
- 合规路径返回空
- forbidden 优先于 allowed
- 绝对路径/空路径/.. 违规
- git status 解析
- guard_llm_write 触碰 frozen core raise
"""
from __future__ import annotations

import json
import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest


# ============================================================================
# 模块加载工具
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXEC_ENGINE_DIR = os.path.join(ROOT, "skills", "execution-monitor-engine")
SCRIPTS_DIR = os.path.join(EXEC_ENGINE_DIR, "scripts")


def _load_path_policy_module():
    """显式加载 execution-monitor-engine/scripts/path_policy.py 为独立模块。"""
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    init_py = os.path.join(SCRIPTS_DIR, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[SCRIPTS_DIR],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    try:
        policy_path = os.path.join(SCRIPTS_DIR, "path_policy.py")
        spec = ilu.spec_from_file_location("scripts.path_policy", policy_path)
        mod = ilu.module_from_spec(spec)
        sys.modules["scripts.path_policy"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


# ============================================================================
# 单元测试：validate_changed_paths
# ============================================================================

@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestValidateChangedPaths:
    """路径校验函数测试"""

    def test_compliant_paths_return_empty(self):
        """合规路径返回空列表"""
        mod = _load_path_policy_module()
        # 普通文件不在 frozen core，且 allowed 为 None（不限制白名单）
        violations = mod.validate_changed_paths(
            changed=["reports-engine/output.html", "workspace/data/test.json"],
            allowed=None,
            forbidden=[],  # 空 forbidden
        )
        assert violations == []

    def test_empty_path_is_violation(self):
        """空路径违规"""
        mod = _load_path_policy_module()
        violations = mod.validate_changed_paths(
            changed=["", "  ", "valid.txt"],
            allowed=None,
            forbidden=[],
        )
        assert "" in violations
        assert "  " in violations
        assert "valid.txt" not in violations

    def test_absolute_path_is_violation(self):
        """绝对路径违规"""
        mod = _load_path_policy_module()
        violations = mod.validate_changed_paths(
            changed=["/etc/passwd", "C:\\Windows\\system32", "valid.txt"],
            allowed=None,
            forbidden=[],
        )
        assert "/etc/passwd" in violations
        assert "C:\\Windows\\system32" in violations

    def test_parent_dir_reference_is_violation(self):
        """含 ../ 的路径违规"""
        mod = _load_path_policy_module()
        violations = mod.validate_changed_paths(
            changed=["../etc/passwd", "dir/../other", "valid.txt"],
            allowed=None,
            forbidden=[],
        )
        assert "../etc/passwd" in violations
        assert "dir/../other" in violations

    def test_forbidden_priority_over_allowed(self):
        """forbidden 优先于 allowed：即使匹配 allowed，匹配 forbidden 也违规"""
        mod = _load_path_policy_module()
        violations = mod.validate_changed_paths(
            changed=["engine.py"],  # 同时匹配 allowed 和 forbidden
            allowed=["engine.py", "*.md"],
            forbidden=["engine.py"],
        )
        assert "engine.py" in violations

    def test_allowed_whitelist_blocks_non_matching(self):
        """allowed 非空时，不匹配 allowed 的路径违规"""
        mod = _load_path_policy_module()
        violations = mod.validate_changed_paths(
            changed=["reports/output.html", "workspace/data.json"],
            allowed=["reports/**"],  # 只允许 reports 目录
            forbidden=[],
        )
        assert "reports/output.html" not in violations
        assert "workspace/data.json" in violations

    def test_frozen_paths_default_6_items(self):
        """默认 FROZEN_PATHS 有 6 项"""
        mod = _load_path_policy_module()
        assert len(mod.FROZEN_PATHS) == 6
        assert "engine.py" in mod.FROZEN_PATHS

    def test_env_extra_appends_to_frozen(self, monkeypatch):
        """QUANT_FROZEN_PATHS_EXTRA 追加 frozen 路径"""
        monkeypatch.setenv("QUANT_FROZEN_PATHS_EXTRA", os.pathsep.join(["custom/secret.py"]))
        mod = _load_path_policy_module()
        frozen = mod.get_frozen_paths()
        assert "custom/secret.py" in frozen
        assert len(frozen) == 7  # 6 + 1


# ============================================================================
# 单元测试：GitChangeTracker
# ============================================================================

@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestGitChangeTracker:
    """GitChangeTracker 测试"""

    def test_post_diff_returns_new_files_only(self, tmp_path):
        """post_diff 只返回 pre_snapshot 之后的新增/修改文件"""
        mod = _load_path_policy_module()
        # 初始化 git 仓库
        os.system(f'cd "{tmp_path}" && git init -q && git config user.email t@t.com && git config user.name t')
        # 创建并提交初始文件
        (tmp_path / "committed.txt").write_text("initial")
        os.system(f'cd "{tmp_path}" && git add . && git commit -q -m init')

        tracker = mod.GitChangeTracker(repo_root=str(tmp_path))
        tracker.pre_snapshot()

        # pre 之后新增文件
        (tmp_path / "new_file.txt").write_text("new")
        diff = tracker.post_diff()
        assert "new_file.txt" in diff

    def test_post_diff_returns_empty_when_no_change(self, tmp_path):
        """无变更时 post_diff 返回空"""
        mod = _load_path_policy_module()
        os.system(f'cd "{tmp_path}" && git init -q && git config user.email t@t.com && git config user.name t')
        (tmp_path / "f.txt").write_text("x")
        os.system(f'cd "{tmp_path}" && git add . && git commit -q -m init')

        tracker = mod.GitChangeTracker(repo_root=str(tmp_path))
        tracker.pre_snapshot()
        diff = tracker.post_diff()
        assert diff == []

    def test_git_status_parse_handles_rename(self, tmp_path):
        """git status 解析能处理重命名"""
        mod = _load_path_policy_module()
        os.system(f'cd "{tmp_path}" && git init -q && git config user.email t@t.com && git config user.name t')
        (tmp_path / "old_name.txt").write_text("content")
        os.system(f'cd "{tmp_path}" && git add . && git commit -q -m init')
        # 重命名
        os.system(f'cd "{tmp_path}" && git mv old_name.txt new_name.txt')

        tracker = mod.GitChangeTracker(repo_root=str(tmp_path))
        status = tracker._git_status_porcelain()
        # 重命名后 new_name.txt 应出现在 status 中
        assert "new_name.txt" in status


# ============================================================================
# 单元测试：guard_llm_write
# ============================================================================

@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestGuardLlmWrite:
    """guard_llm_write 前置校验测试"""

    def test_raises_when_touching_frozen_core(self, tmp_path, monkeypatch):
        """触碰 frozen core → raise PathViolationError"""
        mod = _load_path_policy_module()
        # 审计目录指向临时目录
        monkeypatch.chdir(tmp_path)
        with pytest.raises(mod.PathViolationError) as exc_info:
            mod.guard_llm_write(
                changed_paths=["engine.py"],
                violator="reports-engine",
            )
        assert "engine.py" in str(exc_info.value)

    def test_returns_other_violations_without_raise(self, tmp_path, monkeypatch):
        """非 frozen 违规（如绝对路径）→ 返回违规列表，不 raise"""
        mod = _load_path_policy_module()
        monkeypatch.chdir(tmp_path)
        violations = mod.guard_llm_write(
            changed_paths=["/etc/passwd"],
            violator="reports-engine",
        )
        assert "/etc/passwd" in violations

    def test_returns_empty_when_compliant(self, tmp_path, monkeypatch):
        """合规路径 → 返回空列表"""
        mod = _load_path_policy_module()
        monkeypatch.chdir(tmp_path)
        violations = mod.guard_llm_write(
            changed_paths=["reports/output.html"],
            violator="reports-engine",
        )
        assert violations == []


# ============================================================================
# 单元测试：audit_path_violation
# ============================================================================

@pytest.mark.skill_execution_monitor_engine
@pytest.mark.unit
class TestAuditPathViolation:
    """审计日志写入测试"""

    def test_audit_log_written_to_jsonl(self, tmp_path):
        """审计日志写入 path_violations.jsonl"""
        mod = _load_path_policy_module()
        audit_dir = str(tmp_path / "audit")
        mod.audit_path_violation(
            violator="reports-engine",
            violations=["engine.py"],
            rejected=True,
            audit_dir=audit_dir,
        )
        log_path = os.path.join(audit_dir, "path_violations.jsonl")
        assert os.path.exists(log_path)
        with open(log_path, "r", encoding="utf-8") as f:
            entry = json.loads(f.readline())
        assert entry["violator"] == "reports-engine"
        assert entry["violations"] == ["engine.py"]
        assert entry["rejected"] is True
        assert "timestamp" in entry
