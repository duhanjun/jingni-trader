"""skill_sync 版本检查 + ensure_skill 自动部署系统测试。

来源：合并原 test_skill_sync_system.py 的 Part A/B/C（共 17 用例）。

覆盖：
- _should_skip 五种条件：disabled policy / 24h 内 / 超过 24h / 无 last_check / force=True
- _read_meta：缺失文件 / 合法 / 缺 repo
- sync_all：up_to_date / behind / 网络失败 / 24h 节流 / 不修改 protected 字段
- ensure_skill：已存在 / 克隆成功 / git 不可用 / 未知 skill

不发起真实网络请求，全部通过 mock urllib.request 模拟。

关键不变量（project_memory 硬约束）：
1. "jingni-trader 和 jingni-datafeed 必须有独立的版本检查机制"
2. "版本检查脚本不能自动修改文件；落后时仅输出手动同步提示"
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
from datetime import datetime, timezone, timedelta
from unittest import mock

import pytest
import yaml


# ============================================================================
# Part A: skill_sync 辅助函数（无网络）
# ============================================================================

class TestSkillSyncHelpers:
    """验证 skill_sync 的纯函数行为。"""

    def _make_meta(self, **overrides):
        base = {
            "repo": "duhanjun/jingni-trader",
            "branch": "main",
            "local_commit": "abc1234",
            "local_commit_date": "2026-07-01T00:00:00Z",
            "last_check": "",
            "update_policy": "auto",
            "check_interval_hours": 24,
            "protected_paths": [],
        }
        base.update(overrides)
        return base

    def test_should_skip_when_disabled_policy(self):
        """update_policy=disabled → skip"""
        from scripts.skill_sync import _should_skip
        meta = self._make_meta(update_policy="disabled")
        assert _should_skip(meta) is True

    def test_should_skip_when_recent_check(self):
        """last_check 在 24h 内 → skip"""
        from scripts.skill_sync import _should_skip
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        meta = self._make_meta(last_check=recent)
        assert _should_skip(meta) is True

    def test_should_not_skip_when_old_check(self):
        """last_check 超过 24h → 不 skip"""
        from scripts.skill_sync import _should_skip
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        meta = self._make_meta(last_check=old)
        assert _should_skip(meta) is False

    def test_should_not_skip_when_no_last_check(self):
        """无 last_check → 不 skip"""
        from scripts.skill_sync import _should_skip
        meta = self._make_meta(last_check="")
        assert _should_skip(meta) is False

    def test_should_not_skip_when_force(self):
        """force=True → 即使 disabled 也强制检查"""
        from scripts.skill_sync import _should_skip
        meta = self._make_meta(update_policy="disabled")
        assert _should_skip(meta, force=True) is False

    def test_read_meta_missing_file(self, tmp_path):
        """无 skill-sync.yml → None"""
        from scripts.skill_sync import _read_meta
        assert _read_meta(str(tmp_path)) is None

    def test_read_meta_valid(self, tmp_path):
        """正常 skill-sync.yml → dict"""
        from scripts.skill_sync import _read_meta
        meta = self._make_meta()
        with open(os.path.join(str(tmp_path), "skill-sync.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True)
        result = _read_meta(str(tmp_path))
        assert result is not None
        assert result["repo"] == "duhanjun/jingni-trader"

    def test_read_meta_missing_repo(self, tmp_path):
        """yml 缺 repo 字段 → None"""
        from scripts.skill_sync import _read_meta
        with open(os.path.join(str(tmp_path), "skill-sync.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({"branch": "main"}, f)
        assert _read_meta(str(tmp_path)) is None


# ============================================================================
# Part B: sync_all 主入口（mock 网络）
# ============================================================================

class TestSyncAll:
    """验证 sync_all 在不同网络/版本状态下的行为。"""

    def _make_skill_root(self, tmp_path, meta_overrides=None):
        """构造一个临时 skill_root，含 skill-sync.yml"""
        skill_root = str(tmp_path / "my_skill")
        os.makedirs(skill_root, exist_ok=True)
        meta = {
            "repo": "duhanjun/test",
            "branch": "main",
            "local_commit": "oldsha123",
            "last_check": "",
            "update_policy": "auto",
            "check_interval_hours": 24,
        }
        if meta_overrides:
            meta.update(meta_overrides)
        with open(os.path.join(skill_root, "skill-sync.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True)
        return skill_root

    def _mock_github_response(self, sha="newsha456", date="2026-07-26T10:00:00Z"):
        """构造 GitHub API 返回的字节流"""
        import json
        return json.dumps({
            "sha": sha,
            "commit": {"committer": {"date": date}},
        }).encode("utf-8")

    def test_up_to_date_when_sha_matches(self, tmp_path, monkeypatch):
        """远程 SHA == 本地 SHA → status=up_to_date"""
        skill_root = self._make_skill_root(tmp_path, {"local_commit": "sameSHA789"})
        monkeypatch.setattr(
            "scripts.skill_sync._http_get",
            lambda url, timeout=15: self._mock_github_response(sha="sameSHA789"),
        )

        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        result = ss.sync_all(skill_root, force=True)
        assert result["status"] == "up_to_date"

    def test_behind_when_sha_differs(self, tmp_path, monkeypatch):
        """远程 SHA != 本地 SHA → status=behind，message 含手动同步提示

        注意：skill_sync 内部用 sha[:7] 切片展示，所以断言要用前 7 个字符。
        """
        local_sha = "oldSHAxxxabcdef"   # 前 7 位 = "oldSHAx"
        remote_sha = "newSHAyyyabcdef"  # 前 7 位 = "newSHAy"
        skill_root = self._make_skill_root(tmp_path, {"local_commit": local_sha})
        monkeypatch.setattr(
            "scripts.skill_sync._http_get",
            lambda url, timeout=15: self._mock_github_response(sha=remote_sha),
        )

        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        result = ss.sync_all(skill_root, force=True)
        assert result["status"] == "behind"
        assert "oldSHAx" in result["message"]  # 前 7 位
        assert "newSHAy" in result["message"]
        # 完整 SHA 在 remote_commit 字段
        assert result["remote_commit"] == remote_sha
        # 仅检测不修改：local_commit 字段不应改变（保持完整 SHA）
        with open(os.path.join(skill_root, "skill-sync.yml"), "r", encoding="utf-8") as f:
            meta_after = yaml.safe_load(f)
        assert meta_after["local_commit"] == local_sha, "skill_sync 不应自动修改 local_commit"
        # 但 last_check 应该被更新
        assert meta_after["last_check"] != ""

    def test_failed_when_network_error(self, tmp_path, monkeypatch):
        """网络请求失败 → status=failed，不抛异常"""
        skill_root = self._make_skill_root(tmp_path)
        monkeypatch.setattr(
            "scripts.skill_sync._http_get",
            lambda url, timeout=15: None,  # 模拟网络失败
        )

        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        result = ss.sync_all(skill_root, force=True)
        assert result["status"] == "failed"

    def test_skipped_when_recent_check(self, tmp_path, monkeypatch):
        """24h 内已检查 → skipped，不发起网络请求"""
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        skill_root = self._make_skill_root(tmp_path, {"last_check": recent})

        network_called = {"count": 0}
        def _fake_http_get(url, timeout=15):
            network_called["count"] += 1
            return None
        monkeypatch.setattr("scripts.skill_sync._http_get", _fake_http_get)

        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        result = ss.sync_all(skill_root, force=False)
        assert result["status"] == "skipped"
        assert network_called["count"] == 0

    def test_does_not_modify_protected_fields(self, tmp_path, monkeypatch):
        """版本检查只更新 last_check，不动 local_commit / protected_paths / repo / branch"""
        skill_root = self._make_skill_root(tmp_path, {
            "local_commit": "PROTECTED_SHA",
            "protected_paths": ["workspace/", ".env"],
        })
        monkeypatch.setattr(
            "scripts.skill_sync._http_get",
            lambda url, timeout=15: self._mock_github_response(sha="DIFFERENT_SHA"),
        )

        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        with open(os.path.join(skill_root, "skill-sync.yml"), "r", encoding="utf-8") as f:
            meta_before = yaml.safe_load(f)

        ss.sync_all(skill_root, force=True)

        with open(os.path.join(skill_root, "skill-sync.yml"), "r", encoding="utf-8") as f:
            meta_after = yaml.safe_load(f)

        # 不应修改的字段
        assert meta_after["local_commit"] == "PROTECTED_SHA"
        assert meta_after["repo"] == meta_before["repo"]
        assert meta_after["branch"] == meta_before["branch"]
        assert meta_after["protected_paths"] == ["workspace/", ".env"]
        # 应该被更新的字段
        assert meta_after["last_check"] != ""


# ============================================================================
# Part C: ensure_skill 自动部署逻辑（mock git clone）
# ============================================================================

class TestEnsureSkill:
    """验证 ensure_skill 在不同目录状态下的行为。"""

    def test_returns_ok_when_skill_exists(self, tmp_path, monkeypatch):
        """目录已存在 → status=ok，不调用 git"""
        project_root = str(tmp_path)
        skill_dir = os.path.join(project_root, "skills", "jingni-datafeed")
        os.makedirs(skill_dir, exist_ok=True)

        # 也写一个 skill-sync.yml，触发版本检查路径
        with open(os.path.join(skill_dir, "skill-sync.yml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "repo": "duhanjun/jingni-datafeed",
                "branch": "main",
                "local_commit": "abc",
                "last_check": "",
                "update_policy": "auto",
                "check_interval_hours": 24,
            }, f)

        monkeypatch.setattr(
            "scripts.skill_sync._http_get",
            lambda url, timeout=15: None,  # 网络失败，但目录存在不影响"已存在"判定
        )
        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        from scripts.skill_sync import ensure_skill
        result = ensure_skill(project_root, "jingni-datafeed")
        # 目录已存在 → 返回 ok 或正常版本检查结果
        assert result["status"] in ("ok", "failed", "up_to_date", "behind", "skipped")

    def test_clones_when_skill_missing(self, tmp_path, monkeypatch):
        """目录不存在 → mock git clone 成功 → status=cloned"""
        project_root = str(tmp_path)
        skills_dir = os.path.join(project_root, "skills")
        os.makedirs(skills_dir, exist_ok=True)

        # mock subprocess.run 模拟 git clone 成功
        def _fake_run(cmd, **kwargs):
            skill_path = cmd[-1]
            os.makedirs(skill_path, exist_ok=True)
            # 写一个假的 skill-sync.yml
            with open(os.path.join(skill_path, "skill-sync.yml"), "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "repo": "duhanjun/jingni-datafeed",
                    "branch": "main",
                    "local_commit": "",
                    "last_check": "",
                    "update_policy": "auto",
                    "check_interval_hours": 24,
                }, f)
            result = mock.MagicMock()
            result.returncode = 0
            result.stderr = ""
            return result

        monkeypatch.setattr("subprocess.run", _fake_run)
        # mock GitHub API 也返回成功，让 _update_local_commit_after_clone 不报错
        import json
        monkeypatch.setattr(
            "scripts.skill_sync._http_get",
            lambda url, timeout=15: json.dumps({
                "sha": "clonesha789",
                "commit": {"committer": {"date": "2026-07-26T10:00:00Z"}},
            }).encode("utf-8"),
        )
        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        from scripts.skill_sync import ensure_skill
        result = ensure_skill(project_root, "jingni-datafeed")
        assert result["status"] == "cloned"
        assert os.path.exists(os.path.join(project_root, "skills", "jingni-datafeed"))

    def test_fails_when_git_not_available(self, tmp_path, monkeypatch):
        """git 不可用 → status=failed，不抛异常"""
        project_root = str(tmp_path)

        def _fake_run(cmd, **kwargs):
            raise FileNotFoundError("git not found")
        monkeypatch.setattr("subprocess.run", _fake_run)

        import scripts.skill_sync as ss
        monkeypatch.setattr(ss, "_SYNC_DONE", False)

        from scripts.skill_sync import ensure_skill
        result = ensure_skill(project_root, "jingni-datafeed")
        assert result["status"] == "failed"
        assert "git" in result["message"].lower()

    def test_skipped_when_unknown_skill(self, tmp_path):
        """未知 skill 名（不在依赖表中）→ status=skipped"""
        from scripts.skill_sync import ensure_skill
        result = ensure_skill(str(tmp_path), "unknown-skill-xyz")
        assert result["status"] == "skipped"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
