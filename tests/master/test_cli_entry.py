"""CLI 入口测试：`python engine.py -i ... -o ...`

来源：原 test_e2e_archive_cli.py::TestCliEntryPoint（1 用例）。

覆盖：
- CLI 命令能完整跑通最小回测链路，并把结果写入 -o 指定的 JSON 文件
- 退出码 0，输出 JSON 含 success/completed_stages/context 字段
"""
from __future__ import annotations

import os
import sys
import json
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCliEntryPoint:
    """验证 `python engine.py -i ... -o ...` 命令行入口可用。"""

    def test_cli_with_synthetic_data(self, tmp_path):
        """CLI 命令能完整跑通最小回测链路，并把结果写入 -o 指定的 JSON 文件"""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        output_json = tmp_path / "result.json"

        env = os.environ.copy()
        env["QUANT_WORK_DIR"] = str(work_dir)
        env["ALLOW_SYNTHETIC_FALLBACK"] = "true"
        env["DATA_BACKENDS"] = "websearch"
        env["LOG_LEVEL"] = "INFO"
        # 避免真实 GitHub API 调用拖慢测试
        env["JINGNI_URL"] = ""
        env["JINGNI_TOKEN"] = ""

        intent = "获取近3年A股数据做一个反转因子选股回测并生成绩效报告"
        cmd = [
            sys.executable, "engine.py",
            "-i", intent,
            "-o", str(output_json),
        ]
        result = subprocess.run(
            cmd, cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )

        # CLI 不应该崩溃
        assert result.returncode == 0, (
            f"CLI 退出码非 0: returncode={result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )

        # 输出 JSON 文件应被写入
        assert output_json.exists(), f"输出文件未生成: {output_json}"
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "success" in data
        assert "completed_stages" in data
        assert "context" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
