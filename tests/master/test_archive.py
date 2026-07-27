"""RunArchiver 归档模块测试。

来源：原 test_system_smoke.py::TestRunArchiver（1 用例）。

覆盖：
- RunArchiver 能创建归档目录、步骤目录、写入 summary.md 和 pipeline_summary.md
"""
from __future__ import annotations

import os

import pytest


class TestRunArchiver:
    def test_archiver_creates_run_dir_and_summaries(self, tmp_path):
        """RunArchiver 能创建归档目录、步骤目录、写入 summary.md 和 pipeline_summary.md"""
        from scripts.archive import RunArchiver
        archiver = RunArchiver(str(tmp_path))
        run_dir = archiver.create_run("test_task_001")
        assert os.path.isdir(run_dir)

        archiver.create_step_dir(1, "DATA")
        archiver.record_step_result("DATA", {"success": True, "artifact_path": "/tmp/data.parquet", "metadata": {"rows": 100}})
        archiver.write_step_summary("DATA", 1)
        assert os.path.isfile(os.path.join(run_dir, "step_1_DATA", "summary.md"))

        archiver.write_pipeline_summary(
            completed=["DATA"],
            failed=[],
            target_stages=["DATA"],
            user_intent="测试",
            task_id="test_task_001",
            errors=[],
        )
        assert os.path.isfile(os.path.join(run_dir, "pipeline_summary.md"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
