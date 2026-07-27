"""归档目录结构契约验证测试。

来源：原 test_e2e_archive_cli.py::TestArchiveStructure（2 用例）。

覆盖：
- 归档目录应包含 pipeline_summary.md 和每个 step 的 summary.md + artifacts/
- pipeline_summary.md 应包含完成/失败的阶段列表与任务 ID

验证 SKILL.md 第 109-123 行约定的归档目录结构契约。
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestArchiveStructure:
    """验证运行后归档目录结构符合 SKILL.md 约定。"""

    def _run_pipeline(self, work_dir):
        """复用最小回测链路跑一次 pipeline"""
        import numpy as np
        import pandas as pd

        codes = ["000001.SZ", "600000.SH"]
        frames = []
        rng = np.random.RandomState(20240101)
        for code in codes:
            dates = pd.bdate_range("2024-01-01", "2024-06-30")
            n = len(dates)
            base = rng.uniform(8, 20)
            closes = base * (1 + np.cumsum(rng.normal(0, 0.01, n)))
            opens = closes * (1 + rng.normal(0, 0.002, n))
            highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n)))
            lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n)))
            vol = rng.randint(1_000_000, 10_000_000, n)
            frames.append(pd.DataFrame({
                "code": code, "date": dates,
                "open": opens.round(2), "high": highs.round(2),
                "low": lows.round(2), "close": closes.round(2),
                "volume": vol,
            }))
        external_data = pd.concat(frames, ignore_index=True)

        import engine
        intent = "获取近3年A股数据做一个反转因子选股回测并生成绩效报告"
        master = engine.MasterEngine()
        ctx = master.parse_intent(intent)
        ctx.target_stages = ["DATA", "FACTOR", "BACKTEST", "REPORT"]
        ctx.stock_pool = ["000001.SZ", "600000.SH"]
        ctx.start_date = "2024-01-01"
        ctx.end_date = "2024-06-30"
        ctx.external_data = {"daily": external_data, "source": "e2e-test"}
        return master.run_pipeline(ctx=ctx)

    def test_archive_dir_structure(self, tmp_path, monkeypatch):
        """归档目录应包含 pipeline_summary.md 和每个 step 的 summary.md + artifacts/"""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")

        results = self._run_pipeline(str(work_dir))
        assert results["success"] is True

        archive_dir = results["archive_dir"]
        assert os.path.isdir(archive_dir)

        # 顶层 pipeline_summary.md 必须存在
        pipeline_summary = os.path.join(archive_dir, "pipeline_summary.md")
        assert os.path.isfile(pipeline_summary), f"pipeline_summary.md 缺失: {pipeline_summary}"

        # 4 个步骤子目录都要存在
        for i, stage in enumerate(results["completed_stages"], 1):
            step_dir = os.path.join(archive_dir, f"step_{i}_{stage}")
            assert os.path.isdir(step_dir), f"步骤目录缺失: {step_dir}"
            assert os.path.isfile(os.path.join(step_dir, "summary.md")), \
                f"step summary.md 缺失: {step_dir}/summary.md"
            assert os.path.isdir(os.path.join(step_dir, "artifacts")), \
                f"artifacts/ 子目录缺失: {step_dir}/artifacts"

    def test_pipeline_summary_contains_stages(self, tmp_path, monkeypatch):
        """pipeline_summary.md 应包含完成/失败的阶段列表"""
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")

        results = self._run_pipeline(str(work_dir))
        archive_dir = results["archive_dir"]
        with open(os.path.join(archive_dir, "pipeline_summary.md"), "r", encoding="utf-8") as f:
            content = f.read()

        # 至少应包含任务ID 与用户意图
        assert results["context"]["task_id"] in content
        assert "DATA" in content
        assert "FACTOR" in content
        assert "BACKTEST" in content
        assert "REPORT" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
