"""P1-3 工件版本化 + sha256 sidecar manifest 测试

覆盖（PRD P1-3.8）6 个用例：
1. compute_sha256 计算正确性 + 大文件流式
2. write_artifact / read_artifact 往返 + manifest 结构
3. inputs 血缘追踪
4. read_artifact sha256 不匹配 raise
5. generate_run_manifest 生成 run_manifest.json
6. replay_check 对比两次 run

兼容性：
- 旧产物（无 manifest）读取时只 warning，不 raise
"""
from __future__ import annotations

import json
import os
import shutil

import pytest


# ============================================================================
# 用例 1: compute_sha256
# ============================================================================

class TestComputeSha256:
    def test_sha256_correct_for_known_content(self, tmp_path):
        """sha256 计算正确性（对已知内容用 hashlib 直接验证）"""
        import hashlib
        from scripts.artifact_store import compute_sha256

        # 已知内容的 sha256
        content = b"hello jingni-trader"
        p = tmp_path / "sample.txt"
        p.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        actual = compute_sha256(str(p))
        assert actual == expected
        assert len(actual) == 64  # sha256 hex 摘要长度

    def test_sha256_chunked_for_large_file(self, tmp_path):
        """大文件按 chunk 流式计算（>1MB 跨多个 chunk 边界）"""
        import hashlib
        from scripts.artifact_store import compute_sha256

        # 3MB 内容（跨 3 个 1MB chunk）
        content = os.urandom(3 * 1024 * 1024)
        p = tmp_path / "large.bin"
        p.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        actual = compute_sha256(str(p))
        assert actual == expected

    def test_sha256_missing_file_raises(self, tmp_path):
        """文件不存在 raise FileNotFoundError"""
        from scripts.artifact_store import compute_sha256
        with pytest.raises(FileNotFoundError):
            compute_sha256(str(tmp_path / "nonexistent.txt"))

    def test_sha256_directory_raises(self, tmp_path):
        """路径是目录 raise IsADirectoryError"""
        from scripts.artifact_store import compute_sha256
        sub = tmp_path / "subdir"
        sub.mkdir()
        with pytest.raises(IsADirectoryError):
            compute_sha256(str(sub))


# ============================================================================
# 用例 2: write_artifact / read_artifact 往返 + manifest 结构
# ============================================================================

class TestWriteReadArtifact:
    def test_roundtrip_data_and_manifest(self, tmp_path):
        """write_artifact 写产物 + manifest，read_artifact 读回并校验 sha256"""
        from scripts.artifact_store import write_artifact, read_artifact, MANIFEST_SUFFIX

        data = {"metrics": {"sharpe": 1.2}, "trades": 100}
        output_dir = str(tmp_path / "out")

        path = write_artifact("backtest_result", data, output_dir)
        assert os.path.isfile(path)

        # manifest sidecar 存在（路径为 <output_dir>/<name>.manifest.json）
        manifest_path = os.path.join(output_dir, f"backtest_result{MANIFEST_SUFFIX}")
        assert os.path.isfile(manifest_path)

        # manifest 结构校验（PRD P1-3.2）
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["name"] == "backtest_result"
        assert manifest["version"] == "V1"
        assert "sha256" in manifest and len(manifest["sha256"]) == 64
        assert "created_at" in manifest
        assert manifest["inputs"] == []  # 无 inputs

        # read_artifact 往返
        read_data, read_manifest = read_artifact("backtest_result", output_dir)
        assert read_data == data
        assert read_manifest["sha256"] == manifest["sha256"]

    def test_read_artifact_without_manifest_warns(self, tmp_path):
        """旧产物（无 manifest）读取时只 warning，返回空 manifest dict"""
        from scripts.artifact_store import read_artifact

        # 手动写一个没有 manifest 的产物
        output_dir = str(tmp_path / "legacy")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "legacy.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"old": True}, f)

        # 不应 raise
        data, manifest = read_artifact("legacy", output_dir)
        assert data == {"old": True}
        assert manifest == {}


# ============================================================================
# 用例 3: inputs 血缘追踪
# ============================================================================

class TestInputsLineage:
    def test_inputs_recorded_in_manifest(self, tmp_path):
        """write_artifact 把 inputs 路径的 sha256 记入 manifest.inputs"""
        from scripts.artifact_store import write_artifact, read_artifact, compute_sha256

        # 准备上游 inputs
        input_dir = str(tmp_path / "inputs")
        os.makedirs(input_dir, exist_ok=True)
        inp1 = os.path.join(input_dir, "data.json")
        inp2 = os.path.join(input_dir, "factor.json")
        with open(inp1, "w", encoding="utf-8") as f:
            json.dump({"a": 1}, f)
        with open(inp2, "w", encoding="utf-8") as f:
            json.dump({"b": 2}, f)

        # 写下游产物
        output_dir = str(tmp_path / "out")
        path = write_artifact(
            "downstream",
            {"result": "ok"},
            output_dir,
            inputs=[inp1, inp2],
        )

        # 校验 manifest 中的 inputs 血缘
        _, manifest = read_artifact("downstream", output_dir)
        assert len(manifest["inputs"]) == 2
        inp_names = [i["name"] for i in manifest["inputs"]]
        assert "data.json" in inp_names
        assert "factor.json" in inp_names
        # 每个 input 的 sha256 与实际计算一致
        for inp in manifest["inputs"]:
            inp_path = os.path.join(input_dir, inp["name"])
            assert inp["sha256"] == compute_sha256(inp_path)

    def test_inputs_missing_file_skipped(self, tmp_path):
        """inputs 中的不存在文件被跳过（sha256 留空）"""
        from scripts.artifact_store import write_artifact, read_artifact

        output_dir = str(tmp_path / "out")
        path = write_artifact(
            "downstream",
            {"x": 1},
            output_dir,
            inputs=["/nonexistent/path1.json", "/nonexistent/path2.json"],
        )

        _, manifest = read_artifact("downstream", output_dir)
        assert len(manifest["inputs"]) == 2
        for inp in manifest["inputs"]:
            assert inp["sha256"] == ""


# ============================================================================
# 用例 4: read_artifact sha256 不匹配 raise
# ============================================================================

class TestSha256Mismatch:
    def test_tampered_artifact_raises(self, tmp_path):
        """产物被篡改后 read_artifact 校验失败 raise ValueError"""
        from scripts.artifact_store import write_artifact, read_artifact

        output_dir = str(tmp_path / "out")
        write_artifact("target", {"v": 1}, output_dir)

        # 篡改产物文件
        artifact_path = os.path.join(output_dir, "target.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump({"v": 999}, f)  # 内容已变，但 manifest 里的 sha256 仍是原始值

        with pytest.raises(ValueError, match=r"sha256 不匹配"):
            read_artifact("target", output_dir)


# ============================================================================
# 用例 5: generate_run_manifest
# ============================================================================

class TestRunManifest:
    def test_generate_run_manifest_structure(self, tmp_path):
        """generate_run_manifest 落盘 run_manifest.json，结构符合 PRD P1-3.6"""
        from scripts.artifact_store import generate_run_manifest, load_run_manifest

        run_dir = str(tmp_path / "run_001")
        os.makedirs(run_dir, exist_ok=True)

        stages = [
            {
                "name": "DATA",
                "status": "success",
                "latency_sec": 1.23,
                "artifacts": [{"name": "cleaned_data.parquet", "sha256": "abc123"}],
            },
            {
                "name": "FACTOR",
                "status": "success",
                "latency_sec": 0.45,
                "artifacts": [{"name": "factor_data.parquet", "sha256": "def456"}],
            },
        ]
        inputs_sha = {"DATA": "abc123", "FACTOR": "def456"}

        manifest_path = generate_run_manifest(
            run_dir=run_dir,
            run_id="20260802_103000",
            start_at="2026-08-02T10:30:00",
            end_at="2026-08-02T10:31:00",
            stages=stages,
            inputs_sha256=inputs_sha,
        )
        assert os.path.isfile(manifest_path)
        assert manifest_path == os.path.join(run_dir, "run_manifest.json")

        manifest = load_run_manifest(run_dir)
        assert manifest["run_id"] == "20260802_103000"
        assert manifest["start_at"] == "2026-08-02T10:30:00"
        assert manifest["end_at"] == "2026-08-02T10:31:00"
        assert len(manifest["stages"]) == 2
        assert manifest["stages"][0]["name"] == "DATA"
        assert manifest["stages"][0]["artifacts"][0]["sha256"] == "abc123"
        assert manifest["inputs_sha256"] == inputs_sha
        assert manifest["version"] == "V1"

    def test_load_run_manifest_missing_raises(self, tmp_path):
        """run_manifest.json 不存在时 load_run_manifest raise FileNotFoundError"""
        from scripts.artifact_store import load_run_manifest
        with pytest.raises(FileNotFoundError):
            load_run_manifest(str(tmp_path))


# ============================================================================
# 用例 6: replay_check 对比两次 run
# ============================================================================

class TestReplayCheck:
    def test_identical_runs_no_diffs(self, tmp_path):
        """两次 run 完全一致 → inputs_identical=True, outputs_identical=True"""
        from scripts.artifact_store import generate_run_manifest
        from scripts.replay_check import compare_runs, format_report

        # 两次 run 用相同的 manifest
        stages = [{
            "name": "DATA",
            "status": "success",
            "latency_sec": 1.0,
            "artifacts": [{"name": "data.json", "sha256": "same_sha"}],
        }]
        inputs_sha = {"DATA": "same_input_sha"}

        for run_id in ("run_a", "run_b"):
            run_dir = str(tmp_path / run_id)
            os.makedirs(run_dir, exist_ok=True)
            generate_run_manifest(
                run_dir=run_dir,
                run_id=run_id,
                start_at="2026-08-02T10:00:00",
                end_at="2026-08-02T10:01:00",
                stages=stages,
                inputs_sha256=inputs_sha,
            )

        report = compare_runs(str(tmp_path / "run_a"), str(tmp_path / "run_b"))
        assert report["inputs_identical"] is True
        assert report["outputs_identical"] is True
        assert report["input_diffs"] == []
        assert report["output_diffs"] == []
        assert "确定性确认" in format_report(report)

    def test_different_outputs_detected(self, tmp_path):
        """输入相同但输出不同 → 标记为非确定性差异"""
        from scripts.artifact_store import generate_run_manifest
        from scripts.replay_check import compare_runs, format_report

        inputs_sha = {"DATA": "same_input_sha"}
        # run_a 输出 sha=aaa
        # run_b 输出 sha=bbb
        stages_a = [{
            "name": "DATA",
            "status": "success",
            "latency_sec": 1.0,
            "artifacts": [{"name": "data.json", "sha256": "aaa"}],
        }]
        stages_b = [{
            "name": "DATA",
            "status": "success",
            "latency_sec": 1.0,
            "artifacts": [{"name": "data.json", "sha256": "bbb"}],
        }]
        for run_id, stages in (("run_a", stages_a), ("run_b", stages_b)):
            run_dir = str(tmp_path / run_id)
            os.makedirs(run_dir, exist_ok=True)
            generate_run_manifest(
                run_dir=run_dir, run_id=run_id,
                start_at="2026-08-02T10:00:00", end_at="2026-08-02T10:01:00",
                stages=stages, inputs_sha256=inputs_sha,
            )

        report = compare_runs(str(tmp_path / "run_a"), str(tmp_path / "run_b"))
        assert report["inputs_identical"] is True
        assert report["outputs_identical"] is False
        assert len(report["output_diffs"]) == 1
        assert report["output_diffs"][0]["stage"] == "DATA"
        assert "非确定性" in format_report(report)

    def test_only_in_a_stages_reported(self, tmp_path):
        """只在 run_a 执行的阶段被标记到 only_in_a"""
        from scripts.artifact_store import generate_run_manifest
        from scripts.replay_check import compare_runs

        # run_a 有 DATA + FACTOR
        stages_a = [
            {"name": "DATA", "status": "success", "latency_sec": 1.0, "artifacts": []},
            {"name": "FACTOR", "status": "success", "latency_sec": 1.0, "artifacts": []},
        ]
        # run_b 只有 DATA
        stages_b = [
            {"name": "DATA", "status": "success", "latency_sec": 1.0, "artifacts": []},
        ]
        for run_id, stages in (("run_a", stages_a), ("run_b", stages_b)):
            run_dir = str(tmp_path / run_id)
            os.makedirs(run_dir, exist_ok=True)
            generate_run_manifest(
                run_dir=run_dir, run_id=run_id,
                start_at="2026-08-02T10:00:00", end_at="2026-08-02T10:01:00",
                stages=stages, inputs_sha256={},
            )

        report = compare_runs(str(tmp_path / "run_a"), str(tmp_path / "run_b"))
        assert "FACTOR" in report["only_in_a"]
        assert report["only_in_b"] == []


# ============================================================================
# 用例 7: RunArchiver 集成（sidecar manifest + run_manifest.json）
# ============================================================================

class TestRunArchiverIntegration:
    def test_save_artifact_copy_generates_sidecar_manifest(self, tmp_path):
        """RunArchiver.save_artifact_copy 复制产物后生成 sidecar manifest"""
        from scripts.archive import RunArchiver

        # 准备源产物
        src_path = str(tmp_path / "source.json")
        with open(src_path, "w", encoding="utf-8") as f:
            json.dump({"v": 1}, f)

        archiver = RunArchiver(str(tmp_path / "archive"))
        archiver.create_run("task_001")
        archiver.create_step_dir(1, "DATA")
        archiver.save_artifact_copy("DATA", src_path)

        # 验证 sidecar manifest 存在且结构正确
        step_dir = archiver.step_dirs["DATA"]
        dest_path = os.path.join(step_dir, "artifacts", "source.json")
        manifest_path = dest_path + ".manifest.json"
        assert os.path.isfile(manifest_path)

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["name"] == "source.json"
        assert manifest["version"] == "V1"
        assert len(manifest["sha256"]) == 64
        assert manifest["inputs"] == []

    def test_write_run_manifest_aggregates_stages(self, tmp_path):
        """RunArchiver.write_run_manifest 汇总各阶段状态/耗时/产物 sha256"""
        from scripts.archive import RunArchiver
        from scripts.artifact_store import load_run_manifest

        # 准备源产物
        src1 = str(tmp_path / "data.json")
        src2 = str(tmp_path / "factor.json")
        with open(src1, "w", encoding="utf-8") as f:
            json.dump({"d": 1}, f)
        with open(src2, "w", encoding="utf-8") as f:
            json.dump({"f": 2}, f)

        archiver = RunArchiver(str(tmp_path / "archive"))
        archiver.create_run("task_002")

        archiver.create_step_dir(1, "DATA")
        archiver.save_artifact_copy("DATA", src1)
        archiver.record_stage_end("DATA", "success")

        archiver.create_step_dir(2, "FACTOR")
        archiver.save_artifact_copy("FACTOR", src2)
        archiver.record_stage_end("FACTOR", "success")

        manifest_path = archiver.write_run_manifest()
        assert os.path.isfile(manifest_path)

        manifest = load_run_manifest(archiver.run_dir)
        assert manifest["run_id"] == "task_002"
        assert len(manifest["stages"]) == 2
        stage_names = [s["name"] for s in manifest["stages"]]
        assert "DATA" in stage_names
        assert "FACTOR" in stage_names
        # 每个阶段应有产物 sha256
        for stage in manifest["stages"]:
            assert stage["status"] == "success"
            assert len(stage["artifacts"]) == 1
            assert len(stage["artifacts"][0]["sha256"]) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
