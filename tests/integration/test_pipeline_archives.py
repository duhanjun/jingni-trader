"""归档目录结构契约验证测试。

来源：原 test_e2e_archive_cli.py::TestArchiveStructure（2 用例）。

覆盖：
- 归档目录应包含 pipeline_summary.md 和每个 step 的 summary.md + artifacts/
- pipeline_summary.md 应包含完成/失败的阶段列表与任务 ID
- T3-10: Alphalens 因子分析报告归档（QUANT_ALPHALENS_REPORT=1 时）
- T3-10: reports-engine 自动聚合 factor_analysis_report.html

验证 SKILL.md 第 109-123 行约定的归档目录结构契约。
"""
from __future__ import annotations

import os
import sys
import glob as _glob

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_pipeline_helper(work_dir, enable_alphalens=False, target_stages=None):
    """复用最小回测链路跑一次 pipeline（模块级共享函数）。

    参数
    ----
    work_dir:          QUANT_WORK_DIR 路径（外部已设置环境变量）
    enable_alphalens:  是否启用 QUANT_ALPHALENS_REPORT=1
    target_stages:     自定义目标阶段；默认走完整回测链路
    """
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

    # 必须在 import engine 之前设置环境变量，否则子 skill 加载时已固化
    if enable_alphalens:
        os.environ["QUANT_ALPHALENS_REPORT"] = "1"
        # 强制刷新：跳过主调度器对 FACTOR 等阶段产物的缓存检查。
        # 因为 FACTOR_DIR 等路径在 scripts/config.py 加载时固化（conftest 预加载
        # engine.py 时的 QUANT_WORK_DIR），多个测试共用同一份固化路径，
        # 前序测试遗留的 factor_data.parquet 会让主调度器跳过 factor-engine 调用，
        # 导致 alphalens_report_dir 无法生成。
        os.environ["QUANT_FORCE_REFRESH"] = "1"
    else:
        os.environ.pop("QUANT_ALPHALENS_REPORT", None)
        # 注意：不主动删除 QUANT_FORCE_REFRESH，由调用方控制（T1-12 集成测试
        # 需要强制刷新以验证 YAML 配置生效）

    import engine
    intent = "获取近3年A股数据做一个反转因子选股回测并生成绩效报告"
    master = engine.MasterEngine()
    ctx = master.parse_intent(intent)
    ctx.target_stages = target_stages or ["DATA", "FACTOR", "BACKTEST", "REPORT"]
    ctx.stock_pool = ["000001.SZ", "600000.SH"]
    ctx.start_date = "2024-01-01"
    ctx.end_date = "2024-06-30"
    ctx.external_data = {"daily": external_data, "source": "e2e-test"}
    return master.run_pipeline(ctx=ctx)


class TestArchiveStructure:
    """验证运行后归档目录结构符合 SKILL.md 约定。"""

    def _run_pipeline(self, work_dir, enable_alphalens=False, target_stages=None):
        """向后兼容包装：委托给模块级 _run_pipeline_helper。"""
        return _run_pipeline_helper(work_dir, enable_alphalens, target_stages)

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

    # ── T3-10: Alphalens 因子分析报告归档验证 ──────────────────────

    def test_alphalens_report_archived_when_enabled(self, tmp_path, monkeypatch):
        """QUANT_ALPHALENS_REPORT=1 时，FACTOR 阶段归档中应包含 alphalens 报告子目录。

        验证要点：
        1. FACTOR 步骤的 artifacts/ 下存在以 task_id 命名的子目录
        2. 该子目录中至少存在一份 *_metrics.json（alphalens 不可用时降级为方案C）
        3. alphalens-reloaded 可用时还应存在 *_report.html
        4. pipeline_summary.md 中应能查到 alphalens_report_dir 元数据记录
        """
        work_dir = tmp_path / "workspace_alphalens"
        work_dir.mkdir()
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")

        # 走分析路径（无 BACKTEST），避免 reports-engine 走 performance report 分支
        results = self._run_pipeline(
            str(work_dir),
            enable_alphalens=True,
            target_stages=["DATA", "FACTOR", "REPORT"],
        )
        assert results["success"] is True, (
            f"分析路径未成功: errors={results.get('errors')}"
        )

        archive_dir = results["archive_dir"]
        task_id = results["context"]["task_id"]

        # 诊断：打印 metadata 中的 alphalens_report_dir
        factor_meta = results["context"].get("metadata", {}).get("FACTOR", {})
        alpha_dir_in_meta = factor_meta.get("alphalens_report_dir", "<MISSING>")
        print(f"\n[DIAG] task_id={task_id}")
        print(f"[DIAG] archive_dir={archive_dir}")
        print(f"[DIAG] alphalens_report_dir in meta={alpha_dir_in_meta!r}")
        print(f"[DIAG] FACTOR metadata keys={list(factor_meta.keys())}")
        print(f"[DIAG] os.path.isdir(alpha_dir_in_meta)={os.path.isdir(alpha_dir_in_meta) if alpha_dir_in_meta != '<MISSING>' else 'N/A'}")
        print(f"[DIAG] QUANT_ALPHALENS_REPORT={os.environ.get('QUANT_ALPHALENS_REPORT', '<unset>')!r}")
        print(f"[DIAG] QUANT_WORK_DIR={os.environ.get('QUANT_WORK_DIR', '<unset>')!r}")
        # 检查 skills.factor-engine.engine 模块是否包含 _maybe_generate_alphalens_reports
        import sys as _sys
        _fe_mod = _sys.modules.get("skills.factor-engine.engine")
        if _fe_mod is not None:
            _has_alpha_fn = hasattr(_fe_mod, "_maybe_generate_alphalens_reports")
            print(f"[DIAG] skills.factor-engine.engine loaded=True, has_fn={_has_alpha_fn}")
            # 检查 run 函数源码是否包含 alphalens
            import inspect as _insp
            _src = _insp.getsource(_fe_mod.run)
            print(f"[DIAG] run() source contains 'alphalens': {'alphalens' in _src}")
        else:
            print(f"[DIAG] skills.factor-engine.engine not in sys.modules")

        # 1) 定位 FACTOR 步骤归档目录
        factor_step_dirs = [
            d for d in os.listdir(archive_dir)
            if d.startswith("step_") and "FACTOR" in d
        ]
        assert factor_step_dirs, "FACTOR 步骤归档目录不存在"
        factor_artifacts_dir = os.path.join(archive_dir, factor_step_dirs[0], "artifacts")
        assert os.path.isdir(factor_artifacts_dir)

        # 2) task_id 命名的子目录应存在（save_artifact_copy 对目录走 copytree）
        alphalens_subdir = os.path.join(factor_artifacts_dir, task_id)
        assert os.path.isdir(alphalens_subdir), (
            f"Alphalens 报告归档子目录缺失: {alphalens_subdir}，"
            f"现有内容: {os.listdir(factor_artifacts_dir)}"
        )

        # 3) 至少存在一份 *_metrics.json（alphalens 可用或方案C降级都满足）
        metrics_files = sorted(_glob.glob(os.path.join(alphalens_subdir, "*_metrics.json")))
        assert metrics_files, (
            f"Alphalens 报告目录中未找到 *_metrics.json: {alphalens_subdir}"
        )

        # 4) 至少存在一份 *_report.html
        html_files = sorted(_glob.glob(os.path.join(alphalens_subdir, "*_report.html")))
        assert html_files, (
            f"Alphalens 报告目录中未找到 *_report.html: {alphalens_subdir}"
        )

        # 5) 验证 metrics.json 内容字段（8 个必填字段）
        import json
        with open(metrics_files[0], "r", encoding="utf-8") as f:
            metrics = json.load(f)
        required_fields = {
            "factor", "top_quantile_return", "bottom_quantile_return",
            "long_short_return", "long_short_sharpe",
            "ic_mean", "ic_ir", "avg_turnover_top_quantile",
            "suggested_verdict",
        }
        missing = required_fields - set(metrics.keys())
        assert not missing, f"metrics.json 缺少字段: {missing}"

        # 6) FACTOR 步骤的 summary.md 应记录 alphalens_report_dir 元数据
        #    (archive.write_step_summary 会把 metadata 字段写入步骤小结)
        factor_summary_path = os.path.join(
            archive_dir, factor_step_dirs[0], "summary.md"
        )
        with open(factor_summary_path, "r", encoding="utf-8") as f:
            factor_summary_content = f.read()
        assert "alphalens_report_dir" in factor_summary_content, (
            "FACTOR 步骤 summary.md 未记录 alphalens_report_dir 元数据"
        )

    def test_factor_analysis_summary_report_aggregated(self, tmp_path, monkeypatch):
        """reports-engine 应自动聚合 alphalens metrics.json 生成 factor_analysis_report.html。

        验证要点：
        1. REPORT 阶段产物中包含 factor_analysis_report.html
        2. 该 HTML 文件内容应包含至少一个因子名（来自 metrics.json 的 factor 字段）
        3. 该 HTML 文件应包含 ACCEPT/REVIEW 结论标记
        """
        work_dir = tmp_path / "workspace_summary"
        work_dir.mkdir()
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")

        results = self._run_pipeline(
            str(work_dir),
            enable_alphalens=True,
            target_stages=["DATA", "FACTOR", "REPORT"],
        )
        assert results["success"] is True

        # 1) reports-engine 生成 factor_analysis_report.html 在 REPORT_DIR
        report_dir = os.path.join(str(work_dir), "reports")
        factor_summary_html = os.path.join(report_dir, "factor_analysis_report.html")
        assert os.path.isfile(factor_summary_html), (
            f"因子分析汇总报告未生成: {factor_summary_html}"
        )

        # 2) 验证 HTML 内容包含因子名与结论
        with open(factor_summary_html, "r", encoding="utf-8") as f:
            html_content = f.read()
        # 至少包含 ACCEPT 或 REVIEW 标记（来自 metrics.suggested_verdict）
        assert ("ACCEPT" in html_content or "REVIEW" in html_content), (
            "汇总报告未包含 ACCEPT/REVIEW 结论标记"
        )
        # 包含因子卡片容器
        assert "factor-card" in html_content, "汇总报告缺少因子卡片结构"

        # 3) 同时验证 REPORT 阶段归档中也复制了 factor_analysis_report.html
        archive_dir = results["archive_dir"]
        report_step_dirs = [
            d for d in os.listdir(archive_dir)
            if d.startswith("step_") and "REPORT" in d
        ]
        assert report_step_dirs, "REPORT 步骤归档目录不存在"
        archived_summary = os.path.join(
            archive_dir, report_step_dirs[0], "artifacts", "factor_analysis_report.html"
        )
        assert os.path.isfile(archived_summary), (
            f"汇总报告未归档到 REPORT artifacts: {archived_summary}"
        )

    def test_alphalens_report_not_archived_when_disabled(self, tmp_path, monkeypatch):
        """QUANT_ALPHALENS_REPORT 未启用时，FACTOR 归档不应包含 alphalens 报告目录。

        验证默认关闭行为：FACTOR artifacts 下不应有以 task_id 命名且含 *_metrics.json 的子目录。
        """
        work_dir = tmp_path / "workspace_disabled"
        work_dir.mkdir()
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")
        monkeypatch.delenv("QUANT_ALPHALENS_REPORT", raising=False)

        results = self._run_pipeline(
            str(work_dir),
            enable_alphalens=False,
            target_stages=["DATA", "FACTOR", "REPORT"],
        )
        assert results["success"] is True

        archive_dir = results["archive_dir"]
        factor_step_dirs = [
            d for d in os.listdir(archive_dir)
            if d.startswith("step_") and "FACTOR" in d
        ]
        assert factor_step_dirs
        factor_artifacts_dir = os.path.join(archive_dir, factor_step_dirs[0], "artifacts")

        # FACTOR artifacts 下不应有 *_metrics.json
        metrics_in_archive = _glob.glob(os.path.join(factor_artifacts_dir, "**", "*_metrics.json"), recursive=True)
        assert not metrics_in_archive, (
            f"默认关闭时不应归档 alphalens metrics.json，但找到: {metrics_in_archive}"
        )

        # REPORT_DIR 中不应生成 factor_analysis_report.html
        report_dir = os.path.join(str(work_dir), "reports")
        assert not os.path.isfile(os.path.join(report_dir, "factor_analysis_report.html")), (
            "默认关闭时不应生成 factor_analysis_report.html"
        )


# ============================================================================
# T1-12: Processor Pipeline 集成测试（方向一）
# 验证 PRD 6.3 节要求：
# 1. 端到端跑一次完整 pipeline，校验 manifest 7 字段可重放
# 2. 4 种 YAML 配置组合跑通（全开 / 关 Winsorize / 关 Neutralize / 仅 IC+Fusion）
# ============================================================================


@pytest.mark.integration
@pytest.mark.skill_factor_engine
class TestProcessorPipelineIntegration:
    """T1-12: Processor Pipeline 端到端集成测试。

    覆盖：
    - 默认 ProcessorChain 路径下 manifest.json 含 7 必填字段
    - 4 种 YAML 配置组合都能跑通且产物结构正确
    - 同一输入跑两次 IC 偏差 < 1e-10（可重放性）
    """

    _MANIFEST_REQUIRED_FIELDS = {
        "run_id", "start_time", "pipeline_config",
        "input_data_hash", "steps", "output_artifacts", "env",
    }

    def _run_factor_only(self, work_dir, monkeypatch):
        """仅跑 DATA → FACTOR，缩短集成测试耗时。

        返回 (results, archive_dir, factor_engine_archive_dir)
        factor_engine_archive_dir: <work_dir>/archives/factor_engine/，存放 manifest
        """
        monkeypatch.setenv("QUANT_WORK_DIR", str(work_dir))
        monkeypatch.setenv("ALLOW_SYNTHETIC_FALLBACK", "true")
        monkeypatch.setenv("DATA_BACKENDS", "websearch")
        # 强制刷新，跳过 factor-engine 缓存
        monkeypatch.setenv("QUANT_FORCE_REFRESH", "1")
        # 关闭 alphalens 报告，避免干扰
        monkeypatch.delenv("QUANT_ALPHALENS_REPORT", raising=False)

        results = _run_pipeline_helper(
            str(work_dir),
            enable_alphalens=False,
            target_stages=["DATA", "FACTOR"],
        )
        assert results["success"] is True, (
            f"DATA→FACTOR 未成功: errors={results.get('errors')}"
        )
        factor_engine_archive_dir = os.path.join(
            str(work_dir), "archives", "factor_engine"
        )
        return results, results["archive_dir"], factor_engine_archive_dir

    def _write_pipeline_yaml(self, work_dir: str, config: dict) -> str:
        """在 work_dir 下写入 pipeline.yaml，返回文件路径。"""
        import yaml as _yaml
        yaml_path = os.path.join(str(work_dir), "pipeline.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        return yaml_path

    def _find_latest_manifest(self, factor_engine_archive_dir: str):
        """在 factor_engine 归档目录下找最新的 run_*/manifest.json。"""
        import json
        if not os.path.isdir(factor_engine_archive_dir):
            return None, None
        run_dirs = sorted(
            [d for d in os.listdir(factor_engine_archive_dir)
             if d.startswith("run_")],
            reverse=True,
        )
        for run_dir_name in run_dirs:
            manifest_path = os.path.join(
                factor_engine_archive_dir, run_dir_name, "manifest.json"
            )
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f), manifest_path
        return None, None

    # ── 测试 1：默认路径 manifest 7 字段齐全 ──────────────────────

    def test_default_pipeline_manifest_has_7_fields(self, tmp_path, monkeypatch):
        """默认 ProcessorChain 路径下 manifest.json 含 7 必填字段（PRD CR-3）。"""
        work_dir = tmp_path / "wp_default"
        work_dir.mkdir()
        results, archive_dir, fe_archive_dir = self._run_factor_only(work_dir, monkeypatch)

        manifest, manifest_path = self._find_latest_manifest(fe_archive_dir)
        assert manifest is not None, (
            f"未找到 manifest.json，factor_engine 归档目录: {fe_archive_dir}，"
            f"现有内容: {os.listdir(fe_archive_dir) if os.path.isdir(fe_archive_dir) else 'N/A'}"
        )

        missing = self._MANIFEST_REQUIRED_FIELDS - set(manifest.keys())
        assert not missing, f"manifest.json 缺少字段: {missing}，实际: {list(manifest.keys())}"

        # run_id 应为非空字符串（uuid4().hex = 32 字符）
        assert isinstance(manifest["run_id"], str) and len(manifest["run_id"]) >= 16
        # steps 应为列表（即使部分 Processor 跳过，也应有记录）
        assert isinstance(manifest["steps"], list)
        # pipeline_config 应为列表，每项含 processor 字段
        assert isinstance(manifest["pipeline_config"], list)
        if manifest["pipeline_config"]:
            assert "processor" in manifest["pipeline_config"][0]
        # env 应包含 python_version
        assert "python_version" in manifest["env"]

    # ── 测试 2：YAML 全开配置跑通 ─────────────────────────────────

    def test_pipeline_yaml_config_full_enabled(self, tmp_path, monkeypatch):
        """4 种 YAML 配置组合之一：全开（Neutralize+Winsorize+Fillna+IC+Correlation+Fusion）。"""
        work_dir = tmp_path / "wp_full"
        work_dir.mkdir()
        self._write_pipeline_yaml(work_dir, {
            "pipeline": [
                {"processor": "NeutralizeProcessor", "enabled": True,
                 "params": {"neutralize_mcap": True, "neutralize_industry": True, "min_count": 30}},
                {"processor": "WinsorizeProcessor", "enabled": True,
                 "params": {"method": "mad", "threshold": 3.0}},
                {"processor": "FillnaProcessor", "enabled": True,
                 "params": {"method": "rank_pct", "fill_value": 0.5}},
                {"processor": "ICAnalysisProcessor", "enabled": True,
                 "params": {"ic_type": "normal", "min_count": 10}},
                {"processor": "CorrelationFilterProcessor", "enabled": True,
                 "params": {"max_correlation": 0.7}},
                {"processor": "FusionProcessor", "enabled": True,
                 "params": {"method": "ic_weighted"}},
            ]
        })
        results, _, fe_archive_dir = self._run_factor_only(work_dir, monkeypatch)

        # 校验 FACTOR 产物存在且含 alpha_score
        factor_path = results["context"]["artifacts"].get("FACTOR")
        assert factor_path and os.path.exists(factor_path)
        import pandas as _pd
        df = _pd.read_parquet(factor_path)
        assert "alpha_score" in df.columns

        # manifest 应记录 6 个 Processor 的执行
        manifest, _ = self._find_latest_manifest(fe_archive_dir)
        assert manifest is not None
        executed = [s["processor"] for s in manifest["steps"]]
        # 全开配置下应至少跑了 IC + Correlation + Fusion（其余可能因数据不足跳过）
        assert "ICAnalysisProcessor" in executed
        assert "FusionProcessor" in executed

    # ── 测试 3：YAML 关 Winsorize 跑通 ────────────────────────────

    def test_pipeline_yaml_config_disable_winsorize(self, tmp_path, monkeypatch):
        """4 种 YAML 配置组合之二：关 Winsorize（其余默认）。"""
        work_dir = tmp_path / "wp_no_winsor"
        work_dir.mkdir()
        self._write_pipeline_yaml(work_dir, {
            "pipeline": [
                {"processor": "NeutralizeProcessor", "enabled": True,
                 "params": {"neutralize_mcap": True, "neutralize_industry": True, "min_count": 30}},
                {"processor": "WinsorizeProcessor", "enabled": False},
                {"processor": "ICAnalysisProcessor", "enabled": True,
                 "params": {"ic_type": "normal", "min_count": 10}},
                {"processor": "CorrelationFilterProcessor", "enabled": True,
                 "params": {"max_correlation": 0.7}},
                {"processor": "FusionProcessor", "enabled": True,
                 "params": {"method": "ic_weighted"}},
            ]
        })
        results, _, fe_archive_dir = self._run_factor_only(work_dir, monkeypatch)
        manifest, _ = self._find_latest_manifest(fe_archive_dir)
        assert manifest is not None
        executed = [s["processor"] for s in manifest["steps"]]
        # Winsorize 应被跳过（不在 executed 中）
        assert "WinsorizeProcessor" not in executed, (
            f"WinsorizeProcessor 应被禁用，但出现在 steps: {executed}"
        )
        # pipeline_config 中应仍包含 Winsorize（描述链含所有声明项）
        declared = [p["processor"] for p in manifest["pipeline_config"]]
        # 注意：loader.parse_yaml_to_processors 会过滤 enabled=False 的项，
        # 因此 pipeline_config 中也不应包含 WinsorizeProcessor
        assert "WinsorizeProcessor" not in declared

    # ── 测试 4：YAML 关 Neutralize 跑通 ───────────────────────────

    def test_pipeline_yaml_config_disable_neutralize(self, tmp_path, monkeypatch):
        """4 种 YAML 配置组合之三：关 Neutralize（其余默认）。"""
        work_dir = tmp_path / "wp_no_neutral"
        work_dir.mkdir()
        self._write_pipeline_yaml(work_dir, {
            "pipeline": [
                {"processor": "NeutralizeProcessor", "enabled": False},
                {"processor": "WinsorizeProcessor", "enabled": True,
                 "params": {"method": "mad", "threshold": 3.0}},
                {"processor": "ICAnalysisProcessor", "enabled": True,
                 "params": {"ic_type": "normal", "min_count": 10}},
                {"processor": "CorrelationFilterProcessor", "enabled": True,
                 "params": {"max_correlation": 0.7}},
                {"processor": "FusionProcessor", "enabled": True,
                 "params": {"method": "ic_weighted"}},
            ]
        })
        results, _, fe_archive_dir = self._run_factor_only(work_dir, monkeypatch)
        manifest, _ = self._find_latest_manifest(fe_archive_dir)
        assert manifest is not None
        executed = [s["processor"] for s in manifest["steps"]]
        assert "NeutralizeProcessor" not in executed, (
            f"NeutralizeProcessor 应被禁用，但出现在 steps: {executed}"
        )

    # ── 测试 5：YAML 仅 IC+Fusion 跑通 ────────────────────────────

    def test_pipeline_yaml_config_ic_fusion_only(self, tmp_path, monkeypatch):
        """4 种 YAML 配置组合之四：仅 IC + Fusion（最简链路）。"""
        work_dir = tmp_path / "wp_ic_fusion"
        work_dir.mkdir()
        self._write_pipeline_yaml(work_dir, {
            "pipeline": [
                {"processor": "ICAnalysisProcessor", "enabled": True,
                 "params": {"ic_type": "normal", "min_count": 10}},
                {"processor": "FusionProcessor", "enabled": True,
                 "params": {"method": "equal_weighted"}},
            ]
        })
        results, _, fe_archive_dir = self._run_factor_only(work_dir, monkeypatch)
        manifest, _ = self._find_latest_manifest(fe_archive_dir)
        assert manifest is not None
        executed = [s["processor"] for s in manifest["steps"]]
        # 仅 IC + Fusion 被执行
        assert "ICAnalysisProcessor" in executed
        assert "FusionProcessor" in executed
        assert "NeutralizeProcessor" not in executed
        assert "WinsorizeProcessor" not in executed

    # ── 测试 6：可重放性 - 同一数据跑两次 IC 偏差 < 1e-10 ─────────

    def test_manifest_replay_consistency(self, tmp_path, monkeypatch):
        """同一输入跑两次，两次输出的 ic_report.json 偏差 < 1e-10（PRD CR-4）。

        注意：``FACTOR_DIR`` 在 ``scripts/config.py`` 模块加载时固化，
        ``monkeypatch.setenv`` 不会重置该路径，因此两次运行的
        ``ic_report.json`` 会写入同一固化的 ``FACTOR_DIR``（后写覆盖前写）。
        为对比两次独立结果，本测试在第一次运行后立即读取并保存内容，
        然后再跑第二次。
        """
        import json as _json

        # 第一次运行
        work_dir_1 = tmp_path / "wp_replay_1"
        work_dir_1.mkdir()
        results_1, _, _ = self._run_factor_only(work_dir_1, monkeypatch)
        factor_path_1 = results_1["context"]["artifacts"].get("FACTOR")
        assert factor_path_1 and os.path.exists(factor_path_1), (
            f"FACTOR 产物缺失: {factor_path_1}"
        )
        # ic_report.json 与 factor_data.parquet 同目录（均在固化的 FACTOR_DIR 下）
        ic_report_path = os.path.join(os.path.dirname(factor_path_1), "ic_report.json")
        assert os.path.isfile(ic_report_path), f"ic_report.json 未生成: {ic_report_path}"

        # 立即读取第一次结果（避免第二次运行覆盖）
        with open(ic_report_path, "r", encoding="utf-8") as f:
            ic_1 = _json.load(f)

        # 第二次运行（新 work_dir，相同数据；ic_report.json 会被覆盖）
        work_dir_2 = tmp_path / "wp_replay_2"
        work_dir_2.mkdir()
        results_2, _, _ = self._run_factor_only(work_dir_2, monkeypatch)
        factor_path_2 = results_2["context"]["artifacts"].get("FACTOR")
        assert factor_path_2 and os.path.exists(factor_path_2)
        # 第二次 ic_report.json 应与 factor_path_2 同目录
        ic_report_path_2 = os.path.join(os.path.dirname(factor_path_2), "ic_report.json")
        assert os.path.isfile(ic_report_path_2), (
            f"第二次 ic_report.json 未生成: {ic_report_path_2}"
        )

        with open(ic_report_path_2, "r", encoding="utf-8") as f:
            ic_2 = _json.load(f)

        # 结构应一致（相同的 forward_period 键）
        assert set(ic_1.keys()) == set(ic_2.keys()), (
            f"两次 IC 报告结构不一致: {set(ic_1.keys())} vs {set(ic_2.keys())}"
        )

        # 逐 forward_period 比较每个因子的 ic_mean
        for period in ic_1:
            entries_1 = {e["factor"]: e.get("ic_mean", 0.0) for e in ic_1[period]}
            entries_2 = {e["factor"]: e.get("ic_mean", 0.0) for e in ic_2[period]}
            assert set(entries_1.keys()) == set(entries_2.keys()), (
                f"period={period} 因子集合不一致: {set(entries_1)} vs {set(entries_2)}"
            )
            for factor in entries_1:
                diff = abs(entries_1[factor] - entries_2[factor])
                assert diff < 1e-10, (
                    f"period={period} factor={factor} IC 偏差过大: "
                    f"{entries_1[factor]} vs {entries_2[factor]} (diff={diff})"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
