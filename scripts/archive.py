"""
运行归档模块
每次完整流程执行时，创建时间戳目录保存所有过程和结果

P1-3 增强：
- save_artifact_copy 复制产物时同时生成 sidecar manifest（含 sha256）
- record_stage_latency 记录每个阶段耗时（供 run_manifest 使用）
- write_run_manifest 在 run 结束时落盘 run_manifest.json
"""
import os
import json
import logging
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# P1-3: artifact_store 提供 sha256 计算与 run_manifest 生成
# 延迟 import 避免循环依赖（archive 被 engine 引用，artifact_store 不引用 engine）
logger = logging.getLogger("archive")


STAGE_NAMES = {
    "DATA": "数据采集",
    "FACTOR": "因子构建",
    "MODEL": "模型训练",
    "BACKTEST": "回测验证",
    "PORTFOLIO": "组合优化",
    "EXECUTION": "执行监控",
    "REPORT": "报告生成",
}


class RunArchiver:
    """运行归档器"""

    def __init__(self, archive_root: str):
        self.archive_root = archive_root
        self.run_dir: str = ""
        self.step_dirs: Dict[str, str] = {}
        self.step_results: Dict[str, dict] = {}
        # P1-3: 阶段耗时与产物 sha256 记录（供 write_run_manifest 使用）
        self._stage_timings: Dict[str, Dict[str, Any]] = {}
        self._run_started_at: Optional[str] = None
        self._run_task_id: str = ""

    def create_run(self, task_id: str) -> str:
        """创建运行归档目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.archive_root, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        # P1-3: 记录 run 起始时间与 task_id，供 run_manifest 使用
        self._run_started_at = datetime.now().isoformat()
        self._run_task_id = task_id or timestamp
        return self.run_dir

    def create_step_dir(self, step_num: int, stage: str) -> str:
        """创建步骤子文件夹"""
        stage_name = STAGE_NAMES.get(stage, stage)
        dir_name = f"step_{step_num}_{stage}"
        step_dir = os.path.join(self.run_dir, dir_name)
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(os.path.join(step_dir, "artifacts"), exist_ok=True)
        self.step_dirs[stage] = step_dir
        # P1-3: 记录阶段开始时间
        self._stage_timings[stage] = {"start": time.time(), "end": None}
        return step_dir

    def record_stage_end(self, stage: str, status: str) -> None:
        """P1-3: 记录阶段结束时间与状态（供 run_manifest 使用）。

        参数：
            stage:  阶段名
            status: "success" / "failed" / "skipped"
        """
        if stage not in self._stage_timings:
            self._stage_timings[stage] = {"start": time.time(), "end": None}
        self._stage_timings[stage]["end"] = time.time()
        self._stage_timings[stage]["status"] = status

    def save_artifact_copy(
        self,
        stage: str,
        artifact_path: str,
        inputs: Optional[List[str]] = None,
    ):
        """复制产物到归档目录，并生成 sidecar manifest（P1-3.5）。

        P1-3 增强：
        - 复制文件后计算 sha256，生成 <name>.manifest.json sidecar
        - manifest 含 inputs 血缘（从 ctx.artifacts 推断上游依赖）

        参数：
            stage:         阶段名
            artifact_path: 源产物路径
            inputs:        上游依赖产物路径列表（可选，用于血缘追踪）
        """
        if not artifact_path or not os.path.exists(artifact_path):
            return
        step_dir = self.step_dirs.get(stage)
        if not step_dir:
            return
        dest = os.path.join(step_dir, "artifacts", os.path.basename(artifact_path))
        if os.path.isfile(artifact_path):
            shutil.copy2(artifact_path, dest)
            # P1-3.5: 生成 sidecar manifest（仅对文件，目录暂不生成）
            self._write_sidecar_manifest(dest, inputs)
        elif os.path.isdir(artifact_path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(artifact_path, dest)
            # 目录产物不生成 manifest（sha256 仅适用于文件）

    def _write_sidecar_manifest(
        self,
        artifact_path: str,
        inputs: Optional[List[str]] = None,
    ) -> None:
        """P1-3.5: 为已复制的产物生成 sidecar manifest。

        manifest 路径: <artifact_path>.manifest.json
        内容: {"name":..., "version":"V1", "sha256":..., "created_at":..., "inputs":[...]}
        """
        try:
            from scripts.artifact_store import compute_sha256, MANIFEST_SUFFIX, DEFAULT_VERSION
        except ImportError:
            # 跨 skill 加载场景下 scripts 包可能指向子 skill，延迟到运行时再 import
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location(
                "_artifact_store_tmp",
                os.path.join(os.path.dirname(__file__), "artifact_store.py"),
            )
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            compute_sha256 = mod.compute_sha256
            MANIFEST_SUFFIX = mod.MANIFEST_SUFFIX
            DEFAULT_VERSION = mod.DEFAULT_VERSION

        try:
            sha = compute_sha256(artifact_path)
        except Exception as e:
            logger.warning(f"P1-3 计算 sha256 失败（跳过 manifest）: {artifact_path}: {e}")
            return

        # 构建 inputs 血缘
        inputs_manifest: List[Dict[str, str]] = []
        if inputs:
            for inp in inputs:
                if not inp or not os.path.isfile(inp):
                    continue
                try:
                    inp_sha = compute_sha256(inp)
                except Exception:
                    inp_sha = ""
                inputs_manifest.append({
                    "name": os.path.basename(inp),
                    "sha256": inp_sha,
                })

        manifest = {
            "name": os.path.basename(artifact_path),
            "version": DEFAULT_VERSION,
            "sha256": sha,
            "created_at": datetime.now().isoformat(),
            "inputs": inputs_manifest,
        }
        manifest_path = artifact_path + MANIFEST_SUFFIX
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"P1-3 写 manifest 失败: {manifest_path}: {e}")

    def write_run_manifest(
        self,
        inputs_sha256: Optional[Dict[str, str]] = None,
    ) -> str:
        """P1-3.6: 在 run 结束时落盘 run_manifest.json。

        汇总每个已执行阶段的 status / latency_sec / artifacts sha256，
        写到 <run_dir>/run_manifest.json。

        参数：
            inputs_sha256: 各阶段上游输入的 sha256 映射（可选）

        返回：
            run_manifest.json 路径；若 run_dir 未初始化则返回空字符串
        """
        if not self.run_dir:
            return ""

        # 延迟 import artifact_store.generate_run_manifest（避免循环依赖）
        try:
            from scripts.artifact_store import generate_run_manifest, compute_sha256
        except ImportError:
            import importlib.util as _ilu
            spec = _ilu.spec_from_file_location(
                "_artifact_store_tmp",
                os.path.join(os.path.dirname(__file__), "artifact_store.py"),
            )
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            generate_run_manifest = mod.generate_run_manifest
            compute_sha256 = mod.compute_sha256

        # 汇总各阶段记录
        stages_record: List[Dict[str, Any]] = []
        for stage, step_dir in self.step_dirs.items():
            timing = self._stage_timings.get(stage, {})
            start = timing.get("start")
            end = timing.get("end")
            latency = (end - start) if (start and end) else None
            status = timing.get("status") or (
                "success" if self.step_results.get(stage, {}).get("success") else "failed"
            )

            # 收集该阶段所有产物的 sha256
            artifacts_dir = os.path.join(step_dir, "artifacts")
            artifacts_record: List[Dict[str, str]] = []
            if os.path.isdir(artifacts_dir):
                for fname in sorted(os.listdir(artifacts_dir)):
                    fpath = os.path.join(artifacts_dir, fname)
                    # 跳过 manifest sidecar 文件本身
                    if fname.endswith(".manifest.json"):
                        continue
                    if not os.path.isfile(fpath):
                        continue
                    try:
                        sha = compute_sha256(fpath)
                    except Exception:
                        sha = ""
                    artifacts_record.append({"name": fname, "sha256": sha})

            stages_record.append({
                "name": stage,
                "status": status,
                "latency_sec": latency,
                "artifacts": artifacts_record,
            })

        end_at = datetime.now().isoformat()
        manifest_path = generate_run_manifest(
            run_dir=self.run_dir,
            run_id=self._run_task_id or os.path.basename(self.run_dir),
            start_at=self._run_started_at or end_at,
            end_at=end_at,
            stages=stages_record,
            inputs_sha256=inputs_sha256,
        )
        return manifest_path

    def record_step_result(self, stage: str, result: dict):
        """记录步骤结果"""
        self.step_results[stage] = result

    def write_step_summary(self, stage: str, step_num: int):
        """写入步骤小结报告"""
        step_dir = self.step_dirs.get(stage)
        if not step_dir:
            return

        stage_name = STAGE_NAMES.get(stage, stage)
        result = self.step_results.get(stage, {})

        lines = [
            f"# Step {step_num}: {stage_name} ({stage})",
            "",
            f"**执行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**执行状态**: {'成功' if result.get('success') else '失败'}",
            "",
        ]

        if result.get("success"):
            lines.extend([
                "## 产物",
                f"- 路径: `{result.get('artifact_path', 'N/A')}`",
                "",
            ])

            metadata = result.get("metadata", {})
            if metadata:
                lines.append("## 元数据")
                lines.append("")
                for key, value in metadata.items():
                    lines.append(f"- **{key}**: {value}")
                lines.append("")
        else:
            lines.extend([
                "## 错误信息",
                f"```",
                result.get("error", "未知错误"),
                f"```",
                "",
            ])

        summary_path = os.path.join(step_dir, "summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def write_pipeline_summary(self, completed: List[str], failed: List[str],
                                target_stages: List[str], user_intent: str,
                                task_id: str, errors: List[str]):
        """写入全流程汇总报告"""
        if not self.run_dir:
            return

        lines = [
            "# 运行全流程汇总报告",
            "",
            f"**任务ID**: {task_id}",
            f"**运行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**用户意图**: {user_intent}",
            "",
            "## 执行概览",
            "",
            f"- 目标阶段: {' → '.join(target_stages)}",
            f"- 已完成: {len(completed)}/{len(target_stages)} ({', '.join(completed) if completed else '无'})",
            f"- 已失败: {len(failed)}/{len(target_stages)} ({', '.join(failed) if failed else '无'})",
            "",
        ]

        if completed:
            lines.append("## 已完成步骤")
            lines.append("")
            for i, stage in enumerate(completed, 1):
                stage_name = STAGE_NAMES.get(stage, stage)
                result = self.step_results.get(stage, {})
                lines.append(f"### Step {i}: {stage_name} ({stage})")
                lines.append(f"- 状态: 成功")
                lines.append(f"- 产物: `{result.get('artifact_path', 'N/A')}`")
                lines.append("")
                step_dir = self.step_dirs.get(stage, "")
                if step_dir:
                    lines.append(f"详细报告: [{step_dir}/summary.md]({os.path.basename(step_dir)}/summary.md)")
                    lines.append("")

        if failed:
            lines.append("## 失败步骤")
            lines.append("")
            for i, stage in enumerate(failed, len(completed) + 1):
                stage_name = STAGE_NAMES.get(stage, stage)
                result = self.step_results.get(stage, {})
                lines.append(f"### Step {i}: {stage_name} ({stage})")
                lines.append(f"- 状态: 失败")
                lines.append(f"- 错误: {result.get('error', '未知错误')}")
                lines.append("")

        if errors:
            lines.append("## 全局错误")
            lines.append("")
            for error in errors:
                lines.append(f"- {error}")
            lines.append("")

        lines.extend([
            "## 归档目录结构",
            "",
            "```",
            f"{os.path.basename(self.run_dir)}/",
            f"├── pipeline_summary.md  (当前文件)",
            "",
        ])

        for i, target in enumerate(target_stages, 1):
            step_dir_name = f"step_{i}_{target}"
            result = self.step_results.get(target, {})
            status = "✓" if result.get("success") else "✗"
            stage_name = STAGE_NAMES.get(target, target)
            lines.append(f"├── {step_dir_name}/  {status} {stage_name}")
            lines.append(f"│   └── summary.md")
            lines.append(f"│   └── artifacts/")
            lines.append("")

        lines.append("```")

        summary_path = os.path.join(self.run_dir, "pipeline_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))