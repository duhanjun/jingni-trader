"""
运行归档模块
每次完整流程执行时，创建时间戳目录保存所有过程和结果
"""
import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional


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

    def create_run(self, task_id: str) -> str:
        """创建运行归档目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.archive_root, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        return self.run_dir

    def create_step_dir(self, step_num: int, stage: str) -> str:
        """创建步骤子文件夹"""
        stage_name = STAGE_NAMES.get(stage, stage)
        dir_name = f"step_{step_num}_{stage}"
        step_dir = os.path.join(self.run_dir, dir_name)
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(os.path.join(step_dir, "artifacts"), exist_ok=True)
        self.step_dirs[stage] = step_dir
        return step_dir

    def save_artifact_copy(self, stage: str, artifact_path: str):
        """复制产物到归档目录"""
        if not artifact_path or not os.path.exists(artifact_path):
            return
        step_dir = self.step_dirs.get(stage)
        if not step_dir:
            return
        dest = os.path.join(step_dir, "artifacts", os.path.basename(artifact_path))
        if os.path.isfile(artifact_path):
            shutil.copy2(artifact_path, dest)
        elif os.path.isdir(artifact_path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(artifact_path, dest)

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