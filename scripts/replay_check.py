"""P1-3.7 同输入回放校验工具

对比两次 run 的 run_manifest.json，输出：
- 输入 sha256 是否相同
- 输出 sha256 差异报告（哪些阶段的产物发生了变化）

用途：调试确定性。如果两次 run 的输入 sha256 相同但输出 sha256 不同，
说明管线存在非确定性来源（如随机种子未固定、时间戳污染等）。

使用方式：
    # 命令行
    python -m scripts.replay_check <run_dir_1> <run_dir_2>

    # 代码调用
    from scripts.replay_check import compare_runs
    report = compare_runs("/path/to/run1", "/path/to/run2")
    if report["inputs_identical"] and not report["outputs_identical"]:
        print("⚠️ 非确定性差异：输入相同但输出不同")
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List

from scripts.artifact_store import load_run_manifest

logger = logging.getLogger("replay_check")


def compare_runs(run_dir_a: str, run_dir_b: str) -> Dict[str, Any]:
    """对比两次 run 的 run_manifest.json（PRD P1-3.7）。

    参数：
        run_dir_a: 第一次 run 的归档目录
        run_dir_b: 第二次 run 的归档目录

    返回：
        {
            "run_a": "...",
            "run_b": "...",
            "inputs_identical": bool,        # 上游输入 sha256 是否完全相同
            "outputs_identical": bool,       # 各阶段产物 sha256 是否完全相同
            "common_stages": ["DATA", ...],  # 两次 run 都执行了的阶段
            "input_diffs": [                 # 输入 sha256 差异
                {"stage": "DATA", "sha_a": "...", "sha_b": "..."}
            ],
            "output_diffs": [                # 输出 sha256 差异
                {"stage": "BACKTEST", "artifacts": [
                    {"name": "backtest_result.json", "sha_a": "...", "sha_b": "..."}
                ]}
            ],
            "only_in_a": ["MODEL"],          # 只在 run_a 执行的阶段
            "only_in_b": ["EXECUTION"],      # 只在 run_b 执行的阶段
        }

    异常：
        FileNotFoundError: 任一 run 缺少 run_manifest.json
    """
    manifest_a = load_run_manifest(run_dir_a)
    manifest_b = load_run_manifest(run_dir_b)

    # 按阶段名索引
    stages_a = {s["name"]: s for s in manifest_a.get("stages", [])}
    stages_b = {s["name"]: s for s in manifest_b.get("stages", [])}

    common = sorted(set(stages_a.keys()) & set(stages_b.keys()))
    only_a = sorted(set(stages_a.keys()) - set(stages_b.keys()))
    only_b = sorted(set(stages_b.keys()) - set(stages_a.keys()))

    # 输入 sha256 对比
    inputs_a = manifest_a.get("inputs_sha256", {})
    inputs_b = manifest_b.get("inputs_sha256", {})
    input_diffs: List[Dict[str, str]] = []
    for stage in common:
        sha_a = inputs_a.get(stage, "")
        sha_b = inputs_b.get(stage, "")
        if sha_a != sha_b:
            input_diffs.append({"stage": stage, "sha_a": sha_a, "sha_b": sha_b})

    # 输出 sha256 对比
    output_diffs: List[Dict[str, Any]] = []
    for stage in common:
        arts_a = {a["name"]: a.get("sha256", "") for a in stages_a[stage].get("artifacts", [])}
        arts_b = {a["name"]: a.get("sha256", "") for a in stages_b[stage].get("artifacts", [])}
        stage_diffs: List[Dict[str, str]] = []
        for name in sorted(set(arts_a.keys()) | set(arts_b.keys())):
            sha_a = arts_a.get(name, "")
            sha_b = arts_b.get(name, "")
            if sha_a != sha_b:
                stage_diffs.append({"name": name, "sha_a": sha_a, "sha_b": sha_b})
        if stage_diffs:
            output_diffs.append({"stage": stage, "artifacts": stage_diffs})

    return {
        "run_a": run_dir_a,
        "run_b": run_dir_b,
        "inputs_identical": len(input_diffs) == 0,
        "outputs_identical": len(output_diffs) == 0,
        "common_stages": common,
        "input_diffs": input_diffs,
        "output_diffs": output_diffs,
        "only_in_a": only_a,
        "only_in_b": only_b,
    }


def format_report(report: Dict[str, Any]) -> str:
    """将对比报告格式化为可读字符串。"""
    lines = [
        "=" * 60,
        "Run Replay Check Report",
        "=" * 60,
        f"Run A: {report['run_a']}",
        f"Run B: {report['run_b']}",
        "",
        f"Common stages: {', '.join(report['common_stages']) or '(none)'}",
        f"Only in A:     {', '.join(report['only_in_a']) or '(none)'}",
        f"Only in B:     {', '.join(report['only_in_b']) or '(none)'}",
        "",
        f"Inputs identical:  {'YES' if report['inputs_identical'] else 'NO'}",
        f"Outputs identical: {'YES' if report['outputs_identical'] else 'NO'}",
        "",
    ]

    if report["input_diffs"]:
        lines.append("Input sha256 differences:")
        for d in report["input_diffs"]:
            lines.append(f"  - {d['stage']}:")
            lines.append(f"      A: {d['sha_a'][:16] or '(empty)'}...")
            lines.append(f"      B: {d['sha_b'][:16] or '(empty)'}...")
        lines.append("")

    if report["output_diffs"]:
        lines.append("Output sha256 differences:")
        for d in report["output_diffs"]:
            lines.append(f"  - {d['stage']}:")
            for art in d["artifacts"]:
                lines.append(f"      {art['name']}:")
                lines.append(f"        A: {art['sha_a'][:16] or '(empty)'}...")
                lines.append(f"        B: {art['sha_b'][:16] or '(empty)'}...")
        lines.append("")

    # 结论
    if report["inputs_identical"] and report["outputs_identical"]:
        lines.append("✓ 两次 run 输入与输出完全一致（确定性确认）")
    elif report["inputs_identical"] and not report["outputs_identical"]:
        lines.append("⚠️ 输入相同但输出不同 → 存在非确定性来源（检查随机种子/时间戳）")
    elif not report["inputs_identical"] and report["outputs_identical"]:
        lines.append("ℹ 输入不同但输出相同（可能是输入差异不影响输出）")
    else:
        lines.append("ℹ 输入与输出均不同（常规情况，需结合业务判断）")

    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: List[str] = None) -> int:
    """命令行入口：python -m scripts.replay_check <dir_a> <dir_b>"""
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print("Usage: python -m scripts.replay_check <run_dir_a> <run_dir_b>")
        return 1
    try:
        report = compare_runs(argv[0], argv[1])
        print(format_report(report))
        return 0 if report["outputs_identical"] else 2
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
