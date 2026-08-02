"""P1-3 工件版本化 + sha256 sidecar manifest

提供产物完整性校验与血缘追踪能力（PRD P1-3.1 ~ P1-3.6）：

- compute_sha256: 流式计算文件 sha256（支持大文件，1MB chunk）
- write_artifact: 写入 <name>.json + <name>.manifest.json（含 sha256 与 inputs 血缘）
- read_artifact:  读数据 + manifest，sha256 不匹配时 raise
- generate_run_manifest: 生成 <run_dir>/run_manifest.json，记录每个阶段的产物指纹

manifest 结构（PRD P1-3.2）：
    {
        "name": "backtest_result",
        "version": "V1",
        "sha256": "abc123...",
        "created_at": "2026-08-02T10:30:00",
        "inputs": [{"name": "factor_data", "sha256": "def456..."}]
    }

run_manifest 结构（PRD P1-3.6）：
    {
        "run_id": "20260802_103000",
        "start_at": "...",
        "end_at": "...",
        "stages": [
            {"name": "DATA", "status": "success", "latency_sec": 1.2,
             "artifacts": [{"name": "cleaned_data.parquet", "sha256": "..."}]}
        ],
        "inputs_sha256": {"DATA": "...", "FACTOR": "..."}
    }

兼容性：
- 旧产物（无 manifest）读取时不校验，记 warning（向后兼容）
- 现有 ctx.artifacts 字典结构保留，仅新增 sidecar manifest 文件
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("artifact_store")


# sha256 计算的 chunk 大小：1MB（平衡内存与 I/O 调用次数）
_CHUNK_SIZE = 1024 * 1024

# manifest 文件后缀
MANIFEST_SUFFIX = ".manifest.json"

# 默认工件版本（PRD P1-3.4：所有 V1 模型使用 "V1"）
DEFAULT_VERSION = "V1"


def compute_sha256(path: str) -> str:
    """对任意文件计算 sha256（PRD P1-3.1）。

    按 1MB chunk 流式读取，支持大文件（如 parquet/模型 pkl）。

    参数：
        path: 文件路径

    返回：
        sha256 十六进制摘要（64 字符）

    异常：
        FileNotFoundError: 文件不存在
        IsADirectoryError: 路径是目录而非文件
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"compute_sha256: 文件不存在: {path}")
    if p.is_dir():
        raise IsADirectoryError(f"compute_sha256: 路径是目录: {path}")

    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_artifact(
    name: str,
    data: Any,
    output_dir: str,
    inputs: Optional[List[str]] = None,
    version: str = DEFAULT_VERSION,
) -> str:
    """写入产物 + sidecar manifest（PRD P1-3.2）。

    生成两个文件：
        <output_dir>/<name>.json          产物数据（JSON 序列化）
        <output_dir>/<name>.manifest.json  manifest（含 sha256 与 inputs 血缘）

    参数：
        name:      产物名（不含扩展名）
        data:      可 JSON 序列化的数据
        output_dir: 输出目录（不存在则创建）
        inputs:    上游依赖产物路径列表（用于血缘追踪）
        version:   产物版本（默认 "V1"）

    返回：
        产物文件路径（<output_dir>/<name>.json）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 写产物文件
    artifact_path = os.path.join(output_dir, f"{name}.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # 计算 sha256
    sha = compute_sha256(artifact_path)

    # 构建 inputs 血缘
    inputs_manifest: List[Dict[str, str]] = []
    if inputs:
        for inp_path in inputs:
            try:
                inp_sha = compute_sha256(inp_path)
                inputs_manifest.append({
                    "name": os.path.basename(inp_path),
                    "sha256": inp_sha,
                })
            except (FileNotFoundError, IsADirectoryError) as e:
                logger.warning(f"write_artifact: 跳过 input {inp_path}: {e}")
                inputs_manifest.append({
                    "name": os.path.basename(inp_path),
                    "sha256": "",
                })

    # 写 manifest
    manifest = {
        "name": name,
        "version": version,
        "sha256": sha,
        "created_at": datetime.now().isoformat(),
        "inputs": inputs_manifest,
    }
    manifest_path = os.path.join(output_dir, f"{name}{MANIFEST_SUFFIX}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.debug(f"write_artifact: {artifact_path} (sha256={sha[:12]}..., inputs={len(inputs_manifest)})")
    return artifact_path


def read_artifact(name: str, output_dir: str) -> Tuple[Any, Dict]:
    """读产物 + manifest，校验 sha256（PRD P1-3.3）。

    参数：
        name:      产物名（不含扩展名）
        output_dir: 产物所在目录

    返回：
        (data, manifest) 元组

    异常：
        FileNotFoundError: 产物或 manifest 不存在
        ValueError: sha256 不匹配（产物被篡改或损坏）

    兼容性：
        旧产物（无 manifest）读取时不校验，记 warning，返回空 manifest dict
    """
    artifact_path = os.path.join(output_dir, f"{name}.json")
    manifest_path = os.path.join(output_dir, f"{name}{MANIFEST_SUFFIX}")

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"read_artifact: 产物不存在: {artifact_path}")

    # 读产物
    with open(artifact_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 读 manifest（向后兼容：无 manifest 时只 warning）
    if not os.path.exists(manifest_path):
        logger.warning(
            f"read_artifact: 产物 {name} 无 sidecar manifest，跳过 sha256 校验"
        )
        return data, {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 校验 sha256
    actual_sha = compute_sha256(artifact_path)
    expected_sha = manifest.get("sha256", "")
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"read_artifact: sha256 不匹配 (产物 {name}): "
            f"expected={expected_sha[:12]}..., actual={actual_sha[:12]}..."
        )

    return data, manifest


def generate_run_manifest(
    run_dir: str,
    run_id: str,
    start_at: str,
    end_at: str,
    stages: List[Dict[str, Any]],
    inputs_sha256: Optional[Dict[str, str]] = None,
) -> str:
    """生成 run_manifest.json（PRD P1-3.6）。

    在每次 run 结束时写到 <run_dir>/run_manifest.json。

    参数：
        run_dir:      运行归档目录
        run_id:       运行 ID（通常是 task_id 或时间戳）
        start_at:     运行开始时间 ISO 字符串
        end_at:       运行结束时间 ISO 字符串
        stages:       每个阶段的记录列表，每项含：
                      {"name": "DATA", "status": "success", "latency_sec": 1.2,
                       "artifacts": [{"name": "cleaned_data.parquet", "sha256": "..."}]}
        inputs_sha256: 各阶段上游输入的 sha256 映射（可选，用于调试确定性）

    返回：
        run_manifest.json 路径
    """
    manifest = {
        "run_id": run_id,
        "start_at": start_at,
        "end_at": end_at,
        "stages": stages,
        "inputs_sha256": inputs_sha256 or {},
        "version": DEFAULT_VERSION,
    }
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"run_manifest.json 已落盘: {manifest_path} (stages={len(stages)})")
    return manifest_path


def load_run_manifest(run_dir: str) -> Dict[str, Any]:
    """加载 run_manifest.json（供 replay_check 使用）。

    参数：
        run_dir: 运行归档目录

    返回：
        manifest dict

    异常：
        FileNotFoundError: run_manifest.json 不存在
    """
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"run_manifest.json 不存在: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
