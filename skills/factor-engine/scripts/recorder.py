"""T1-4: ExperimentRecorder - 实验可重放记录器

借鉴 Microsoft Qlib Recorder + MLflow 风格，记录每次 pipeline 运行的完整元数据。

manifest 结构（PRD FE-PP-005）
-----------------------------
    {
        "run_id": "uuid4().hex",
        "start_time": "2026-08-02T10:30:00",
        "pipeline_config": [...],         # 各 Processor 的 describe() 输出
        "input_data_hash": {...},         # 输入文件 sha256（如 cleaned_data.parquet）
        "steps": [                        # 每步执行后状态
            {
                "processor": "NeutralizeProcessor",
                "params": {...},
                "rows_after": 10000,
                "cols_after": [...],
                "nan_ratio": 0.02,
                "before_rows": 10000,
                "before_cols": [...],
                "after_rows": 10000,
                "after_cols": [...]
            }
        ],
        "output_artifacts": [...],       # 输出文件路径 + sha256
        "env": {                          # 环境快照
            "python_version": "3.9.x",
            "pandas_version": "2.0.x",
            "polars_version": "0.20.x",   # 缺失时为 null
            "quant_legacy_pipeline": "0",
            "quant_factor_backend": "pandas",
            ...
        }
    }

接入 artifact_store（PRD Q1-2）
------------------------------
- ``finalize()`` 写入 manifest.json 后，调用 ``compute_sha256`` 计算 manifest 自身指纹
- 该 sha256 纳入 P1-3 已建立的 sha256 Manifest 覆盖范围
- 写盘失败不阻塞主流程（PRD NFR-005）
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

if TYPE_CHECKING:
    from scripts.processors.base import Processor

logger = logging.getLogger("recorder")


class ExperimentRecorder:
    """实验可重放记录器。

    Parameters
    ----------
    archive_dir:
        归档根目录；每次 ``__init__`` 创建 ``run_YYYYMMDD_HHMMSS`` 子目录
    pipeline_config:
        可选的 pipeline 描述（``ProcessorChain.describe_chain()`` 输出）；
        也可后续通过 ``set_pipeline_config`` 设置
    input_data_paths:
        输入数据文件路径列表，用于计算 sha256 血缘
    """

    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        archive_dir: Path,
        pipeline_config: Optional[List[Dict[str, Any]]] = None,
        input_data_paths: Optional[List[str]] = None,
    ) -> None:
        self.archive_dir = Path(archive_dir)
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = self.archive_dir / timestamp
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # 写盘失败不阻塞主流程（PRD NFR-005），降级为内存记录
            logger.warning(f"Recorder 创建目录失败，降级为内存模式: {e}")
            self.run_dir = None

        self.run_id = uuid4().hex
        self.start_time = datetime.now().isoformat()
        self._pipeline_config: Optional[List[Dict[str, Any]]] = pipeline_config
        self._input_data_paths: List[str] = list(input_data_paths) if input_data_paths else []
        self._steps: List[Dict[str, Any]] = []
        self._output_artifacts: List[Dict[str, Any]] = []
        self._env_snapshot: Dict[str, Any] = self._snapshot_env()
        self._finalized = False
        self._manifest_sha256: Optional[str] = None

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def set_pipeline_config(self, config: List[Dict[str, Any]]) -> None:
        """设置 pipeline 配置（通常在 Chain.run 前调用）"""
        self._pipeline_config = config

    def log_step(
        self,
        processor: "Processor",
        df_after: pd.DataFrame,
        before_rows: Optional[int] = None,
        before_cols: Optional[List[str]] = None,
        after_rows: Optional[int] = None,
        after_cols: Optional[List[str]] = None,
    ) -> None:
        """记录单个 Processor 执行后的状态。

        Parameters
        ----------
        processor:
            Processor 实例
        df_after:
            执行后的 DataFrame（用于计算 nan_ratio；不复制，仅读取统计量）
        before_rows/before_cols:
            执行前的行数与列名（可选）
        after_rows/after_cols:
            执行后的行数与列名（可选；为空时从 df_after 推断）
        """
        if self._finalized:
            logger.warning("Recorder 已 finalize，忽略后续 log_step 调用")
            return

        try:
            nan_ratio = float(df_after.isna().mean().mean()) if not df_after.empty else 0.0
        except Exception:
            nan_ratio = -1.0

        if after_rows is None:
            after_rows = len(df_after) if df_after is not None else 0
        if after_cols is None:
            after_cols = list(df_after.columns) if df_after is not None else []

        step_record = {
            "processor": processor.name,
            "params": processor.describe().get("params", {}),
            "before_rows": before_rows,
            "before_cols": before_cols,
            "after_rows": after_rows,
            "after_cols": after_cols,
            "nan_ratio": round(nan_ratio, 6),
        }
        self._steps.append(step_record)

    def log_output_artifact(self, name: str, path: str) -> None:
        """记录输出产物（如 factor_data.parquet）"""
        self._output_artifacts.append({
            "name": name,
            "path": path,
        })

    def finalize(self) -> Optional[str]:
        """写入 manifest.json，返回其 sha256 指纹。

        若目录创建失败（内存模式），返回 None。
        """
        if self._finalized:
            logger.warning("Recorder 已 finalize，跳过重复调用")
            return self._manifest_sha256

        if self.run_dir is None:
            logger.warning("Recorder 处于内存模式，不写盘")
            self._finalized = True
            return None

        manifest = {
            "run_id": self.run_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "pipeline_config": self._pipeline_config,
            "input_data_hash": self._compute_input_hashes(),
            "steps": self._steps,
            "output_artifacts": self._output_artifacts,
            "env": self._env_snapshot,
        }

        manifest_path = self.run_dir / self.MANIFEST_FILENAME
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        except OSError as e:
            logger.warning(f"Recorder 写 manifest 失败: {e}")
            self._finalized = True
            return None

        # 接入 artifact_store 的 sha256 机制（PRD FE-PP-009 / Q1-2）
        try:
            self._manifest_sha256 = self._compute_sha256_safe(str(manifest_path))
            logger.info(
                f"Recorder: manifest 落盘 {manifest_path} (sha256={self._manifest_sha256[:12]}...)"
            )
        except Exception as e:
            logger.warning(f"Recorder 计算 manifest sha256 失败: {e}")

        self._finalized = True
        return self._manifest_sha256

    @property
    def manifest_path(self) -> Optional[Path]:
        """manifest.json 的完整路径（内存模式下为 None）"""
        if self.run_dir is None:
            return None
        return self.run_dir / self.MANIFEST_FILENAME

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _snapshot_env(self) -> Dict[str, Any]:
        """环境快照（Python/pandas/polars 版本 + QUANT_ 环境变量）"""
        env = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "pandas_version": None,
            "numpy_version": None,
            "polars_version": None,
            "scipy_version": None,
            "sklearn_version": None,
            "quant_env": {},
        }
        try:
            import pandas as _pd
            env["pandas_version"] = _pd.__version__
        except Exception:
            pass
        try:
            import numpy as _np
            env["numpy_version"] = _np.__version__
        except Exception:
            pass
        try:
            import polars as _pl
            env["polars_version"] = _pl.__version__
        except Exception:
            pass
        try:
            import scipy as _sp
            env["scipy_version"] = _sp.__version__
        except Exception:
            pass
        try:
            import sklearn as _sk
            env["sklearn_version"] = _sk.__version__
        except Exception:
            pass

        # 收集所有 QUANT_ 前缀环境变量
        for k, v in os.environ.items():
            if k.startswith("QUANT_"):
                env["quant_env"][k] = v

        return env

    def _compute_input_hashes(self) -> Dict[str, str]:
        """计算输入数据文件的 sha256（用于血缘追踪）"""
        hashes: Dict[str, str] = {}
        for path in self._input_data_paths:
            if not path or not os.path.isfile(path):
                continue
            sha = self._compute_sha256_safe(path)
            if sha:
                hashes[os.path.basename(path)] = sha
        return hashes

    def _compute_sha256_safe(self, path: str) -> Optional[str]:
        """安全计算 sha256，失败时返回 None"""
        try:
            # 优先使用 P1-3 的 artifact_store.compute_sha256
            try:
                from scripts.artifact_store import compute_sha256
                return compute_sha256(path)
            except ImportError:
                # 跨 skill 加载场景：尝试从 factor-engine/scripts 同级目录加载
                import importlib.util as _ilu
                artifact_store_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "artifact_store.py",
                )
                if os.path.exists(artifact_store_path):
                    spec = _ilu.spec_from_file_location("_artifact_store_tmp", artifact_store_path)
                    mod = _ilu.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod.compute_sha256(path)
                # 最终 fallback：内置实现
                import hashlib
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                return h.hexdigest()
        except Exception as e:
            logger.warning(f"compute_sha256 失败 ({path}): {e}")
            return None
