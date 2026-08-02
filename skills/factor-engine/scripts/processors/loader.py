"""T1-5: pipeline.yaml 加载器

加载顺序（PRD FE-PP-010 / Q1-3）
--------------------------------
1. ``work_dir/pipeline.yaml``（用户覆盖）
2. ``skill/scripts/processors/pipeline.yaml``（默认配置）
3. 兜底默认链（``_default_processors()``）

YAML schema
-----------
.. code-block:: yaml

    pipeline:
      - processor: NeutralizeProcessor
        enabled: true
        params:
          neutralize_mcap: true
          neutralize_industry: true
          min_count: 30

      - processor: WinsorizeProcessor
        enabled: false              # enabled=false 时跳过该工序
        params:
          method: mad
          threshold: 3.0

Processor 名解析
----------------
通过 ``PROCESSOR_REGISTRY`` 字典映射类名到类对象。
新增 Processor 时需在 ``PROCESSOR_REGISTRY`` 中注册。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scripts.processors.base import Processor
from scripts.processors.neutralize import NeutralizeProcessor
from scripts.processors.winsorize import WinsorizeProcessor
from scripts.processors.fillna import FillnaProcessor
from scripts.processors.standardize import StandardizeProcessor
from scripts.processors.ic_analysis import ICAnalysisProcessor
from scripts.processors.correlation_filter import CorrelationFilterProcessor
from scripts.processors.fusion import FusionProcessor

logger = logging.getLogger("processors.loader")


#: Processor 类名 → 类对象 注册表
PROCESSOR_REGISTRY: Dict[str, type] = {
    "NeutralizeProcessor": NeutralizeProcessor,
    "WinsorizeProcessor": WinsorizeProcessor,
    "FillnaProcessor": FillnaProcessor,
    "StandardizeProcessor": StandardizeProcessor,
    "ICAnalysisProcessor": ICAnalysisProcessor,
    "CorrelationFilterProcessor": CorrelationFilterProcessor,
    "FusionProcessor": FusionProcessor,
}


def register_processor(name: str, cls: type) -> None:
    """注册自定义 Processor（供外部扩展使用）"""
    if not issubclass(cls, Processor):
        raise TypeError(f"register_processor: {cls} 必须继承 Processor")
    PROCESSOR_REGISTRY[name] = cls
    logger.debug(f"Processor 注册: {name} → {cls.__name__}")


def load_pipeline_config(
    work_dir: Optional[Path] = None,
    config_filename: str = "pipeline.yaml",
) -> List[Processor]:
    """加载 pipeline 配置，返回 Processor 列表。

    加载顺序：
    1. ``work_dir/config_filename``（用户覆盖）
    2. ``<skill>/scripts/processors/config_filename``（默认配置）
    3. ``_default_processors()``（兜底默认链）

    Parameters
    ----------
    work_dir:
        工作目录；为空时仅查找 skill 默认配置
    config_filename:
        配置文件名，默认 ``"pipeline.yaml"``

    Returns
    -------
    有序的 Processor 实例列表
    """
    candidates: List[Path] = []
    if work_dir:
        candidates.append(Path(work_dir) / config_filename)
    # skill 默认配置：本文件所在目录
    candidates.append(Path(__file__).parent / config_filename)

    for path in candidates:
        if path.exists():
            logger.info(f"加载 pipeline 配置: {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                return parse_yaml_to_processors(config)
            except Exception as e:
                logger.warning(f"解析 pipeline 配置失败 ({path}): {e}，回退默认链")

    logger.info("未找到 pipeline.yaml，使用兜底默认链")
    return _default_processors()


def parse_yaml_to_processors(config: Dict[str, Any]) -> List[Processor]:
    """将 YAML 配置解析为 Processor 实例列表。

    Parameters
    ----------
    config:
        已解析的 YAML 字典，结构为 ``{"pipeline": [{"processor": ..., "enabled": ..., "params": ...}, ...]}``

    Returns
    -------
    有序的 Processor 实例列表（``enabled=false`` 的工序被跳过）

    Raises
    ------
    ValueError:
        配置格式错误或引用了未注册的 Processor
    """
    if not isinstance(config, dict):
        raise ValueError(f"pipeline 配置必须是 dict，实际类型: {type(config).__name__}")

    pipeline = config.get("pipeline")
    if not isinstance(pipeline, list):
        raise ValueError(
            f"pipeline 配置必须包含 'pipeline' 列表，实际: {type(pipeline).__name__}"
        )

    processors: List[Processor] = []
    for i, item in enumerate(pipeline):
        if not isinstance(item, dict):
            raise ValueError(f"pipeline[{i}] 必须是 dict，实际: {type(item).__name__}")

        name = item.get("processor")
        if not name:
            raise ValueError(f"pipeline[{i}] 缺少 'processor' 字段")

        if name not in PROCESSOR_REGISTRY:
            available = ", ".join(sorted(PROCESSOR_REGISTRY.keys()))
            raise ValueError(
                f"pipeline[{i}] 引用了未注册的 Processor '{name}'；"
                f"可用: {available}"
            )

        enabled = item.get("enabled", True)
        if not enabled:
            logger.info(f"pipeline[{i}] {name} 已禁用（enabled=false），跳过")
            continue

        params = item.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError(
                f"pipeline[{i}] {name} 的 params 必须是 dict，实际: {type(params).__name__}"
            )

        cls = PROCESSOR_REGISTRY[name]
        try:
            instance = cls(**params)
        except TypeError as e:
            raise ValueError(f"pipeline[{i}] {name} 参数构造失败: {e}") from e

        processors.append(instance)

    if not processors:
        raise ValueError("pipeline 配置为空（所有工序都被禁用或未配置）")

    return processors


def _default_processors() -> List[Processor]:
    """兜底默认链（与 ``pipeline.yaml`` 默认配置 + 旧 ``run()`` 行为对齐）

    仅启用 IC + Correlation + Fusion 三步（PRD CR-3 / CR-6 兼容性要求），
    Neutralize/Winsorize/Fillna 作为可选工序，用户需通过 YAML 显式启用。

    顺序：
    1. ICAnalysisProcessor（IC 分析，只读，写入 ctx.ic_results）
    2. CorrelationFilterProcessor（相关性去冗余，只读，写入 ctx.selected_factors）
    3. FusionProcessor（IC 加权融合，输出 alpha_score 列）
    """
    return [
        ICAnalysisProcessor(),
        CorrelationFilterProcessor(),
        FusionProcessor(),
    ]
