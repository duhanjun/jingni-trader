"""
配置化流水线引擎 - Pipeline Runner
借鉴来源: Qlib qrun (YAML config-driven workflow)
"""
from .runner import PipelineRunner, PipelineConfig

__all__ = ["PipelineRunner", "PipelineConfig"]