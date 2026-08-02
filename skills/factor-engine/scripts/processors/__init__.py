"""factor-engine Processor Pipeline 模块（方向一 T1-1 ~ T1-5）

借鉴 Microsoft Qlib 的 Processor + Recorder 设计，将因子处理流程抽象为可插拔的工序链 + 实验可重放记录器。

模块结构
--------
- ``base``:                   Processor 抽象基类 + ProcessContext 工序状态载体
- ``chain``:                  ProcessorChain 调度器（拓扑校验 + 依赖检查）
- ``loader``:                 pipeline.yaml 加载器（work_dir 优先 / skill 默认 / 兜底默认链）
- ``neutralize``:             NeutralizeProcessor（行业+市值中性化）
- ``winsorize``:              WinsorizeProcessor（MAD / 分位数去极值）
- ``fillna``:                 FillnaProcessor（rank_pct / mean / zero / ffill 填充）
- ``standardize``:            StandardizeProcessor（z-score / min-max 标准化）
- ``ic_analysis``:            ICAnalysisProcessor（Pearson / Spearman IC 分析）
- ``correlation_filter``:     CorrelationFilterProcessor（相关性去冗余）
- ``fusion``:                 FusionProcessor（多因子等权 / IC 加权融合）

环境变量
--------
- ``QUANT_LEGACY_PIPELINE``:  ``"1"`` 时强制走旧 5 步硬编码路径（兼容回滚）
"""
from scripts.processors.base import (
    Processor,
    ProcessContext,
    ProcessorRequirementError,
)
from scripts.processors.chain import (
    ProcessorChain,
    ChainValidationError,
)
from scripts.processors.loader import (
    load_pipeline_config,
    parse_yaml_to_processors,
)

__all__ = [
    "Processor",
    "ProcessContext",
    "ProcessorRequirementError",
    "ProcessorChain",
    "ChainValidationError",
    "load_pipeline_config",
    "parse_yaml_to_processors",
]
