"""T1-1: Processor 抽象基类 + ProcessContext 工序状态载体

借鉴 Microsoft Qlib 的 Processor 设计：
- 每个 Processor 声明 ``requires``（依赖的列/字段）
- ``__call__`` 接收 (df, ctx) 并返回新的 DataFrame
- ``describe`` 返回元数据，供 Recorder 落盘

设计约束（PRD Q1-1）
-------------------
- ctx 仅传引用，禁止复制大 DataFrame
- 大数据如需落盘，显式写 parquet 到 ctx.work_dir
- 轻量元数据（IC 结果、selected_factors）放 ctx

异常处理
--------
- ``ProcessorRequirementError``: 依赖字段缺失时抛出，由 Chain 捕获决定是否中断
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from scripts.recorder import ExperimentRecorder

logger = logging.getLogger("processors.base")


class ProcessorRequirementError(RuntimeError):
    """Processor 依赖的列/字段缺失时抛出。"""


@dataclass
class ProcessContext:
    """工序间状态传递载体（轻量元数据 + 大数据引用）。

    设计原则
    --------
    - 大 DataFrame（如 factor_df 中间结果）不复制，仅传引用
    - 轻量元数据（IC 结果、selected_factors）放本对象
    - 如需落盘中间结果，显式写 parquet 到 ``work_dir``

    Attributes
    ----------
    industry_df:
        行业数据 DataFrame（引用，含 code/industry 列）；中性化工序需要
    recorder:
        ExperimentRecorder 实例，工序完成后调用 ``log_step``
    ic_results:
        IC 分析结果（按 forward_period 分组的 dict），供 Fusion 工序读取权重
    selected_factors:
        去冗余后的因子列表（CorrelationFilter 输出，Fusion 输入）
    forward_returns:
        前瞻收益 DataFrame（IC 工序输入），含 code/date/ret_forward_*
    factor_names:
        当前 pipeline 处理的因子名列表（初始由调用方注入）
    task_id:
        当前任务 ID（用于 Recorder 目录命名）
    work_dir:
        工作目录 Path（用于落盘中间结果）
    backend:
        DataFrame 后端（"pandas" / "polars" / "auto" / None），透传给支持双后端的 Processor
    metadata:
        自由扩展字段（任意工序可读写）
    """

    industry_df: Optional[pd.DataFrame] = None
    recorder: Optional["ExperimentRecorder"] = None
    ic_results: Dict[str, Any] = field(default_factory=dict)
    selected_factors: List[str] = field(default_factory=list)
    forward_returns: Optional[pd.DataFrame] = None
    factor_names: List[str] = field(default_factory=list)
    task_id: str = ""
    work_dir: Optional[Path] = None
    backend: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Processor(ABC):
    """Processor 抽象基类。

    子类必须实现：
    - ``requires``:  依赖的列名列表（如 ``["industry", "lncap"]``）
    - ``__call__``:  处理 DataFrame，返回新 DataFrame
    - ``describe``:  返回工序元数据 dict（供 Recorder 落盘）

    子类可选实现：
    - ``provides``:  输出的列名列表（仅用于文档化，不强制校验）
    """

    #: 依赖的列名（子类覆盖）
    requires: List[str] = []

    #: 提供的列名（子类覆盖；仅文档化）
    provides: List[str] = []

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    @abstractmethod
    def __call__(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        """处理 DataFrame，返回新 DataFrame。

        参数
        ----
        df:
            当前 pipeline 中的因子 DataFrame（含 code/date/factor 列）
        ctx:
            工序状态载体（可读写 ic_results / selected_factors 等）

        返回
        ----
        新的 DataFrame（通常新增列或修改列，避免就地修改）
        """

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        """返回工序元数据，用于 Recorder 落盘。

        建议包含：processor 名、参数、输入输出列说明。
        """

    @property
    def name(self) -> str:
        """工序名（默认为类名，子类可覆盖）"""
        return self.__class__.__name__

    def check_requirements(self, df: pd.DataFrame) -> None:
        """检查 df 是否包含 requires 声明的所有列。

        缺失时抛 ``ProcessorRequirementError``，由 Chain 决定是否中断。
        """
        missing = [c for c in self.requires if c not in df.columns]
        if missing:
            raise ProcessorRequirementError(
                f"{self.name} 依赖的列缺失: {missing}；当前 df 列: {list(df.columns)}"
            )

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
