"""T1-3: ProcessorChain 调度器

职责：
- 串联多个 Processor 顺序执行
- 每步执行前检查 ``requires`` 声明的列是否存在
- 每步执行后调用 ``ctx.recorder.log_step`` 记录中间状态（如果 recorder 已配置）
- 异常隔离：单个 Processor 失败时，根据 ``fail_fast`` 决定是否中断整条链

设计要点
--------
- 不做拓扑排序：Processor 列表顺序即执行顺序（YAML 配置显式声明）
- 依赖检查基于 ``requires`` 列表，校验 df 是否包含必需列
- 中间结果不复制大 DataFrame，仅传引用（PRD Q1-1 决策）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.processors.base import (
    Processor,
    ProcessContext,
    ProcessorRequirementError,
)

logger = logging.getLogger("processors.chain")


class ChainValidationError(RuntimeError):
    """ProcessorChain 初始化时拓扑校验失败。"""


class ProcessorChain:
    """Processor 调度器。

    Parameters
    ----------
    processors:
        有序的 Processor 列表（按执行顺序）
    fail_fast:
        ``True`` 时首个 Processor 失败立即抛异常；
        ``False`` 时记录错误并跳过该 Processor，继续执行后续（默认 True）

    用法
    ----
    >>> chain = ProcessorChain([NeutralizeProcessor(), WinsorizeProcessor()])
    >>> result = chain.run(df, ctx)
    """

    def __init__(
        self,
        processors: List[Processor],
        fail_fast: bool = True,
    ) -> None:
        if not processors:
            raise ChainValidationError("ProcessorChain 至少需要一个 Processor")
        self.processors = list(processors)
        self.fail_fast = fail_fast
        self._validate_basic()

    def _validate_basic(self) -> None:
        """基础校验：类型检查 + 重名检查"""
        seen_names = set()
        for i, p in enumerate(self.processors):
            if not isinstance(p, Processor):
                raise ChainValidationError(
                    f"processors[{i}] 不是 Processor 实例: {type(p).__name__}"
                )
            if p.name in seen_names:
                logger.warning(
                    f"ProcessorChain: 检测到同名 Processor '{p.name}'，"
                    f"如需多次执行同一工序请显式重命名"
                )
            seen_names.add(p.name)

    def run(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        """顺序执行所有 Processor。

        Parameters
        ----------
        df:
            输入 DataFrame
        ctx:
            工序状态载体

        Returns
        -------
        处理后的 DataFrame

        Raises
        ------
        ProcessorRequirementError:
            ``fail_fast=True`` 且依赖列缺失时抛出
        Exception:
            ``fail_fast=True`` 且 Processor 执行异常时原样抛出
        """
        if df is None or df.empty:
            logger.warning("ProcessorChain.run: 输入 df 为空，跳过所有工序")
            return df

        current_df = df
        executed: List[str] = []
        skipped: List[str] = []
        errors: List[Dict[str, Any]] = []

        for p in self.processors:
            step_name = p.name
            try:
                # 依赖检查
                p.check_requirements(current_df)

                # 执行
                logger.info(f"ProcessorChain: 执行 {step_name}...")
                before_rows = len(current_df)
                before_cols = list(current_df.columns)

                current_df = p(current_df, ctx)

                after_rows = len(current_df) if current_df is not None else 0
                after_cols = list(current_df.columns) if current_df is not None else []

                executed.append(step_name)

                # 记录到 Recorder（如果已配置）
                if ctx.recorder is not None:
                    ctx.recorder.log_step(
                        processor=p,
                        df_after=current_df,
                        before_rows=before_rows,
                        before_cols=before_cols,
                        after_rows=after_rows,
                        after_cols=after_cols,
                    )

                logger.debug(
                    f"ProcessorChain: {step_name} 完成 "
                    f"(rows: {before_rows}→{after_rows}, cols: {len(before_cols)}→{len(after_cols)})"
                )

            except ProcessorRequirementError as e:
                skipped.append(step_name)
                errors.append({"processor": step_name, "error": str(e), "type": "requirement"})
                logger.warning(f"ProcessorChain: {step_name} 依赖检查失败: {e}")
                if self.fail_fast:
                    raise
                logger.warning(f"ProcessorChain: fail_fast=False，跳过 {step_name} 继续执行")

            except Exception as e:
                skipped.append(step_name)
                errors.append({"processor": step_name, "error": str(e), "type": "execution"})
                logger.exception(f"ProcessorChain: {step_name} 执行异常: {e}")
                if self.fail_fast:
                    raise
                logger.warning(f"ProcessorChain: fail_fast=False，跳过 {step_name} 继续执行")

        # 汇总日志
        logger.info(
            f"ProcessorChain: 执行完成 (executed={executed}, skipped={skipped})"
        )
        if errors:
            ctx.metadata.setdefault("chain_errors", []).extend(errors)

        return current_df

    def describe_chain(self) -> List[Dict[str, Any]]:
        """返回整条链的描述（供 Recorder 落盘）"""
        return [p.describe() for p in self.processors]
