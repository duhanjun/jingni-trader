"""T1-2: CorrelationFilterProcessor - 因子相关性去冗余

复用现有 ``scripts/optimizations/vectorized_correlation.py`` 的实现，
将其包装为 Processor。

行为：
- 计算因子间相关性矩阵（按 date 聚合后求 Pearson 相关性）
- 高于阈值的因子对中，剔除 IC_IR 较低的那个（保强剔弱）
- 结果写入 ctx.selected_factors（供 FusionProcessor 使用）

兼容性：
- 与旧 ``FactorEngine.correlation_analysis`` 输出结构完全一致
- 支持 polars 后端（透传 ``ctx.backend``）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.correlation_filter")


class CorrelationFilterProcessor(Processor):
    """因子相关性去冗余工序。

    Parameters
    ----------
    max_correlation:
        最大允许相关性阈值，默认 0.7
    factor_names:
        显式指定待分析的因子列；为空时自动推断
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        max_correlation: float = 0.7,
        factor_names: List[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_correlation=max_correlation,
            factor_names=factor_names,
            **kwargs,
        )
        self.max_correlation = float(max_correlation)
        self.factor_names = list(factor_names) if factor_names else None

    def _infer_factor_names(self, df: pd.DataFrame) -> List[str]:
        if self.factor_names:
            return [f for f in self.factor_names if f in df.columns]
        if ctx_factor_names := getattr(self, "_ctx_factor_names", None):
            return [f for f in ctx_factor_names if f in df.columns]
        exclude = {"code", "date", "industry", "estimated_mv", "money_flow_raw",
                   "ret_1d", "ret_5d", "ret_20d", "ret_60d", "turnover_5d"}
        return [c for c in df.columns if c not in exclude]

    def __call__(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        if df.empty:
            ctx.selected_factors = []
            return df

        self._ctx_factor_names = ctx.factor_names
        factors = self._infer_factor_names(df)
        if not factors:
            logger.warning("CorrelationFilterProcessor: 未找到待分析的因子列，跳过")
            ctx.selected_factors = []
            return df

        from scripts.optimizations.vectorized_correlation import correlation_analysis

        result = correlation_analysis(
            factor_df=df,
            factor_names=factors,
            max_correlation=self.max_correlation,
            backend=ctx.backend,
        )

        ctx.selected_factors = result["selected_factors"]
        # 保留完整结果到 metadata，便于 Recorder 落盘
        ctx.metadata["correlation_result"] = {
            "selected_factors": result["selected_factors"],
            "removed_factors": result["removed_factors"],
            "max_correlation": self.max_correlation,
        }

        logger.info(
            f"CorrelationFilterProcessor: 选中 {len(ctx.selected_factors)} 个因子，"
            f"剔除 {len(result['removed_factors'])} 个高相关因子"
        )
        return df  # 相关性分析不修改 df

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "max_correlation": self.max_correlation,
                "factor_names": self.factor_names,
            },
            "requires": list(self.requires),
            "provides": ["ctx.selected_factors"],
        }
