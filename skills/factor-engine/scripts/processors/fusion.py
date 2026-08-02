"""T1-2: FusionProcessor - 多因子融合为复合 Alpha 信号

复用现有 ``FactorEngine.factor_fusion`` 逻辑，将其包装为 Processor。

行为：
- 读取 ctx.selected_factors（由 CorrelationFilterProcessor 写入）
- 读取 ctx.ic_results（由 ICAnalysisProcessor 写入）
- 支持 ``ic_weighted`` / ``equal_weighted`` 两种融合方法
- 输出新增 ``alpha_score`` 列（与旧 factor_fusion 一致）

兼容性：
- 与旧 ``FactorEngine.factor_fusion`` 输出结构完全一致
- NaN 隔离：缺失因子的 rank 填充为 0.5 中性值，避免 0 权重×NaN 污染
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.fusion")


class FusionProcessor(Processor):
    """多因子融合工序。

    Parameters
    ----------
    method:
        ``"ic_weighted"`` 或 ``"equal_weighted"``，默认 ``"ic_weighted"``
    forward_period_for_weight:
        IC 加权时使用的前瞻期（默认 ``"ret_forward_5d"``），与旧实现一致
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        method: str = "ic_weighted",
        forward_period_for_weight: str = "ret_forward_5d",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            method=method,
            forward_period_for_weight=forward_period_for_weight,
            **kwargs,
        )
        if method not in ("ic_weighted", "equal_weighted"):
            raise ValueError(
                f"FusionProcessor 不支持的 method: {method}"
                f"（可选 ic_weighted/equal_weighted）"
            )
        self.method = method
        self.forward_period_for_weight = forward_period_for_weight

    def __call__(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        if df.empty:
            return df

        selected_factors = ctx.selected_factors
        if not selected_factors:
            logger.warning("FusionProcessor: ctx.selected_factors 为空，跳过融合")
            return df

        # 计算权重
        if self.method == "ic_weighted":
            weights = self._get_ic_weights(ctx.ic_results, selected_factors)
        else:
            weights = {f: 1.0 / len(selected_factors) for f in selected_factors}

        # 按截面 rank 百分位 + NaN 隔离
        normalized = df[["code", "date"]].copy()
        for factor in selected_factors:
            if factor not in df.columns:
                logger.warning(f"FusionProcessor: 因子 {factor} 不在 df 中，跳过")
                continue
            normalized[f"{factor}_rank"] = df.groupby("date")[factor].transform(
                lambda x: x.rank(pct=True)
            ).fillna(0.5)  # NaN 隔离

        rank_cols = [
            f"{f}_rank" for f in selected_factors if f"{f}_rank" in normalized.columns
        ]
        normalized["alpha_score"] = 0.0
        for f, col in zip(selected_factors, rank_cols):
            w = weights.get(f, 0)
            normalized["alpha_score"] += w * normalized[col]

        # 将 alpha_score 合并回 df
        result = df.merge(
            normalized[["code", "date", "alpha_score"]],
            on=["code", "date"],
            how="left",
        )

        logger.info(f"FusionProcessor: 完成融合 (method={self.method}, weights={weights})")
        return result

    def _get_ic_weights(
        self,
        ic_results: Dict[str, Any],
        selected_factors: List[str],
    ) -> Dict[str, float]:
        """根据 IC_IR 计算因子权重（与 FactorEngine._get_ic_weights 一致）"""
        weights = {}
        total_ic_ir = 0

        ic_list = ic_results.get(self.forward_period_for_weight, [])
        ic_map = {item["factor"]: item["ic_ir"] for item in ic_list}

        for factor in selected_factors:
            ic_ir = abs(ic_map.get(factor, 0))
            weights[factor] = ic_ir
            total_ic_ir += ic_ir

        if total_ic_ir > 0:
            weights = {k: v / total_ic_ir for k, v in weights.items()}
        else:
            n = len(selected_factors)
            weights = {k: 1.0 / n for k in selected_factors}

        return weights

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "method": self.method,
                "forward_period_for_weight": self.forward_period_for_weight,
            },
            "requires": list(self.requires) + ["ctx.selected_factors", "ctx.ic_results"],
            "provides": ["alpha_score"],
        }
