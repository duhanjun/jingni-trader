"""T1-2: NeutralizeProcessor - 行业 + 市值中性化

复用现有 ``scripts/optimizations/vectorized_neutralize.py`` 的实现，
将其包装为 Processor，接入 ProcessorChain。

与旧 ``FactorEngine.neutralize()`` 行为一致：
- 行业列缺失时自动从 ``ctx.industry_df`` merge
- 支持 polars 后端（透传 ``ctx.backend``）
- 输出新增 ``{factor}_neutral`` 列
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.neutralize")


class NeutralizeProcessor(Processor):
    """行业 + 市值中性化工序。

    Parameters
    ----------
    neutralize_mcap:
        是否市值中性化，默认 True
    neutralize_industry:
        是否行业中性化，默认 True
    mcap_col:
        对数市值列名，默认 ``"lncap"``
    industry_col:
        行业列名，默认 ``"industry"``
    min_count:
        截面最少样本数，默认 30
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        neutralize_mcap: bool = True,
        neutralize_industry: bool = True,
        mcap_col: str = "lncap",
        industry_col: str = "industry",
        min_count: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            neutralize_mcap=neutralize_mcap,
            neutralize_industry=neutralize_industry,
            mcap_col=mcap_col,
            industry_col=industry_col,
            min_count=min_count,
            **kwargs,
        )
        self.neutralize_mcap = neutralize_mcap
        self.neutralize_industry = neutralize_industry
        self.mcap_col = mcap_col
        self.industry_col = industry_col
        self.min_count = min_count

    def __call__(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        from scripts.optimizations.vectorized_neutralize import neutralize_factor

        if df.empty:
            return df

        if not self.neutralize_mcap and not self.neutralize_industry:
            logger.info("NeutralizeProcessor: 市值/行业中性化均关闭，直接返回")
            return df

        result = df.copy()

        # 行业列缺失时从 ctx.industry_df merge
        if (
            self.neutralize_industry
            and self.industry_col not in result.columns
            and ctx.industry_df is not None
        ):
            industry_cols = [c for c in ["code", self.industry_col] if c in ctx.industry_df.columns]
            if industry_cols:
                result = result.merge(ctx.industry_df[industry_cols], on="code", how="left")
                logger.info(f"NeutralizeProcessor: 从 ctx.industry_df 合并 {self.industry_col} 列")

        # 确定待中性化的因子列
        # 默认排除非因子列；如果 ctx.factor_names 指定，优先用之
        exclude_cols = {"code", "date", "industry", "estimated_mv", "money_flow_raw",
                        "ret_1d", "ret_5d", "ret_20d", "ret_60d", "turnover_5d"}
        if ctx.factor_names:
            factor_names = [f for f in ctx.factor_names if f in result.columns]
        else:
            factor_names = [c for c in result.columns if c not in exclude_cols]

        if not factor_names:
            logger.warning("NeutralizeProcessor: 未找到待中性化的因子列，跳过")
            return result

        # 检查必需的列
        needed = []
        if self.neutralize_mcap and self.mcap_col not in result.columns:
            needed.append(self.mcap_col)
        if self.neutralize_industry and self.industry_col not in result.columns:
            needed.append(self.industry_col)
        if needed:
            logger.warning(
                f"NeutralizeProcessor: 必需列 {needed} 缺失，无法中性化，原样返回"
            )
            return result

        # 调用现有向量化实现，透传 backend
        result = neutralize_factor(
            factor_df=result,
            factor_names=factor_names,
            neutralize_mcap=self.neutralize_mcap,
            neutralize_industry=self.neutralize_industry,
            mcap_col=self.mcap_col,
            industry_col=self.industry_col,
            min_count=self.min_count,
            backend=ctx.backend,
        )

        # 中性化后，ctx.factor_names 应更新为 _neutral 后缀
        # 但下游工序（如 ICAnalysis）应能识别 _neutral 列
        # 这里通过 ctx.metadata 标记，由调用方决定是否更新 factor_names
        ctx.metadata["neutralized_factors"] = [f"{f}_neutral" for f in factor_names]

        logger.info(
            f"NeutralizeProcessor: 完成 {len(factor_names)} 个因子中性化 "
            f"(mcap={self.neutralize_mcap}, industry={self.neutralize_industry})"
        )
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "neutralize_mcap": self.neutralize_mcap,
                "neutralize_industry": self.neutralize_industry,
                "mcap_col": self.mcap_col,
                "industry_col": self.industry_col,
                "min_count": self.min_count,
            },
            "requires": list(self.requires),
            "provides": ["{factor}_neutral"],
        }
