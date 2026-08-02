"""T1-2: FillnaProcessor - 缺失值填充

支持四种方法：
- ``rank_pct``: 按截面 rank 后填充为百分位（默认 0.5 中性值，避免 0 权重×NaN 污染）
- ``mean``:     按截面均值填充
- ``zero``:     填充为 0
- ``ffill``:    按代码（code）组内前向填充

注意：与原 ``factor_fusion`` 中内嵌的 ``rank().fillna(0.5)`` 行为一致，
单独抽出便于在 Fusion 工序前统一处理 NaN。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.fillna")


class FillnaProcessor(Processor):
    """缺失值填充工序。

    Parameters
    ----------
    method:
        ``"rank_pct"`` / ``"mean"`` / ``"zero"`` / ``"ffill"``，默认 ``"rank_pct"``
    fill_value:
        ``rank_pct`` 方法的中性填充值，默认 0.5
    factor_names:
        显式指定待处理的因子列；为空时自动推断
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        method: str = "rank_pct",
        fill_value: float = 0.5,
        factor_names: List[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            method=method,
            fill_value=fill_value,
            factor_names=factor_names,
            **kwargs,
        )
        if method not in ("rank_pct", "mean", "zero", "ffill"):
            raise ValueError(
                f"FillnaProcessor 不支持的 method: {method}"
                f"（可选 rank_pct/mean/zero/ffill）"
            )
        self.method = method
        self.fill_value = float(fill_value)
        self.factor_names = list(factor_names) if factor_names else None

    def _infer_factor_names(self, df: pd.DataFrame) -> List[str]:
        if self.factor_names:
            return [f for f in self.factor_names if f in df.columns]
        exclude = {"code", "date", "industry", "estimated_mv", "money_flow_raw",
                   "ret_1d", "ret_5d", "ret_20d", "ret_60d", "turnover_5d"}
        return [c for c in df.columns if c not in exclude]

    def __call__(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        if df.empty:
            return df

        factors = self._infer_factor_names(df)
        if not factors:
            logger.warning("FillnaProcessor: 未找到待处理的因子列，跳过")
            return df

        result = df.copy()

        if self.method == "rank_pct":
            for f in factors:
                # 截面 rank 百分位，NaN 填充为中性值
                result[f] = (
                    result.groupby("date")[f]
                    .transform(lambda x: x.rank(pct=True))
                    .fillna(self.fill_value)
                )
        elif self.method == "mean":
            for f in factors:
                result[f] = result.groupby("date")[f].transform(
                    lambda x: x.fillna(x.mean())
                )
                # 全 NaN 截面 fallback
                result[f] = result[f].fillna(0.0)
        elif self.method == "zero":
            for f in factors:
                result[f] = result[f].fillna(0.0)
        elif self.method == "ffill":
            for f in factors:
                result[f] = result.groupby("code")[f].ffill()
                # ffill 后仍可能存在开头 NaN（首日无前值），fallback 为 0
                result[f] = result[f].fillna(0.0)

        logger.info(
            f"FillnaProcessor: 完成 {len(factors)} 个因子缺失值填充 "
            f"(method={self.method}, fill_value={self.fill_value})"
        )
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "method": self.method,
                "fill_value": self.fill_value,
                "factor_names": self.factor_names,
            },
            "requires": list(self.requires),
            "provides": [],
        }
