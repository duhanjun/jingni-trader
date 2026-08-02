"""T1-2: StandardizeProcessor - 因子标准化

支持两种方法：
- ``zscore``:  按截面 z-score 标准化（均值 0、标准差 1），默认
- ``minmax``:  按截面 min-max 标准化到 [0, 1]

按日期截面处理，避免跨时序数据泄露。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.standardize")


class StandardizeProcessor(Processor):
    """因子标准化工序。

    Parameters
    ----------
    method:
        ``"zscore"`` 或 ``"minmax"``，默认 ``"zscore"``
    factor_names:
        显式指定待处理的因子列；为空时自动推断
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        method: str = "zscore",
        factor_names: List[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(method=method, factor_names=factor_names, **kwargs)
        if method not in ("zscore", "minmax"):
            raise ValueError(
                f"StandardizeProcessor 不支持的 method: {method}（可选 zscore/minmax）"
            )
        self.method = method
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
            logger.warning("StandardizeProcessor: 未找到待处理的因子列，跳过")
            return df

        result = df.copy()

        if self.method == "zscore":
            for f in factors:
                result[f] = result.groupby("date")[f].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
                )
                # std=0 时返回 0（避免 NaN 污染下游）
                result[f] = result[f].fillna(0.0)
        else:  # minmax
            for f in factors:
                def _minmax(s: pd.Series) -> pd.Series:
                    lo, hi = s.min(), s.max()
                    if hi - lo == 0:
                        return pd.Series(0.0, index=s.index)
                    return (s - lo) / (hi - lo)
                result[f] = result.groupby("date")[f].transform(_minmax)
                result[f] = result[f].fillna(0.0)

        logger.info(
            f"StandardizeProcessor: 完成 {len(factors)} 个因子标准化 (method={self.method})"
        )
        return result

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "method": self.method,
                "factor_names": self.factor_names,
            },
            "requires": list(self.requires),
            "provides": [],
        }
