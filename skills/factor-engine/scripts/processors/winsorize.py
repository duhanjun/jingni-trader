"""T1-2: WinsorizeProcessor - 去极值处理

支持两种方法：
- ``mad``:    基于 MAD（Median Absolute Deviation）的去极值，阈值默认 3.0
- ``quantile``: 分位数截尾，默认 [0.01, 0.99]

按日期截面处理（每个交易日独立去极值），保持时序无偏。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.winsorize")


class WinsorizeProcessor(Processor):
    """去极值工序。

    Parameters
    ----------
    method:
        ``"mad"`` 或 ``"quantile"``，默认 ``"mad"``
    threshold:
        MAD 方法的阈值（默认 3.0，即超过 median ± 3×MAD 的值被截断）
    quantile_range:
        分位数方法的上下界，默认 ``(0.01, 0.99)``
    factor_names:
        显式指定待处理的因子列；为空时自动推断（排除 code/date/industry 等）
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        method: str = "mad",
        threshold: float = 3.0,
        quantile_range: Tuple[float, float] = (0.01, 0.99),
        factor_names: List[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            method=method,
            threshold=threshold,
            quantile_range=quantile_range,
            factor_names=factor_names,
            **kwargs,
        )
        if method not in ("mad", "quantile"):
            raise ValueError(f"WinsorizeProcessor 不支持的 method: {method}（可选 mad/quantile）")
        self.method = method
        self.threshold = float(threshold)
        self.quantile_range = tuple(quantile_range)
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
            logger.warning("WinsorizeProcessor: 未找到待处理的因子列，跳过")
            return df

        result = df.copy()

        if self.method == "mad":
            result = self._winsorize_mad(result, factors)
        else:
            result = self._winsorize_quantile(result, factors)

        logger.info(
            f"WinsorizeProcessor: 完成 {len(factors)} 个因子去极值 "
            f"(method={self.method}, threshold={self.threshold})"
        )
        return result

    def _winsorize_mad(self, df: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
        """MAD 法：每个截面内 median ± threshold × MAD 截断"""
        def _clip_section(s: pd.Series) -> pd.Series:
            median = s.median()
            mad = (s - median).abs().median()
            if mad == 0 or np.isnan(mad):
                return s
            lower = median - self.threshold * mad
            upper = median + self.threshold * mad
            return s.clip(lower=lower, upper=upper)

        for f in factors:
            df[f] = df.groupby("date")[f].transform(_clip_section)
        return df

    def _winsorize_quantile(self, df: pd.DataFrame, factors: List[str]) -> pd.DataFrame:
        """分位数法：每个截面内按 quantile_range 截尾"""
        lower_q, upper_q = self.quantile_range

        def _clip_section(s: pd.Series) -> pd.Series:
            lower = s.quantile(lower_q)
            upper = s.quantile(upper_q)
            return s.clip(lower=lower, upper=upper)

        for f in factors:
            df[f] = df.groupby("date")[f].transform(_clip_section)
        return df

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "method": self.method,
                "threshold": self.threshold,
                "quantile_range": list(self.quantile_range),
                "factor_names": self.factor_names,
            },
            "requires": list(self.requires),
            "provides": [],
        }
