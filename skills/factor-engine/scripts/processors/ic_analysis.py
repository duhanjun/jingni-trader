"""T1-2: ICAnalysisProcessor - 因子 IC 分析

复用现有 ``FactorEngine.ic_analysis`` 逻辑，将其包装为 Processor。

行为：
- 读取 ctx.forward_returns（前瞻收益 DataFrame，含 ret_forward_1d/5d/20d）
- 对每个因子 × 每个前瞻期计算 IC 序列与统计量
- 结果写入 ctx.ic_results（供 FusionProcessor 读取权重）
- 不修改 df（IC 分析是只读的，仅产生元数据）

兼容性：
- 与旧 ``FactorEngine.ic_analysis`` 输出结构完全一致
- 支持 ``IC_TYPE`` 环境变量（spearman / normal）
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from scripts.processors.base import Processor, ProcessContext

logger = logging.getLogger("processors.ic_analysis")


class ICAnalysisProcessor(Processor):
    """因子 IC 分析工序。

    Parameters
    ----------
    ic_type:
        ``"spearman"`` / ``"normal"``（pearson）；为空时读取环境变量 ``IC_TYPE``
    forward_periods:
        前瞻期列表，默认 ``[1, 5, 20]``
    min_count:
        单日截面最小样本数，默认 10
    factor_names:
        显式指定待分析的因子列；为空时自动推断
    """

    requires: List[str] = ["code", "date"]

    def __init__(
        self,
        ic_type: Optional[str] = None,
        forward_periods: List[int] = None,
        min_count: int = 10,
        factor_names: List[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            ic_type=ic_type,
            forward_periods=forward_periods,
            min_count=min_count,
            factor_names=factor_names,
            **kwargs,
        )
        self.ic_type = ic_type or os.environ.get("IC_TYPE", "normal")
        self.forward_periods = list(forward_periods) if forward_periods else [1, 5, 20]
        self.min_count = int(min_count)
        self.factor_names = list(factor_names) if factor_names else None

    def _infer_factor_names(self, df: pd.DataFrame) -> List[str]:
        """推断待分析的因子列。

        优先级：
        1. 显式指定的 factor_names（过滤存在的列）
        2. ctx.factor_names
        3. 自动推断（排除 code/date/industry/estimated_mv 等非因子列）
        """
        if self.factor_names:
            return [f for f in self.factor_names if f in df.columns]
        if ctx_factor_names := getattr(self, "_ctx_factor_names", None):
            return [f for f in ctx_factor_names if f in df.columns]
        exclude = {"code", "date", "industry", "estimated_mv", "money_flow_raw",
                   "ret_1d", "ret_5d", "ret_20d", "ret_60d", "turnover_5d"}
        return [c for c in df.columns if c not in exclude]

    def __call__(self, df: pd.DataFrame, ctx: ProcessContext) -> pd.DataFrame:
        if df.empty:
            return df

        # 前瞻收益必须存在
        if ctx.forward_returns is None or ctx.forward_returns.empty:
            logger.warning(
                "ICAnalysisProcessor: ctx.forward_returns 为空，跳过 IC 分析"
            )
            ctx.ic_results = {}
            return df

        # 同步 ctx.factor_names 到本实例（供 _infer_factor_names 使用）
        self._ctx_factor_names = ctx.factor_names

        factors = self._infer_factor_names(df)
        if not factors:
            logger.warning("ICAnalysisProcessor: 未找到待分析的因子列，跳过")
            ctx.ic_results = {}
            return df

        # 合并 df 与 forward_returns
        fwd_cols = ["code", "date"] + [
            f"ret_forward_{p}d" for p in self.forward_periods
            if f"ret_forward_{p}d" in ctx.forward_returns.columns
        ]
        data = df.merge(
            ctx.forward_returns[fwd_cols],
            on=["code", "date"],
            how="inner",
        )

        if data.empty:
            logger.warning("ICAnalysisProcessor: 合并 forward_returns 后数据为空")
            ctx.ic_results = {}
            return df

        results: Dict[str, Any] = {}
        for period in self.forward_periods:
            fwd_col = f"ret_forward_{period}d"
            if fwd_col not in data.columns:
                continue

            ic_results = []
            for factor in factors:
                if factor not in data.columns:
                    continue

                ic_series = self._calc_ic(data, factor, fwd_col)
                if ic_series is None or ic_series.empty:
                    continue

                ic_mean = ic_series.mean()
                ic_std = ic_series.std()
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                ic_positive_ratio = (ic_series > 0).mean()

                ic_results.append({
                    "factor": factor,
                    "forward_period": fwd_col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_std": round(float(ic_std), 6),
                    "ic_ir": round(float(ic_ir), 4),
                    "ic_positive_ratio": round(float(ic_positive_ratio), 4),
                    "ic_t_stat": round(
                        float(ic_mean / (ic_std / np.sqrt(len(ic_series))))
                        if ic_std > 0 else 0,
                        4,
                    ),
                })

            results[fwd_col] = ic_results

        ctx.ic_results = results
        logger.info(
            f"ICAnalysisProcessor: 完成 {len(factors)} 个因子 × "
            f"{len(self.forward_periods)} 个前瞻期的 IC 分析"
        )
        return df  # IC 分析不修改 df

    def _calc_ic(
        self,
        data: pd.DataFrame,
        factor_col: str,
        forward_col: str,
    ) -> Optional[pd.Series]:
        """计算单个因子的 IC 时间序列（与 FactorEngine._calc_ic 一致）"""
        ic_list = []
        dates = sorted(data["date"].unique())

        for dt in dates:
            cross = data[data["date"] == dt].dropna(subset=[factor_col, forward_col])
            if len(cross) < self.min_count:
                continue

            if self.ic_type == "spearman":
                ic, _ = stats.spearmanr(
                    cross[factor_col], cross[forward_col], nan_policy="omit"
                )
            else:
                ic, _ = stats.pearsonr(
                    cross[factor_col].fillna(0), cross[forward_col].fillna(0)
                )

            if not np.isnan(ic):
                ic_list.append({"date": dt, "ic": ic})

        if not ic_list:
            return None

        ic_df = pd.DataFrame(ic_list)
        ic_df["date"] = pd.to_datetime(ic_df["date"])
        return ic_df.set_index("date")["ic"]

    def describe(self) -> Dict[str, Any]:
        return {
            "processor": self.name,
            "params": {
                "ic_type": self.ic_type,
                "forward_periods": self.forward_periods,
                "min_count": self.min_count,
                "factor_names": self.factor_names,
            },
            "requires": list(self.requires),
            "provides": ["ctx.ic_results"],
        }
