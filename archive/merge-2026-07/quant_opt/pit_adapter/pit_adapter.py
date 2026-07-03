"""
PIT (Point-in-Time) 数据适配器实现
==================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class PITField:
    """声明一个 PIT 字段

    parameters
    ----------
    name:
        输出列名
    value_col:
        在原始数据中的取值列
    announce_col:
        在原始数据中的公告时间列 (该列 <= asof 才视为"已发布")
    fallback_delay:
        若没有 announce_col, 用 ``period_date + fallback_delay`` 模拟
    """

    name: str
    value_col: str
    announce_col: Optional[str] = None
    fallback_delay: Optional[int] = None  # 单位: 天


class PITDataAdapter:
    """PIT 适配器

    使用示例
    --------
    >>> adapter = PITDataAdapter([
    ...     PITField("roe", "roe_q", announce_col="announce_date"),
    ...     PITField("revenue", "revenue_q", fallback_delay=45),
    ... ])
    >>> snap = adapter.asof(df, asof=pd.Timestamp("2024-08-15"))
    """

    def __init__(self, fields: List[PITField]) -> None:
        if not fields:
            raise ValueError("fields must be non-empty")
        self.fields = fields

    def asof(
        self,
        df: pd.DataFrame,
        asof: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """返回 ``asof`` 时刻所有"已发布"字段的快照

        parameters
        ----------
        df:
            必须包含 ``code`` 与 ``date`` 列, 以及各 PITField 指定的列
        asof:
            asof 时间
        """
        asof = pd.Timestamp(asof)
        unique_codes = pd.DataFrame({"code": df["code"].unique()})
        out = unique_codes.copy()
        for f in self.fields:
            col = f.value_col
            if col not in df.columns:
                out[f.name] = np.nan
                continue

            announce = self._announce_series(df, f)
            mask = announce <= asof
            # 取每个 code 在 <= asof 范围内最新一期的 value
            tmp = df.loc[mask, ["code", "date", col]].copy()
            if tmp.empty:
                out[f.name] = np.nan
                continue
            tmp = tmp.sort_values(["code", "date"])
            latest = tmp.groupby("code").tail(1).set_index("code")[col]
            out[f.name] = out["code"].map(latest)
        return out

    def panel(
        self,
        df: pd.DataFrame,
        asof_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """按一系列 asof 日期生成面板数据, 长格式 ``[code, asof_date, fields...]``

        parameters
        ----------
        df:
            原始数据
        asof_dates:
            asof 日期列表
        """
        if len(asof_dates) == 0:
            return pd.DataFrame()
        frames = []
        for asof in asof_dates:
            snap = self.asof(df, asof)
            snap["asof_date"] = pd.Timestamp(asof)
            frames.append(snap)
        return pd.concat(frames, ignore_index=True)

    # -----------------------------------------------------------------
    # 内部
    # -----------------------------------------------------------------

    def _announce_series(self, df: pd.DataFrame, f: PITField) -> pd.Series:
        if f.announce_col and f.announce_col in df.columns:
            return pd.to_datetime(df[f.announce_col])
        if f.fallback_delay is not None:
            return pd.to_datetime(df["date"]) + pd.Timedelta(days=f.fallback_delay)
        # 退而求其次: 当期即可见 (无 look-ahead 保护)
        return pd.to_datetime(df["date"])
