"""
优化方向 2：Point-in-Time (PIT) 数据提供者

借鉴来源：Microsoft Qlib 的 PIT 数据系统
- Qlib PIT 文档: https://github.com/microsoft/qlib/blob/main/docs/advanced/PIT.rst
- Qlib pit.py: https://github.com/microsoft/qlib/blob/main/qlib/data/pit.py

问题分析（jingni-trader 现状）：
- skills/data-engine/scripts/base/base_data_provider.py 的 get_financial 仅按 report_date 取数
- 回测时若直接用财报数据，会用 "最终修订值" 而非 "当时可得值"，造成未来函数（look-ahead bias）
- 例如：2024Q2 财报在 2024-08-15 发布，2024-10-20 修订；若回测 2024-09-01 的策略，
  错误地用了 10-20 的修订值 → 未来数据泄露

优化方案：
- 实现 PITProvider：维护 (code, period, publish_date, value, revision_seq) 修订链
- 查询接口 get_pit(code, field, observe_date)：返回 observe_date 当天 "已公开的最新值"
- 严格保证：只返回 publish_date <= observe_date 的最新一次修订
- 提供 as_of_pit(df, observe_date_col) 方法，对回测数据按观察日对齐财报
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class PITProvider:
    """Point-in-Time 财务数据提供者

    数据结构（内部）:
        每条记录: (code, period, field, publish_date, value, revision_seq)
        修订链: 同一 (code, period, field) 可有多条，按 publish_date 升序

    查询语义:
        get_pit(code, field, observe_date) 返回满足
        publish_date <= observe_date 的最新一条记录的 value
    """

    def __init__(self):
        # records: List[dict]
        self._records: List[dict] = []
        # 索引: (code, field) -> DataFrame[code, period, publish_date, value, seq]
        self._index: Dict[tuple, pd.DataFrame] = {}
        self._dirty = False

    # ------------------------------------------------------------------
    # 数据装载
    # ------------------------------------------------------------------

    def load_records(self, df: pd.DataFrame) -> None:
        """批量装载财务数据记录

        必需列: code, period, field, publish_date, value
        可选列: revision_seq（缺省则按 publish_date 排序自动生成）
        """
        required = {"code", "period", "field", "publish_date", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"PIT 记录缺少列: {missing}")

        df = df.copy()
        df["publish_date"] = pd.to_datetime(df["publish_date"])
        if "revision_seq" not in df.columns:
            df = df.sort_values(["code", "field", "period", "publish_date"])
            df["revision_seq"] = df.groupby(["code", "field", "period"]).cumcount()

        self._records.extend(df.to_dict("records"))
        self._dirty = True

    def _rebuild_index(self) -> None:
        if not self._dirty:
            return
        full = pd.DataFrame(self._records)
        if full.empty:
            self._index = {}
            self._dirty = False
            return
        for (code, field), g in full.groupby(["code", "field"]):
            g = g.sort_values(["period", "publish_date", "revision_seq"]).reset_index(drop=True)
            self._index[(code, field)] = g[
                ["code", "period", "publish_date", "value", "revision_seq"]
            ]
        self._dirty = False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_pit(
        self,
        code: str,
        field: str,
        observe_date,
        period: Optional[str] = None,
    ) -> Optional[float]:
        """查询单只股票单个字段在 observe_date 当天可得的最新值

        返回 None 表示该日期尚无任何公开数据
        """
        self._rebuild_index()
        key = (code, field)
        if key not in self._index:
            return None
        sub = self._index[key]
        obs = pd.Timestamp(observe_date)
        avail = sub[sub["publish_date"] <= obs]
        if avail.empty:
            return None
        if period is not None:
            avail = avail[avail["period"] == period]
            if avail.empty:
                return None
        # 取最新发布的一条
        latest = avail.sort_values("publish_date").iloc[-1]
        return float(latest["value"])

    def as_of_pit(
        self,
        panel: pd.DataFrame,
        observe_date_col: str = "date",
        code_col: str = "code",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """对回测面板数据按观察日对齐 PIT 财务字段

        panel: 含 code, date 的 DataFrame
        fields: 需要拼接的财务字段列表；若 None 则装载所有字段
        返回: 在 panel 基础上追加 {field}_pit 列
        """
        self._rebuild_index()
        out = panel.copy()
        obs_dates = pd.to_datetime(out[observe_date_col])

        # 收集所有字段
        if fields is None:
            fields = sorted({k[1] for k in self._index.keys()})

        for field in fields:
            col_values = []
            for code, obs in zip(out[code_col], obs_dates):
                v = self.get_pit(code, field, obs)
                col_values.append(v)
            out[f"{field}_pit"] = col_values
        return out

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def revision_chain(
        self, code: str, field: str, period: str
    ) -> pd.DataFrame:
        """返回某 (code, field, period) 的完整修订链，便于审计"""
        self._rebuild_index()
        key = (code, field)
        if key not in self._index:
            return pd.DataFrame()
        sub = self._index[key]
        return sub[sub["period"] == period].reset_index(drop=True)

    def stats(self) -> Dict[str, int]:
        self._rebuild_index()
        full = pd.DataFrame(self._records)
        if full.empty:
            return {"records": 0, "codes": 0, "fields": 0, "periods": 0}
        return {
            "records": len(full),
            "codes": full["code"].nunique(),
            "fields": full["field"].nunique(),
            "periods": full["period"].nunique(),
        }
