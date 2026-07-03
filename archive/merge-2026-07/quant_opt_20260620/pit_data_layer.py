"""
Point-in-Time (PIT) 数据层
借鉴来源: Microsoft Qlib PIT Database (qlib/data/pit.py)

设计目标:
    解决 jingni-trader 数据引擎缺乏 PIT 感知的问题.
    财务报告会多次修订, 若回测中使用最新版本数据, 会产生"未来数据泄漏"(look-ahead bias).
    PIT 层确保在任意历史时间点 t, 只返回 t 时刻已公开的数据版本.

Qlib 的核心设计 (我们借鉴并简化):
    - 每条记录含: 发布日期(date), 报告期(period), 数值(value), 修订链下一跳(_next)
    - 按 period 维护修订链 (链表), 查询时沿链找到 date <= 观测时间 的最新版本
    - 文件式存储 (.data + .index), 这里用内存 DataFrame 模拟以便验证

与 jingni-trader 现状对比:
    现有 data-engine: 直接 merge 财务数据, 无版本概念, 存在泄漏风险
    优化后: 通过 PITProvider 查询, 自动取观测时点可得版本
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PITRecord:
    """单条 PIT 记录 (对应 Qlib 中 .data 的一行)"""

    __slots__ = ('date', 'period', 'value', 'next_idx')

    def __init__(self, date: int, period: int, value: float, next_idx: int = -1):
        # date/period 用 YYYYMMDD / YYYYQQ 整数编码, 与 Qlib 一致
        self.date = date
        self.period = period
        self.value = value
        self.next_idx = next_idx  # 修订链下一跳的行索引, -1 表示链尾

    def __repr__(self):
        return f"PITRecord(date={self.date}, period={self.period}, value={self.value}, next={self.next_idx})"


class PITStorage:
    """
    PIT 存储 (内存版, 模拟 Qlib 的文件式存储)

    内部结构:
        self._data[field][code] = List[PITRecord]  按 date 升序
        self._period_head[field][code][period] = 链头索引
    """

    def __init__(self):
        # field -> code -> List[PITRecord]
        self._data: Dict[str, Dict[str, List[PITRecord]]] = {}
        # field -> code -> period -> 链头索引
        self._heads: Dict[str, Dict[str, Dict[int, int]]] = {}

    def append(
        self,
        field: str,
        code: str,
        date: int,
        period: int,
        value: float,
    ):
        """追加一条记录, 自动维护修订链"""
        self._data.setdefault(field, {}).setdefault(code, [])
        self._heads.setdefault(field, {}).setdefault(code, {})
        records = self._data[field][code]
        heads = self._heads[field][code]

        idx = len(records)
        records.append(PITRecord(date, period, value, next_idx=-1))

        if period in heads:
            # 沿链找到尾, 把新记录接上
            cur = heads[period]
            while records[cur].next_idx != -1:
                cur = records[cur].next_idx
            records[cur].next_idx = idx
        else:
            heads[period] = idx

    def query(self, field: str, code: str, observe_date: int) -> Optional[float]:
        """
        查询在 observe_date 当天可见的最新值.

        遍历所有 period 的修订链, 找到:
          - 发布日期 <= observe_date
          - 在满足条件的同一 period 链中取最新版本
        然后在所有 period 中取 period 最大的 (最近报告期)
        """
        records = self._data.get(field, {}).get(code, [])
        heads = self._heads.get(field, {}).get(code, {})
        if not records or not heads:
            return None

        best_period = -1
        best_value = None

        for period, head_idx in heads.items():
            # 沿链找 date <= observe_date 的最新版本
            cur = head_idx
            chosen = None
            while cur != -1:
                rec = records[cur]
                if rec.date <= observe_date:
                    chosen = rec  # 继续往后找更新的修订
                cur = rec.next_idx
            if chosen is not None:
                if period > best_period:
                    best_period = period
                    best_value = chosen.value

        return best_value

    def query_panel(
        self,
        field: str,
        codes: List[str],
        observe_dates: List[int],
    ) -> pd.DataFrame:
        """
        批量查询, 返回面板 (code, observe_date, value)
        """
        rows = []
        for code in codes:
            for od in observe_dates:
                v = self.query(field, code, od)
                rows.append({'code': code, 'date': od, field: v})
        return pd.DataFrame(rows)

    def stats(self) -> Dict[str, int]:
        """统计信息"""
        total = 0
        for field, codes in self._data.items():
            for code, recs in codes.items():
                total += len(recs)
        return {'fields': len(self._data), 'total_records': total}


class PITProvider:
    """
    PIT 数据提供者

    封装 PITStorage, 提供面向回测引擎的接口.
    核心方法: get_feature_series(field, code, dates) -> pd.Series
    """

    def __init__(self, storage: PITStorage):
        self.storage = storage

    def get_feature_series(
        self,
        field: str,
        code: str,
        dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """获取某字段在某股票上的 PIT 时间序列"""
        values = []
        for d in dates:
            od = int(d.strftime('%Y%m%d'))
            v = self.storage.query(field, code, od)
            values.append(v)
        return pd.Series(values, index=dates, name=field)

    def get_feature_panel(
        self,
        field: str,
        codes: List[str],
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """获取某字段的面板数据 (long format)"""
        observe_ints = [int(d.strftime('%Y%m%d')) for d in dates]
        return self.storage.query_panel(field, codes, observe_ints)


# ============================================================
# 辅助: 从普通财务 DataFrame 构建 PIT 存储
# ============================================================

def build_pit_storage_from_records(
    records_df: pd.DataFrame,
    field_col: str = 'field',
    code_col: str = 'code',
    pub_date_col: str = 'pub_date',
    period_col: str = 'period',
    value_col: str = 'value',
) -> PITStorage:
    """
    从记录 DataFrame 构建 PIT 存储.

    records_df 列: field, code, pub_date(YYYYMMDD int 或 str), period(YYYYQQ int), value
    """
    storage = PITStorage()
    df = records_df.copy()
    df[pub_date_col] = df[pub_date_col].astype(int)
    df[period_col] = df[period_col].astype(int)

    # 按 pub_date 排序, 保证链表顺序正确
    df = df.sort_values([field_col, code_col, pub_date_col])

    for _, row in df.iterrows():
        storage.append(
            field=str(row[field_col]),
            code=str(row[code_col]),
            date=int(row[pub_date_col]),
            period=int(row[period_col]),
            value=float(row[value_col]),
        )
    return storage


def detect_lookahead_bias(
    pit_values: pd.Series,
    latest_values: pd.Series,
    pub_dates: pd.Series,
) -> Dict[str, int]:
    """
    检测未来数据泄漏

    比较 PIT 查询值与"直接用最新值"的差异, 统计泄漏次数.

    参数:
        pit_values: PIT 查询得到的序列 (按日期)
        latest_values: 直接用最新修订值的序列 (有泄漏)
        pub_dates: 每个最新值对应的发布日期
    返回:
        {'leakage_count': n, 'total': n, 'leakage_ratio': r}
    """
    leakage_count = 0
    total = 0
    for i in range(len(pit_values)):
        if pd.isna(pit_values.iloc[i]) and pd.isna(latest_values.iloc[i]):
            continue
        total += 1
        # 若 latest 值的发布日期晚于当前观测日期, 且与 PIT 值不同, 则存在泄漏
        if not pd.isna(latest_values.iloc[i]):
            pub = pub_dates.iloc[i]
            obs = pit_values.index[i]
            if hasattr(pub, 'strftime'):
                pub_int = int(pub.strftime('%Y%m%d'))
            else:
                pub_int = int(pub)
            if hasattr(obs, 'strftime'):
                obs_int = int(obs.strftime('%Y%m%d'))
            else:
                obs_int = int(obs)
            if pub_int > obs_int and not pd.isna(pit_values.iloc[i]):
                if abs(latest_values.iloc[i] - pit_values.iloc[i]) > 1e-9:
                    leakage_count += 1
    return {
        'leakage_count': leakage_count,
        'total': total,
        'leakage_ratio': leakage_count / total if total > 0 else 0.0,
    }
