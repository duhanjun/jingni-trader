"""
Point-in-Time (PIT) 数据合并工具

借鉴自：
  - Qlib Data Layer 的 PIT 设计（Point-in-Time 数据库 / 避免 look-ahead bias）
  - 学术 / 业界对 Financial ML 数据切分最佳实践
    (Advances in Financial ML, Marcos López de Prado)

问题背景：
  A 股的"财务数据"通常以"公告日"为可见时间点，但回测中常见的错误
  是把"报告期"(period_end) 当成可观察时间，从而引入未来函数。
  PIT 合并要求"任何指标只有在 announce_date <= asof_date 时才能被使用"。

本模块提供：
  1. PIT 合并：按 asof_date 找到每行最后一条"已发布"记录
  2. PIT 校验：检测合并结果是否出现未来函数
  3. 与 jingni-trader 当前"普通 merge"的对比报告
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class PITConfig:
    """PIT 合并配置"""
    asof_col: str = "date"              # 观察日（左侧 df 的时间列）
    announce_col: str = "announce_date" # 数据发布日（右侧 df 的时间列）
    by: str = "code"                    # 实体键
    allow_exact_match: bool = True      # asof == announce 是否允许


def pit_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: PITConfig,
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    对 left 的每一行 (code, asof_date)，
    在 right 中找到 announce_date <= asof_date 的最近一条记录并合并。

    Parameters
    ----------
    left : pd.DataFrame
        主表，必须包含 config.by 与 config.asof_col 列
    right : pd.DataFrame
        待合并的"指标"表，必须包含 config.by、config.announce_col 与 value_cols
    config : PITConfig
        合并配置
    value_cols : list[str], optional
        要合并的列；默认取 right 中除 by/announce_col 之外的全部列

    Returns
    -------
    pd.DataFrame
        与 left 行数一致的合并结果，新增列为 value_cols
    """
    if config.by not in left.columns or config.asof_col not in left.columns:
        raise KeyError(f"left 缺少必要列 {config.by} 或 {config.asof_col}")
    if config.by not in right.columns or config.announce_col not in right.columns:
        raise KeyError(f"right 缺少必要列 {config.by} 或 {config.announce_col}")

    if value_cols is None:
        skip = {config.by, config.announce_col}
        value_cols = [c for c in right.columns if c not in skip]

    if not value_cols:
        return left.copy()

    # 关键：先记录原始行号 → 再排序，结束时按原始行号还原
    left_sorted = left.copy()
    left_sorted["_orig_idx"] = np.arange(len(left_sorted))
    # merge_asof 要求 left 的 key 列全局单调；因此先按 (asof, by) 排序，
    # 保证日期全局有序，同时同一日内 by 有序，便于后续 groupby 还原。
    left_sorted = left_sorted.sort_values([config.asof_col, config.by])

    # 排序：右表按 (announce_date, code) 升序，同样让 announce_date 全局有序
    right_sorted = right[[config.by, config.announce_col] + value_cols].sort_values(
        [config.announce_col, config.by]
    )

    # 用 merge_asof（backward 方向）实现 PIT
    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=config.asof_col,
        right_on=config.announce_col,
        by=config.by,
        direction="backward",
        allow_exact_matches=config.allow_exact_match,
    )

    # 还原原始顺序
    merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return merged


def detect_lookahead(
    base: pd.DataFrame,
    pit_merged: pd.DataFrame,
    value_cols: List[str],
    asof_col: str = "date",
    announce_col: str = "announce_date",
) -> dict:
    """
    对比 base (raw merge) 与 pit_merged (PIT merge) 的结果，
    统计 PIT 剔除掉的"未来函数"行数和比例。
    """
    report: dict = {"per_column": {}, "total_rows": len(base)}
    for col in value_cols:
        if col not in base.columns or col not in pit_merged.columns:
            continue
        diff_mask = ~base[col].fillna(-1e18).eq(pit_merged[col].fillna(-1e18))
        if diff_mask.any():
            # 进一步看"被修改"的方向：原始有值，PIT 为 NaN 即典型未来函数
            nan_after_pit = diff_mask & base[col].notna() & pit_merged[col].isna()
            report["per_column"][col] = {
                "rows_diff": int(diff_mask.sum()),
                "rows_lookahead_eliminated": int(nan_after_pit.sum()),
                "ratio_diff": float(diff_mask.mean()),
            }
    # 汇总
    total_lookahead = sum(c["rows_lookahead_eliminated"] for c in report["per_column"].values())
    report["total_lookahead_eliminated"] = total_lookahead
    return report


__all__ = ["PITConfig", "pit_merge", "detect_lookahead"]
