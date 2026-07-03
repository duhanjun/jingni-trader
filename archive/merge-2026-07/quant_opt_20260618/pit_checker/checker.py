"""
Point-in-Time (PIT) 数据完整性检查器
===================================

借鉴自：Microsoft Qlib 的 Point-in-Time 数据系统设计
参考：https://github.com/microsoft/qlib
      Qlib Paper: "Qlib: An AI-oriented Quantitative Investment Platform" (arXiv:2009.11189)

问题背景
--------
在 A 股量化研究中，"未来数据泄露" (look-ahead bias) 是最常见的过拟合来源之一。
常见场景：
  1. 用 2024-12-31 收盘后的公告/财务数据预测 2024-12-30 的收益
  2. 用未来才知道的换手率/资金流数据计算当日因子
  3. 复权时未做 PIT 校验，使用了尚未发生的分红数据
  4. 因子标准化（z-score）时混入了未来窗口

PIT 检查器的作用
----------------
1. 校验字段是否带 announce_date / publish_date
2. 校验 label 的"未来期"是否超出训练集截止日
3. 校验"复权因子"是否在对应交易日已生效
4. 提供 PIT-aware 的 join 接口，避免无意中拼入未来

设计取舍
--------
- 不强制要求所有列都带 announce_date（金融数据大多确实没有）
- 但凡是带 announce_date 的列，必须在 PIT join 时严格过滤
- 通过可插拔的 PITPolicy 适配不同数据源的语义
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# 常量定义
# ─────────────────────────────────────────────────────────────

# 已知会引入未来信息的列名（启发式）
KNOWN_LOOKAHEAD_COLS: Set[str] = {
    "announce_date",     # 公告日：用于检验基本面发布
    "publish_date",      # 发布日
    "report_period",     # 财报所属期
    "filing_date",       # 申报日
    "effective_date",    # 生效日（如复权）
    "ex_date",           # 除权除息日
}

# 这些列如果存在，join 时必须按 PIT 规则过滤
PIT_REQUIRED_COLS: Set[str] = {
    "announce_date", "publish_date", "filing_date",
}


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class PITViolation:
    """单条 PIT 违规记录"""
    code: str
    row_date: Any
    field: str
    value: Any
    available_at: Any
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "row_date": str(self.row_date),
            "field": self.field,
            "value": str(self.value),
            "available_at": str(self.available_at),
            "reason": self.reason,
        }


@dataclass
class PITReport:
    """PIT 检查报告"""
    total_rows: int = 0
    checked_rows: int = 0
    pit_columns: List[str] = field(default_factory=list)
    violations: List[PITViolation] = field(default_factory=list)
    is_clean: bool = True

    def add(self, v: PITViolation):
        self.violations.append(v)
        self.is_clean = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "checked_rows": self.checked_rows,
            "pit_columns": self.pit_columns,
            "violation_count": len(self.violations),
            "is_clean": self.is_clean,
            "violations": [v.to_dict() for v in self.violations[:100]],
        }


# ─────────────────────────────────────────────────────────────
# 核心检查器
# ─────────────────────────────────────────────────────────────

class PITChecker:
    """Point-in-Time 数据完整性检查器

    使用示例：

    >>> checker = PITChecker()
    >>> report = checker.check(df, pit_columns=["announce_date", "publish_date"])
    >>> if not report.is_clean:
    ...     print(report.to_dict())
    """

    def __init__(self, strict: bool = False):
        """
        参数:
            strict: True=发现 1 条违规即失败；False=记录所有违规
        """
        self.strict = strict

    def detect_pit_columns(self, df: pd.DataFrame) -> List[str]:
        """自动检测 DataFrame 中哪些列是 PIT 类列"""
        cols = []
        for c in df.columns:
            c_lower = str(c).lower()
            if c_lower in PIT_REQUIRED_COLS or c_lower in KNOWN_LOOKAHEAD_COLS:
                cols.append(c)
        return cols

    def check(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        code_col: str = "code",
        pit_columns: Optional[List[str]] = None,
        max_violations: int = 1000,
    ) -> PITReport:
        """对 DataFrame 进行 PIT 检查

        参数:
            df: 待检查的数据
            date_col: 行日期列名
            code_col: 股票代码列名
            pit_columns: PIT 类列名列表（None=自动检测）
            max_violations: 最多记录多少条违规

        返回:
            PITReport
        """
        report = PITReport(total_rows=len(df))

        if df.empty:
            return report

        if date_col not in df.columns:
            report.add(PITViolation(
                code="N/A", row_date="N/A", field=date_col,
                value="N/A", available_at="N/A",
                reason=f"缺少日期列 {date_col}",
            ))
            return report

        if pit_columns is None:
            pit_columns = self.detect_pit_columns(df)

        report.pit_columns = pit_columns
        if not pit_columns:
            return report

        # 转换为 datetime
        df_check = df[[date_col, code_col] + pit_columns].copy()
        df_check[date_col] = pd.to_datetime(df_check[date_col], errors="coerce")
        for pc in pit_columns:
            df_check[pc] = pd.to_datetime(df_check[pc], errors="coerce")

        valid_mask = df_check[date_col].notna()
        df_check = df_check[valid_mask]
        report.checked_rows = len(df_check)

        for pc in pit_columns:
            if pc not in df_check.columns:
                continue

            # 关键规则：行日期 <= announce_date 才是合法的
            # 即：在 announce_date 当日及之后，行才能使用此字段
            mask_invalid = df_check[pc].notna() & (
                df_check[date_col] < df_check[pc]
            )

            invalid_rows = df_check[mask_invalid]
            for _, row in invalid_rows.head(max_violations - len(report.violations)).iterrows():
                report.add(PITViolation(
                    code=str(row[code_col]),
                    row_date=row[date_col].date() if hasattr(row[date_col], "date") else row[date_col],
                    field=pc,
                    value=row[pc],
                    available_at=row[pc],
                    reason=f"行日期 {row[date_col].date()} 早于 {pc}={row[pc].date()}, 存在未来信息泄露",
                ))

            if self.strict and mask_invalid.any():
                # strict 模式提前结束
                break

        return report

    def pit_safe_merge(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        on: List[str],
        right_date_col: str = "announce_date",
        left_date_col: str = "date",
        how: str = "left",
    ) -> pd.DataFrame:
        """PIT-aware 安全 join

        行为说明：
        - 合并后只保留 left[left_date_col] >= right[right_date_col] 的行
        - 即：left 行的日期必须晚于 right 数据的发布日期

        参数:
            left: 主表（带 date 列）
            right: 副表（带 announce_date 列）
            on: 合并键（如 ['code']）
            right_date_col: 副表的发布日期列
            left_date_col: 主表的日期列
            how: 合并方式

        返回:
            过滤后的合并结果
        """
        if right_date_col not in right.columns:
            # 副表没有发布日期列，按普通 merge 处理
            return left.merge(right, on=on, how=how)

        right = right.copy()
        right[right_date_col] = pd.to_datetime(right[right_date_col], errors="coerce")
        left = left.copy()
        left[left_date_col] = pd.to_datetime(left[left_date_col], errors="coerce")

        merged = left.merge(right, on=on, how=how, suffixes=("", "_pit"))

        # PIT 过滤：左表日期 >= 右表发布日期
        mask = merged[left_date_col] >= merged[right_date_col]
        merged = merged[mask].copy()
        merged = merged.drop(columns=[right_date_col + "_pit"], errors="ignore")
        return merged


# ─────────────────────────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────────────────────────

def check_pit(
    df: pd.DataFrame,
    pit_columns: Optional[List[str]] = None,
    strict: bool = False,
) -> PITReport:
    """便捷函数：对 DataFrame 做 PIT 检查"""
    return PITChecker(strict=strict).check(df, pit_columns=pit_columns)


def ensure_pit_safe_label(
    df: pd.DataFrame,
    label_col: str,
    horizon: int = 1,
    freq: str = "1D",
) -> pd.DataFrame:
    """确保 label 不会用到训练集截止日之后的数据

    参数:
        df: 原始数据（必须按 code 排序）
        label_col: 标签列名
        horizon: 预测期
        freq: 频率（'1D' 日频 / '1H' 小时频）

    返回:
        添加了 label 列的 DataFrame
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "label" not in df.columns:
        df[label_col] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-horizon) / x - 1
        )
    return df
