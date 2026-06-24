"""
Look-ahead Bias Detector
========================

借鉴来源
--------
- **VectorBT / Qlib 讨论**  https://blog.iconfinder.com/
  "How I Built an Event-Driven Backtesting Engine in Python"
  详细列举了 Pandas 向量化代码中 6 类典型前视偏差。
- **Microsoft Qlib**  Point-in-Time 数据系统设计
  https://github.com/microsoft/qlib
- **AKQuant**  "防止前视偏差" 原则
  https://akquant.akfamily.xyz/en/advanced/ml/

设计目标
--------
为 jingni-trader 增加一个轻量级"前视偏差体检"工具，
覆盖最常见的几类错误：
  1. 用了 ``.shift(-n)``（负数 shift = 来自未来）
  2. 用了 ``.rolling(window).xxx()`` 但未 ``.shift(1)``，导致当天数据
     在当天决策中被使用
  3. 回测时使用了当日 ``close`` 计算信号，却在 ``open`` 上成交
  4. 因子计算时把 ``alpha_score`` 同时作为 label 的一部分
  5. 训练/测试集日期重叠 (time leakage)

输入接口
--------
``detect_in_dataframe(df, label_col, feature_cols, date_col='date')``
   对已生成的因子/特征 DataFrame 做静态检查 (类型 1/2/4)。

``detect_in_code(source: str)``
   对策略/因子源码字符串做静态扫描 (类型 1/2/3 的代码级检查)。

``detect_train_test_leakage(train_dates, test_dates, purge_gap_days)``
   检查时间序列切分是否有泄漏 (类型 5)。
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("quant_opt.lookahead")


@dataclass
class BiasIssue:
    """单个前视偏差问题"""

    code: str
    severity: str          # "error" | "warning"
    location: str          # 行列或代码位置
    description: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "description": self.description,
        }


@dataclass
class BiasReport:
    issues: List[BiasIssue] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    def to_dict(self) -> dict:
        return {
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# 1) 静态代码扫描
# ---------------------------------------------------------------------------
_NEGATIVE_SHIFT_RE = re.compile(r"\.shift\(\s*-\s*\d+")
# 更宽松: 匹配 .rolling(...) 后跟 1~4 个方法调用 (但不一定有 shift)
_ROLLING_PATTERN_RE = re.compile(
    r"\.rolling\([^)]*\)"
)
_SHIFT_AFTER_ROLLING_RE = re.compile(
    r"\.rolling\([^)]*\)(?:\.[a-z_][a-z_0-9]*\([^)]*\))*\s*(?:\.[a-z_][a-z_0-9]*\([^)]*\))*"
)


def detect_in_code(source: str, filename: str = "<source>") -> BiasReport:
    """在源码中检测典型前视偏差模式"""
    report = BiasReport()
    lines = source.splitlines()
    for lineno, line in enumerate(lines, 1):
        # 忽略注释行
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # 1) 负数 shift
        if _NEGATIVE_SHIFT_RE.search(line):
            report.issues.append(
                BiasIssue(
                    code="LOOKAHEAD_NEG_SHIFT",
                    severity="error",
                    location=f"{filename}:{lineno}",
                    description="使用 .shift(-n) 引用未来数据，请改用正向 shift 或当日数据。",
                )
            )

        # 2) rolling 后未 shift
        if _ROLLING_PATTERN_RE.search(line):
            # 简化启发: 找 .rolling(...) 出现位置，看其后面 (同一行内) 是否出现 .shift(
            for m in _ROLLING_PATTERN_RE.finditer(line):
                after = line[m.end():]
                # 若 .rolling 之后还调用了 .shift( 正数 ) 则视为 OK
                if re.search(r"\.shift\(\s*-?\d*\s*\)", after):
                    continue
                # 如果再无任何方法调用 (即直接是赋值), 仍报 warning
                report.issues.append(
                    BiasIssue(
                        code="LOOKAHEAD_ROLLING_NO_SHIFT",
                        severity="warning",
                        location=f"{filename}:{lineno}",
                        description=(
                            ".rolling(...) 后未观察到 .shift(1)，"
                            "可能将当日数据用于当日决策，建议在 rolling 后 shift(1)。"
                        ),
                    )
                )
                break  # 一行只报一次

        # 3) 同一行中先用 close 计算信号再用 open 成交
        if "close" in line and "open" in line and ".shift(0)" in line:
            report.issues.append(
                BiasIssue(
                    code="LOOKAHEAD_PRICE_MIX",
                    severity="warning",
                    location=f"{filename}:{lineno}",
                    description="同时使用 close/open 但未做 shift，请确认执行价不含未来信息。",
                )
            )

    return report


# ---------------------------------------------------------------------------
# 2) DataFrame 静态检查
# ---------------------------------------------------------------------------
def detect_in_dataframe(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    date_col: str = "date",
) -> BiasReport:
    """对一组 (X, y) 检查是否含有未来信息特征

    检查项目:
    - label_col 是否在 feature_cols 中 (直接泄漏)
    - 同一 date 上是否存在未来期收益字段被混入
    """
    report = BiasReport()

    if label_col in feature_cols:
        report.issues.append(
            BiasIssue(
                code="LOOKAHEAD_LABEL_IN_FEATURES",
                severity="error",
                location=f"<dataframe> columns",
                description=f"标签列 {label_col} 出现在特征列中，会导致 100% 训练准确率。",
            )
        )

    # 启发: 如果某列名包含 "future" / "next" / "lead" / "fwd" / "forward" 而非标签列
    for c in feature_cols:
        if c == label_col:
            continue
        cl = c.lower()
        if any(k in cl for k in ("fwd_", "forward_", "next_", "lead_", "future_")):
            report.issues.append(
                BiasIssue(
                    code="LOOKAHEAD_FORWARD_FEATURE",
                    severity="warning",
                    location=f"<dataframe> column {c}",
                    description=(
                        f"特征 {c} 命名含 forward/next/future 等字样，请确认其仅使用 t 时刻及以前的数据。"
                    ),
                )
            )

    return report


# ---------------------------------------------------------------------------
# 3) 训练/测试集时间泄漏检查
# ---------------------------------------------------------------------------
def detect_train_test_leakage(
    train_dates: Iterable[pd.Timestamp],
    test_dates: Iterable[pd.Timestamp],
    purge_gap_days: int = 0,
) -> BiasReport:
    """检查训练集和测试集是否存在时间重叠 / 间隔不足"""
    report = BiasReport()
    tr = pd.to_datetime(pd.Series(list(train_dates))).dropna()
    te = pd.to_datetime(pd.Series(list(test_dates))).dropna()
    if tr.empty or te.empty:
        return report

    train_max = tr.max()
    test_min = te.min()
    train_min = tr.min()
    test_max = te.max()

    # 重叠?
    if train_max >= test_min:
        report.issues.append(
            BiasIssue(
                code="LEAKAGE_OVERLAP",
                severity="error",
                location="<time split>",
                description=(
                    f"训练集最大日期 {train_max.date()} >= 测试集最小日期 {test_min.date()}，"
                    f"存在时间重叠。"
                ),
            )
        )

    gap = (test_min - train_max).days
    if 0 < gap < purge_gap_days:
        report.issues.append(
            BiasIssue(
                code="LEAKAGE_INSUFFICIENT_PURGE",
                severity="warning",
                location="<time split>",
                description=(
                    f"训练末与测试起点间隔仅 {gap} 天，少于 purge_gap_days={purge_gap_days}。"
                ),
            )
        )

    # 训练集是否在测试集"未来"
    if test_max < train_min:
        report.issues.append(
            BiasIssue(
                code="LEAKAGE_TEST_BEFORE_TRAIN",
                severity="error",
                location="<time split>",
                description=(
                    f"测试集最大日期 {test_max.date()} 早于训练集最小日期 {train_min.date()}，"
                    "切分时序颠倒。"
                ),
            )
        )

    return report


# ---------------------------------------------------------------------------
# 4) 高层便捷函数
# ---------------------------------------------------------------------------
def run_full_check(
    source: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    label_col: Optional[str] = None,
    feature_cols: Optional[Sequence[str]] = None,
    train_dates: Optional[Iterable[pd.Timestamp]] = None,
    test_dates: Optional[Iterable[pd.Timestamp]] = None,
    purge_gap_days: int = 0,
    filename: str = "<source>",
) -> BiasReport:
    """聚合多种检查入口"""
    report = BiasReport()
    if source is not None:
        r = detect_in_code(source, filename=filename)
        report.issues.extend(r.issues)
    if df is not None and label_col is not None and feature_cols is not None:
        r = detect_in_dataframe(df, label_col, list(feature_cols))
        report.issues.extend(r.issues)
    if train_dates is not None and test_dates is not None:
        r = detect_train_test_leakage(train_dates, test_dates, purge_gap_days)
        report.issues.extend(r.issues)
    return report