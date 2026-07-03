"""
借鉴来源: kernc/backtesting.py + Qlib PIT (Point-In-Time) Provider
- 官方仓库: https://github.com/kernc/backtesting.py
- 设计理念: progressive data revelation —— 引擎一次只"看"到当前 bar
- 论文: "Advances in Cross-Sectional Equity Strategies" (Qlib paper)

jingni-trader 现状:
  factor-engine/engine.py 与 strategy-model-engine/engine.py 中存在多处
  容易引入"前视偏差" (look-ahead bias) 的写法,例如:
    1) `df.groupby('code')['close'].pct_change()`  (line 64-67)
       -> 实际包含 T 日的 close,会用到 T 日收盘价,
          对 T 日开盘买入的策略来说属于"偷看"当日收盘价
    2) `df.groupby('code')['close'].rolling(20).mean()` (line 87-94)
       -> rolling 默认 min_periods=1,首日就把"未来"算进来
    3) `forward_returns` 计算 (line 405-408)
       -> `x.shift(-period) / x - 1` 在 T 日就能看到 T+period 的 close
    4) 训练数据 `forward_return` 与特征同时间戳,模型在 T 日训练时
       已经看到 T+1 ~ T+5 的标签
    5) `close.shift(-1)` / `Ref($close, -1)` 等负向 shift

借鉴方案:
  提供一个 LookAheadDetector,自动扫描因子表达式与回测代码,标注
  所有"偷看未来"的位置,输出人类可读的报告。
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LookAheadIssue:
    """一处前视偏差问题。"""
    severity: str          # "critical" / "warning" / "info"
    location: str          # 文件:行号 or 表达式
    pattern: str           # 触发模式
    description: str       # 人类可读解释
    fix: str               # 建议修复


@dataclass
class DetectionReport:
    """完整的检测报告。"""
    issues: List[LookAheadIssue] = field(default_factory=list)
    total_scanned: int = 0

    def to_dict(self):
        return {
            "total_scanned": self.total_scanned,
            "n_issues": len(self.issues),
            "n_critical": sum(1 for i in self.issues if i.severity == "critical"),
            "n_warning":  sum(1 for i in self.issues if i.severity == "warning"),
            "issues": [
                {
                    "severity": i.severity, "location": i.location,
                    "pattern": i.pattern, "description": i.description, "fix": i.fix,
                }
                for i in self.issues
            ],
        }


# ===========================================================================
# 1. 表达式层检测 (针对 ExpressionEngine)
# ===========================================================================
class ExpressionLookAheadDetector:
    """
    检测 Qlib 风格表达式中的负向 shift / 未来信息泄漏。

    检测规则:
      1) Ref($field, -N)            — 负 shift = 偷看未来,Qlib 显式禁止
      2) Mean($field, -N)           — 滚动负数
      3) $close / Ref($close, 0)    — T 日与 T 日做运算,语义不清
      4) forward_return             — 与未来收益同表
    """

    NEGATIVE_SHIFT_RE = re.compile(r"(Ref|Mean|Sum|Std|Max|Min|Slope)\s*\(\s*\$([a-zA-Z_]\w*)\s*,\s*(-?\d+)\s*\)")

    def scan_expression(self, expr: str, source: str = "") -> List[LookAheadIssue]:
        issues = []
        for match in self.NEGATIVE_SHIFT_RE.finditer(expr):
            op, field, n = match.groups()
            n_int = int(n)
            if n_int < 0:
                issues.append(LookAheadIssue(
                    severity="critical",
                    location=source or expr,
                    pattern=match.group(0),
                    description=f"{op}(${field}, {n_int}) 使用了负向窗口,等同于从未来取值",
                    fix=f"改为 {op}(${field}, {abs(n_int)}) 并通过 shift(-N) 在数据层把标签提前",
                ))
            elif n_int == 0:
                issues.append(LookAheadIssue(
                    severity="warning",
                    location=source or expr,
                    pattern=match.group(0),
                    description=f"{op}(${field}, 0) 窗口为 0,语义不清,建议明确窗口",
                    fix=f"使用正窗口如 {op}(${field}, 1)",
                ))
        if "forward_return" in expr or "future" in expr.lower():
            issues.append(LookAheadIssue(
                severity="critical",
                location=source or expr,
                pattern="forward_return/future",
                description="表达式中直接使用了 forward_return,模型在 T 日训练时已经包含 T+1 ~ T+N 标签",
                fix="训练时严格用 T 日及之前的特征预测 T+1 ~ T+N 收益,绝不可让特征列同时包含未来标签",
            ))
        return issues


# ===========================================================================
# 2. 代码层检测 (针对 .py 文件做 AST 扫描)
# ===========================================================================
class CodeLookAheadDetector:
    """
    用 ast 扫描 Python 代码,标记以下高危模式:
      1) .shift(-N)                       — 负向 shift
      2) .rolling(N)  without min_periods  — 滚动默认 min_periods=1 易引入首日偷看
      3) 同时存在 .groupby('code')[col] 与 .shift(-)                — 用未来 close 算因子
      4) x.shift(-period) / x - 1                                 — 典型 forward_return 写法
    """

    ROLLING_WITHOUT_MIN = re.compile(r"\.rolling\s*\([^)]*\)(?!\s*\.min_periods)")
    NEGATIVE_SHIFT = re.compile(r"\.shift\s*\(\s*-\s*[\w]+")  # 允许 -1, -FORWARD_PERIOD 等
    FORWARD_RETURN = re.compile(r"\.shift\s*\(\s*-\s*\w+\s*\)\s*/\s*x\s*-\s*1")

    def scan_file(self, file_path: str) -> List[LookAheadIssue]:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        return self.scan_source(source, file_path)

    def scan_source(self, source: str, file_path: str = "<memory>") -> List[LookAheadIssue]:
        issues = []
        lines = source.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if self.NEGATIVE_SHIFT.search(line):
                m = self.NEGATIVE_SHIFT.search(line)
                # 区分 forward_return (计算未来收益,允许) 和直接 .shift(-N) 偷看 (不允许)
                # 但 forward_return 不应该与特征同表,这是一个 warning
                is_forward_return = "/ x - 1" in line or "/ close - 1" in line
                severity = "warning" if is_forward_return else "critical"
                desc = (
                    "直接使用 .shift(-N) 取得未来 N 期数据,引入前视偏差"
                    if not is_forward_return
                    else "此处计算 forward_return (未来收益),应与特征表严格分离,训练时使用 T 日及之前的数据预测 T+N 收益"
                )
                issues.append(LookAheadIssue(
                    severity=severity,
                    location=f"{file_path}:{i}",
                    pattern=m.group(0),
                    description=desc,
                    fix="forward_return 必须独立成表;若在因子计算代码中看到此模式,应改为 Ref($field, N) (N>0) 从历史中取",
                ))
            if self.FORWARD_RETURN.search(line):
                issues.append(LookAheadIssue(
                    severity="warning",
                    location=f"{file_path}:{i}",
                    pattern="x.shift(-period) / x - 1",
                    description="计算 forward_return 的典型写法;若该列与特征同表存储,模型在 T 日训练时容易把未来收益当作特征",
                    fix="forward_return 必须独立成表,且训练时严格使用 T 日及之前的数据来预测 T+1 收益",
                ))
            if self.ROLLING_WITHOUT_MIN.search(line) and ".rolling" in line and "min_periods" not in line:
                issues.append(LookAheadIssue(
                    severity="warning",
                    location=f"{file_path}:{i}",
                    pattern=".rolling(N) 未指定 min_periods",
                    description="rolling 默认 min_periods=1,首日就纳入计算,可能产生『首日偷看』",
                    fix="显式写 .rolling(N, min_periods=N) 或 .rolling(N, min_periods=max(2, N//2))",
                ))
        return issues


# ===========================================================================
# 3. 数据层检测 (针对 DataFrame,验证没有跨期泄漏)
# ===========================================================================
class DataLookAheadDetector:
    """
    对已经计算好的因子 DataFrame 做"金标测试"。

    常见场景:
      - 因子值与"未来 1 日"价格涨跌的 IC 异常高 (>0.3) 时,
        极可能是 forward 特征被错误混入;
      - 因子在 T 日 close 之后才能计算出来,但回测在 T 日开盘就用了。

    检测项:
      1) 因子与未来 1 / 5 / 20 日收益的 IC
      2) 因子值的 "可用日期" 是否早于"应当可计算的日期"
    """

    def check_factor_ic(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        factor_col: str,
        forward_periods: List[int] = (1, 5, 20),
        ic_threshold: float = 0.30,
    ) -> List[LookAheadIssue]:
        """
        若因子与 forward_return 1 日的 |IC| > ic_threshold,极可能有前视偏差。
        注意: 这只是一个启发式,不能替代人工审查。
        """
        issues: List[LookAheadIssue] = []
        if factor_df.empty or price_df.empty:
            return issues
        df = factor_df[["date", "code", factor_col]].merge(
            price_df[["date", "code", "close"]], on=["date", "code"], how="inner"
        ).sort_values(["date", "code"])
        for p in forward_periods:
            df[f"fwd_{p}"] = df.groupby("code")["close"].transform(lambda x: x.shift(-p) / x - 1)
            cross = df[[factor_col, f"fwd_{p}"]].dropna()
            if len(cross) < 30:
                continue
            from scipy.stats import spearmanr
            ic, _ = spearmanr(cross[factor_col], cross[f"fwd_{p}"])
            if abs(ic) > ic_threshold:
                issues.append(LookAheadIssue(
                    severity="critical",
                    location=f"factor={factor_col} forward={p}d",
                    pattern=f"|IC({factor_col}, fwd_{p})| = {abs(ic):.3f} > {ic_threshold}",
                    description=f"因子 {factor_col} 与 {p} 日 forward_return 的秩相关系数过高,极可能存在前视偏差",
                    fix="检查该因子的实现,确认是否在 T 日用到了 T+1 之后的数据 (如 close.shift(-1))",
                ))
        return issues


# ===========================================================================
# 4. 一键扫描器
# ===========================================================================
class LookAheadDetector:
    """组合三种检测器,提供统一入口。"""

    def __init__(self):
        self.expr_det = ExpressionLookAheadDetector()
        self.code_det = CodeLookAheadDetector()
        self.data_det = DataLookAheadDetector()

    def scan_all(
        self,
        expressions: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        factor_df: Optional[pd.DataFrame] = None,
        price_df: Optional[pd.DataFrame] = None,
        factor_cols: Optional[List[str]] = None,
    ) -> DetectionReport:
        report = DetectionReport()
        if expressions:
            for expr in expressions:
                report.issues.extend(self.expr_det.scan_expression(expr, source=expr))
                report.total_scanned += 1
        if source_files:
            for fp in source_files:
                report.issues.extend(self.code_det.scan_file(fp))
                report.total_scanned += 1
        if factor_df is not None and price_df is not None and factor_cols:
            for col in factor_cols:
                report.issues.extend(self.data_det.check_factor_ic(factor_df, price_df, col))
                report.total_scanned += 1
        return report
