"""
PIT (Point-in-Time) Data Validator

借鉴自 Microsoft Qlib 的 Point-in-Time 数据库设计：
  https://qlib-xiaoge.readthedocs.io/en/latest/advanced/PIT.html

核心思想：保证在任何回测时点 t，只能使用 <= t 时刻已经发布的数据，
防止"未来函数"（look-ahead bias）导致回测指标虚高。

jingni-trader 现状问题：
  1. data-engine 直接落 parquet 文件，没有"发布时点"概念
  2. factor-engine 在每个时点计算因子时，可能用到未发布的财报数据
  3. 缺失 PIT 校验手段，用户容易写出"未来函数"而不自知

本模块提供：
  - PITDataFrame：带发布时点 (asof) 标签的数据结构
  - PITValidator：校验任意特征是否满足 PIT 约束
  - filter_asof()：按当前时点筛选可见数据
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PITSpec:
    """
    PIT 字段规范

    Attributes:
        feature_name: 特征名 (e.g. 'eps_ttm', 'roe', 'pe')
        period_col: 数据所属期间的列名 (e.g. 'period' for 财报季度)
        asof_col: 数据发布时间列 (e.g. 'announce_date' or 'filing_date')
        freq: 数据频率，'q'(季度) / 'y'(年度) / 'd'(日频, 默认发布即得)
    """
    feature_name: str
    period_col: Optional[str] = "period"
    asof_col: Optional[str] = "asof"
    freq: str = "d"


class PITDataFrame:
    """
    PIT 感知的 DataFrame 容器

    存储规范：
        code     period      asof        value
        000001   2023Q1   2023-04-25    0.45
        000001   2023Q2   2023-08-25    0.52
        000001   2023Q3   2023-10-25    0.48
        ...
    """

    def __init__(self, df: pd.DataFrame, spec: PITSpec):
        self.spec = spec
        if spec.asof_col not in df.columns:
            raise ValueError(f"PIT 数据缺少 asof 列: {spec.asof_col}")
        if spec.freq in ("q", "y") and spec.period_col not in df.columns:
            raise ValueError(f"PIT 数据缺少 period 列: {spec.period_col}")
        self.df = df.copy()
        self.df[spec.asof_col] = pd.to_datetime(self.df[spec.asof_col])

    def filter_asof(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        在给定时点 as_of，仅返回"已发布"的数据行
        """
        mask = self.df[self.spec.asof_col] <= as_of
        return self.df.loc[mask].copy()

    def get_latest(self, code: str, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """
        在 as_of 时点，获取某只股票最近一期已发布的特征值
        """
        sub = self.filter_asof(as_of)
        if sub.empty:
            return None
        sub = sub[sub["code"] == code]
        if sub.empty:
            return None
        return sub.sort_values(self.spec.asof_col).iloc[-1]


class PITValidator:
    """
    PIT 校验器

    校验数据集是否满足：
      1. asof <= 任意回测时点 t
      2. 同 (code, period) 没有"未发布就被使用"的情况
      3. 财报修订版本不混用（最新发布版本优于历史修订版）
    """

    def __init__(self, df: pd.DataFrame, spec: PITSpec):
        self.pit_df = PITDataFrame(df, spec)
        self._issues: List[Dict] = []

    def check_lookahead(
        self,
        as_of: pd.Timestamp,
        feature_col: str = "value",
    ) -> List[Dict]:
        """
        检查是否存在"as_of 之后才发布但已被使用"的数据
        返回问题列表
        """
        issues = []
        for _, row in self.pit_df.df.iterrows():
            if row[self.pit_df.spec.asof_col] > as_of:
                issues.append({
                    "code": row.get("code"),
                    "period": row.get(self.pit_df.spec.period_col, ""),
                    "asof": row[self.pit_df.spec.asof_col],
                    "feature": feature_col,
                    "value": row[feature_col],
                    "severity": "high",
                    "message": f"数据 {row.get('code')}@{row.get(self.pit_df.spec.period_col, '')} "
                               f"在 {as_of.date()} 之后才发布 ({row[self.pit_df.spec.asof_col].date()})，"
                               f"存在 look-ahead 风险",
                })
        self._issues.extend(issues)
        return issues

    def check_version_consistency(self) -> List[Dict]:
        """
        检查同一 (code, period) 是否使用了多个版本的特征值
        （即同一财报被多次修订/重述时，是否混用了不同版本）
        """
        issues = []
        if self.pit_df.spec.period_col is None:
            return issues

        grp = self.pit_df.df.groupby(["code", self.pit_df.spec.period_col])
        for (code, period), sub in grp:
            if sub["value"].nunique() > 1:
                issues.append({
                    "code": code,
                    "period": period,
                    "versions": len(sub),
                    "severity": "medium",
                    "message": f"{code}@{period} 有 {len(sub)} 个不同版本的值，"
                               f"未严格按 asof 时点选最新版",
                })
        self._issues.extend(issues)
        return issues

    def audit_pipeline(
        self,
        eval_timestamps: List[pd.Timestamp],
    ) -> Dict:
        """
        全流程 PIT 审计：对每个回测时点执行 look-ahead 检查
        """
        report = {
            "n_eval_points": len(eval_timestamps),
            "total_issues": 0,
            "by_severity": {"high": 0, "medium": 0, "low": 0},
            "details": [],
        }
        for ts in eval_timestamps:
            ts_issues = self.check_lookahead(ts)
            ver_issues = self.check_version_consistency()
            n = len(ts_issues) + len(ver_issues)
            report["total_issues"] += n
            for iss in ts_issues + ver_issues:
                sev = iss.get("severity", "low")
                report["by_severity"][sev] = report["by_severity"].get(sev, 0) + 1
            if n > 0:
                report["details"].append({
                    "as_of": ts,
                    "n_lookahead": len(ts_issues),
                    "n_version_conflict": len(ver_issues),
                })
        return report


def make_synthetic_pit(
    n_stocks: int = 5,
    n_periods: int = 12,
    freq: str = "q",
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成合成 PIT 数据用于测试
    """
    np.random.seed(seed)
    rows = []
    base_date = pd.Timestamp("2023-01-01")
    for code_idx in range(n_stocks):
        code = f"{600000 + code_idx:06d}"
        for p in range(n_periods):
            if freq == "q":
                period = f"2023Q{(p % 4) + 1}" if p < 4 else f"2024Q{(p % 4) + 1}"
                publish_offset = 30 + np.random.randint(-5, 15)
            else:
                period = str(2023 + p // 4)
                publish_offset = 90 + np.random.randint(-10, 30)
            asof = base_date + pd.DateOffset(months=3 * p) + pd.Timedelta(days=publish_offset)
            value = np.random.randn()
            rows.append({
                "code": code,
                "period": period,
                "asof": asof,
                "value": value,
            })
    return pd.DataFrame(rows)