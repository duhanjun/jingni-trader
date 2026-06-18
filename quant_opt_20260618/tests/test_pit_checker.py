"""
PIT Checker 单元测试 + 集成测试
================================

测试场景：
1. 正常数据：全部合规 → 报告 is_clean=True
2. 存在未来数据泄露 → 报告 is_clean=False，且定位到具体行列
3. 边界条件：announce_date 与 row_date 同日 → 合规
4. 边界条件：announce_date 缺失 → 应被忽略
5. PIT-safe merge 行为
6. 自动检测 PIT 列
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from quant_opt_20260618.pit_checker.checker import (
    PITChecker, PITReport, PITViolation,
    check_pit, ensure_pit_safe_label,
    KNOWN_LOOKAHEAD_COLS, PIT_REQUIRED_COLS,
)


# ─────────────────────────────────────────────────────────────
# 工具：生成测试数据
# ─────────────────────────────────────────────────────────────

def make_clean_data(n_stocks: int = 5, n_days: int = 30) -> pd.DataFrame:
    """生成完全合规的测试数据

    规则：行日期 row_date 必须 >= announce_date
         即：基本面数据在 announce_date 当日及之后才可用
         所以 row_date >= announce_date 才合法
         这里我们让 announce_date <= row_date（announce 在过去或同日）
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    for code in codes:
        for i, dt in enumerate(dates):
            # 公告日 = 行日期的前 5 天（数据在 row_date 之前已发布，合法）
            announce_date = dt - timedelta(days=5)
            rows.append({
                "code": code,
                "date": dt,
                "close": 10.0 + i * 0.1,
                "announce_date": announce_date,
            })
    return pd.DataFrame(rows)


def make_dirty_data(n_stocks: int = 3, n_days: int = 20) -> pd.DataFrame:
    """生成含未来泄露的测试数据"""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    for ci, code in enumerate(codes):
        for i, dt in enumerate(dates):
            # 前 10 天：合规；后 10 天：announce_date 在 row_date 之前（违规）
            if i < 10:
                announce_date = dt + timedelta(days=1)
            else:
                announce_date = dt - timedelta(days=2)
            rows.append({
                "code": code,
                "date": dt,
                "close": 10.0 + i * 0.1,
                "announce_date": announce_date,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 正确性测试
# ─────────────────────────────────────────────────────────────

class TestPITCheckerCorrectness:
    """正确性测试"""

    def test_clean_data_passes(self):
        """完全合规的数据应通过 PIT 检查"""
        df = make_clean_data(n_stocks=3, n_days=30)
        report = check_pit(df, pit_columns=["announce_date"])
        assert report.is_clean, f"Expected clean, got {report.violations[:2]}"
        assert report.total_rows == 90
        assert report.checked_rows == 90
        assert "announce_date" in report.pit_columns

    def test_dirty_data_detected(self):
        """含未来泄露的数据应被检测到"""
        df = make_dirty_data(n_stocks=3, n_days=20)
        report = check_pit(df, pit_columns=["announce_date"])
        assert not report.is_clean
        # 每天 3 只股票 × 后 10 天 = 30 条违规
        assert len(report.violations) >= 30, \
            f"Expected >=30 violations, got {len(report.violations)}"
        # 验证违规原因
        v = report.violations[0]
        assert "未来信息泄露" in v.reason
        assert v.field == "announce_date"

    def test_same_day_announcement_is_legal(self):
        """announce_date == row_date 应被视作合规（当日及之后可用）"""
        df = pd.DataFrame([
            {"code": "000001.SZ", "date": pd.Timestamp("2024-01-15"),
             "close": 10.0, "announce_date": pd.Timestamp("2024-01-15")},
        ])
        report = check_pit(df, pit_columns=["announce_date"])
        assert report.is_clean, f"Same-day should be legal: {report.violations}"

    def test_missing_announce_date_ignored(self):
        """announce_date 为 NaT 应被忽略（无法判断即视为不违规）"""
        df = pd.DataFrame([
            {"code": "000001.SZ", "date": pd.Timestamp("2024-01-15"),
             "close": 10.0, "announce_date": pd.NaT},
        ])
        report = check_pit(df, pit_columns=["announce_date"])
        assert report.is_clean

    def test_multiple_pit_columns(self):
        """多个 PIT 列应都被检查"""
        df = pd.DataFrame([
            {"code": "000001.SZ", "date": pd.Timestamp("2024-01-15"),
             "close": 10.0,
             "announce_date": pd.Timestamp("2024-01-10"),  # 合规：announce 在过去
             "publish_date": pd.Timestamp("2024-01-20")},   # 违规：publish 在未来
        ])
        report = check_pit(df, pit_columns=["announce_date", "publish_date"])
        assert not report.is_clean
        assert len(report.violations) == 1
        assert report.violations[0].field == "publish_date"

    def test_auto_detect_pit_columns(self):
        """自动检测 PIT 列"""
        df = pd.DataFrame({
            "code": ["000001.SZ"],
            "date": [pd.Timestamp("2024-01-15")],
            "close": [10.0],
            "announce_date": [pd.Timestamp("2024-01-20")],
            "publish_date": [pd.Timestamp("2024-01-21")],
            "normal_col": ["abc"],
        })
        checker = PITChecker()
        detected = checker.detect_pit_columns(df)
        assert "announce_date" in detected
        assert "publish_date" in detected
        assert "normal_col" not in detected

    def test_empty_dataframe(self):
        """空 DataFrame 应安全处理"""
        df = pd.DataFrame(columns=["code", "date", "close", "announce_date"])
        report = check_pit(df, pit_columns=["announce_date"])
        assert report.is_clean
        assert report.total_rows == 0


# ─────────────────────────────────────────────────────────────
# PIT-safe merge 测试
# ─────────────────────────────────────────────────────────────

class TestPITSafeMerge:

    def test_pit_merge_filters_future(self):
        """PIT merge 应过滤掉未来信息"""
        checker = PITChecker()
        left = pd.DataFrame({
            "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "date": [pd.Timestamp("2024-01-10"),
                     pd.Timestamp("2024-01-15"),
                     pd.Timestamp("2024-01-20")],
            "value": [1.0, 2.0, 3.0],
        })
        right = pd.DataFrame({
            "code": ["000001.SZ", "000001.SZ"],
            "announce_date": [pd.Timestamp("2024-01-12"),
                              pd.Timestamp("2024-01-18")],
            "fundamental": [100.0, 200.0],
        })
        merged = checker.pit_safe_merge(left, right, on=["code"])
        # 行日期 >= 公告日 才保留
        # 2024-01-10: 1.0 （不匹配 1-12、1-18，保留但 fundamental 为 NaN）
        # 2024-01-15: 匹配 1-12，保留
        # 2024-01-20: 匹配 1-12、1-18，保留
        assert len(merged) == 3
        # 2024-01-15 行的 fundamental 应来自 2024-01-12 的公告
        row_15 = merged[merged["date"] == pd.Timestamp("2024-01-15")].iloc[0]
        assert row_15["fundamental"] == 100.0

    def test_pit_merge_no_announce_col(self):
        """若右表无公告日列，按普通 merge 处理"""
        checker = PITChecker()
        left = pd.DataFrame({
            "code": ["A", "B"],
            "date": pd.to_datetime(["2024-01-10", "2024-01-15"]),
        })
        right = pd.DataFrame({
            "code": ["A", "B"],
            "value": [10, 20],
        })
        merged = checker.pit_safe_merge(left, right, on=["code"])
        assert len(merged) == 2


# ─────────────────────────────────────────────────────────────
# 性能测试
# ─────────────────────────────────────────────────────────────

class TestPITCheckerPerformance:
    """性能测试：1万行数据应在 1s 内完成"""

    def test_10k_rows_under_1s(self):
        import time
        np.random.seed(42)
        n = 10_000
        df = pd.DataFrame({
            "code": np.random.choice([f"{i:06d}.SZ" for i in range(100)], n),
            "date": pd.to_datetime("2024-01-01") + pd.to_timedelta(
                np.random.randint(0, 100, n), unit="D"
            ),
            "close": np.random.uniform(5, 50, n),
            "announce_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(
                np.random.randint(0, 100, n), unit="D"
            ),
        })
        t0 = time.time()
        report = check_pit(df, pit_columns=["announce_date"])
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s"
        assert report.total_rows == n


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
