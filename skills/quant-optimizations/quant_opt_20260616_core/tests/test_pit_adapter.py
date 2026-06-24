"""
PIT 数据适配器测试
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_fundamental() -> pd.DataFrame:
    """模拟季报数据: 日期为报告期, announce_date 滞后 30-60 天"""
    rows = []
    for code in ["000001.SZ", "000002.SZ"]:
        for year in [2023, 2024]:
            for q, delay in [(1, 45), (2, 30), (3, 30), (4, 60)]:
                period = pd.Timestamp(f"{year}-{q * 3:02d}-30")
                announce = period + pd.Timedelta(days=delay)
                rows.append({
                    "code": code,
                    "date": period,
                    "announce_date": announce,
                    "roe": 0.1 + 0.01 * q,
                    "revenue": 1e9 * (1 + 0.1 * q),
                })
    return pd.DataFrame(rows)


def test_asof_basic():
    from skills.quant-optimizations.quant_opt_20260616_core.pit_adapter import PITDataAdapter, PITField
    df = _make_fundamental()
    adapter = PITDataAdapter([
        PITField("roe", "roe", announce_col="announce_date"),
        PITField("revenue", "revenue", announce_col="announce_date"),
    ])
    # 选 2024-08-15: 应该只能看到 2024Q1 (announce=2024-04-30)
    snap = adapter.asof(df, asof=pd.Timestamp("2024-08-15"))
    assert set(snap["code"]) == {"000001.SZ", "000002.SZ"}
    # 2024Q2 (period=2024-06-30, announce=2024-07-30) 也应可见
    # 但 Q3 (period=2024-09-30) 不应可见
    # roe 应该来自 Q2
    expected_q2_roe = 0.12
    for code in ["000001.SZ", "000002.SZ"]:
        val = snap[snap["code"] == code]["roe"].iloc[0]
        assert abs(val - expected_q2_roe) < 1e-6, (code, val)
    print("  [OK] asof basic")


def test_fallback_delay():
    from skills.quant-optimizations.quant_opt_20260616_core.pit_adapter import PITDataAdapter, PITField
    # 没有 announce_col 时, 用 date + delay
    rows = [
        {"code": "X", "date": pd.Timestamp("2024-03-31"), "roe": 0.10},
        {"code": "X", "date": pd.Timestamp("2024-06-30"), "roe": 0.15},
    ]
    df = pd.DataFrame(rows)
    adapter = PITDataAdapter([PITField("roe", "roe", fallback_delay=30)])
    # 2024-05-01: 只能看到 2024-03-31 (announce 2024-04-30)
    snap = adapter.asof(df, asof=pd.Timestamp("2024-05-01"))
    assert abs(snap["roe"].iloc[0] - 0.10) < 1e-9
    # 2024-08-15: 能看到 2024-06-30 (announce 2024-07-30)
    snap = adapter.asof(df, asof=pd.Timestamp("2024-08-15"))
    assert abs(snap["roe"].iloc[0] - 0.15) < 1e-9
    print("  [OK] fallback delay")


def test_no_lookahead_protection():
    """asof 早于所有公告日, 应得到 NaN"""
    from skills.quant-optimizations.quant_opt_20260616_core.pit_adapter import PITDataAdapter, PITField
    df = _make_fundamental()
    adapter = PITDataAdapter([
        PITField("roe", "roe", announce_col="announce_date"),
    ])
    snap = adapter.asof(df, asof=pd.Timestamp("2022-01-01"))
    assert snap["roe"].isna().all()
    print("  [OK] no-lookahead protection")


def test_panel_generation():
    from skills.quant-optimizations.quant_opt_20260616_core.pit_adapter import PITDataAdapter, PITField
    df = _make_fundamental()
    adapter = PITDataAdapter([PITField("roe", "roe", announce_col="announce_date")])
    dates = pd.DatetimeIndex(["2024-05-01", "2024-08-15", "2024-12-31"])
    panel = adapter.panel(df, dates)
    assert "asof_date" in panel.columns
    assert len(panel) == len(dates) * df["code"].nunique()
    # 2024-12-31: 4 份季报都已发布
    last = panel[panel["asof_date"] == pd.Timestamp("2024-12-31")]
    assert last["roe"].notna().all()
    print("  [OK] panel generation")


def run() -> dict:
    test_asof_basic()
    test_fallback_delay()
    test_no_lookahead_protection()
    test_panel_generation()
    return {"status": "passed", "cases": 4}


if __name__ == "__main__":
    run()