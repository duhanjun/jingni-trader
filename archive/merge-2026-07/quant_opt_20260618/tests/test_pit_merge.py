"""
PIT (Point-in-Time) 合并单元测试
"""
import numpy as np
import pandas as pd
import pytest

from quant_opt_20260618.pit_join import PITConfig, pit_merge, detect_lookahead
from quant_opt_20260618.tests.fixtures import make_synthetic_ashare_data, make_financial_data


def test_pit_merge_basic():
    """正常 PIT 合并：左表 asof 早于 right announce 时应为 NaN"""
    financial = make_financial_data(n_stocks=2, n_periods=4)
    # 重命名：让 announce_date 出现在 result 中（fixture 自带 announce_date）
    # 保持 period_end 不冲突即可
    # 左表选 4 个观察日：1 个在所有公告前，1 个在中间，1 个在最后公告后
    sample_dates = pd.to_datetime([
        "2023-01-15",  # 远早于所有公告
        "2023-08-01",  # 中间
        "2024-06-30",  # 大部分公告都已发布
    ])
    left = pd.DataFrame([
        {"code": "600000.SH", "date": sample_dates[0]},
        {"code": "600000.SH", "date": sample_dates[1]},
        {"code": "600000.SH", "date": sample_dates[2]},
        {"code": "600001.SH", "date": sample_dates[0]},
        {"code": "600001.SH", "date": sample_dates[1]},
        {"code": "600001.SH", "date": sample_dates[2]},
    ])

    cfg = PITConfig(asof_col="date", announce_col="announce_date", by="code", allow_exact_match=True)
    merged = pit_merge(left, financial, cfg, value_cols=["pe_ttm", "roe"])
    # 6 行
    assert len(merged) == 6
    # 第一行（2023-01-15）应该找不到任何已发布财务
    row0 = merged.iloc[0]
    assert pd.isna(row0["pe_ttm"]), f"earlier asof should be NaN, got {row0['pe_ttm']}"
    assert pd.isna(row0["roe"])


def test_pit_merge_no_lookahead():
    """
    对比错误实现（直接按 period_end merge 会引入未来函数），
    验证 PIT 合并确实剔除了未来函数。
    """
    financial = make_financial_data(n_stocks=3, n_periods=4).copy()
    # 让 left 的 asof 早于所有 announce_date，
    # 此时：错误实现按 period_end join 会得到"未来函数"的值，
    #       PIT 实现则会判定为 NaN。
    # 用一个安全的基础日期：所有 period_end 都在它之后，所有 announce_date 都在它之后
    safe_base = financial["announce_date"].min() - pd.Timedelta(days=10)
    left = pd.DataFrame([
        {"code": code, "date": safe_base}
        for code in financial["code"].unique()
    ])

    cfg = PITConfig(asof_col="date", announce_col="announce_date", by="code")
    pit_merged = pit_merge(left, financial, cfg, value_cols=["pe_ttm"])

    # 错误实现：直接按 period_end join，会引入未来函数
    bad_input_right = financial.rename(columns={"period_end": "_period_end"})
    bad_merge = left.merge(
        bad_input_right,
        left_on="code",
        right_on="code",
        how="left",
    )

    # PIT 应当全部为 NaN（因为 asof 都早于任何 announce）
    assert pit_merged["pe_ttm"].isna().all(), "PIT 不应引入未发布的财务数据"
    # 而错误实现则会填充上值
    assert bad_merge["pe_ttm"].notna().sum() > 0, "错误实现应当至少填上一行"
    diff_rows = bad_merge["pe_ttm"].notna() & pit_merged["pe_ttm"].isna()
    assert diff_rows.sum() > 0, "PIT 应当剔除至少一行未来函数"
    report = detect_lookahead(bad_merge, pit_merged, value_cols=["pe_ttm"])
    assert report["total_lookahead_eliminated"] > 0


def test_pit_merge_uses_latest_announcement():
    """对于同一 asof，应取 <= asof 的最近一条 announce"""
    financial = pd.DataFrame([
        # 同一 code，两条不同期间的财务，announce 越来越晚
        {"code": "X", "announce_date": pd.Timestamp("2024-01-01"), "pe_ttm": 10.0},
        {"code": "X", "announce_date": pd.Timestamp("2024-05-01"), "pe_ttm": 12.0},
        {"code": "X", "announce_date": pd.Timestamp("2024-09-01"), "pe_ttm": 15.0},
    ])
    left = pd.DataFrame([
        {"code": "X", "date": pd.Timestamp("2024-04-15")},  # 只能看到 1 月那一条
        {"code": "X", "date": pd.Timestamp("2024-06-01")},  # 看到 5 月
        {"code": "X", "date": pd.Timestamp("2024-12-01")},  # 看到 9 月
    ])
    cfg = PITConfig(asof_col="date", announce_col="announce_date", by="code", allow_exact_match=True)
    merged = pit_merge(left, financial, cfg, value_cols=["pe_ttm"])
    assert merged.iloc[0]["pe_ttm"] == 10.0
    assert merged.iloc[1]["pe_ttm"] == 12.0
    assert merged.iloc[2]["pe_ttm"] == 15.0


def test_pit_merge_empty_value_cols():
    df = make_synthetic_ashare_data(n_stocks=2, n_days=5)
    # 构造一个只含 code+announce_date 的 right
    right = df[["code", "date"]].rename(columns={"date": "announce_date"})
    cfg = PITConfig()
    out = pit_merge(df, right, cfg, value_cols=[])
    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == df.shape[1]


def test_pit_merge_preserves_left_order():
    financial = make_financial_data(n_stocks=1, n_periods=3)
    # 直接使用 fixture 的 announce_date 列
    left = pd.DataFrame([
        {"code": "600000.SH", "date": pd.Timestamp("2024-04-01")},
        {"code": "600000.SH", "date": pd.Timestamp("2023-04-01")},
        {"code": "600000.SH", "date": pd.Timestamp("2024-08-01")},
        {"code": "600000.SH", "date": pd.Timestamp("2023-09-01")},
    ])
    cfg = PITConfig(asof_col="date", announce_col="announce_date", by="code")
    merged = pit_merge(left, financial, cfg, value_cols=["pe_ttm"])
    # 顺序应保持与 left 一致
    pd.testing.assert_series_equal(
        merged["date"].reset_index(drop=True),
        left["date"].reset_index(drop=True),
        check_names=False,
    )
