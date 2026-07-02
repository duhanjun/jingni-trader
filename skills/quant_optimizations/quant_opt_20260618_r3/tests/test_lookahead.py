"""Look-ahead Bias 检测器单元测试"""
import numpy as np
import pandas as pd
import pytest

from skills.quant_optimizations.quant_opt_20260618_r3.lookahead_detector import (
    detect_in_code,
    detect_in_dataframe,
    detect_train_test_leakage,
    run_full_check,
)


# ---------------------------------------------------------------------------
# 代码扫描
# ---------------------------------------------------------------------------
def test_detect_negative_shift():
    src = "df['future'] = df['close'].shift(-5)\n"
    report = detect_in_code(src)
    assert any(i.code == "LOOKAHEAD_NEG_SHIFT" for i in report.issues)


def test_detect_rolling_no_shift():
    src = "df['ma5'] = df['close'].rolling(5).mean()\n"
    report = detect_in_code(src)
    # 注释行应被忽略
    has_rolling = any(i.code == "LOOKAHEAD_ROLLING_NO_SHIFT" for i in report.issues)
    assert has_rolling


def test_no_warning_when_rolling_followed_by_shift():
    src = "df['ma5_lag'] = df['close'].rolling(5).mean().shift(1)\n"
    report = detect_in_code(src)
    assert not any(i.code == "LOOKAHEAD_ROLLING_NO_SHIFT" for i in report.issues)
    assert not any(i.code == "LOOKAHEAD_NEG_SHIFT" for i in report.issues)


def test_comment_lines_ignored():
    src = "# df['x'] = df['close'].shift(-5)\n"
    report = detect_in_code(src)
    assert not any(i.code == "LOOKAHEAD_NEG_SHIFT" for i in report.issues)


# ---------------------------------------------------------------------------
# DataFrame 检查
# ---------------------------------------------------------------------------
def test_detect_label_in_features():
    df = pd.DataFrame({"feat1": [1, 2], "label": [0.1, 0.2]})
    report = detect_in_dataframe(df, "label", ["feat1", "label"])
    assert any(i.code == "LOOKAHEAD_LABEL_IN_FEATURES" for i in report.issues)


def test_detect_forward_named_feature():
    df = pd.DataFrame({"feat1": [1, 2], "fwd_ret_5": [0.1, 0.2]})
    report = detect_in_dataframe(df, "label", ["feat1", "fwd_ret_5"])
    assert any(i.code == "LOOKAHEAD_FORWARD_FEATURE" for i in report.issues)


# ---------------------------------------------------------------------------
# 训练/测试集时间泄漏
# ---------------------------------------------------------------------------
def test_overlap_detected():
    tr = pd.date_range("2024-01-01", "2024-06-30")
    te = pd.date_range("2024-06-15", "2024-12-31")
    report = detect_train_test_leakage(tr, te)
    assert any(i.code == "LEAKAGE_OVERLAP" for i in report.issues)


def test_chronological_split_ok():
    tr = pd.date_range("2024-01-01", "2024-06-30")
    te = pd.date_range("2024-07-15", "2024-12-31")
    report = detect_train_test_leakage(tr, te, purge_gap_days=10)
    assert not any(i.severity == "error" for i in report.issues)
    # 间隔 15 天 >= purge 10 -> 不应有 warning
    assert not any(i.code == "LEAKAGE_INSUFFICIENT_PURGE" for i in report.issues)


def test_insufficient_purge():
    tr = pd.date_range("2024-01-01", "2024-06-30")
    te = pd.date_range("2024-07-03", "2024-12-31")
    report = detect_train_test_leakage(tr, te, purge_gap_days=10)
    assert any(i.code == "LEAKAGE_INSUFFICIENT_PURGE" for i in report.issues)


def test_test_before_train():
    tr = pd.date_range("2024-07-01", "2024-12-31")
    te = pd.date_range("2024-01-01", "2024-06-30")
    report = detect_train_test_leakage(tr, te)
    assert any(i.code == "LEAKAGE_TEST_BEFORE_TRAIN" for i in report.issues)


# ---------------------------------------------------------------------------
# run_full_check
# ---------------------------------------------------------------------------
def test_run_full_check_aggregates():
    src = "df['x'] = df['close'].shift(-1)\n"
    tr = pd.date_range("2024-01-01", "2024-06-30")
    te = pd.date_range("2024-07-01", "2024-12-31")
    report = run_full_check(
        source=src,
        train_dates=tr,
        test_dates=te,
        purge_gap_days=5,
    )
    assert any(i.code == "LOOKAHEAD_NEG_SHIFT" for i in report.issues)
    # 此时 train/test 间隔 1 天 < purge 5 -> 应有 warning
    assert any(i.code == "LEAKAGE_INSUFFICIENT_PURGE" for i in report.issues)