"""
前视偏差检测工具测试

测试内容：
1. 正向测试：无泄漏的数据应通过检测
2. 反向测试：含泄漏的数据应被检测到并抛异常
3. 边界条件测试：空数据、缺列
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest_engine_opt.look_ahead_guard import (
    check_forward_return_leakage,
    check_signal_timestamp_order,
    check_feature_alignment,
    LookAheadBiasError,
)


class TestForwardReturnLeakage:
    def test_clean_features_pass(self):
        """合法特征应无泄漏"""
        np.random.seed(0)
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=10),
            "code": ["000001.SZ"] * 10,
            "momentum": np.random.randn(10),
            "ret_forward_1d": np.random.randn(10),
        })
        issues = check_forward_return_leakage(
            df, ["momentum"], ["ret_forward_1d"], raise_on_fail=False
        )
        assert issues == []

    def test_same_name_detected(self):
        """特征与未来收益同名应被检测"""
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=5),
            "code": ["000001.SZ"] * 5,
            "ret_forward_1d": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        with pytest.raises(LookAheadBiasError):
            check_forward_return_leakage(df, ["ret_forward_1d"], ["ret_forward_1d"])

    def test_identical_values_detected(self):
        """特征与未来收益数值相同应被检测"""
        np.random.seed(1)
        vals = np.random.randn(10)
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=10),
            "code": ["000001.SZ"] * 10,
            "fake_feature": vals,
            "ret_forward_1d": vals.copy(),
        })
        issues = check_forward_return_leakage(
            df, ["fake_feature"], ["ret_forward_1d"], raise_on_fail=False
        )
        assert len(issues) > 0, "数值相同的特征应被检测为泄漏"

    def test_no_raise_mode(self):
        """raise_on_fail=False 应返回问题列表而非抛异常"""
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=3),
            "code": ["000001.SZ"] * 3,
            "ret_forward_1d": [0.1, 0.2, 0.3],
        })
        issues = check_forward_return_leakage(
            df, ["ret_forward_1d"], ["ret_forward_1d"], raise_on_fail=False
        )
        assert len(issues) > 0


class TestSignalTimestampOrder:
    def test_valid_order_pass(self):
        """信号日不超出数据日应通过"""
        dates = pd.bdate_range("2023-01-02", periods=10)
        signals = pd.DataFrame({"date": dates[:8], "signal": [1] * 8})
        data = pd.DataFrame({"date": dates, "close": np.arange(10)})
        issues = check_signal_timestamp_order(signals, data, raise_on_fail=False)
        assert issues == []

    def test_future_signal_detected(self):
        """信号日超出数据末日应被检测"""
        data_dates = pd.bdate_range("2023-01-02", periods=5)
        sig_dates = pd.bdate_range("2023-01-02", periods=7)  # 超出
        signals = pd.DataFrame({"date": sig_dates, "signal": [1] * 7})
        data = pd.DataFrame({"date": data_dates, "close": np.arange(5)})
        with pytest.raises(LookAheadBiasError):
            check_signal_timestamp_order(signals, data)

    def test_empty_data_pass(self):
        """空数据应安全返回"""
        issues = check_signal_timestamp_order(
            pd.DataFrame(), pd.DataFrame(), raise_on_fail=False
        )
        assert issues == []


class TestFeatureAlignment:
    def test_valid_alignment_pass(self):
        """合法对齐应通过"""
        np.random.seed(2)
        n = 50
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "code": ["000001.SZ"] * n,
            "momentum": np.random.randn(n),
        })
        fr = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "code": ["000001.SZ"] * n,
            "ret_forward_1d": np.random.normal(0, 0.01, n),
        })
        issues = check_feature_alignment(
            df, fr, ["momentum"], "ret_forward_1d", raise_on_fail=False
        )
        assert issues == []

    def test_row_count_mismatch_detected(self):
        """行数不匹配应被检测"""
        df = pd.DataFrame({"date": [1, 2, 3], "code": ["A"] * 3, "f": [1, 2, 3]})
        fr = pd.DataFrame({"date": [1, 2], "code": ["A"] * 2, "ret_forward_1d": [0.1, 0.2]})
        issues = check_feature_alignment(
            df, fr, ["f"], "ret_forward_1d", raise_on_fail=False
        )
        assert any("行数不匹配" in i for i in issues)

    def test_abnormal_forward_return_detected(self):
        """未来收益均值异常大应被警告"""
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=10),
            "code": ["000001.SZ"] * 10,
            "f": np.random.randn(10),
        })
        fr = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=10),
            "code": ["000001.SZ"] * 10,
            "ret_forward_1d": [10.0] * 10,  # 异常大
        })
        issues = check_feature_alignment(
            df, fr, ["f"], "ret_forward_1d", raise_on_fail=False
        )
        assert any("异常大" in i for i in issues)

    def test_missing_forward_col(self):
        """未来收益列不存在应抛异常"""
        df = pd.DataFrame({"date": [1], "code": ["A"], "f": [1]})
        fr = pd.DataFrame({"date": [1], "code": ["A"]})
        with pytest.raises(LookAheadBiasError):
            check_feature_alignment(df, fr, ["f"], "ret_forward_1d")
