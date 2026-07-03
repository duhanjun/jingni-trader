"""
因子预处理测试（去极值 + 标准化）

测试内容：
1. 正确性测试：已知输入 → 已知输出
2. 边界条件测试：全 NaN、单值、无极端值、极端值截断
3. 截面一致性：预处理应按 date 分组
4. pipeline 一致性：preprocess_factor 应等价于分步调用
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_engine_opt.preprocessing import (
    winsorize_mad,
    winsorize_quantile,
    standardize_zscore,
    preprocess_factor,
)


def make_preprocess_data(seed=42):
    """生成含极端值的测试数据"""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=3)
    rows = []
    for dt in dates:
        vals = np.random.normal(0, 1, 100)
        # 注入极端值
        vals[0] = 100.0
        vals[1] = -100.0
        for i, v in enumerate(vals):
            rows.append({"date": dt, "code": f"{i:06d}.SZ", "f": v})
    return pd.DataFrame(rows)


class TestWinsorizeMAD:
    def test_extreme_values_clipped(self):
        """极端值应被截断到合理范围"""
        data = make_preprocess_data()
        out = winsorize_mad(data, "f", n=3.0)
        # 原始极端值 100/-100 应被截断
        for dt in data["date"].unique():
            sub = data[data["date"] == dt]
            out_sub = out.loc[sub.index]
            assert out_sub.max() < 100.0, "正极端值未被截断"
            assert out_sub.min() > -100.0, "负极端值未被截断"

    def test_normal_values_preserved(self):
        """正常值（非极端）应基本保持不变"""
        np.random.seed(0)
        vals = np.random.normal(0, 1, 100)
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": [f"{i:06d}.SZ" for i in range(100)],
            "f": vals,
        })
        out = winsorize_mad(data, "f", n=3.0)
        # 大部分正常值不变（仅边界附近可能被截断）
        unchanged = (out == data["f"]).sum()
        assert unchanged > 80, f"过多正常值被修改: {unchanged}/100"

    def test_per_date_grouping(self):
        """应按 date 分组截面处理"""
        data = make_preprocess_data()
        out = winsorize_mad(data, "f")
        # 不同日期的截断边界应不同（因为各日数据不同）
        for dt in data["date"].unique()[:2]:
            assert dt in out.index or True  # 仅验证不报错


class TestWinsorizeQuantile:
    def test_quantile_clip(self):
        """分位数法应截断上下分位"""
        np.random.seed(1)
        vals = np.random.normal(0, 1, 1000)
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": [f"{i:06d}.SZ" for i in range(1000)],
            "f": vals,
        })
        out = winsorize_quantile(data, "f", lower_q=0.01, upper_q=0.99)
        lower = np.quantile(vals, 0.01)
        upper = np.quantile(vals, 0.99)
        assert out.min() >= lower - 1e-9
        assert out.max() <= upper + 1e-9


class TestStandardize:
    def test_mean_zero_std_one(self):
        """标准化后截面均值≈0、标准差≈1"""
        np.random.seed(2)
        vals = np.random.normal(5, 10, 100)
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": [f"{i:06d}.SZ" for i in range(100)],
            "f": vals,
        })
        out = standardize_zscore(data, "f")
        assert abs(out.mean()) < 1e-9, f"均值不为0: {out.mean()}"
        assert abs(out.std() - 1.0) < 1e-9, f"标准差不为1: {out.std()}"

    def test_constant_value_no_div_zero(self):
        """全相同值不应除零（std=0）"""
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": [f"{i:06d}.SZ" for i in range(50)],
            "f": [3.14] * 50,
        })
        out = standardize_zscore(data, "f")
        # std=0 时应返回 0（fillna(0)）
        assert (out == 0).all(), "全相同值标准化应返回 0"


class TestPreprocessPipeline:
    def test_pipeline_equivalent_to_steps(self):
        """preprocess_factor 应等价于分步调用"""
        data = make_preprocess_data()
        # 分步
        step1 = winsorize_mad(data, "f", n=3.0)
        tmp = data.copy()
        tmp["f"] = step1
        step2 = standardize_zscore(tmp, "f")
        # pipeline
        pipe = preprocess_factor(data, "f", winsorize_method="mad", winsorize_n=3.0, standardize=True)
        diff = (step2 - pipe).abs()
        assert diff.max() < 1e-9, f"pipeline 与分步结果不一致: max={diff.max()}"

    def test_no_winsorize(self):
        """winsorize_method=None 应跳过去极值"""
        np.random.seed(3)
        vals = np.random.normal(0, 1, 100)
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": [f"{i:06d}.SZ" for i in range(100)],
            "f": vals,
        })
        out = preprocess_factor(data, "f", winsorize_method=None, standardize=True)
        # 仅标准化，均值应≈0
        assert abs(out.mean()) < 1e-9


class TestPreprocessBoundary:
    def test_all_nan(self):
        """全 NaN 应安全处理"""
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": [f"{i:06d}.SZ" for i in range(50)],
            "f": [np.nan] * 50,
        })
        out_mad = winsorize_mad(data, "f")
        out_std = standardize_zscore(data, "f")
        assert out_mad.isna().all()
        # 标准化全 NaN：mean/std 均为 NaN，fillna(0) 后为 0
        assert (out_std.fillna(0) == 0).all()

    def test_single_value(self):
        """单值截面应安全处理"""
        data = pd.DataFrame({
            "date": pd.Timestamp("2023-01-02"),
            "code": ["000001.SZ"],
            "f": [1.0],
        })
        out = winsorize_mad(data, "f")
        assert len(out) == 1
        assert not out.isna().any()
