"""T1-11: L2 单元测试 - 7 个内置 Processor + Processor 基类 + ProcessContext

覆盖 PRD 6.2 节要求：
- 每个 Processor 100% 行覆盖
- 基类 Processor.check_requirements 依赖检查
- ProcessContext 字段传递
- 各 Processor 的 describe() 返回正确元数据

测试策略：
- 使用 fe_scripts_env fixture 隔离 sys.modules
- 使用 sample_panel / sample_panel_with_nan fixture 提供测试数据
- 每个测试用例独立，不依赖执行顺序
"""
from __future__ import annotations

import os
import sys
import importlib.util
from unittest import mock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 基类 Processor + ProcessContext 测试
# ---------------------------------------------------------------------------


class TestProcessorBase:
    """T1-11: Processor 抽象基类测试"""

    def test_cannot_instantiate_abstract_class(self, fe_scripts_env):
        """Processor 是抽象类，不能直接实例化"""
        from scripts.processors.base import Processor
        with pytest.raises(TypeError, match="abstract"):
            Processor()

    def test_subclass_must_implement_abstract_methods(self, fe_scripts_env):
        """子类必须实现 __call__ 和 describe"""
        from scripts.processors.base import Processor

        class IncompleteProcessor(Processor):
            requires = ["code"]

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProcessor()

    def test_check_requirements_passes(self, fe_scripts_env):
        """依赖列存在时不抛异常"""
        from scripts.processors.base import Processor, ProcessContext

        class DummyProcessor(Processor):
            requires = ["code", "date"]

            def __call__(self, df, ctx):
                return df

            def describe(self):
                return {"processor": self.name}

        p = DummyProcessor()
        df = pd.DataFrame({"code": ["000001"], "date": ["2024-01-01"]})
        # 不抛异常即通过
        p.check_requirements(df)

    def test_check_requirements_raises_on_missing(self, fe_scripts_env):
        """依赖列缺失时抛 ProcessorRequirementError"""
        from scripts.processors.base import Processor, ProcessorRequirementError

        class DummyProcessor(Processor):
            requires = ["nonexistent"]

            def __call__(self, df, ctx):
                return df

            def describe(self):
                return {}

        p = DummyProcessor()
        df = pd.DataFrame({"code": ["000001"]})
        with pytest.raises(ProcessorRequirementError, match="nonexistent"):
            p.check_requirements(df)

    def test_processor_name_property(self, fe_scripts_env):
        """name 属性默认返回类名"""
        from scripts.processors.base import Processor

        class MyTestProcessor(Processor):
            def __call__(self, df, ctx):
                return df

            def describe(self):
                return {}

        p = MyTestProcessor()
        assert p.name == "MyTestProcessor"

    def test_processor_params_stored(self, fe_scripts_env):
        """构造参数被存储到 self.params"""
        from scripts.processors.base import Processor

        class P(Processor):
            def __init__(self, alpha=1.0, **kwargs):
                super().__init__(alpha=alpha, **kwargs)
            def __call__(self, df, ctx):
                return df
            def describe(self):
                return {}

        p = P(alpha=0.5)
        assert p.params == {"alpha": 0.5}

    def test_processor_repr(self, fe_scripts_env):
        """__repr__ 返回类名+参数"""
        from scripts.processors.base import Processor

        class P(Processor):
            def __init__(self, x=1, **kwargs):
                super().__init__(x=x, **kwargs)
            def __call__(self, df, ctx):
                return df
            def describe(self):
                return {}

        p = P(x=42)
        assert "P(" in repr(p)
        assert "x=42" in repr(p)

    def test_process_context_default_fields(self, fe_scripts_env):
        """ProcessContext 默认字段值正确"""
        from scripts.processors.base import ProcessContext

        ctx = ProcessContext()
        assert ctx.industry_df is None
        assert ctx.recorder is None
        assert ctx.ic_results == {}
        assert ctx.selected_factors == []
        assert ctx.forward_returns is None
        assert ctx.factor_names == []
        assert ctx.task_id == ""
        assert ctx.work_dir is None
        assert ctx.backend is None
        assert ctx.metadata == {}


# ---------------------------------------------------------------------------
# NeutralizeProcessor 测试
# ---------------------------------------------------------------------------


class TestNeutralizeProcessor:
    """T1-11: NeutralizeProcessor 测试"""

    def test_basic_neutralize(self, fe_scripts_env, sample_panel):
        """行业+市值中性化后，因子均值更接近 0"""
        from scripts.processors.neutralize import NeutralizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)
        p = NeutralizeProcessor(min_count=10)
        result = p(factor_df, ctx)

        # 结果应包含 _neutral 后缀列
        assert "factor_0_neutral" in result.columns
        assert "factor_1_neutral" in result.columns
        # 中性化后残差均值应接近 0（每个截面）
        for dt in result["date"].unique()[:5]:
            section = result[result["date"] == dt]
            if len(section) >= 10:
                assert abs(section["factor_0_neutral"].mean()) < 0.5

    def test_skip_when_both_disabled(self, fe_scripts_env, sample_panel):
        """市值+行业中性化都关闭时，原样返回"""
        from scripts.processors.neutralize import NeutralizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)
        p = NeutralizeProcessor(neutralize_mcap=False, neutralize_industry=False)
        result = p(factor_df, ctx)
        pd.testing.assert_frame_equal(result, factor_df)

    def test_skip_empty_df(self, fe_scripts_env):
        """空 DataFrame 原样返回"""
        from scripts.processors.neutralize import NeutralizeProcessor
        from scripts.processors.base import ProcessContext

        ctx = ProcessContext()
        p = NeutralizeProcessor()
        result = p(pd.DataFrame(), ctx)
        assert result.empty

    def test_missing_required_columns(self, fe_scripts_env, sample_panel):
        """lncap/industry 缺失时原样返回并记录 warning"""
        from scripts.processors.neutralize import NeutralizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        df_no_lncap = factor_df.drop(columns=["lncap"])
        ctx = ProcessContext(factor_names=names)
        p = NeutralizeProcessor(neutralize_mcap=True, neutralize_industry=False)
        result = p(df_no_lncap, ctx)
        # 缺 lncap 时原样返回
        assert "factor_0_neutral" not in result.columns

    def test_merge_industry_from_ctx(self, fe_scripts_env, sample_panel):
        """industry 列缺失时从 ctx.industry_df merge"""
        from scripts.processors.neutralize import NeutralizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        # 移除 industry 列
        df_no_industry = factor_df.drop(columns=["industry"])
        # ctx 提供 industry_df
        industry_df = factor_df[["code", "industry"]].drop_duplicates()
        ctx = ProcessContext(industry_df=industry_df, factor_names=names)
        p = NeutralizeProcessor(neutralize_mcap=False, neutralize_industry=True, min_count=10)
        result = p(df_no_industry, ctx)
        assert "factor_0_neutral" in result.columns

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.neutralize import NeutralizeProcessor

        p = NeutralizeProcessor(min_count=50)
        desc = p.describe()
        assert desc["processor"] == "NeutralizeProcessor"
        assert desc["params"]["min_count"] == 50
        assert "code" in desc["requires"]


# ---------------------------------------------------------------------------
# WinsorizeProcessor 测试
# ---------------------------------------------------------------------------


class TestWinsorizeProcessor:
    """T1-11: WinsorizeProcessor 测试"""

    def test_mad_method_clips_outliers(self, fe_scripts_env, sample_panel):
        """MAD 法截断极端值"""
        from scripts.processors.winsorize import WinsorizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        # 注入极端值
        df = factor_df.copy()
        df.loc[0, "factor_0"] = 1000.0
        ctx = ProcessContext(factor_names=names)
        p = WinsorizeProcessor(method="mad", threshold=3.0)
        result = p(df, ctx)

        # 极端值应被截断
        assert result.loc[0, "factor_0"] < 1000.0

    def test_quantile_method(self, fe_scripts_env, sample_panel):
        """分位数法截尾：截尾后值域被夹在原始 [0.05, 0.95] 分位数之间"""
        from scripts.processors.winsorize import WinsorizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)
        p = WinsorizeProcessor(method="quantile", quantile_range=(0.05, 0.95))
        result = p(factor_df, ctx)

        # 截尾后每个截面的 max/min 应被夹在【原始】数据的 [0.05, 0.95] 分位数之间
        for dt in result["date"].unique()[:3]:
            orig = factor_df[factor_df["date"] == dt]["factor_0"]
            section = result[result["date"] == dt]["factor_0"]
            orig_lower, orig_upper = orig.quantile(0.05), orig.quantile(0.95)
            assert section.max() <= orig_upper + 1e-9
            assert section.min() >= orig_lower - 1e-9

    def test_invalid_method_raises(self, fe_scripts_env):
        """未知 method 抛 ValueError"""
        from scripts.processors.winsorize import WinsorizeProcessor
        with pytest.raises(ValueError, match="不支持的 method"):
            WinsorizeProcessor(method="invalid")

    def test_empty_df(self, fe_scripts_env):
        """空 DataFrame 原样返回"""
        from scripts.processors.winsorize import WinsorizeProcessor
        from scripts.processors.base import ProcessContext

        ctx = ProcessContext()
        p = WinsorizeProcessor()
        result = p(pd.DataFrame(), ctx)
        assert result.empty

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.winsorize import WinsorizeProcessor

        p = WinsorizeProcessor(method="mad", threshold=5.0)
        desc = p.describe()
        assert desc["processor"] == "WinsorizeProcessor"
        assert desc["params"]["method"] == "mad"
        assert desc["params"]["threshold"] == 5.0


# ---------------------------------------------------------------------------
# FillnaProcessor 测试
# ---------------------------------------------------------------------------


class TestFillnaProcessor:
    """T1-11: FillnaProcessor 测试"""

    def test_rank_pct_method(self, fe_scripts_env, sample_panel_with_nan):
        """rank_pct 方法填充 NaN 为 0.5"""
        from scripts.processors.fillna import FillnaProcessor
        from scripts.processors.base import ProcessContext

        df, _, names = sample_panel_with_nan
        ctx = ProcessContext(factor_names=names)
        p = FillnaProcessor(method="rank_pct", fill_value=0.5)
        result = p(df, ctx)

        # 不应再有 NaN
        assert not result["factor_0"].isna().any()
        # 原本 NaN 的位置应填充为 0.5
        nan_mask = df["factor_0"].isna()
        assert (result.loc[nan_mask, "factor_0"] == 0.5).all()

    def test_zero_method(self, fe_scripts_env, sample_panel_with_nan):
        """zero 方法填充 NaN 为 0"""
        from scripts.processors.fillna import FillnaProcessor
        from scripts.processors.base import ProcessContext

        df, _, names = sample_panel_with_nan
        ctx = ProcessContext(factor_names=names)
        p = FillnaProcessor(method="zero")
        result = p(df, ctx)
        assert not result["factor_0"].isna().any()
        nan_mask = df["factor_0"].isna()
        assert (result.loc[nan_mask, "factor_0"] == 0.0).all()

    def test_mean_method(self, fe_scripts_env, sample_panel_with_nan):
        """mean 方法按截面均值填充"""
        from scripts.processors.fillna import FillnaProcessor
        from scripts.processors.base import ProcessContext

        df, _, names = sample_panel_with_nan
        ctx = ProcessContext(factor_names=names)
        p = FillnaProcessor(method="mean")
        result = p(df, ctx)
        assert not result["factor_0"].isna().any()

    def test_ffill_method(self, fe_scripts_env, sample_panel_with_nan):
        """ffill 方法按 code 组内前向填充"""
        from scripts.processors.fillna import FillnaProcessor
        from scripts.processors.base import ProcessContext

        df, _, names = sample_panel_with_nan
        ctx = ProcessContext(factor_names=names)
        p = FillnaProcessor(method="ffill")
        result = p(df, ctx)
        # ffill 后首日可能仍为 NaN，但 fallback 为 0
        assert not result["factor_0"].isna().any()

    def test_invalid_method_raises(self, fe_scripts_env):
        """未知 method 抛 ValueError"""
        from scripts.processors.fillna import FillnaProcessor
        with pytest.raises(ValueError, match="不支持的 method"):
            FillnaProcessor(method="invalid")

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.fillna import FillnaProcessor

        p = FillnaProcessor(method="rank_pct", fill_value=0.3)
        desc = p.describe()
        assert desc["processor"] == "FillnaProcessor"
        assert desc["params"]["method"] == "rank_pct"
        assert desc["params"]["fill_value"] == 0.3


# ---------------------------------------------------------------------------
# StandardizeProcessor 测试
# ---------------------------------------------------------------------------


class TestStandardizeProcessor:
    """T1-11: StandardizeProcessor 测试"""

    def test_zscore_method(self, fe_scripts_env, sample_panel):
        """z-score 标准化后截面均值≈0、标准差≈1"""
        from scripts.processors.standardize import StandardizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)
        p = StandardizeProcessor(method="zscore")
        result = p(factor_df, ctx)

        for dt in result["date"].unique()[:5]:
            section = result[result["date"] == dt]
            if len(section) >= 2:
                assert abs(section["factor_0"].mean()) < 1e-6
                assert abs(section["factor_0"].std() - 1.0) < 1e-6 or section["factor_0"].std() == 0

    def test_minmax_method(self, fe_scripts_env, sample_panel):
        """min-max 标准化到 [0, 1]"""
        from scripts.processors.standardize import StandardizeProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)
        p = StandardizeProcessor(method="minmax")
        result = p(factor_df, ctx)

        for dt in result["date"].unique()[:5]:
            section = result[result["date"] == dt]
            assert section["factor_0"].min() >= -1e-6
            assert section["factor_0"].max() <= 1.0 + 1e-6

    def test_invalid_method_raises(self, fe_scripts_env):
        """未知 method 抛 ValueError"""
        from scripts.processors.standardize import StandardizeProcessor
        with pytest.raises(ValueError, match="不支持的 method"):
            StandardizeProcessor(method="invalid")

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.standardize import StandardizeProcessor

        p = StandardizeProcessor(method="zscore")
        desc = p.describe()
        assert desc["processor"] == "StandardizeProcessor"
        assert desc["params"]["method"] == "zscore"


# ---------------------------------------------------------------------------
# ICAnalysisProcessor 测试
# ---------------------------------------------------------------------------


class TestICAnalysisProcessor:
    """T1-11: ICAnalysisProcessor 测试"""

    def test_basic_ic_analysis(self, fe_scripts_env, sample_panel):
        """IC 分析后 ctx.ic_results 非空"""
        from scripts.processors.ic_analysis import ICAnalysisProcessor
        from scripts.processors.base import ProcessContext

        factor_df, forward_returns, names = sample_panel
        ctx = ProcessContext(
            forward_returns=forward_returns,
            factor_names=names,
        )
        p = ICAnalysisProcessor(ic_type="normal", min_count=10)
        result = p(factor_df, ctx)

        # IC 分析不修改 df
        pd.testing.assert_frame_equal(result, factor_df)
        # ctx.ic_results 应有内容
        assert len(ctx.ic_results) > 0
        # 应包含 ret_forward_1d/5d/20d 三个键
        assert "ret_forward_1d" in ctx.ic_results

    def test_skip_when_no_forward_returns(self, fe_scripts_env, sample_panel):
        """ctx.forward_returns 为空时跳过"""
        from scripts.processors.ic_analysis import ICAnalysisProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)  # 无 forward_returns
        p = ICAnalysisProcessor()
        p(factor_df, ctx)
        assert ctx.ic_results == {}

    def test_spearman_method(self, fe_scripts_env, sample_panel):
        """spearman 方法应正常工作"""
        from scripts.processors.ic_analysis import ICAnalysisProcessor
        from scripts.processors.base import ProcessContext

        factor_df, forward_returns, names = sample_panel
        ctx = ProcessContext(
            forward_returns=forward_returns,
            factor_names=names,
        )
        p = ICAnalysisProcessor(ic_type="spearman", min_count=10)
        p(factor_df, ctx)
        assert len(ctx.ic_results) > 0

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.ic_analysis import ICAnalysisProcessor

        p = ICAnalysisProcessor(ic_type="spearman", min_count=20)
        desc = p.describe()
        assert desc["processor"] == "ICAnalysisProcessor"
        assert desc["params"]["ic_type"] == "spearman"
        assert desc["params"]["min_count"] == 20


# ---------------------------------------------------------------------------
# CorrelationFilterProcessor 测试
# ---------------------------------------------------------------------------


class TestCorrelationFilterProcessor:
    """T1-11: CorrelationFilterProcessor 测试"""

    def test_basic_correlation_filter(self, fe_scripts_env, sample_panel):
        """相关性分析后 ctx.selected_factors 非空"""
        from scripts.processors.correlation_filter import CorrelationFilterProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(factor_names=names)
        p = CorrelationFilterProcessor(max_correlation=0.9)  # 高阈值，保留所有因子
        result = p(factor_df, ctx)

        # 相关性分析不修改 df
        pd.testing.assert_frame_equal(result, factor_df)
        # 应选中至少一个因子
        assert len(ctx.selected_factors) > 0

    def test_high_correlation_removes_factor(self, fe_scripts_env, sample_panel):
        """高相关因子被剔除"""
        from scripts.processors.correlation_filter import CorrelationFilterProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        # 构造一个与 factor_0 高相关的因子
        df = factor_df.copy()
        df["factor_dup"] = df["factor_0"] * 1.001  # 几乎相同
        ctx = ProcessContext(factor_names=["factor_0", "factor_1", "factor_dup"])
        p = CorrelationFilterProcessor(max_correlation=0.5)
        p(df, ctx)
        # factor_dup 应被剔除
        assert "factor_dup" not in ctx.selected_factors

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.correlation_filter import CorrelationFilterProcessor

        p = CorrelationFilterProcessor(max_correlation=0.6)
        desc = p.describe()
        assert desc["processor"] == "CorrelationFilterProcessor"
        assert desc["params"]["max_correlation"] == 0.6


# ---------------------------------------------------------------------------
# FusionProcessor 测试
# ---------------------------------------------------------------------------


class TestFusionProcessor:
    """T1-11: FusionProcessor 测试"""

    def test_basic_fusion(self, fe_scripts_env, sample_panel):
        """融合后输出 alpha_score 列"""
        from scripts.processors.fusion import FusionProcessor
        from scripts.processors.ic_analysis import ICAnalysisProcessor
        from scripts.processors.correlation_filter import CorrelationFilterProcessor
        from scripts.processors.base import ProcessContext

        factor_df, forward_returns, names = sample_panel
        ctx = ProcessContext(
            forward_returns=forward_returns,
            factor_names=names,
        )

        # 先跑 IC + Correlation 填充 ctx
        ICAnalysisProcessor(min_count=10)(factor_df, ctx)
        CorrelationFilterProcessor(max_correlation=0.9)(factor_df, ctx)

        # 再跑 Fusion
        p = FusionProcessor(method="ic_weighted")
        result = p(factor_df, ctx)

        assert "alpha_score" in result.columns
        assert not result["alpha_score"].isna().all()

    def test_equal_weighted_method(self, fe_scripts_env, sample_panel):
        """equal_weighted 方法正常工作"""
        from scripts.processors.fusion import FusionProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, names = sample_panel
        ctx = ProcessContext(
            factor_names=names,
            selected_factors=names,
            ic_results={},  # 等权不需要 IC
        )
        p = FusionProcessor(method="equal_weighted")
        result = p(factor_df, ctx)
        assert "alpha_score" in result.columns

    def test_skip_when_no_selected_factors(self, fe_scripts_env, sample_panel):
        """ctx.selected_factors 为空时跳过"""
        from scripts.processors.fusion import FusionProcessor
        from scripts.processors.base import ProcessContext

        factor_df, _, _ = sample_panel
        ctx = ProcessContext(selected_factors=[])
        p = FusionProcessor()
        result = p(factor_df, ctx)
        # 原样返回
        pd.testing.assert_frame_equal(result, factor_df)

    def test_invalid_method_raises(self, fe_scripts_env):
        """未知 method 抛 ValueError"""
        from scripts.processors.fusion import FusionProcessor
        with pytest.raises(ValueError, match="不支持的 method"):
            FusionProcessor(method="invalid")

    def test_describe(self, fe_scripts_env):
        """describe 返回正确元数据"""
        from scripts.processors.fusion import FusionProcessor

        p = FusionProcessor(method="ic_weighted")
        desc = p.describe()
        assert desc["processor"] == "FusionProcessor"
        assert desc["params"]["method"] == "ic_weighted"
