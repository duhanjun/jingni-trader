"""data-engine L2 单元测试：DataEngine.fetch_financial / fetch_valuation。

覆盖 data-engine/engine.py 中 v3 新增的财务/估值数据拉取方法：
- fetch_financial：mock provider 返回财务数据 → 返回标准 schema
- fetch_financial 降级链：首源失败（RateLimitError）→ 次源成功
- fetch_financial 标准 schema 列补齐
- fetch_valuation：返回 code/trade_date/pe_ttm/pb/ps_ttm/dv_ratio
- fetch_financial 所有源都失败 → 返回空 DataFrame（不抛异常）

mock 策略：
- 用 importlib 显式加载 data-engine/engine.py 为独立模块（避免与主调度器 engine.py 同名冲突）
- mock _load_adapter 控制每个 backend 的 provider 行为
- 通过 mod.RateLimitError / mod.NetworkError 等访问 engine 模块已导入的异常类
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# 模块加载工具：把 data-engine/engine.py 加载为独立模块
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
DATA_ENGINE_PATH = os.path.join(DATA_ENGINE_DIR, "engine.py")


def _load_data_engine_module():
    """显式加载 data-engine/engine.py 为独立模块。

    需要把 data-engine/scripts 注册为 sys.modules['scripts']，
    使 engine.py 顶层的 `from scripts.config import ...` 能解析到 data-engine/scripts。
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    scripts_dir = os.path.join(DATA_ENGINE_DIR, "scripts")
    init_py = os.path.join(scripts_dir, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            "scripts", init_py,
            submodule_search_locations=[scripts_dir],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)

    # mock 重量级依赖
    for _m in ("tushare", "baostock", "akshare", "xtquant", "talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    try:
        spec = ilu.spec_from_file_location("data_engine_engine", DATA_ENGINE_PATH)
        mod = ilu.module_from_spec(spec)
        sys.modules["data_engine_engine"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        # 恢复 scripts 缓存（让主调度器后续可用）
        for key in list(sys.modules.keys()):
            if key == "scripts" or key.startswith("scripts."):
                sys.modules.pop(key, None)
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v


# ============================================================================
# 合成数据与 mock 工具
# ============================================================================

def _make_financial_df(n=2) -> pd.DataFrame:
    """构造 provider.get_financial 返回的合成财务数据。"""
    codes = [f"60000{i}.SH" for i in range(n)]
    return pd.DataFrame({
        "code": codes,
        "report_date": ["20240930"] * n,
        "pe_ttm": [12.5, 8.3][:n],
        "pb": [1.5, 0.9][:n],
        "ps_ttm": [2.1, 1.4][:n],
        "dv_ratio": [2.5, 3.8][:n],
        "roe": [15.0, 12.0][:n],
        "roa": [1.2, 1.0][:n],
        "gross_margin": [25.0, 20.0][:n],
        "net_margin": [8.0, 6.5][:n],
        "revenue_growth": [10.0, 8.0][:n],
        "profit_growth": [15.0, 12.0][:n],
        "debt_ratio": [40.0, 50.0][:n],
        "current_ratio": [1.5, 1.2][:n],
        "quick_ratio": [1.2, 1.0][:n],
        "ocf": [1e8, 8e7][:n],
        "industry": ["银行"] * n,
        "name": [f"股票{i}" for i in range(n)],
    })


def _make_provider_returning(df: pd.DataFrame) -> mock.MagicMock:
    """构造一个 get_financial 返回 df 的 mock provider。"""
    p = mock.MagicMock()
    p.get_financial.return_value = df
    return p


def _make_provider_raising(exc: Exception) -> mock.MagicMock:
    """构造一个 get_financial 抛出 exc 的 mock provider。"""
    p = mock.MagicMock()
    p.get_financial.side_effect = exc
    return p


# ============================================================================
# 单元测试：fetch_financial
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestFetchFinancial:
    """验证 DataEngine.fetch_financial。"""

    def test_fetch_financial_returns_data_from_mocked_provider(self):
        """mock provider 返回财务数据 → fetch_financial 透传数据。"""
        mod = _load_data_engine_module()
        fin_df = _make_financial_df(n=2)
        mock_provider = _make_provider_returning(fin_df)

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_financial(
                ["600000.SH", "600001.SH"], "20240930"
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["code"]) == ["600000.SH", "600001.SH"]
        # provider 应被调用一次
        mock_provider.get_financial.assert_called()

    def test_fetch_financial_returns_standard_schema_columns(self):
        """fetch_financial 返回的 DataFrame 应含全部标准 schema 列。"""
        mod = _load_data_engine_module()
        # provider 只返回部分列
        partial_df = pd.DataFrame({
            "code": ["600000.SH"],
            "report_date": ["20240930"],
            "pe_ttm": [12.5],
            "roe": [15.0],
        })
        mock_provider = _make_provider_returning(partial_df)

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_financial(["600000.SH"], "20240930")

        standard_cols = mod.DataEngine._FINANCIAL_STANDARD_COLS
        assert list(result.columns) == standard_cols
        # 缺失列应被补为 NaN
        assert pd.isna(result["pb"].iloc[0])
        assert pd.isna(result["industry"].iloc[0])
        # 已有列保留
        assert result["pe_ttm"].iloc[0] == 12.5
        assert result["roe"].iloc[0] == 15.0

    def test_fetch_financial_fields_filter(self):
        """指定 fields 时返回 code/report_date + 指定字段。"""
        mod = _load_data_engine_module()
        fin_df = _make_financial_df(n=1)
        mock_provider = _make_provider_returning(fin_df)

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_financial(
                ["600000.SH"], "20240930",
                fields=["pe_ttm", "roe"],
            )

        # code + report_date + 指定字段
        assert "code" in result.columns
        assert "report_date" in result.columns
        assert "pe_ttm" in result.columns
        assert "roe" in result.columns
        # 未指定的字段不应出现
        assert "pb" not in result.columns
        assert "industry" not in result.columns

    def test_fetch_financial_fallback_first_fails_second_succeeds(self):
        """降级链：tushare 抛 RateLimitError → baostock 返回数据。"""
        mod = _load_data_engine_module()
        fin_df = _make_financial_df(n=1)

        providers = {
            "tushare": _make_provider_raising(
                mod.RateLimitError("tushare", "访问频率超限 1次/小时")
            ),
            "baostock": _make_provider_returning(fin_df),
        }

        def fake_load_adapter(backend, **kwargs):
            return providers[backend]

        with mock.patch.object(mod, "_load_adapter", side_effect=fake_load_adapter):
            engine = mod.DataEngine(data_sources=["tushare", "baostock"])
            result = engine.fetch_financial(["600000.SH"], "20240930")

        assert len(result) == 1
        assert result["code"].iloc[0] == "600000.SH"
        # 最终用了 baostock
        assert engine.backend == "baostock"
        # 两个 provider 都被调用过 get_financial
        providers["tushare"].get_financial.assert_called_once()
        providers["baostock"].get_financial.assert_called_once()

    def test_fetch_financial_all_providers_fail_returns_empty_df(self):
        """所有源都失败 → 返回空 DataFrame（不抛异常）。"""
        mod = _load_data_engine_module()

        providers = {
            "tushare": _make_provider_raising(
                mod.RateLimitError("tushare", "限频")
            ),
            "baostock": _make_provider_raising(
                mod.NetworkError("baostock", "网络错误")
            ),
            "akshare": _make_provider_raising(
                mod.BlacklistedError("akshare", "黑名单")
            ),
        }

        def fake_load_adapter(backend, **kwargs):
            return providers[backend]

        with mock.patch.object(mod, "_load_adapter", side_effect=fake_load_adapter):
            engine = mod.DataEngine(data_sources=["tushare", "baostock", "akshare"])
            # 不应抛异常
            result = engine.fetch_financial(["600000.SH"], "20240930")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # 空 DataFrame 应有标准列
        standard_cols = mod.DataEngine._FINANCIAL_STANDARD_COLS
        assert list(result.columns) == standard_cols

    def test_fetch_financial_empty_symbols_returns_empty_df(self):
        """symbols 为空 → 直接返回空 DataFrame。"""
        mod = _load_data_engine_module()
        mock_provider = _make_provider_returning(_make_financial_df())

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_financial([], "20240930")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        # provider 不应被调用
        mock_provider.get_financial.assert_not_called()

    def test_fetch_financial_provider_returns_empty_df(self):
        """provider 返回空 DataFrame → 降级链走完后返回空。"""
        mod = _load_data_engine_module()
        empty_df = pd.DataFrame()
        mock_provider = _make_provider_returning(empty_df)

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_financial(["600000.SH"], "20240930")

        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ============================================================================
# 单元测试：fetch_valuation
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.unit
class TestFetchValuation:
    """验证 DataEngine.fetch_valuation。"""

    def test_fetch_valuation_returns_pe_pb_ps_columns(self):
        """fetch_valuation 返回 code/trade_date/pe_ttm/pb/ps_ttm/dv_ratio 列。"""
        mod = _load_data_engine_module()
        fin_df = _make_financial_df(n=2)
        mock_provider = _make_provider_returning(fin_df)

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_valuation(
                ["600000.SH", "600001.SH"], "20240930"
            )

        expected_cols = ["code", "trade_date", "pe_ttm", "pb", "ps_ttm", "dv_ratio"]
        assert list(result.columns) == expected_cols
        assert len(result) == 2
        # trade_date 应来自 report_date
        assert (result["trade_date"] == "20240930").all()
        # PE/PB/PS/股息率值应正确传递
        assert result["pe_ttm"].iloc[0] == 12.5
        assert result["pb"].iloc[0] == 1.5
        assert result["ps_ttm"].iloc[0] == 2.1
        assert result["dv_ratio"].iloc[0] == 2.5

    def test_fetch_valuation_empty_symbols_returns_empty_df(self):
        """symbols 为空 → 返回空 DataFrame。"""
        mod = _load_data_engine_module()
        mock_provider = _make_provider_returning(_make_financial_df())

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_valuation([], "20240930")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = ["code", "trade_date", "pe_ttm", "pb", "ps_ttm", "dv_ratio"]
        assert list(result.columns) == expected_cols
        mock_provider.get_financial.assert_not_called()

    def test_fetch_valuation_all_providers_fail_returns_empty_df(self):
        """所有源失败 → 返回空 DataFrame（不抛异常）。"""
        mod = _load_data_engine_module()
        mock_provider = _make_provider_raising(
            mod.RateLimitError("tushare", "限频")
        )

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_valuation(["600000.SH"], "20240930")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        expected_cols = ["code", "trade_date", "pe_ttm", "pb", "ps_ttm", "dv_ratio"]
        assert list(result.columns) == expected_cols

    def test_fetch_valuation_fallback_chain(self):
        """fetch_valuation 也走降级链：tushare 失败 → baostock 成功。"""
        mod = _load_data_engine_module()
        fin_df = _make_financial_df(n=1)

        providers = {
            "tushare": _make_provider_raising(
                mod.QuotaExceededError("tushare", "积分不足")
            ),
            "baostock": _make_provider_returning(fin_df),
        }

        def fake_load_adapter(backend, **kwargs):
            return providers[backend]

        with mock.patch.object(mod, "_load_adapter", side_effect=fake_load_adapter):
            engine = mod.DataEngine(data_sources=["tushare", "baostock"])
            result = engine.fetch_valuation(["600000.SH"], "20240930")

        assert len(result) == 1
        assert result["code"].iloc[0] == "600000.SH"
        assert engine.backend == "baostock"

    def test_fetch_valuation_partial_columns_filled(self):
        """provider 只返回部分估值列 → 缺失列应补 NaN。"""
        mod = _load_data_engine_module()
        # 只返回 pe_ttm，缺 pb/ps_ttm/dv_ratio
        partial_df = pd.DataFrame({
            "code": ["600000.SH"],
            "report_date": ["20240930"],
            "pe_ttm": [12.5],
        })
        mock_provider = _make_provider_returning(partial_df)

        with mock.patch.object(mod, "_load_adapter", return_value=mock_provider):
            engine = mod.DataEngine(data_sources=["tushare"])
            result = engine.fetch_valuation(["600000.SH"], "20240930")

        expected_cols = ["code", "trade_date", "pe_ttm", "pb", "ps_ttm", "dv_ratio"]
        assert list(result.columns) == expected_cols
        assert result["pe_ttm"].iloc[0] == 12.5
        # 缺失列应为 NaN
        assert pd.isna(result["pb"].iloc[0])
        assert pd.isna(result["ps_ttm"].iloc[0])
        assert pd.isna(result["dv_ratio"].iloc[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
