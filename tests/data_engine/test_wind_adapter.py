"""WindAdapter 单元测试。

验证 wind_adapter.py 的核心逻辑，不依赖真实 Wind 终端：
- 模块可正常 import
- wsd 调用参数（fields/PriceAdj/日期格式）正确
- DataFrame 字段映射到标准 schema
- 复权方式 hfq/qfq/none 映射到 PriceAdj B/F/U
- 连接失败/查询错误抛出 DataSourceError/NetworkError

设计：用 unittest.mock 注入假的 WindPy 模块，验证适配器调用 wsd 的参数。
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
# 模块加载：把 data-engine/scripts 注册为 scripts 包，加载 wind_adapter
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
SCRIPTS_DIR = os.path.join(DATA_ENGINE_DIR, "scripts")


def _load_wind_adapter(monkeypatch):
    """加载 wind_adapter 模块，注入假的 WindPy。"""
    # 清理 scripts 缓存
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    # 注册 data-engine/scripts 为 scripts 包
    init_py = os.path.join(SCRIPTS_DIR, "__init__.py")
    spec = ilu.spec_from_file_location(
        "scripts", init_py,
        submodule_search_locations=[SCRIPTS_DIR],
    )
    pkg = ilu.module_from_spec(spec)
    sys.modules["scripts"] = pkg
    spec.loader.exec_module(pkg)

    # 注入假的 WindPy（wind_adapter 在 __init__ 中 from WindPy import w）
    fake_w = mock.MagicMock()
    fake_w.isconnected.return_value = False
    fake_w.start.return_value = mock.MagicMock(ErrorCode=0)
    fake_windpy_module = mock.MagicMock()
    fake_windpy_module.w = fake_w
    sys.modules["WindPy"] = fake_windpy_module

    # 加载 wind_adapter
    adapter_path = os.path.join(SCRIPTS_DIR, "adapters", "wind_adapter.py")
    spec = ilu.spec_from_file_location("scripts.adapters.wind_adapter", adapter_path)
    mod = ilu.module_from_spec(spec)
    sys.modules["scripts.adapters.wind_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod, fake_w


# ============================================================================
# 辅助：构造 Wind wsd 返回的 DataFrame
# ============================================================================

def _make_wsd_df(symbol: str, n: int = 5) -> pd.DataFrame:
    """构造 Wind wsd 返回的 DataFrame（列名大写，index 是日期）。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "OPEN": np.linspace(10, 11, n),
        "HIGH": np.linspace(11, 12, n),
        "LOW": np.linspace(9, 10, n),
        "CLOSE": np.linspace(10.5, 11.5, n),
        "VOLUME": np.arange(100, 100 + n) * 1000,
        "AMT": np.arange(200, 200 + n) * 1000,
        "OI": [np.nan] * n,
    }, index=dates)
    return df


# ============================================================================
# 测试用例
# ============================================================================

class TestWindAdapterInit:
    """WindAdapter 初始化测试"""

    def test_init_success_when_windpy_available(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()
        assert adapter._connected is True
        fake_w.start.assert_called_once()

    def test_init_raises_when_windpy_missing(self, monkeypatch):
        mod, _ = _load_wind_adapter(monkeypatch)
        # 移除假 WindPy，模拟未安装
        sys.modules.pop("WindPy", None)
        with pytest.raises(Exception) as exc_info:
            mod.WindAdapter()
        assert "WindPy" in str(exc_info.value) or "wind" in str(exc_info.value).lower()

    def test_init_raises_when_start_fails(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        fake_w.isconnected.return_value = False
        fake_w.start.return_value = mock.MagicMock(ErrorCode=-1)
        with pytest.raises(Exception) as exc_info:
            mod.WindAdapter()
        assert "连接失败" in str(exc_info.value) or "ErrorCode" in str(exc_info.value)


class TestWindAdapterGetDaily:
    """get_daily 核心逻辑测试"""

    def test_get_daily_returns_standard_schema(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        symbol = "600000.SH"
        wsd_df = _make_wsd_df(symbol)
        # wsd(usedf=True) 返回 (ErrorCode, DataFrame)
        fake_w.wsd.return_value = (0, wsd_df)

        df = adapter.get_daily([symbol], "2024-01-01", "2024-01-07", adjust="hfq")

        assert not df.empty
        # 验证标准 schema 列存在
        required_cols = {"code", "date", "open", "high", "low", "close", "volume", "amount",
                         "pre_close", "change_pct", "is_st", "is_limit_up", "is_limit_down"}
        assert required_cols.issubset(set(df.columns))
        assert (df["code"] == symbol).all()
        assert len(df) == 5

    def test_get_daily_passes_correct_fields_to_wsd(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        symbol = "000001.SZ"
        wsd_df = _make_wsd_df(symbol)
        fake_w.wsd.return_value = (0, wsd_df)

        adapter.get_daily([symbol], "2024-01-01", "2024-01-05", adjust="hfq")

        fake_w.wsd.assert_called_once()
        _, kwargs = fake_w.wsd.call_args
        # 验证 fields 包含核心字段
        assert "open" in kwargs["fields"]
        assert "close" in kwargs["fields"]
        assert "volume" in kwargs["fields"]
        # 验证日期格式化为 YYYY-MM-DD
        assert kwargs["beginTime"] == "2024-01-01"
        assert kwargs["endTime"] == "2024-01-05"
        # 验证 usedf=True
        assert kwargs["usedf"] is True

    @pytest.mark.parametrize("adjust,expected_price_adj", [
        ("hfq", "B"),   # 后复权 -> Back
        ("qfq", "F"),   # 前复权 -> Front
        ("none", "U"), # 不复权 -> Unadjusted
        ("", "U"),
    ])
    def test_get_daily_adjust_maps_to_price_adj(self, monkeypatch, adjust, expected_price_adj):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        symbol = "600000.SH"
        wsd_df = _make_wsd_df(symbol)
        fake_w.wsd.return_value = (0, wsd_df)

        adapter.get_daily([symbol], "2024-01-01", "2024-01-05", adjust=adjust)

        _, kwargs = fake_w.wsd.call_args
        assert f"PriceAdj={expected_price_adj}" in kwargs["options"]

    def test_get_daily_raises_on_error_code(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        fake_w.wsd.return_value = (-1, pd.DataFrame())

        with pytest.raises(Exception) as exc_info:
            adapter.get_daily(["600000.SH"], "2024-01-01", "2024-01-05")
        assert "错误码" in str(exc_info.value) or "wsd" in str(exc_info.value).lower()

    def test_get_daily_returns_empty_when_no_data(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        # Wind 返回空 DataFrame
        fake_w.wsd.return_value = (0, pd.DataFrame())

        df = adapter.get_daily(["600000.SH"], "2024-01-01", "2024-01-05")
        assert df.empty

    def test_get_daily_rejects_invalid_symbol_format(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        # 代码缺少交易所后缀
        with pytest.raises(Exception) as exc_info:
            adapter.get_daily(["600000"], "2024-01-01", "2024-01-05")
        assert "不支持" in str(exc_info.value) or "格式" in str(exc_info.value)


class TestWindAdapterOtherMethods:
    """get_stock_list / get_adj_factor / get_financial 测试"""

    def test_get_adj_factor_returns_empty(self, monkeypatch):
        mod, _ = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()
        df = adapter.get_adj_factor(["600000.SH"], "2024-01-01", "2024-01-05")
        assert df.empty
        assert "code" in df.columns and "date" in df.columns

    def test_get_financial_returns_standard_cols(self, monkeypatch):
        mod, fake_w = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()

        # wss 返回 (ErrorCode, DataFrame)
        wss_df = pd.DataFrame({
            "SEC_NAME": ["浦发银行"],
            "INDUSTRY": ["银行"],
            "PE_TTM": [5.2],
            "PB": [0.6],
            "PS_TTM": [1.8],
            "DV_RATIO": [5.0],
            "ROE": [12.0],
            "ROA": [0.8],
        }, index=["600000.SH"])
        fake_w.wss.return_value = (0, wss_df)

        df = adapter.get_financial(["600000.SH"], "20240930", [])
        assert not df.empty
        assert "code" in df.columns
        assert "pe_ttm" in df.columns
        assert "pb" in df.columns
        assert df.iloc[0]["name"] == "浦发银行"

    def test_get_financial_returns_empty_when_no_symbols(self, monkeypatch):
        mod, _ = _load_wind_adapter(monkeypatch)
        adapter = mod.WindAdapter()
        df = adapter.get_financial([], "20240930", [])
        assert df.empty
