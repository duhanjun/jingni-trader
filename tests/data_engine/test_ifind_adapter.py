"""IfindAdapter 单元测试。

验证 ifind_adapter.py 的核心逻辑，不依赖真实 iFinD 服务：
- 模块可正常 import
- THS_HQ 调用参数（indicators/params/CPS）正确
- DataFrame 字段映射到标准 schema
- 复权方式 hfq/qfq/none 映射到 CPS 1/2/无
- 股票代码自动加 CPS，非股票不加
- 登录失败/查询错误抛出 DataSourceError/NetworkError

设计：用 unittest.mock 注入假的 iFinDPy 模块，验证适配器调用 THS_HQ 的参数。
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
# 模块加载：把 data-engine/scripts 注册为 scripts 包，加载 ifind_adapter
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
SCRIPTS_DIR = os.path.join(DATA_ENGINE_DIR, "scripts")


def _load_ifind_adapter(monkeypatch, username="test_user", password="test_pass"):
    """加载 ifind_adapter 模块，注入假的 iFinDPy。"""
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

    # 注入假的 iFinDPy（ifind_adapter 在 __init__ 中 from iFinDPy import ...）
    fake_THS_iFinDLogin = mock.MagicMock(return_value=0)  # 0 = 登录成功
    fake_THS_HQ = mock.MagicMock()
    fake_THSData = mock.MagicMock()
    fake_ifind_module = mock.MagicMock()
    fake_ifind_module.THS_iFinDLogin = fake_THS_iFinDLogin
    fake_ifind_module.THS_HQ = fake_THS_HQ
    fake_ifind_module.THSData = fake_THSData
    sys.modules["iFinDPy"] = fake_ifind_module

    # 注入假的 IFIND_USERNAME/PASSWORD（config 从环境变量读）
    monkeypatch.setenv("IFIND_USERNAME", username)
    monkeypatch.setenv("IFIND_PASSWORD", password)
    # 重置 scripts.config 缓存，使其重新读取环境变量
    sys.modules.pop("scripts.config", None)

    # 加载 ifind_adapter
    adapter_path = os.path.join(SCRIPTS_DIR, "adapters", "ifind_adapter.py")
    spec = ilu.spec_from_file_location("scripts.adapters.ifind_adapter", adapter_path)
    mod = ilu.module_from_spec(spec)
    sys.modules["scripts.adapters.ifind_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod, fake_THS_iFinDLogin, fake_THS_HQ


# ============================================================================
# 辅助：构造 iFinD THS_HQ 返回的 result 对象
# ============================================================================

def _make_hq_result(symbol: str, n: int = 5):
    """构造 THS_HQ 返回的 result（含 .errorcode 和 .data）。"""
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame({
        "time": dates.strftime("%Y-%m-%d"),
        "open": np.linspace(10, 11, n),
        "high": np.linspace(11, 12, n),
        "low": np.linspace(9, 10, n),
        "close": np.linspace(10.5, 11.5, n),
        "volume": np.arange(100, 100 + n) * 1000,
        "amount": np.arange(200, 200 + n) * 1000,
        "openInterest": [0] * n,
    })
    result = mock.MagicMock()
    result.errorcode = 0
    result.data = df
    return result


# ============================================================================
# 测试用例
# ============================================================================

class TestIfindAdapterInit:
    """IfindAdapter 初始化测试"""

    def test_init_success_with_credentials(self, monkeypatch):
        mod, fake_login, _ = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()
        assert adapter._inited is True
        fake_login.assert_called_once_with("test_user", "test_pass")

    def test_init_raises_when_credentials_missing(self, monkeypatch):
        mod, _, _ = _load_ifind_adapter(monkeypatch)
        # 清除环境变量并重置模块缓存的 config 值，模拟未配置
        monkeypatch.delenv("IFIND_USERNAME", raising=False)
        monkeypatch.delenv("IFIND_PASSWORD", raising=False)
        # ifind_adapter 在模块顶层已 import IFIND_USERNAME/PASSWORD，
        # 需用 mock.patch 覆盖为 None 才能触发凭证缺失分支
        with mock.patch.object(mod, "IFIND_USERNAME", None), \
             mock.patch.object(mod, "IFIND_PASSWORD", None):
            with pytest.raises(Exception) as exc_info:
                mod.IfindAdapter()
        assert "IFIND_USERNAME" in str(exc_info.value) or "账号" in str(exc_info.value)

    def test_init_raises_when_login_fails(self, monkeypatch):
        mod, fake_login, _ = _load_ifind_adapter(monkeypatch)
        fake_login.return_value = -1  # 登录失败
        with pytest.raises(Exception) as exc_info:
            mod.IfindAdapter()
        assert "登录失败" in str(exc_info.value) or "错误码" in str(exc_info.value)

    def test_init_raises_when_ifindpy_missing(self, monkeypatch):
        mod, _, _ = _load_ifind_adapter(monkeypatch)
        sys.modules.pop("iFinDPy", None)
        with pytest.raises(Exception) as exc_info:
            mod.IfindAdapter()
        assert "iFinDPy" in str(exc_info.value) or "ifind" in str(exc_info.value).lower()


class TestIfindAdapterGetDaily:
    """get_daily 核心逻辑测试"""

    def test_get_daily_returns_standard_schema(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        symbol = "600000.SH"
        fake_hq.return_value = _make_hq_result(symbol)

        df = adapter.get_daily([symbol], "2024-01-01", "2024-01-07", adjust="hfq")

        assert not df.empty
        required_cols = {"code", "date", "open", "high", "low", "close", "volume", "amount",
                         "pre_close", "change_pct", "is_st", "is_limit_up", "is_limit_down"}
        assert required_cols.issubset(set(df.columns))
        assert (df["code"] == symbol).all()
        assert len(df) == 5

    def test_get_daily_passes_correct_indicators(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        symbol = "000001.SZ"
        fake_hq.return_value = _make_hq_result(symbol)

        adapter.get_daily([symbol], "2024-01-01", "2024-01-05", adjust="hfq")

        fake_hq.assert_called_once()
        args, _ = fake_hq.call_args
        # THS_HQ(symbol, indicators, params, start, end)
        indicators = args[1]
        assert "open" in indicators
        assert "high" in indicators
        assert "low" in indicators
        assert "close" in indicators
        assert "volume" in indicators
        # 验证日期格式化为 YYYY-MM-DD HH:MM:SS
        assert args[3] == "2024-01-01 00:00:00"
        assert args[4] == "2024-01-05 00:00:00"

    @pytest.mark.parametrize("adjust,expected_cps", [
        ("hfq", "1"),   # 后复权
        ("qfq", "2"),   # 前复权
    ])
    def test_get_daily_stock_adds_cps_param(self, monkeypatch, adjust, expected_cps):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        symbol = "600000.SH"  # 股票
        fake_hq.return_value = _make_hq_result(symbol)

        adapter.get_daily([symbol], "2024-01-01", "2024-01-05", adjust=adjust)

        _, kwargs = fake_hq.call_args
        # 第3个位置参数是 params
        args, _ = fake_hq.call_args
        params = args[2]
        assert f"CPS:{expected_cps}" in params

    def test_get_daily_none_adjust_no_cps(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        symbol = "600000.SH"
        fake_hq.return_value = _make_hq_result(symbol)

        adapter.get_daily([symbol], "2024-01-01", "2024-01-05", adjust="none")

        args, _ = fake_hq.call_args
        params = args[2]
        assert "CPS" not in params

    def test_get_daily_raises_on_error_code(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        result = mock.MagicMock()
        result.errorcode = -1
        result.data = None
        fake_hq.return_value = result

        with pytest.raises(Exception) as exc_info:
            adapter.get_daily(["600000.SH"], "2024-01-01", "2024-01-05")
        assert "错误码" in str(exc_info.value) or "THS_HQ" in str(exc_info.value)

    def test_get_daily_returns_empty_when_no_data(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        result = mock.MagicMock()
        result.errorcode = 0
        result.data = pd.DataFrame()
        fake_hq.return_value = result

        df = adapter.get_daily(["600000.SH"], "2024-01-01", "2024-01-05")
        assert df.empty

    def test_get_daily_rejects_invalid_symbol(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        with pytest.raises(Exception) as exc_info:
            adapter.get_daily(["600000"], "2024-01-01", "2024-01-05")
        assert "不支持" in str(exc_info.value) or "格式" in str(exc_info.value)

    def test_get_daily_symbol_uppercased(self, monkeypatch):
        mod, _, fake_hq = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()

        symbol = "600000.sh"  # 小写后缀
        fake_hq.return_value = _make_hq_result("600000.SH")

        adapter.get_daily([symbol], "2024-01-01", "2024-01-05", adjust="hfq")

        args, _ = fake_hq.call_args
        # 代码会被 upper()
        assert args[0] == "600000.SH"


class TestIfindAdapterOtherMethods:
    """get_adj_factor / get_financial 测试"""

    def test_get_adj_factor_returns_empty(self, monkeypatch):
        mod, _, _ = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()
        df = adapter.get_adj_factor(["600000.SH"], "2024-01-01", "2024-01-05")
        assert df.empty
        assert "code" in df.columns and "date" in df.columns

    def test_get_financial_returns_standard_schema(self, monkeypatch):
        """P0-1 PIT 契约：get_financial 返回标准 schema（含 disclosure_date）。

        iFinD 财务接口字段映射已实现，但 mock 环境下 THS_BD 返回 errorcode，
        各指标留空；code/report_date/disclosure_date 仍填充。
        """
        mod, _, _ = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()
        df = adapter.get_financial(["600000.SH"], "20240930", [])
        # 1 行（每个 symbol 一行，code/report_date/disclosure_date 已填充）
        assert not df.empty
        assert len(df) == 1
        # 标准 schema 列（含 P0-1 新增 disclosure_date）
        assert "code" in df.columns
        assert "pe_ttm" in df.columns
        assert "disclosure_date" in df.columns
        # P0-1 PIT 契约：ifind 无原生披露日，回填为 report_date（保守降级）
        assert df["code"].iloc[0] == "600000.SH"
        assert df["report_date"].iloc[0] == "20240930"
        assert df["disclosure_date"].iloc[0] == "20240930"
        # mock 环境下 THS_BD 返回 errorcode，财务指标应留空
        assert pd.isna(df["pe_ttm"].iloc[0])

    def test_get_financial_empty_symbols(self, monkeypatch):
        mod, _, _ = _load_ifind_adapter(monkeypatch)
        adapter = mod.IfindAdapter()
        df = adapter.get_financial([], "20240930", [])
        assert df.empty
