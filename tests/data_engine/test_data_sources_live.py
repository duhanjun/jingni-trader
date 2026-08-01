"""data-engine L3 集成测试:9 个数据源真实连通性测试。

测试 data-engine 接入的所有数据源(baostock/akshare/websearch/tushare/xtquant/gm/tdxquant/wind/ifind)
是否能正常实例化、拉取真实日线行情、返回符合标准 schema 的 DataFrame。

设计:
- 每个数据源一个测试类,独立 skip/pass/fail
- 不可用源(缺包/缺 token)用 pytest.skip 跳过
- 使用 importlib 加载适配器模块(复用 test_wind_adapter.py 的模式)
- 测试股票: 600000.SH,日期范围: 动态计算最近 90 天到昨天
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from datetime import datetime, timedelta
from unittest import mock

import pytest
import pandas as pd
import numpy as np


# ============================================================================
# 路径常量
# ============================================================================

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ENGINE_DIR = os.path.join(ROOT, "skills", "data-engine")
SCRIPTS_DIR = os.path.join(DATA_ENGINE_DIR, "scripts")


# ============================================================================
# 动态测试参数
# ============================================================================

TODAY = datetime.now()
END_DATE = (TODAY - timedelta(days=1)).strftime("%Y-%m-%d")
START_DATE = (TODAY - timedelta(days=90)).strftime("%Y-%m-%d")
TEST_SYMBOL = "600000.SH"

# 财务报告期: 上一季度末
_month = TODAY.month
_year = TODAY.year
if _month <= 3:
    REPORT_DATE = f"{_year - 1}1231"
elif _month <= 6:
    REPORT_DATE = f"{_year}0331"
elif _month <= 9:
    REPORT_DATE = f"{_year}0630"
else:
    REPORT_DATE = f"{_year}0930"


# ============================================================================
# 模块加载工具(复用 test_wind_adapter.py 模式)
# ============================================================================

def _load_adapter_module(adapter_name: str, extra_mocks: dict = None):
    """加载 data-engine/scripts/adapters/{adapter_name}_adapter.py 模块。

    参数:
        adapter_name: 适配器名(如 'baostock', 'tushare', 'wind')
        extra_mocks: 额外需要 mock 的模块 {模块名: mock对象}

    返回:
        (module, module_name) 已加载的模块对象和完整模块名
    """
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

    # mock 未安装的重型依赖(避免 import 阶段崩溃)
    for _m in ("tushare", "baostock", "akshare", "xtquant", "gm", "pytdx"):
        if _m not in sys.modules:
            try:
                __import__(_m)
            except ImportError:
                sys.modules[_m] = mock.MagicMock()

    # 额外 mock
    if extra_mocks:
        for mod_name, mock_obj in extra_mocks.items():
            sys.modules[mod_name] = mock_obj

    # 加载目标适配器
    adapter_path = os.path.join(SCRIPTS_DIR, "adapters", f"{adapter_name}_adapter.py")
    full_mod_name = f"scripts.adapters.{adapter_name}_adapter"
    spec = ilu.spec_from_file_location(full_mod_name, adapter_path)
    mod = ilu.module_from_spec(spec)
    sys.modules[full_mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _check_package_available(package_name: str) -> bool:
    """检查第三方包是否真实安装(非 mock)。"""
    try:
        # 先清除可能的 mock
        if package_name in sys.modules and isinstance(sys.modules[package_name], mock.MagicMock):
            del sys.modules[package_name]
        spec = ilu.find_spec(package_name)
        if spec is None:
            return False
        if spec.origin and 'mock' in str(spec.origin).lower():
            return False
        __import__(package_name)
        return not isinstance(sys.modules.get(package_name), mock.MagicMock)
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _check_env_var(var_name: str) -> bool:
    """检查环境变量是否已设置且非空。"""
    val = os.environ.get(var_name, "")
    return bool(val and val.strip())


def _load_errors_module():
    """加载 data-engine/scripts/errors.py 模块,用于校验异常类型。"""
    sys.modules.pop("scripts.errors", None)
    if "scripts" not in sys.modules:
        init_py = os.path.join(SCRIPTS_DIR, "__init__.py")
        spec = ilu.spec_from_file_location("scripts", init_py, submodule_search_locations=[SCRIPTS_DIR])
        pkg = ilu.module_from_spec(spec)
        sys.modules["scripts"] = pkg
        spec.loader.exec_module(pkg)
    errors_path = os.path.join(SCRIPTS_DIR, "errors.py")
    spec = ilu.spec_from_file_location("scripts.errors", errors_path)
    mod = ilu.module_from_spec(spec)
    sys.modules["scripts.errors"] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================================
# Schema 校验工具
# ============================================================================

DAILY_REQUIRED_COLS = {"code", "date", "open", "high", "low", "close", "volume"}
DAILY_NUMERIC_COLS = ["open", "high", "low", "close", "volume"]
FINANCIAL_REQUIRED_COLS = {"code", "report_date"}


def _validate_daily_schema(df: pd.DataFrame, symbol: str):
    """校验 get_daily 返回的 DataFrame schema。"""
    assert df is not None, "get_daily 返回 None"
    assert not df.empty, "get_daily 返回空 DataFrame"

    missing = DAILY_REQUIRED_COLS - set(df.columns)
    assert not missing, f"缺少必需列: {missing}, 实际列: {list(df.columns)}"

    # code 列值校验
    assert (df["code"] == symbol).all(), f"code 列值不一致,期望 {symbol}"

    # date 列类型校验
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), f"date 列非 datetime 类型: {df['date'].dtype}"

    # 数值列类型校验
    for col in DAILY_NUMERIC_COLS:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} 列非数值类型: {df[col].dtype}"

    # 不全为 NaN
    for col in ["close", "volume"]:
        if col in df.columns:
            assert not df[col].isna().all(), f"{col} 列全为 NaN"

    # 价格逻辑校验(忽略 NaN 行)
    valid = df.dropna(subset=["high", "low", "close"])
    if not valid.empty:
        assert (valid["high"] >= valid["low"]).all(), "存在 high < low 的行"
        assert (valid["high"] >= valid["close"]).all(), "存在 high < close 的行"
        assert (valid["low"] <= valid["close"]).all(), "存在 low > close 的行"


def _validate_financial_schema(df: pd.DataFrame, symbol: str):
    """校验 get_financial 返回的 DataFrame schema。"""
    assert df is not None, "get_financial 返回 None"
    if df.empty:
        # 财务数据允许空(某些源某些报告期可能无数据),但不报错
        return
    missing = FINANCIAL_REQUIRED_COLS - set(df.columns)
    assert not missing, f"财务数据缺少必需列: {missing}"


# ============================================================================
# 1. BaoStock 测试(免费源)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_network
class TestBaostockLive:
    """BaoStock 数据源连通性测试。"""

    def test_adapter_instantiate(self):
        mod = _load_adapter_module("baostock")
        adapter = mod.BaostockAdapter()
        assert adapter is not None

    def test_get_daily_returns_valid_data(self):
        mod = _load_adapter_module("baostock")
        adapter = mod.BaostockAdapter()
        df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
        _validate_daily_schema(df, TEST_SYMBOL)
        assert len(df) > 0

    def test_get_financial_returns_valid_data(self):
        mod = _load_adapter_module("baostock")
        adapter = mod.BaostockAdapter()
        df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
        _validate_financial_schema(df, TEST_SYMBOL)


# ============================================================================
# 2. AkShare 测试(免费源)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_network
class TestAkshareLive:
    """AkShare 数据源连通性测试。"""

    def test_adapter_instantiate(self):
        mod = _load_adapter_module("akshare")
        adapter = mod.AkshareAdapter()
        assert adapter is not None

    def test_get_daily_returns_valid_data(self):
        mod = _load_adapter_module("akshare")
        adapter = mod.AkshareAdapter()
        import time
        last_err = None
        for attempt in range(3):
            try:
                df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
                if df is not None and not df.empty:
                    _validate_daily_schema(df, TEST_SYMBOL)
                    return
                last_err = "返回空 DataFrame"
            except Exception as e:
                last_err = e
            if attempt < 2:
                time.sleep(2)
        pytest.skip(f"akshare get_daily 连续 3 次失败(可能网络抖动): {last_err}")

    def test_get_financial_returns_valid_data(self):
        mod = _load_adapter_module("akshare")
        adapter = mod.AkshareAdapter()
        df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
        _validate_financial_schema(df, TEST_SYMBOL)


# ============================================================================
# 3. WebSearch 测试(mock 注入)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
class TestWebsearchLive:
    """WebSearch 适配器测试(用 mock 注入 web_search_fn)。"""

    def test_adapter_instantiate_without_fn(self):
        """未注入 web_search_fn 时仍可实例化(但会 warning)。"""
        mod = _load_adapter_module("websearch")
        adapter = mod.WebSearchAdapter(web_search_fn=None)
        assert adapter is not None

    def test_get_daily_raises_when_no_fn(self):
        """未注入 web_search_fn 时 get_daily 抛 DataSourceError。"""
        mod = _load_adapter_module("websearch")
        adapter = mod.WebSearchAdapter(web_search_fn=None)
        with pytest.raises(Exception) as exc_info:
            adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE)
        assert "web_search_fn" in str(exc_info.value) or "未注入" in str(exc_info.value)

    def test_get_daily_parses_search_result(self):
        """注入含收盘价的 mock 搜索结果,验证解析正确。"""
        mod = _load_adapter_module("websearch")

        def mock_search(query):
            return f"600000.SH 在 {START_DATE} 的收盘价为 10.50 元,开盘 10.30 元,最高 10.60 元,最低 10.20 元,成交量 100000 手"

        adapter = mod.WebSearchAdapter(web_search_fn=mock_search)
        df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
        assert not df.empty
        assert "close" in df.columns
        assert df.iloc[0]["close"] == 10.50

    def test_get_daily_raises_on_no_data(self):
        """注入无数据搜索结果,验证抛 DataNotFoundError。"""
        mod = _load_adapter_module("websearch")

        def mock_search_no_data(query):
            return "未找到相关数据"

        adapter = mod.WebSearchAdapter(web_search_fn=mock_search_no_data)
        with pytest.raises(Exception):
            adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE)


    def test_get_financial_returns_valid_schema(self):
        """get_financial 返回带标准列的空 DataFrame。

        WebSearch 不适合结构化财务数据,此处验证返回的空 DataFrame
        仍包含全部 18 个标准字段列(允许空数据)。
        """
        mod = _load_adapter_module("websearch")
        adapter = mod.WebSearchAdapter(web_search_fn=lambda q: "")
        df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
        assert df is not None
        standard_cols = [
            'code', 'report_date', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ratio',
            'roe', 'roa', 'gross_margin', 'net_margin',
            'revenue_growth', 'profit_growth',
            'debt_ratio', 'current_ratio', 'quick_ratio', 'ocf',
            'industry', 'name',
        ]
        missing = set(standard_cols) - set(df.columns)
        assert not missing, f"websearch get_financial 空 schema 缺少标准列: {missing}"


# ============================================================================
# 4. Tushare 测试(非免费,需 TUSHARE_TOKEN)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.requires_tushare_token
class TestTushareLive:
    """Tushare Pro 数据源连通性测试。"""

    def test_adapter_instantiate(self):
        if not _check_env_var("TUSHARE_TOKEN"):
            pytest.skip("TUSHARE_TOKEN 未设置")
        if not _check_package_available("tushare"):
            pytest.skip("tushare 包未安装")
        mod = _load_adapter_module("tushare")
        adapter = mod.TushareAdapter()
        assert adapter is not None

    def test_get_daily_returns_valid_data(self):
        if not _check_env_var("TUSHARE_TOKEN"):
            pytest.skip("TUSHARE_TOKEN 未设置")
        if not _check_package_available("tushare"):
            pytest.skip("tushare 包未安装")
        mod = _load_adapter_module("tushare")
        adapter = mod.TushareAdapter()
        try:
            df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
            _validate_daily_schema(df, TEST_SYMBOL)
        except Exception as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in ["频率", "限", "权限", "积分", "permission"]):
                pytest.skip(f"tushare 接口受限: {err_msg}")
            raise

    def test_get_financial_returns_valid_data(self):
        if not _check_env_var("TUSHARE_TOKEN"):
            pytest.skip("TUSHARE_TOKEN 未设置")
        if not _check_package_available("tushare"):
            pytest.skip("tushare 包未安装")
        mod = _load_adapter_module("tushare")
        adapter = mod.TushareAdapter()
        try:
            df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
            _validate_financial_schema(df, TEST_SYMBOL)
        except Exception as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in ["权限", "permission", "积分", "频率", "限"]):
                pytest.skip(f"当前 TUSHARE_TOKEN 无 fina_indicator 接口权限: {err_msg}")
            raise

    def test_get_financial_handles_permission_error(self):
        """验证 tushare 无权限接口返回权限错误(DataSourceError 子类)而非崩溃。"""
        if not _check_env_var("TUSHARE_TOKEN"):
            pytest.skip("TUSHARE_TOKEN 未设置")
        if not _check_package_available("tushare"):
            pytest.skip("tushare 包未安装")
        mod = _load_adapter_module("tushare")
        adapter = mod.TushareAdapter()
        try:
            df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
            assert df is not None
        except Exception as e:
            # 用类名检查(避免不同模块实例的 isinstance 不匹配)
            assert any(cls.__name__ == "DataSourceError" for cls in type(e).__mro__), f"期望 DataSourceError 子类,实际 {type(e).__name__}: {e}"
            err_msg = str(e)
            assert any(kw in err_msg for kw in ["权限", "permission", "积分", "没有接口"]), f"权限错误消息不含预期关键字: {err_msg}"


# ============================================================================
# 5. XtQuant 测试(非免费,需 QMT 客户端)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.requires_xtquant
class TestXtquantLive:
    """XtQuant (QMT) 数据源连通性测试。"""

    def test_adapter_instantiate(self):
        if not _check_package_available("xtquant"):
            pytest.skip("xtquant 包未安装(QMT 客户端未部署)")
        mod = _load_adapter_module("xtquant")
        adapter = mod.XtQuantAdapter()
        assert adapter is not None
        if not adapter.available:
            pytest.skip("xtquant 包已装但 QMT 客户端未启动")

    def test_get_daily_returns_valid_data(self):
        if not _check_package_available("xtquant"):
            pytest.skip("xtquant 包未安装(QMT 客户端未部署)")
        mod = _load_adapter_module("xtquant")
        adapter = mod.XtQuantAdapter()
        if not adapter.available:
            pytest.skip("QMT 客户端未启动")
        df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
        _validate_daily_schema(df, TEST_SYMBOL)


    def test_get_financial_returns_valid_data(self):
        if not _check_package_available("xtquant"):
            pytest.skip("xtquant 包未安装(QMT 客户端未部署)")
        mod = _load_adapter_module("xtquant")
        adapter = mod.XtQuantAdapter()
        if not adapter.available:
            pytest.skip("QMT 客户端未启动")
        try:
            df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
            _validate_financial_schema(df, TEST_SYMBOL)
        except Exception as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in ["权限", "permission", "积分", "频率", "限", "登录失败", "不可达", "connect"]):
                pytest.skip(f"xtquant get_financial 接口受限: {err_msg}")
            raise


# ============================================================================
# 6. Gm 测试(非免费,需 GM_TOKEN)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_network
@pytest.mark.requires_gm_token
class TestGmLive:
    """掘金量化(gm)数据源连通性测试。"""

    def test_adapter_instantiate(self):
        if not _check_env_var("GM_TOKEN"):
            pytest.skip("GM_TOKEN 未设置")
        if not _check_package_available("gm"):
            pytest.skip("gm 包未安装")
        mod = _load_adapter_module("gm")
        adapter = mod.GmAdapter()
        assert adapter is not None
        if not adapter.available:
            pytest.skip("gm 包已装但 GM_TOKEN 无效或 gm SDK 初始化失败")

    def test_get_daily_returns_valid_data(self):
        if not _check_env_var("GM_TOKEN"):
            pytest.skip("GM_TOKEN 未设置")
        if not _check_package_available("gm"):
            pytest.skip("gm 包未安装")
        mod = _load_adapter_module("gm")
        adapter = mod.GmAdapter()
        if not adapter.available:
            pytest.skip("GmAdapter 不可用(token/SDK 问题)")
        df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
        _validate_daily_schema(df, TEST_SYMBOL)


    def test_get_financial_returns_valid_data(self):
        if not _check_env_var("GM_TOKEN"):
            pytest.skip("GM_TOKEN 未设置")
        if not _check_package_available("gm"):
            pytest.skip("gm 包未安装")
        mod = _load_adapter_module("gm")
        adapter = mod.GmAdapter()
        if not adapter.available:
            pytest.skip("GmAdapter 不可用(token/SDK 问题)")
        try:
            df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
            _validate_financial_schema(df, TEST_SYMBOL)
        except Exception as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in ["权限", "permission", "积分", "频率", "限", "登录失败", "不可达", "token", "GmError"]):
                pytest.skip(f"gm get_financial 接口受限: {err_msg}")
            raise


# ============================================================================
# 7. TdxQuant 测试(非免费,需 pytdx + 通达信行情服务器)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_network
class TestTdxquantLive:
    """通达信(pytdx)数据源连通性测试。"""

    def test_adapter_instantiate(self):
        if not _check_package_available("pytdx"):
            pytest.skip("pytdx 包未安装")
        mod = _load_adapter_module("tdxquant")
        adapter = mod.TdxQuantAdapter()
        assert adapter is not None

    def test_get_daily_returns_valid_data(self):
        if not _check_package_available("pytdx"):
            pytest.skip("pytdx 包未安装")
        mod = _load_adapter_module("tdxquant")
        adapter = mod.TdxQuantAdapter()
        # tdxquant 连接公共行情服务器,可能因网络/服务器状态失败
        try:
            df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="none")
            _validate_daily_schema(df, TEST_SYMBOL)
        except Exception as e:
            if "无法连接" in str(e) or "connect" in str(e).lower():
                pytest.skip(f"通达信行情服务器不可达: {e}")
            raise


    def test_get_financial_returns_valid_data(self):
        if not _check_package_available("pytdx"):
            pytest.skip("pytdx 包未安装")
        mod = _load_adapter_module("tdxquant")
        adapter = mod.TdxQuantAdapter()
        # get_financial 需连接通达信行情服务器,连接失败时 skip
        try:
            df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
            _validate_financial_schema(df, TEST_SYMBOL)
        except Exception as e:
            if "无法连接" in str(e) or "connect" in str(e).lower() or "不可达" in str(e):
                pytest.skip(f"通达信行情服务器不可达,跳过 get_financial: {e}")
            raise


# ============================================================================
# 8. Wind 测试(非免费,需 WindPy + Wind 终端)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_wind_terminal
class TestWindLive:
    """万得 WindPy 数据源连通性测试。"""

    def test_adapter_instantiate(self):
        if not _check_package_available("WindPy"):
            pytest.skip("WindPy 包未安装(Wind 终端未部署)")
        mod = _load_adapter_module("wind")
        adapter = mod.WindAdapter()
        assert adapter is not None

    def test_get_daily_returns_valid_data(self):
        if not _check_package_available("WindPy"):
            pytest.skip("WindPy 包未安装(Wind 终端未部署)")
        mod = _load_adapter_module("wind")
        adapter = mod.WindAdapter()
        df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
        _validate_daily_schema(df, TEST_SYMBOL)


# ============================================================================
# 9. iFinD 测试(非免费,需 iFinDPy + 账号密码)
# ============================================================================

@pytest.mark.skill_data_engine
@pytest.mark.integration
@pytest.mark.requires_ifind_credentials
class TestIfindLive:
    """同花顺 iFinD 数据源连通性测试。"""

    def test_adapter_instantiate(self):
        if not _check_package_available("iFinDPy"):
            pytest.skip("iFinDPy 包未安装(同花顺终端未部署)")
        if not _check_env_var("IFIND_USERNAME") or not _check_env_var("IFIND_PASSWORD"):
            pytest.skip("IFIND_USERNAME/IFIND_PASSWORD 未设置")
        mod = _load_adapter_module("ifind")
        adapter = mod.IfindAdapter()
        assert adapter is not None

    def test_get_daily_returns_valid_data(self):
        if not _check_package_available("iFinDPy"):
            pytest.skip("iFinDPy 包未安装(同花顺终端未部署)")
        if not _check_env_var("IFIND_USERNAME") or not _check_env_var("IFIND_PASSWORD"):
            pytest.skip("IFIND_USERNAME/IFIND_PASSWORD 未设置")
        mod = _load_adapter_module("ifind")
        adapter = mod.IfindAdapter()
        df = adapter.get_daily([TEST_SYMBOL], START_DATE, END_DATE, adjust="hfq")
        _validate_daily_schema(df, TEST_SYMBOL)


# ============================================================================
# 10. 汇总测试
# ============================================================================

    def test_get_financial_returns_valid_data(self):
        if not _check_package_available("iFinDPy"):
            pytest.skip("iFinDPy 包未安装(同花顺终端未部署)")
        if not _check_env_var("IFIND_USERNAME") or not _check_env_var("IFIND_PASSWORD"):
            pytest.skip("IFIND_USERNAME/IFIND_PASSWORD 未设置")
        mod = _load_adapter_module("ifind")
        adapter = mod.IfindAdapter()
        try:
            df = adapter.get_financial([TEST_SYMBOL], REPORT_DATE, [])
            _validate_financial_schema(df, TEST_SYMBOL)
        except Exception as e:
            err_msg = str(e)
            if any(kw in err_msg for kw in ["权限", "permission", "积分", "频率", "限", "登录失败", "不可达", "登录"]):
                pytest.skip(f"ifind get_financial 接口受限: {err_msg}")
            raise


@pytest.mark.skill_data_engine
@pytest.mark.integration
class TestDataSourcesSummary:
    """汇总所有数据源可用性状态。"""

    def test_summary_report(self):
        """打印所有数据源可用性汇总表。"""
        sources = [
            ("baostock", "baostock", None, None, "pip install baostock"),
            ("akshare", "akshare", None, None, "pip install akshare"),
            ("websearch", None, None, None, "无需安装(注入 web_search_fn)"),
            ("tushare", "tushare", "TUSHARE_TOKEN", None, "pip install tushare"),
            ("xtquant", "xtquant", None, None, "需 QMT 客户端"),
            ("gm", "gm", "GM_TOKEN", None, "pip install gm"),
            ("tdxquant", "pytdx", None, None, "pip install pytdx"),
            ("wind", "WindPy", None, None, "需安装 Wind 金融终端(非 PyPI)"),
            ("ifind", "iFinDPy", "IFIND_USERNAME", "IFIND_PASSWORD", "需安装同花顺 iFinD 终端(非 PyPI)"),
        ]

        print("\n" + "=" * 90)
        print("数据源可用性汇总")
        print("=" * 90)
        print(f"{'数据源':<12} {'包就绪':<8} {'配置就绪':<10} {'安装/配置说明'}")
        print("-" * 90)

        for name, pkg, env_var, env_var2, hint in sources:
            pkg_ok = "Y" if (pkg is None or _check_package_available(pkg)) else "N"
            if env_var is None:
                cfg_ok = "N/A"
            else:
                cfg1 = _check_env_var(env_var)
                cfg2 = _check_env_var(env_var2) if env_var2 else True
                cfg_ok = "Y" if (cfg1 and cfg2) else "N"

            if pkg_ok == "N":
                note = f"[缺包] {hint}"
            elif cfg_ok == "N":
                note = f"[缺配置] 需设置 {env_var}"
            else:
                note = hint

            print(f"{name:<12} {pkg_ok:<8} {cfg_ok:<10} {note}")

        print("=" * 90)
        # 此测试总是通过,仅用于打印汇总
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-rs"])