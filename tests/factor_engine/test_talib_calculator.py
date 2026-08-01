"""factor-engine L2 单元测试：registry-based TalibCalculator。

覆盖：
- get_available_factors() 返回 150+ 因子
- get_factor_info() 元信息查询（已知因子 / 未知因子）
- calculate() 在 mocked talib 下的行为
- 多输出缓存（macd/macd_signal/macd_hist 只调用一次 talib.MACD）
- 61 个 CDL 形态全部注册
- 未知因子 → ValueError
- 空 DataFrame → 原样返回
- 类别覆盖至少 8 类
"""
from __future__ import annotations

import os
import sys
import importlib.util as ilu
from unittest import mock

import pytest
import pandas as pd
import numpy as np

from synthetic_data import make_synthetic_daily


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACTOR_ENGINE_DIR = os.path.join(ROOT, "skills", "factor-engine")


def _setup_pkg(pkg_name: str, pkg_path: str) -> None:
    """把指定路径注册为 sys.modules 中的包。"""
    init_py = os.path.join(pkg_path, "__init__.py")
    if os.path.exists(init_py):
        spec = ilu.spec_from_file_location(
            pkg_name, init_py,
            submodule_search_locations=[pkg_path],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)
    else:
        spec = ilu.spec_from_file_location(
            pkg_name, None,
            submodule_search_locations=[pkg_path],
        )
        pkg = ilu.module_from_spec(spec)
        sys.modules[pkg_name] = pkg


def _load_talib_calculator():
    """加载 scripts.adapters.talib_calculator 为独立模块。

    talib_calculator.py 使用了 `from ..base.base_factor_calculator import ...`
    相对导入，因此需要先建立 scripts / scripts.base / scripts.adapters 包层级。
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    # 确保 talib 已 mock（conftest 已做，这里做幂等保护）
    for _m in ("talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    scripts_dir = os.path.join(FACTOR_ENGINE_DIR, "scripts")
    _setup_pkg("scripts", scripts_dir)
    _setup_pkg("scripts.base", os.path.join(scripts_dir, "base"))
    _setup_pkg("scripts.adapters", os.path.join(scripts_dir, "adapters"))

    # 预加载 base_factor_calculator
    bfc_path = os.path.join(scripts_dir, "base", "base_factor_calculator.py")
    spec = ilu.spec_from_file_location("scripts.base.base_factor_calculator", bfc_path)
    bfc_mod = ilu.module_from_spec(spec)
    sys.modules["scripts.base.base_factor_calculator"] = bfc_mod
    spec.loader.exec_module(bfc_mod)

    # 加载 talib_calculator（使用全限定名以支持相对导入）
    tc_path = os.path.join(scripts_dir, "adapters", "talib_calculator.py")
    spec = ilu.spec_from_file_location("scripts.adapters.talib_calculator", tc_path)
    tc_mod = ilu.module_from_spec(spec)
    sys.modules["scripts.adapters.talib_calculator"] = tc_mod
    spec.loader.exec_module(tc_mod)

    # 恢复 scripts.* 缓存（模块本身的命名空间已填充，不受影响）
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v

    return tc_mod


def _make_fake_talib(func_returns: dict | None = None):
    """构造一个 talib 替身对象。

    Args:
        func_returns: {函数名: 返回值或 side_effect}，
                      未指定的函数返回 MagicMock。
    """
    fake = mock.MagicMock()
    if func_returns:
        for name, ret in func_returns.items():
            if callable(ret):
                setattr(fake, name, mock.MagicMock(side_effect=ret))
            else:
                setattr(fake, name, mock.MagicMock(return_value=ret))
    return fake


# =====================================================================
# 注册表元信息测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestTalibCalculatorRegistry:
    """注册表元信息与枚举测试。"""

    def test_get_available_factors_returns_150_plus(self):
        """get_available_factors() 返回 150+ 因子"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()
        factors = calc.get_available_factors()
        assert len(factors) >= 150

    def test_get_factor_info_returns_metadata(self):
        """get_factor_info() 对已知因子返回完整元信息"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()
        info = calc.get_factor_info("macd")
        assert info, "macd 元信息不应为空"
        assert info["category"] == "momentum"
        assert info["func"] == "MACD"
        assert "direction" in info
        assert "params" in info
        assert "inputs" in info
        assert info["inputs"] == ["close"]
        assert "output_idx" in info
        assert info["output_idx"] == 0

    def test_get_factor_info_unknown_returns_empty(self):
        """get_factor_info() 对未知因子返回空 dict"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()
        info = calc.get_factor_info("nonexistent_factor_xyz")
        assert info == {}

    def test_all_61_cdl_patterns_in_registry(self):
        """全部 61 个 CDL 形态名（小写）都在注册表中"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()
        factors = calc.get_available_factors()
        cdl_factors = [f for f in factors if f.startswith("cdl")]
        assert len(cdl_factors) == 61

    def test_category_coverage_at_least_8(self):
        """注册表覆盖至少 8 个类别"""
        mod = _load_talib_calculator()
        categories = {
            entry["category"]
            for entry in mod.TALIB_FUNCTION_REGISTRY.values()
        }
        expected = {
            "overlap", "momentum", "volume", "volatility",
            "price", "statistic", "cycle", "pattern",
        }
        assert expected.issubset(categories)
        assert len(categories) >= 8


# =====================================================================
# calculate() 行为测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestTalibCalculatorCalculate:
    """calculate() 在 mocked talib 下的行为测试。"""

    def test_calculate_with_mocked_talib(self):
        """calculate() 在 mocked talib 下正常返回带因子列的 DataFrame"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()

        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-04-30"
        )
        n = len(df)

        # 替换 talib 为可控 mock
        def _mock_ma(*args, **kwargs):
            return np.random.rand(len(args[0]))

        calc._func_cache.clear()
        mod.talib = _make_fake_talib({"MA": _mock_ma})

        result = calc.calculate(df, ["ma"])
        assert "ma" in result.columns
        assert len(result) == n
        assert result["ma"].notna().any()

    def test_multi_output_caching(self):
        """macd/macd_signal/macd_hist 只调用一次 talib.MACD（单只股票）"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()

        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-04-30"
        )

        call_count = [0]

        def _mock_macd(*args, **kwargs):
            call_count[0] += 1
            n = len(args[0])
            return (
                np.random.rand(n),
                np.random.rand(n),
                np.random.rand(n),
            )

        calc._func_cache.clear()
        mod.talib = _make_fake_talib({"MACD": _mock_macd})

        result = calc.calculate(df, ["macd", "macd_signal", "macd_hist"])
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
        # 单只股票 → 缓存命中 → talib.MACD 只调用一次
        assert call_count[0] == 1

    def test_calculate_raises_value_error_for_unknown(self):
        """calculate() 对未知因子抛 ValueError"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()
        df = make_synthetic_daily(
            codes=["000001.SZ"], start="2024-01-01", end="2024-02-28"
        )
        with pytest.raises(ValueError, match="不支持的因子"):
            calc.calculate(df, ["unknown_factor_xyz"])

    def test_calculate_handles_empty_dataframe(self):
        """calculate() 对空 DataFrame 原样返回"""
        mod = _load_talib_calculator()
        calc = mod.TalibCalculator()
        empty_df = pd.DataFrame()
        result = calc.calculate(empty_df, ["ma"])
        assert result.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
