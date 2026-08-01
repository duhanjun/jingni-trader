"""factor-engine L2 单元测试：SupportResistanceCalculator + FibonacciCalculator。

覆盖：
- fibonacci.calculate() 返回 7 个回撤位且结构正确
- support_resistance.calculate_all() 返回 resistance/support 列表
- current_price 正确设置
- nearest_resistance / nearest_support 计算
- 空 DataFrame 边界
- 上涨趋势 vs 下跌趋势数据
- _round_numbers()
- _ma_levels()
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


def _load_levels_modules():
    """加载 scripts.levels.fibonacci 与 scripts.levels.support_resistance。

    support_resistance.py 使用 `from .fibonacci import FibonacciCalculator`
    相对导入，需建立 scripts / scripts.levels 包层级。

    Returns:
        (fibonacci_mod, support_resistance_mod)
    """
    saved = {k: sys.modules.get(k) for k in list(sys.modules.keys())
             if k == "scripts" or k.startswith("scripts.")}
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)

    for _m in ("talib", "pandas_ta"):
        if _m not in sys.modules:
            sys.modules[_m] = mock.MagicMock()

    scripts_dir = os.path.join(FACTOR_ENGINE_DIR, "scripts")
    _setup_pkg("scripts", scripts_dir)
    _setup_pkg("scripts.levels", os.path.join(scripts_dir, "levels"))

    # 先加载 fibonacci（被 support_resistance 依赖）
    fib_path = os.path.join(scripts_dir, "levels", "fibonacci.py")
    spec = ilu.spec_from_file_location("scripts.levels.fibonacci", fib_path)
    fib_mod = ilu.module_from_spec(spec)
    sys.modules["scripts.levels.fibonacci"] = fib_mod
    spec.loader.exec_module(fib_mod)

    # 再加载 support_resistance
    sr_path = os.path.join(scripts_dir, "levels", "support_resistance.py")
    spec = ilu.spec_from_file_location("scripts.levels.support_resistance", sr_path)
    sr_mod = ilu.module_from_spec(spec)
    sys.modules["scripts.levels.support_resistance"] = sr_mod
    spec.loader.exec_module(sr_mod)

    # 恢复 scripts.* 缓存
    for key in list(sys.modules.keys()):
        if key == "scripts" or key.startswith("scripts."):
            sys.modules.pop(key, None)
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v

    return fib_mod, sr_mod


def _make_trend_df(n_days=120, trend="up", base_price=10.0, seed=42):
    """构造单只股票的 OHLCV 数据，指定上涨或下跌趋势。

    Args:
        n_days: 交易日数
        trend: "up" 或 "down"
        base_price: 起始价格
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)

    if trend == "up":
        # 收盘价单调递增（带噪声）
        step = 0.05
        closes = base_price + np.arange(n_days) * step + rng.normal(0, 0.1, n_days)
    else:
        # 收盘价单调递减
        step = -0.05
        closes = base_price + np.arange(n_days) * step + rng.normal(0, 0.1, n_days)

    closes = closes.round(2)
    opens = closes * (1 + rng.normal(0, 0.002, n_days))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    vol = rng.randint(1_000_000, 10_000_000, n_days)

    return pd.DataFrame({
        "code": "000001.SZ",
        "date": dates,
        "open": opens.round(2),
        "high": highs.round(2),
        "low": lows.round(2),
        "close": closes,
        "volume": vol,
    })


# =====================================================================
# FibonacciCalculator 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestFibonacciCalculator:
    """FibonacciCalculator 行为测试。"""

    def test_calculate_returns_7_levels(self):
        """calculate() 返回 7 个回撤位"""
        fib_mod, _ = _load_levels_modules()
        calc = fib_mod.FibonacciCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        levels = calc.calculate(df)
        assert len(levels) == 7

    def test_calculate_level_structure(self):
        """每个回撤位含 level/name/price/type 字段"""
        fib_mod, _ = _load_levels_modules()
        calc = fib_mod.FibonacciCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        levels = calc.calculate(df)
        expected_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        for i, lv in enumerate(levels):
            assert lv["level"] == expected_levels[i]
            assert "name" in lv
            assert "price" in lv
            assert "type" in lv
            assert lv["type"] in ("support", "resistance")

    def test_calculate_with_empty_dataframe(self):
        """空 DataFrame → 返回空列表"""
        fib_mod, _ = _load_levels_modules()
        calc = fib_mod.FibonacciCalculator()
        assert calc.calculate(pd.DataFrame()) == []

    def test_calculate_uptrend_vs_downtrend(self):
        """上涨趋势与下跌趋势的回撤价格不同"""
        fib_mod, _ = _load_levels_modules()
        calc = fib_mod.FibonacciCalculator()

        up_df = _make_trend_df(n_days=120, trend="up", base_price=10.0, seed=1)
        down_df = _make_trend_df(n_days=120, trend="down", base_price=20.0, seed=2)

        up_levels = calc.calculate(up_df)
        down_levels = calc.calculate(down_df)

        assert len(up_levels) == 7
        assert len(down_levels) == 7

        # 上涨趋势: 0.0=高点, 1.0=低点 → price 递减
        # 下跌趋势: 0.0=低点, 1.0=高点 → price 递增
        up_prices = [lv["price"] for lv in up_levels]
        down_prices = [lv["price"] for lv in down_levels]

        # 上涨趋势：0.0(高点) > 1.0(低点)
        assert up_prices[0] > up_prices[-1]
        # 下跌趋势：0.0(低点) < 1.0(高点)
        assert down_prices[0] < down_prices[-1]


# =====================================================================
# SupportResistanceCalculator 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestSupportResistanceCalculator:
    """SupportResistanceCalculator 行为测试。"""

    def test_calculate_all_returns_required_keys(self):
        """calculate_all() 返回 resistance/support/current_price/nearest_*"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        result = calc.calculate_all(df)

        assert "resistance" in result
        assert "support" in result
        assert "current_price" in result
        assert "nearest_resistance" in result
        assert "nearest_support" in result

        assert isinstance(result["resistance"], list)
        assert isinstance(result["support"], list)

    def test_current_price_set_correctly(self):
        """current_price 默认取最新收盘价"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        result = calc.calculate_all(df)
        expected = float(df["close"].iloc[-1])
        assert result["current_price"] == expected

    def test_current_price_explicit(self):
        """显式传入 current_price 时使用传入值"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        result = calc.calculate_all(df, current_price=999.0)
        assert result["current_price"] == 999.0

    def test_nearest_resistance_and_support_computed(self):
        """nearest_resistance > current_price > nearest_support（当两者都存在时）"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        result = calc.calculate_all(df)

        # 当存在阻力位和支撑位时，验证相对关系
        if result["nearest_resistance"] is not None and result["nearest_support"] is not None:
            assert result["nearest_resistance"] > result["current_price"]
            assert result["nearest_support"] < result["current_price"]

        # 阻力升序：最近者优先（第一个就是 nearest）
        if result["resistance"]:
            assert result["resistance"][0]["price"] == result["nearest_resistance"]
        # 支撑降序：最近者优先（第一个就是 nearest）
        if result["support"]:
            assert result["support"][0]["price"] == result["nearest_support"]

    def test_calculate_all_with_empty_dataframe(self):
        """空 DataFrame → 返回空结果"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        result = calc.calculate_all(pd.DataFrame())
        assert result["resistance"] == []
        assert result["support"] == []
        assert result["current_price"] is None
        assert result["nearest_resistance"] is None
        assert result["nearest_support"] is None

    def test_resistance_entries_have_required_fields(self):
        """resistance/support 每条含 price/type/strength/method"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        result = calc.calculate_all(df)

        for entry in result["resistance"] + result["support"]:
            assert "price" in entry
            assert "type" in entry
            assert "strength" in entry
            assert "method" in entry

    def test_uptrend_vs_downtrend(self):
        """上涨 vs 下跌数据，结果都应正常返回"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()

        up_df = _make_trend_df(n_days=120, trend="up")
        down_df = _make_trend_df(n_days=120, trend="down")

        up_result = calc.calculate_all(up_df)
        down_result = calc.calculate_all(down_df)

        # 两种趋势都应能算出 current_price
        assert up_result["current_price"] is not None
        assert down_result["current_price"] is not None
        # 上涨趋势末尾价格应高于下跌趋势末尾价格
        assert up_result["current_price"] > down_result["current_price"]


# =====================================================================
# _round_numbers() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestRoundNumbers:
    """_round_numbers() 整数关口计算测试。"""

    def test_round_numbers_returns_list(self):
        """_round_numbers() 返回列表"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        levels = calc._round_numbers(14.32)
        assert isinstance(levels, list)
        assert len(levels) > 0

    def test_round_numbers_prices_are_multiples(self):
        """返回的价位是 1/5/10/50/100 的倍数"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        levels = calc._round_numbers(14.32)
        for lv in levels:
            assert lv["method"] == "round_number"
            assert lv["price"] > 0
            # 价格应是某个步长的整数倍
            steps = [1, 5, 10, 50, 100]
            assert any(abs(lv["price"] - round(lv["price"])) < 1e-6
                       for _ in steps)

    def test_round_numbers_with_zero_or_negative(self):
        """current_price <= 0 → 返回空列表"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        assert calc._round_numbers(0) == []
        assert calc._round_numbers(-5) == []


# =====================================================================
# _ma_levels() 测试
# =====================================================================

@pytest.mark.skill_factor_engine
@pytest.mark.unit
class TestMaLevels:
    """_ma_levels() 均线支撑/阻力位测试。"""

    def test_ma_levels_returns_entries(self):
        """数据足够时 _ma_levels() 返回均线位"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        levels = calc._ma_levels(df)
        assert isinstance(levels, list)
        assert len(levels) > 0

    def test_ma_levels_method_naming(self):
        """每条均线位 method 为 ma_XXX"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        df = _make_trend_df(n_days=120, trend="up")
        levels = calc._ma_levels(df)
        for lv in levels:
            assert lv["method"].startswith("ma_")
            assert "price" in lv
            assert "type_label" in lv
            assert "strength" in lv

    def test_ma_levels_with_short_data(self):
        """数据不足以计算任何均线时返回空列表"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        # MA5 至少需要 5 行
        df = _make_trend_df(n_days=3, trend="up")
        levels = calc._ma_levels(df)
        assert levels == []

    def test_ma_levels_empty_dataframe(self):
        """空 DataFrame → 返回空列表"""
        _, sr_mod = _load_levels_modules()
        calc = sr_mod.SupportResistanceCalculator()
        assert calc._ma_levels(pd.DataFrame()) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
