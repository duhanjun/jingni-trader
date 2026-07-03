"""
边界条件测试

测试各种边界情况下的行为正确性：
- 空数据、单行数据
- 全 NaN / 部分缺失
- 单一行业、单一股票
- 极端值
- 常数列（零方差）
- 日期不足 min_samples
"""
import sys
import os
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optimizations.polars_ic_analysis import calc_ic_series_polars, ic_summary_stats
from optimizations.polars_neutralize import (
    neutralize_mcap_polars,
    neutralize_industry_mcap_polars,
)
from optimizations.vectorized_metrics import calc_enhanced_metrics
from optimizations.tests import generate_panel_data, generate_equity_curve


def test_empty_data():
    """测试空数据"""
    empty = pl.DataFrame(schema={
        "date": pl.Date, "code": pl.Utf8,
        "factor_1": pl.Float64, "ret_forward_5d": pl.Float64,
    })
    ic = calc_ic_series_polars(empty, "factor_1", "ret_forward_5d")
    assert ic.height == 0, "空数据应返回空 IC 序列"
    stats = ic_summary_stats(ic)
    assert stats["ic_mean"] == 0.0, "空数据统计应为 0"
    print("  [PASS] 空数据处理正确")


def test_single_row():
    """测试单行数据"""
    df = pl.DataFrame({
        "date": [pd.Timestamp("2024-01-01")],
        "code": ["000001.SZ"],
        "factor_1": [1.0],
        "ret_forward_5d": [0.01],
    })
    ic = calc_ic_series_polars(df, "factor_1", "ret_forward_5d", min_samples=10)
    assert ic.height == 0, "单行数据不足 min_samples 应返回空"
    print("  [PASS] 单行数据处理正确")


def test_all_nan():
    """测试全 NaN 列"""
    data = generate_panel_data(n_stocks=50, n_days=30, seed=10)
    data = data.with_columns(pl.lit(None).cast(pl.Float64).alias("bad_factor"))
    ic = calc_ic_series_polars(data, "bad_factor", "ret_forward_5d")
    assert ic.height == 0, "全 NaN 因子应返回空 IC 序列"
    print("  [PASS] 全 NaN 数据处理正确")


def test_partial_nan():
    """测试部分缺失值"""
    data = generate_panel_data(n_stocks=50, n_days=30, seed=11)
    # 随机置 20% 为 null
    rng = np.random.default_rng(11)
    mask = rng.random(data.height) < 0.2
    factor_vals = data["factor_1"].to_numpy().astype(float).copy()
    factor_vals[mask] = np.nan
    data = data.with_columns(pl.Series("factor_1", factor_vals))

    ic = calc_ic_series_polars(data, "factor_1", "ret_forward_5d")
    # 应能正常计算（drop_nulls 后）
    assert ic.height > 0, "部分缺失应仍能计算 IC"
    assert ic["ic"].is_not_null().all(), "IC 值不应有 null"
    print(f"  [PASS] 部分缺失值处理正确: {ic.height} 天有有效 IC")


def test_constant_factor():
    """测试常数因子（零方差）"""
    data = generate_panel_data(n_stocks=50, n_days=30, seed=12)
    data = data.with_columns(pl.lit(1.0).alias("const_factor"))
    ic = calc_ic_series_polars(data, "const_factor", "ret_forward_5d")
    # 常数因子相关性为 null/NaN，应被过滤
    assert ic.height == 0 or ic["ic"].null_count() == ic.height, "常数因子应无有效 IC"
    print("  [PASS] 常数因子处理正确")


def test_single_industry():
    """测试单一行业"""
    data = generate_panel_data(n_stocks=50, n_days=30, seed=13)
    data = data.with_columns(pl.lit("唯一行业").alias("industry"))
    result = neutralize_industry_mcap_polars(data, "factor_1", "industry", "lncap")
    assert "factor_1_neutral" in result.columns, "应生成 neutral 列"
    neutral = result["factor_1_neutral"].to_numpy()
    assert not np.all(np.isnan(neutral)), "中性化结果不应全为 NaN"
    print("  [PASS] 单一行业处理正确")


def test_few_samples():
    """测试样本数不足 min_samples"""
    # 每天只有 5 只股票，低于默认 min_samples=30
    data = generate_panel_data(n_stocks=5, n_days=30, seed=14)
    result = neutralize_mcap_polars(data, "factor_1", "lncap", min_samples=30)
    neutral = result["factor_1_neutral"].to_numpy()
    original = data["factor_1"].to_numpy()
    # 样本不足应返回原值
    np.testing.assert_allclose(neutral, original, rtol=1e-6)
    print("  [PASS] 样本不足时返回原值")


def test_extreme_values():
    """测试极端值"""
    data = generate_panel_data(n_stocks=50, n_days=30, seed=15)
    # 注入极端值
    rng = np.random.default_rng(15)
    vals = data["factor_1"].to_numpy().astype(float).copy()
    extreme_idx = rng.integers(0, len(vals), 5)
    vals[extreme_idx] = 1e10
    data = data.with_columns(pl.Series("factor_1", vals))

    ic = calc_ic_series_polars(data, "factor_1", "ret_forward_5d")
    assert ic.height > 0, "极端值下应仍能计算 IC"
    assert np.all(np.isfinite(ic["ic"].to_numpy())), "IC 值应为有限数"
    print("  [PASS] 极端值处理正确")


def test_metrics_short_series():
    """测试短净值序列"""
    # 仅 5 天
    eq = pd.Series(
        [1e6, 1.01e6, 0.99e6, 1.02e6, 1.03e6],
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    metrics = calc_enhanced_metrics(eq)
    assert len(metrics) > 0, "短序列应仍能计算指标"
    assert "sharpe_ratio" in metrics
    print(f"  [PASS] 短序列指标计算: {len(metrics)} 个指标")


def test_metrics_empty():
    """测试空净值序列"""
    metrics = calc_enhanced_metrics(pd.Series([], dtype=float))
    assert metrics == {}, "空序列应返回空字典"
    print("  [PASS] 空净值序列处理正确")


def test_metrics_flat():
    """测试无波动的净值序列（全相等）"""
    eq = pd.Series(
        [1e6] * 100,
        index=pd.bdate_range("2024-01-01", periods=100),
    )
    metrics = calc_enhanced_metrics(eq)
    assert metrics["volatility"] == 0.0 or abs(metrics["volatility"]) < 1e-10, "无波动时 volatility 应为 0"
    assert metrics["max_drawdown"] == 0.0, "无波动时 max_drawdown 应为 0"
    print("  [PASS] 无波动序列处理正确")


def test_metrics_single_day():
    """测试单日净值"""
    eq = pd.Series([1e6], index=pd.bdate_range("2024-01-01", periods=1))
    metrics = calc_enhanced_metrics(eq)
    assert metrics == {}, "单日应返回空字典"
    print("  [PASS] 单日净值处理正确")


def run_all_boundary_tests():
    """运行所有边界条件测试"""
    print("=" * 60)
    print("边界条件测试")
    print("=" * 60)
    tests = [
        ("空数据", test_empty_data),
        ("单行数据", test_single_row),
        ("全 NaN", test_all_nan),
        ("部分缺失", test_partial_nan),
        ("常数因子", test_constant_factor),
        ("单一行业", test_single_industry),
        ("样本不足", test_few_samples),
        ("极端值", test_extreme_values),
        ("短净值序列", test_metrics_short_series),
        ("空净值序列", test_metrics_empty),
        ("无波动序列", test_metrics_flat),
        ("单日净值", test_metrics_single_day),
    ]
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[测试] {name}")
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"边界条件测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return passed, failed


if __name__ == "__main__":
    run_all_boundary_tests()
