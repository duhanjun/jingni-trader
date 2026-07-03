"""
边界条件测试：空数据、单股票、缺失列、极端值等
"""
from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.quant_opt_20260620.factor_engine_polars import (
    PolarsFactorEngine,
    FactorDef,
    compile_factor,
    vectorized_ic_analysis,
)
from experiments.quant_opt_20260620.backtest_vectorized import (
    VectorizedBacktester,
    BacktestConfig,
)
from experiments.quant_opt_20260620.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_folds,
    walk_forward_split,
)
from experiments.quant_opt_20260620.tests.data_gen import make_synthetic_data, make_signals


def test_empty_data():
    """空数据"""
    print("\n[边界] 空数据...")
    engine = PolarsFactorEngine()
    empty = pd.DataFrame(columns=["code", "date", "open", "high", "low", "close", "volume"])
    result = engine.compute(empty)
    assert result.empty, "空数据应返回空 DataFrame"

    bt = VectorizedBacktester()
    result = bt.run(empty, empty)
    assert "metrics" in result
    assert result["metrics"] == {} or len(result["equity_curve"]) == 0

    print("  ✓ 空数据处理通过")


def test_single_stock():
    """单只股票"""
    print("\n[边界] 单只股票...")
    data = make_synthetic_data(n_codes=1, n_days=100, seed=42)
    engine = PolarsFactorEngine()
    result = engine.compute(data)
    assert len(result) == 100, "应返回 100 行"
    assert "rev_5d" in result.columns or "vol_20d" in result.columns

    # 单股票回测
    signals = make_signals(data)
    # 单股票时信号可能很少，确保不崩溃
    bt = VectorizedBacktester()
    result = bt.run(data, signals)
    assert "equity_curve" in result

    print("  ✓ 单只股票处理通过")


def test_missing_columns():
    """缺失必要列"""
    print("\n[边界] 缺失列...")
    engine = PolarsFactorEngine()
    bad = pd.DataFrame({"code": ["1"], "date": [pd.Timestamp("2024-01-01")]})
    try:
        engine.compute(bad)
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "缺少必要列" in str(e)
    print("  ✓ 缺失列处理通过")


def test_missing_factor_field():
    """因子表达式中引用不存在的字段"""
    print("\n[边界] 未知字段...")
    try:
        compile_factor("UnknownField")
        assert False, "应抛出 ValueError"
    except ValueError:
        pass
    print("  ✓ 未知字段处理通过")


def test_extreme_values():
    """极端值：0 价格、NaN、inf"""
    print("\n[边界] 极端值...")
    data = make_synthetic_data(n_codes=5, n_days=50, seed=42)
    # 注入 0 价格
    data.loc[data.index[:5], "close"] = 0
    # 注入 NaN
    data.loc[data.index[5:10], "volume"] = np.nan

    engine = PolarsFactorEngine()
    # 不应崩溃
    result = engine.compute(data)
    assert len(result) == len(data)

    print("  ✓ 极端值处理通过")


def test_walk_forward_insufficient_data():
    """walk-forward 数据不足"""
    print("\n[边界] walk-forward 数据不足...")
    # 只有 10 天数据，远小于训练窗口
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=10))
    config = WalkForwardConfig(train_months=12, test_months=3, min_train_samples=100)
    folds = generate_walk_forward_folds(dates, config)
    assert len(folds) == 0, "数据不足时应返回空折列表"
    print("  ✓ 数据不足处理通过")


def test_walk_forward_min_samples():
    """walk-forward 最小样本数过滤"""
    print("\n[边界] walk-forward 最小样本数...")
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=500))
    # 设置很高的 min_train_samples，应过滤掉所有折
    config = WalkForwardConfig(train_months=12, test_months=3, min_train_samples=10000)
    folds = generate_walk_forward_folds(dates, config)
    assert len(folds) == 0, "min_train_samples 过高时应返回空"
    print("  ✓ 最小样本数过滤通过")


def test_signal_with_no_action():
    """所有信号都是 0（无交易）"""
    print("\n[边界] 无交易信号...")
    data = make_synthetic_data(n_codes=10, n_days=50, seed=42)
    signals = pd.DataFrame({
        "code": data["code"].values,
        "date": data["date"].values,
        "signal": 0,
    })
    bt = VectorizedBacktester()
    result = bt.run(data, signals)
    # 应该有净值曲线但无成交
    assert len(result["equity_curve"]) > 0
    assert len(result["trades"]) == 0
    # 末日净值应等于初始资金
    final = float(result["equity_curve"]["equity"].iloc[-1])
    assert abs(final - 1e6) < 1e-6, f"无交易时净值应等于初始资金, 实际: {final}"
    print("  ✓ 无交易信号处理通过")


def test_all_limit_up():
    """所有股票涨停（无法买入）"""
    print("\n[边界] 全部涨停...")
    data = make_synthetic_data(n_codes=10, n_days=30, seed=42)
    data["is_limit_up"] = True  # 全部涨停
    signals = make_signals(data)
    bt = VectorizedBacktester()
    result = bt.run(data, signals)
    # 应该没有买入成交
    buy_trades = result["trades"][result["trades"]["action"] == "buy"] if not result["trades"].empty else result["trades"]
    assert len(buy_trades) == 0, "涨停时应无法买入"
    print("  ✓ 全部涨停处理通过")


if __name__ == "__main__":
    test_empty_data()
    test_single_stock()
    test_missing_columns()
    test_missing_factor_field()
    test_extreme_values()
    test_walk_forward_insufficient_data()
    test_walk_forward_min_samples()
    test_signal_with_no_action()
    test_all_limit_up()
    print("\n=== 所有边界测试通过 ===")
