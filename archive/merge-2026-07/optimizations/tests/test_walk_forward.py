"""
Walk-Forward 滚动验证测试

测试内容：
1. 窗口生成正确性
2. 滚动窗口 vs 扩展窗口
3. 完整 walk-forward 验证流程
4. 过拟合概率计算
5. 边界条件
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optimizations.walk_forward_v2 import (
    generate_walk_forward_windows,
    walk_forward_validate,
)


def test_window_generation():
    """测试 1：窗口生成"""
    print("\n=== 测试 1: 窗口生成 ===")
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    windows = generate_walk_forward_windows(dates, train_months=12, test_months=3)

    print(f"  日期范围: {dates[0].date()} ~ {dates[-1].date()}")
    print(f"  生成窗口数: {len(windows)}")
    assert len(windows) > 0, "应生成至少 1 个窗口"

    for i, (train, test) in enumerate(windows[:3]):
        print(f"  窗口 {i}: train {train[0].date()}~{train[-1].date()}, test {test[0].date()}~{test[-1].date()}")
        # 测试期应在训练期之后
        assert test[0] > train[-1], "测试期应在训练期之后"
        # 训练期和测试期不应重叠
        assert not train.equals(test), "训练期和测试期不应相同"
    print("  ✓ 窗口生成正确")


def test_expand_window():
    """测试 2：扩展窗口模式"""
    print("\n=== 测试 2: 扩展窗口 ===")
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    windows_rolling = generate_walk_forward_windows(dates, train_months=12, test_months=6, expand=False)
    windows_expand = generate_walk_forward_windows(dates, train_months=12, test_months=6, expand=True)

    assert len(windows_rolling) > 0
    assert len(windows_expand) > 0

    # 扩展窗口：训练期起点应相同
    if len(windows_expand) >= 2:
        assert windows_expand[0][0][0] == windows_expand[1][0][0], "扩展窗口训练期起点应相同"
        # 滚动窗口：训练期起点应不同
        assert windows_rolling[0][0][0] != windows_rolling[1][0][0], "滚动窗口训练期起点应不同"
    print("  ✓ 扩展窗口 vs 滚动窗口正确")


def test_embargo():
    """测试 3：隔离期"""
    print("\n=== 测试 3: 隔离期 ===")
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    windows = generate_walk_forward_windows(dates, train_months=12, test_months=3, embargo_days=5)

    for train, test in windows:
        gap = (test[0] - train[-1]).days
        assert gap >= 5, f"隔离期不足: {gap} 天"
    print(f"  隔离期 5 天生效，共 {len(windows)} 个窗口")
    print("  ✓ 隔离期正确")


def test_walk_forward_validate():
    """测试 4：完整 walk-forward 验证"""
    print("\n=== 测试 4: Walk-Forward 验证 ===")
    np.random.seed(42)
    n_dates = 500
    n_stocks = 30
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    codes = [f"c{i:04d}" for i in range(n_stocks)]

    # 生成行情数据
    data_rows = []
    for c in codes:
        price = 10.0
        for d in dates:
            price *= (1 + np.random.randn() * 0.02)
            data_rows.append({
                "date": d, "code": c, "close": price,
                "is_limit_up": False, "is_limit_down": False,
                "volume": 1000, "change_pct": 0,
            })
    data = pd.DataFrame(data_rows)

    # 生成因子数据
    factor_rows = []
    for c in codes:
        for d in dates:
            factor_rows.append({
                "date": d, "code": c,
                "alpha_score": np.random.randn(),
            })
    factor_df = pd.DataFrame(factor_rows)

    param_grid = {"quantile": [0.7, 0.8, 0.9], "holding_days": [5]}

    result = walk_forward_validate(
        data, factor_df, param_grid,
        train_months=12, test_months=3,
    )

    print(f"  窗口数: {len(result['windows'])}")
    print(f"  样本外收益点数: {len(result['oos_returns'])}")
    if result["oos_metrics"]:
        print(f"  样本外 Sharpe: {result['oos_metrics'].get('sharpe_ratio', 'N/A')}")
    if result["is_oos_consistency"]:
        print(f"  样本内 Sharpe 均值: {result['is_oos_consistency'].get('is_sharpe_mean', 'N/A')}")
        print(f"  样本外 Sharpe 均值: {result['is_oos_consistency'].get('oos_sharpe_mean', 'N/A')}")
    print(f"  过拟合概率: {result['overfitting_probability']}")

    assert len(result["windows"]) > 0, "应有窗口结果"
    print("  ✓ Walk-Forward 验证功能正常")


def test_edge_cases():
    """测试 5：边界条件"""
    print("\n=== 测试 5: 边界条件 ===")

    # 数据不足
    short_dates = pd.bdate_range("2024-01-01", periods=30)
    windows = generate_walk_forward_windows(short_dates, train_months=12, test_months=3)
    assert len(windows) == 0, "数据不足应返回空"
    print("  ✓ 数据不足正确处理")

    # 空数据
    windows = generate_walk_forward_windows(pd.DatetimeIndex([]), 12, 3)
    assert len(windows) == 0
    print("  ✓ 空数据正确处理")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_window_generation()
    test_expand_window()
    test_embargo()
    test_walk_forward_validate()
    test_edge_cases()
    print("\n🎉 全部 Walk-Forward 测试通过")
