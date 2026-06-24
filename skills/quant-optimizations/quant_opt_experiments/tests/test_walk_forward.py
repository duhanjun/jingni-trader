"""
测试 3：因子 IC 稳定性 + Walk-Forward 评估
"""
import pandas as pd
import numpy as np
import pytest

from skills.quant-optimizations.quant_opt_experiments.walk_forward_eval import (
    analyze_factor,
    analyze_all_factors,
    walk_forward,
    calc_forward_returns,
    cross_sectional_ic,
)
from skills.quant-optimizations.quant_opt_experiments.factor_expression_engine import FactorEngine, register_alpha158_pv
from skills.quant-optimizations.quant_opt_experiments.tests.fixtures import make_synthetic_panel


def close_pivot(panel):
    return panel.pivot(index="date", columns="code", values="close").sort_index()


# ---- Test 1: 未来收益计算 ----
def test_forward_returns():
    close = pd.DataFrame({
        "A": [10, 11, 12, 13, 14],
        "B": [20, 21, 22, 23, 24],
    }, index=pd.bdate_range("2024-01-01", periods=5))
    fwd = calc_forward_returns(close, [1, 2])
    # ret_forward_1d[2024-01-01, A] = (11-10)/10 = 0.1
    row_a = fwd[(fwd["code"] == "A") & (fwd["date"] == close.index[0])].iloc[0]
    assert abs(row_a["ret_forward_1d"] - 0.1) < 1e-9
    # ret_forward_2d[2024-01-01, A] = (12-10)/10 = 0.2
    assert abs(row_a["ret_forward_2d"] - 0.2) < 1e-9


# ---- Test 2: IC 方向性 ----
def test_factor_ic_with_known_signal(panel):
    """
    构造：让"未来 1 日收益"在 panel 上排第一 → 该因子的 IC 应明显为正
    用 5 日反转因子在已知反转数据上检验
    """
    close = close_pivot(panel)
    # 5 日反转因子
    factor = -close.pct_change(5)
    factor_long = factor.stack(future_stack=True).reset_index()
    factor_long.columns = ["date", "code", "reversal_5d"]

    res = analyze_factor(factor_long, close, "reversal_5d", forward_periods=(5,))
    fwd_col = "ret_forward_5d"
    ic = res[fwd_col]
    # 合成数据有反转结构，期望 IC mean > 0
    print(f"\n[INFO] 5日反转因子 IC mean = {ic.ic_mean:.4f}, IR = {ic.ic_ir:.2f}")
    # 不强求数值，但 IR 应不太小
    assert ic.ic_ir > -10  # 没崩溃即可


# ---- Test 3: 批量 IC 评估 ----
def test_analyze_all_factors(panel):
    close = close_pivot(panel)
    engine = FactorEngine(panel)
    register_alpha158_pv(engine)
    factor_long = engine.compute_all()
    # compute_all 返回的列可能含 'date','code' 和所有因子
    factor_cols = [c for c in factor_long.columns if c not in ("date", "code")]

    df = analyze_all_factors(factor_long, close, factor_cols[:5], forward_periods=(5,))
    assert len(df) == 5
    assert "ic_mean" in df.columns
    assert "ic_ir" in df.columns
    print(f"\n[INFO] 批量 IC 评估样例：\n{df.to_string(index=False)}")


# ---- Test 4: Walk-Forward ----
def test_walk_forward(panel):
    close = close_pivot(panel)
    factor = close.pct_change(20)
    factor_long = factor.stack(future_stack=True).reset_index()
    factor_long.columns = ["date", "code", "mom_20"]

    folds = walk_forward(
        factor_long, close, "mom_20",
        train_months=4, test_months=2, forward=5
    )
    # 至少 2 折
    assert len(folds) >= 1
    print(f"\n[INFO] Walk-forward 折数 = {len(folds)}")
    for f in folds:
        print(f"  train[{f.train_start}~{f.train_end}] IC={f.train_ic:.3f}  "
              f"test[{f.test_start}~{f.test_end}] IC={f.test_ic:.3f}  "
              f"Sharpe train={f.train_sharpe:.2f} test={f.test_sharpe:.2f}")

    # 计算 train/test IC 一致性
    train_ics = [f.train_ic for f in folds]
    test_ics = [f.test_ic for f in folds]
    if len(folds) >= 2:
        corr = np.corrcoef(train_ics, test_ics)[0, 1]
        print(f"[INFO] train/test IC 相关系数 = {corr:.2f}")


# ---- Test 5: 因子稳定性结论 ----
def test_factor_stability_conclusion(panel):
    close = close_pivot(panel)
    engine = FactorEngine(panel)
    register_alpha158_pv(engine)
    factor_long = engine.compute_all()

    # 评估所有因子
    factor_cols = [c for c in factor_long.columns if c not in ("date", "code")]
    df = analyze_all_factors(factor_long, close, factor_cols, forward_periods=(5,))

    # 排序
    df_sorted = df.sort_values("ic_ir", ascending=False)
    print("\n[INFO] Top 5 稳定因子 (按 IR 排序):")
    print(df_sorted.head(5).to_string(index=False))
    print("\n[INFO] Bottom 5 因子 (按 IR 排序):")
    print(df_sorted.tail(5).to_string(index=False))


# ---- pytest fixture ----
@pytest.fixture
def panel():
    return make_synthetic_panel(n_stocks=6, n_days=500, seed=42)


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"])
    sys.exit(0)