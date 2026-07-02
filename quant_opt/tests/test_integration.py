"""
端到端集成测试：串联 表达式引擎 + 因子注册 + 向量化回测

模拟 jingni-trader 工作流：因子 → 选股信号 → PortfolioWeight → 回测
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_opt.expression_engine import (
    Evaluator,
    F,
    Rank,
    Ref,
    Zscore,
)
from quant_opt.factor_registry import REGISTRY
from quant_opt.tests._fixtures import make_synthetic_a_share_data
from quant_opt.vectorized_backtest import (
    signals_to_weights,
    vectorized_backtest,
)


def test_end_to_end_factor_signal_backtest():
    """完整流程：表达式因子 → 排序 → 信号 → 权重 → 回测"""
    # 1) 准备数据
    data = make_synthetic_a_share_data(n_stocks=30, n_days=120, seed=2024)

    # 2) 用表达式引擎构造因子：20日反转
    expr = -(F("close") / Ref(F("close"), 20) - 1)
    expr.name = "reversal_20d"
    ev = Evaluator(data)
    factor_values = ev.eval(expr)

    # 3) 把因子值转成 0/1 信号：取前 20% 多头
    factor_df = pd.DataFrame({
        "date": data["date"].values,
        "code": data["code"].values,
        "signal": factor_values.values,
    })
    # top 20% 多头
    factor_df["signal"] = (
        factor_df.groupby("date")["signal"]
        .transform(lambda s: s.rank(pct=True, method="first") <= 0.2)
        .astype(int)
    )

    # 4) 生成权重
    pw = signals_to_weights(factor_df, top_quantile=0.2, long_only=True)
    assert pw.weight_frame.shape == (120, 30)

    # 5) 回测
    result = vectorized_backtest(
        price_df=data, weights=pw, init_capital=1_000_000.0
    )
    m = result["metrics"]
    print("\n[集成测试] 20日反转因子回测结果:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    assert "total_return" in m
    assert "sharpe_ratio" in m
    assert "max_drawdown" in m
    # 净值首日应等于初始资金
    assert abs(result["equity_curve"]["equity"].iloc[0] - 1_000_000.0) < 1.0


def test_registry_evaluate_via_expression_engine():
    """验证因子注册表与表达式引擎互通"""
    data = make_synthetic_a_share_data(n_stocks=20, n_days=60, seed=2024)

    # 1) 因子注册表批量计算
    out = REGISTRY.compute_many(
        ["ret_1d", "ret_5d", "volatility_20d", "expr_reversal_20d"],
        data,
        shared_evaluator=True,
    )

    # 2) 与表达式引擎直接计算对比
    expr = -(F("close") / Ref(F("close"), 20) - 1)
    direct = Evaluator(data).eval(expr)
    np.testing.assert_array_equal(
        out["expr_reversal_20d"].fillna(-999).values,
        direct.fillna(-999).values,
        "expr_reversal_20d 在注册表与表达式引擎下应一致",
    )


def test_performance_summary():
    """性能汇总：收集所有模块的耗时"""
    data = make_synthetic_a_share_data(n_stocks=100, n_days=500, seed=2024)
    results = {}

    # 表达式引擎
    from quant_opt.expression_engine import TsStd
    t0 = time.perf_counter()
    ev = Evaluator(data)
    out = ev.eval_many({
        "ret_5d": F("close") / Ref(F("close"), 5) - 1,
        "ma_20_diff": F("close") / Ref(F("close"), 20) - 1,
        "vol_20": TsStd(F("close") / Ref(F("close"), 1) - 1, 20),
    })
    results["expression_engine_eval_many (3 factors)"] = time.perf_counter() - t0

    # 因子注册表
    t0 = time.perf_counter()
    out2 = REGISTRY.compute_many(
        ["ret_1d", "ret_5d", "ret_20d", "volatility_20d", "ma_deviation_20"],
        data,
        shared_evaluator=True,
    )
    results["factor_registry_compute_many (5 factors)"] = time.perf_counter() - t0

    # 向量化回测
    from quant_opt.tests._fixtures import make_signals
    sig = make_signals(data, n_dates=500, top_quantile=0.1)
    pw = signals_to_weights(sig, top_quantile=0.1)
    t0 = time.perf_counter()
    result = vectorized_backtest(price_df=data, weights=pw, init_capital=1_000_000.0)
    results["vectorized_backtest (100×500)"] = time.perf_counter() - t0

    print("\n[性能汇总]")
    for k, v in results.items():
        print(f"  {k}: {v*1000:.1f} ms")

    # 向量化回测应至少在 1s 内完成
    assert results["vectorized_backtest (100×500)"] < 5.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
