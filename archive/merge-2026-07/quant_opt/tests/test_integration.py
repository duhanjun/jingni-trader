"""
集成测试: end-to-end 验证 4 个优化模块之间的协作
====================================================

模拟 jingni-trader 完整流程中可能用到的组合场景:
1. 用 factor_dsl 计算反转因子
2. 用 factor_tearsheet 做完整 IC/turnover 分析
3. 用 walk_forward 做 ML 滚动训练 (使用因子作为特征)
4. 用 benchmarks 计算回测 vs 基准的相对指标
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests._synth_data import make_synth_panel, make_synth_factor, make_synth_equity

from factor_dsl.evaluator import eval_preset
from factor_tearsheet.tearsheet import create_full_tear_sheet
from walk_forward.validator import (
    WalkForwardConfig,
    MeanReversionSignal,
    run_walk_forward_validation,
)
from benchmarks.relative_metrics import relative_metrics


@pytest.fixture(scope="module")
def panel():
    return make_synth_panel(n_codes=12, n_days=300)


def test_end_to_end_dsl_to_tearsheet(panel):
    """
    1) 用 DSL 计算 5 日反转因子
    2) 送入 tearsheet 跑完整分析
    3) 验证 IC 报告 + turnover
    """
    factor = eval_preset("reversal_5d", panel)
    factor_df = factor.reset_index()
    factor_df.columns = ["code", "date", "reversal_5d"]

    ts = create_full_tear_sheet(factor_df, panel, quantiles=5, periods=(1, 5))
    assert ts["clean_factor_summary"]["rows"] > 0
    assert "ret_forward_1D" in ts["ic_summary"]
    # turn-over 应在 [0, 1]
    if not ts["turnover"].empty:
        assert ts["turnover"]["top_turnover"].dropna().between(0, 1).all()


def test_end_to_end_dsl_to_walk_forward(panel):
    """
    1) 用 DSL 计算反转因子作为特征
    2) 构造目标: 下日收益
    3) walk-forward 验证
    """
    df = panel.sort_values(["code", "date"]).copy()
    factor = eval_preset("reversal_5d", panel).reset_index()
    factor.columns = ["code", "date", "reversal_5d"]
    df = df.merge(factor, on=["code", "date"], how="left")
    df["y"] = df.groupby("code")["close"].pct_change().shift(-1)
    df = df.dropna(subset=["reversal_5d", "y"])

    X = df[["close", "volume", "reversal_5d"]].reset_index(drop=True)
    y = df["y"].reset_index(drop=True)
    cfg = WalkForwardConfig(train_window=120, test_window=20, rolling_step=20)
    res = run_walk_forward_validation(X, y, MeanReversionSignal, cfg, threshold=0.0)

    # 至少生成一个 fold
    assert res["folds"], "no folds produced"
    assert res["oos_signals"].size > 0


def test_end_to_end_tearsheet_to_benchmarks(panel):
    """
    1) tearsheet 给出 long-short 收益 spread
    2) 转化为 equity curve
    3) 与基准做相对指标对比
    """
    factor_df = make_synth_factor(panel, signal_strength=0.4)
    ts = create_full_tear_sheet(factor_df, panel, quantiles=5, periods=(1, 5))

    # 构造一个等权 long-short 的 equity curve
    mrq = ts["mean_return_by_quantile"].get("ret_forward_5D")
    assert mrq is not None and not mrq.empty

    # 简化: 取每日 top - bottom 的均值作为每日 PnL
    pnl = (
        mrq.groupby("date")
        .apply(lambda x: x[x["factor_quantile"] == 4]["ret_forward_5D"].mean()
               - x[x["factor_quantile"] == 0]["ret_forward_5D"].mean())
    )
    pnl = pnl.reindex(pd.bdate_range(panel["date"].min(), panel["date"].max())).fillna(0)
    strategy_eq = (1 + pnl / 5).cumprod() * 1_000_000  # 5 日再平衡
    # 同步生成基准
    bench_eq = pd.Series(
        1_000_000 * (1 + np.random.default_rng(0).normal(0.0003, 0.01, len(strategy_eq))).cumprod(),
        index=strategy_eq.index,
    )

    rm = relative_metrics(strategy_eq, bench_eq)
    assert "alpha" in rm
    assert "beta" in rm
    assert "information_ratio" in rm
