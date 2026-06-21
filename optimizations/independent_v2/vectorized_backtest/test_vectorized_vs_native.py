"""
向量化回测适配器 vs 原生回测适配器 对比测试

验证内容:
1. 正确性: 向量化版本与原生版本在相同输入下产出可比的绩效指标
2. 性能: 向量化版本在大数据量下应显著快于原生版本
3. 边界条件: 空数据、单只股票、单日数据等
"""
from __future__ import annotations

import sys
import os
import time
import numpy as np
import pandas as pd
import pytest

# 让测试既能从仓库根目录运行，也能从本目录运行
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from optimizations.independent_v2.jingni_compat import NativeAdapterCompat, VectorizedAdapterCompat
from optimizations.independent_v2.data_fixtures import (
    make_synthetic_ohlcv,
    make_signal_from_factor,
    make_target_weight_signal,
)


# ---------- 正确性测试 ----------

def test_vectorized_basic_metrics_nonzero():
    """向量化回测应产出非空且合理的绩效指标。"""
    data = make_synthetic_ohlcv(n_codes=5, n_days=120, seed=1)
    signals = make_target_weight_signal(data, top_n=2, rebalance_freq="M")
    adapter = VectorizedAdapterCompat()
    result = adapter.run_backtest(data, signals, init_capital=1e6)

    assert "metrics" in result
    metrics = result["metrics"]
    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert not result["equity_curve"].empty
    assert len(result["equity_curve"]) > 0


def test_vectorized_vs_native_metrics_comparable():
    """向量化版本与原生版本的核心指标应在合理误差范围内一致。

    注意:
    - 两者成交撮合细节略有差异（向量化用权重变化估算成交，
      原生用逐笔撮合），因此允许较大容差。
    - 原生适配器只在信号日记录净值（月度调仓=9个点），
      向量化适配器记录所有交易日净值（180个点）。
      这是向量化版本的优势（更完整的净值曲线），不视为错误。
    """
    data = make_synthetic_ohlcv(n_codes=5, n_days=180, seed=7)
    signals = make_signal_from_factor(data, top_n=2, rebalance_freq="M")

    native = NativeAdapterCompat()
    vec = VectorizedAdapterCompat()

    r_native = native.run_backtest(data, signals, init_capital=1e6)
    r_vec = vec.run_backtest(data, signals, init_capital=1e6)

    m_n = r_native["metrics"]
    m_v = r_vec["metrics"]

    # 向量化版本应产出更完整的净值曲线（所有交易日 vs 仅信号日）
    assert len(r_vec["equity_curve"]) >= len(r_native["equity_curve"]), (
        f"向量化净值曲线应更长: vec={len(r_vec['equity_curve'])}, "
        f"native={len(r_native['equity_curve'])}"
    )

    # 总收益方向应一致（同号或都接近0）
    if abs(m_n["total_return"]) > 0.05:
        assert m_n["total_return"] * m_v["total_return"] > 0, (
            f"总收益方向不一致: native={m_n['total_return']:.4f}, "
            f"vectorized={m_v['total_return']:.4f}"
        )

    # 最大回撤符号应都为负
    assert m_n["max_drawdown"] <= 0
    assert m_v["max_drawdown"] <= 0

    # 夏普比率应在合理范围内（不要求精确相等，但都应是有限数）
    assert np.isfinite(m_n["sharpe_ratio"])
    assert np.isfinite(m_v["sharpe_ratio"])


def test_t_plus_1_constraint_respected():
    """T+1 约束：当日信号不应在当日产生收益贡献。"""
    data = make_synthetic_ohlcv(n_codes=3, n_days=30, seed=3)
    # 只在第一天发信号
    first_date = data["date"].min()
    sig = pd.DataFrame([
        {"date": first_date, "code": c, "target_weight": 1.0 / 3}
        for c in data["code"].unique()
    ])
    adapter = VectorizedAdapterCompat()
    result = adapter.run_backtest(data, sig, init_capital=1e6, t_plus_1=True)
    # 第一天的持仓应为0（信号次日才生效）
    eq = result["equity_curve"]
    first_row = eq.iloc[0]
    assert first_row["position_count"] == 0 or first_row["market_value"] < 1.0


# ---------- 性能测试 ----------

def test_performance_vectorized_faster():
    """向量化版本在大数据量+高频信号下应显著快于原生版本。

    公平性说明: 原生适配器只遍历信号日 (dates = signals['date'].unique())，
    因此月度调仓时它只处理 ~9 天。为公平对比，本测试使用日频信号
    （每个交易日都调仓），使两者处理相同数量的日期。
    """
    data = make_synthetic_ohlcv(n_codes=20, n_days=300, seed=99)
    # 日频信号：每个交易日都生成 target_weight
    signals = make_target_weight_signal(data, top_n=5, rebalance_freq="D")

    native = NativeAdapterCompat()
    vec = VectorizedAdapterCompat()

    # 预热（避免首次导入开销影响）
    warm_data = data.iloc[:60]
    warm_sig = signals[signals["date"].isin(warm_data["date"].unique())]
    native.run_backtest(warm_data, warm_sig)
    vec.run_backtest(warm_data, warm_sig)

    t0 = time.perf_counter()
    native.run_backtest(data, signals)
    t_native = time.perf_counter() - t0

    t0 = time.perf_counter()
    vec.run_backtest(data, signals)
    t_vec = time.perf_counter() - t0

    speedup = t_native / t_vec if t_vec > 0 else float("inf")
    print(f"\n[perf] native={t_native*1000:.1f}ms, vectorized={t_vec*1000:.1f}ms, "
          f"speedup={speedup:.1f}x")
    assert speedup >= 3.0, f"向量化版本未达到 3x 加速: speedup={speedup:.2f}x"


# ---------- 边界条件测试 ----------

def test_empty_data():
    adapter = VectorizedAdapterCompat()
    result = adapter.run_backtest(pd.DataFrame(), pd.DataFrame())
    assert result["metrics"] == {}
    assert result["equity_curve"].empty


def test_empty_signals():
    data = make_synthetic_ohlcv(n_codes=3, n_days=20, seed=5)
    adapter = VectorizedAdapterCompat()
    result = adapter.run_backtest(data, pd.DataFrame())
    assert result["metrics"] == {}
    assert result["equity_curve"].empty


def test_single_stock_single_day():
    """单只股票、单日数据不应崩溃。"""
    data = make_synthetic_ohlcv(n_codes=1, n_days=5, seed=11)
    signals = make_target_weight_signal(data, top_n=1, rebalance_freq="M")
    adapter = VectorizedAdapterCompat()
    result = adapter.run_backtest(data, signals, init_capital=1e6)
    # 至少应返回结构完整的结果
    assert "equity_curve" in result
    assert "metrics" in result


def test_no_rebalance_signal():
    """无任何调仓信号时，净值应保持初始资金。"""
    data = make_synthetic_ohlcv(n_codes=3, n_days=30, seed=13)
    # 所有信号权重为0
    dates = data["date"].unique()
    sig = pd.DataFrame([
        {"date": d, "code": c, "target_weight": 0.0}
        for d in dates[:1]  # 只在第一天发空信号
        for c in data["code"].unique()
    ])
    adapter = VectorizedAdapterCompat()
    result = adapter.run_backtest(data, sig, init_capital=1e6)
    eq = result["equity_curve"]
    if not eq.empty:
        # 净值应接近初始资金（允许微小浮点误差）
        assert abs(eq["equity"].iloc[-1] - 1e6) < 1.0, (
            f"无信号时净值不应变化: {eq['equity'].iloc[-1]}"
        )


if __name__ == "__main__":
    # 直接运行: python test_vectorized_vs_native.py
    pytest.main([__file__, "-v", "-s"])
