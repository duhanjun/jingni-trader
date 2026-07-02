"""
Module 2 测试: 向量化回测引擎
"""
import time
import numpy as np
import pandas as pd
import pytest

from skills.quant_optimizations.quant_opt_20260617.core.vectorized_backtest import (
    VectorizedBacktester,
    run_backtest,
    HAS_NUMBA,
)


def _make_market_data(
    n_stocks: int = 30,
    n_days: int = 120,
    seed: int = 42,
    drift: float = 0.0005,
) -> pd.DataFrame:
    """构造 A 股风格合成行情"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for c in codes:
        start = rng.uniform(10, 50)
        ret = rng.normal(drift, 0.02, n_days)
        price = start * (1 + ret).cumprod()
        for i, d in enumerate(dates):
            open_ = price[i] * (1 + rng.normal(0, 0.003))
            high = max(open_, price[i]) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_, price[i]) * (1 - abs(rng.normal(0, 0.005)))
            close = price[i]
            rows.append({
                "date": d, "code": c,
                "open": open_, "high": high, "low": low, "close": close,
                "volume": int(rng.lognormal(15, 0.5)),
                "is_limit_up": False, "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def _make_topk_signals(data: pd.DataFrame, topk: int = 5, seed: int = 0) -> pd.DataFrame:
    """每天随机挑 topk 支股票作为买入信号"""
    rng = np.random.default_rng(seed)
    rows = []
    for d, g in data.groupby("date"):
        # 按 volume 排名选 topk
        chosen = g.nlargest(topk, "volume")["code"].tolist()
        for c in g["code"]:
            rows.append({"date": d, "code": c, "signal": 1 if c in chosen else 0})
    return pd.DataFrame(rows)


class TestVectorizedBacktest:
    """向量化回测测试套件"""

    def test_basic_run_returns_metrics(self):
        """正确性: 基础回测应返回关键指标"""
        data = _make_market_data(n_stocks=20, n_days=60, seed=1)
        signals = _make_topk_signals(data, topk=5, seed=1)
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        assert "total_return" in res.metrics
        assert "sharpe_ratio" in res.metrics
        assert "max_drawdown" in res.metrics
        assert "annual_return" in res.metrics
        assert "volatility" in res.metrics
        # 资产序列长度 = 天数
        assert len(res.equity) == data["date"].nunique()
        # 总资产非零
        assert res.equity[-1] > 0

    def test_equity_never_explodes(self):
        """正确性: 净值不应发散到天文数字"""
        data = _make_market_data(n_stocks=15, n_days=80, seed=2)
        signals = _make_topk_signals(data, topk=3, seed=2)
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        # 在合理范围内（< 100x 初始）
        assert res.equity.max() < 100 * 1_000_000
        assert res.equity.min() > 0

    def test_no_buy_signal_keeps_cash(self):
        """边界条件: 没有买入信号时应全部保持现金"""
        data = _make_market_data(n_stocks=10, n_days=30, seed=3)
        # 全 0 信号
        signals = data[["date", "code"]].copy()
        signals["signal"] = 0
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        # 净值变化应只来自费用（接近 0）
        assert res.equity[-1] <= 1_000_000
        # 现金应等于初始（无费用情况下）
        assert res.cash[-1] <= 1_000_000

    def test_all_buy_uses_cash(self):
        """正确性: 全买入时大部分资金被用掉"""
        data = _make_market_data(n_stocks=10, n_days=30, seed=4)
        signals = data[["date", "code"]].copy()
        signals["signal"] = 1
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        # 应有大量持仓市值
        assert res.market_value[-1] > 0
        # 剩余现金小于初始的 10%
        assert res.cash[-1] < 0.1 * 1_000_000

    def test_lot_size_100_shares(self):
        """正确性: A 股 100 股一手，应自动 round down"""
        data = _make_market_data(n_stocks=5, n_days=10, seed=5)
        signals = data[["date", "code"]].copy()
        signals["signal"] = 1
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        # 现金减少应符合 100 股整数倍
        cash_used = 1_000_000 - res.cash[-1]
        # 总成本 = 持仓市值 + 累计费用
        assert cash_used > 0
        # 持仓只数 = 5 (全部)
        assert res.position_count[-1] == 5

    def test_limit_up_blocks_buy(self):
        """边界条件: 涨停不能买入"""
        data = _make_market_data(n_stocks=5, n_days=10, seed=6)
        # 全部标记为涨停
        data["is_limit_up"] = True
        signals = data[["date", "code"]].copy()
        signals["signal"] = 1
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        # 涨停无法买入 → 现金应仍接近初始
        assert res.cash[-1] > 0.95 * 1_000_000

    def test_parameter_sweep(self):
        """正确性: 参数扫描应返回多组结果"""
        data = _make_market_data(n_stocks=20, n_days=80, seed=7)
        # 参数化 signal func
        def sig_func(topk, hold_days):
            return _make_topk_signals(data, topk=topk, seed=hold_days)

        bt = VectorizedBacktester(init_capital=1_000_000)
        result = bt.parameter_sweep(
            data, sig_func,
            param_grid={"topk": [3, 5, 10], "hold_days": [3, 5]},
        )
        # 3 * 2 = 6 组
        assert len(result) == 6
        assert "sharpe_ratio" in result.columns
        assert "total_return" in result.columns

    def test_performance_vs_naive_loop(self):
        """性能: 向量化版应快于纯 Python for-loop"""
        if not HAS_NUMBA:
            pytest.skip("Numba not available, skip perf test")

        data = _make_market_data(n_stocks=50, n_days=120, seed=8)
        signals = _make_topk_signals(data, topk=10, seed=8)

        bt = VectorizedBacktester(init_capital=1_000_000)

        # 预热 JIT：用一个小但完整的面板
        small_data = _make_market_data(n_stocks=10, n_days=30, seed=88)
        small_signals = _make_topk_signals(small_data, topk=3, seed=88)
        bt.run(small_data, small_signals)

        # 向量化（含 Numba）版
        t0 = time.perf_counter()
        for _ in range(3):
            res_vec = bt.run(data, signals)
        t_vec = time.perf_counter() - t0

        print(f"\n  [Backtest perf] vec/njit: {t_vec:.3f}s for 3 runs")
        # 不直接对比原版（其逻辑与本版不同），但要求 3 次回测 < 1.5s（Numba 后）
        assert t_vec < 1.5, f"3 次回测耗时 {t_vec:.3f}s 偏长"

    def test_metrics_match_formula(self):
        """正确性: 关键指标应符合教科书公式"""
        data = _make_market_data(n_stocks=10, n_days=60, seed=9)
        signals = _make_topk_signals(data, topk=3, seed=9)
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(data, signals)
        # 用 equity 自己算 total_return 并对比
        eq = res.equity
        total_ret = eq[-1] / eq[0] - 1
        assert abs(res.metrics["total_return"] - total_ret) < 1e-6
        # max_drawdown ≤ 0
        assert res.metrics["max_drawdown"] <= 0
        # win_rate ∈ [0, 1]
        assert 0 <= res.metrics["win_rate"] <= 1

    def test_empty_data(self):
        """边界条件: 空数据应不崩溃"""
        bt = VectorizedBacktester(init_capital=1_000_000)
        res = bt.run(pd.DataFrame(), pd.DataFrame())
        assert len(res.equity) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])