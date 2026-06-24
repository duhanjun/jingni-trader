"""
jingni-trader 优化验证测试

验证内容：
1. 正确性测试：向量化回测适配器 vs main 分支原生适配器（结果等价性）
2. 性能对比测试：向量化 vs 原生（耗时对比）
3. 边界条件测试：空数据、单股票、全涨跌停、T+1
4. 向量化 IC 正确性：vs scipy.stats.spearmanr
5. 扩展指标正确性：手算 vs 函数

运行：python -m tests.optimizations.test_optimizations
"""
import os
import sys
import time
import json
import unittest
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# 将项目根目录加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "skills", "backtest-engine"))

from scripts.adapters.native_adapter import NativeAdapter
from scripts.optimizations.vectorized_adapter import VectorizedAdapter
from scripts.optimizations.extended_metrics import ExtendedMetrics
from scripts.optimizations.vectorized_ic import VectorizedIC


# ── 测试数据生成 ─────────────────────────────────────────
def make_market_data(n_codes=50, n_days=120, seed=42):
    """生成合成 A 股日线数据"""
    rng = np.random.RandomState(seed)
    base_dates = pd.bdate_range("2024-01-02", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rows = []
    for code in codes:
        price = 10.0 + rng.randn() * 3
        for dt in base_dates:
            ret = rng.randn() * 0.02
            price = max(price * (1 + ret), 1.0)
            volume = int(rng.randint(100000, 1000000))
            amount = volume * price
            # 涨跌停标记（约 2% 概率）
            is_limit_up = bool(rng.rand() < 0.02)
            is_limit_down = bool(rng.rand() < 0.02)
            rows.append({
                "date": dt, "code": code,
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": volume, "amount": amount,
                "turnover_rate": float(rng.uniform(0.5, 5.0)),
                "is_limit_up": is_limit_up, "is_limit_down": is_limit_down,
            })
    return pd.DataFrame(rows)


def make_signals(data, buy_ratio=0.2, seed=7):
    """生成交易信号（每日在 buy_ratio 比例股票上发出买入信号）"""
    rng = np.random.RandomState(seed)
    dates = sorted(data["date"].unique())
    rows = []
    for dt in dates:
        day_codes = data[data["date"] == dt]["code"].tolist()
        n_buy = max(1, int(len(day_codes) * buy_ratio))
        buy_codes = rng.choice(day_codes, size=n_buy, replace=False)
        for c in day_codes:
            sig = 1 if c in buy_codes else 0
            rows.append({"date": dt, "code": c, "signal": sig})
    # 隔日对买入的股票发卖出信号
    sig_df = pd.DataFrame(rows)
    return sig_df


def make_hold_signals(data, hold_days=20, buy_ratio=0.2, seed=7):
    """生成买入持有信号：首日买入，持有 hold_days 后卖出（避免每日换仓耗尽现金）"""
    rng = np.random.RandomState(seed)
    dates = sorted(data["date"].unique())
    if len(dates) < hold_days + 2:
        hold_days = max(1, len(dates) // 3)
    buy_date = dates[0]
    sell_date = dates[hold_days]
    day_codes = data[data["date"] == buy_date]["code"].tolist()
    n_buy = max(1, int(len(day_codes) * buy_ratio))
    buy_codes = set(rng.choice(day_codes, size=n_buy, replace=False))
    rows = []
    for c in buy_codes:
        rows.append({"date": buy_date, "code": c, "signal": 1})
        rows.append({"date": sell_date, "code": c, "signal": -1})
    return pd.DataFrame(rows)


def make_rebalance_signals(data, buy_ratio=0.2, seed=7):
    """生成每日换仓信号：每日卖出前一日买入的，再买入新的（用于性能压测）"""
    rng = np.random.RandomState(seed)
    dates = sorted(data["date"].unique())
    rows = []
    holdings = set()
    for dt in dates:
        day_codes = data[data["date"] == dt]["code"].tolist()
        for c in list(holdings):
            if c in day_codes:
                rows.append({"date": dt, "code": c, "signal": -1})
                holdings.discard(c)
        n_buy = max(1, int(len(day_codes) * buy_ratio))
        buy_codes = rng.choice(day_codes, size=n_buy, replace=False)
        for c in buy_codes:
            rows.append({"date": dt, "code": c, "signal": 1})
            holdings.add(c)
    return pd.DataFrame(rows)


# ── 测试用例 ─────────────────────────────────────────────
class TestVectorizedBacktest(unittest.TestCase):
    """向量化回测适配器测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = make_market_data(n_codes=50, n_days=120)
        cls.signals = make_hold_signals(cls.data, hold_days=20, buy_ratio=0.2)

    def test_correctness_vs_native(self):
        """正确性：向量化结果与原生适配器等价（买入持有场景，无现金耗尽干扰）"""
        native = NativeAdapter().run_backtest(self.data, self.signals)
        vectorized = VectorizedAdapter().run_backtest(self.data, self.signals)

        # 净值曲线长度一致
        self.assertEqual(len(native["equity_curve"]), len(vectorized["equity_curve"]))

        # 终值应接近（整手取整顺序可能导致微小差异，容差 1%）
        nav_end = native["equity_curve"]["equity"].iloc[-1]
        vec_end = vectorized["equity_curve"]["equity"].iloc[-1]
        rel_diff = abs(nav_end - vec_end) / max(abs(nav_end), 1.0)
        self.assertLess(rel_diff, 0.01, f"终值差异过大: native={nav_end}, vec={vec_end}")

        # 成交笔数应一致
        self.assertEqual(len(native["trades"]), len(vectorized["trades"]),
                         f"成交笔数不一致: native={len(native['trades'])}, vec={len(vectorized['trades'])}")

        # 关键指标接近
        for key in ["total_return", "sharpe_ratio", "max_drawdown"]:
            nv = native["metrics"].get(key, 0)
            vv = vectorized["metrics"].get(key, 0)
            if abs(nv) > 1e-9:
                self.assertLess(abs(nv - vv) / abs(nv), 0.05,
                                f"{key} 差异过大: native={nv}, vec={vv}")
        print(f"[正确性] native终值={nav_end:.2f}, vectorized终值={vec_end:.2f}, 相对差异={rel_diff:.4%}")

    def test_performance(self):
        """性能对比：向量化应快于原生"""
        # 用更大数据量 + 每日换仓信号（交易量大，体现向量化优势）
        big_data = make_market_data(n_codes=200, n_days=200, seed=99)
        big_signals = make_rebalance_signals(big_data, buy_ratio=0.2)

        t0 = time.time()
        native_res = NativeAdapter().run_backtest(big_data, big_signals)
        t_native = time.time() - t0

        t0 = time.time()
        vec_res = VectorizedAdapter().run_backtest(big_data, big_signals)
        t_vec = time.time() - t0

        speedup = t_native / t_vec if t_vec > 0 else float("inf")
        print(f"[性能] 数据规模=200股×200天")
        print(f"  原生适配器: {t_native:.3f}s")
        print(f"  向量化适配器: {t_vec:.3f}s")
        print(f"  加速比: {speedup:.2f}x")
        # 向量化应不慢于原生
        self.assertGreater(speedup, 1.0, "向量化应快于原生")
        # 记录用于报告
        self.perf_result = {
            "n_codes": 200, "n_days": 200,
            "native_seconds": round(t_native, 4),
            "vectorized_seconds": round(t_vec, 4),
            "speedup": round(speedup, 2),
        }

    def test_edge_empty_data(self):
        """边界：空数据"""
        res = VectorizedAdapter().run_backtest(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(res["metrics"], {})
        self.assertTrue(res["equity_curve"].empty)

    def test_edge_single_stock(self):
        """边界：单股票"""
        data = make_market_data(n_codes=1, n_days=30, seed=1)
        sig = make_hold_signals(data, hold_days=10, buy_ratio=1.0)
        res = VectorizedAdapter().run_backtest(data, sig)
        self.assertFalse(res["equity_curve"].empty)
        self.assertGreater(res["equity_curve"]["equity"].iloc[-1], 0)

    def test_edge_all_limit_up(self):
        """边界：全部涨停无法买入"""
        data = make_market_data(n_codes=5, n_days=5, seed=2)
        data["is_limit_up"] = True  # 全涨停
        sig = pd.DataFrame([
            {"date": data["date"].iloc[0], "code": c, "signal": 1}
            for c in data["code"].unique()
        ])
        res = VectorizedAdapter().run_backtest(data, sig, price_limit=True)
        # 无成交，权益≈初始资金
        self.assertEqual(len(res["trades"]), 0)
        self.assertAlmostEqual(res["equity_curve"]["equity"].iloc[-1], 1e6, places=0)

    def test_t_plus_1(self):
        """边界：T+1 当日买入不可卖出"""
        data = make_market_data(n_codes=3, n_days=3, seed=3)
        code = data["code"].unique()[0]
        # 同日对同一股票先买后卖
        sig = pd.DataFrame([
            {"date": data["date"].iloc[0], "code": code, "signal": 1},
            {"date": data["date"].iloc[0], "code": code, "signal": -1},
        ])
        res = VectorizedAdapter().run_backtest(data, sig, t_plus_1=True)
        # T+1 下当日买入不可卖，应有买入无卖出
        actions = res["trades"]["action"].tolist() if not res["trades"].empty else []
        self.assertIn("buy", actions)
        self.assertNotIn("sell", actions)


class TestVectorizedIC(unittest.TestCase):
    """向量化 IC 分析测试"""

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(123)
        n_codes, n_days = 100, 60
        dates = pd.bdate_range("2024-01-02", periods=n_days)
        codes = [f"{i:06d}.SH" for i in range(n_codes)]
        rows = []
        for c in codes:
            factor_vals = rng.randn(n_days)
            # 构造与远期收益正相关的关系 + 噪声
            fwd = 0.3 * factor_vals + rng.randn(n_days) * 0.5
            for j, dt in enumerate(dates):
                rows.append({
                    "date": dt, "code": c,
                    "f1": factor_vals[j],
                    "f2": rng.randn(),  # 无效因子
                    "ret_forward_5d": fwd[j],
                })
        cls.data = pd.DataFrame(rows)

    def test_ic_vs_scipy(self):
        """正确性：向量化 IC 与 scipy.spearmanr 一致"""
        ic_vec = VectorizedIC.calc_ic_series(self.data, "f1", "ret_forward_5d")
        self.assertIsNotNone(ic_vec)
        self.assertGreater(len(ic_vec), 0)

        # 用 scipy 逐日计算对比
        ic_scipy = []
        for dt, g in self.data.groupby("date"):
            if len(g) < 10:
                continue
            r, _ = stats.spearmanr(g["f1"], g["ret_forward_5d"])
            ic_scipy.append({"date": dt, "ic": r})
        ic_scipy_s = pd.DataFrame(ic_scipy).set_index("date")["ic"]

        # 对齐比较
        common = ic_vec.index.intersection(ic_scipy_s.index)
        diff = (ic_vec.loc[common] - ic_scipy_s.loc[common]).abs().max()
        print(f"[IC正确性] 向量化 vs scipy 最大差异: {diff:.6f}")
        self.assertLess(diff, 1e-6, "向量化 IC 应与 scipy 一致")

    def test_effective_factor(self):
        """有效因子 f1 的 IC_IR 应高于无效因子 f2"""
        stat_f1 = VectorizedIC.summarize_ic(
            VectorizedIC.calc_ic_series(self.data, "f1", "ret_forward_5d"))
        stat_f2 = VectorizedIC.summarize_ic(
            VectorizedIC.calc_ic_series(self.data, "f2", "ret_forward_5d"))
        print(f"[IC有效性] f1 IC_IR={stat_f1['ic_ir']}, f2 IC_IR={stat_f2['ic_ir']}")
        self.assertGreater(abs(stat_f1["ic_ir"]), abs(stat_f2["ic_ir"]))


class TestExtendedMetrics(unittest.TestCase):
    """扩展绩效指标测试"""

    def test_profit_factor(self):
        """利润因子 = 总盈利 / 总亏损"""
        returns = pd.Series([0.1, -0.05, 0.08, -0.02])
        pf = ExtendedMetrics.calc_profit_factor(returns)
        expected = (0.1 + 0.08) / (0.05 + 0.02)
        self.assertAlmostEqual(pf, expected, places=6)

    def test_payoff_ratio(self):
        """平均盈亏比"""
        returns = pd.Series([0.1, -0.05, 0.08, -0.02])
        pr = ExtendedMetrics.calc_payoff_ratio(returns)
        expected = (0.09) / (0.035)  # avg_win=0.09, avg_loss=0.035
        self.assertAlmostEqual(pr, expected, places=6)

    def test_max_consecutive_loss(self):
        """最大连续亏损天数"""
        returns = pd.Series([0.01, -0.01, -0.02, -0.03, 0.01, -0.01, -0.02])
        mcl = ExtendedMetrics.calc_max_consecutive_loss_days(returns)
        self.assertEqual(mcl, 3)

    def test_alpha_beta(self):
        """Alpha/Beta 计算"""
        rng = np.random.RandomState(0)
        bench = pd.Series(rng.randn(100) * 0.01)
        # 构造 beta=1.2 的组合
        port = 1.2 * bench + rng.randn(100) * 0.005
        ab = ExtendedMetrics.calc_alpha_beta(port, bench)
        self.assertAlmostEqual(ab["beta"], 1.2, delta=0.1)
        print(f"[Alpha/Beta] beta={ab['beta']:.4f}, alpha={ab['alpha']:.4f}")

    def test_information_ratio(self):
        """信息比率应为有限值"""
        rng = np.random.RandomState(1)
        port = pd.Series(rng.randn(100) * 0.01 + 0.001)
        bench = pd.Series(rng.randn(100) * 0.01)
        ir = ExtendedMetrics.calc_information_ratio(port, bench)
        self.assertTrue(np.isfinite(ir))
        print(f"[信息比率] IR={ir:.4f}")


def run_all_and_collect():
    """运行全部测试并收集结果用于报告"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_all_and_collect()
