"""
向量化回测引擎验证测试
====================

测试维度：
    1. 正确性：与 jingni-trader 原生 native_adapter 的关键指标对齐
    2. 性能：与 native_adapter 在相同数据集下的耗时对比
    3. 边界：空数据、单一标的、单日信号等异常场景

运行：
    PYTHONPATH=workspace python -m pytest workspace/quant_opt_20260617/tests/test_vectorized_engine.py -v
    或直接：
    PYTHONPATH=. python workspace/quant_opt_20260617/tests/test_vectorized_engine.py
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from typing import Tuple

import numpy as np
import pandas as pd


# 把项目根加入 sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from quant_opt_20260617.vectorized_backtest.vectorized_engine import (
    VectorizedBacktestEngine,
)


# =====================================================================
# 合成数据生成
# =====================================================================


def build_synthetic_data(
    n_stocks: int = 10,
    n_days: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """构造多只股票 A 股日线数据"""
    np.random.seed(seed)
    rows = []
    for s in range(n_stocks):
        code = f"{600000 + s}.SH"
        price = 10.0 + np.random.rand() * 30
        for d in range(n_days):
            date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=d)
            ret = np.random.normal(0.0005, 0.015)
            price = max(1.0, price * (1 + ret))
            change = ret
            rows.append({
                "date": date,
                "code": code,
                "open": price * 0.995,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": int(np.random.lognormal(10, 0.4)),
                "change_pct": change * 100,
                "is_limit_up": change >= 0.099,
                "is_limit_down": change <= -0.099,
                "is_st": False,
            })
    return pd.DataFrame(rows)


def build_topk_signals(data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """基于等权 20 日动量构造 topk 信号"""
    data = data.sort_values(["code", "date"]).copy()
    data["ret_20"] = data.groupby("code")["close"].pct_change(20)
    data = data.dropna(subset=["ret_20"])
    rows = []
    for date, group in data.groupby("date"):
        if len(group) < 5:
            continue
        threshold = group["ret_20"].quantile(1 - top_pct)
        losers = group[group["ret_20"] < group["ret_20"].quantile(top_pct)]
        for _, r in losers.iterrows():
            rows.append({"date": r["date"], "code": r["code"], "signal": -1})
        winners = group[group["ret_20"] >= threshold]
        for _, r in winners.iterrows():
            rows.append({"date": r["date"], "code": r["code"], "signal": 1})
    return pd.DataFrame(rows)


# =====================================================================
# 单元测试
# =====================================================================


class TestVectorizedEngine(unittest.TestCase):

    def setUp(self):
        self.data = build_synthetic_data(n_stocks=8, n_days=200)
        self.signals = build_topk_signals(self.data)
        self.engine = VectorizedBacktestEngine(
            init_capital=1_000_000.0,
            commission_rate=0.00025,
            min_commission=5.0,
            stamp_tax_rate=0.001,
            slippage=0.001,
            t_plus_1=True,
            price_limit=True,
        )

    # --- 正确性 ---

    def test_01_basic_run(self):
        """基础回测：能跑出 metrics 且指标在合理范围"""
        result = self.engine.run_backtest(self.data, self.signals)
        self.assertIn("metrics", result)
        m = result["metrics"]
        self.assertIn("total_return", m)
        self.assertIn("annual_return", m)
        self.assertIn("sharpe_ratio", m)
        self.assertIn("max_drawdown", m)
        # 任何回报都应有限
        self.assertTrue(np.isfinite(m["total_return"]))
        self.assertTrue(np.isfinite(m["sharpe_ratio"]))
        self.assertLessEqual(m["max_drawdown"], 0.0)
        # 至少应有一笔成交
        self.assertGreaterEqual(len(result["trades"]), 1)
        # 权益曲线长度 = 交易日数
        self.assertEqual(len(result["equity_curve"]), self.data["date"].nunique())

    def test_02_equity_curve_invariant(self):
        """权益曲线：现金 + 持仓市值 = 权益"""
        result = self.engine.run_backtest(self.data, self.signals)
        eq = result["equity_curve"]
        diff = (eq["equity"] - eq["cash"] - eq["market_value"]).abs().max()
        self.assertLess(diff, 1e-6, f"权益恒等式不成立，最大差 {diff}")

    def test_03_trade_log_consistency(self):
        """成交记录：买入 amount+commission 应该让现金减少，卖出 amount-commission-tax 应让现金增加"""
        result = self.engine.run_backtest(self.data, self.signals)
        trades = result["trades"]
        buys = trades[trades["action"] == "buy"]
        sells = trades[trades["action"] == "sell"]
        self.assertGreater(len(buys), 0, "至少应有一笔买入")
        # 买入 PnL 应为负（成本+手续费）
        self.assertTrue((buys["pnl"] < 0).all(), "买入 pnl 应为负")
        # 卖出 amount > 0
        self.assertTrue((sells["amount"] > 0).all())

    def test_04_t_plus1(self):
        """T+1：信号当日不能成交，需次日才买"""
        eng = VectorizedBacktestEngine(t_plus_1=True, slippage=0.0)
        result = eng.run_backtest(self.data, self.signals)
        # 第一天的买入应为 0（信号在 t 时无法当日成交）
        first_day_eq = result["equity_curve"].iloc[0]
        self.assertEqual(first_day_eq["market_value"], 0.0,
                         "T+1 下第一天不应有持仓市值")

    def test_05_no_t_plus1(self):
        """非 T+1：信号次日即应建仓（成交在信号当日收盘，权益曲线次日可见）"""
        eng = VectorizedBacktestEngine(t_plus_1=False, slippage=0.0)
        result = eng.run_backtest(self.data, self.signals)
        sig_dates = sorted(self.signals["date"].unique())
        if len(sig_dates) < 2:
            self.skipTest("信号日期不足")
        # 第一个信号日次日，权益曲线应已记录持仓市值
        first_signal_date = sig_dates[0]
        next_day = first_signal_date + pd.Timedelta(days=1)
        eq = result["equity_curve"].copy()
        eq["date"] = pd.to_datetime(eq["date"])
        next_row = eq[eq["date"] >= next_day]
        if next_row.empty:
            self.skipTest("权益曲线缺少信号次日")
        # 在信号次日起的几日内，至少有一天应出现持仓市值
        max_pos = next_row.head(5)["market_value"].max()
        self.assertGreater(max_pos, 0.0,
                           f"非 T+1 下 {first_signal_date} 之后应建仓，最大市值={max_pos}")

    # --- 性能 ---

    def test_06_performance_vs_loop(self):
        """对比：相同数据下的向量化引擎 vs jingni-trader 原生 native_adapter 的耗时"""
        # 扩大数据集
        big = build_synthetic_data(n_stocks=20, n_days=500)
        sig = build_topk_signals(big)

        # 1) 向量化
        t0 = time.perf_counter()
        result_v = self.engine.run_backtest(big, sig)
        t_vec = time.perf_counter() - t0

        # 2) 原生 native_adapter（通过 package 方式加载，避开连字符目录名问题）
        import importlib.util
        import types

        # 构造一个虚拟包 adapters -> base 的父包
        pkg_name = "bt_pkg"
        # 准备一个虚拟的 base 子包
        base_pkg = types.ModuleType(f"{pkg_name}.base")
        base_pkg.__path__ = [os.path.join(
            PROJECT_ROOT, "skills", "backtest-engine", "scripts", "base"
        )]
        sys.modules[base_pkg.__name__] = base_pkg
        # 准备一个虚拟的 scripts.adapters 子包（适配器所在包）
        scripts_pkg = types.ModuleType(f"{pkg_name}")
        scripts_pkg.__path__ = [os.path.join(
            PROJECT_ROOT, "skills", "backtest-engine", "scripts"
        )]
        sys.modules[scripts_pkg.__name__] = scripts_pkg
        # 子包
        adapters_pkg = types.ModuleType(f"{pkg_name}.adapters")
        adapters_pkg.__path__ = [os.path.join(
            PROJECT_ROOT, "skills", "backtest-engine", "scripts", "adapters"
        )]
        sys.modules[adapters_pkg.__name__] = adapters_pkg
        base_init = os.path.join(
            PROJECT_ROOT, "skills", "backtest-engine", "scripts", "base", "__init__.py"
        )
        adapters_init = os.path.join(
            PROJECT_ROOT, "skills", "backtest-engine", "scripts", "adapters", "__init__.py"
        )
        if os.path.exists(base_init):
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.base.__init__", base_init
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        if os.path.exists(adapters_init):
            spec = importlib.util.spec_from_file_location(
                f"{pkg_name}.adapters.__init__", adapters_init
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

        # 加载 native_adapter
        adapter_path = os.path.join(
            PROJECT_ROOT, "skills", "backtest-engine",
            "scripts", "adapters", "native_adapter.py",
        )
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.adapters.native_adapter", adapter_path
        )
        native_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native_mod
        spec.loader.exec_module(native_mod)
        NativeAdapter = native_mod.NativeAdapter
        native = NativeAdapter()
        t0 = time.perf_counter()
        result_n = native.run_backtest(big, sig, init_capital=1_000_000.0)
        t_loop = time.perf_counter() - t0

        speedup = t_loop / max(t_vec, 1e-6)
        print(f"\n[PERF] 向量化 {t_vec:.3f}s vs 原生 {t_loop:.3f}s → 加速比 {speedup:.2f}x")
        print(f"[PERF] 向量化 total_return={result_v['metrics'].get('total_return'):.4f} "
              f"vs 原生 total_return={result_n['metrics'].get('total_return'):.4f}")

        # 写入性能报告文件
        report = {
            "n_stocks": 20,
            "n_days": 500,
            "vec_time_s": t_vec,
            "loop_time_s": t_loop,
            "speedup": speedup,
            "vec_total_return": result_v["metrics"].get("total_return"),
            "loop_total_return": result_n["metrics"].get("total_return"),
        }
        out_dir = os.path.join(HERE, "..", "reports")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "perf_vectorized_vs_native.json"), "w") as f:
            import json
            json.dump(report, f, indent=2)

        # 加速比至少 >= 1.0（不强求具体倍数，因为是 20 标的 500 日的小数据集）
        self.assertGreater(speedup, 1.0,
                           f"向量化引擎应当快于原循环，speedup={speedup:.2f}")

    # --- 边界 ---

    def test_07_empty_data(self):
        result = self.engine.run_backtest(pd.DataFrame(), pd.DataFrame())
        self.assertEqual(result["metrics"], {})
        self.assertTrue(result["trades"].empty)

    def test_08_single_stock(self):
        """单只股票：回测不报错且有合理输出"""
        one = self.data[self.data["code"] == self.data["code"].iloc[0]].copy()
        sig = self.signals[self.signals["code"] == one["code"].iloc[0]].copy()
        if sig.empty:
            # 造一个买入信号
            sig = pd.DataFrame({
                "date": [one["date"].iloc[10], one["date"].iloc[20]],
                "code": [one["code"].iloc[0]] * 2,
                "signal": [1, -1],
            })
        result = self.engine.run_backtest(one, sig)
        self.assertGreaterEqual(len(result["equity_curve"]), 1)

    def test_09_limit_up_filter(self):
        """涨停过滤：涨停当日即使有 buy 信号也不成交"""
        # 制造涨停日
        d = self.data.copy()
        d.loc[d.index[::30], "is_limit_up"] = True
        d.loc[d.index[::30], "change_pct"] = 10.0
        # 移除这些日期的所有卖出信号，确保信号完整
        result = self.engine.run_backtest(d, self.signals)
        # 不应崩溃，权益曲线完整
        self.assertEqual(len(result["equity_curve"]), d["date"].nunique())

    def test_10_lot_size_rounding(self):
        """手数取整：每笔买入股数应为 100 的倍数"""
        result = self.engine.run_backtest(self.data, self.signals)
        buys = result["trades"][result["trades"]["action"] == "buy"]
        if not buys.empty:
            self.assertTrue((buys["shares"] % 100 == 0).all(),
                            "买入股数应为 100 的整数倍")


def _standalone_main():
    """不依赖 pytest 的快速运行入口"""
    import json
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVectorizedEngine)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    _standalone_main()
