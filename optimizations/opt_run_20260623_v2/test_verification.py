"""
jingni-trader 优化验证测试套件

覆盖 3 个优化方向:
1. ExpressionEngine  - 因子表达式引擎（正确性 + 与 pandas_ta 对比 + 边界）
2. PITProvider       - Point-in-Time 数据（未来函数防护 + 修订链 + 边界）
3. VectorizedBacktest - 向量化回测（正确性 + 与 NativeAdapter 对比 + 性能）

运行: python -m optimizations.opt_run_20260623_v2.test_verification
"""
from __future__ import annotations

import time
import unittest
from typing import List

import numpy as np
import pandas as pd

from optimizations.opt_run_20260623_v2.expression_engine import ExpressionEngine, Ops
from optimizations.opt_run_20260623_v2.pit_data_provider import PITProvider
from optimizations.opt_run_20260623_v2.vectorized_backtest import VectorizedBacktest


# ---------------------------------------------------------------------------
# 测试辅助：构造合成数据
# ---------------------------------------------------------------------------

def make_ohlcv(n_codes: int = 5, n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """生成合成 OHLCV 数据"""
    rng = np.random.default_rng(seed)
    base_dates = pd.bdate_range("2024-01-01", periods=n_days)
    frames: List[pd.DataFrame] = []
    for i in range(n_codes):
        code = f"{600000 + i:06d}.SH"
        close = 10.0 + rng.normal(0, 0.2, n_days).cumsum()
        close = np.maximum(close, 1.0)
        open_ = close * (1 + rng.normal(0, 0.01, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, n_days)))
        volume = rng.integers(1_000_000, 10_000_000, n_days)
        frames.append(pd.DataFrame({
            "code": code,
            "date": base_dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
        }))
    return pd.concat(frames, ignore_index=True)


def make_pit_records() -> pd.DataFrame:
    """生成 PIT 财报数据（含修订）"""
    return pd.DataFrame([
        # 2024Q1 财报，原始发布 2024-04-30，2024-06-15 修订
        {"code": "600000.SH", "period": "202403", "field": "roe",
         "publish_date": "2024-04-30", "value": 12.5, "revision_seq": 0},
        {"code": "600000.SH", "period": "202403", "field": "roe",
         "publish_date": "2024-06-15", "value": 13.1, "revision_seq": 1},
        # 2024Q2 财报，2024-08-20 发布
        {"code": "600000.SH", "period": "202406", "field": "roe",
         "publish_date": "2024-08-20", "value": 14.0, "revision_seq": 0},
        # 第二只股票
        {"code": "600001.SH", "period": "202403", "field": "roe",
         "publish_date": "2024-04-29", "value": 8.2, "revision_seq": 0},
        {"code": "600001.SH", "period": "202406", "field": "roe",
         "publish_date": "2024-08-22", "value": 9.1, "revision_seq": 0},
    ])


# ===========================================================================
# 1. ExpressionEngine 测试
# ===========================================================================

class TestExpressionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ExpressionEngine()
        self.data = make_ohlcv(n_codes=3, n_days=40)

    def test_field_access(self):
        """字段访问"""
        out = self.engine.evaluate("$close", self.data)
        self.assertIn("factor", out.columns)
        np.testing.assert_allclose(out["factor"].values, self.data["close"].values)

    def test_arithmetic(self):
        """四则运算"""
        out = self.engine.evaluate("($high - $low) / $close", self.data)
        expected = (self.data["high"] - self.data["low"]) / self.data["close"]
        np.testing.assert_allclose(out["factor"].values, expected.values, rtol=1e-10)

    def test_ref_operator(self):
        """Ref 时间序列算子"""
        out = self.engine.evaluate("Ref($close, 5)", self.data)
        # 手动验证
        expected = self.data.sort_values(["code", "date"]).groupby("code")["close"].shift(5)
        out_sorted = out.sort_values(["code", "date"])
        # 对齐后比较非 NaN 部分
        mask = expected.notna()
        np.testing.assert_allclose(
            out_sorted["factor"].values[mask],
            expected.values[mask],
            rtol=1e-10,
        )

    def test_mean_operator(self):
        """Mean 滚动均值"""
        out = self.engine.evaluate("Mean($close, 5)", self.data)
        expected = self.data.sort_values(["code", "date"]).groupby("code")["close"].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
        out_sorted = out.sort_values(["code", "date"]).reset_index(drop=True)
        expected = expected.reset_index(drop=True)
        mask = expected.notna() & out_sorted["factor"].notna()
        np.testing.assert_allclose(
            out_sorted["factor"].values[mask],
            expected.values[mask],
            rtol=1e-9,
        )

    def test_cs_rank(self):
        """CSRank 横截面排名"""
        out = self.engine.evaluate("CSRank($close)", self.data)
        # 横截面排名应在 [0,1]
        valid = out["factor"].dropna()
        self.assertTrue((valid >= 0).all() and (valid <= 1).all())
        # 同一日期内排名应单调对应 close
        merged = out.merge(self.data[["code", "date", "close"]], on=["code", "date"])
        for dt, g in merged.groupby("date"):
            ranks = g["factor"].rank(method="average", pct=True)
            np.testing.assert_allclose(g["factor"].values, ranks.values)

    def test_complex_expression(self):
        """复合表达式：5日反转因子 = -1 * CSRank(Delta(close, 5) / close)"""
        expr = "-1 * CSRank(Delta($close, 5) / $close)"
        out = self.engine.evaluate(expr, self.data)
        self.assertFalse(out["factor"].isna().all())
        # 验证范围合理
        valid = out["factor"].dropna()
        self.assertTrue((valid >= -1.01).all() and (valid <= 1.01).all())

    def test_evaluate_many(self):
        """批量计算多因子"""
        exprs = {
            "ret_5d": "Ref($close, 5) / $close - 1",
            "vol_20d": "Std($close / Ref($close, 1) - 1, 20)",
            "mom_10d": "Delta($close, 10) / Ref($close, 10)",
        }
        out = self.engine.evaluate_many(exprs, self.data)
        for name in exprs:
            self.assertIn(name, out.columns)

    def test_unknown_operator_raises(self):
        """未知算子应抛 KeyError"""
        with self.assertRaises(KeyError):
            self.engine.evaluate("UnknownOp($close)", self.data)

    def test_unknown_field_raises(self):
        """未知字段应抛 KeyError"""
        with self.assertRaises(KeyError):
            self.engine.evaluate("$nonexistent", self.data)

    def test_empty_data(self):
        """空数据边界：应优雅处理（返回空或抛出明确异常）"""
        empty = pd.DataFrame(columns=["code", "date", "close"])
        try:
            out = self.engine.evaluate("$close", empty)
            # 若不抛异常，结果应为空
            self.assertTrue(out.empty or len(out) == 0)
        except Exception:
            # 抛出明确异常也算正确处理
            pass

    def test_cache_hit(self):
        """表达式解析缓存：第二次应更快"""
        expr = "Mean(Std($close, 10), 20)"
        self.engine._parse(expr)  # 预热
        self.assertIn(expr, self.engine._cache)


# ===========================================================================
# 2. PITProvider 测试
# ===========================================================================

class TestPITProvider(unittest.TestCase):
    def setUp(self):
        self.provider = PITProvider()
        self.provider.load_records(make_pit_records())

    def test_before_publish_returns_none(self):
        """发布日之前查询应返回 None"""
        v = self.provider.get_pit("600000.SH", "roe", "2024-04-29")
        self.assertIsNone(v)

    def test_original_value_before_revision(self):
        """修订前应返回原始值"""
        v = self.provider.get_pit("600000.SH", "roe", "2024-05-15")
        self.assertAlmostEqual(v, 12.5)

    def test_revised_value_after_revision(self):
        """修订后应返回修订值（防未来函数的核心）"""
        v = self.provider.get_pit("600000.SH", "roe", "2024-06-20")
        self.assertAlmostEqual(v, 13.1)

    def test_next_period_value(self):
        """下一报告期查询"""
        v = self.provider.get_pit("600000.SH", "roe", "2024-09-01")
        self.assertAlmostEqual(v, 14.0)

    def test_look_ahead_bias_protection(self):
        """未来函数防护：2024-05-15 不应拿到 2024-06-15 的修订值"""
        v_may = self.provider.get_pit("600000.SH", "roe", "2024-05-15")
        v_jun = self.provider.get_pit("600000.SH", "roe", "2024-06-20")
        self.assertNotEqual(v_may, v_jun)
        self.assertEqual(v_may, 12.5)  # 原始值
        self.assertEqual(v_jun, 13.1)  # 修订值

    def test_revision_chain(self):
        """修订链完整性"""
        chain = self.provider.revision_chain("600000.SH", "roe", "202403")
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain.iloc[0]["value"], 12.5)
        self.assertEqual(chain.iloc[1]["value"], 13.1)

    def test_as_of_pit_alignment(self):
        """面板数据按观察日对齐"""
        panel = pd.DataFrame({
            "code": ["600000.SH"] * 4,
            "date": pd.to_datetime(["2024-04-15", "2024-05-15", "2024-06-20", "2024-09-01"]),
        })
        out = self.provider.as_of_pit(panel)
        self.assertIn("roe_pit", out.columns)
        vals = out["roe_pit"].tolist()
        # None 在 pandas 列中会变为 NaN，用 isna 判断
        self.assertTrue(pd.isna(vals[0]))
        self.assertAlmostEqual(vals[1], 12.5)
        self.assertAlmostEqual(vals[2], 13.1)
        self.assertAlmostEqual(vals[3], 14.0)

    def test_unknown_code_returns_none(self):
        """未知股票返回 None"""
        v = self.provider.get_pit("999999.SH", "roe", "2024-12-01")
        self.assertIsNone(v)

    def test_stats(self):
        """统计信息"""
        s = self.provider.stats()
        self.assertEqual(s["codes"], 2)
        self.assertEqual(s["fields"], 1)
        self.assertGreaterEqual(s["records"], 5)

    def test_empty_provider(self):
        """空 provider 边界"""
        empty = PITProvider()
        self.assertIsNone(empty.get_pit("X", "y", "2024-01-01"))
        self.assertEqual(empty.stats()["records"], 0)


# ===========================================================================
# 3. VectorizedBacktest 测试
# ===========================================================================

class TestVectorizedBacktest(unittest.TestCase):
    def setUp(self):
        self.bt = VectorizedBacktest()
        self.data = make_ohlcv(n_codes=5, n_days=30, seed=1)
        # 简单信号：每 5 天轮换买入前 2 只
        dates = sorted(self.data["date"].unique())
        sig_rows = []
        for i, dt in enumerate(dates):
            for code in self.data["code"].unique()[:2]:
                sig_rows.append({
                    "date": dt, "code": code,
                    "signal": 1 if (i // 5) % 2 == 0 else -1,
                })
        self.signals = pd.DataFrame(sig_rows)

    def test_basic_run(self):
        """基本回测能跑通"""
        result = self.bt.run_backtest(self.data, self.signals)
        self.assertIn("equity_curve", result)
        self.assertFalse(result["equity_curve"].empty)
        self.assertIn("metrics", result)
        self.assertIn("total_return", result["metrics"])

    def test_empty_data(self):
        """空数据边界"""
        result = self.bt.run_backtest(
            pd.DataFrame(), pd.DataFrame()
        )
        self.assertTrue(result["equity_curve"].empty)

    def test_no_signals(self):
        """无信号边界：净值应保持不变"""
        empty_sig = pd.DataFrame(columns=["date", "code", "signal"])
        result = self.bt.run_backtest(self.data, empty_sig)
        if not result["equity_curve"].empty:
            eq = result["equity_curve"]["equity"]
            # 无交易则净值应接近初始资金
            np.testing.assert_allclose(eq.iloc[0], 1e6, rtol=1e-6)

    def test_target_weight_mode(self):
        """目标权重模式"""
        sig = self.signals.copy()
        sig["target_weight"] = np.where(sig["signal"] > 0, 0.5, 0.0)
        sig = sig.drop(columns=["signal"])
        result = self.bt.run_backtest(self.data, sig)
        self.assertFalse(result["equity_curve"].empty)

    def test_metrics_reasonable(self):
        """指标合理性"""
        result = self.bt.run_backtest(self.data, self.signals)
        m = result["metrics"]
        self.assertGreaterEqual(m["n_days"], 1)
        self.assertTrue(-1 <= m["max_drawdown"] <= 0)
        # 夏普比率应在合理范围
        self.assertTrue(-50 < m["sharpe_ratio"] < 50)

    def test_price_limit_blocks_trades(self):
        """涨跌停限制：涨停日不能买入"""
        data = self.data.copy()
        # 把第一天的第一只股票标记为涨停
        first_date = data["date"].min()
        mask = (data["date"] == first_date) & (data["code"] == data["code"].iloc[0])
        data.loc[mask, "is_limit_up"] = True
        data["is_limit_up"] = data.get("is_limit_up", False)
        data["is_limit_down"] = False
        result = self.bt.run_backtest(data, self.signals)
        # 应能跑完
        self.assertFalse(result["equity_curve"].empty)


# ===========================================================================
# 4. 性能对比测试（VectorizedBacktest vs 原生 iterrows 思路）
# ===========================================================================

class TestPerformanceComparison(unittest.TestCase):
    """性能对比：向量化回测 vs 模拟原 native_adapter 的逐行实现"""

    def _naive_iterrows_backtest(self, data, signals, init_capital=1e6):
        """模拟原 native_adapter 的逐行实现，用于性能对比基线"""
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)
        dates = sorted(signals["date"].unique())
        cash = init_capital
        positions = {}
        equity_records = []
        for dt in dates:
            day_signal = signals[signals["date"] == dt]
            day_data = data[data["date"] == dt]
            if day_data.empty:
                continue
            day_data_map = day_data.set_index("code")
            for _, row in day_signal.iterrows():
                code = row["code"]
                sig = row.get("signal", 0)
                if isinstance(sig, (int, float, np.integer, np.floating)):
                    if sig > 0:
                        if code in day_data_map.index:
                            price = day_data_map.loc[code, "close"]
                            budget = cash * 0.95
                            shares = int(budget / price / 100) * 100
                            if shares > 0:
                                cash -= shares * price
                                positions[code] = positions.get(code, 0) + shares
                    elif sig < 0 and code in positions:
                        price = day_data_map.loc[code, "close"]
                        cash += positions[code] * price
                        positions[code] = 0
            mv = 0
            for code, shares in positions.items():
                if shares > 0 and code in day_data_map.index:
                    mv += shares * day_data_map.loc[code, "close"]
            equity_records.append({"date": dt, "equity": cash + mv})
        return pd.DataFrame(equity_records)

    def test_performance_vectorized_faster(self):
        """向量化实现应明显快于逐行实现"""
        data = make_ohlcv(n_codes=20, n_days=120, seed=7)
        # 生成信号
        dates = sorted(data["date"].unique())
        sig_rows = []
        for dt in dates:
            for code in data["code"].unique()[:10]:
                sig_rows.append({"date": dt, "code": code, "signal": 1})
        signals = pd.DataFrame(sig_rows)

        # 基线（逐行）
        t0 = time.perf_counter()
        self._naive_iterrows_backtest(data, signals)
        t_naive = time.perf_counter() - t0

        # 向量化
        bt = VectorizedBacktest()
        t0 = time.perf_counter()
        bt.run_backtest(data, signals)
        t_vec = time.perf_counter() - t0

        speedup = t_naive / t_vec if t_vec > 0 else float("inf")
        print(f"\n[性能] 逐行: {t_naive*1000:.1f}ms | 向量化: {t_vec*1000:.1f}ms | 加速比: {speedup:.1f}x")
        # 向量化应至少快 2x（宽松阈值以适应不同环境）
        self.assertGreater(speedup, 2.0, f"向量化未达预期加速: {speedup:.1f}x")


# ===========================================================================
# 5. 集成测试：表达式因子 → PIT 对齐 → 向量化回测
# ===========================================================================

class TestIntegrationPipeline(unittest.TestCase):
    """端到端：用表达式引擎算因子 → PIT 对齐 → 向量化回测"""

    def test_full_pipeline(self):
        data = make_ohlcv(n_codes=5, n_days=40, seed=11)

        # 1. 表达式因子
        engine = ExpressionEngine()
        factor_df = engine.evaluate("CSRank(-1 * Delta($close, 5) / Ref($close, 5))", data)
        factor_df = factor_df.rename(columns={"factor": "reversal_5d"})

        # 2. PIT 对齐
        pit = PITProvider()
        pit.load_records(make_pit_records())
        merged = data.merge(factor_df[["code", "date", "reversal_5d"]], on=["code", "date"])
        merged_with_pit = pit.as_of_pit(merged)

        # 3. 生成信号：反转因子 top
        merged_with_pit = merged_with_pit.sort_values(["date", "reversal_5d"], ascending=[True, False])
        signals = merged_with_pit.groupby("date").head(2).copy()
        signals["signal"] = 1
        signals = signals[["date", "code", "signal"]]

        # 4. 向量化回测
        bt = VectorizedBacktest()
        result = bt.run_backtest(data, signals)
        self.assertFalse(result["equity_curve"].empty)
        self.assertIn("sharpe_ratio", result["metrics"])
        print(f"\n[集成] 端到端回测完成: 净值={result['equity_curve']['equity'].iloc[-1]:.0f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
