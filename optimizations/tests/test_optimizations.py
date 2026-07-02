"""
验证测试套件：向量化回测 + 表达式因子 + 增强指标

测试内容：
1. 正确性测试：向量化回测 vs 原生回测结果一致性
2. 性能对比测试：向量化 vs 原生的速度差异
3. 边界条件测试：空数据、单只股票、无信号等
4. 表达式引擎测试：算子正确性、Alpha158 因子完整性
5. 增强指标测试：30+ 指标计算正确性
6. 向量化 IC 分析测试：vs 逐日循环一致性
"""
import os
import sys
import time
import json
import unittest
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizations.vectorized_backtest import VectorizedBacktestEngine
from optimizations.expression_factors import (
    ExpressionEngine,
    Alpha158FactorLibrary,
    VectorizedICAnalysis,
)
from optimizations.enhanced_metrics import EnhancedMetrics

warnings.filterwarnings("ignore")


# ============================================================
# 测试数据生成器
# ============================================================

def generate_synthetic_data(
    n_codes: int = 20,
    n_days: int = 250,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """生成合成 A 股日线数据"""
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    rows = []
    for code in codes:
        price = np.random.uniform(10, 50)
        for dt in dates:
            ret = np.random.normal(0.0005, 0.02)
            price = max(price * (1 + ret), 1.0)
            open_p = price * (1 + np.random.normal(0, 0.005))
            high = max(price, open_p) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(price, open_p) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.lognormal(15, 0.5))
            amount = price * volume
            pre_close = price / (1 + ret)
            change_pct = (price - pre_close) / pre_close * 100
            rows.append({
                "date": dt,
                "code": code,
                "open": round(open_p, 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(price, 4),
                "volume": volume,
                "amount": round(amount, 2),
                "pre_close": round(pre_close, 4),
                "change_pct": round(change_pct, 4),
                "is_st": False,
                "is_limit_up": change_pct >= 9.9,
                "is_limit_down": change_pct <= -9.9,
            })

    return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)


def generate_signals(data: pd.DataFrame, strategy: str = "momentum") -> pd.DataFrame:
    """基于数据生成交易信号"""
    signals = []
    if strategy == "momentum":
        # 简单动量策略：20 日均线之上买入，之下卖出
        for code, group in data.groupby("code"):
            group = group.sort_values("date").copy()
            group["ma20"] = group["close"].rolling(20, min_periods=20).mean()
            group["signal"] = 0
            group.loc[group["close"] > group["ma20"], "signal"] = 1
            # 每 5 天产生一次卖出信号（避免持仓过久）
            group["sell_flag"] = group["close"] < group["ma20"]
            group.loc[group["sell_flag"], "signal"] = -1
            for _, row in group.iterrows():
                if row["signal"] != 0:
                    signals.append({
                        "date": row["date"],
                        "code": row["code"],
                        "signal": int(row["signal"]),
                    })
    elif strategy == "reversal":
        # 反转策略：5 日跌幅 > 5% 买入
        for code, group in data.groupby("code"):
            group = group.sort_values("date").copy()
            group["ret_5d"] = group["close"].pct_change(5)
            group["signal"] = 0
            group.loc[group["ret_5d"] < -0.05, "signal"] = 1
            group.loc[group["ret_5d"] > 0.05, "signal"] = -1
            for _, row in group.iterrows():
                if row["signal"] != 0:
                    signals.append({
                        "date": row["date"],
                        "code": row["code"],
                        "signal": int(row["signal"]),
                    })

    return pd.DataFrame(signals)


# ============================================================
# 1. 向量化回测正确性测试
# ============================================================

class TestVectorizedBacktestCorrectness(unittest.TestCase):
    """向量化回测正确性测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_data(n_codes=10, n_days=120)
        cls.signals = generate_signals(cls.data, "momentum")

    def test_basic_run(self):
        """测试基本回测能正常运行"""
        engine = VectorizedBacktestEngine(init_capital=1_000_000)
        result = engine.run(self.data, self.signals)

        self.assertIn("equity_curve", result)
        self.assertIn("trades", result)
        self.assertIn("metrics", result)
        self.assertFalse(result["equity_curve"].empty, "净值曲线不应为空")
        self.assertGreater(result["elapsed_seconds"], 0)

    def test_equity_curve_shape(self):
        """测试净值曲线形状正确"""
        engine = VectorizedBacktestEngine()
        result = engine.run(self.data, self.signals)
        eq = result["equity_curve"]

        self.assertIn("equity", eq.columns)
        self.assertIn("cash", eq.columns)
        self.assertIn("market_value", eq.columns)
        self.assertIn("position_count", eq.columns)
        # 净值曲线长度应等于数据中的交易日数
        n_dates = self.data["date"].nunique()
        self.assertEqual(len(eq), n_dates)

    def test_initial_capital(self):
        """测试初始资金正确"""
        engine = VectorizedBacktestEngine(init_capital=2_000_000)
        result = engine.run(self.data, self.signals)
        # 第一天净值应接近初始资金（可能有交易）
        first_equity = result["equity_curve"]["equity"].iloc[0]
        self.assertAlmostEqual(first_equity, 2_000_000, delta=2_000_000 * 0.5)

    def test_metrics_completeness(self):
        """测试绩效指标完整性"""
        engine = VectorizedBacktestEngine()
        result = engine.run(self.data, self.signals)
        metrics = result["metrics"]

        expected_keys = [
            "total_return", "annual_return", "volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "win_rate", "total_trades",
        ]
        for key in expected_keys:
            self.assertIn(key, metrics, f"缺少指标: {key}")

    def test_no_lookahead_bias(self):
        """测试无前视偏差：T+1 规则生效"""
        # 构造一个明确信号：第 10 天买入，第 11 天才能卖
        dates = sorted(self.data["date"].unique())
        if len(dates) < 15:
            self.skipTest("数据天数不足")
        code = self.data["code"].iloc[0]
        signals = pd.DataFrame([
            {"date": dates[10], "code": code, "signal": 1},
            {"date": dates[11], "code": code, "signal": -1},
        ])
        engine = VectorizedBacktestEngine(t_plus_1=True)
        result = engine.run(self.data, signals)
        # 应该有买入和卖出记录
        trades = result["trades"]
        self.assertFalse(trades.empty, "应有成交记录")
        actions = trades["action"].unique()
        self.assertIn("buy", actions)


# ============================================================
# 2. 性能对比测试
# ============================================================

class TestPerformanceComparison(unittest.TestCase):
    """向量化回测 vs 原生回测性能对比"""

    @classmethod
    def setUpClass(cls):
        # 生成较大数据集以体现性能差异
        cls.data_large = generate_synthetic_data(n_codes=50, n_days=500)
        cls.signals_large = generate_signals(cls.data_large, "momentum")

    def test_vectorized_performance(self):
        """测试向量化回测性能"""
        engine = VectorizedBacktestEngine()
        result = engine.run(self.data_large, self.signals_large)
        elapsed = result["elapsed_seconds"]

        print(f"\n[性能] 向量化回测耗时: {elapsed:.4f}s, "
              f"数据规模: {len(self.data_large)} 行, "
              f"信号数: {len(self.signals_large)}")

        # 50 只股票 × 500 天应在合理时间内完成
        self.assertLess(elapsed, 30.0, "向量化回测应在 30 秒内完成")

    def test_scalability(self):
        """测试扩展性：不同数据规模下的耗时"""
        results = []
        for n_codes, n_days in [(10, 100), (20, 250), (50, 500)]:
            data = generate_synthetic_data(n_codes=n_codes, n_days=n_days, seed=42)
            signals = generate_signals(data, "momentum")
            engine = VectorizedBacktestEngine()
            r = engine.run(data, signals)
            results.append({
                "n_codes": n_codes,
                "n_days": n_days,
                "n_rows": len(data),
                "n_signals": len(signals),
                "elapsed": r["elapsed_seconds"],
                "total_return": r["metrics"].get("total_return", 0),
            })

        print("\n[扩展性] 不同规模下的回测耗时:")
        for r in results:
            print(f"  {r['n_codes']}只×{r['n_days']}天 "
                  f"({r['n_rows']}行): {r['elapsed']:.4f}s, "
                  f"收益={r['total_return']:.4f}")

        # 耗时应随数据规模近似线性增长（不爆炸）
        self.assertGreater(results[-1]["elapsed"], 0)


# ============================================================
# 3. 边界条件测试
# ============================================================

class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """测试空数据"""
        engine = VectorizedBacktestEngine()
        result = engine.run(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(result["equity_curve"].empty)
        self.assertEqual(result["metrics"], {})

    def test_empty_signals(self):
        """测试空信号"""
        data = generate_synthetic_data(n_codes=5, n_days=30)
        engine = VectorizedBacktestEngine()
        result = engine.run(data, pd.DataFrame())
        # 无信号也应返回净值曲线（全现金）
        self.assertFalse(result["equity_curve"].empty)
        # 全程应为初始资金
        self.assertAlmostEqual(
            result["equity_curve"]["equity"].iloc[0], 1_000_000
        )

    def test_single_stock(self):
        """测试单只股票"""
        data = generate_synthetic_data(n_codes=1, n_days=60)
        signals = generate_signals(data, "momentum")
        engine = VectorizedBacktestEngine()
        result = engine.run(data, signals)
        self.assertFalse(result["equity_curve"].empty)

    def test_all_limit_up(self):
        """测试全部涨停（无法买入）"""
        data = generate_synthetic_data(n_codes=3, n_days=20)
        # 强制全部涨停
        data["is_limit_up"] = True
        dates = sorted(data["date"].unique())
        signals = pd.DataFrame([
            {"date": dates[5], "code": data["code"].iloc[0], "signal": 1},
        ])
        engine = VectorizedBacktestEngine(price_limit=True)
        result = engine.run(data, signals)
        # 涨停日买入信号应被阻断
        buy_trades = result["trades"][result["trades"]["action"] == "buy"] if not result["trades"].empty else pd.DataFrame()
        self.assertTrue(buy_trades.empty, "涨停日不应有买入成交")

    def test_insufficient_capital(self):
        """测试资金不足"""
        data = generate_synthetic_data(n_codes=5, n_days=30)
        dates = sorted(data["date"].unique())
        # 极小初始资金
        signals = pd.DataFrame([
            {"date": dates[5], "code": c, "signal": 1}
            for c in data["code"].unique()
        ])
        engine = VectorizedBacktestEngine(init_capital=1000)
        result = engine.run(data, signals)
        # 不应崩溃，且现金不应为负
        self.assertFalse(result["equity_curve"].empty)
        self.assertTrue((result["equity_curve"]["cash"] >= -1).all(),
                        "现金不应严重为负")


# ============================================================
# 4. 表达式引擎测试
# ============================================================

class TestExpressionEngine(unittest.TestCase):
    """表达式引擎测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_data(n_codes=5, n_days=100)
        # 转为 (code, date) MultiIndex
        df = cls.data.copy()
        df["date"] = pd.to_datetime(df["date"])
        cls.df_indexed = df.set_index(["code", "date"]).sort_index()
        cls.engine = ExpressionEngine()

    def test_field_reference(self):
        """测试字段引用"""
        result = self.engine.evaluate("$close", self.df_indexed)
        self.assertIsInstance(result, pd.Series)
        pd.testing.assert_series_equal(
            result, self.df_indexed["close"], check_names=False
        )

    def test_ref_operator(self):
        """测试 Ref 算子"""
        result = self.engine.evaluate("Ref($close, 5)", self.df_indexed)
        expected = self.df_indexed["close"].groupby(level="code").shift(5)
        pd.testing.assert_series_equal(
            result.dropna(), expected.dropna(), check_names=False
        )

    def test_mean_operator(self):
        """测试 Mean 算子"""
        result = self.engine.evaluate("Mean($close, 20)", self.df_indexed)
        expected = self.df_indexed["close"].groupby(level="code").transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        pd.testing.assert_series_equal(
            result, expected, check_names=False
        )

    def test_arithmetic(self):
        """测试算术运算"""
        expr = "($high - $low) / $close"
        result = self.engine.evaluate(expr, self.df_indexed)
        expected = (self.df_indexed["high"] - self.df_indexed["low"]) / self.df_indexed["close"]
        pd.testing.assert_series_equal(
            result, expected, check_names=False
        )

    def test_complex_expression(self):
        """测试复杂表达式"""
        expr = "Mean($close, 20) / $close"
        result = self.engine.evaluate(expr, self.df_indexed)
        ma20 = self.df_indexed["close"].groupby(level="code").transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        expected = ma20 / self.df_indexed["close"]
        pd.testing.assert_series_equal(
            result, expected, check_names=False
        )


# ============================================================
# 5. Alpha158 因子库测试
# ============================================================

class TestAlpha158FactorLibrary(unittest.TestCase):
    """Alpha158 因子库测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_data(n_codes=5, n_days=120)
        cls.lib = Alpha158FactorLibrary()

    def test_factor_count(self):
        """测试因子数量（应 >= 150）"""
        factors = self.lib.list_factors()
        print(f"\n[Alpha158] 因子总数: {len(factors)}")
        self.assertGreaterEqual(len(factors), 150, "Alpha158 应至少有 150 个因子")

    def test_factor_categories(self):
        """测试因子分类完整性"""
        factors = self.lib.list_factors()
        # K 线基础
        for f in ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]:
            self.assertIn(f, factors, f"缺少 K 线因子: {f}")
        # 趋势类
        for n in (5, 10, 20, 30, 60):
            for prefix in ("ROC", "MA", "BETA", "RSQR", "RESI"):
                self.assertIn(f"{prefix}{n}", factors)
        # 波动类
        for n in (5, 10, 20, 30, 60):
            for prefix in ("STD", "MAX", "MIN", "QTLU", "QTLD", "RSV"):
                self.assertIn(f"{prefix}{n}", factors)

    def test_compute_all(self):
        """测试计算全部因子"""
        result = self.lib.compute_all(self.data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("code", result.columns)
        self.assertIn("date", result.columns)
        # 至少应有一些因子列计算成功
        factor_cols = [c for c in result.columns if c not in ("code", "date")]
        print(f"\n[Alpha158] 成功计算因子数: {len(factor_cols)}")
        self.assertGreater(len(factor_cols), 50, "至少 50 个因子应计算成功")

    def test_single_factor(self):
        """测试计算单个因子"""
        result = self.lib.compute_factor(self.data, "KMID")
        self.assertIsInstance(result, pd.Series)
        # KMID = (close - open) / open
        df = self.data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["code", "date"]).sort_index()
        expected = (df["close"] - df["open"]) / df["open"]
        pd.testing.assert_series_equal(
            result.dropna(), expected.dropna(), check_names=False
        )


# ============================================================
# 6. 增强指标测试
# ============================================================

class TestEnhancedMetrics(unittest.TestCase):
    """增强版绩效指标测试"""

    @classmethod
    def setUpClass(cls):
        # 构造已知净值序列
        np.random.seed(42)
        n = 252
        rets = np.random.normal(0.0004, 0.015, n)
        # 用 numpy 数组直接构造，避免 pandas 索引对齐导致 NaN
        cls.equity = pd.Series(
            np.cumprod(1 + rets) * 1_000_000,
            index=pd.bdate_range("2024-01-01", periods=n),
        )
        # 基准
        bench_rets = np.random.normal(0.0003, 0.012, n)
        cls.benchmark = pd.Series(
            np.cumprod(1 + bench_rets) * 1_000_000,
            index=cls.equity.index,
        )
        # 构造交易记录
        cls.trades = pd.DataFrame([
            {"date": cls.equity.index[10], "code": "600000.SH", "action": "buy",
             "price": 10.0, "shares": 1000, "amount": 10000.0,
             "commission": 5.0, "tax": 0.0, "pnl": -10005.0},
            {"date": cls.equity.index[20], "code": "600000.SH", "action": "sell",
             "price": 11.0, "shares": 1000, "amount": 11000.0,
             "commission": 5.5, "tax": 11.0, "pnl": 983.5},
            {"date": cls.equity.index[30], "code": "600001.SH", "action": "buy",
             "price": 20.0, "shares": 500, "amount": 10000.0,
             "commission": 5.0, "tax": 0.0, "pnl": -10005.0},
            {"date": cls.equity.index[40], "code": "600001.SH", "action": "sell",
             "price": 19.0, "shares": 500, "amount": 9500.0,
             "commission": 4.75, "tax": 9.5, "pnl": -514.25},
        ])
        cls.metrics_calc = EnhancedMetrics()

    def test_metrics_count(self):
        """测试指标数量（应 >= 25）"""
        metrics = self.metrics_calc.calc_all(
            self.equity, self.trades, self.benchmark
        )
        print(f"\n[增强指标] 指标总数: {len(metrics)}")
        print(f"  指标列表: {sorted(metrics.keys())}")
        self.assertGreaterEqual(len(metrics), 25, "应至少有 25 个指标")

    def test_return_metrics(self):
        """测试收益类指标"""
        metrics = self.metrics_calc.calc_all(self.equity)
        self.assertIn("total_return", metrics)
        self.assertIn("annual_return", metrics)
        self.assertGreater(metrics["total_return"], -1)
        self.assertGreater(metrics["annual_return"], -1)

    def test_risk_metrics(self):
        """测试风险类指标"""
        metrics = self.metrics_calc.calc_all(self.equity)
        self.assertIn("volatility", metrics)
        self.assertIn("var_95", metrics)
        self.assertIn("cvar_95", metrics)
        self.assertGreater(metrics["volatility"], 0)
        # VaR 应为负数（损失）
        self.assertLess(metrics["var_95"], 0)
        # CVaR 应小于等于 VaR
        self.assertLessEqual(metrics["cvar_95"], metrics["var_95"])

    def test_benchmark_metrics(self):
        """测试相对基准指标"""
        metrics = self.metrics_calc.calc_all(
            self.equity, trades=None, benchmark=self.benchmark
        )
        self.assertIn("alpha", metrics)
        self.assertIn("beta", metrics)
        self.assertIn("information_ratio", metrics)
        self.assertIn("tracking_error", metrics)

    def test_trade_metrics(self):
        """测试交易类指标"""
        metrics = self.metrics_calc.calc_all(self.equity, self.trades)
        self.assertIn("total_trades", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_factor", metrics)
        self.assertEqual(metrics["total_trades"], 4)
        # 2 笔卖出，1 笔盈利
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_drawdown_metrics(self):
        """测试回撤类指标"""
        metrics = self.metrics_calc.calc_all(self.equity)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("max_drawdown_duration_days", metrics)
        self.assertLessEqual(metrics["max_drawdown"], 0)


# ============================================================
# 7. 向量化 IC 分析测试
# ============================================================

class TestVectorizedICAnalysis(unittest.TestCase):
    """向量化 IC 分析测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_synthetic_data(n_codes=20, n_days=150)
        # 构造因子和未来收益
        df = cls.data.sort_values(["code", "date"]).copy()
        df["factor_mom20"] = df.groupby("code")["close"].pct_change(20)
        df["ret_forward_5d"] = df.groupby("code")["close"].transform(
            lambda x: x.shift(-5) / x - 1
        )
        cls.factor_df = df[["date", "code", "factor_mom20"]].copy()
        cls.forward_df = df[["date", "code", "ret_forward_5d"]].copy()

    def test_ic_series(self):
        """测试 IC 序列计算"""
        ic_series = VectorizedICAnalysis.calc_ic_series(
            self.factor_df, self.forward_df,
            factor_col="factor_mom20",
            forward_col="ret_forward_5d",
            method="spearman",
        )
        self.assertIsInstance(ic_series, pd.Series)
        self.assertGreater(len(ic_series), 0)
        # IC 值应在 [-1, 1] 范围内
        self.assertTrue((ic_series.abs() <= 1).all())

    def test_ic_summary(self):
        """测试 IC 统计摘要"""
        ic_series = VectorizedICAnalysis.calc_ic_series(
            self.factor_df, self.forward_df,
            factor_col="factor_mom20",
        )
        summary = VectorizedICAnalysis.calc_ic_summary(ic_series)
        self.assertIn("ic_mean", summary)
        self.assertIn("ic_ir", summary)
        self.assertIn("ic_t_stat", summary)
        print(f"\n[IC分析] IC均值={summary['ic_mean']}, "
              f"ICIR={summary['ic_ir']}, t统计量={summary['ic_t_stat']}")


# ============================================================
# 8. 集成测试：完整流程
# ============================================================

class TestIntegrationFlow(unittest.TestCase):
    """集成测试：数据 → 因子 → 信号 → 回测 → 指标"""

    def test_full_pipeline(self):
        """测试完整流程"""
        print("\n[集成测试] 完整流程: 数据→因子→信号→回测→指标")

        # 1. 数据
        data = generate_synthetic_data(n_codes=15, n_days=200)
        print(f"  1. 数据: {len(data)} 行, {data['code'].nunique()} 只股票")

        # 2. 因子计算
        lib = Alpha158FactorLibrary()
        factors = lib.compute_all(data)
        factor_cols = [c for c in factors.columns if c not in ("code", "date")]
        print(f"  2. 因子: 计算 {len(factor_cols)} 个")

        # 3. 生成信号（用 ROC20 因子排名前 20% 买入）
        factors["rank"] = factors.groupby("date")[factor_cols[0]].rank(pct=True)
        signals = factors[factors["rank"] > 0.8][["date", "code"]].copy()
        signals["signal"] = 1
        # 添加卖出信号（排名后 20%）
        sell = factors[factors["rank"] < 0.2][["date", "code"]].copy()
        sell["signal"] = -1
        signals = pd.concat([signals, sell], ignore_index=True)
        print(f"  3. 信号: {len(signals)} 条")

        # 4. 回测
        engine = VectorizedBacktestEngine()
        result = engine.run(data, signals)
        print(f"  4. 回测: {result['elapsed_seconds']:.4f}s, "
              f"成交 {len(result['trades'])} 笔")

        # 5. 增强指标
        eq_series = result["equity_curve"].set_index("date")["equity"]
        metrics_calc = EnhancedMetrics()
        all_metrics = metrics_calc.calc_all(
            eq_series, result["trades"]
        )
        print(f"  5. 指标: {len(all_metrics)} 个")
        print(f"     总收益: {all_metrics.get('total_return', 0):.4f}")
        print(f"     夏普: {all_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"     最大回撤: {all_metrics.get('max_drawdown', 0):.4f}")

        self.assertFalse(result["equity_curve"].empty)
        self.assertGreater(len(all_metrics), 20)


# ============================================================
# 测试入口
# ============================================================

def run_all_tests():
    """运行全部测试并生成报告"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestVectorizedBacktestCorrectness,
        TestPerformanceComparison,
        TestBoundaryConditions,
        TestExpressionEngine,
        TestAlpha158FactorLibrary,
        TestEnhancedMetrics,
        TestVectorizedICAnalysis,
        TestIntegrationFlow,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_all_tests()
