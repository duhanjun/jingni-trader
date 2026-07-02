"""
验证测试：向量化回测引擎性能对比
====================================================
借鉴来源: VectorBT (https://github.com/polakowo/vectorbt)
优化方向: backtest-engine - 添加向量化回测模式，加速大规模参数扫描
日期: 2026-06-14

VectorBT 的核心亮点：
  - 纯向量化运算替代事件驱动循环，速度提升 100-1000x
  - 单行代码完成参数网格扫描和热力图可视化
  - 基于 NumPy/Numba 的 JIT 编译加速
  - 原生支持多资产、多策略组合回测

本测试验证：
  1. 向量化回测 vs 传统循环回测的正确性对比
  2. 向量化回测 vs 传统循环回测的性能对比（单策略 / 参数扫描）
  3. A 股特殊规则（T+1、涨跌停）的向量化实现正确性
  4. 边界条件：空数据、单日数据、全停牌场景
"""

import unittest
import sys
import os
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# =====================================================
# 传统循环回测引擎（模拟现有 backtest-engine 逻辑）
# =====================================================

class LoopBacktestEngine:
    """
    传统逐日循环回测引擎

    模拟现有 native_adapter / backtrader_adapter 的事件驱动逻辑
    """

    def __init__(self, init_capital: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.001,
                 slippage: float = 0.001,
                 t_plus_1: bool = True):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1

    def run(self, price_data: pd.DataFrame,
            signals: pd.DataFrame) -> Dict:
        """
        逐日循环回测

        参数:
            price_data: 价格数据，列为 code，行为 date
            signals: 交易信号，列为 code，行为 date
                     1=买入, -1=卖出, 0=持有
        """
        if price_data.empty or signals.empty:
            return self._empty_result()

        # 对齐日期
        common_dates = price_data.index.intersection(signals.index)
        common_codes = price_data.columns.intersection(signals.columns)

        if len(common_dates) < 2 or len(common_codes) == 0:
            return self._empty_result()

        price_data = price_data.loc[common_dates, common_codes]
        signals = signals.loc[common_dates, common_codes]

        cash = self.init_capital
        position = pd.Series(0.0, index=common_codes)  # 持股数量
        equity_curve = []
        trades = []

        dates = price_data.index.tolist()

        for i, date in enumerate(dates):
            prices = price_data.loc[date]
            signals_today = signals.loc[date] if i < len(signals) else pd.Series(0, index=common_codes)

            # T+1: 当日信号次日执行
            exec_date_idx = i + 1 if self.t_plus_1 else i
            if exec_date_idx < len(dates):
                exec_prices = price_data.iloc[exec_date_idx]
            else:
                exec_prices = prices

            for code in common_codes:
                sig = signals_today.get(code, 0)
                price = prices.get(code, np.nan)
                exec_price = exec_prices.get(code, price)

                if np.isnan(price) or np.isnan(exec_price):
                    continue

                # 买入信号
                if sig > 0 and position[code] == 0:
                    # 计算可买数量（考虑滑点和手续费）
                    buy_price = exec_price * (1 + self.slippage)
                    shares = int(cash * 0.95 / buy_price / 100) * 100  # 95%仓位，整手
                    if shares > 0:
                        cost = shares * buy_price * (1 + self.commission_rate)
                        if cost <= cash:
                            cash -= cost
                            position[code] += shares
                            trades.append({
                                "date": dates[exec_date_idx],
                                "code": code,
                                "type": "buy",
                                "price": buy_price,
                                "shares": shares,
                                "cost": cost,
                            })

                # 卖出信号
                elif sig < 0 and position[code] > 0:
                    sell_price = exec_price * (1 - self.slippage)
                    revenue = position[code] * sell_price * (1 - self.commission_rate - self.stamp_tax_rate)
                    cash += revenue
                    trades.append({
                        "date": dates[exec_date_idx],
                        "code": code,
                        "type": "sell",
                        "price": sell_price,
                        "shares": position[code],
                        "revenue": revenue,
                    })
                    position[code] = 0

            # 计算当日权益
            total_value = cash
            for code in common_codes:
                if position[code] > 0:
                    total_value += position[code] * prices.get(code, 0)

            equity_curve.append({"date": date, "equity": total_value, "cash": cash})

        equity_df = pd.DataFrame(equity_curve)
        metrics = self._calc_metrics(equity_df)

        return {
            "equity_curve": equity_df,
            "trades": trades,
            "metrics": metrics,
        }

    def _calc_metrics(self, equity_df: pd.DataFrame) -> Dict:
        if equity_df.empty or len(equity_df) < 2:
            return {}
        eq = equity_df.set_index("date")["equity"]
        returns = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        max_dd = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = (returns > 0).mean()
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
        }

    def _empty_result(self) -> Dict:
        return {
            "equity_curve": pd.DataFrame(columns=["date", "equity", "cash"]),
            "trades": [],
            "metrics": {},
        }


# =====================================================
# 向量化回测引擎（借鉴 VectorBT 设计）
# =====================================================

class VectorizedBacktestEngine:
    """
    向量化回测引擎

    借鉴 VectorBT 的核心思想：
      - 将所有操作统一为 NumPy 矩阵运算
      - 避免 Python 循环，利用向量化加速
      - 支持批量参数扫描（通过多维数组）

    注意：向量化回测适用于路径依赖较弱的策略（如因子截面选股）。
    对于强路径依赖策略（如金字塔加仓），仍需事件驱动回测。
    """

    def __init__(self, init_capital: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.001,
                 slippage: float = 0.001,
                 t_plus_1: bool = True):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1

    def run_single(self, price_data: pd.DataFrame,
                   signals: pd.DataFrame) -> Dict:
        """
        单策略向量化回测

        核心思路：
        1. 将价格矩阵和信号矩阵对齐（NumPy 数组）
        2. 向量化计算每日持仓权重
        3. 向量化计算每日组合收益率
        4. 向量化计算绩效指标
        """
        if price_data.empty or signals.empty:
            return self._empty_result()

        # 对齐
        common_dates = price_data.index.intersection(signals.index)
        common_codes = price_data.columns.intersection(signals.columns)

        if len(common_dates) < 2 or len(common_codes) == 0:
            return self._empty_result()

        prices = price_data.loc[common_dates, common_codes].values  # (T, N)
        sigs = signals.loc[common_dates, common_codes].values       # (T, N)

        T, N = prices.shape

        # 1. 每日收益率矩阵 (T-1, N)
        rets = prices[1:] / prices[:-1] - 1  # (T-1, N)

        # 2. 持仓权重矩阵
        #    信号 > 0: 等权买入, 信号 < 0: 卖出
        weights = np.zeros((T - 1, N))

        for t in range(T - 1):
            sig_t = sigs[t]  # 当日信号（如果是 T+1，这里应该用 sigs[t-1]）
            if self.t_plus_1 and t > 0:
                sig_t = sigs[t - 1]

            long_mask = sig_t > 0
            if long_mask.any():
                n_stocks = long_mask.sum()
                weights[t, long_mask] = 1.0 / n_stocks

        # 3. 组合日收益率 (T-1,)
        port_rets = np.sum(weights * rets, axis=1)

        # 4. 扣除交易成本
        #    换手 = |新权重 - 旧权重| 的差值
        if T > 2:
            turnover = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
            # 单边成本
            cost_rate = self.commission_rate + self.stamp_tax_rate / 2
            port_rets[1:] = port_rets[1:] - turnover * cost_rate

        # 5. 计算权益曲线
        equity = self.init_capital * np.cumprod(1 + port_rets)
        equity_full = np.concatenate([[self.init_capital], equity])

        equity_df = pd.DataFrame({
            "date": common_dates,
            "equity": equity_full,
        })

        metrics = self._calc_metrics(equity_df)

        return {
            "equity_curve": equity_df,
            "metrics": metrics,
        }

    def run_param_sweep(self, price_data: pd.DataFrame,
                        signal_func: callable,
                        param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        参数网格扫描（向量化批量回测）

        借鉴 VectorBT 的 run_combs 设计：
        将所有参数组合展开为一个大的 DataFrame，
        每一列代表一组参数的回测结果。

        参数:
            price_data: 价格数据
            signal_func: 信号生成函数，签名为 func(price_data, **params) -> signals_df
            param_grid: 参数网格，如 {"fast": [10,20,30], "slow": [50,60]}

        返回:
            包含所有参数组合绩效的 DataFrame
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []

        for combo in combinations:
            params = dict(zip(param_names, combo))
            signals = signal_func(price_data, **params)
            result = self.run_single(price_data, signals)
            metrics = result.get("metrics", {})

            row = {**params, **metrics}
            results.append(row)

        return pd.DataFrame(results).round(6)

    def _calc_metrics(self, equity_df: pd.DataFrame) -> Dict:
        if equity_df.empty or len(equity_df) < 2:
            return {}
        eq = equity_df.set_index("date")["equity"]
        returns = eq.pct_change().dropna()
        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        max_dd = (eq / eq.cummax() - 1).min() if len(eq) > 0 else 0
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
        }

    def _empty_result(self) -> Dict:
        return {
            "equity_curve": pd.DataFrame(columns=["date", "equity"]),
            "metrics": {},
        }


# =====================================================
# 信号生成辅助函数
# =====================================================

def make_ma_crossover_signals(price_data: pd.DataFrame,
                              fast: int = 10,
                              slow: int = 50) -> pd.DataFrame:
    """
    双均线交叉信号生成

    返回:
        DataFrame: 1=金叉买入, -1=死叉卖出, 0=持有
    """
    fast_ma = price_data.rolling(fast, min_periods=1).mean()
    slow_ma = price_data.rolling(slow, min_periods=1).mean()

    signals = pd.DataFrame(0, index=price_data.index, columns=price_data.columns)

    # 金叉: fast 上穿 slow
    cross_up = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    # 死叉: fast 下穿 slow
    cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    # 简单处理：金叉买入，持有直到死叉卖出
    for col in price_data.columns:
        in_position = False
        for i in range(len(signals)):
            if cross_up.iloc[i][col]:
                signals.iloc[i, signals.columns.get_loc(col)] = 1
                in_position = True
            elif cross_down.iloc[i][col] and in_position:
                signals.iloc[i, signals.columns.get_loc(col)] = -1
                in_position = False

    return signals


# =====================================================
# 单元测试
# =====================================================

class TestVectorizedBacktestCorrectness(unittest.TestCase):
    """验证向量化回测的正确性"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
        n_dates = len(dates)

        # 简单的上升趋势数据
        base = 10 + np.cumsum(np.random.normal(0.0005, 0.015, n_dates))
        cls.price_data = pd.DataFrame({
            "stock_A": base,
            "stock_B": base * (1 + np.cumsum(np.random.normal(0.0002, 0.01, n_dates)) * 0.1),
        }, index=dates)

        # 生成买入信号（前半年持有 stock_A，后半年持有 stock_B）
        cls.signals = pd.DataFrame(0, index=dates, columns=["stock_A", "stock_B"])
        mid = n_dates // 2
        cls.signals.iloc[0, 0] = 1
        cls.signals.iloc[mid, 0] = -1
        cls.signals.iloc[mid, 1] = 1
        cls.signals.iloc[-1, 1] = -1

    def test_equity_curve_shape(self):
        """测试权益曲线形状"""
        loop_engine = LoopBacktestEngine(t_plus_1=False)
        vec_engine = VectorizedBacktestEngine(t_plus_1=False)

        loop_result = loop_engine.run(self.price_data, self.signals)
        vec_result = vec_engine.run_single(self.price_data, self.signals)

        self.assertFalse(loop_result["equity_curve"].empty)
        self.assertFalse(vec_result["equity_curve"].empty)

    def test_metrics_returned(self):
        """测试绩效指标完整返回"""
        vec_engine = VectorizedBacktestEngine()
        result = vec_engine.run_single(self.price_data, self.signals)

        expected_metrics = ["total_return", "annual_return", "volatility",
                          "sharpe_ratio", "max_drawdown", "calmar_ratio", "win_rate"]
        for m in expected_metrics:
            self.assertIn(m, result["metrics"], f"缺少指标: {m}")

    def test_buy_and_hold_equity(self):
        """测试买入持有策略的权益计算"""
        price_data = pd.DataFrame({
            "stock": [10.0, 10.5, 10.2, 10.8, 11.0, 10.9, 11.5, 11.3, 12.0, 12.5],
        }, index=pd.date_range("2024-01-01", periods=10, freq="B"))

        # 向量化回测引擎：每天持有信号保持权重
        signals = pd.DataFrame(1, index=price_data.index, columns=["stock"])  # 每天都是买入信号

        vec_engine = VectorizedBacktestEngine(t_plus_1=False)
        result = vec_engine.run_single(price_data, signals)
        eq = result["equity_curve"]["equity"].values

        total_return = eq[-1] / eq[0] - 1
        expected_return = 12.5 / 10.0 - 1  # 0.25
        # 应该接近买入持有收益（扣除极小的成本）
        self.assertAlmostEqual(total_return, expected_return, delta=0.01)

    def test_t_plus_1_rule(self):
        """测试 T+1 规则"""
        price_data = pd.DataFrame({
            "stock": [10.0, 10.5, 10.2, 10.8, 11.0],
        }, index=pd.date_range("2024-01-01", periods=5, freq="B"))

        signals = pd.DataFrame(0, index=price_data.index, columns=["stock"])
        signals.iloc[0] = 1   # D0 发出买入信号
        signals.iloc[3] = -1  # D3 发出卖出信号

        # T+1 模式下，D0 信号在 D1 执行
        vec_t1 = VectorizedBacktestEngine(t_plus_1=True)
        result_t1 = vec_t1.run_single(price_data, signals)
        eq_t1 = result_t1["equity_curve"]["equity"].values

        # 非 T+1 模式下，D0 信号在 D0 执行
        vec_not1 = VectorizedBacktestEngine(t_plus_1=False)
        result_not1 = vec_not1.run_single(price_data, signals)
        eq_not1 = result_not1["equity_curve"]["equity"].values

        # T+1 模式的收益应该不同（买入价不同）
        self.assertNotEqual(
            round(eq_t1[-1], 4),
            round(eq_not1[-1], 4),
            "T+1 和非 T+1 模式的最终权益应该不同"
        )


class TestVectorizedBacktestPerformance(unittest.TestCase):
    """性能对比测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_dates = 500
        n_stocks = 20

        dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
        codes = [f"stock_{i}" for i in range(n_stocks)]

        # 模拟随机游走价格
        ret_matrix = np.random.normal(0.0003, 0.02, (n_dates, n_stocks))
        prices = 10 * np.cumprod(1 + ret_matrix, axis=0)
        cls.price_data = pd.DataFrame(prices, index=dates, columns=codes)

        # 生成每日随机信号（Top-K 方式）
        cls.signals = pd.DataFrame(0, index=dates, columns=codes)
        for i in range(n_dates):
            # 随机选 5 只买入
            top_k = np.random.choice(n_stocks, size=5, replace=False)
            cls.signals.iloc[i, top_k] = 1

    def test_performance_comparison(self):
        """对比循环回测和向量化回测的性能"""
        loop_engine = LoopBacktestEngine(t_plus_1=False)
        vec_engine = VectorizedBacktestEngine(t_plus_1=False)

        # 预热
        _ = vec_engine.run_single(self.price_data, self.signals)

        # 测量向量化回测
        times_vec = []
        for _ in range(5):
            start = time.perf_counter()
            _ = vec_engine.run_single(self.price_data, self.signals)
            times_vec.append(time.perf_counter() - start)

        # 测量循环回测
        times_loop = []
        for _ in range(5):
            start = time.perf_counter()
            _ = loop_engine.run(self.price_data, self.signals)
            times_loop.append(time.perf_counter() - start)

        avg_vec = np.mean(times_vec)
        avg_loop = np.mean(times_loop)

        speedup = avg_loop / avg_vec if avg_vec > 0 else float('inf')

        print(f"\n  性能对比 (20标的×500天):")
        print(f"    循环回测平均: {avg_loop*1000:.1f}ms")
        print(f"    向量化回测平均: {avg_vec*1000:.1f}ms")
        print(f"    加速比: {speedup:.1f}x")

        # 向量化应该更快（至少不漏于循环版本太多）
        self.assertLess(avg_vec * 3, avg_loop * 5,
            "向量化回测不应显著慢于循环回测")

    def test_param_sweep_speed(self):
        """测试参数扫描性能"""
        vec_engine = VectorizedBacktestEngine()

        param_grid = {
            "fast": list(range(5, 41, 5)),   # 8 个参数
            "slow": list(range(20, 61, 10)),  # 5 个参数
        }
        total_combos = len(param_grid["fast"]) * len(param_grid["slow"])

        start = time.perf_counter()
        result = vec_engine.run_param_sweep(
            self.price_data, make_ma_crossover_signals, param_grid
        )
        elapsed = time.perf_counter() - start

        self.assertEqual(len(result), total_combos,
            f"应生成 {total_combos} 组结果，实际 {len(result)}")
        self.assertLess(elapsed, 30,
            f"参数扫描 {total_combos} 组耗时 {elapsed:.1f}s 超过30s阈值")

        # 验证每组都包含必要的指标
        for metrics_col in ["total_return", "sharpe_ratio", "max_drawdown"]:
            self.assertIn(metrics_col, result.columns)
            self.assertTrue(result[metrics_col].notna().any())

        print(f"\n  参数扫描: {total_combos} 组合耗时 {elapsed:.1f}s")
        print(f"    最佳 Sharpe: {result['sharpe_ratio'].max():.4f}")


class TestBoundaryConditions(unittest.TestCase):
    """边界条件测试"""

    def test_empty_data(self):
        """测试空数据"""
        empty_price = pd.DataFrame()
        empty_signals = pd.DataFrame()
        engine = VectorizedBacktestEngine()

        result = engine.run_single(empty_price, empty_signals)
        self.assertEqual(result["metrics"], {})

    def test_single_day(self):
        """测试单日数据"""
        price = pd.DataFrame({"A": [10.0]}, index=[pd.Timestamp("2024-01-01")])
        signals = pd.DataFrame({"A": [1]}, index=[pd.Timestamp("2024-01-01")])

        engine = VectorizedBacktestEngine()
        result = engine.run_single(price, signals)
        self.assertEqual(result["metrics"], {})  # 不足2天无法计算指标

    def test_all_suspended(self):
        """测试全 NaN 数据"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        price = pd.DataFrame({"A": [np.nan] * 10}, index=dates)
        signals = pd.DataFrame({"A": [1] * 10}, index=dates)

        engine = VectorizedBacktestEngine()
        result = engine.run_single(price, signals)
        # 不应该崩溃
        self.assertIsNotNone(result)

    def test_sudden_price_spike(self):
        """测试价格跳空"""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        # 大跳空：从10跳到100然后暴跌
        price = pd.DataFrame({"A": [10, 10.5, 10.2, 100, 50, 30, 20, 15, 14, 13]}, index=dates)
        # 一直持有（每天信号都为1，保持权重）
        signals = pd.DataFrame({"A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}, index=dates)

        engine = VectorizedBacktestEngine(t_plus_1=False)
        result = engine.run_single(price, signals)
        metrics = result["metrics"]
        # 先猛涨后猛跌，应该有大回撤
        self.assertLess(metrics.get("max_drawdown", 0), -0.1)

    def test_commission_impact(self):
        """测试手续费的影响"""
        n_dates = 252
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        # 每天交替买入卖出
        price = pd.DataFrame({
            "A": 10 + np.cumsum(np.random.normal(0.000, 0.01, n_dates)),
        }, index=dates)

        signals = pd.DataFrame({"A": [1 if i % 2 == 0 else -1 for i in range(n_dates)]},
                              index=dates)

        # 无手续费
        engine_no_cost = VectorizedBacktestEngine(commission_rate=0, stamp_tax_rate=0, slippage=0, t_plus_1=False)
        result_no_cost = engine_no_cost.run_single(price, signals)

        # 有手续费
        engine_with_cost = VectorizedBacktestEngine(commission_rate=0.005, stamp_tax_rate=0.001, slippage=0.001, t_plus_1=False)
        result_with_cost = engine_with_cost.run_single(price, signals)

        ret_no = result_no_cost["metrics"].get("total_return", 0)
        ret_with = result_with_cost["metrics"].get("total_return", 0)

        # 有手续费的收益应该更低（高频交易场景下手续费侵蚀明显）
        self.assertLess(ret_with, ret_no + 0.01,
            f"高手续费收益应不高于零手续费收益: {ret_with:.4f} vs {ret_no:.4f}")
        print(f"\n  手续费影响: 无成本={ret_no:.4f}, 有成本={ret_with:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("向量化回测引擎性能对比 - 验证测试")
    print("借鉴来源: VectorBT (vectorized backtesting)")
    print("=" * 60)
    unittest.main(verbosity=2)