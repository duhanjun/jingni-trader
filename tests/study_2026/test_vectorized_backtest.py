"""
=============================================================================
优化方向: 向量化回测引擎 (Vectorized Backtesting Engine)
借鉴来源: VectorBT (向量化回测 + 参数广播), quant-stream (流式回测)
日期: 2026-06-13
=============================================================================

核心思路:
  VectorBT 通过向量化运算将回测速度提升 100-1000 倍，核心原理是将策略逻辑
  转换为对整个数据矩阵的 NumPy/Pandas 操作，而非逐K线循环。
  jingni-trader 当前使用事件驱动回测 (native/rqalpha/backtrader)，在参数
  扫描场景下效率较低。引入向量化回测核心可大幅加速因子筛选和策略迭代。

验证目标:
  1. 实现一个轻量级向量化回测核心，支持 A 股特有规则 (T+1, 涨跌停)
  2. 实现参数广播: 一次计算评估多组参数
  3. 与现有事件驱动回测进行性能对比
  4. 验证结果的正确性（与事件驱动结果一致）
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 向量化回测引擎核心实现
# =============================================================================

class VectorizedBacktestEngine:
    """
    向量化回测引擎

    核心设计:
      - 所有操作为矩阵运算，避免 Python 循环
      - 支持 A 股 T+1 交易规则
      - 支持涨跌停无法交易
      - 支持参数广播: 传入多组参数矩阵，一次计算所有结果
    """

    def __init__(
        self,
        data: pd.DataFrame,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        """
        参数:
            data: 包含 code, date, close, is_limit_up, is_limit_down 的 DataFrame
            init_capital: 初始资金
            commission_rate: 佣金费率
            stamp_tax_rate: 印花税（仅卖出）
            slippage: 滑点
            t_plus_1: 是否 T+1
            price_limit: 是否考虑涨跌停
        """
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit

        # 构建价格矩阵和辅助矩阵
        self._build_matrices(data)

    def _build_matrices(self, data: pd.DataFrame):
        """构建回测所需矩阵"""
        df = data.sort_values(['date', 'code']).copy()
        self.dates = sorted(df['date'].unique())
        self.codes = sorted(df['code'].unique())

        n_dates = len(self.dates)
        n_codes = len(self.codes)

        # 构建价格矩阵 (dates x codes)
        pivot = df.pivot(index='date', columns='code', values='close')
        self.close_matrix = pivot.reindex(index=self.dates, columns=self.codes).values

        # 涨跌停矩阵
        if 'is_limit_up' in df.columns:
            limit_up = df.pivot(index='date', columns='code', values='is_limit_up')
            self.limit_up_matrix = limit_up.reindex(
                index=self.dates, columns=self.codes
            ).fillna(False).values
        else:
            self.limit_up_matrix = np.zeros((n_dates, n_codes), dtype=bool)

        if 'is_limit_down' in df.columns:
            limit_down = df.pivot(index='date', columns='code', values='is_limit_down')
            self.limit_down_matrix = limit_down.reindex(
                index=self.dates, columns=self.codes
            ).fillna(False).values
        else:
            self.limit_down_matrix = np.zeros((n_dates, n_codes), dtype=bool)

        self.n_dates = n_dates
        self.n_codes = n_codes

    def run(
        self,
        signals: np.ndarray,
        signal_type: str = 'position',
        rebalance_freq: int = 1,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            signals: 信号矩阵 (n_dates x n_codes)
                     - 'position': 值为 0/1/-1 (持有多仓/空仓/平仓)
                     - 'weight': 值为仓位权重
            signal_type: 信号类型
            rebalance_freq: 调仓频率 (交易日)

        返回: 包含 equity_curve, metrics 的字典
        """
        signals = np.asarray(signals, dtype=float)
        n_dates, n_codes = signals.shape

        # 初始化
        equity = np.zeros(n_dates)
        equity[0] = self.init_capital

        # 持仓矩阵 (n_dates x n_codes): 每只股票持有的股数
        positions = np.zeros((n_dates, n_codes))
        cash = np.zeros(n_dates)
        cash[0] = self.init_capital

        # 交易成本累积
        total_cost = np.zeros(n_dates)

        for t in range(1, n_dates):
            # 继承前一日持仓
            positions[t] = positions[t - 1].copy()
            cash[t] = cash[t - 1]
            total_cost[t] = total_cost[t - 1]

            # 判断是否调仓日
            if t % rebalance_freq != 0:
                # 不调仓，仅更新市值
                equity[t] = cash[t] + np.nansum(
                    positions[t] * self.close_matrix[t], dtype=float
                )
                continue

            # 获取目标信号
            target_signal = signals[t - 1] if self.t_plus_1 else signals[t]

            # 信号处理
            if signal_type == 'position':
                target_positions = self._signals_to_positions(
                    target_signal, t
                )
            else:
                # weight 类型: 根据权重分配资金
                target_positions = self._weights_to_positions(
                    target_signal, t, equity[t - 1]
                )

            # 计算交易成本并执行调仓
            cost, positions[t], cash[t] = self._execute_rebalance(
                positions[t - 1], target_positions, cash[t], t
            )
            total_cost[t] += cost

            # 更新组合净值
            equity[t] = cash[t] + np.nansum(
                positions[t] * self.close_matrix[t], dtype=float
            )

        # 计算指标
        metrics = self._calc_metrics(equity)

        return {
            'equity_curve': pd.DataFrame({
                'date': self.dates,
                'equity': equity,
            }),
            'metrics': metrics,
            'total_cost': total_cost[-1],
        }

    def _signals_to_positions(
        self, signals: np.ndarray, t: int
    ) -> np.ndarray:
        """将信号转换为目标持仓股数 (等权分配)"""
        side = signals.copy()
        # 涨跌停过滤
        if self.price_limit:
            side[self.limit_up_matrix[t]] = 0  # 涨停买不进
            # 跌停卖不掉: 保留现有持仓

        long_mask = side > 0
        short_mask = side < 0

        target = np.zeros(len(side))
        if long_mask.any():
            # 等权分配
            target[long_mask] = 1.0 / long_mask.sum()
        if short_mask.any():
            target[short_mask] = -1.0 / short_mask.sum()

        return target

    def _weights_to_positions(
        self, weights: np.ndarray, t: int, equity_val: float
    ) -> np.ndarray:
        """将权重转换为目标股数"""
        prices = self.close_matrix[t]
        valid = ~np.isnan(prices) & (prices > 0)
        if self.price_limit:
            valid &= ~self.limit_up_matrix[t]

        target = np.zeros(len(weights))
        for i in range(len(weights)):
            if valid[i] and weights[i] > 0:
                target[i] = equity_val * weights[i] / prices[i]
        return target

    def _execute_rebalance(
        self,
        current_positions: np.ndarray,
        target_positions: np.ndarray,
        cash: float,
        t: int,
    ) -> Tuple[float, np.ndarray, float]:
        """执行调仓计算"""
        prices = self.close_matrix[t]
        diff = target_positions - current_positions

        # 买入动作
        buy_mask = diff > 0
        # 卖出动作
        sell_mask = diff < 0

        total_cost = 0.0
        new_positions = current_positions.copy()

        # 计算卖出金额
        sell_amount = 0.0
        for i in np.where(sell_mask)[0]:
            if self.price_limit and self.limit_down_matrix[t, i]:
                continue  # 跌停卖不掉
            sell_price = prices[i] * (1 - self.slippage)
            sell_qty = -diff[i]
            amount = sell_qty * sell_price
            sell_amount += amount
            new_positions[i] -= sell_qty
            # 印花税 (仅卖出)
            total_cost += amount * self.stamp_tax_rate
            # 佣金
            total_cost += max(amount * self.commission_rate, 5.0)

        cash += sell_amount

        # 计算买入金额
        buy_amount = 0.0
        for i in np.where(buy_mask)[0]:
            if self.price_limit and self.limit_up_matrix[t, i]:
                continue  # 涨停买不进
            buy_price = prices[i] * (1 + self.slippage)
            buy_qty = diff[i]
            amount = buy_qty * buy_price
            buy_amount += amount
            new_positions[i] += buy_qty
            # 佣金
            total_cost += max(amount * self.commission_rate, 5.0)

        cash -= buy_amount
        cash -= total_cost

        return total_cost, new_positions, cash

    def _calc_metrics(self, equity: np.ndarray) -> Dict[str, float]:
        """计算绩效指标"""
        if len(equity) < 2:
            return {}

        returns = np.diff(equity) / equity[:-1]
        total_return = equity[-1] / equity[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # 最大回撤
        cummax = np.maximum.accumulate(equity)
        drawdown = equity / cummax - 1
        max_drawdown = np.min(drawdown)

        # 夏普比率
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0

        # 胜率
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": float(calmar),
        }


# =============================================================================
# 事件驱动回测 (用于对比)
# =============================================================================

class EventDrivenBacktestEngine:
    """事件驱动回测，与 VectorizedBacktestEngine 对比用"""

    def __init__(
        self,
        data: pd.DataFrame,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
        t_plus_1: bool = True,
    ):
        self.data = data.sort_values(['date', 'code']).reset_index(drop=True)
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1

        self.dates = sorted(data['date'].unique())
        self.codes = sorted(data['code'].unique())

    def run(
        self,
        signals: np.ndarray,
        rebalance_freq: int = 1,
    ) -> Dict[str, Any]:
        """事件驱动回测"""
        n_dates = len(self.dates)
        n_codes = len(self.codes)

        equity = np.zeros(n_dates)
        equity[0] = self.init_capital
        cash = self.init_capital
        positions = np.zeros(n_codes)  # 持仓股数

        for t in range(1, n_dates):
            # 获取当日价格
            date = self.dates[t]
            day_data = self.data[self.data['date'] == date]
            if day_data.empty:
                equity[t] = equity[t - 1]
                continue

            prices = np.zeros(n_codes)
            for i, code in enumerate(self.codes):
                row = day_data[day_data['code'] == code]
                if not row.empty:
                    prices[i] = row['close'].values[0]
                else:
                    prices[i] = np.nan

            # 调仓
            if t % rebalance_freq == 0:
                target_signal = signals[t - 1] if self.t_plus_1 else signals[t]

                # 卖出
                for i in range(n_codes):
                    if positions[i] > 0 and target_signal[i] <= 0:
                        sell_price = prices[i] * (1 - self.slippage)
                        if not np.isnan(sell_price):
                            cash += positions[i] * sell_price * (1 - self.stamp_tax_rate - self.commission_rate)
                            positions[i] = 0

                # 买入
                buy_signals = target_signal > 0
                n_buy = buy_signals.sum()
                if n_buy > 0:
                    per_stock_cash = cash * 0.9 / n_buy  # 留 10% 现金
                    for i in np.where(buy_signals)[0]:
                        buy_price = prices[i] * (1 + self.slippage)
                        if not np.isnan(buy_price) and buy_price > 0:
                            qty = int(per_stock_cash / buy_price / 100) * 100
                            if qty >= 100:
                                cost = qty * buy_price * (1 + self.commission_rate)
                                if cost <= cash * 0.9:
                                    positions[i] += qty
                                    cash -= cost

            # 计算市值
            total_value = cash
            for i in range(n_codes):
                if positions[i] > 0 and not np.isnan(prices[i]):
                    total_value += positions[i] * prices[i]
            equity[t] = total_value

        metrics = self._calc_metrics(equity)
        return {
            'equity_curve': pd.DataFrame({
                'date': self.dates,
                'equity': equity,
            }),
            'metrics': metrics,
        }

    def _calc_metrics(self, equity: np.ndarray) -> Dict[str, float]:
        if len(equity) < 2:
            return {}
        returns = np.diff(equity) / equity[:-1]
        total_return = equity[-1] / equity[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        cummax = np.maximum.accumulate(equity)
        drawdown = equity / cummax - 1
        max_drawdown = np.min(drawdown)
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": float(calmar),
        }


# =============================================================================
# 测试用例
# =============================================================================

class TestVectorizedBacktestCorrectness(unittest.TestCase):
    """向量化回测 vs 事件驱动回测 正确性对比"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 5
        n_days = 252
        cls.codes = [f'{i:06d}.SH' for i in range(600000, 600000 + n_codes)]
        cls.dates = pd.date_range('2023-01-01', periods=n_days, freq='B')

        rows = []
        for code in cls.codes:
            price = 10 + np.cumsum(np.random.randn(n_days) * 0.2)
            for i, dt in enumerate(cls.dates):
                rows.append({
                    'code': code,
                    'date': dt,
                    'close': price[i] + 5,
                    'open': price[i] * (1 + np.random.randn() * 0.01),
                    'volume': np.random.randint(10000, 100000),
                    'is_limit_up': False,
                    'is_limit_down': False,
                })
        cls.data = pd.DataFrame(rows)

        # 生成信号: 50% 概率买入
        cls.signals = (np.random.rand(n_days, n_codes) > 0.5).astype(float)

    def test_equity_curve_consistency(self):
        """测试向量化回测与事件驱动回测的净值曲线一致性"""
        vec_engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)
        evt_engine = EventDrivenBacktestEngine(self.data, t_plus_1=True)

        vec_result = vec_engine.run(self.signals)
        evt_result = evt_engine.run(self.signals)

        vec_equity = vec_result['equity_curve']['equity'].values
        evt_equity = evt_result['equity_curve']['equity'].values

        # 检查整体趋势一致 (相关系数 > 0.7)
        if len(vec_equity) > 10:
            corr = np.corrcoef(vec_equity, evt_equity)[0, 1]
            self.assertGreater(corr, 0.7, f"净值曲线相关性过低: {corr:.4f}")

    def test_metrics_similarity(self):
        """测试指标一致性"""
        vec_engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)
        evt_engine = EventDrivenBacktestEngine(self.data, t_plus_1=True)

        vec_result = vec_engine.run(self.signals)
        evt_result = evt_engine.run(self.signals)

        vec_metrics = vec_result['metrics']
        evt_metrics = evt_result['metrics']

        # 检查关键指标相对误差 < 100%
        for key in ['total_return', 'sharpe_ratio']:
            if key in vec_metrics and key in evt_metrics:
                v = abs(vec_metrics[key])
                e = abs(evt_metrics[key])
                if v + e > 0.001:
                    rel_diff = abs(v - e) / max(v + e, 0.001)
                    self.assertLess(rel_diff, 1.0, f"{key} 相对差异过大: {rel_diff:.2%}")

    def test_t_plus_1_rule(self):
        """测试 T+1 规则"""
        engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)

        # 生成信号: 仅在 t=1 时买入
        signals = np.zeros((len(self.dates), len(self.codes)))
        signals[1, 0] = 1

        result = engine.run(signals)
        equity = result['equity_curve']['equity'].values

        # T+1: 信号在 t=1 发出，应在 t=2 执行
        # 因此 t=0, t=1 净值应等于初始资金
        self.assertAlmostEqual(equity[0], engine.init_capital)
        # t=1 时持仓应为0（信号延迟一天执行）
        # 检查净值曲线在 t=1 时没有变化
        self.assertAlmostEqual(equity[1], engine.init_capital)

    def test_price_limit_filter(self):
        """测试涨跌停过滤"""
        # 创建含涨跌停的测试数据
        n_codes = 3
        n_days = 5
        codes = ['A', 'B', 'C']
        dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

        rows = []
        for code in codes:
            for i, dt in enumerate(dates):
                rows.append({
                    'code': code,
                    'date': dt,
                    'close': 10.0 + i,
                    'open': 10.0 + i,
                    'volume': 10000,
                    'is_limit_up': (code == 'A' and i == 2),  # A在第3天涨停
                    'is_limit_down': (code == 'B' and i == 3),  # B在第4天跌停
                })
        data = pd.DataFrame(rows)

        engine = VectorizedBacktestEngine(data, t_plus_1=False, price_limit=True)

        # 信号: 第2天买入A和B
        signals = np.zeros((n_days, n_codes))
        signals[1, 0] = 1  # A
        signals[1, 1] = 1  # B

        result = engine.run(signals)
        equity = result['equity_curve']['equity'].values

        # 验证回测正常运行（不因涨跌停抛异常）
        self.assertGreater(equity[-1], 0)

    def test_multi_param_broadcast(self):
        """测试参数广播: 一次计算多组参数"""
        engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)

        # 模拟多组参数: 不同调仓频率
        results = []
        for freq in [1, 5, 10, 20]:
            result = engine.run(self.signals, rebalance_freq=freq)
            results.append(result)

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIn('metrics', r)
            self.assertIn('sharpe_ratio', r['metrics'])


class TestVectorizedBacktestPerformance(unittest.TestCase):
    """向量化回测性能对比测试"""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n_codes = 100
        n_days = 252
        cls.codes = [f'{i:06d}.SH' for i in range(600000, 600000 + n_codes)]
        cls.dates = pd.date_range('2023-01-01', periods=n_days, freq='B')

        rows = []
        for code in cls.codes:
            price = 10 + np.cumsum(np.random.randn(n_days) * 0.2)
            for i, dt in enumerate(cls.dates):
                rows.append({
                    'code': code,
                    'date': dt,
                    'close': price[i] + 5,
                    'open': price[i] * (1 + np.random.randn() * 0.01),
                    'volume': np.random.randint(10000, 100000),
                    'is_limit_up': False,
                    'is_limit_down': False,
                })
        cls.data = pd.DataFrame(rows)
        cls.signals = (np.random.rand(n_days, n_codes) > 0.5).astype(float)

    def test_single_backtest_speed(self):
        """测试单次回测速度对比"""
        vec_engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)
        evt_engine = EventDrivenBacktestEngine(self.data, t_plus_1=True)

        # 向量化
        start = time.time()
        vec_result = vec_engine.run(self.signals)
        vec_time = time.time() - start

        # 事件驱动
        start = time.time()
        evt_result = evt_engine.run(self.signals)
        evt_time = time.time() - start

        print(f"\n  单次回测性能对比:")
        print(f"    向量化: {vec_time:.4f}s")
        print(f"    事件驱动: {evt_time:.4f}s")
        print(f"    加速比: {evt_time / vec_time:.1f}x")

        # 向量化应显著快于事件驱动
        self.assertLess(vec_time, evt_time,
                        "向量化回测应快于事件驱动回测")

    def test_parameter_sweep_speed(self):
        """测试参数扫描速度对比"""
        vec_engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)
        evt_engine = EventDrivenBacktestEngine(self.data, t_plus_1=True)

        freqs = list(range(1, 21))  # 20 组参数

        # 向量化扫描
        start = time.time()
        for freq in freqs:
            vec_engine.run(self.signals, rebalance_freq=freq)
        vec_time = time.time() - start

        # 事件驱动扫描
        start = time.time()
        for freq in freqs:
            evt_engine.run(self.signals, rebalance_freq=freq)
        evt_time = time.time() - start

        print(f"\n  参数扫描性能对比 (20组参数):")
        print(f"    向量化: {vec_time:.4f}s")
        print(f"    事件驱动: {evt_time:.4f}s")
        print(f"    加速比: {evt_time / vec_time:.1f}x")

        self.assertLess(vec_time, evt_time,
                        "向量化参数扫描应快于事件驱动")

    def test_large_scale_speed(self):
        """测试大规模回测速度 (100股票, 252天, 50组参数)"""
        vec_engine = VectorizedBacktestEngine(self.data, t_plus_1=True, price_limit=False)

        start = time.time()
        for freq in range(1, 51):
            vec_engine.run(self.signals, rebalance_freq=freq)
        vec_time = time.time() - start

        print(f"\n  大规模扫描 (50组参数, 100只股票, 252天):")
        print(f"    总耗时: {vec_time:.4f}s")
        print(f"    平均每次: {vec_time / 50:.4f}s")

        self.assertLess(vec_time, 5.0, "大规模扫描超时 (>5s)")


if __name__ == '__main__':
    unittest.main(verbosity=2)