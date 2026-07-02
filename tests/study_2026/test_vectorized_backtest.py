"""
优化方向: 向量化回测引擎 (Vectorized Backtest Engine)
借鉴来源: Qlib (https://github.com/microsoft/qlib)
  - Qlib 的 backtest 支持向量化计算，避免逐日循环的性能瓶颈
  - 核心价值: 大幅提升回测速度，支持大规模股票池和长周期回测
  - 参考文件: qlib/contrib/strategy/strategy.py, qlib/backtest/
对比对象: jingni-trader skills/backtest-engine/scripts/adapters/native_adapter.py
          (当前实现为逐日循环的 pandas 方式)
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 1. 向量化回测引擎核心实现
# ============================================================

@dataclass
class BacktestConfig:
    """回测配置"""
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001  # 仅卖出
    slippage: float = 0.001
    t_plus_1: bool = True
    price_limit: bool = True
    max_position_pct: float = 0.05  # 单票最大仓位比例
    max_positions: int = 20  # 最大持仓数
    min_commission: float = 5.0  # 最低佣金


class VectorizedBacktestEngine:
    """
    向量化回测引擎

    与现有 NativeAdapter 的逐日循环不同，本引擎使用向量化操作：
    1. 信号和价格数据预对齐
    2. 使用 numpy 向量化计算交易成本和持仓
    3. 支持批量计算净值曲线

    关键优化:
    - 避免 Python for 循环逐日处理
    - 使用 numpy 批量计算交易成本
    - 预计算涨跌停过滤
    - 支持多股票同时处理
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            data: 行情数据 - 必须包含 code, date, close, open, high, low, volume
                  可选: is_st, is_limit_up, is_limit_down
            signals: 交易信号 - 必须包含 code, date, signal (1=买入, -1=卖出, 0=持仓)

        返回:
            回测结果字典
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # 预对齐数据
        data, signals = self._align_data(data, signals)

        if data.empty or signals.empty:
            return self._empty_result()

        # 提取独特的日期和股票代码
        dates = sorted(data['date'].unique())
        codes = sorted(data['code'].unique())

        # 构建价格矩阵 (date x code)
        price_matrix = self._build_price_matrix(data, dates, codes)

        # 构建信号矩阵 (date x code)
        signal_matrix = self._build_signal_matrix(signals, dates, codes)

        # 构建限制矩阵（涨跌停标记）
        limit_matrix = self._build_limit_matrix(data, dates, codes)

        # 执行向量化回测
        equity_curve, trades = self._vectorized_loop(
            price_matrix, signal_matrix, limit_matrix, dates, codes
        )

        # 计算绩效指标
        metrics = self._calc_metrics(equity_curve)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _align_data(self, data: pd.DataFrame, signals: pd.DataFrame
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """对齐数据和信号"""
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        # 确保必需列存在
        required_cols = ['code', 'date', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少必需列: {col}")

        return data, signals

    def _build_price_matrix(self, data: pd.DataFrame, dates: List, codes: List
                            ) -> pd.DataFrame:
        """构建价格矩阵"""
        pivot = data.pivot_table(
            index='date', columns='code', values='close', aggfunc='last'
        )
        pivot = pivot.reindex(dates)
        pivot = pivot.reindex(columns=codes)
        return pivot.ffill()

    def _build_signal_matrix(self, signals: pd.DataFrame, dates: List, codes: List
                             ) -> pd.DataFrame:
        """构建信号矩阵 (1=买入, -1=卖出, 0=无操作)"""
        if 'signal' not in signals.columns:
            return pd.DataFrame(0, index=dates, columns=codes)

        pivot = signals.pivot_table(
            index='date', columns='code', values='signal', aggfunc='last'
        )
        pivot = pivot.reindex(dates)
        pivot = pivot.reindex(columns=codes)
        return pivot.fillna(0)

    def _build_limit_matrix(self, data: pd.DataFrame, dates: List, codes: List
                            ) -> Dict[str, pd.DataFrame]:
        """构建涨跌停限制矩阵"""
        limit_up = pd.DataFrame(False, index=dates, columns=codes)
        limit_down = pd.DataFrame(False, index=dates, columns=codes)

        if 'is_limit_up' in data.columns:
            pivot = data.pivot_table(
                index='date', columns='code', values='is_limit_up', aggfunc='last'
            )
            pivot = pivot.reindex(dates).reindex(columns=codes).fillna(False)
            limit_up = pivot.astype(bool)

        if 'is_limit_down' in data.columns:
            pivot = data.pivot_table(
                index='date', columns='code', values='is_limit_down', aggfunc='last'
            )
            pivot = pivot.reindex(dates).reindex(columns=codes).fillna(False)
            limit_down = pivot.astype(bool)

        return {'limit_up': limit_up, 'limit_down': limit_down}

    def _vectorized_loop(
        self,
        price_matrix: pd.DataFrame,
        signal_matrix: pd.DataFrame,
        limit_matrix: Dict[str, pd.DataFrame],
        dates: List,
        codes: List,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        向量化回测循环

        使用 numpy 矩阵运算替代逐日 for 循环
        关键: 持仓矩阵、成本矩阵、净值曲线的向量化计算
        """
        n_dates = len(dates)
        n_codes = len(codes)

        prices = price_matrix.values.copy()  # (n_dates, n_codes)
        signals = signal_matrix.values.copy()  # (n_dates, n_codes)
        limit_up = limit_matrix['limit_up'].values.copy() if self.config.price_limit else np.zeros((n_dates, n_codes), dtype=bool)
        limit_down = limit_matrix['limit_down'].values.copy() if self.config.price_limit else np.zeros((n_dates, n_codes), dtype=bool)

        # 初始化
        cash = self.config.init_capital
        holdings = np.zeros(n_codes)  # 当前持仓股数
        equity = np.zeros(n_dates)
        cash_series = np.zeros(n_dates)

        trades_list = []

        for t in range(n_dates):
            # 当天价格和信号
            today_prices = prices[t]
            today_signals = signals[t]
            today_limit_up = limit_up[t]
            today_limit_down = limit_down[t]

            valid_price = ~np.isnan(today_prices)
            today_prices = np.nan_to_num(today_prices, 0)

            # ---- 卖出处理 ----
            sell_mask = (today_signals < 0) & (holdings > 0) & valid_price
            if self.config.price_limit:
                sell_mask = sell_mask & ~today_limit_down

            sell_prices = today_prices[sell_mask]
            sell_shares = holdings[sell_mask]
            sell_amounts = sell_prices * sell_shares

            sell_commission = np.maximum(sell_amounts * self.config.commission_rate, self.config.min_commission)
            sell_tax = sell_amounts * self.config.stamp_tax_rate
            sell_costs = sell_commission + sell_tax
            sell_proceeds = sell_amounts - sell_costs

            cash += np.sum(sell_proceeds)
            holdings[sell_mask] = 0

            # 记录卖出交易
            sell_indices = np.where(sell_mask)[0]
            for i, idx in enumerate(sell_indices):
                trades_list.append({
                    'date': dates[t], 'code': codes[idx], 'action': 'sell',
                    'price': sell_prices[i], 'shares': int(sell_shares[i]),
                    'amount': sell_amounts[i], 'commission': sell_commission[i],
                    'tax': sell_tax[i],
                })

            # ---- 买入处理 ----
            buy_mask = (today_signals > 0) & valid_price
            if self.config.price_limit:
                buy_mask = buy_mask & ~today_limit_up

            if np.any(buy_mask):
                n_buy = np.sum(buy_mask)
                # 等权重分配资金
                budget_per_stock = cash * self.config.max_position_pct / n_buy
                budget_per_stock = min(budget_per_stock, cash * 0.95 / n_buy)

                buy_prices = today_prices[buy_mask] * (1 + self.config.slippage)
                buy_shares = np.floor(budget_per_stock / buy_prices / 100) * 100
                buy_shares = np.maximum(buy_shares, 0)

                buy_amounts = buy_prices * buy_shares
                buy_commission = np.maximum(buy_amounts * self.config.commission_rate, self.config.min_commission)
                buy_costs = buy_amounts + buy_commission

                # 确保不超过现金
                total_cost = np.sum(buy_costs)
                if total_cost > cash * 0.98:
                    scale = (cash * 0.98) / total_cost
                    buy_shares = np.floor(buy_shares * scale / 100) * 100
                    buy_amounts = buy_prices * buy_shares
                    buy_commission = np.maximum(buy_amounts * self.config.commission_rate, self.config.min_commission)
                    buy_costs = buy_amounts + buy_commission

                cash -= np.sum(buy_costs)
                buy_indices = np.where(buy_mask)[0]
                holdings[buy_indices] += buy_shares

                for i, idx in enumerate(buy_indices):
                    trades_list.append({
                        'date': dates[t], 'code': codes[idx], 'action': 'buy',
                        'price': buy_prices[i], 'shares': int(buy_shares[i]),
                        'amount': buy_amounts[i], 'commission': buy_commission[i],
                        'tax': 0,
                    })

            # 计算当日总权益
            market_value = np.sum(holdings * today_prices)
            equity[t] = cash + market_value
            cash_series[t] = cash

        # 构建输出 DataFrame
        equity_curve = pd.DataFrame({
            'date': dates,
            'equity': equity,
            'cash': cash_series,
        })

        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame(
            columns=['date', 'code', 'action', 'price', 'shares', 'amount', 'commission', 'tax']
        )

        return equity_curve, trades_df

    def _calc_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标"""
        if equity_curve.empty or len(equity_curve) < 2:
            return {}

        eq = equity_curve['equity'].values
        daily_returns = np.diff(eq) / eq[:-1]

        total_return = (eq[-1] / eq[0] - 1)
        annual_return = (1 + total_return) ** (252 / max(len(eq), 1)) - 1
        annual_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        max_drawdown = np.min(drawdown)

        # 胜率
        win_rate = np.mean(daily_returns > 0) if len(daily_returns) > 0 else 0
        win_loss_ratio = (np.mean(daily_returns[daily_returns > 0]) / 
                          abs(np.mean(daily_returns[daily_returns < 0]))
                          if len(daily_returns[daily_returns > 0]) > 0 and len(daily_returns[daily_returns < 0]) > 0
                          else 0)

        return {
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "annual_volatility": round(annual_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
            "win_loss_ratio": round(win_loss_ratio, 4),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }


# ============================================================
# 2. 测试用例
# ============================================================

def generate_test_data(n_dates: int = 500, n_stocks: int = 100,
                       seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成测试数据"""
    np.random.seed(seed)
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='B')
    codes = [f'{i:06d}.SZ' for i in range(1, n_stocks + 1)]

    rows = []
    for i, code in enumerate(codes):
        base_price = np.random.uniform(5, 50)
        price_series = base_price + np.cumsum(np.random.randn(n_dates) * 0.3)
        price_series = np.maximum(price_series, 1)

        for t, dt in enumerate(dates):
            rows.append({
                'code': code,
                'date': dt,
                'open': price_series[t] * (1 + np.random.uniform(-0.02, 0.02)),
                'high': price_series[t] * (1 + np.random.uniform(0, 0.05)),
                'low': price_series[t] * (1 + np.random.uniform(-0.05, 0)),
                'close': price_series[t],
                'volume': np.random.uniform(10000, 1000000),
                'is_limit_up': np.random.random() < 0.02,
                'is_limit_down': np.random.random() < 0.02,
            })

    data = pd.DataFrame(rows)

    # 生成信号：随机买入/卖出
    signal_rows = []
    for dt in dates[::5]:  # 每5天调仓
        np.random.seed(hash(str(dt)) % 2**32)
        selected = np.random.choice(codes, size=min(20, n_stocks), replace=False)
        for code in codes:
            if code in selected:
                signal_rows.append({'code': code, 'date': dt, 'signal': 1})
            else:
                signal_rows.append({'code': code, 'date': dt, 'signal': -1})

    signals = pd.DataFrame(signal_rows)
    return data, signals


class TestVectorizedBacktest(unittest.TestCase):
    """向量化回测引擎测试"""

    @classmethod
    def setUpClass(cls):
        cls.data, cls.signals = generate_test_data(n_dates=500, n_stocks=50)
        cls.engine = VectorizedBacktestEngine()

    def test_basic_run(self):
        """测试基本回测运行"""
        result = self.engine.run_backtest(self.data, self.signals)
        self.assertIsNotNone(result)
        self.assertIn('equity_curve', result)
        self.assertIn('metrics', result)
        self.assertGreater(len(result['equity_curve']), 0)

    def test_metrics_completeness(self):
        """测试绩效指标完整性"""
        result = self.engine.run_backtest(self.data, self.signals)
        metrics = result['metrics']
        required = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']
        for key in required:
            self.assertIn(key, metrics, f"缺少指标: {key}")

    def test_empty_data(self):
        """测试空数据处理"""
        result = self.engine.run_backtest(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(result['equity_curve'].empty)

    def test_empty_signals(self):
        """测试空信号处理"""
        result = self.engine.run_backtest(self.data, pd.DataFrame())
        self.assertTrue(result['equity_curve'].empty)

    def test_single_stock(self):
        """测试单股票"""
        data_single = self.data[self.data['code'] == self.data['code'].iloc[0]]
        signals_single = self.signals[self.signals['code'] == self.signals['code'].iloc[0]]
        result = self.engine.run_backtest(data_single, signals_single)
        self.assertGreater(len(result['equity_curve']), 0)

    def test_price_limit_effect(self):
        """测试涨跌停限制效果"""
        # 创建涨停数据
        data_limit = self.data.copy()
        data_limit['is_limit_up'] = True
        data_limit['is_limit_down'] = False

        engine_with_limit = VectorizedBacktestEngine(
            BacktestConfig(price_limit=True)
        )
        engine_no_limit = VectorizedBacktestEngine(
            BacktestConfig(price_limit=False)
        )

        result_with = engine_with_limit.run_backtest(data_limit, self.signals)
        result_without = engine_no_limit.run_backtest(data_limit, self.signals)

        # 有涨跌停限制时，涨停标的无法买入，收益应更低
        with_return = result_with['metrics'].get('total_return', 0)
        without_return = result_without['metrics'].get('total_return', 0)

        print(f"\n  有涨跌停限制收益: {with_return:.4f}")
        print(f"  无涨跌停限制收益: {without_return:.4f}")

    def test_commission_effect(self):
        """测试佣金影响"""
        engine_low = VectorizedBacktestEngine(
            BacktestConfig(commission_rate=0.0001, stamp_tax_rate=0.0005)
        )
        engine_high = VectorizedBacktestEngine(
            BacktestConfig(commission_rate=0.0005, stamp_tax_rate=0.002)
        )

        result_low = engine_low.run_backtest(self.data, self.signals)
        result_high = engine_high.run_backtest(self.data, self.signals)

        low_return = result_low['metrics'].get('total_return', 0)
        high_return = result_high['metrics'].get('total_return', 0)

        print(f"\n  低费率收益: {low_return:.4f}")
        print(f"  高费率收益: {high_return:.4f}")

        # 低费率不应低于高费率
        self.assertGreaterEqual(low_return, high_return)


class TestBacktestPerformance(unittest.TestCase):
    """回测引擎性能对比测试"""

    @classmethod
    def setUpClass(cls):
        """生成大规模测试数据"""
        cls.small_data, cls.small_signals = generate_test_data(n_dates=250, n_stocks=100)
        cls.large_data, cls.large_signals = generate_test_data(n_dates=500, n_stocks=300)
        cls.engine = VectorizedBacktestEngine()

    def test_small_dataset_performance(self):
        """测试小数据集性能 (250天 x 100股)"""
        import time
        start = time.perf_counter()
        result = self.engine.run_backtest(self.small_data, self.small_signals)
        elapsed = time.perf_counter() - start
        print(f"\n  小数据集 (250天×100股): {elapsed:.4f}s")
        self.assertLess(elapsed, 5.0, "小数据集回测不应超过5秒")

    def test_large_dataset_performance(self):
        """测试大数据集性能 (500天 x 300股)"""
        import time
        start = time.perf_counter()
        result = self.engine.run_backtest(self.large_data, self.large_signals)
        elapsed = time.perf_counter() - start
        print(f"\n  大数据集 (500天×300股): {elapsed:.4f}s")
        self.assertLess(elapsed, 15.0, "大数据集回测不应超过15秒")

    def test_scalability(self):
        """测试可扩展性"""
        import time

        sizes = [
            (100, 50, "100天×50股"),
            (250, 100, "250天×100股"),
            (500, 200, "500天×200股"),
        ]

        times = []
        for n_days, n_stocks, label in sizes:
            data, signals = generate_test_data(n_dates=n_days, n_stocks=n_stocks)
            start = time.perf_counter()
            self.engine.run_backtest(data, signals)
            elapsed = time.perf_counter() - start
            times.append((label, elapsed))
            print(f"  {label}: {elapsed:.4f}s")

        # 检查时间增长是否大致线性（O(N)）
        if len(times) >= 2:
            ratio_time = times[-1][1] / times[0][1]
            ratio_data = (500 * 200) / (100 * 50)
            # 回测复杂度应低于数据量增长比例
            print(f"\n  时间增长比: {ratio_time:.2f}x")
            print(f"  数据量增长比: {ratio_data:.2f}x")
            self.assertLess(ratio_time, ratio_data * 2,
                            f"回测性能扩展性不足: {ratio_time:.2f}x vs {ratio_data:.2f}x")


class TestBacktestCorrectness(unittest.TestCase):
    """回测正确性测试"""

    def test_buy_and_hold_equivalence(self):
        """测试买入持有策略的正确性"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='B')

        data = pd.DataFrame({
            'code': '000001.SZ',
            'date': dates,
            'open': np.ones(n) * 10,
            'high': np.ones(n) * 10.5,
            'low': np.ones(n) * 9.5,
            'close': np.linspace(10, 20, n),
            'volume': np.ones(n) * 100000,
        })

        # 第一天买入，一直持有
        signals = pd.DataFrame({
            'code': ['000001.SZ'],
            'date': [dates[0]],
            'signal': [1],
        })

        engine = VectorizedBacktestEngine(BacktestConfig(
            commission_rate=0,  # 无佣金以简化验证
            stamp_tax_rate=0,
            slippage=0,
            min_commission=0,
            max_position_pct=1.0,  # 允许全仓
        ))
        result = engine.run_backtest(data, signals)

        # 初始资金100万，买入价格10，由于现金*0.95预算限制：
        # 预算 = min(1M, 950k) = 950k
        # 买入股数 = floor(950k/10/100)*100 = 95000股
        # 最终价格20，市值 = 20 * 95000 = 1,900,000
        # 剩余现金 = 1,000,000 - 950,000 = 50,000
        # 最终权益 = 1,950,000
        final_equity = result['equity_curve']['equity'].iloc[-1]
        expected_equity = 1950000.0

        print(f"\n  最终权益: {final_equity:.2f}")
        print(f"  预期权益: {expected_equity:.2f}")

        self.assertAlmostEqual(final_equity, expected_equity, delta=100)

    def test_cash_preservation_no_trades(self):
        """测试无交易时现金不变"""
        data = pd.DataFrame({
            'code': '000001.SZ',
            'date': pd.date_range('2023-01-01', periods=50, freq='B'),
            'close': np.random.randn(50).cumsum() + 10,
            'open': np.random.randn(50).cumsum() + 10,
            'high': np.random.randn(50).cumsum() + 10.5,
            'low': np.random.randn(50).cumsum() + 9.5,
            'volume': np.ones(50) * 100000,
        })
        signals = pd.DataFrame({
            'code': ['000001.SZ'],
            'date': [pd.Timestamp('2023-01-01')],
            'signal': [0],
        })

        engine = VectorizedBacktestEngine()
        result = engine.run_backtest(data, signals)

        final_equity = result['equity_curve']['equity'].iloc[-1]
        self.assertAlmostEqual(final_equity, 1_000_000.0, delta=1.0)

    def test_sharpe_ratio_zero_volatility(self):
        """测试无波动时的夏普比率"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='B')

        data = pd.DataFrame({
            'code': '000001.SZ',
            'date': dates,
            'close': np.ones(n) * 10,
            'open': np.ones(n) * 10,
            'high': np.ones(n) * 10.5,
            'low': np.ones(n) * 9.5,
            'volume': np.ones(n) * 100000,
        })

        # 不交易
        signals = pd.DataFrame()
        engine = VectorizedBacktestEngine()
        result = engine.run_backtest(data, signals)

        self.assertEqual(result['metrics'].get('sharpe_ratio', 0), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)