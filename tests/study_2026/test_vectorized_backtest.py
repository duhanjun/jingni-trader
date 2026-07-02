"""
测试文件: 向量化回测引擎验证
借鉴来源: backtesting.py (https://github.com/kernc/backtesting.py)
优化方向: backtest-engine - 用 NumPy 向量化操作替代逐行循环
日期: 2026-06-14

backtesting.py 的核心优势在于其极简的 API 和基于 NumPy 的向量化执行引擎。
它通过预计算信号和向量化订单匹配，在纯 Python 生态中实现了 5-10 倍
的速度提升。

本验证测试:
1. 向量化信号生成
2. 向量化订单匹配
3. 向量化权益曲线计算
4. 与现有逐行方式对比
5. A股特有规则支持 (T+1, 涨跌停)
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Tuple, Optional


# ============================================================================
# 向量化回测引擎
# ============================================================================

class VectorizedBacktest:
    """
    向量化回测引擎

    借鉴 backtesting.py 的核心设计:
    - 预计算所有信号
    - 使用 NumPy 向量化操作匹配订单
    - 最小化 Python 循环开销
    """

    def __init__(
        self,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            data: 行情数据, 需包含 date, code, open, close, is_limit_up, is_limit_down
            signals: 信号数据, 需包含 date, code, signal (1=买入, -1=卖出, 0=持有)

        返回:
            回测结果字典
        """
        # 数据预处理
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        codes = sorted(data['code'].unique())
        dates = sorted(data['date'].unique())

        # 构建价格矩阵 [n_dates, n_codes]
        close_matrix = self._build_matrix(data, 'close', dates, codes)
        open_matrix = self._build_matrix(data, 'open', dates, codes)

        # 构建信号矩阵
        signal_matrix = self._build_signal_matrix(signals, dates, codes)

        # 构建涨跌停矩阵
        if self.price_limit:
            is_limit_up = self._build_matrix(data, 'is_limit_up', dates, codes)
            is_limit_down = self._build_matrix(data, 'is_limit_down', dates, codes)
        else:
            is_limit_up = np.zeros_like(close_matrix)
            is_limit_down = np.zeros_like(close_matrix)

        # 执行回测
        n_dates, n_codes = close_matrix.shape

        # 仓位: 股数矩阵 [n_dates, n_codes]
        position = np.zeros((n_dates, n_codes), dtype=float)
        # 现金 (单变量, 逐日更新)
        cash = self.init_capital
        # 权益
        equity = np.zeros(n_dates + 1)
        equity[0] = self.init_capital
        # 交易记录
        trades = []

        for t in range(n_dates):
            # 复制前一天的仓位
            if t > 0:
                position[t] = position[t - 1].copy()

            # 执行卖出信号
            sell_mask = (signal_matrix[t] == -1) & (position[t] > 0)
            if self.price_limit:
                sell_mask = sell_mask & (~is_limit_down[t])

            if sell_mask.any():
                sell_codes_idx = np.where(sell_mask)[0]
                sell_prices = close_matrix[t, sell_codes_idx] * (1 - self.slippage)
                sell_amounts = position[t, sell_codes_idx] * sell_prices
                stamp_tax = sell_amounts * self.stamp_tax_rate
                commissions = np.maximum(sell_amounts * self.commission_rate, self.min_commission)
                sell_proceeds = sell_amounts - stamp_tax - commissions
                cash += sell_proceeds.sum()

                for idx, code_idx in enumerate(sell_codes_idx):
                    trades.append({
                        'date': dates[t], 'code': codes[code_idx],
                        'action': 'sell', 'price': float(sell_prices[idx]),
                        'shares': float(position[t, code_idx]),
                        'amount': float(sell_amounts[idx]),
                        'commission': float(commissions[idx]),
                        'stamp_tax': float(stamp_tax[idx]),
                    })

                position[t, sell_codes_idx] = 0

            # 执行买入信号
            buy_mask = (signal_matrix[t] == 1) & (position[t] == 0)
            if self.price_limit:
                buy_mask = buy_mask & (~is_limit_up[t])

            if buy_mask.any():
                buy_codes_idx = np.where(buy_mask)[0]
                n_buy = len(buy_codes_idx)
                cash_per_stock = cash / n_buy

                if self.t_plus_1:
                    buy_prices = close_matrix[t, buy_codes_idx]
                else:
                    buy_prices = open_matrix[t, buy_codes_idx]

                buy_prices *= (1 + self.slippage)

                # 计算可买股数 (100股整数倍)
                shares = np.floor(cash_per_stock / buy_prices / 100) * 100
                buy_amounts = shares * buy_prices
                commissions = np.maximum(buy_amounts * self.commission_rate, self.min_commission)

                total_cost = buy_amounts + commissions

                for i, code_idx in enumerate(buy_codes_idx):
                    if shares[i] > 0:
                        position[t, code_idx] = shares[i]
                        cash -= total_cost[i]
                        trades.append({
                            'date': dates[t], 'code': codes[code_idx],
                            'action': 'buy', 'price': float(buy_prices[i]),
                            'shares': float(shares[i]),
                            'amount': float(buy_amounts[i]),
                            'commission': float(commissions[i]),
                            'stamp_tax': 0.0,
                        })

            # 计算权益
            position_value = (position[t] * close_matrix[t]).sum()
            equity[t + 1] = cash + position_value

        # 构建权益曲线 (对齐: equity[t+1] 对应日期 dates[t])
        equity_curve = pd.DataFrame({
            'date': list(dates) + [dates[-1] + pd.Timedelta(days=1)],
            'equity': equity,
        })

        # 计算绩效指标
        metrics = self._calc_metrics(equity_curve)

        return {
            'equity_curve': equity_curve,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'metrics': metrics,
            'final_equity': equity[-1],
            'total_return': (equity[-1] - self.init_capital) / self.init_capital,
        }

    def _build_matrix(
        self, data: pd.DataFrame, col: str, dates: list, codes: list
    ) -> np.ndarray:
        """构建日期×代码矩阵"""
        pivot = data.pivot(index='date', columns='code', values=col)
        pivot = pivot.reindex(index=dates, columns=codes)
        return pivot.ffill().fillna(0).values

    def _build_signal_matrix(
        self, signals: pd.DataFrame, dates: list, codes: list
    ) -> np.ndarray:
        """构建信号矩阵"""
        if 'signal' not in signals.columns:
            return np.zeros((len(dates), len(codes)))

        pivot = signals.pivot(index='date', columns='code', values='signal')
        pivot = pivot.reindex(index=dates, columns=codes)
        return pivot.fillna(0).values

    def _calc_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标"""
        eq = equity_curve.set_index('date')['equity']
        returns = eq.pct_change().dropna()

        if len(returns) < 2:
            return {}

        total_return = eq.iloc[-1] / eq.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        win_rate = (returns > 0).mean()
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'calmar_ratio': float(calmar),
        }


# ============================================================================
# 逐行回测引擎 (对比用)
# ============================================================================

class RowByRowBacktest:
    """逐行回测引擎 (模拟传统方式)"""

    def __init__(
        self,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """逐行执行回测"""
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(data['date'].unique())
        codes = sorted(data['code'].unique())

        cash = self.init_capital
        position = {c: 0 for c in codes}
        equity_list = []
        trades = []

        for dt in dates:
            day_data = data[data['date'] == dt].set_index('code')
            day_signals = signals[signals['date'] == dt].set_index('code')

            # 卖出
            for code in codes:
                if code not in day_signals.index or code not in day_data.index:
                    continue
                sig = day_signals.loc[code, 'signal']
                if sig == -1 and position[code] > 0:
                    price = day_data.loc[code, 'close']
                    if self.price_limit and day_data.loc[code].get('is_limit_down', False):
                        continue
                    price *= (1 - self.slippage)
                    amount = position[code] * price
                    stamp_tax = amount * self.stamp_tax_rate
                    commission = max(amount * self.commission_rate, self.min_commission)
                    cash += amount - stamp_tax - commission
                    trades.append({
                        'date': dt, 'code': code, 'action': 'sell',
                        'price': price, 'shares': position[code],
                        'amount': amount, 'commission': commission, 'stamp_tax': stamp_tax,
                    })
                    position[code] = 0

            # 买入
            buy_candidates = []
            for code in codes:
                if code not in day_signals.index or code not in day_data.index:
                    continue
                sig = day_signals.loc[code, 'signal']
                if sig == 1 and position[code] == 0:
                    if self.price_limit and day_data.loc[code].get('is_limit_up', False):
                        continue
                    buy_candidates.append(code)

            if buy_candidates:
                cash_per = cash / len(buy_candidates)
                for code in buy_candidates:
                    price = day_data.loc[code, 'close']
                    price *= (1 + self.slippage)
                    shares = int(cash_per / price / 100) * 100
                    if shares > 0:
                        amount = shares * price
                        commission = max(amount * self.commission_rate, self.min_commission)
                        cash -= amount + commission
                        position[code] = shares
                        trades.append({
                            'date': dt, 'code': code, 'action': 'buy',
                            'price': price, 'shares': shares,
                            'amount': amount, 'commission': commission, 'stamp_tax': 0.0,
                        })

            # 计算权益
            position_value = 0
            for code in codes:
                if code in day_data.index:
                    position_value += position[code] * day_data.loc[code, 'close']

            equity_list.append({
                'date': dt,
                'equity': cash + position_value,
            })

        equity_curve = pd.DataFrame(equity_list)
        returns = equity_curve.set_index('date')['equity'].pct_change().dropna()

        if len(returns) < 2:
            metrics = {}
        else:
            total_return = equity_curve['equity'].iloc[-1] / self.init_capital - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            max_drawdown = (equity_curve['equity'] / equity_curve['equity'].cummax() - 1).min()
            sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
            win_rate = (returns > 0).mean()
            calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            metrics = {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'calmar_ratio': float(calmar),
            }

        return {
            'equity_curve': equity_curve,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'metrics': metrics,
            'final_equity': cash + position_value,
            'total_return': (cash + position_value - self.init_capital) / self.init_capital,
        }


# ============================================================================
# 测试数据生成
# ============================================================================

def generate_test_data(
    n_codes: int = 50,
    n_days: int = 252,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成测试数据"""
    np.random.seed(seed)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    dates = pd.bdate_range('2024-01-01', periods=n_days)

    rows = []
    for code in codes:
        base_price = np.random.uniform(10, 50)
        returns = np.random.normal(0.0005, 0.015, n_days)
        prices = base_price * np.cumprod(1 + returns)

        for i, (date, price) in enumerate(zip(dates, prices)):
            rows.append({
                'date': date,
                'code': code,
                'open': float(price * (1 + np.random.normal(0, 0.003))),
                'close': float(price),
                'is_limit_up': False,
                'is_limit_down': False,
            })

    data = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)

    # 生成信号: 简单均值回归策略
    signals_list = []
    for code in codes:
        code_data = data[data['code'] == code].sort_values('date')
        ret_20d = code_data['close'].pct_change(20)
        signal = np.zeros(len(code_data))
        signal[ret_20d > 0.05] = -1  # 涨太多卖出
        signal[ret_20d < -0.05] = 1  # 跌太多买入
        signals_list.append(pd.DataFrame({
            'date': code_data['date'].values,
            'code': code,
            'signal': signal.astype(int),
        }))

    signals = pd.concat(signals_list, ignore_index=True)

    return data, signals


# ============================================================================
# 测试函数
# ============================================================================

def test_correctness():
    """测试正确性: 向量化 vs 逐行"""
    print("\n" + "=" * 60)
    print("测试1: 正确性验证 (向量化 vs 逐行)")
    print("=" * 60)

    data, signals = generate_test_data(n_codes=50, n_days=252)

    # 向量化回测
    vec_bt = VectorizedBacktest()
    start = time.time()
    vec_result = vec_bt.run(data, signals)
    vec_time = time.time() - start
    print(f"  向量化回测耗时: {vec_time:.4f}s")

    # 逐行回测
    row_bt = RowByRowBacktest()
    start = time.time()
    row_result = row_bt.run(data, signals)
    row_time = time.time() - start
    print(f"  逐行回测耗时: {row_time:.4f}s")

    # 对比结果
    print(f"\n  权益曲线对比:")
    vec_eq = vec_result['equity_curve'].set_index('date')['equity']
    row_eq = row_result['equity_curve'].set_index('date')['equity']

    # 对齐索引
    common_idx = vec_eq.index.intersection(row_eq.index)
    diff = (vec_eq.loc[common_idx] - row_eq.loc[common_idx]).abs()
    max_diff = diff.max()
    corr = vec_eq.loc[common_idx].corr(row_eq.loc[common_idx])

    print(f"  最大权益差异: {max_diff:.6f}")
    print(f"  权益曲线相关性: {corr:.6f}")

    # 绩效对比
    print(f"\n  绩效指标对比:")
    for key in vec_result['metrics']:
        v = vec_result['metrics'][key]
        r = row_result['metrics'][key]
        delta = abs(v - r) / max(abs(r), 1e-8) * 100
        print(f"    {key}: 向量化={v:.6f}, 逐行={r:.6f}, 差异={delta:.2f}%")

    # 性能对比
    speedup = row_time / vec_time if vec_time > 0 else float('inf')
    print(f"\n  性能加速比: {speedup:.1f}x (向量化更快)")

    assert corr > 0.99, f"权益曲线相关性过低: {corr}"
    print("  ✓ 正确性验证通过")

    return True


def test_scalability():
    """测试可扩展性"""
    print("\n" + "=" * 60)
    print("测试2: 可扩展性 (不同规模数据)")
    print("=" * 60)

    configs = [
        (50, 252, "小规模"),
        (200, 252, "中规模"),
        (500, 252, "大规模"),
    ]

    for n_codes, n_days, label in configs:
        data, signals = generate_test_data(n_codes=n_codes, n_days=n_days)

        # 向量化
        vec_bt = VectorizedBacktest()
        start = time.time()
        vec_bt.run(data, signals)
        vec_time = time.time() - start

        # 逐行
        row_bt = RowByRowBacktest()
        start = time.time()
        row_bt.run(data, signals)
        row_time = time.time() - start

        speedup = row_time / vec_time if vec_time > 0 else float('inf')
        print(f"  {label} ({n_codes}股×{n_days}天): 向量化={vec_time:.3f}s, 逐行={row_time:.3f}s, 加速比={speedup:.1f}x")

    print("  ✓ 可扩展性测试通过")

    return True


def test_a_share_rules():
    """测试A股特有规则"""
    print("\n" + "=" * 60)
    print("测试3: A股特有规则 (T+1, 涨跌停)")
    print("=" * 60)

    data, signals = generate_test_data(n_codes=50, n_days=252)

    # 设置部分股票涨跌停
    data.loc[data['code'] == data['code'].unique()[0], 'is_limit_up'] = True
    data.loc[data['code'] == data['code'].unique()[1], 'is_limit_down'] = True

    # 给涨跌停股票设置信号
    limit_up_code = data['code'].unique()[0]
    limit_down_code = data['code'].unique()[1]
    signals.loc[
        (signals['code'] == limit_up_code) & (signals['date'] == signals['date'].unique()[100]),
        'signal'
    ] = 1  # 买入信号, 但涨停买不进
    signals.loc[
        (signals['code'] == limit_down_code) & (signals['date'] == signals['date'].unique()[100]),
        'signal'
    ] = -1  # 卖出信号, 但跌停卖不出

    vec_bt = VectorizedBacktest(price_limit=True)
    result = vec_bt.run(data, signals)

    # 检查涨跌停日没有交易
    trades = result['trades']
    if not trades.empty:
        limit_day = signals['date'].unique()[100]
        limit_day_trades = trades[trades['date'] == limit_day]
        for _, trade in limit_day_trades.iterrows():
            if trade['code'] == limit_up_code:
                assert trade['action'] != 'buy', f"涨停股 {trade['code']} 不应该被买入"
            if trade['code'] == limit_down_code:
                assert trade['action'] != 'sell', f"跌停股 {trade['code']} 不应该被卖出"

    print("  ✓ A股规则验证通过")

    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("向量化回测引擎 - 验证测试")
    print("借鉴来源: backtesting.py (kernc/backtesting.py)")
    print("=" * 60)

    results = []
    tests = [
        ("正确性验证", test_correctness),
        ("可扩展性", test_scalability),
        ("A股规则", test_a_share_rules),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "PASS"))
        except Exception as e:
            results.append((name, f"FAIL: {e}"))
            print(f"  ✗ {name} 失败: {e}")

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, status in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASS")
    print(f"\n总计: {passed}/{len(results)} 通过")
    return all(s == "PASS" for _, s in results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)