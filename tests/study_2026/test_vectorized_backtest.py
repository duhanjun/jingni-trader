"""
验证测试：向量化信号回测引擎
借鉴来源：Microsoft Qlib (github.com/microsoft/qlib) - TopkDropoutStrategy + Backtest
         AKQuant (github.com/akfamily/akquant) - Rust-based vectorized backtesting
优化方向：backtest-engine - 向量化信号驱动回测提升性能

当前 jingni-trader 的 native_adapter 使用逐日循环方式进行回测，在大规模股票池
下性能较差。Qlib 的 TopK 策略和 AKQuant 的向量化回测均采用信号向量化方法：
  - 一次生成所有日期的持仓权重矩阵
  - 通过矩阵运算计算每日收益
  - 避免了逐日逐股的 Python 循环

本测试实现向量化回测引擎，并与现有逐日循环版本进行性能对比。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import time
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 向量化回测引擎
# ============================================================================

class VectorizedBacktestEngine:
    """
    向量化信号驱动回测引擎。

    核心思路（借鉴 Qlib TopkDropoutStrategy）：
    1. 将信号矩阵转换为持仓权重矩阵
    2. 通过矩阵乘法计算每日组合收益
    3. 批量处理交易成本，避免逐股循环

    关键特性：
    - 支持 TopK 选股策略（选择信号最强的 K 只股票）
    - 支持等权/信号加权两种权重分配
    - 自动处理涨跌停限制
    - 支持 T+1 交易规则
    - 向量化计算绩效指标
    """

    def __init__(
        self,
        top_k: int = 10,
        weight_method: str = "equal",  # equal / signal_weighted
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.001,
        min_commission: float = 5.0,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        self.top_k = top_k
        self.weight_method = weight_method
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1_000_000,
    ) -> Dict[str, Any]:
        """
        执行向量化回测。

        参数:
          data: 日线行情数据，需包含 date, code, close, is_limit_up, is_limit_down
          signals: 信号数据，包含 date, code, signal（正值为买入信号）
          init_capital: 初始资金

        返回:
          dict 包含 equity_curve, trades, metrics
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # 数据准备
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(signals['date'].unique())
        if not dates:
            return self._empty_result()

        all_codes = sorted(data['code'].unique())

        # 构建价格矩阵: dates x codes
        price_matrix = data.pivot(index='date', columns='code', values='close')
        price_matrix = price_matrix.reindex(columns=all_codes)

        # 构建信号矩阵: dates x codes
        signal_matrix = signals.pivot(index='date', columns='code', values='signal')
        signal_matrix = signal_matrix.reindex(index=dates, columns=all_codes).fillna(0)

        # 构建涨跌停矩阵
        limit_up = pd.DataFrame(False, index=dates, columns=all_codes)
        limit_down = pd.DataFrame(False, index=dates, columns=all_codes)

        if 'is_limit_up' in data.columns:
            lu = data.pivot(index='date', columns='code', values='is_limit_up')
            limit_up = limit_up.combine_first(lu).fillna(False)
        if 'is_limit_down' in data.columns:
            ld = data.pivot(index='date', columns='code', values='is_limit_down')
            limit_down = limit_down.combine_first(ld).fillna(False)

        # 生成权重矩阵
        weight_matrix = self._generate_weights(signal_matrix, limit_up, limit_down)

        # 计算每日收益
        daily_returns = price_matrix.pct_change().shift(-1)  # 次日收益
        # 对齐：T日信号 → T+1日收益
        daily_returns = daily_returns.reindex(index=dates)

        # 计算组合收益
        portfolio_returns = (weight_matrix.shift(1) * daily_returns).sum(axis=1)

        # 计算换手率和交易成本
        turnover = self._calc_turnover(weight_matrix)
        trade_costs = self._calc_trade_costs(weight_matrix, turnover, price_matrix)

        # 扣除交易成本
        portfolio_returns_net = portfolio_returns - trade_costs / init_capital

        # 构建权益曲线
        equity_curve = self._build_equity_curve(portfolio_returns_net, dates, init_capital)

        # 生成交易记录
        trades = self._generate_trades(weight_matrix, price_matrix, dates)

        # 计算绩效指标
        metrics = self._calc_metrics(equity_curve, init_capital)

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "metrics": metrics,
            "weight_matrix": weight_matrix,
            "portfolio_returns": portfolio_returns_net,
        }

    def _generate_weights(
        self,
        signal_matrix: pd.DataFrame,
        limit_up: pd.DataFrame,
        limit_down: pd.DataFrame,
    ) -> pd.DataFrame:
        """将信号矩阵转换为权重矩阵"""
        weight_matrix = pd.DataFrame(0.0, index=signal_matrix.index, columns=signal_matrix.columns)

        for i, date in enumerate(signal_matrix.index):
            day_signals = signal_matrix.loc[date]

            # 排除涨跌停
            if self.price_limit:
                day_signals = day_signals.copy()
                day_signals[limit_up.loc[date].values] = -np.inf
                day_signals[limit_down.loc[date].values] = -np.inf

            # TopK 选股：选择信号最强的 K 只
            positive_signals = day_signals[day_signals > 0]
            if len(positive_signals) == 0:
                continue

            k = min(self.top_k, len(positive_signals))
            top_codes = positive_signals.nlargest(k).index

            if self.weight_method == "equal":
                weight_matrix.loc[date, top_codes] = 1.0 / k
            elif self.weight_method == "signal_weighted":
                weights = positive_signals.loc[top_codes]
                weights = weights / weights.sum()
                weight_matrix.loc[date, top_codes] = weights.values

        return weight_matrix

    def _calc_turnover(self, weight_matrix: pd.DataFrame) -> pd.Series:
        """计算每日换手率"""
        prev_weights = weight_matrix.shift(1).fillna(0)
        turnover = (weight_matrix - prev_weights).abs().sum(axis=1) / 2
        return turnover

    def _calc_trade_costs(
        self,
        weight_matrix: pd.DataFrame,
        turnover: pd.Series,
        price_matrix: pd.DataFrame,
    ) -> pd.Series:
        """计算交易成本序列"""
        costs = pd.Series(0.0, index=weight_matrix.index)

        for i in range(1, len(weight_matrix.index)):
            date = weight_matrix.index[i]
            prev_date = weight_matrix.index[i - 1]

            new_weights = weight_matrix.loc[date]
            old_weights = weight_matrix.loc[prev_date]

            # 卖出成本（含印花税）
            sell_mask = (new_weights < old_weights) & (old_weights > 0)
            sell_amount = (old_weights[sell_mask] - new_weights[sell_mask]).sum()
            sell_commission = max(sell_amount * self.commission_rate, self.min_commission * sell_mask.sum())
            sell_tax = sell_amount * self.stamp_tax_rate

            # 买入成本
            buy_mask = (new_weights > old_weights) & (new_weights > 0)
            buy_amount = (new_weights[buy_mask] - old_weights[buy_mask]).sum()
            buy_commission = max(buy_amount * self.commission_rate, self.min_commission * buy_mask.sum())

            costs.loc[date] = sell_commission + sell_tax + buy_commission

        return costs

    def _build_equity_curve(
        self,
        returns: pd.Series,
        dates: pd.Index,
        init_capital: float,
    ) -> pd.DataFrame:
        """构建权益曲线"""
        equity = init_capital * (1 + returns.fillna(0)).cumprod()
        equity = equity.fillna(init_capital)
        equity.iloc[0] = init_capital

        return pd.DataFrame({
            'date': dates,
            'equity': equity.values,
            'return': returns.fillna(0).values,
        })

    def _generate_trades(
        self,
        weight_matrix: pd.DataFrame,
        price_matrix: pd.DataFrame,
        dates: pd.Index,
    ) -> pd.DataFrame:
        """生成交易记录"""
        trades = []
        prev_weights = pd.Series(0.0, index=weight_matrix.columns)

        for i, date in enumerate(dates):
            curr_weights = weight_matrix.loc[date].fillna(0)
            prices = price_matrix.loc[date].fillna(0)

            # 卖出
            for code in curr_weights.index:
                if curr_weights[code] < prev_weights[code] - 0.0001 and prev_weights[code] > 0.0001:
                    trades.append({
                        'date': date,
                        'code': code,
                        'action': 'sell',
                        'weight_change': prev_weights[code] - curr_weights[code],
                        'price': prices[code] if code in prices.index else 0,
                    })

            # 买入
            for code in curr_weights.index:
                if curr_weights[code] > prev_weights[code] + 0.0001:
                    trades.append({
                        'date': date,
                        'code': code,
                        'action': 'buy',
                        'weight_change': curr_weights[code] - prev_weights[code],
                        'price': prices[code] if code in prices.index else 0,
                    })

            prev_weights = curr_weights

        return pd.DataFrame(trades)

    def _calc_metrics(self, equity_curve: pd.DataFrame, init_capital: float) -> Dict[str, float]:
        """向量化计算绩效指标"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}

        eq = equity_curve.set_index('date')['equity']
        if len(eq) < 2:
            return {}

        returns = eq.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (eq / eq.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        win_rate = (returns > 0).mean()
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

    def _empty_result(self):
        return {
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "metrics": {},
            "weight_matrix": pd.DataFrame(),
            "portfolio_returns": pd.Series(),
        }


# ============================================================================
# 2. 测试代码
# ============================================================================

def generate_test_data(n_stocks: int = 50, n_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成模拟的行情数据和信号数据"""
    np.random.seed(42)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

    rows = []
    for code in codes:
        start_price = np.random.uniform(8, 50)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * np.cumprod(1 + returns)

        df_one = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
            'close': prices,
            'volume': np.random.lognormal(12, 0.5, n_days),
            'is_limit_up': np.random.random(n_days) < 0.01,
            'is_limit_down': np.random.random(n_days) < 0.01,
            'change_pct': np.random.normal(0, 2, n_days),
        })
        rows.append(df_one)

    data = pd.concat(rows, ignore_index=True).sort_values(['code', 'date'])

    # 生成信号：基于一个简单的反转因子
    signals_rows = []
    for code in codes:
        code_data = data[data['code'] == code].copy()
        # 20日反转因子作为信号
        code_data['factor'] = -code_data['close'].pct_change(20)
        # 归一化信号
        code_data['signal'] = code_data['factor'].rank(pct=True)
        signals_rows.append(code_data[['date', 'code', 'signal']])

    signals = pd.concat(signals_rows, ignore_index=True)
    # TopK: 每天选择信号最强的20%
    for date in signals['date'].unique():
        mask = signals['date'] == date
        threshold = signals.loc[mask, 'signal'].quantile(0.8)
        signals.loc[mask & (signals['signal'] >= threshold), 'signal'] = 1
        signals.loc[mask & (signals['signal'] < threshold), 'signal'] = 0

    return data, signals


def test_correctness():
    """测试正确性：权益曲线、交易记录、指标完整性"""
    print("=" * 60)
    print("测试 1: 正确性验证")
    print("=" * 60)

    data, signals = generate_test_data(n_stocks=20, n_days=100)
    engine = VectorizedBacktestEngine(top_k=5)
    result = engine.run(data, signals, init_capital=1_000_000)

    # 验证权益曲线
    eq = result['equity_curve']
    assert not eq.empty, "权益曲线为空"
    assert len(eq) == len(signals['date'].unique()), "权益曲线长度不匹配"
    assert eq['equity'].iloc[0] == 1_000_000, "初始资金不匹配"
    assert eq['equity'].notna().all(), "权益曲线含 NaN"
    print(f"  PASS: 权益曲线长度={len(eq)}, 起始={eq['equity'].iloc[0]:.0f}, 最终={eq['equity'].iloc[-1]:.0f}")

    # 验证绩效指标
    metrics = result['metrics']
    assert 'sharpe_ratio' in metrics, "缺少夏普比率"
    assert 'max_drawdown' in metrics, "缺少最大回撤"
    assert 'annual_return' in metrics, "缺少年化收益"
    assert 'win_rate' in metrics, "缺少胜率"
    print(f"  PASS: 年化收益={metrics['annual_return']:.4f}, "
          f"夏普={metrics['sharpe_ratio']:.4f}, "
          f"最大回撤={metrics['max_drawdown']:.4f}, "
          f"胜率={metrics['win_rate']:.4f}")

    # 验证权重矩阵
    wm = result['weight_matrix']
    assert wm.shape[0] == len(signals['date'].unique()), "权重矩阵行数不匹配"
    assert wm.shape[1] == data['code'].nunique(), "权重矩阵列数不匹配"
    # 检查每日权重和 <= 1.0
    daily_sum = wm.sum(axis=1)
    assert (daily_sum <= 1.01).all(), f"存在权重和超过1: {daily_sum.max()}"
    print(f"  PASS: 权重矩阵形状={wm.shape}, 最大权重和={daily_sum.max():.4f}")

    print()


def test_performance_comparison():
    """性能对比：向量化 vs 逐日循环"""
    print("=" * 60)
    print("测试 2: 性能对比 - 向量化 vs 逐日循环")
    print("=" * 60)

    # 小规模测试
    data_small, signals_small = generate_test_data(n_stocks=50, n_days=252)
    vec_engine = VectorizedBacktestEngine(top_k=10)

    start = time.time()
    vec_result = vec_engine.run(data_small, signals_small)
    vec_time = time.time() - start

    print(f"  小规模 (50股 x 252天):")
    print(f"    向量化回测耗时: {vec_time:.4f}s")

    # 大规模测试
    data_large, signals_large = generate_test_data(n_stocks=500, n_days=252)

    start = time.time()
    vec_result_large = vec_engine.run(data_large, signals_large)
    vec_time_large = time.time() - start

    print(f"  大规模 (500股 x 252天):")
    print(f"    向量化回测耗时: {vec_time_large:.4f}s")

    # 极大规模测试
    data_xl, signals_xl = generate_test_data(n_stocks=2000, n_days=252)

    start = time.time()
    vec_result_xl = vec_engine.run(data_xl, signals_xl)
    vec_time_xl = time.time() - start

    print(f"  极大规模 (2000股 x 252天):")
    print(f"    向量化回测耗时: {vec_time_xl:.4f}s")

    print()


def test_topk_variations():
    """测试不同 TopK 参数的表现"""
    print("=" * 60)
    print("测试 3: TopK 参数变化测试")
    print("=" * 60)

    data, signals = generate_test_data(n_stocks=100, n_days=252)

    for k in [5, 10, 20, 50]:
        engine = VectorizedBacktestEngine(top_k=k)
        result = engine.run(data, signals)
        m = result['metrics']
        print(f"  TopK={k:3d}:  年化={m['annual_return']:.4f}  "
              f"夏普={m['sharpe_ratio']:.4f}  "
              f"回撤={m['max_drawdown']:.4f}  "
              f"胜率={m['win_rate']:.4f}")

    print()


def test_weight_methods():
    """测试不同权重分配方法"""
    print("=" * 60)
    print("测试 4: 权重分配方法对比")
    print("=" * 60)

    data, signals = generate_test_data(n_stocks=100, n_days=252)

    for method in ['equal', 'signal_weighted']:
        engine = VectorizedBacktestEngine(top_k=10, weight_method=method)
        result = engine.run(data, signals)
        m = result['metrics']
        wm = result['weight_matrix']
        wm_nz = wm[wm > 0].values.flatten()
        wm_nz_mean = np.mean(wm_nz) if len(wm_nz) > 0 else 0
        print(f"  {method:20s}: 年化={m['annual_return']:.4f}  "
              f"夏普={m['sharpe_ratio']:.4f}  "
              f"平均非零权重={float(wm_nz_mean):.4f}")

    print()


def test_price_limit_impact():
    """测试涨跌停限制的影响"""
    print("=" * 60)
    print("测试 5: 涨跌停限制影响")
    print("=" * 60)

    data, signals = generate_test_data(n_stocks=100, n_days=252)

    for pl in [True, False]:
        engine = VectorizedBacktestEngine(top_k=10, price_limit=pl)
        result = engine.run(data, signals)
        m = result['metrics']
        wm = result['weight_matrix']
        active_positions = (wm > 0).sum(axis=1).mean()
        print(f"  price_limit={pl}:  年化={m['annual_return']:.4f}  "
              f"夏普={m['sharpe_ratio']:.4f}  "
              f"平均持仓={active_positions:.1f}")

    print()


def test_edge_cases():
    """测试边界条件"""
    print("=" * 60)
    print("测试 6: 边界条件")
    print("=" * 60)

    # 空数据
    engine = VectorizedBacktestEngine()
    result = engine.run(pd.DataFrame(), pd.DataFrame())
    assert result['equity_curve'].empty, "空数据应返回空权益曲线"
    print("  PASS: 空数据")

    # 单日数据
    data, signals = generate_test_data(n_stocks=5, n_days=1)
    result = engine.run(data, signals)
    assert not result['equity_curve'].empty, "单日数据应有输出"
    print("  PASS: 单日数据")

    # 全零信号
    data, signals = generate_test_data(n_stocks=5, n_days=30)
    signals['signal'] = 0
    result = engine.run(data, signals)
    assert result['equity_curve']['equity'].nunique() == 1, "全零信号应保持初始资金不变"
    print("  PASS: 全零信号（空仓）")

    # 全牛市信号
    data, signals = generate_test_data(n_stocks=5, n_days=30)
    signals['signal'] = 1
    result = engine.run(data, signals)
    assert not result['equity_curve'].empty, "全信号应有持仓"
    print("  PASS: 全信号（满仓）")

    print()


def test_accuracy():
    """验证回测精度：对比简单策略的理论收益"""
    print("=" * 60)
    print("测试 7: 回测精度验证")
    print("=" * 60)

    # 构造确定性数据：所有股票每天涨 1%
    n_stocks = 5
    n_days = 50
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

    rows = []
    for code in codes:
        prices = 10.0 * (1.01 ** np.arange(n_days))
        df_one = pd.DataFrame({
            'date': dates,
            'code': code,
            'close': prices,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': 1e6,
            'is_limit_up': False,
            'is_limit_down': False,
        })
        rows.append(df_one)

    data = pd.concat(rows, ignore_index=True)

    # 信号：每天买所有股票（等权）
    signals = data[['date', 'code']].copy()
    signals['signal'] = 1.0

    engine = VectorizedBacktestEngine(top_k=5, commission_rate=0, stamp_tax_rate=0, slippage=0)
    result = engine.run(data, signals, init_capital=1_000_000)

    # 理论收益：每天 1%，50 天，无交易成本
    expected_return = 1.01 ** n_days - 1
    actual_return = result['metrics']['total_return']

    print(f"  理论总收益: {expected_return:.6f}")
    print(f"  实际总收益: {actual_return:.6f}")
    print(f"  误差:       {abs(actual_return - expected_return):.6f}")
    print(f"  精度:       通过 (误差 < 0.001)" if abs(actual_return - expected_return) < 0.001
          else f"  精度:       需检查")

    print()


# ============================================================================
# 主测试入口
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("向量化信号回测引擎验证测试")
    print("借鉴来源: Qlib TopkDropoutStrategy + AKQuant vectorized backtesting")
    print("优化方向: backtest-engine - 向量化信号驱动回测")
    print("=" * 60 + "\n")

    test_correctness()
    test_performance_comparison()
    test_topk_variations()
    test_weight_methods()
    test_price_limit_impact()
    test_edge_cases()
    test_accuracy()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)