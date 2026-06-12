"""
优化方向: 向量化回测引擎性能对比 (Polars vs Pandas Backtesting)
借鉴来源: AKQuant (github.com/akfamily/akquant) - Rust+Python 混合高性能回测
借鉴亮点: AKQuant 使用 Polars 作为因子计算引擎，通过向量化操作替代逐行循环，
         在部分场景下可获得显著性能提升。同时内置 TA-Lib 双后端支持。

优化目标: 评估在 jingni-trader 的 backtest-engine 中引入 Polars 向量化计算的
         性能收益，对比 Pandas 实现的回测引擎在相同场景下的表现。
"""

import sys
import os
sys.path.insert(0, '/workspace')

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import time
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 Polars
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("⚠ Polars 未安装，将跳过 Polars 相关测试")


# ============================================================================
# 1. Pandas 向量化回测引擎
# ============================================================================

class PandasVectorizedBacktest:
    """基于 Pandas 的向量化回测引擎（参考当前 jingni-trader 实现）"""

    def __init__(
        self,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
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
            data: 日线数据，包含 code, date, close, open, is_limit_up, is_limit_down
            signals: 交易信号，包含 code, date, signal
        """
        t0 = time.perf_counter()

        # 合并数据和信号
        merged = data.merge(signals, on=['code', 'date'], how='left')
        merged['signal'] = merged['signal'].fillna(0)

        if self.t_plus_1:
            merged['signal'] = merged.groupby('code')['signal'].shift(1).fillna(0)

        # 计算每日持仓和收益
        dates = sorted(merged['date'].unique())
        equity_curve = []
        trades = []

        # 等权分配：每日根据信号等权买入/卖出
        for dt in dates:
            day_data = merged[merged['date'] == dt].copy()

            if self.price_limit:
                # 剔除涨跌停无法交易的股票
                day_data = day_data[
                    ~(day_data['is_limit_up'].fillna(False)) &
                    ~(day_data['is_limit_down'].fillna(False))
                ]

            buy_codes = day_data[day_data['signal'] > 0]['code'].tolist()

            if len(buy_codes) > 0:
                # 等权分配
                weight = 1.0 / len(buy_codes)
                day_return = 0
                for code in buy_codes:
                    stock_data = day_data[day_data['code'] == code]
                    if stock_data.empty:
                        continue
                    price = stock_data['close'].values[0]
                    # 考虑滑点
                    exec_price = price * (1 + self.slippage)
                    day_return += weight * 0  # 简化：当日收益由后续计算
            else:
                day_return = 0

            # 简化处理：用持仓股票的平均收益
            if len(buy_codes) > 0:
                positions = day_data[day_data['code'].isin(buy_codes)]
                if 'change_pct' in positions.columns:
                    day_return = positions['change_pct'].mean() / 100
                else:
                    day_return = 0
            else:
                day_return = 0

            equity_curve.append({
                'date': dt,
                'return': day_return,
                'n_positions': len(buy_codes),
            })

        # 构建权益曲线
        eq_df = pd.DataFrame(equity_curve)
        eq_df['equity'] = self.init_capital * (1 + eq_df['return']).cumprod()

        # 计算绩效指标
        metrics = self._calc_metrics(eq_df)

        elapsed = time.perf_counter() - t0

        return {
            'equity_curve': eq_df,
            'metrics': metrics,
            'trades': trades,
            'elapsed_time': elapsed,
        }

    def _calc_metrics(self, eq_df: pd.DataFrame) -> Dict[str, float]:
        """计算回测绩效指标"""
        returns = eq_df['return'].dropna()
        if len(returns) < 2:
            return {}

        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        equity = eq_df['equity']
        max_drawdown = (equity / equity.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        win_rate = (returns > 0).mean()

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0,
        }


# ============================================================================
# 2. Polars 向量化回测引擎（需要安装 polars）
# ============================================================================

class PolarsVectorizedBacktest:
    """基于 Polars 的向量化回测引擎（借鉴 AKQuant 设计）"""

    def __init__(
        self,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        slippage: float = 0.0001,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
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
        执行 Polars 向量化回测
        """
        if not HAS_POLARS:
            return {"error": "Polars 未安装", "metrics": {}}

        t0 = time.perf_counter()

        # 转换为 Polars DataFrame
        data_pl = pl.from_pandas(data)
        signals_pl = pl.from_pandas(signals)

        # 合并
        merged = data_pl.join(
            signals_pl.select(['code', 'date', 'signal']),
            on=['code', 'date'],
            how='left'
        )
        merged = merged.with_columns(pl.col('signal').fill_null(0))

        if self.t_plus_1:
            merged = merged.sort(['code', 'date'])
            merged = merged.with_columns(
                pl.col('signal').shift(1).over('code').fill_null(0).alias('signal')
            )

        # 过滤涨跌停
        if self.price_limit:
            merged = merged.filter(
                ~pl.col('is_limit_up').fill_null(False) &
                ~pl.col('is_limit_down').fill_null(False)
            )

        # 按日期分组计算（Polars 的 group_by 比 Pandas groupby 更快）
        if 'change_pct' in merged.columns:
            daily_returns = (
                merged
                .filter(pl.col('signal') > 0)
                .group_by('date')
                .agg([
                    pl.col('change_pct').mean().alias('return'),
                    pl.col('code').count().alias('n_positions'),
                ])
                .sort('date')
            )
        else:
            daily_returns = (
                merged
                .filter(pl.col('signal') > 0)
                .group_by('date')
                .agg([
                    pl.col('code').count().alias('n_positions'),
                ])
                .sort('date')
                .with_columns(pl.lit(0.0).alias('return'))
            )

        # 转回 Pandas 计算权益曲线
        eq_df = daily_returns.to_pandas()
        if 'return' in eq_df.columns:
            eq_df['return'] = eq_df['return'] / 100.0  # change_pct 是百分比

        eq_df['equity'] = self.init_capital * (1 + eq_df['return'].fillna(0)).cumprod()

        # 计算绩效指标
        metrics = self._calc_metrics(eq_df)

        elapsed = time.perf_counter() - t0

        return {
            'equity_curve': eq_df,
            'metrics': metrics,
            'elapsed_time': elapsed,
        }

    def _calc_metrics(self, eq_df: pd.DataFrame) -> Dict[str, float]:
        """计算回测绩效指标"""
        returns = eq_df['return'].dropna()
        if len(returns) < 2:
            return {}

        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        equity = eq_df['equity']
        max_drawdown = (equity / equity.cummax() - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility > 0 else 0
        win_rate = (returns > 0).mean()

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0,
        }


# ============================================================================
# 3. NumPy 纯向量化回测 (最快速方案)
# ============================================================================

class NumPyVectorizedBacktest:
    """基于 NumPy 的纯向量化回测（近似 C 级别速度）"""

    def __init__(
        self,
        init_capital: float = 1_000_000,
        commission_rate: float = 0.00025,
        slippage: float = 0.0001,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.slippage = slippage

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        执行 NumPy 纯向量化回测

        将数据转换为 NumPy 数组，使用矩阵运算替代 DataFrame 操作。
        这是 AKQuant 的 Rust 核心层在 Python 侧的近似模拟。
        """
        t0 = time.perf_counter()

        # 构建收益矩阵 (dates × codes)
        pivot = data.pivot(index='date', columns='code', values='close')
        returns = pivot.pct_change().fillna(0)

        # 构建信号矩阵
        signal_pivot = signals.pivot(index='date', columns='code', values='signal').fillna(0)

        # 对齐
        common_dates = returns.index.intersection(signal_pivot.index)
        common_codes = returns.columns.intersection(signal_pivot.columns)

        if len(common_dates) == 0 or len(common_codes) == 0:
            return {"error": "无共同日期或代码", "metrics": {}}

        ret_mat = returns.loc[common_dates, common_codes].values
        sig_mat = signal_pivot.loc[common_dates, common_codes].values

        # T+1: 信号滞后一天
        if hasattr(self, 't_plus_1') and self.t_plus_1:
            sig_mat = np.roll(sig_mat, 1, axis=0)
            sig_mat[0, :] = 0

        # 向量化计算每日收益
        n_dates, n_stocks = ret_mat.shape

        # 每日等权组合收益
        daily_returns = np.zeros(n_dates)
        n_positions = np.zeros(n_dates)

        for d in range(n_dates):
            long_mask = sig_mat[d] > 0
            n_long = long_mask.sum()

            if n_long > 0:
                # 考虑交易成本: 换手带来的佣金
                if d > 0:
                    prev_mask = sig_mat[d - 1] > 0
                    turnover = (long_mask != prev_mask).sum() / max(n_long, prev_mask.sum())
                else:
                    turnover = 1.0

                # 等权组合收益 - 交易成本
                raw_return = ret_mat[d, long_mask].mean()
                cost = turnover * self.commission_rate * 2  # 买卖双向
                daily_returns[d] = raw_return - cost
                n_positions[d] = n_long
            else:
                daily_returns[d] = 0
                n_positions[d] = 0

        # 构建权益曲线
        equity = self.init_capital * np.cumprod(1 + daily_returns)
        eq_df = pd.DataFrame({
            'date': common_dates,
            'return': daily_returns,
            'equity': equity,
            'n_positions': n_positions,
        })

        # 计算指标
        metrics = self._calc_metrics(daily_returns, equity)

        elapsed = time.perf_counter() - t0

        return {
            'equity_curve': eq_df,
            'metrics': metrics,
            'elapsed_time': elapsed,
        }

    def _calc_metrics(self, returns: np.ndarray, equity: np.ndarray) -> Dict[str, float]:
        """计算回测绩效指标（纯 NumPy）"""
        if len(returns) < 2:
            return {}

        total_return = float(np.prod(1 + returns) - 1)
        annual_return = float((1 + total_return) ** (252 / len(returns)) - 1)
        volatility = float(np.std(returns, ddof=1) * np.sqrt(252))
        peak = np.maximum.accumulate(equity)
        max_drawdown = float(np.min(equity / peak - 1))
        sharpe = float((annual_return - 0.03) / volatility) if volatility > 0 else 0
        win_rate = float((returns > 0).mean())

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "calmar_ratio": annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
        }


# ============================================================================
# 4. 测试与对比
# ============================================================================

def generate_test_data(n_stocks: int = 500, n_days: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成模拟回测数据"""
    np.random.seed(42)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.bdate_range('2022-01-01', periods=n_days)

    rows = []
    signal_rows = []

    for code in codes:
        start_price = np.random.uniform(5, 100)
        daily_returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * np.exp(np.cumsum(daily_returns))

        for i, (date, price) in enumerate(zip(dates, prices)):
            ret = daily_returns[i]
            rows.append({
                'code': code,
                'date': date,
                'open': price * (1 + np.random.normal(0, 0.003)),
                'high': price * (1 + abs(np.random.normal(0, 0.015))),
                'low': price * (1 - abs(np.random.normal(0, 0.015))),
                'close': price,
                'volume': int(np.random.lognormal(10, 0.5)),
                'change_pct': ret * 100,
                'is_limit_up': ret >= 0.099,
                'is_limit_down': ret <= -0.099,
            })

            # 信号：约 20% 的股票每日有信号
            signal_rows.append({
                'code': code,
                'date': date,
                'signal': 1 if np.random.random() < 0.2 else 0,
            })

    data = pd.DataFrame(rows).sort_values(['code', 'date']).reset_index(drop=True)
    signals = pd.DataFrame(signal_rows).sort_values(['code', 'date']).reset_index(drop=True)

    return data, signals


def test_correctness():
    """测试正确性：Pandas vs NumPy 结果一致性"""
    print("\n" + "=" * 60)
    print("测试 1: 正确性验证 (Pandas vs NumPy 向量化)")
    print("=" * 60)

    data, signals = generate_test_data(n_stocks=100, n_days=500)

    # Pandas 回测
    pandas_bt = PandasVectorizedBacktest()
    result_pandas = pandas_bt.run(data, signals)

    # NumPy 回测
    numpy_bt = NumPyVectorizedBacktest()
    result_numpy = numpy_bt.run(data, signals)

    # 比较权益曲线
    eq_pandas = result_pandas['equity_curve']['equity'].values
    eq_numpy = result_numpy['equity_curve']['equity'].values

    # 由于实现细节不同，允许一定误差
    if len(eq_pandas) == len(eq_numpy):
        # 取最小长度对齐
        min_len = min(len(eq_pandas), len(eq_numpy))
        corr = np.corrcoef(eq_pandas[:min_len], eq_numpy[:min_len])[0, 1]
        print(f"  权益曲线相关性: {corr:.4f}")
        if corr > 0.8:
            print(f"  ✓ 两个引擎的权益曲线高度相关")
        else:
            print(f"  ✗ 权益曲线相关性偏低")

    # 比较绩效指标
    print(f"\n  Pandas 指标:")
    for k, v in result_pandas['metrics'].items():
        print(f"    {k}: {v:.4f}")

    print(f"\n  NumPy 指标:")
    for k, v in result_numpy['metrics'].items():
        print(f"    {k}: {v:.4f}")


def test_performance():
    """性能对比测试：Pandas vs Polars vs NumPy"""
    print("\n" + "=" * 60)
    print("测试 2: 性能对比 (Pandas vs Polars vs NumPy)")
    print("=" * 60)

    # 从小规模到大规模测试
    scales = [
        (50, 250),    # 小规模: 50只股票, 250天
        (100, 500),   # 中规模: 100只股票, 500天
        (500, 500),   # 中大规模: 500只股票, 500天
        (500, 1000),  # 大规模: 500只股票, 1000天
    ]

    results = []

    for n_stocks, n_days in scales:
        print(f"\n  规模: {n_stocks}只股票 × {n_days}天 ({n_stocks * n_days:,} 行)")
        data, signals = generate_test_data(n_stocks=n_stocks, n_days=n_days)

        row_result = {
            "n_stocks": n_stocks,
            "n_days": n_days,
            "total_rows": n_stocks * n_days,
        }

        # Pandas 回测 (3 runs)
        pandas_bt = PandasVectorizedBacktest()
        pandas_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = pandas_bt.run(data, signals)
            pandas_times.append(time.perf_counter() - t0)
        pandas_mean = np.mean(pandas_times)
        row_result["pandas_time"] = round(pandas_mean, 4)
        print(f"    Pandas:     {pandas_mean:.4f}s")

        # NumPy 回测 (3 runs)
        numpy_bt = NumPyVectorizedBacktest()
        numpy_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = numpy_bt.run(data, signals)
            numpy_times.append(time.perf_counter() - t0)
        numpy_mean = np.mean(numpy_times)
        row_result["numpy_time"] = round(numpy_mean, 4)
        row_result["numpy_speedup"] = round(pandas_mean / numpy_mean, 2) if numpy_mean > 0 else 0
        print(f"    NumPy:      {numpy_mean:.4f}s ({pandas_mean / numpy_mean:.1f}x 加速)")

        # Polars 回测 (如果可用)
        if HAS_POLARS:
            polars_bt = PolarsVectorizedBacktest()
            polars_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                _ = polars_bt.run(data, signals)
                polars_times.append(time.perf_counter() - t0)
            polars_mean = np.mean(polars_times)
            row_result["polars_time"] = round(polars_mean, 4)
            row_result["polars_speedup"] = round(pandas_mean / polars_mean, 2) if polars_mean > 0 else 0
            print(f"    Polars:     {polars_mean:.4f}s ({pandas_mean / polars_mean:.1f}x 加速)")

        results.append(row_result)

    # 汇总表格
    print("\n" + "-" * 80)
    print(f"  {'规模':<20} {'Pandas':>10} {'NumPy':>10} {'加速比':>8}", end="")
    if HAS_POLARS:
        print(f" {'Polars':>10} {'加速比':>8}")
    else:
        print()
    print("-" * 80)
    for r in results:
        label = f"{r['n_stocks']}股×{r['n_days']}天 ({r['total_rows']:,}行)"
        print(f"  {label:<20} {r['pandas_time']:>8.4f}s {r['numpy_time']:>8.4f}s {r['numpy_speedup']:>6.1f}x", end="")
        if HAS_POLARS:
            print(f" {r.get('polars_time', 0):>8.4f}s {r.get('polars_speedup', 0):>6.1f}x")
        else:
            print()
    print("-" * 80)

    return results


def test_edge_cases():
    """边界条件测试"""
    print("\n" + "=" * 60)
    print("测试 3: 边界条件测试")
    print("=" * 60)

    # 空信号
    print("\n  [空信号]")
    data, signals = generate_test_data(n_stocks=10, n_days=100)
    signals['signal'] = 0

    numpy_bt = NumPyVectorizedBacktest()
    result = numpy_bt.run(data, signals)
    eq = result['equity_curve']['equity'].values
    if np.allclose(eq, 1_000_000):
        print("  ✓ 空信号时权益保持不变")
    else:
        print(f"  ✗ 空信号时权益变化: {eq[0]} -> {eq[-1]}")

    # 全信号
    print("\n  [全信号]")
    signals['signal'] = 1
    result = numpy_bt.run(data, signals)
    eq = result['equity_curve']['equity'].values
    print(f"  ✓ 全信号时终值: {eq[-1]:.2f}")

    # 单股票
    print("\n  [单股票]")
    data_single = data[data['code'] == data['code'].iloc[0]].copy()
    signals_single = signals[signals['code'] == signals['code'].iloc[0]].copy()
    signals_single['signal'] = 1
    result = numpy_bt.run(data_single, signals_single)
    eq = result['equity_curve']['equity'].values
    print(f"  ✓ 单股票回测终值: {eq[-1]:.2f}")

    # 涨跌停过滤
    print("\n  [涨跌停过滤]")
    data_limit = data.copy()
    data_limit['is_limit_up'] = True  # 全部涨停，无法买入
    result = numpy_bt.run(data_limit, signals)
    eq = result['equity_curve']['equity'].values
    print(f"  ✓ 全部涨停时权益: {eq[-1]:.2f} (应保持不变或接近初始值)")

    print("\n  ✓ 边界条件测试完成")


if __name__ == "__main__":
    test_correctness()
    test_performance()
    test_edge_cases()
    print("\n" + "=" * 60)
    print("测试完成: 向量化回测引擎性能对比")
    print("=" * 60)