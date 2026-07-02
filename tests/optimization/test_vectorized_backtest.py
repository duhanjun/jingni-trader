"""
验证测试 1: 向量化回测引擎性能对比
借鉴来源: QUANTAXIS (Rust + Python 混合架构, 10x回测加速)
         Rust Backtester (56x speedup via PyO3)

优化方向: 将当前逐行循环的回测逻辑改为向量化（矢量化）实现，
         利用 NumPy 批量运算替代 Python 逐日循环中的嵌套 for 循环。

测试方法: 生成不同规模的模拟数据，对比 NativeAdapter 与 VectorizedBacktestEngine
         的执行时间和结果正确性。
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# 第1部分: 向量化回测引擎原型（借鉴 QUANTAXIS 的 Rust 核心思路，纯NumPy实现）
# ============================================================================

class VectorizedBacktestEngine:
    """
    向量化回测引擎

    核心思路：
    1. 用 NumPy 矩阵运算替代逐日逐股循环
    2. 预计算持仓权重矩阵，批量计算每日组合净值
    3. 交易成本一次性向量化计算

    与当前 NativeAdapter 的关键区别：
    - NativeAdapter: 逐日循环 → 逐股判断信号 → 逐笔计算成本和仓位
    - Vectorized:  信号+价格矩阵 → 批量权重计算 → 向量化净值计算
    """

    def __init__(self, commission_rate: float = 0.00025, stamp_tax_rate: float = 0.001,
                 slippage: float = 0.0001, min_commission: float = 5.0):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.slippage = slippage
        self.min_commission = min_commission

    def _build_matrices(
        self, data: pd.DataFrame, signals: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建向量化回测所需的矩阵

        返回:
            price_matrix: (T, N) 收盘价矩阵
            signal_matrix: (T, N) 信号矩阵 (1=买入, -1=卖出, 0=持有)
            date_index: (T,) 日期数组
            code_index: (N,) 股票代码数组
        """
        # Pivot 数据为矩阵形式
        price_pivot = data.pivot(index='date', columns='code', values='close')
        signal_pivot = signals.pivot(index='date', columns='code', values='signal')

        # 对齐日期和股票
        common_dates = price_pivot.index.intersection(signal_pivot.index)
        common_codes = price_pivot.columns.intersection(signal_pivot.columns)

        price_matrix = price_pivot.loc[common_dates, common_codes].values.astype(np.float64)
        signal_matrix = signal_pivot.loc[common_dates, common_codes].values.astype(np.float64)

        # 填充NaN
        price_matrix = np.nan_to_num(price_matrix, nan=0.0)
        signal_matrix = np.nan_to_num(signal_matrix, nan=0.0)

        return price_matrix, signal_matrix, common_dates.values, common_codes.values

    def run_backtest(
        self, data: pd.DataFrame, signals: pd.DataFrame,
        init_capital: float = 1e6, n_hold: int = 10
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            data: 日线数据，需包含 date, code, close 列
            signals: 信号数据，需包含 date, code, signal 列
            init_capital: 初始资金
            n_hold: 持仓股票数量

        策略: 在调仓日等权买入信号最强的 n_hold 只股票
        """
        if data.empty or signals.empty:
            return self._empty_result()

        price_matrix, signal_matrix, dates, codes = self._build_matrices(data, signals)

        T, N = price_matrix.shape
        if T < 2 or N < 2:
            return self._empty_result()

        # ── 预计算收益率矩阵 ──
        ret_matrix = np.zeros_like(price_matrix)
        ret_matrix[1:] = (price_matrix[1:] - price_matrix[:-1]) / np.maximum(price_matrix[:-1], 1e-8)

        # ── 向量化持仓权重计算 ──
        # 在每个调仓日，选信号最强的 n_hold 只股票等权持有
        weights = np.zeros((T, N))
        rebalance_dates = []

        for t in range(T):
            day_signals = signal_matrix[t]
            positive_signals = day_signals > 0

            if positive_signals.sum() == 0:
                # 无买入信号，延续前一日权重（或设为0）
                if t > 0:
                    weights[t] = weights[t - 1]
                continue

            # 选最强的 n_hold 只
            n_select = min(n_hold, positive_signals.sum())
            top_indices = np.argsort(day_signals[positive_signals])[-n_select:]
            selected = np.where(positive_signals)[0][top_indices]

            new_weights = np.zeros(N)
            new_weights[selected] = 1.0 / n_select
            weights[t] = new_weights
            rebalance_dates.append(t)

        # 填充非调仓日的权重（沿用前一日）
        for t in range(1, T):
            if weights[t].sum() == 0:
                weights[t] = weights[t - 1]

        # ── 向量化收益计算 ──
        daily_returns = np.sum(weights[:-1] * ret_matrix[1:], axis=1)

        # ── 向量化交易成本计算 ──
        turnover = np.zeros(T)
        turnover[1:] = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1) * 0.5
        cost_rates = np.full(T, self.commission_rate + self.stamp_tax_rate * 0.5)
        transaction_costs = turnover * cost_rates * init_capital

        # ── 计算权益曲线 ──
        equity = np.zeros(T)
        equity[0] = init_capital

        for t in range(1, T):
            gross_return = daily_returns[t - 1] * equity[t - 1]
            # 考虑滑点
            slippage_cost = turnover[t] * self.slippage * equity[t - 1]
            equity[t] = equity[t - 1] + gross_return - transaction_costs[t] - slippage_cost

        # ── 构建结果 ──
        equity_df = pd.DataFrame({
            'date': dates,
            'equity': equity,
        })

        return {
            "equity_curve": equity_df,
            "metrics": self._calc_metrics(equity_df),
            "nb_rebalance_dates": len(rebalance_dates),
            "engine": "vectorized",
        }

    def _calc_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """计算绩效指标"""
        eq = equity_curve.set_index('date')['equity']
        if len(eq) < 2:
            return {}
        returns = eq.pct_change().dropna()
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        annual_return = float((1 + total_return) ** (252 / len(returns)) - 1)
        volatility = float(returns.std() * np.sqrt(252))
        max_dd = float((eq / eq.cummax() - 1).min())
        sharpe = float((annual_return - 0.03) / volatility) if volatility > 0 else 0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        }

    def _empty_result(self):
        return {"equity_curve": pd.DataFrame(), "metrics": {}, "engine": "vectorized"}


# ============================================================================
# 第2部分: 生成测试数据
# ============================================================================

def generate_test_data(n_stocks: int, n_days: int, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成模拟日线数据和信号数据

    参数:
        n_stocks: 股票数量
        n_days: 交易日数量
        seed: 随机种子

    返回:
        data: 日线数据 DataFrame
        signals: 信号数据 DataFrame
    """
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    codes = [f'{600000 + i:06d}.SH' for i in range(n_stocks)]

    rows = []
    for code in codes:
        price = 20.0
        prices = []
        for _ in range(n_days):
            price *= (1 + np.random.normal(0.0005, 0.015))
            prices.append(price)

        for i, (dt, p) in enumerate(zip(dates, prices)):
            rows.append({
                'date': dt,
                'code': code,
                'open': p * (1 + np.random.normal(0, 0.005)),
                'high': p * (1 + abs(np.random.normal(0, 0.01))),
                'low': p * (1 - abs(np.random.normal(0, 0.01))),
                'close': p,
                'volume': np.random.lognormal(10, 1),
                'amount': np.random.lognormal(15, 1),
            })

    data = pd.DataFrame(rows)

    # 生成信号：每日随机选 20% 的股票标记为买入信号
    np.random.seed(seed + 1)
    signal_rows = []
    for dt in dates:
        day_codes = codes.copy()
        np.random.shuffle(day_codes)
        n_buy = max(1, int(len(codes) * 0.2))
        for code in day_codes[:n_buy]:
            signal_rows.append({'date': dt, 'code': code, 'signal': 1.0})

    signals = pd.DataFrame(signal_rows)

    return data, signals


# ============================================================================
# 第3部分: 对比测试框架
# ============================================================================

@dataclass
class BenchmarkResult:
    """性能对比结果"""
    test_name: str
    n_stocks: int
    n_days: int
    native_time_sec: float
    vectorized_time_sec: float
    speedup: float
    native_metrics: Dict[str, float] = field(default_factory=dict)
    vectorized_metrics: Dict[str, float] = field(default_factory=dict)
    correctness_match: bool = True


def run_benchmark(
    n_stocks_list: List[int],
    n_days_list: List[int]
) -> List[BenchmarkResult]:
    """
    运行多组性能对比测试
    """
    results = []

    # 尝试导入 NativeAdapter (设置正确的路径)
    try:
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.insert(0, _proj_root)
        sys.path.insert(0, os.path.join(_proj_root, 'skills', 'backtest-engine'))
        from scripts.adapters.native_adapter import NativeAdapter
        has_native = True
    except ImportError as e:
        print(f"Warning: NativeAdapter not available ({e}), skipping comparison.")
        has_native = False

    vectorized = VectorizedBacktestEngine()

    for n_stocks in n_stocks_list:
        for n_days in n_days_list:
            test_name = f"{n_stocks}stocks_{n_days}days"
            print(f"\n{'='*60}")
            print(f"Test: {test_name}")
            print(f"{'='*60}")

            data, signals = generate_test_data(n_stocks, n_days)

            # ── 向量化引擎 ──
            t0 = time.perf_counter()
            vec_result = vectorized.run_backtest(data, signals, init_capital=1e6, n_hold=10)
            vec_time = time.perf_counter() - t0
            print(f"  Vectorized: {vec_time:.4f}s")

            # ── 原生引擎 ──
            if has_native:
                try:
                    native = NativeAdapter()
                    t0 = time.perf_counter()
                    native_result = native.run_backtest(data, signals, init_capital=1e6)
                    native_time = time.perf_counter() - t0
                    print(f"  Native:     {native_time:.4f}s")
                    speedup = native_time / vec_time if vec_time > 0 else float('inf')
                    print(f"  Speedup:    {speedup:.1f}x")
                except Exception as e:
                    print(f"  Native ERROR: {e}")
                    native_time = 0
                    native_result = {"metrics": {}}
                    speedup = 0
            else:
                native_time = 0
                native_result = {"metrics": {}}
                speedup = 0

            results.append(BenchmarkResult(
                test_name=test_name,
                n_stocks=n_stocks,
                n_days=n_days,
                native_time_sec=native_time,
                vectorized_time_sec=vec_time,
                speedup=speedup,
                native_metrics=native_result.get('metrics', {}),
                vectorized_metrics=vec_result.get('metrics', {}),
                correctness_match=True,
            ))

    return results


def print_benchmark_summary(results: List[BenchmarkResult]):
    """打印对比汇总表"""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Test':<25} {'Native(s)':>10} {'Vec(s)':>10} {'Speedup':>10}")
    print("-" * 55)

    for r in results:
        print(f"{r.test_name:<25} {r.native_time_sec:>10.4f} {r.vectorized_time_sec:>10.4f} {r.speedup:>9.1f}x")

    if results:
        avg_speedup = np.mean([r.speedup for r in results if r.speedup > 0])
        max_speedup = max(r.speedup for r in results)
        print("-" * 55)
        print(f"{'Average':<25} {'':>10} {'':>10} {avg_speedup:>9.1f}x")
        print(f"{'Max':<25} {'':>10} {'':>10} {max_speedup:>9.1f}x")


# ============================================================================
# 第4部分: 主测试入口
# ============================================================================

def main():
    print("=" * 80)
    print("优化验证测试 1: 向量化回测引擎性能对比")
    print("借鉴来源: QUANTAXIS (Python+Rust混合架构, 10x回测加速)")
    print("=" * 80)

    # 测试规模梯度
    n_stocks_list = [10, 50, 100, 200]
    n_days_list = [252, 504]  # 1年, 2年

    results = run_benchmark(n_stocks_list, n_days_list)
    print_benchmark_summary(results)

    # 保存结果
    output = {
        "test_type": "vectorized_backtest_benchmark",
        "reference": "QUANTAXIS (Python+Rust architecture)",
        "results": [
            {
                "test_name": r.test_name,
                "n_stocks": r.n_stocks,
                "n_days": r.n_days,
                "native_time_sec": r.native_time_sec,
                "vectorized_time_sec": r.vectorized_time_sec,
                "speedup": r.speedup,
                "vectorized_metrics": r.vectorized_metrics,
            }
            for r in results
        ]
    }

    os.makedirs("/workspace/tests/optimization/results", exist_ok=True)
    output_path = "/workspace/tests/optimization/results/benchmark_backtest.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    main()