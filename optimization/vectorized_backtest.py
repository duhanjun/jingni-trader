"""
向量化回测引擎 (Vectorized Backtest Engine)

借鉴来源: VectorBT (https://github.com/polakowo/vectorbt)
核心思想: 将策略表示为多维数组，用 NumPy 向量化运算替代逐 bar 的 Python 循环，
         从而在参数搜索和大规模回测中获得数量级的性能提升。

与 jingni-trader 现有 native_adapter.py (逐日 for 循环) 的对比:
  - 原实现: for dt in dates: 逐日处理信号、买卖、记账 (O(N_days) 次 Python 循环)
  - 本实现: 将信号转为目标权重矩阵，用矩阵运算一次性完成换手、成本、净值计算

同时借鉴 FinRL-X 的 weight-centric 接口思想:
  策略层输出目标权重向量 w_t，回测/执行层共享同一套权重→订单的转换逻辑。

本文件为验证性实现，不修改 main 分支的任何代码。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """回测配置 (借鉴 FinRL-X Pydantic 配置思想，此处用 dataclass 轻量实现)"""
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025      # 双边佣金率
    min_commission: float = 5.0           # 单笔最低佣金
    stamp_tax_rate: float = 0.001         # 印花税 (卖出)
    slippage: float = 0.001               # 滑点
    t_plus_1: bool = True                 # A股 T+1 规则
    price_limit: bool = True              # 涨跌停限制
    risk_free_rate: float = 0.03          # 无风险利率 (年化)
    trading_days_per_year: int = 252


@dataclass
class VectorizedBacktestResult:
    """回测结果"""
    equity_curve: pd.DataFrame
    weights_history: pd.DataFrame
    turnover_series: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float] = field(default_factory=dict)


class VectorizedBacktester:
    """
    向量化回测引擎

    核心创新:
    1. 权重中心接口 (借鉴 FinRL-X): 策略输出目标权重矩阵 W (dates x codes)
    2. 向量化换手计算: turnover = |W_t - W_{t-1}|，无需逐日循环
    3. 向量化成本计算: 成本 = turnover * 成本率，矩阵运算
    4. 向量化净值: 用累计收益的向量化计算替代逐日记账

    A股规则处理:
    - T+1: 目标权重延迟一天生效 (W_effective[t] = W_target[t-1])
    - 涨跌停: 涨停日买入权重清零，跌停日卖出权重清零 (向量化掩码)
    - 印花税: 仅卖出时收取 (换手的一半)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run_from_weights(
        self,
        prices: pd.DataFrame,
        target_weights: pd.DataFrame,
        is_limit_up: Optional[pd.DataFrame] = None,
        is_limit_down: Optional[pd.DataFrame] = None,
    ) -> VectorizedBacktestResult:
        """
        基于目标权重矩阵的向量化回测

        参数:
            prices: 收盘价矩阵 (dates x codes), 已前复权
            target_weights: 目标权重矩阵 (dates x codes), 每行之和应 <= 1
            is_limit_up: 涨停标记矩阵 (dates x codes), True 表示当日涨停
            is_limit_down: 跌停标记矩阵 (dates x codes), True 表示当日跌停

        返回:
            VectorizedBacktestResult
        """
        cfg = self.config

        # 空数据处理
        if prices.empty or target_weights.empty:
            return VectorizedBacktestResult(
                equity_curve=pd.DataFrame(columns=['equity', 'return', 'turnover', 'transaction_cost']),
                weights_history=pd.DataFrame(),
                turnover_series=pd.Series(dtype=float),
                trades=pd.DataFrame(columns=['date', 'code', 'action', 'weight_delta', 'price']),
                metrics={},
            )

        # 对齐索引
        prices = prices.sort_index()
        target_weights = target_weights.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        if is_limit_up is not None:
            is_limit_up = is_limit_up.reindex(index=prices.index, columns=prices.columns).fillna(False)
        if is_limit_down is not None:
            is_limit_down = is_limit_down.reindex(index=prices.index, columns=prices.columns).fillna(False)

        # 转为 numpy 加速
        P = prices.values.astype(float)               # (T, N) 价格
        W_target = target_weights.values.astype(float)  # (T, N) 目标权重

        # 涨跌停掩码
        limit_up = is_limit_up.values if is_limit_up is not None else np.zeros_like(P, dtype=bool)
        limit_down = is_limit_down.values if is_limit_down is not None else np.zeros_like(P, dtype=bool)

        T, N = P.shape

        # ---- T+1 规则: 实际持仓权重延迟一天 ----
        if cfg.t_plus_1:
            W_effective = np.zeros_like(W_target)
            W_effective[1:] = W_target[:-1]  # 昨天的目标权重 = 今天的实际持仓
        else:
            W_effective = W_target.copy()

        # ---- 涨跌停处理 (向量化掩码) ----
        # 涨停日无法买入: 新增权重清零
        # 跌停日无法卖出: 减少的权重清零 (保持原仓位)
        if cfg.price_limit:
            # 买入受限: 涨停时该标的的目标权重增量置零
            W_prev = np.zeros_like(W_effective)
            W_prev[1:] = W_effective[:-1]
            weight_increase = np.maximum(W_effective - W_prev, 0)
            weight_increase = np.where(limit_up, 0, weight_increase)
            weight_decrease = np.maximum(W_prev - W_effective, 0)
            weight_decrease = np.where(limit_down, 0, weight_decrease)
            W_effective = W_prev + weight_increase - weight_decrease

        # 归一化: 确保权重和 <= 1 (现金补足)
        row_sums = W_effective.sum(axis=1, keepdims=True)
        W_effective = np.where(row_sums > 1, W_effective / np.where(row_sums > 0, row_sums, 1), W_effective)

        # ---- 向量化计算: 每日收益率 ----
        # 个股日收益率
        returns = np.zeros_like(P)
        returns[1:] = P[1:] / P[:-1] - 1.0
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # 组合日收益率 = sum(权重 * 个股收益)
        # 注意: 用 t-1 的权重乘以 t 的收益 (持仓在 t-1 收盘时确定)
        W_for_return = np.zeros_like(W_effective)
        W_for_return[1:] = W_effective[:-1]
        portfolio_returns = (W_for_return * returns).sum(axis=1)
        # 现金部分收益为0, 已隐含 (权重和 < 1 的部分)

        # ---- 向量化换手率计算 ----
        # turnover[t] = sum(|W[t] - W[t-1]|) / 2  (双边换手)
        W_diff = np.zeros_like(W_effective)
        W_diff[1:] = W_effective[1:] - W_effective[:-1]
        turnover = np.abs(W_diff).sum(axis=1) / 2.0

        # ---- 向量化交易成本 ----
        # 买入成本 = turnover_buy * (commission + slippage)
        # 卖出成本 = turnover_sell * (commission + stamp_tax + slippage)
        turnover_buy = np.maximum(W_diff, 0).sum(axis=1)
        turnover_sell = np.maximum(-W_diff, 0).sum(axis=1)

        buy_cost_rate = cfg.commission_rate + cfg.slippage
        sell_cost_rate = cfg.commission_rate + cfg.stamp_tax_rate + cfg.slippage

        # 成本以占组合总值的比例表示
        transaction_costs = turnover_buy * buy_cost_rate + turnover_sell * sell_cost_rate

        # ---- 向量化净值曲线 ----
        # net_return[t] = portfolio_return[t] - transaction_cost[t]
        net_returns = portfolio_returns - transaction_costs
        equity = cfg.init_capital * np.cumprod(1.0 + net_returns)

        # ---- 构建结果 DataFrame ----
        equity_curve = pd.DataFrame({
            'date': prices.index,
            'equity': equity,
            'return': net_returns,
            'turnover': turnover,
            'transaction_cost': transaction_costs * cfg.init_capital,
        }).set_index('date')

        weights_history = pd.DataFrame(
            W_effective, index=prices.index, columns=prices.columns
        )

        turnover_series = pd.Series(turnover, index=prices.index, name='turnover')

        # 提取交易记录 (权重变化不为零的点)
        trades = self._extract_trades(W_diff, P, prices.index, prices.columns)

        # 计算绩效指标
        metrics = self._calc_metrics(net_returns, equity)

        return VectorizedBacktestResult(
            equity_curve=equity_curve,
            weights_history=weights_history,
            turnover_series=turnover_series,
            trades=trades,
            metrics=metrics,
        )

    def run_from_signals(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        is_limit_up: Optional[pd.DataFrame] = None,
        is_limit_down: Optional[pd.DataFrame] = None,
        top_n: int = 20,
    ) -> VectorizedBacktestResult:
        """
        借鉴 VectorBT 的 from_signals API: 从买卖信号生成等权目标权重

        参数:
            prices: 收盘价矩阵 (dates x codes)
            signals: 信号矩阵 (dates x codes), >0 买入, <0 卖出, 0 持有
            top_n: 每日最多持有的股票数 (等权分配 1/top_n)
        """
        # 向量化: 将信号转为目标权重
        target_weights = self._signals_to_weights(signals, top_n)
        return self.run_from_weights(prices, target_weights, is_limit_up, is_limit_down)

    def _signals_to_weights(self, signals: pd.DataFrame, top_n: int) -> pd.DataFrame:
        """
        向量化信号转权重:
        - 买入信号 (>0) 的标的等权分配 1/top_n
        - 卖出信号 (<0) 的标的权重清零
        - 涨跌停过滤在 run_from_weights 中处理
        """
        S = signals.reindex(index=signals.index).fillna(0).values.astype(float)
        T, N = S.shape

        # 买入掩码
        buy_mask = S > 0
        # 每日买入数量
        daily_buy_count = buy_mask.sum(axis=1, keepdims=True)
        daily_buy_count = np.where(daily_buy_count > top_n, top_n, daily_buy_count)

        # 等权权重
        W = np.zeros_like(S)
        nonzero = daily_buy_count.flatten() > 0
        W[nonzero] = np.where(
            buy_mask[nonzero],
            1.0 / np.maximum(daily_buy_count[nonzero], 1),
            0.0,
        )
        return pd.DataFrame(W, index=signals.index, columns=signals.columns)

    def _extract_trades(
        self,
        W_diff: np.ndarray,
        prices: np.ndarray,
        dates: pd.Index,
        codes: pd.Index,
    ) -> pd.DataFrame:
        """向量化提取交易记录 (权重变化点)"""
        rows = []
        T, N = W_diff.shape
        for t in range(1, T):
            for n in range(N):
                delta = W_diff[t, n]
                if abs(delta) < 1e-8:
                    continue
                rows.append({
                    'date': dates[t],
                    'code': codes[n],
                    'action': 'buy' if delta > 0 else 'sell',
                    'weight_delta': delta,
                    'price': prices[t, n],
                })
        if not rows:
            return pd.DataFrame(columns=['date', 'code', 'action', 'weight_delta', 'price'])
        return pd.DataFrame(rows)

    def _calc_metrics(self, returns: np.ndarray, equity: np.ndarray) -> Dict[str, float]:
        """向量化计算绩效指标"""
        cfg = self.config
        if len(returns) < 2:
            return {}

        returns = np.nan_to_num(returns, nan=0.0)
        cumulative = equity[-1] / equity[0] - 1
        n_years = len(returns) / cfg.trading_days_per_year
        annual_return = (1 + cumulative) ** (1 / n_years) - 1 if n_years > 0 else 0

        volatility = np.std(returns, ddof=1) * np.sqrt(cfg.trading_days_per_year)
        sharpe = (annual_return - cfg.risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤 (向量化)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio (只用下行波动)
        downside = returns[returns < 0]
        downside_vol = np.std(downside, ddof=1) * np.sqrt(cfg.trading_days_per_year) if len(downside) > 1 else 0
        sortino = (annual_return - cfg.risk_free_rate) / downside_vol if downside_vol > 0 else 0

        return {
            'total_return': float(cumulative),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'calmar_ratio': float(calmar),
        }


def benchmark_against_loop(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """
    性能对比基准: 向量化回测 vs 逐日循环回测

    返回两种实现的耗时和结果对比
    """
    results = {'vectorized_times': [], 'loop_times': [], 'n_days': len(prices), 'n_codes': prices.shape[1]}

    # 向量化回测
    vt = VectorizedBacktester()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        v_result = vt.run_from_weights(prices, target_weights)
        t1 = time.perf_counter()
        results['vectorized_times'].append(t1 - t0)

    # 逐日循环回测 (模拟 native_adapter.py 的 for dt in dates 模式)
    # 使用与原 native_adapter 相同的 DataFrame 逐日过滤模式
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _loop_backtest(prices, target_weights)
        t1 = time.perf_counter()
        results['loop_times'].append(t1 - t0)

    results['vectorized_avg'] = float(np.mean(results['vectorized_times']))
    results['loop_avg'] = float(np.mean(results['loop_times']))
    results['speedup'] = results['loop_avg'] / results['vectorized_avg'] if results['vectorized_avg'] > 0 else 0
    results['vectorized_metrics'] = v_result.metrics
    return results


def benchmark_parameter_sweep(
    prices: pd.DataFrame,
    n_configs: int = 50,
    n_runs: int = 3,
) -> Dict[str, Any]:
    """
    参数搜索性能对比 (VectorBT 的核心优势场景)

    对比在参数搜索场景下:
    - 向量化: 一次性生成所有配置的权重矩阵，批量回测
    - 循环: 逐个配置循环回测

    参数:
        n_configs: 参数配置数量 (不同的 top_n / rebalance_freq 组合)
    """
    cfg = BacktestConfig()
    T, N = prices.shape
    configs = [(top_n, freq) for top_n in [5, 10, 15, 20, 30] for freq in [3, 5, 10, 15, 20]]
    configs = configs[:n_configs]

    vt = VectorizedBacktester()

    # 向量化批量: 逐配置但用向量化引擎
    v_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        v_results = []
        for top_n, freq in configs:
            weights = _gen_weights(prices, top_n, freq)
            r = vt.run_from_weights(prices, weights)
            v_results.append(r.metrics.get('sharpe_ratio', 0))
        t1 = time.perf_counter()
        v_times.append(t1 - t0)

    # 循环批量: 逐配置用循环引擎
    l_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        l_results = []
        for top_n, freq in configs:
            weights = _gen_weights(prices, top_n, freq)
            r = _loop_backtest(prices, weights)
            l_results.append(r.get('sharpe_ratio', 0))
        t1 = time.perf_counter()
        l_times.append(t1 - t0)

    return {
        'n_configs': len(configs),
        'vectorized_total': float(np.mean(v_times)),
        'loop_total': float(np.mean(l_times)),
        'speedup': float(np.mean(l_times) / np.mean(v_times)) if np.mean(v_times) > 0 else 0,
        'vectorized_sharpes': v_results,
        'loop_sharpes': l_results,
    }


def _gen_weights(prices: pd.DataFrame, top_n: int, freq: int) -> pd.DataFrame:
    """生成权重矩阵"""
    np.random.seed(42)
    T, N = prices.shape
    weights = np.zeros((T, N))
    for t in range(0, T, freq):
        selected = np.random.choice(N, size=min(top_n, N), replace=False)
        for s in selected:
            for tt in range(t, min(t + freq, T)):
                weights[tt, s] = 1.0 / top_n
    return pd.DataFrame(weights, index=prices.index, columns=prices.columns)


def _loop_backtest(prices: pd.DataFrame, target_weights: pd.DataFrame) -> Dict[str, float]:
    """
    逐日循环回测 (模拟 native_adapter.py 的 for dt in dates 模式)
    使用与原 native_adapter 相同的逐日 DataFrame 操作模式，用于性能对比基准

    关键: 这里使用 prices DataFrame 的逐日 .loc[date] 访问模式，
    与 native_adapter.py 中 `day_data = data[data['date'] == dt]` 的模式一致，
    体现了逐日循环 + DataFrame 过滤的开销。
    """
    cfg = BacktestConfig()
    # 转为长表以模拟 native_adapter 的逐日过滤模式
    prices_long = prices.stack().reset_index()
    prices_long.columns = ['date', 'code', 'close']
    weights_long = target_weights.stack().reset_index()
    weights_long.columns = ['date', 'code', 'weight']

    dates = sorted(prices.index.unique())
    cash = cfg.init_capital
    positions = {}  # code -> shares
    equity_list = []

    for dt in dates:
        day_prices = prices_long[prices_long['date'] == dt].set_index('code')
        day_weights = weights_long[weights_long['date'] == dt].set_index('code')

        # T+1: 用昨天的目标权重
        dt_idx = dates.index(dt)
        if cfg.t_plus_1 and dt_idx > 0:
            prev_dt = dates[dt_idx - 1]
            day_weights = weights_long[weights_long['date'] == prev_dt].set_index('code')

        # 计算总市值
        market_value = sum(shares * day_prices.loc[c, 'close']
                          for c, shares in positions.items() if c in day_prices.index)
        total_value = cash + market_value

        # 计算目标持仓
        target_positions = {}
        for code in day_weights.index:
            w = day_weights.loc[code, 'weight']
            if w > 0 and code in day_prices.index:
                target_value = w * total_value
                price = day_prices.loc[code, 'close']
                target_positions[code] = target_value / price

        # 换手
        all_codes = set(positions.keys()) | set(target_positions.keys())
        buy_value = 0
        sell_value = 0
        for code in all_codes:
            old = positions.get(code, 0)
            new = target_positions.get(code, 0)
            price = day_prices.loc[code, 'close'] if code in day_prices.index else 0
            delta = new - old
            if delta > 0:
                buy_value += delta * price
            else:
                sell_value += (-delta) * price

        buy_cost = max(buy_value * cfg.commission_rate, 5) if buy_value > 0 else 0
        sell_cost = max(sell_value * cfg.commission_rate, 5) if sell_value > 0 else 0
        sell_tax = sell_value * cfg.stamp_tax_rate

        cash += sell_value - buy_value - buy_cost - sell_cost - sell_tax
        positions = target_positions
        mv = sum(shares * day_prices.loc[c, 'close']
                for c, shares in positions.items() if c in day_prices.index)
        equity_list.append(cash + mv)

    equity = np.array(equity_list)
    returns = np.diff(equity) / equity[:-1]
    returns = np.insert(returns, 0, 0)

    cumulative = equity[-1] / equity[0] - 1
    n_years = len(dates) / cfg.trading_days_per_year
    annual_return = (1 + cumulative) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = np.std(returns, ddof=1) * np.sqrt(cfg.trading_days_per_year)
    sharpe = (annual_return - cfg.risk_free_rate) / volatility if volatility > 0 else 0
    running_max = np.maximum.accumulate(equity)
    max_dd = ((equity - running_max) / running_max).min()

    return {
        'total_return': float(cumulative),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
    }
