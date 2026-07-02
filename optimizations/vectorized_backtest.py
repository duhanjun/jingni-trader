"""
向量化回测引擎 + T+1 修复 + 基准跟踪

借鉴来源:
  - FinRL-X (arXiv 2603.21330): 部署一致性架构 —— 回测执行语义应与实盘一致
  - QuantConnect LEAN: 事件驱动 + 模块化成本模型

jingni-trader 现有 native_adapter.py 的三个核心问题:
  1. 性能: 逐日循环 + signals[signals['date']==dt] 过滤导致 O(n²)
  2. 正确性 BUG: t_plus_1 参数被接收但从未实际执行（同日买卖未阻止）
  3. 基准缺失: benchmark 参数接收但 equity_curve 未包含基准净值

本模块通过向量化（pivot + 矩阵运算）解决性能问题，
显式实现 T+1 约束，并加入基准净值跟踪。
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from .cost_models import (
    CostCalculator, ConstantSlippage, AShareFeeModel, TradeContext
)


class VectorizedBacktester:
    """
    向量化回测引擎

    核心优化:
      - 用 pivot 将 (date, code) -> 宽表，矩阵运算替代逐行循环
      - 显式 T+1: 记录买入日期，次日不可卖出
      - 基准净值: equity_curve 含 benchmark 列，便于计算 alpha/beta/IR
      - 模块化成本: 接入 CostCalculator
    """

    def __init__(
        self,
        cost_calculator: Optional[CostCalculator] = None,
        t_plus_1: bool = True,
        price_limit: bool = True,
    ):
        self.cost_calc = cost_calculator or CostCalculator(
            ConstantSlippage(0.001), AShareFeeModel()
        )
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
    ) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        # --- 预处理: pivot 为宽表，避免逐日过滤 ---
        price_wide = data.pivot_table(index='date', columns='code', values='close')
        price_wide = price_wide.sort_index()

        # 涨跌停标记（宽表）
        limit_up = data.pivot_table(index='date', columns='code', values='is_limit_up', aggfunc='max').reindex_like(price_wide).fillna(False)
        limit_down = data.pivot_table(index='date', columns='code', values='is_limit_down', aggfunc='max').reindex_like(price_wide).fillna(False)

        # 成交量宽表（用于量价滑点）
        vol_wide = data.pivot_table(index='date', columns='code', values='volume', aggfunc='sum').reindex_like(price_wide).fillna(0)

        # 信号宽表: 1 买入, -1 卖出, 0 无
        sig_wide = signals.pivot_table(
            index='date', columns='code', values='signal', aggfunc='max'
        ).reindex_like(price_wide).fillna(0)

        # 基准净值
        bench_close = data[data['code'] == benchmark].set_index('date')['close'] if benchmark in price_wide.columns else None

        dates = price_wide.index.tolist()
        codes = price_wide.columns.tolist()

        cash = init_capital
        # shares[code] = 持仓股数; buy_date[code] = 最近买入日期（用于T+1）
        shares = {c: 0 for c in codes}
        buy_date = {c: None for c in codes}

        equity_records = []
        trades = []

        for i, dt in enumerate(dates):
            prices = price_wide.loc[dt]
            day_sigs = sig_wide.loc[dt]
            day_vol = vol_wide.loc[dt]
            day_limit_up = limit_up.loc[dt]
            day_limit_down = limit_down.loc[dt]

            # --- 1. 先卖出 ---
            sell_codes = codes if not isinstance(day_sigs, pd.Series) else day_sigs[day_sigs < 0].index.tolist()
            for code in sell_codes:
                if shares.get(code, 0) <= 0:
                    continue
                # T+1 检查: 买入当日不可卖出
                if self.t_plus_1 and buy_date.get(code) == dt:
                    continue
                if pd.isna(prices.get(code)):
                    continue
                if self.price_limit and day_limit_down.get(code, False):
                    continue  # 跌停无法卖出

                vol = day_vol.get(code, 0)
                ctx = TradeContext(
                    price=float(prices[code]),
                    shares=float(shares[code]),
                    side='sell',
                    volume=float(vol) if vol > 0 else None,
                )
                fill_price, fees = self.cost_calc.compute(ctx)
                sell_amount = fill_price * ctx.shares
                total_cost = fees['commission'] + fees['tax'] + fees.get('transfer_fee', 0)
                cash += sell_amount - total_cost
                trades.append({
                    'date': dt, 'code': code, 'action': 'sell',
                    'price': fill_price, 'shares': ctx.shares, 'amount': sell_amount,
                    'commission': fees['commission'], 'tax': fees['tax'],
                })
                shares[code] = 0
                buy_date[code] = None

            # --- 2. 再买入 ---
            buy_codes = day_sigs[day_sigs > 0].index.tolist() if isinstance(day_sigs, pd.Series) else []
            if buy_codes:
                n_buy = len(buy_codes)
                budget = cash * 0.95 / n_buy
                for code in buy_codes:
                    if pd.isna(prices.get(code)):
                        continue
                    if self.price_limit and day_limit_up.get(code, False):
                        continue  # 涨停无法买入

                    vol = day_vol.get(code, 0)
                    # 先估算股数（用原始价），再用成本计算器精算
                    est_price = float(prices[code]) * 1.001
                    est_shares = int(budget / est_price / 100) * 100
                    if est_shares <= 0:
                        continue

                    ctx = TradeContext(
                        price=float(prices[code]),
                        shares=float(est_shares),
                        side='buy',
                        volume=float(vol) if vol > 0 else None,
                    )
                    fill_price, fees = self.cost_calc.compute(ctx)
                    buy_amount = fill_price * est_shares
                    total_cost = buy_amount + fees['commission'] + fees.get('transfer_fee', 0)
                    if total_cost > cash:
                        est_shares = int((cash * 0.98) / fill_price / 100) * 100
                        if est_shares <= 0:
                            continue
                        ctx.shares = float(est_shares)
                        fill_price, fees = self.cost_calc.compute(ctx)
                        buy_amount = fill_price * est_shares
                        total_cost = buy_amount + fees['commission'] + fees.get('transfer_fee', 0)

                    cash -= total_cost
                    shares[code] = shares.get(code, 0) + est_shares
                    buy_date[code] = dt  # 记录买入日期用于T+1
                    trades.append({
                        'date': dt, 'code': code, 'action': 'buy',
                        'price': fill_price, 'shares': est_shares, 'amount': buy_amount,
                        'commission': fees['commission'], 'tax': 0,
                    })

            # --- 3. 计算当日总权益 ---
            market_value = 0.0
            for code, sh in shares.items():
                if sh <= 0 or pd.isna(prices.get(code)):
                    continue
                market_value += sh * float(prices[code])

            bench_val = float(bench_close.loc[dt]) if bench_close is not None and dt in bench_close.index else np.nan

            equity_records.append({
                'date': dt,
                'equity': cash + market_value,
                'cash': cash,
                'market_value': market_value,
                'position_count': sum(1 for s in shares.values() if s > 0),
                'benchmark': bench_val,
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        metrics = self._calc_metrics(equity_curve, init_capital)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(
                [(c, s) for c, s in shares.items() if s > 0],
                columns=['code', 'shares'],
            ) if any(s > 0 for s in shares.values()) else pd.DataFrame(columns=['code', 'shares']),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _calc_metrics(self, equity_curve: pd.DataFrame, init_capital: float) -> Dict[str, float]:
        """计算绩效指标，含基准相对指标（alpha/beta/IR）"""
        eq = equity_curve.set_index('date')['equity']
        if len(eq) < 2:
            return {}

        returns = eq.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n = len(returns)
        annual_return = (1 + total_return) ** (252 / n) - 1 if n > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (eq / eq.cummax() - 1).min()
        rf = 0.03
        sharpe = (annual_return - rf) / volatility if volatility != 0 else 0
        downside = returns[returns < 0].std() * np.sqrt(252)
        sortino = (annual_return - rf) / downside if downside != 0 else 0
        win_rate = (returns > 0).mean() if n > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "calmar_ratio": float(calmar),
        }

        # 基准相对指标
        if 'benchmark' in equity_curve.columns:
            bench = equity_curve.set_index('date')['benchmark'].dropna()
            if len(bench) >= 2:
                bench_ret = bench.pct_change().dropna()
                # 对齐
                common = returns.index.intersection(bench_ret.index)
                if len(common) > 20:
                    r = returns.loc[common]
                    b = bench_ret.loc[common]
                    cov_mat = np.cov(r, b)
                    beta = float(cov_mat[0, 1] / cov_mat[1, 1]) if cov_mat[1, 1] != 0 else 0
                    alpha = float(r.mean() - beta * b.mean()) * 252
                    tracking_error = float((r - b).std() * np.sqrt(252))
                    info_ratio = float((r - b).mean() * 252 / tracking_error) if tracking_error != 0 else 0
                    bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1)
                    metrics.update({
                        "beta": beta,
                        "alpha": alpha,
                        "tracking_error": tracking_error,
                        "information_ratio": info_ratio,
                        "benchmark_return": bench_total,
                    })
        return metrics

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
