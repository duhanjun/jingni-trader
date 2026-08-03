"""
原生回测引擎适配器
纯 pandas/numpy 实现，不依赖外部回测框架
支持A股 T+1、涨跌停、印花税、过户费等规则

合并 NativeAdapterV2 的 6 项修复：
1. T+1 真正实现：记录每只股票的最近买入日期，卖出时强制 current_date > buy_date
2. PnL 正确计算：positions 记录 avg_cost，卖出 pnl = (sell_price - avg_cost) * shares - costs
3. 滑点双侧应用：卖出价也乘以 (1 - slippage)
4. 过户费计算：买卖双侧加 transfer_fee = amount * transfer_fee_rate
5. 基准对比：equity_curve 增加 benchmark 列，metrics 增加 alpha/beta/excess_return
6. 成本分离：metrics 同时输出 gross_return（不含费用）与 net_return（含费用）
"""
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from ..base.base_backtest_engine import BaseBacktestEngine
from ..base.base_backtest import BaseBacktestMetrics


class Position:
    """单只股票持仓，记录股数、平均成本、最近买入日期。

    记录 avg_cost 以支持正确的 PnL 计算，记录 last_buy_date 以支持 T+1。
    """

    __slots__ = ("shares", "avg_cost", "last_buy_date")

    def __init__(self, shares: int = 0, avg_cost: float = 0.0, last_buy_date=None):
        self.shares = shares
        self.avg_cost = avg_cost
        self.last_buy_date = last_buy_date

    def add(self, shares: int, price: float, date):
        """加仓，更新平均成本与最近买入日期。"""
        old_total = self.avg_cost * self.shares
        self.shares += shares
        if self.shares > 0:
            self.avg_cost = (old_total + shares * price) / self.shares
        self.last_buy_date = date

    def reduce(self, shares: int):
        """减仓，股数减为 0 时不清零 avg_cost（便于后续审计），仅减股数。"""
        self.shares = max(0, self.shares - shares)


class NativeAdapter(BaseBacktestEngine):
    """原生回测适配器"""

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        slippage: float = 0.001,
        transfer_fee_rate: float = 0.00002,
        min_commission: float = 5.0,
        budget_ratio: float = 0.95,
    ) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(signals['date'].unique())
        if not dates:
            return self._empty_result()

        cash = init_capital
        positions: Dict[str, Position] = {}
        equity_records = []
        trades = []
        gross_equity_records = []  # 不含费用的权益
        cumulative_fees = 0.0  # 累计已支付费用

        # 预提取基准净值（若有）
        benchmark_prices = {}
        bm_data = data[data["code"] == benchmark] if benchmark else pd.DataFrame()
        if not bm_data.empty:
            benchmark_prices = dict(zip(bm_data["date"], bm_data["close"]))
        bm_initial = None

        for dt in dates:
            day_signal = signals[signals['date'] == dt]
            day_data = data[data['date'] == dt]

            if day_data.empty:
                continue

            day_data_map = day_data.set_index('code')

            sell_codes, buy_codes = self._split_signals(day_signal)

            # ---- 卖出（先卖后买，释放资金）----
            for code in sell_codes:
                pos = positions.get(code)
                if pos is None or pos.shares <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                # T+1 检查：买入当日不得卖出
                if t_plus_1 and pos.last_buy_date is not None and dt <= pos.last_buy_date:
                    continue
                # 跌停不得卖出
                if price_limit and bool(price_row.get('is_limit_down', False)):
                    continue
                # 滑点双侧应用：卖出价乘以 (1 - slippage)
                fill_price = float(price_row['close']) * (1.0 - slippage)
                shares = pos.shares
                sell_amount = fill_price * shares
                # 费用：佣金（最低5元）+ 印花税（卖出）+ 过户费（双侧）
                commission = max(sell_amount * commission_rate, min_commission)
                stamp_tax = sell_amount * stamp_tax_rate
                transfer_fee = sell_amount * transfer_fee_rate
                total_cost = commission + stamp_tax + transfer_fee
                # 真实盈亏 = (卖出价 - 平均成本) * 股数 - 费用
                realized_pnl = (fill_price - pos.avg_cost) * shares - total_cost
                cash += sell_amount - total_cost
                cumulative_fees += total_cost
                trades.append({
                    'date': dt, 'code': code, 'action': 'sell',
                    'price': fill_price, 'shares': shares, 'amount': sell_amount,
                    'commission': commission, 'stamp_tax': stamp_tax,
                    'transfer_fee': transfer_fee, 'pnl': realized_pnl,
                    'avg_cost': pos.avg_cost,
                })
                pos.reduce(shares)

            # ---- 买入 ----
            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * budget_ratio / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    # 涨停不得买入
                    if price_limit and bool(price_row.get('is_limit_up', False)):
                        continue
                    # 滑点双侧应用：买入价乘以 (1 + slippage)
                    fill_price = float(price_row['close']) * (1.0 + slippage)
                    shares = int(budget_per_stock / fill_price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = fill_price * shares
                    # 费用：佣金（最低5元）+ 过户费（买入无印花税）
                    commission = max(buy_amount * commission_rate, min_commission)
                    transfer_fee = buy_amount * transfer_fee_rate
                    total_cost = commission + transfer_fee
                    if buy_amount + total_cost > cash:
                        shares = int((cash * 0.98) / fill_price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = fill_price * shares
                        commission = max(buy_amount * commission_rate, min_commission)
                        transfer_fee = buy_amount * transfer_fee_rate
                        total_cost = commission + transfer_fee
                    cash -= buy_amount + total_cost
                    cumulative_fees += total_cost
                    if code not in positions:
                        positions[code] = Position()
                    positions[code].add(shares, fill_price, dt)
                    trades.append({
                        'date': dt, 'code': code, 'action': 'buy',
                        'price': fill_price, 'shares': shares, 'amount': buy_amount,
                        'commission': commission, 'stamp_tax': 0.0,
                        'transfer_fee': transfer_fee, 'pnl': 0.0,
                        'avg_cost': positions[code].avg_cost,
                    })

            # ---- 估值 ----
            market_value = 0.0
            for code, pos in positions.items():
                if pos.shares <= 0:
                    continue
                if code in day_data_map.index:
                    close = float(day_data_map.loc[code, 'close'])
                    market_value += pos.shares * close
            total_equity = cash + market_value
            equity_records.append({
                'date': dt,
                'equity': total_equity,
                'cash': cash,
                'market_value': market_value,
                'position_count': sum(1 for p in positions.values() if p.shares > 0),
                'benchmark': benchmark_prices.get(dt, np.nan),
            })
            # 毛权益 = 净权益 + 累计费用（假设未支付费用的情景）
            gross_equity_records.append({
                'date': dt,
                'equity': total_equity + cumulative_fees,
            })

        equity_curve = pd.DataFrame(equity_records)
        gross_curve = pd.DataFrame(gross_equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        # 初始化基准首值
        if not equity_curve["benchmark"].isna().all():
            bm_initial = equity_curve["benchmark"].dropna().iloc[0]

        # 基础绩效指标（保持 key 兼容 RuleJudge：sharpe_ratio/calmar_ratio/max_drawdown）
        eq_series = equity_curve.set_index('date')['equity']
        metrics = BaseBacktestMetrics.calc_all_metrics(eq_series, trades_df)

        # 追加成本分离与基准对比指标（V2 修复点 5/6）
        metrics.update(self._calc_extended_metrics(
            equity_curve, gross_curve, trades_df, init_capital, bm_initial
        ))

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(
                [(c, p.shares, p.avg_cost) for c, p in positions.items() if p.shares > 0],
                columns=['code', 'shares', 'avg_cost'],
            ),
            "equity_curve": equity_curve,
            "gross_equity_curve": gross_curve,
            "metrics": metrics,
            "report_path": "",
        }

    @staticmethod
    def _split_signals(day_signal: pd.DataFrame) -> Tuple[list, list]:
        """分离买卖信号：signal>0 买入，signal<0 卖出"""
        sell_codes, buy_codes = [], []
        for _, row in day_signal.iterrows():
            code = row['code']
            sig = row.get('signal', 0)
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    buy_codes.append(code)
                elif sig < 0:
                    sell_codes.append(code)
        return sell_codes, buy_codes

    @staticmethod
    def _calc_extended_metrics(
        equity_curve: pd.DataFrame,
        gross_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        init_capital: float,
        bm_initial: Optional[float],
    ) -> Dict[str, Any]:
        """计算扩展指标：成本分离 + 基准对比。

        基础指标（sharpe_ratio/calmar_ratio/max_drawdown 等）由
        BaseBacktestMetrics.calc_all_metrics 提供，本方法仅补充 V2 新增指标。
        """
        result: Dict[str, Any] = {}

        # 成本分离（借鉴 Qlib excess_return_without_cost）
        gross_eq = gross_curve.set_index("date")["equity"]
        if len(gross_eq) > 1:
            gross_total_return = float(gross_eq.iloc[-1] / gross_eq.iloc[0] - 1.0)
            n_days = len(gross_eq)
            gross_annual = (1 + gross_total_return) ** (252.0 / max(n_days, 1)) - 1.0 \
                if gross_total_return > -1 else -1.0
            net_total_return = float(
                equity_curve.set_index("date")["equity"].iloc[-1] / init_capital - 1.0
            )
            result["gross_total_return"] = gross_total_return
            result["gross_annual_return"] = float(gross_annual)
            result["total_cost_drag"] = float(gross_total_return - net_total_return)

        # 基准对比
        if bm_initial is not None and "benchmark" in equity_curve.columns:
            bm_series = equity_curve.set_index("date")["benchmark"].dropna()
            if len(bm_series) > 1:
                bm_return = float(bm_series.iloc[-1] / bm_series.iloc[0] - 1.0)
                bm_daily = bm_series.pct_change().dropna()
                bm_vol = float(bm_daily.std() * np.sqrt(252)) if len(bm_daily) > 1 else 0.0
                bm_cummax = bm_series.cummax()
                bm_dd = float(((bm_series - bm_cummax) / bm_cummax).min())

                # alpha / beta（协方差法）
                eq_series = equity_curve.set_index("date")["equity"]
                daily_ret = eq_series.pct_change().dropna()
                aligned = pd.concat([daily_ret, bm_daily], axis=1, join="inner").dropna()
                aligned.columns = ["strat", "bench"]
                if len(aligned) > 2 and aligned["bench"].std() > 0:
                    beta = float(aligned.cov().iloc[0, 1] / aligned["bench"].var())
                    # 年化 alpha = 年化策略收益 - beta * 年化基准收益
                    ann_return = float((1 + (eq_series.iloc[-1] / eq_series.iloc[0] - 1.0)) ** (252.0 / max(len(eq_series), 1)) - 1.0) \
                        if (eq_series.iloc[-1] / eq_series.iloc[0] - 1.0) > -1 else -1.0
                    alpha = float(ann_return - beta * (bm_return if abs(bm_return) < 10 else 0))
                else:
                    beta, alpha = 0.0, 0.0

                net_total = float(eq_series.iloc[-1] / eq_series.iloc[0] - 1.0)
                result["benchmark_return"] = bm_return
                result["benchmark_volatility"] = bm_vol
                result["benchmark_max_drawdown"] = bm_dd
                result["alpha"] = alpha
                result["beta"] = beta
                result["excess_return"] = float(net_total - bm_return)

        return result

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "gross_equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
