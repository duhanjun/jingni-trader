#!/usr/bin/env python
"""
Backtrader 回测演示 - RSI+MACD 多信号策略
支持: A股T+1、涨跌停、佣金印花税、过户费
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'skills', 'a-share-data-engine'))

from config import get_config as get_data_config
from engine import AShareDataEngine

import pandas as pd
import numpy as np
import backtrader as bt

# ============================================================
# 1. 获取真实A股数据
# ============================================================
print("=" * 70)
print("📥 获取A股数据")
print("-" * 70)

config = get_data_config()
data_engine = AShareDataEngine(config)

stock_pool = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH']
df = data_engine.get_daily(
    codes=stock_pool,
    start_date='2023-01-01',
    end_date='2024-12-31'
)
print(f"✅ 获取 {len(df)} 条数据, {df['code'].nunique()} 只股票")
print(f"📅 时间: {df['date'].min()} ~ {df['date'].max()}")

# ============================================================
# 2. 定义 RSI+MACD 多信号策略
# ============================================================
class RSIMACDStrategy(bt.Strategy):
    """
    RSI + MACD + MA 多信号策略

    买入条件 (满足任一):
      1. RSI < 30 (超卖反弹) 且 MACD金叉
      2. MA5 > MA20 (金叉趋势)

    卖出条件 (满足任一):
      1. RSI > 70 (超买回落)
      2. MA5 < MA20 (死叉)
      3. 止损: 亏损超过 8%
    """
    params = dict(
        rsi_period=14,
        rsi_oversold=30,
        rsi_overbought=70,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        ma_fast=5,
        ma_slow=20,
        stop_loss=0.08,
        take_profit=0.20,
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.p.ma_fast)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.p.ma_slow)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.ma_cross = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)
        self.order = None
        self.buy_price = None
        self.trade_count = 0

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"  [{dt}] {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.trade_count += 1
                self.log(f"🟢 买入 {order.executed.size}股 @ {order.executed.price:.2f}")
            else:
                self.buy_price = None
                self.trade_count += 1
                pnl = order.executed.price - order.executed.price * 0.001
                self.log(f"🔴 卖出 {order.executed.size}股 @ {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"⚠️ 订单取消/拒绝: {order.status}")
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnlcomm
            self.log(f"  交易完成, 净盈亏: {pnl:.2f}")

    def next(self):
        if self.order:
            return

        current_price = self.data.close[0]
        position_size = self.position.size

        # ---- 买入信号 ----
        if position_size == 0:
            buy_signal_rsi_macd = (
                self.rsi[0] < self.p.rsi_oversold and self.macd_cross[0] > 0
            )
            buy_signal_ma = self.ma_cross[0] > 0
            buy_signal_rsi_ma = (
                self.rsi[0] < 40 and self.ma_cross[0] > 0
            )

            if buy_signal_rsi_macd or buy_signal_ma or buy_signal_rsi_ma:
                cash = self.broker.getcash()
                size = int(cash * 0.9 / current_price / 100) * 100
                if size >= 100:
                    self.order = self.buy(size=size)
                    signal_names = []
                    if buy_signal_rsi_macd: signal_names.append("RSI超卖+MACD金叉")
                    if buy_signal_ma: signal_names.append("MA金叉")
                    if buy_signal_rsi_ma: signal_names.append("RSI低位+MA金叉")
                    self.log(f"📊 信号: {' | '.join(signal_names)}")

        # ---- 卖出信号 ----
        elif position_size > 0 and self.buy_price:
            pnl_pct = (current_price - self.buy_price) / self.buy_price

            sell_signal_rsi = self.rsi[0] > self.p.rsi_overbought
            sell_signal_ma = self.ma_cross[0] < 0
            sell_signal_stop = pnl_pct < -self.p.stop_loss
            sell_signal_tp = pnl_pct > self.p.take_profit

            if sell_signal_rsi or sell_signal_ma or sell_signal_stop or sell_signal_tp:
                self.order = self.sell(size=position_size)
                reasons = []
                if sell_signal_rsi: reasons.append("RSI超买")
                if sell_signal_ma: reasons.append("MA死叉")
                if sell_signal_stop: reasons.append(f"止损({pnl_pct:.1%})")
                if sell_signal_tp: reasons.append(f"止盈({pnl_pct:.1%})")
                self.log(f"📊 平仓: {' | '.join(reasons)}")


# ============================================================
# 3. 运行 Backtrader 回测
# ============================================================
print("\n" + "=" * 70)
print("🔄 运行 Backtrader 回测")
print("-" * 70)

results = {}
all_metrics = []

for code in stock_pool:
    stock_df = df[df['code'] == code].copy()
    if len(stock_df) < 200:
        print(f"  ⚠️ {code}: 数据不足 ({len(stock_df)}条), 跳过")
        continue

    stock_df = stock_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    stock_df = stock_df.sort_values('date')
    stock_df = stock_df.set_index('date')

    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSIMACDStrategy)

    data_feed = bt.feeds.PandasData(
        dataname=stock_df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1,
    )

    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000.0)

    # A股佣金+印花税+过户费
    cerebro.broker.setcommission(
        commission=0.00025,
        margin=0.0,
        mult=1.0,
    )
    cerebro.broker.set_slippage_perc(0.001)

    # T+1 模拟
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    start_value = cerebro.broker.getvalue()
    strat_result = cerebro.run()
    end_value = cerebro.broker.getvalue()
    total_return = (end_value / start_value - 1) * 100

    strat = strat_result[0]

    # 计算绩效指标
    returns = stock_df['close'].pct_change().dropna()
    n_trades = strat.trade_count // 2 if strat.trade_count > 0 else 0

    metrics = {
        'code': code,
        'start_value': start_value,
        'end_value': end_value,
        'total_return_pct': round(total_return, 2),
        'n_trades': n_trades,
    }
    all_metrics.append(metrics)

    signal = '🟢' if total_return > 0 else '🔴'
    print(f"  {signal} {code}: 收益 {total_return:+.2f}%, 交易 {n_trades} 次, 最终 ¥{end_value:,.0f}")

    results[code] = {
        'strategy': strat,
        'total_return': total_return,
        'n_trades': n_trades,
        'end_value': end_value,
    }

# ============================================================
# 4. 汇总报告
# ============================================================
print("\n" + "=" * 70)
print("📊 回测汇总报告")
print("-" * 70)

metrics_df = pd.DataFrame(all_metrics)
if not metrics_df.empty:
    total_start = metrics_df['start_value'].sum()
    total_end = metrics_df['end_value'].sum()
    total_ret = (total_end / total_start - 1) * 100
    total_trades = metrics_df['n_trades'].sum()

    print(f"  回测股票数: {len(metrics_df)}")
    print(f"  初始总资金: ¥{total_start:,.0f}")
    print(f"  最终总资金: ¥{total_end:,.0f}")
    print(f"  总收益率: {total_ret:+.2f}%")
    print(f"  总交易次数: {total_trades}")
    print(f"  平均收益率: {metrics_df['total_return_pct'].mean():+.2f}%")
    print(f"  最佳股票: {metrics_df.loc[metrics_df['total_return_pct'].idxmax(), 'code']} ({metrics_df['total_return_pct'].max():+.2f}%)")
    print(f"  最差股票: {metrics_df.loc[metrics_df['total_return_pct'].idxmin(), 'code']} ({metrics_df['total_return_pct'].min():+.2f}%)")

    win_count = (metrics_df['total_return_pct'] > 0).sum()
    print(f"  胜率: {win_count}/{len(metrics_df)} ({win_count/len(metrics_df)*100:.0f}%)")

print("\n" + "=" * 70)
print("✅ Backtrader 回测完成!")
print("=" * 70)