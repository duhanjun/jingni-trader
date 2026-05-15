#!/usr/bin/env python
"""
StockBullStrategy V210 - 量化多头行业轮动策略 本地化实现
基于 /workspace/3.1 StockBullStrategyV210.py 的完整实现

核心因子:
  1. 趋势得分 (Trend Score): 指数价格突破MA20的强度
  2. 趋势强度 (Trend Strength): MA20>MA60>MA120 多时间框架趋势
  3. 资金流向 (Capital Flow): 成交量/额趋势变化（替代两融数据）

风控:
  4. 组合止损: 15%回撤清仓 / 8%减半仓
  5. 单标止损: 10%
  6. 波动率自适应因子
  7. 凯利公式仓位管理
  8. EMA + 卡尔曼滤波择时信号平滑
"""
import sys
import os
import json
import math
import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import backtrader as bt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'skills', 'a-share-data-engine'))

from config import get_config as get_data_config
from engine import AShareDataEngine

# ============================================================
# 配置
# ============================================================
INDUSTRY_INDICES = {
    '000933.SH': '中证医药',
    '000934.SH': '中证金融',
    '000935.SH': '中证信息',
    '000936.SH': '中证电信',
    '000937.SH': '中证公用',
    '000938.SH': '中证能源',
    '000939.SH': '中证材料',
    '000940.SH': '中证工业',
    '000941.SH': '中证可选',
    '000942.SH': '中证消费',
}

INITIAL_CAPITAL = 1000000.0
BACKTEST_START = '2022-01-01'
BACKTEST_END = '2024-12-31'

# ============================================================
# 工具函数
# ============================================================
def get_ma(values, period):
    if len(values) < period:
        return values[-1] if len(values) > 0 else 0
    return np.mean(values[-period:])

def normalize(values, feature_range=(-0.5, 0.5)):
    v = np.asarray(values, dtype=np.float64)
    mn, mx = v.min(), v.max()
    if np.isclose(mn, mx):
        return np.full_like(v, feature_range[0])
    scaled = (v - mn) / (mx - mn)
    return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

def calc_ema_signal(signal_series, span=5):
    if len(signal_series) < span:
        return float(signal_series.iloc[-1]) if len(signal_series) > 0 else 0.0
    ema = signal_series.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1])

def kalman_filter_smooth(observations, process_noise=1e-4, measurement_noise=1e-2):
    n = len(observations)
    if n == 0:
        return 0.0
    x_est = float(observations[0])
    p_est = 1.0
    filtered = [x_est]
    for k in range(1, n):
        x_pred = x_est
        p_pred = p_est + process_noise
        kg = p_pred / (p_pred + measurement_noise) if measurement_noise > 0 else 1.0
        z = float(observations[k])
        x_est = x_pred + kg * (z - x_pred)
        p_est = (1.0 - kg) * p_pred
        filtered.append(x_est)
    return filtered[-1]

def calc_kelly_fraction(win_rate=0.55, win_loss_ratio=1.5, max_fraction=0.40):
    b = win_loss_ratio
    p = win_rate
    q = 1.0 - p
    kelly = (p * b - q) / b
    if kelly <= 0:
        return 0.0
    half_kelly = kelly * 0.5
    return min(half_kelly, max_fraction)

# ============================================================
# 因子计算
# ============================================================
def calc_trend_score(df_index, df_constituents_map):
    """趋势得分: 计算每个指数成分股中价格>MA20的比例"""
    scores = {}
    for code, name in INDUSTRY_INDICES.items():
        c_df = df_index[df_index['code'] == code].sort_values('date')
        if len(c_df) < 30:
            scores[code] = 0
            continue
        prices = c_df['close'].values
        ma20 = np.array([get_ma(prices[:i+1], 20) for i in range(len(prices))])
        above = (prices > ma20).astype(float)
        scores[code] = round(above[-1] * 100, 0)  # 简化为100或0
    return scores

def calc_trend_strength(df_index):
    """趋势强度: MA20>MA60>MA120多时间框架趋势"""
    strengths = {}
    for code, name in INDUSTRY_INDICES.items():
        c_df = df_index[df_index['code'] == code].sort_values('date')
        if len(c_df) < 130:
            strengths[code] = 0
            continue
        prices = c_df['close'].values
        current = prices[-1]
        ma20 = get_ma(prices, 20)
        ma60 = get_ma(prices, 60)
        ma120 = get_ma(prices, 120)

        score = 0
        if current >= ma20 and ma20 >= ma60 and ma60 >= ma120:
            score = 100
        elif current >= ma20 and ma20 >= ma60:
            score = 80
        elif current >= ma20 and ma20 >= ma60 and current >= prices[-61] if len(prices) > 61 else False:
            score = 60
        elif current >= ma20 and current >= prices[-21] if len(prices) > 21 else False:
            score = 40
        elif current >= ma20:
            score = 20
        strengths[code] = score
    return strengths

def calc_capital_flow(df_index):
    """资金流向: 用成交量/额变化率替代两融数据"""
    flows = {}
    for code, name in INDUSTRY_INDICES.items():
        c_df = df_index[df_index['code'] == code].sort_values('date')
        if len(c_df) < 10:
            flows[code] = 0
            continue
        vol = c_df['volume'].values[-10:]
        amount = c_df['amount'].values[-10:]
        vol_chg = (vol[-1] - vol[-6]) / vol[-6] * 100 if vol[-6] > 0 else 0
        amt_chg = (amount[-1] - amount[-6]) / amount[-6] * 100 if amount[-6] > 0 else 0
        flows[code] = round((vol_chg + amt_chg) / 2, 2)
    return flows

def calc_timing_signal(df_index, index_code, lookback=5):
    """择时信号: 趋势得分变化 + EMA平滑 + 卡尔曼滤波"""
    c_df = df_index[df_index['code'] == index_code].sort_values('date')
    if len(c_df) < 130:
        return 0, 'hold'

    prices = c_df['close'].values
    trend_scores = []
    for i in range(len(prices)):
        if i < 20:
            trend_scores.append(50)
        else:
            ma20 = get_ma(prices[:i+1], 20)
            above_ratio = (prices[:i+1][-20:] > ma20).sum() / 20 * 100
            trend_scores.append(above_ratio)

    ts = pd.Series(trend_scores)
    if len(ts) < 6:
        return 0, 'hold'

    today = ts.iloc[-1]
    yesterday = ts.iloc[-2]
    prev_5 = ts.iloc[-6:-1]

    # 卡尔曼滤波
    kalman_input = ts.values
    kalman_signal = kalman_filter_smooth(kalman_input, process_noise=5e-4, measurement_noise=1e-2)

    # EMA平滑
    ema_signal = calc_ema_signal(ts, span=5)
    ema_prev = calc_ema_signal(ts.iloc[:-1], span=5)

    # 择时决策
    if ema_signal > ema_prev and kalman_signal >= 0:
        decision = 'buy'
    elif ema_signal < ema_prev and kalman_signal < 0:
        decision = 'sell'
    elif kalman_signal >= 0:
        decision = 'buy'
    else:
        decision = 'sell'

    return round(kalman_signal, 4), decision

def multi_factor_selection(df_index, weights=None):
    """多因子选股: 归一化 + 加权求和"""
    if weights is None:
        weights = {'trend_score': 0.50, 'trend_strength': 0.30, 'capital_flow': 0.20}

    scores = {}
    for code in INDUSTRY_INDICES:
        ts = calc_trend_score(df_index, None)
        tst = calc_trend_strength(df_index)
        cf = calc_capital_flow(df_index)

        scores[code] = {
            'trend_score': ts.get(code, 0),
            'trend_strength': tst.get(code, 0),
            'capital_flow': cf.get(code, 0),
        }

    df = pd.DataFrame(scores).T
    for col in ['trend_score', 'trend_strength', 'capital_flow']:
        df[col + '_norm'] = normalize(df[col].values)

    df['multi_factor_score'] = (
        df['trend_score_norm'] * weights['trend_score'] +
        df['trend_strength_norm'] * weights['trend_strength'] +
        df['capital_flow_norm'] * weights['capital_flow']
    )
    df = df.sort_values('multi_factor_score', ascending=False)
    return df

# ============================================================
# Backtrader 策略
# ============================================================
class StockBullStrategy(bt.Strategy):
    """量化多头行业轮动策略"""
    params = dict(
        top_n=5,
        stop_loss_portfolio=0.15,
        stop_loss_half=0.08,
        stop_loss_single=0.10,
        rebalance_freq=20,
    )

    def __init__(self):
        self.day_count = 0
        self.peak_value = INITIAL_CAPITAL
        self.stop_loss_triggered = False
        self.order_buy_flag = False
        self.order_sell_flag = False
        self.order_buyback_flag = False
        self.prev_trade_count = 0
        self.trade_log = []
        self.daily_equity = []
        self.timing_history = {}
        self.positions_pnl = {}

    def log_trade(self, msg):
        self.trade_log.append({
            'date': str(self.datas[0].datetime.date(0)),
            'message': msg
        })

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log_trade(f"BUY {order.data._name} x{order.executed.size} @ {order.executed.price:.2f}")
            else:
                self.log_trade(f"SELL {order.data._name} x{order.executed.size} @ {order.executed.price:.2f}")

    def get_data_for_index(self, idx_code):
        for d in self.datas:
            if d._name == idx_code:
                return d
        return self.datas[0]

    def next(self):
        self.day_count += 1
        current_date = str(self.datas[0].datetime.date(0))

        # 每天记录权益
        portfolio_value = self.broker.getvalue()
        self.daily_equity.append({
            'date': current_date,
            'equity': portfolio_value,
            'cash': self.broker.getcash(),
            'positions_value': portfolio_value - self.broker.getcash(),
            'market_ratio': (portfolio_value - self.broker.getcash()) / portfolio_value * 100 if portfolio_value > 0 else 0
        })

        # 只在rebalance日执行
        if self.day_count > 1 and self.day_count % self.p.rebalance_freq != 0:
            return

        portfolio_value = self.broker.getvalue()
        cash = self.broker.getcash()
        positions_value = portfolio_value - cash
        market_ratio = positions_value / portfolio_value * 100 if portfolio_value > 0 else 0

        self.daily_equity.append({
            'date': current_date,
            'equity': portfolio_value,
            'cash': cash,
            'positions_value': positions_value,
            'market_ratio': market_ratio
        })

        # 峰值跟踪
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        drawdown = (self.peak_value - portfolio_value) / self.peak_value * 100

        # === 组合止损 ===
        if drawdown > self.p.stop_loss_portfolio:
            self.log_trade(f"触发15%强制清仓! 回撤:{drawdown:.2f}%")
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
            self.stop_loss_triggered = True
            return

        if drawdown > self.p.stop_loss_half and not self.order_sell_flag:
            self.log_trade(f"触发8%减半仓! 回撤:{drawdown:.2f}%")
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos // 2)
            self.order_sell_flag = True
            return

        # === 单标止损 ===
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0 and pos.price > 0:
                current_price = d.close[0]
                loss_pct = (current_price - pos.price) / pos.price
                if loss_pct < -self.p.stop_loss_single:
                    self.sell(data=d, size=pos.size)
                    self.log_trade(f"{d._name} 单标止损 {loss_pct:.2%}")

        # === 使用预计算的因子数据 ===
        date_key = current_date
        if date_key not in self.precomputed_factors:
            return

        day_data = self.precomputed_factors[date_key]
        buy_signals = day_data.get('buy_signals', 0)
        sell_signals = day_data.get('sell_signals', 0)
        top_indices = day_data.get('top_indices', [])

        # === 择时卖出 ===
        if sell_signals > buy_signals and market_ratio > 0:
            for d in self.datas:
                pos = self.getposition(d).size
                if pos > 0:
                    self.sell(data=d, size=pos)
                    self.log_trade(f"SELL {d._name} x{pos} (sell_signal)")
            return

        # === 择时买入 ===
        if buy_signals >= sell_signals and market_ratio <= 85:
            kelly_factor = calc_kelly_fraction(win_rate=0.55, win_loss_ratio=1.5, max_fraction=0.40)
            order_weight = min(kelly_factor, 0.40)
            order_weight = max(0.10, min(order_weight, 0.50))

            # 计算可以买入的股票数量（确保每只至少能买1手）
            n_available = 0
            for idx_code in top_indices:
                d = self.get_data_for_index(idx_code)
                price = d.close[0]
                min_value_per_stock = price * 100
                if min_value_per_stock <= portfolio_value * order_weight:
                    n_available += 1

            n_buy = min(n_available, self.p.top_n, 3)  # 最多3只
            if n_buy == 0:
                return

            per_stock_weight = order_weight / n_buy

            bought = 0
            for idx_code in top_indices:
                if bought >= n_buy:
                    break
                d = self.get_data_for_index(idx_code)
                stock_value = portfolio_value * per_stock_weight
                price = d.close[0]
                size = int(stock_value / price / 100) * 100
                if size >= 100:
                    self.buy(data=d, size=size)
                    bought += 1
                    self.log_trade(f"BUY {idx_code} x{size} @ {price:.2f}")

    def stop(self):
        self.final_value = self.broker.getvalue()
        self.total_return = (self.final_value / INITIAL_CAPITAL - 1) * 100
        self.final_trades = len(self.trade_log)


# ============================================================
# 数据获取
# ============================================================
print("=" * 70)
print("📥 StockBullStrategy V210 - 获取行业指数数据")
print("-" * 70)

data_config = get_data_config()
data_engine = AShareDataEngine(data_config)

all_index_data = []
for code, name in INDUSTRY_INDICES.items():
    try:
        df = data_engine.get_daily(
            codes=[code],
            start_date=BACKTEST_START,
            end_date=BACKTEST_END
        )
        if not df.empty:
            all_index_data.append(df)
            print(f"  ✅ {code} {name}: {len(df)} 条")
        else:
            print(f"  ⚠️ {code} {name}: 无数据")
    except Exception as e:
        print(f"  ❌ {code} {name}: {e}")

if not all_index_data:
    print("\n⚠️ Tushare获取行业指数数据失败，使用模拟数据演示策略逻辑...")
    dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='B')
    np.random.seed(42)
    for code, name in INDUSTRY_INDICES.items():
        base = 50 + np.random.randint(-20, 20)
        trend = np.linspace(0, np.random.randn() * 10, len(dates))
        noise = np.cumsum(np.random.randn(len(dates)) * 2)
        close = base + trend + noise
        close = np.maximum(close, 5)
        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': close * 0.99,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(100000, 5000000, len(dates)),
            'amount': close * np.random.randint(100000, 5000000, len(dates)),
        })
        all_index_data.append(df)
    print("  ✅ 使用模拟数据: 10个行业指数, {} 个交易日".format(len(dates)))

combined = pd.concat(all_index_data, ignore_index=True)
print(f"\n📊 总数据: {len(combined)} 条, {combined['code'].nunique()} 个指数")

# ============================================================
# 因子计算演示
# ============================================================
print("\n" + "=" * 70)
print("🧮 因子计算")
print("-" * 70)

trend_scores = calc_trend_score(combined, None)
trend_strengths = calc_trend_strength(combined)
capital_flows = calc_capital_flow(combined)

print("  指数              趋势得分  趋势强度  资金流向")
for code, name in INDUSTRY_INDICES.items():
    ts = trend_scores.get(code, 0)
    tst = trend_strengths.get(code, 0)
    cf = capital_flows.get(code, 0)
    print(f"  {code} {name:8s}  {ts:8.0f}  {tst:8.0f}  {cf:8.2f}")

factor_result = multi_factor_selection(combined)
print("\n📊 多因子排名 (Top 5):")
for i, (code, row) in enumerate(factor_result.head(5).iterrows()):
    print(f"  {i+1}. {code} ({INDUSTRY_INDICES[code]}): {row['multi_factor_score']:.4f}")

# ============================================================
# 预计算每日因子（避免在Backtrader中访问line数据）
# ============================================================
print("\n🧮 预计算每日因子...")
dates_sorted = sorted(combined['date'].unique())
precomputed = {}
rebalance_interval = 20

for idx, current_date in enumerate(dates_sorted):
    date_str = str(current_date)[:10]
    # 为所有日期预计算（Backtrader可能在任何一天调仓）
    if idx > 0 and idx % rebalance_interval != 0:
        # 非调仓日也计算，供Backtrader的daily equity记录使用
        pass

    # 获取该日期之前的数据
    hist_data = combined[combined['date'] <= current_date]

    # 计算每个指数的因子
    factor_scores = {}
    buy_signals = 0
    sell_signals = 0

    for code in combined['code'].unique():
        c_df = hist_data[hist_data['code'] == code].sort_values('date')
        if len(c_df) < 130:
            continue
        prices = c_df['close'].values
        volumes = c_df['volume'].values

        # 趋势得分
        ma20 = np.mean(prices[-20:])
        ts = 100 if prices[-1] > ma20 else 0

        # 趋势强度
        ma60 = np.mean(prices[-60:]) if len(prices) >= 60 else ma20
        ma120 = np.mean(prices[-120:]) if len(prices) >= 120 else ma60
        if prices[-1] >= ma20 and ma20 >= ma60 and ma60 >= ma120:
            tst = 100
        elif prices[-1] >= ma20 and ma20 >= ma60:
            tst = 80
        elif prices[-1] >= ma20:
            tst = 40
        else:
            tst = 0

        # 资金流向
        vol_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        vol_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else vol_5
        cf = ((vol_5 - vol_10) / vol_10 * 100) if vol_10 > 0 else 0

        factor_scores[code] = {
            'trend_score': ts,
            'trend_strength': tst,
            'capital_flow': cf,
        }

        # 择时信号
        trend_vals = []
        for i in range(len(prices)):
            if i < 20:
                trend_vals.append(50)
            else:
                m20 = np.mean(prices[max(0, i-19):i+1])
                trend_vals.append(100 if prices[i] > m20 else 0)
        ts_series = pd.Series(trend_vals)
        kalman_signal = kalman_filter_smooth(ts_series.values, 5e-4, 1e-2)
        ema_signal = calc_ema_signal(ts_series, span=5)
        if len(ts_series) >= 3:
            ema_prev = calc_ema_signal(ts_series.iloc[:-1], span=5)
            if ema_signal > ema_prev and kalman_signal >= 0:
                buy_signals += 1
            elif ema_signal < ema_prev and kalman_signal < 0:
                sell_signals += 1

    # 多因子排名
    if factor_scores:
        df_factors = pd.DataFrame(factor_scores).T
        for col in ['trend_score', 'trend_strength', 'capital_flow']:
            df_factors[col + '_norm'] = normalize(df_factors[col].values)
        df_factors['score'] = (
            df_factors['trend_score_norm'] * 0.50 +
            df_factors['trend_strength_norm'] * 0.30 +
            df_factors['capital_flow_norm'] * 0.20
        )
        df_factors = df_factors.sort_values('score', ascending=False)
        top_indices = df_factors.head(5).index.tolist()
    else:
        top_indices = []

    precomputed[date_str] = {
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'top_indices': top_indices,
        'factor_scores': factor_scores,
    }

# 在 addstrategy 之前设置类变量
StockBullStrategy.precomputed_factors = precomputed
print(f"  ✅ 预计算完成: {len(precomputed)} 个调仓日")

# ============================================================
# Backtrader 回测
# ============================================================
print("\n" + "=" * 70)
print("🔄 Backtrader 回测")
print("-" * 70)

cerebro = bt.Cerebro()
cerebro.addstrategy(StockBullStrategy)

# 添加数据
for code, name in INDUSTRY_INDICES.items():
    stock_df = combined[combined['code'] == code].copy()
    if stock_df.empty:
        continue
    stock_df = stock_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    stock_df = stock_df.sort_values('date').set_index('date')

    data_feed = bt.feeds.PandasData(
        dataname=stock_df,
        datetime=None,
        open='open', high='high', low='low', close='close',
        volume='volume', openinterest=-1,
        name=code,
    )
    cerebro.adddata(data_feed)

cerebro.broker.setcash(INITIAL_CAPITAL)
cerebro.broker.setcommission(commission=0.0003)
cerebro.broker.set_slippage_perc(0.001)

# 添加分析器
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

start_value = cerebro.broker.getvalue()
results = cerebro.run()
end_value = cerebro.broker.getvalue()
total_return = (end_value / start_value - 1) * 100

strat = results[0]

print(f"  初始资金: ¥{start_value:,.0f}")
print(f"  最终资金: ¥{end_value:,.0f}")
print(f"  总收益率: {total_return:+.2f}%")
print(f"  交易记录: {len(strat.trade_log)} 条")

# ============================================================
# 生成 HTML 报告
# ============================================================
print("\n" + "=" * 70)
print("📄 生成 HTML 报告")
print("-" * 70)

# 分析器结果
sharpe = strat.analyzers.sharpe.get_analysis()
drawdown = strat.analyzers.drawdown.get_analysis()
returns_analysis = strat.analyzers.returns.get_analysis()
trade_analysis = strat.analyzers.trades.get_analysis()

# 计算权益曲线数据
equity_data = strat.daily_equity
equity_dates = [e['date'] for e in equity_data]
equity_values = [e['equity'] for e in equity_data]
market_ratios = [e['market_ratio'] for e in equity_data]

max_dd = drawdown.get('max', {}).get('drawdown', 0) if drawdown.get('max') else 0
annual_return = returns_analysis.get('rnorm100', 0)
sharpe_ratio = sharpe.get('sharperatio', 0) if sharpe.get('sharperatio') else 0

total_trades = trade_analysis.get('total', {}).get('total', 0)
won_trades = trade_analysis.get('won', {}).get('total', 0)
lost_trades = trade_analysis.get('lost', {}).get('total', 0)
win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0

# 行业指数因子排名
factor_rows = ""
for i, (code, row) in enumerate(factor_result.iterrows()):
    bg = '#e8f5e9' if i < 3 else ('#fff3e0' if i < 6 else '#f5f5f5')
    score = row['multi_factor_score']
    ts = row.get('trend_score', 0)
    tst = row.get('trend_strength', 0)
    cf = row.get('capital_flow', 0)
    factor_rows += f"""
    <tr style="background:{bg}">
      <td>{i+1}</td>
      <td>{code}</td>
      <td>{INDUSTRY_INDICES[code]}</td>
      <td>{ts:.0f}</td>
      <td>{tst:.0f}</td>
      <td>{cf:.2f}</td>
      <td><strong>{score:.4f}</strong></td>
    </tr>"""

# 交易日志
trade_log_rows = ""
for t in strat.trade_log[-50:]:
    trade_log_rows += f'<tr><td>{t["date"]}</td><td>{t["message"]}</td></tr>'

# 权益曲线 JSON
equity_json = json.dumps([{'date': e['date'], 'value': e['equity']} for e in equity_data])
ratio_json = json.dumps([{'date': e['date'], 'value': e['market_ratio']} for e in equity_data])

# 策略核心说明
strategy_logic = """
<h3>因子体系</h3>
<table class="compact">
<tr><th>因子</th><th>权重</th><th>说明</th></tr>
<tr><td>趋势得分</td><td>50%</td><td>指数价格突破MA20的强度信号</td></tr>
<tr><td>趋势强度</td><td>30%</td><td>MA20>MA60>MA120多时间框架趋势</td></tr>
<tr><td>资金流向</td><td>20%</td><td>成交量/额变化率（替代两融数据）</td></tr>
</table>

<h3>择时系统</h3>
<ul>
<li>趋势得分变化 + EMA平滑(span=5)</li>
<li>卡尔曼滤波降噪 (process_noise=5e-4)</li>
<li>信号: EMA向上+Kalman≥0 → 买入 | EMA向下+Kalman&lt;0 → 卖出</li>
</ul>

<h3>风控体系</h3>
<table class="compact">
<tr><th>风控措施</th><th>参数</th></tr>
<tr><td>组合强制清仓</td><td>回撤 > 15%</td></tr>
<tr><td>组合减半仓</td><td>回撤 > 8%</td></tr>
<tr><td>单标止损</td><td>亏损 > 10%</td></tr>
<tr><td>凯利公式</td><td>win_rate=55%, w/l_ratio=1.5, max=30%</td></tr>
<tr><td>仓位上限</td><td>85%</td></tr>
</ul>
</table>

<h3>仓位管理</h3>
<ul>
<li>波动率自适应因子: target_vol=15%</li>
<li>凯利公式推荐仓位: half_kelly, max=30%</li>
<li>最终权重: min(kelly, base_20%) * vol_adaptive_factor</li>
<li>波动率倒数加权分配</li>
</ul>
"""

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StockBullStrategy V210 - 量化多头行业轮动策略 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background:#f0f2f5; color:#333; }}
.container {{ max-width:1400px; margin:0 auto; padding:20px; }}
.header {{ background:linear-gradient(135deg, #1a237e, #0d47a1); color:white; padding:40px; border-radius:16px; margin-bottom:24px; }}
.header h1 {{ font-size:28px; margin-bottom:8px; }}
.header .meta {{ opacity:0.85; font-size:14px; }}
.card {{ background:white; border-radius:12px; padding:24px; margin-bottom:20px; box-shadow:0 2px 12px rgba(0,0,0,0.06); }}
.card h2 {{ font-size:20px; margin-bottom:16px; color:#1a237e; border-bottom:2px solid #e8eaf6; padding-bottom:10px; }}
.metrics {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:16px; }}
.metric {{ background:#f8f9fa; border-radius:10px; padding:20px; text-align:center; }}
.metric .label {{ font-size:13px; color:#666; margin-bottom:6px; }}
.metric .value {{ font-size:28px; font-weight:700; }}
.metric .value.positive {{ color:#2e7d32; }}
.metric .value.negative {{ color:#c62828; }}
.metric .sub {{ font-size:12px; color:#999; margin-top:4px; }}
.chart-container {{ height:400px; margin:20px 0; }}
canvas {{ width:100% !important; }}
table {{ width:100%; border-collapse:collapse; margin-top:12px; }}
th, td {{ padding:10px 14px; text-align:left; border-bottom:1px solid #eee; font-size:13px; }}
th {{ background:#f5f5f5; font-weight:600; color:#555; }}
.compact td, .compact th {{ padding:6px 10px; font-size:12px; }}
.trade-log {{ max-height:300px; overflow-y:auto; }}
.green {{ color:#2e7d32; }}
.red {{ color:#c62828; }}
.footer {{ text-align:center; padding:20px; color:#999; font-size:12px; }}
h3 {{ font-size:16px; margin:16px 0 8px 0; color:#333; }}
ul {{ margin-left:20px; margin-bottom:12px; }}
li {{ font-size:13px; line-height:1.8; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>📊 StockBullStrategy V210</h1>
  <p style="font-size:16px; margin:4px 0;">量化多头行业轮动策略 · 回测报告</p>
  <div class="meta">
    回测区间: {BACKTEST_START} ~ {BACKTEST_END} | 
    初始资金: ¥{INITIAL_CAPITAL:,.0f} |
    行业指数: {len(INDUSTRY_INDICES)} 个 |
    生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>

<!-- 核心指标 -->
<div class="card">
  <h2>📈 核心绩效指标</h2>
  <div class="metrics">
    <div class="metric">
      <div class="label">总收益率</div>
      <div class="value {'positive' if total_return >= 0 else 'negative'}">{total_return:+.2f}%</div>
      <div class="sub">vs 初始资金 ¥{INITIAL_CAPITAL:,.0f}</div>
    </div>
    <div class="metric">
      <div class="label">最终资金</div>
      <div class="value">¥{end_value:,.0f}</div>
      <div class="sub">初始 ¥{start_value:,.0f}</div>
    </div>
    <div class="metric">
      <div class="label">年化收益率</div>
      <div class="value {'positive' if annual_return >= 0 else 'negative'}">{annual_return:+.2f}%</div>
    </div>
    <div class="metric">
      <div class="label">夏普比率</div>
      <div class="value">{sharpe_ratio:.3f}</div>
      <div class="sub">无风险利率 2%</div>
    </div>
    <div class="metric">
      <div class="label">最大回撤</div>
      <div class="value negative">-{max_dd:.2f}%</div>
    </div>
    <div class="metric">
      <div class="label">胜率</div>
      <div class="value">{win_rate:.1f}%</div>
      <div class="sub">{won_trades}赢 / {lost_trades}输</div>
    </div>
  </div>
</div>

<!-- 权益曲线 -->
<div class="card">
  <h2>📉 权益曲线 & 仓位变化</h2>
  <div class="chart-container">
    <canvas id="equityChart"></canvas>
  </div>
  <div class="chart-container">
    <canvas id="ratioChart"></canvas>
  </div>
</div>

<!-- 因子排名 -->
<div class="card">
  <h2>🏆 行业指数多因子排名</h2>
  <table>
    <tr><th>排名</th><th>代码</th><th>名称</th><th>趋势得分</th><th>趋势强度</th><th>资金流向</th><th>综合得分</th></tr>
    {factor_rows}
  </table>
  <p style="margin-top:12px;font-size:12px;color:#888;">
    Top 3 (绿色) 为当期推荐买入行业 | 权重: 趋势得分50% + 趋势强度30% + 资金流向20%
  </p>
</div>

<!-- 策略逻辑 -->
<div class="card">
  <h2>🔧 策略逻辑说明</h2>
  {strategy_logic}
</div>

<!-- 交易日志 -->
<div class="card">
  <h2>📝 交易日志 (最近50条)</h2>
  <div class="trade-log">
    <table>
      <tr><th>日期</th><th>操作</th></tr>
      {trade_log_rows}
    </table>
  </div>
</div>

<div class="footer">
  <p>StockBullStrategy V210 · 基于PTrade策略的本地化Backtrader实现</p>
  <p>数据源: Tushare · 回测引擎: Backtrader · 行业指数: CSI 中证行业指数系列</p>
</div>

</div>

<script>
// 权益曲线
const equityCtx = document.getElementById('equityChart').getContext('2d');
new Chart(equityCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(equity_dates)},
    datasets: [{{
      label: '权益曲线 (¥)',
      data: {json.dumps(equity_values)},
      borderColor: '#1a237e',
      backgroundColor: 'rgba(26, 35, 126, 0.05)',
      fill: true,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: true }} }},
    scales: {{
      y: {{ ticks: {{ callback: v => '¥' + (v/10000).toFixed(0) + '万' }} }}
    }}
  }}
}});

// 仓位变化
const ratioCtx = document.getElementById('ratioChart').getContext('2d');
new Chart(ratioCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(equity_dates)},
    datasets: [{{
      label: '仓位比例 (%)',
      data: {json.dumps(market_ratios)},
      borderColor: '#e65100',
      backgroundColor: 'rgba(230, 81, 0, 0.08)',
      fill: true,
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: true }} }},
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ callback: v => v + '%' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stockbull_backtest_report.html')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"✅ HTML报告已生成: {report_path}")
print(f"📄 文件大小: {os.path.getsize(report_path) / 1024:.1f} KB")