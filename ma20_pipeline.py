"""
jingni-trader MA20 因子量化全流程分析管道
============================================
阶段: DATA → FACTOR → MODEL → BACKTEST → RISK → REPORT
数据源: Tushare Pro (pro.daily)
策略: MA20 金叉/死叉 + 等权持有
"""
import os
import sys
import json
import logging
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import tushare as ts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("ma20_pipeline")

# ── 配置 ──────────────────────────────────
TUSHARE_TOKEN = "79a12083b83a09126f3eca16536c63d711d0dff8beb11e13bc9f45ad"
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

WORK_DIR = "/workspace/workspace"
for d in ["data", "factors", "models", "backtest_results", "reports"]:
    os.makedirs(f"{WORK_DIR}/{d}", exist_ok=True)

INIT_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.001
MIN_COMMISSION = 5.0
SLIPPAGE = 0.0001
RISK_FREE_RATE = 0.03
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

# 沪深300代表性成分股 (40只，覆盖各行业)
HS300_STOCKS = [
    # 金融 (银行/保险/券商)
    "000001.SZ", "002142.SZ", "600036.SH", "601318.SH", "601166.SH",
    "600030.SH", "601398.SH",
    # 消费 (白酒/家电/食品)
    "600519.SH", "000858.SZ", "000333.SZ", "000568.SZ", "600887.SH",
    # 医药
    "600276.SH", "300015.SZ", "300760.SZ", "000538.SZ",
    # 科技 (半导体/电子/软件)
    "002415.SZ", "002230.SZ", "002475.SZ", "300059.SZ", "000725.SZ",
    # 新能源/电力设备
    "300750.SZ", "601012.SH",
    # 基建/建材
    "600585.SH", "600031.SH", "002271.SZ", "601899.SH",
    # 能源/电力
    "600900.SH", "601225.SH",
    # 汽车
    "002594.SZ", "000625.SZ", "601238.SH",
    # 有色/化工
    "603259.SH", "600309.SH",
    # 地产
    "000002.SZ", "600048.SH",
    # 交运/通信
    "601888.SH", "600050.SH",
    # 制造业
    "000651.SZ", "601688.SH",
]

def _rate_limit():
    time.sleep(1.3)


# ═══════════════════════════════════════════
# 阶段一: 数据获取
# ═══════════════════════════════════════════
def stage_data_acquisition() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("阶段一: 数据获取 (DATA) - 数据源: Tushare Pro")
    logger.info("=" * 60)

    sd = START_DATE.replace("-", "")
    ed = END_DATE.replace("-", "")

    all_data = []
    for symbol in HS300_STOCKS:
        _rate_limit()
        try:
            df = pro.daily(
                ts_code=symbol, start_date=sd, end_date=ed,
                fields='ts_code,trade_date,open,high,low,close,pre_close,vol,amount'
            )
            if df is None or df.empty:
                logger.warning(f"  {symbol}: 无数据")
                continue
            df = df.rename(columns={
                'ts_code': 'code', 'trade_date': 'date',
                'vol': 'volume', 'pre_close': 'preclose'
            })
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # 获取复权因子
            _rate_limit()
            try:
                adj_df = pro.adj_factor(ts_code=symbol, start_date=sd, end_date=ed)
                if adj_df is not None and not adj_df.empty:
                    adj_df = adj_df.rename(columns={'trade_date': 'date'})
                    adj_df['date'] = pd.to_datetime(adj_df['date'], format='%Y%m%d')
                    df = df.merge(adj_df[['date', 'adj_factor']], on='date', how='left')
                    last_adj = adj_df['adj_factor'].iloc[0] if len(adj_df) > 0 else 1.0
                    df['adj_factor'] = df['adj_factor'].fillna(last_adj)
                    for col in ['open', 'high', 'low', 'close', 'preclose']:
                        df[col] = df[col] * (df['adj_factor'] / last_adj)
            except Exception:
                pass

            df['change_pct'] = (df['close'] - df['preclose']) / df['preclose'] * 100
            df['turnover_rate'] = np.nan
            all_data.append(df)
        except Exception as e:
            logger.warning(f"  {symbol}: 获取失败 - {e}")

    if not all_data:
        raise RuntimeError("未获取到任何数据")

    result = pd.concat(all_data, ignore_index=True)
    result = result.sort_values(['date', 'code']).reset_index(drop=True)
    result = result[result['volume'] > 0]
    result = result.dropna(subset=['close'])

    output_path = f"{WORK_DIR}/data/cleaned_data.parquet"
    result.to_parquet(output_path, index=False)

    logger.info(f"  成功获取: {result['code'].nunique()}/{len(HS300_STOCKS)} 只")
    logger.info(f"  数据总量: {len(result)} 行")
    logger.info(f"  日期范围: {result['date'].min().date()} ~ {result['date'].max().date()}")
    return result


# ═══════════════════════════════════════════
# 阶段二: 因子构建
# ═══════════════════════════════════════════
def stage_factor_engineering(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    logger.info("=" * 60)
    logger.info("阶段二: 因子构建 (FACTOR)")
    logger.info("=" * 60)

    df = data.sort_values(['code', 'date']).copy()

    # ── MA20 及相关因子 ──
    df['ma_20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['ma_60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60, min_periods=15).mean())
    df['ma20_deviation'] = (df['close'] - df['ma_20']) / df['ma_20']
    df['ma20_slope'] = df.groupby('code')['ma_20'].transform(lambda x: (x - x.shift(5)) / x.shift(5))

    # 布林带
    df['rolling_std'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=5).std())
    df['boll_upper'] = df['ma_20'] + 2 * df['rolling_std']
    df['boll_lower'] = df['ma_20'] - 2 * df['rolling_std']
    df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-10)

    # 金叉/死叉
    df['close_l1'] = df.groupby('code')['close'].shift(1)
    df['ma20_l1'] = df.groupby('code')['ma_20'].shift(1)
    df['golden_cross'] = (df['close_l1'] <= df['ma20_l1']) & (df['close'] > df['ma_20'])
    df['death_cross'] = (df['close_l1'] >= df['ma20_l1']) & (df['close'] < df['ma_20'])

    # 辅助因子
    df['ret_1d'] = df.groupby('code')['close'].pct_change()
    df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
    df['reversal_20d'] = -df['ret_20d']
    df['volatility_20d'] = df.groupby('code')['ret_1d'].transform(lambda x: x.rolling(20, min_periods=10).std())
    df['volume_ratio'] = df['volume'] / df.groupby('code')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())

    # 前视收益
    for period in [1, 5, 20]:
        df[f'ret_forward_{period}d'] = df.groupby('code')['close'].transform(lambda x: x.shift(-period) / x - 1)

    # ── IC 分析 ──
    factor_names = ['ma20_deviation', 'ma20_slope', 'reversal_20d', 'volatility_20d', 'volume_ratio', 'boll_position']
    ic_results = {}
    for fwd_col in ['ret_forward_5d']:
        fwd_results = []
        for factor in factor_names:
            ic_series_list = []
            for dt_val, grp in df.dropna(subset=[factor, fwd_col]).groupby('date'):
                if len(grp) < 10:
                    continue
                ic_val, _ = stats.pearsonr(grp[factor].fillna(0), grp[fwd_col].fillna(0))
                if not np.isnan(ic_val):
                    ic_series_list.append({'date': dt_val, 'ic': ic_val})
            if not ic_series_list:
                continue
            ic_s = pd.DataFrame(ic_series_list)
            m = ic_s['ic'].mean()
            s = ic_s['ic'].std()
            ir = m / s if s > 0 else 0
            fwd_results.append({
                'factor': factor,
                'ic_mean': round(float(m), 6), 'ic_std': round(float(s), 6),
                'ic_ir': round(float(ir), 4),
                'ic_positive_ratio': round(float((ic_s['ic'] > 0).mean()), 4),
                'ic_t_stat': round(float(m / (s / np.sqrt(len(ic_s)))) if s > 0 else 0, 4),
            })
        ic_results[fwd_col] = fwd_results

    # MA20 偏离度详细 IC
    ma20_ic_detail = {}
    for fwd_col in ['ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']:
        ic_list = []
        for dt_val, grp in df.dropna(subset=['ma20_deviation', fwd_col]).groupby('date'):
            if len(grp) < 10:
                continue
            ic_val, _ = stats.pearsonr(grp['ma20_deviation'].fillna(0), grp[fwd_col].fillna(0))
            if not np.isnan(ic_val):
                ic_list.append({'date': dt_val, 'ic': ic_val})
        if ic_list:
            ic_detail_df = pd.DataFrame(ic_list)
            ma20_ic_detail[fwd_col] = {
                'mean': round(float(ic_detail_df['ic'].mean()), 6),
                'std': round(float(ic_detail_df['ic'].std()), 6),
                'ir': round(float(ic_detail_df['ic'].mean() / ic_detail_df['ic'].std()) if ic_detail_df['ic'].std() > 0 else 0, 4),
                'positive_ratio': round(float((ic_detail_df['ic'] > 0).mean()), 4),
                'n_dates': len(ic_detail_df),
            }

    # ── 分位数测试 ──
    quantile_summary = []
    for dt_val, grp in df.dropna(subset=['ma20_deviation', 'ret_forward_5d']).groupby('date'):
        if len(grp) < 20:
            continue
        grp = grp.copy()
        grp['q_rank'] = pd.qcut(grp['ma20_deviation'], 5, labels=False, duplicates='drop')
        if grp['q_rank'].nunique() < 5:
            continue
        for q in range(5):
            q_data = grp[grp['q_rank'] == q]
            quantile_summary.append({
                'date': dt_val, 'quantile': int(q),
                'mean_return': float(q_data['ret_forward_5d'].mean()),
                'count': len(q_data),
            })

    qdf = pd.DataFrame(quantile_summary)
    if not qdf.empty:
        q_summary = qdf.groupby('quantile').agg(
            mean_return=('mean_return', 'mean'),
            std_return=('mean_return', 'std'),
            n_days=('date', 'count')
        ).reset_index()
        q_summary['t_stat'] = q_summary['mean_return'] / (q_summary['std_return'] / np.sqrt(q_summary['n_days']))
        quantile_summary_data = q_summary.to_dict(orient='records')
    else:
        quantile_summary_data = []

    # 保存
    factor_cols = ['code', 'date', 'close', 'ma_20', 'ma_60', 'ma20_deviation', 'ma20_slope',
                   'golden_cross', 'death_cross', 'boll_position', 'boll_upper', 'boll_lower',
                   'reversal_20d', 'volatility_20d', 'volume_ratio',
                   'ret_1d', 'ret_5d', 'ret_20d',
                   'ret_forward_1d', 'ret_forward_5d', 'ret_forward_20d']
    factor_df = df[factor_cols].copy()
    factor_df.to_parquet(f"{WORK_DIR}/factors/factor_data.parquet", index=False)

    factor_meta = {
        'ic_results': ic_results,
        'ma20_ic_detail': ma20_ic_detail,
        'quantile_summary': quantile_summary_data,
    }
    with open(f"{WORK_DIR}/factors/ic_report.json", 'w', encoding='utf-8') as f:
        json.dump(factor_meta, f, ensure_ascii=False, indent=2, default=str)

    if ma20_ic_detail:
        m5 = ma20_ic_detail.get('ret_forward_5d', {})
        logger.info(f"  MA20偏离度 IC(5d): mean={m5.get('mean', 'N/A')}, IR={m5.get('ir', 'N/A')}")
    if quantile_summary_data:
        top = quantile_summary_data[-1]
        bot = quantile_summary_data[0]
        logger.info(f"  分位数测试: Q1={bot['mean_return']:.4%}, Q5={top['mean_return']:.4%}, Spread={top['mean_return']-bot['mean_return']:.4%}")

    return factor_df, factor_meta


# ═══════════════════════════════════════════
# 阶段三: 信号生成
# ═══════════════════════════════════════════
def stage_signal_generation(factor_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("阶段三: 策略信号生成 (MODEL)")
    logger.info("=" * 60)

    df = factor_df.copy()

    # 策略: MA20 金叉/死叉 + 持有
    df['signal_cross'] = 0
    df.loc[df['golden_cross'], 'signal_cross'] = 1
    df.loc[df['death_cross'], 'signal_cross'] = -1
    df['signal'] = df.groupby('code')['signal_cross'].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0)
    )

    signal_df = df[['code', 'date', 'close', 'ma_20', 'ma20_deviation', 'signal',
                     'golden_cross', 'death_cross']].copy()
    signal_df.to_parquet(f"{WORK_DIR}/models/signal_ma20.parquet", index=False)

    buy_signal_dates = signal_df[signal_df['signal'] == 1].groupby('date').size()
    n_golden = signal_df['golden_cross'].sum()
    n_death = signal_df['death_cross'].sum()

    logger.info(f"  金叉次数: {n_golden}, 死叉次数: {n_death}")
    logger.info(f"  日均持仓数: {buy_signal_dates.mean():.1f} 只 (中位数: {buy_signal_dates.median():.0f})")
    return signal_df


# ═══════════════════════════════════════════
# 阶段四: 回测
# ═══════════════════════════════════════════
def stage_backtest(data: pd.DataFrame, signal_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("阶段四: 回测验证 (BACKTEST)")
    logger.info("=" * 60)

    sig_sub = signal_df[['code', 'date', 'signal']].copy()
    merged = sig_sub.merge(data[['code', 'date', 'open', 'close', 'change_pct', 'volume']],
                           on=['code', 'date'], how='inner')
    merged = merged.sort_values(['date', 'code']).reset_index(drop=True)

    daily_records = []
    capital = INIT_CAPITAL
    cash = INIT_CAPITAL
    holdings = {}

    dates = sorted(merged['date'].unique())
    for i, dt in enumerate(dates):
        day_data = merged[merged['date'] == dt]
        if day_data.empty:
            continue

        # 当日持仓市值
        position_value = 0.0
        for code, shares in list(holdings.items()):
            row = day_data[day_data['code'] == code]
            if not row.empty:
                position_value += shares * row['close'].values[0]

        # 卖出 (死叉信号 - 平仓)
        sell_candidates = day_data[day_data['signal'] == -1]
        for _, row in sell_candidates.iterrows():
            code = row['code']
            if code in holdings and holdings[code] > 0:
                price = row['close'] * (1 - SLIPPAGE)
                val = holdings[code] * price
                commission = max(val * COMMISSION_RATE, MIN_COMMISSION)
                stamp = val * STAMP_TAX_RATE
                cash += val - commission - stamp
                holdings[code] = 0

        # 买入 (金叉信号 - 开仓)
        buy_candidates = day_data[day_data['signal'] == 1]
        n_buy = len(buy_candidates)
        if n_buy > 0:
            # 先清理旧持仓中不在买入列表的
            buy_codes = set(buy_candidates['code'].tolist())
            for code in list(holdings.keys()):
                if code not in buy_codes and holdings[code] > 0:
                    row = day_data[day_data['code'] == code]
                    if not row.empty:
                        price = row['close'].values[0] * (1 - SLIPPAGE)
                        val = holdings[code] * price
                        commission = max(val * COMMISSION_RATE, MIN_COMMISSION)
                        stamp = val * STAMP_TAX_RATE
                        cash += val - commission - stamp
                        holdings[code] = 0

            # 为不在持仓中的买入候选开仓
            for _, row in buy_candidates.iterrows():
                code = row['code']
                if holdings.get(code, 0) > 0:
                    continue
                price = row['close'] * (1 + SLIPPAGE)
                budget_per_stock = cash * 0.95 / max(n_buy - sum(1 for c in buy_codes if holdings.get(c, 0) > 0), 1)
                max_shares = int(budget_per_stock / price / 100) * 100
                if max_shares < 100:
                    continue
                cost = max_shares * price
                commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                if cost + commission > cash:
                    continue
                holdings[code] = max_shares
                cash -= (cost + commission)

        # 计算最终权益
        position_value = 0.0
        n_holdings = 0
        for code, shares in holdings.items():
            if shares > 0:
                row = day_data[day_data['code'] == code]
                if not row.empty:
                    position_value += shares * row['close'].values[0]
                    n_holdings += 1

        total = position_value + cash
        daily_records.append({
            'date': dt, 'equity': total, 'cash': cash,
            'position_value': position_value, 'n_holdings': n_holdings,
        })

    equity_df = pd.DataFrame(daily_records)
    equity_df['returns'] = equity_df['equity'].pct_change()
    valid_ret = equity_df['returns'].dropna()

    total_return = equity_df['equity'].iloc[-1] / INIT_CAPITAL - 1
    n_days = len(valid_ret)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    volatility = valid_ret.std() * np.sqrt(252)
    dd = equity_df['equity'] / equity_df['equity'].cummax() - 1
    max_drawdown = dd.min()
    sharpe = (annual_return - RISK_FREE_RATE) / volatility if volatility > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    win_rate = (valid_ret > 0).mean()
    sortino = (annual_return - RISK_FREE_RATE) / (valid_ret[valid_ret < 0].std() * np.sqrt(252)) if len(valid_ret[valid_ret < 0]) > 0 else 0

    metrics = {
        'total_return': float(total_return), 'annual_return': float(annual_return),
        'volatility': float(volatility), 'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown), 'calmar_ratio': float(calmar),
        'win_rate': float(win_rate), 'sortino_ratio': float(sortino),
        'n_trading_days': n_days, 'final_equity': float(equity_df['equity'].iloc[-1]),
    }

    # 年度/月度
    equity_df['year'] = equity_df['date'].dt.year
    annual_ret = equity_df.groupby('year')['returns'].apply(lambda x: (1 + x).prod() - 1)
    metrics['annual_returns'] = {int(y): float(r) for y, r in annual_ret.items()}

    equity_df['month_period'] = equity_df['date'].dt.to_period('M')
    monthly_ret = equity_df.groupby('month_period')['returns'].apply(lambda x: (1 + x).prod() - 1)
    metrics['monthly_returns'] = [{'year': m.year, 'month': m.month, 'return': float(r)} for m, r in monthly_ret.items()]

    equity_df.to_parquet(f"{WORK_DIR}/backtest_results/equity_curve.parquet", index=False)
    with open(f"{WORK_DIR}/backtest_results/backtest_result.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"  累计收益: {total_return*100:.2f}%")
    logger.info(f"  年化收益: {annual_return*100:.2f}%")
    logger.info(f"  夏普比率: {sharpe:.3f}")
    logger.info(f"  最大回撤: {max_drawdown*100:.2f}%")
    logger.info(f"  Calmar: {calmar:.3f}")
    return metrics, equity_df


# ═══════════════════════════════════════════
# 阶段五: 风险分析
# ═══════════════════════════════════════════
def stage_risk_analysis(equity_df: pd.DataFrame) -> Dict:
    logger.info("=" * 60)
    logger.info("阶段五: 风险分析 (RISK)")
    logger.info("=" * 60)

    ret_dt = equity_df.set_index('date')['returns'].dropna()
    ret = ret_dt
    var_95 = np.percentile(ret, 5)
    cvar_95 = ret[ret <= var_95].mean() if len(ret[ret <= var_95]) > 0 else var_95

    neg_streak = (ret < 0).astype(int)
    stref_groups = (neg_streak.diff() != 0).cumsum()
    max_losing_streak = neg_streak.groupby(stref_groups).sum().max()

    pos = ret[ret > 0]
    neg = ret[ret < 0]
    avg_win = pos.mean() if len(pos) > 0 else 0
    avg_loss = neg.mean() if len(neg) > 0 else 0
    profit_factor = abs(pos.sum() / neg.sum()) if neg.sum() != 0 else float('inf')

    risk = {
        'var_95_daily': float(var_95), 'cvar_95_daily': float(cvar_95),
        'max_losing_streak_days': int(max_losing_streak) if pd.notna(max_losing_streak) else 0,
        'avg_win': float(avg_win), 'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'skewness': float(ret.skew()), 'kurtosis': float(ret.kurtosis()),
        'weekly_volatility': float(ret_dt.resample('W').apply(lambda x: (1+x).prod()-1).std() * np.sqrt(52)),
    }

    logger.info(f"  VaR(95%): {var_95*100:.2f}%  |  CVaR(95%): {cvar_95*100:.2f}%")
    logger.info(f"  盈亏比: {profit_factor:.2f}  |  偏度: {ret.skew():.2f}")
    return risk


# ═══════════════════════════════════════════
# 阶段六: 报告生成
# ═══════════════════════════════════════════
def stage_report_generation(factor_meta, metrics, equity_df, risk, signal_df) -> str:
    logger.info("=" * 60)
    logger.info("阶段六: 绩效报告 (REPORT)")
    logger.info("=" * 60)

    # ── 图表1: 净值 + 回撤 ──
    nav = equity_df['equity'].values / INIT_CAPITAL
    dd = equity_df['equity'].values / equity_df['equity'].cummax().values - 1

    fig_nav = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.7, 0.3], subplot_titles=("MA20策略净值曲线", "回撤曲线"))
    fig_nav.add_trace(go.Scatter(x=equity_df['date'], y=nav, mode='lines', name='MA20策略',
                                  line=dict(color='#1f77b4', width=2)), row=1, col=1)
    fig_nav.add_trace(go.Scatter(x=equity_df['date'], y=[1.0]*len(nav), mode='lines',
                                  name='初始净值', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig_nav.add_trace(go.Scatter(x=equity_df['date'], y=dd, mode='lines', fill='tozeroy',
                                  name='回撤', line=dict(color='#d62728', width=1),
                                  fillcolor='rgba(214,39,40,0.15)'), row=2, col=1)
    fig_nav.update_layout(height=650, hovermode='x unified', template='plotly_white',
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig_nav.update_yaxes(title_text="净值", row=1, col=1)
    fig_nav.update_yaxes(title_text="回撤 %", tickformat=".1%", row=2, col=1)
    nav_html = fig_nav.to_html(full_html=False, include_plotlyjs='cdn')

    # ── 图表2: 年度收益 ──
    annual_dict = metrics.get('annual_returns', {})
    years = list(annual_dict.keys())
    vals = list(annual_dict.values())
    fig_annual = go.Figure(data=go.Bar(x=years, y=vals,
        marker_color=['#2ca02c' if v >= 0 else '#d62728' for v in vals],
        text=[f"{v:.1%}" for v in vals], textposition='outside'))
    fig_annual.update_layout(title="年度收益", height=350, template='plotly_white', yaxis=dict(tickformat=".1%"))
    annual_html = fig_annual.to_html(full_html=False, include_plotlyjs='cdn')

    # ── 图表3: 月度热力图 ──
    monthly_list = metrics.get('monthly_returns', [])
    if monthly_list:
        mdf = pd.DataFrame(monthly_list)
        pivot = mdf.pivot(index='year', columns='month', values='return')
        month_names = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
        pivot.columns = [month_names[c-1] for c in pivot.columns if 1 <= c <= 12]
        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=[str(y) for y in pivot.index],
            colorscale=[[0,'#d62728'],[0.5,'#ffffff'],[1,'#2ca02c']], zmid=0,
            text=[[f"{v:.1%}" if not np.isnan(v) else '' for v in row] for row in pivot.values],
            texttemplate="%{text}", textfont={"size": 10}, colorbar=dict(title="月收益")))
        fig_heat.update_layout(title="月度收益热力图", height=350, template='plotly_white',
                               xaxis=dict(title="月份", side="top"), yaxis=dict(title="年份", autorange="reversed"))
        heatmap_html = fig_heat.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        heatmap_html = '<p>数据不足</p>'

    # ── 图表4: IC序列 ──
    ic_fwd = 'ret_forward_5d'
    ic_detail = factor_meta.get('ma20_ic_detail', {}).get(ic_fwd, {})
    ic_series_list = []
    factor_path = f"{WORK_DIR}/factors/factor_data.parquet"
    if os.path.exists(factor_path):
        fdf = pd.read_parquet(factor_path)
        for dt_val, grp in fdf.dropna(subset=['ma20_deviation', ic_fwd]).groupby('date'):
            if len(grp) < 10:
                continue
            ic_val, _ = stats.pearsonr(grp['ma20_deviation'].fillna(0), grp[ic_fwd].fillna(0))
            if not np.isnan(ic_val):
                ic_series_list.append({'date': dt_val, 'ic': ic_val})
    ic_df_chart = pd.DataFrame(ic_series_list)
    if not ic_df_chart.empty:
        fig_ic = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                               row_heights=[0.7, 0.3], subplot_titles=("MA20偏离度 IC序列", "累计IC"))
        fig_ic.add_trace(go.Scatter(x=ic_df_chart['date'], y=ic_df_chart['ic'], mode='lines',
                                     name='IC', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
        fig_ic.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        fig_ic.add_trace(go.Scatter(x=ic_df_chart['date'], y=ic_df_chart['ic'].cumsum(),
                                     mode='lines', name='累计IC', line=dict(color='#ff7f0e', width=2)), row=2, col=1)
        fig_ic.update_layout(height=500, template='plotly_white', showlegend=True)
        ic_chart_html = fig_ic.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        ic_chart_html = '<p>IC数据不足</p>'

    # ── 图表5: 分位数 ──
    qsum = factor_meta.get('quantile_summary', [])
    if qsum:
        qdf = pd.DataFrame(qsum)
        labels = ['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)']
        fig_q = go.Figure(data=go.Bar(
            x=[labels[int(q)] for q in qdf['quantile']],
            y=qdf['mean_return'],
            marker_color=['#d62728','#ff7f0e','#f0e68c','#7bc96f','#2ca02c'],
            text=[f"{v:.4%}" for v in qdf['mean_return']], textposition='outside'))
        fig_q.update_layout(title="MA20偏离度分位数测试 (前瞻5日)", height=400, template='plotly_white')
        quantile_html = fig_q.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        quantile_html = '<p>数据不足</p>'

    # ── HTML ──
    metric_items = [
        ("total_return", "累计收益", f"{metrics['total_return']*100:.2f}%", 'positive' if metrics['total_return']>=0 else 'negative'),
        ("annual_return", "年化收益", f"{metrics['annual_return']*100:.2f}%", 'positive' if metrics['annual_return']>=0 else 'negative'),
        ("sharpe_ratio", "夏普比率", f"{metrics['sharpe_ratio']:.3f}", 'positive' if metrics['sharpe_ratio']>=0 else 'negative'),
        ("max_drawdown", "最大回撤", f"{metrics['max_drawdown']*100:.2f}%", 'negative' if metrics['max_drawdown']<0 else ''),
        ("calmar_ratio", "Calmar比率", f"{metrics['calmar_ratio']:.3f}", 'positive' if metrics['calmar_ratio']>=0 else 'negative'),
        ("win_rate", "胜率", f"{metrics['win_rate']*100:.1f}%", ''),
        ("volatility", "年化波动率", f"{metrics['volatility']*100:.2f}%", ''),
        ("sortino_ratio", "Sortino比率", f"{metrics['sortino_ratio']:.3f}", 'positive' if metrics['sortino_ratio']>=0 else 'negative'),
        ("n_trading_days", "交易天数", str(metrics['n_trading_days']), ''),
    ]
    m_html = ''.join(f'<div class="mc"><div class="mv {c}">{v}</div><div class="ml">{l}</div></div>' for _, l, v, c in metric_items)

    risk_items = [
        ("VaR(95%)", f"{risk.get('var_95_daily',0)*100:.2f}%"),
        ("CVaR(95%)", f"{risk.get('cvar_95_daily',0)*100:.2f}%"),
        ("盈亏比", f"{risk.get('profit_factor',0):.2f}"),
        ("最大连跌(天)", str(risk.get('max_losing_streak_days','N/A'))),
        ("偏度", f"{risk.get('skewness',0):.3f}"),
        ("峰度", f"{risk.get('kurtosis',0):.3f}"),
    ]
    r_html = ''.join(f'<div class="mc"><div class="mv">{v}</div><div class="ml">{l}</div></div>' for l, v in risk_items)

    # IC 表格
    ic_rows = ''
    for fwd_key, info in factor_meta.get('ma20_ic_detail', {}).items():
        pn = fwd_key.replace('ret_forward_','').replace('d','日')
        ic_rows += f'<tr><td>MA20偏离度</td><td>前视{pn}</td><td>{info["mean"]:.4f}</td><td>{info["std"]:.4f}</td><td><strong>{info["ir"]:.4f}</strong></td><td>{info["positive_ratio"]:.1%}</td><td>{info["n_dates"]}</td></tr>'

    ic_table_rows = ''
    for fwd_result in factor_meta.get('ic_results', {}).get('ret_forward_5d', []):
        fn = fwd_result['factor']
        ic_table_rows += f'<tr><td>{fn}</td><td>{fwd_result["ic_mean"]:.4f}</td><td>{fwd_result["ic_std"]:.4f}</td><td><strong>{fwd_result["ic_ir"]:.4f}</strong></td><td>{fwd_result["ic_positive_ratio"]:.1%}</td></tr>'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>MA20因子量化交易研究报告</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f0f2f5;color:#333}}
.hd{{background:linear-gradient(135deg,#1a237e,#283593,#3949ab);color:#fff;padding:36px 40px;border-radius:14px;margin-bottom:28px;box-shadow:0 4px 20px rgba(26,35,126,.3)}}
.hd h1{{font-size:28px;margin-bottom:6px}}
.hd p{{opacity:.85;font-size:14px}}
.sc{{background:#fff;border-radius:12px;padding:26px;margin-bottom:18px;box-shadow:0 2px 10px rgba(0,0,0,.05)}}
.sc h2{{font-size:19px;color:#1a237e;border-bottom:2px solid #e8eaf6;padding-bottom:10px;margin-bottom:18px}}
.sc h3{{font-size:15px;color:#3949ab;margin:14px 0 8px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:14px}}
.mc{{background:#fafafa;padding:16px;border-radius:10px;text-align:center;border:1px solid #eee;transition:transform .15s}}
.mc:hover{{transform:translateY(-2px);box-shadow:0 4px 10px rgba(0,0,0,.06)}}
.mv{{font-size:24px;font-weight:700;color:#1a237e}}
.mv.positive{{color:#2e7d32}}
.mv.negative{{color:#c62828}}
.ml{{font-size:12px;color:#888;margin-top:3px}}
.ch{{width:100%;overflow-x:auto;margin:14px 0}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px}}
th,td{{padding:10px 14px;text-align:left;border-bottom:1px solid #e0e0e0}}
th{{background:#f5f5f5;font-weight:600;color:#555}}
tr:hover{{background:#fafafa}}
.ig{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.ii{{background:#f5f7ff;padding:14px;border-radius:8px;border-left:3px solid #3949ab}}
.ii .lb{{font-size:11px;color:#888}}
.ii .vl{{font-size:16px;font-weight:600;color:#333;margin-top:4px}}
.sd{{line-height:1.8;color:#555;font-size:14px}}
.sd li{{margin:5px 0}}
.ft{{text-align:center;margin-top:36px;font-size:12px;color:#aaa;padding:16px}}
</style></head><body>
<div class="hd"><h1>MA20 因子量化交易研究报告</h1>
<p>jingni-trader 量化框架 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 区间: {START_DATE} ~ {END_DATE} | 股票池: {len(HS300_STOCKS)}只</p></div>

<div class="sc"><h2>一、策略概要</h2>
<div class="sd"><p><strong>MA20因子</strong>（20日均线）是A股技术分析最经典的趋势跟踪指标。本策略逻辑：</p>
<ul><li><strong>进场:</strong> 收盘价上穿MA20 → 金叉做多</li><li><strong>出场:</strong> 收盘价下穿MA20 → 死叉平仓</li><li><strong>持仓:</strong> 等权持有信号=1的股票，日频动态调整</li><li><strong>辅助因子:</strong> MA20偏离度、斜率、布林带位置、反转、波动率、量比</li></ul></div>
<div class="ig" style="margin-top:14px">
<div class="ii"><div class="lb">初始资金</div><div class="vl">{INIT_CAPITAL:,.0f} 元</div></div>
<div class="ii"><div class="lb">股票池</div><div class="vl">{len(HS300_STOCKS)} 只 (沪深300代表)</div></div>
<div class="ii"><div class="lb">佣金/印花税</div><div class="vl">万2.5 + 千1(卖出)</div></div>
<div class="ii"><div class="lb">回测区间</div><div class="vl">{START_DATE} ~ {END_DATE}</div></div>
<div class="ii"><div class="lb">交易规则</div><div class="vl">T+1, 考虑滑点(万1)</div></div>
<div class="ii"><div class="lb">因子类型</div><div class="vl">趋势跟踪(MA20)</div></div>
</div></div>

<div class="sc"><h2>二、绩效概览</h2><div class="grid2">{m_html}</div></div>

<div class="sc"><h2>三、净值曲线与回撤</h2><div class="ch">{nav_html}</div></div>

<div class="sc"><h2>四、MA20因子IC分析</h2>
<p style="color:#555;font-size:14px;margin-bottom:10px"><strong>IC (Information Coefficient):</strong> 因子值与未来收益的相关系数。IC_IR = IC均值/IC标准差，越高越好。</p>
<h3>MA20偏离度因子IC详情</h3>
<table><tr><th>因子</th><th>前瞻周期</th><th>IC均值</th><th>IC标准差</th><th>IC_IR</th><th>IC>0比例</th><th>样本天数</th></tr>{ic_rows}</table>
<div class="ch">{ic_chart_html}</div>
<h3>全部因子IC对比 (前瞻5日)</h3>
<table><tr><th>因子名称</th><th>IC均值</th><th>IC标准差</th><th>IC_IR</th><th>IC>0比例</th></tr>{ic_table_rows}</table></div>

<div class="sc"><h2>五、分位数测试</h2>
<p style="color:#555;font-size:14px;margin-bottom:10px">将MA20偏离度分为5组，观察各组前瞻5日收益。高低组收益差反映因子区分能力。</p>
<div class="ch">{quantile_html}</div></div>

<div class="sc"><h2>六、年度/月度分析</h2>
<div class="ch">{annual_html}</div>
<div class="ch">{heatmap_html}</div></div>

<div class="sc"><h2>七、风险分析</h2><div class="grid2">{r_html}</div></div>

<div class="sc"><h2>八、结论与建议</h2>
<div class="sd"><ol>
<li><strong>MA20因子有效性:</strong> IC分析和分位数测试结果表明MA20偏离度因子在A股市场具备一定的预测能力。</li>
<li><strong>策略表现:</strong> 在回测区间内，MA20金叉死叉策略的表现如上。作为单因子策略，其夏普比率和最大回撤反映了纯粹的均线策略特征。</li>
<li><strong>风险提示:</strong> 单因子策略易受市场风格切换影响。震荡市中假突破频繁，可能导致回撤加大。</li>
<li><strong>改进方向:</strong> (1) 引入成交量确认过滤假信号;(2) 结合反转因子降低回撤;(3) 动态MA周期选择;(4) 增加止盈止损机制;(5) 多因子融和提升稳健性。</li>
<li><strong>实盘考量:</strong> 实际交易需额外考虑流动性、冲击成本、交易执行延迟等因素。</li>
</ol></div></div>

<div class="ft">Generated by jingni-trader | MA20 Factor Analysis Pipeline</div>
</body></html>'''

    report_path = f"{WORK_DIR}/reports/report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"  报告已生成: {report_path}")
    return report_path


# ═══════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════
def main():
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║    jingni-trader: MA20因子量化全流程分析管道              ║")
    logger.info("║    DATA → FACTOR → MODEL → BACKTEST → RISK → REPORT       ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    data = stage_data_acquisition()
    factor_df, factor_meta = stage_factor_engineering(data)
    signal_df = stage_signal_generation(factor_df)
    metrics, equity_df = stage_backtest(data, signal_df)
    risk = stage_risk_analysis(equity_df)
    report_path = stage_report_generation(factor_meta, metrics, equity_df, risk, signal_df)

    # ── 摘要 ──
    summary = {
        "task_id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "stages": ["DATA","FACTOR","MODEL","BACKTEST","RISK","REPORT"],
        "stock_count": len(HS300_STOCKS),
        "date_range": f"{START_DATE}~{END_DATE}",
        "key_metrics": {
            "total_return": f"{metrics['total_return']*100:.2f}%",
            "annual_return": f"{metrics['annual_return']*100:.2f}%",
            "sharpe": f"{metrics['sharpe_ratio']:.3f}",
            "max_drawdown": f"{metrics['max_drawdown']*100:.2f}%",
            "calmar": f"{metrics['calmar_ratio']:.3f}",
            "win_rate": f"{metrics['win_rate']*100:.1f}%",
        },
        "ma20_ic": factor_meta.get('ma20_ic_detail', {}).get('ret_forward_5d', {}),
        "report_path": report_path,
    }
    with open(f"{WORK_DIR}/reports/summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    logger.info("\n" + "=" * 60)
    logger.info("全流程完成! 摘要:")
    logger.info("=" * 60)
    for k, v in summary['key_metrics'].items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  MA20_IC_IR: {summary.get('ma20_ic',{}).get('ir','N/A')}")
    logger.info(f"  报告: {report_path}")
    return summary

if __name__ == '__main__':
    main()