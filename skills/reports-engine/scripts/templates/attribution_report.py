"""
绩效归因报告模板

生成完整的 HTML 绩效归因报告，包含：
1. 交易统计概览
2. Round-Trip 盈亏归因
3. 按标的盈亏明细
4. 净值曲线 + 回撤
5. 执行质量分析
6. A 股压力期表现
7. LLM 深度解读
"""
import os
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("reports-engine.attribution_template")


def make_pnl_by_stock_chart(pnl_df: pd.DataFrame, chart_theme: str = "plotly_white") -> str:
    """生成按标的盈亏条形图"""
    if pnl_df.empty:
        return ""

    df = pnl_df.head(20)  # 最多展示 20 只
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in df['total_pnl']]

    fig = go.Figure(data=[
        go.Bar(
            x=df['code'],
            y=df['total_pnl'],
            marker_color=colors,
            text=[f"{v:,.2f}" for v in df['total_pnl']],
            textposition='outside',
        )
    ])

    fig.update_layout(
        title="按标的盈亏归因",
        height=500,
        template=chart_theme,
        yaxis=dict(title="净盈亏（元）", zeroline=True, zerolinecolor='black'),
        xaxis=dict(title="股票代码"),
        showlegend=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def make_round_trip_scatter(round_trips: list, chart_theme: str = "plotly_white") -> str:
    """生成 Round-Trip 散点图：持仓天数 vs 收益率，颜色区分盈亏"""
    if not round_trips:
        return ""

    holding_days = [rt.holding_days for rt in round_trips]
    returns = [rt.return_pct * 100 for rt in round_trips]
    codes = [rt.code for rt in round_trips]
    pnls = [rt.net_pnl for rt in round_trips]
    colors = ['#2ca02c' if rt.is_win else '#d62728' for rt in round_trips]

    fig = go.Figure(data=go.Scatter(
        x=holding_days,
        y=returns,
        mode='markers',
        marker=dict(size=8, color=colors, opacity=0.7),
        text=[f"{c}<br>盈亏: {p:,.2f}<br>持仓: {h}天<br>收益: {r:.2f}%"
              for c, p, h, r in zip(codes, pnls, holding_days, returns)],
        hoverinfo='text',
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Round-Trip 分析（持仓天数 vs 收益率）",
        height=450,
        template=chart_theme,
        xaxis=dict(title="持仓天数"),
        yaxis=dict(title="收益率 (%)"),
        showlegend=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def build_attribution_html(
    metrics: dict,
    tx_stats: dict,
    rt_stats: dict,
    pnl_by_stock: pd.DataFrame,
    exec_quality: dict,
    stress_perf: dict,
    consecutive: dict,
    charts: list,
    round_trips: list,
    chart_theme: str = "plotly_white",
) -> str:
    """构建绩效归因报告 HTML

    参数:
        metrics: 从净值曲线计算的绩效指标
        tx_stats: 交易统计概览
        rt_stats: Round-trip 统计
        pnl_by_stock: 按标的归因 DataFrame
        exec_quality: 执行质量分析
        stress_perf: 压力期表现
        consecutive: 连胜/连败统计
        charts: 图表 HTML 列表
        round_trips: RoundTrip 对象列表
    """
    from datetime import datetime

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 核心指标卡片
    cards_html = []
    if metrics:
        card_items = [
            ("total_return", "累计收益", True),
            ("annual_return", "年化收益", True),
            ("sharpe_ratio", "夏普比率", True),
            ("max_drawdown", "最大回撤", False),
            ("calmar_ratio", "Calmar比率", True),
            ("win_rate", "胜率", None),
        ]
        for key, label, positive_oriented in card_items:
            val = metrics.get(key)
            if val is None:
                continue
            if key in ("total_return", "annual_return", "max_drawdown", "win_rate", "volatility"):
                formatted = f"{val * 100:.2f}%"
            else:
                formatted = f"{val:.3f}"
            cls = ""
            if positive_oriented is True:
                cls = "positive" if val >= 0 else "negative"
            elif positive_oriented is False:
                cls = "negative" if val < 0 else "positive"
            cards_html.append(
                f'<div class="metric-card"><div class="metric-value {cls}">{formatted}</div>'
                f'<div class="metric-label">{label}</div></div>'
            )

    # 交易统计表
    tx_rows = ""
    if tx_stats:
        tx_rows = f"""
        <tr><td>总成交笔数</td><td>{tx_stats.get('total_trades', 0)}</td></tr>
        <tr><td>买入笔数</td><td>{tx_stats.get('total_buys', 0)}</td></tr>
        <tr><td>卖出笔数</td><td>{tx_stats.get('total_sells', 0)}</td></tr>
        <tr><td>涉及标的数</td><td>{tx_stats.get('unique_stocks', 0)}</td></tr>
        <tr><td>总佣金</td><td>{tx_stats.get('total_commission', 0):,.2f}</td></tr>
        <tr><td>总印花税</td><td>{tx_stats.get('total_stamp_tax', 0):,.2f}</td></tr>
        <tr><td>总滑点成本</td><td>{tx_stats.get('total_slippage', 0):,.2f}</td></tr>
        <tr><td>总交易成本</td><td><b>{tx_stats.get('total_cost', 0):,.2f}</b></td></tr>
        """

    # Round-trip 统计
    rt_rows = ""
    if rt_stats:
        win_rate_pct = rt_stats.get('win_rate', 0) * 100
        rt_rows = f"""
        <tr><td>闭环交易数</td><td>{rt_stats.get('total_round_trips', 0)}</td></tr>
        <tr><td>盈利次数</td><td style="color:#2ca02c">{rt_stats.get('win_count', 0)}</td></tr>
        <tr><td>亏损次数</td><td style="color:#d62728">{rt_stats.get('loss_count', 0)}</td></tr>
        <tr><td>胜率</td><td><b>{win_rate_pct:.1f}%</b></td></tr>
        <tr><td>总净盈亏</td><td><b>{rt_stats.get('total_net_pnl', 0):,.2f}</b></td></tr>
        <tr><td>平均收益率</td><td>{rt_stats.get('avg_return_pct', 0):.2f}%</td></tr>
        <tr><td>平均盈利</td><td style="color:#2ca02c">{rt_stats.get('avg_win', 0):,.2f}</td></tr>
        <tr><td>平均亏损</td><td style="color:#d62728">{rt_stats.get('avg_loss', 0):,.2f}</td></tr>
        <tr><td>盈亏比</td><td>{rt_stats.get('profit_factor', 0):.2f}</td></tr>
        <tr><td>平均持仓天数</td><td>{rt_stats.get('avg_holding_days', 0):.1f}</td></tr>
        """
        if consecutive:
            rt_rows += f"""
            <tr><td>最大连胜</td><td style="color:#2ca02c">{consecutive.get('max_win_streak', 0)}</td></tr>
            <tr><td>最大连败</td><td style="color:#d62728">{consecutive.get('max_loss_streak', 0)}</td></tr>
            """

    # 按标的归因表
    stock_rows = ""
    if not pnl_by_stock.empty:
        for _, row in pnl_by_stock.head(20).iterrows():
            pnl_cls = "positive" if row['total_pnl'] >= 0 else "negative"
            stock_rows += (
                f"<tr>"
                f"<td>{row['code']}</td>"
                f"<td class='{pnl_cls}'>{row['total_pnl']:,.2f}</td>"
                f"<td>{row['trade_count']}</td>"
                f"<td>{row['win_count']}</td>"
                f"<td>{row['win_rate']:.1f}%</td>"
                f"<td>{row['avg_return_pct']:.2f}%</td>"
                f"</tr>"
            )

    # 执行质量表
    eq_rows = ""
    if exec_quality:
        eq_rows = f"""
        <tr><td>总成交额</td><td>{exec_quality.get('total_turnover', 0):,.2f}</td></tr>
        <tr><td>总交易成本</td><td>{exec_quality.get('total_cost', 0):,.2f}</td></tr>
        <tr><td>成本占比</td><td>{exec_quality.get('cost_ratio_bps', 0):.2f} bps</td></tr>
        <tr><td>滑点占比</td><td>{exec_quality.get('slippage_ratio_bps', 0):.2f} bps</td></tr>
        <tr><td>平均单笔规模</td><td>{exec_quality.get('avg_trade_size', 0):,.2f}</td></tr>
        <tr><td>最大单笔规模</td><td>{exec_quality.get('max_trade_size', 0):,.2f}</td></tr>
        """

    # 压力期表现表
    stress_rows = ""
    if stress_perf:
        for name, data in stress_perf.items():
            ret_cls = "positive" if data['return_pct'] >= 0 else "negative"
            stress_rows += (
                f"<tr>"
                f"<td>{name}</td>"
                f"<td class='{ret_cls}'>{data['return_pct']:.2f}%</td>"
                f"<td class='negative'>{data['max_drawdown_pct']:.2f}%</td>"
                f"</tr>"
            )

    # Round-trip 明细表
    rt_detail_rows = ""
    if round_trips:
        for rt in round_trips[:50]:  # 最多 50 条
            pnl_cls = "positive" if rt.net_pnl >= 0 else "negative"
            rt_detail_rows += (
                f"<tr>"
                f"<td>{rt.code}</td>"
                f"<td>{rt.buy_date}</td>"
                f"<td>{rt.sell_date}</td>"
                f"<td>{rt.shares}</td>"
                f"<td>{rt.buy_price:.2f}</td>"
                f"<td>{rt.sell_price:.2f}</td>"
                f"<td>{rt.holding_days}</td>"
                f"<td class='{pnl_cls}'>{rt.net_pnl:,.2f}</td>"
                f"<td class='{pnl_cls}'>{rt.return_pct * 100:.2f}%</td>"
                f"</tr>"
            )

    # 图表区
    charts_html = "".join(
        f'<div class="section"><div class="chart-container">{c}</div></div>'
        for c in charts
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>绩效归因报告</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
    .header {{ background: linear-gradient(135deg, #6c5ce7, #0984e3); color: white;
               padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
    .header h1 {{ margin: 0 0 10px 0; font-size: 28px; }}
    .header p {{ margin: 0; opacity: 0.9; }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 16px; margin-bottom: 30px; }}
    .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    text-align: center; }}
    .metric-value {{ font-size: 28px; font-weight: 700; color: #1f77b4; }}
    .metric-label {{ font-size: 13px; color: #888; margin-top: 4px; }}
    .metric-value.positive {{ color: #2ca02c; }}
    .metric-value.negative {{ color: #d62728; }}
    .section {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .section h2 {{ margin: 0 0 16px 0; font-size: 18px; color: #444;
                   border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    .chart-container {{ width: 100%; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #f9f9f9; font-weight: 600; color: #555; }}
    tr:hover {{ background: #fafafa; }}
    .positive {{ color: #2ca02c; }}
    .negative {{ color: #d62728; }}
    .llm-section {{ background: linear-gradient(135deg, #f8f9fa, #e9ecef); }}
    .footer {{ text-align: center; margin-top: 40px; font-size: 12px; color: #aaa; }}
    .disclaimer {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;
                   padding: 16px; margin-top: 20px; font-size: 13px; color: #856404; }}
</style>
</head>
<body>
<div class="header">
    <h1>绩效归因报告</h1>
    <p>生成时间: {now_str} | 报告类型: 实盘/模拟交易绩效复盘</p>
</div>

<div class="section">
    <h2>报告概览</h2>
    <div class="metrics-grid">
        {''.join(cards_html)}
    </div>
</div>

{charts_html}

<div class="section">
    <h2>交易统计概览</h2>
    <table>
        <thead><tr><th>指标</th><th>数值</th></tr></thead>
        <tbody>{tx_rows}</tbody>
    </table>
</div>

<div class="section">
    <h2>Round-Trip 盈亏归因</h2>
    <table>
        <thead><tr><th>指标</th><th>数值</th></tr></thead>
        <tbody>{rt_rows}</tbody>
    </table>
</div>

{'<div class="section"><h2>按标的盈亏明细</h2><table><thead><tr><th>股票代码</th><th>总净盈亏</th><th>交易次数</th><th>盈利次数</th><th>胜率</th><th>平均收益率</th></tr></thead><tbody>' + stock_rows + '</tbody></table></div>' if stock_rows else ''}

<div class="section">
    <h2>执行质量分析</h2>
    <table>
        <thead><tr><th>指标</th><th>数值</th></tr></thead>
        <tbody>{eq_rows}</tbody>
    </table>
</div>

{'<div class="section"><h2>A股压力期表现</h2><table><thead><tr><th>压力期</th><th>区间收益</th><th>最大回撤</th></tr></thead><tbody>' + stress_rows + '</tbody></table></div>' if stress_rows else ''}

{'<div class="section"><h2>Round-Trip 明细</h2><table><thead><tr><th>股票代码</th><th>买入日</th><th>卖出日</th><th>数量</th><th>买入价</th><th>卖出价</th><th>持仓天数</th><th>净盈亏</th><th>收益率</th></tr></thead><tbody>' + rt_detail_rows + '</tbody></table></div>' if rt_detail_rows else ''}

<div class="section llm-section">
    <h2>深度解读</h2>
    <!--LLM_ATTRIBUTION_PLACEHOLDER-->
</div>

<div class="disclaimer">
    <b>风险提示：</b>本报告仅基于历史交易数据进行归因分析，不构成投资建议。历史业绩不代表未来表现。
    交易成本（佣金/印花税/滑点）为实际记录值，可能与理论值存在偏差。
</div>

<div class="footer">Generated by jingni-trader reports-engine</div>
</body>
</html>"""
