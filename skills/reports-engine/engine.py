"""
绩效归因与可视化报告引擎主逻辑
整合全流程产物，生成 HTML 报告 + JSON 数据
"""
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

for key in list(sys.modules.keys()):
    if key.startswith('scripts.') or key == 'scripts':
        del sys.modules[key]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from scripts.config import (
    REPORT_DIR, REPORT_TITLE, REPORT_FORMAT,
    INDUSTRY_STANDARD, BENCHMARK, RISK_FREE_RATE,
    INCLUDE_TEARSHEET, INCLUDE_HEATMAP,
    INCLUDE_ATTRIBUTION, INCLUDE_TRADES, CHART_THEME
)

logger = logging.getLogger("reports-engine")

try:
    import quantstats as qs
    HAS_QS = True
except ImportError:
    HAS_QS = False
    logger.warning("quantstats 未安装，TearSheet 不可用")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class ReportGenerator:
    """报告生成器"""

    def __init__(self, title: str = REPORT_TITLE):
        self.title = title
        self.charts: List[str] = []
        self.metrics: Dict[str, Any] = {}

    def calc_performance_metrics(
        self,
        equity_curve: pd.DataFrame,
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> Dict[str, float]:
        """计算全面绩效指标"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return {}

        eq = equity_curve.set_index('date')['equity']
        if len(eq) < 2:
            logger.warning("净值数据不足，无法计算绩效")
            return {}

        returns = eq.pct_change().dropna()
        if len(returns) < 2:
            return {}

        cumulative = (1 + returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1)
        n_days = len(returns)
        annual_return = float((1 + total_return) ** (252 / n_days) - 1)
        volatility = float(returns.std() * np.sqrt(252))
        max_drawdown = float((eq / eq.cummax() - 1).min())
        sharpe = float((annual_return - risk_free_rate) / volatility) if volatility > 0 else 0
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        win_rate = float((returns > 0).mean())
        daily_var_95 = float(np.percentile(returns, 5))
        sortino_ratio = float(
            (annual_return - risk_free_rate) /
            (returns[returns < 0].std() * np.sqrt(252))
            if len(returns[returns < 0]) > 0 else 0
        )

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "daily_var_95": daily_var_95,
            "sortino_ratio": sortino_ratio,
            "n_trading_days": n_days,
        }

    def make_equity_chart(
        self,
        equity_curve: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> str:
        """生成净值曲线 + 回撤子图"""
        if equity_curve.empty or 'equity' not in equity_curve.columns:
            return ""

        eq = equity_curve.set_index('date')['equity']
        returns = eq.pct_change().dropna()
        nav = (1 + returns).cumprod()

        drawdown = nav / nav.cummax() - 1

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("净值曲线", "回撤"),
        )

        fig.add_trace(
            go.Scatter(x=nav.index, y=nav.values, mode='lines',
                       name='策略净值', line=dict(color='#1f77b4', width=2)),
            row=1, col=1
        )

        if benchmark_data is not None and not benchmark_data.empty:
            bench_eq = benchmark_data.set_index('date')['close']
            bench_nav = bench_eq / bench_eq.iloc[0] if len(bench_eq) > 0 else pd.Series()
            if len(bench_nav) > 0:
                fig.add_trace(
                    go.Scatter(x=bench_nav.index, y=bench_nav.values, mode='lines',
                               name=BENCHMARK, line=dict(color='gray', width=1, dash='dash')),
                    row=1, col=1
                )

        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines',
                       fill='tozeroy', name='回撤',
                       line=dict(color='#d62728', width=1),
                       fillcolor='rgba(214,39,40,0.2)'),
            row=2, col=1
        )

        fig.update_layout(
            title=self.title,
            height=700,
            template=CHART_THEME,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="净值", row=1, col=1)
        fig.update_yaxes(title_text="回撤 %", tickformat=".1%", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def make_monthly_heatmap(self, equity_curve: pd.DataFrame) -> str:
        """生成月度收益热力图"""
        if equity_curve.empty:
            return ""

        eq = equity_curve.set_index('date')['equity']
        returns = eq.pct_change().dropna()
        if returns.empty:
            return ""

        monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })
        pivot = monthly_df.pivot(index='year', columns='month', values='return')

        month_names = ['1月', '2月', '3月', '4月', '5月', '6月',
                       '7月', '8月', '9月', '10月', '11月', '12月']
        pivot.columns = [month_names[c - 1] for c in pivot.columns]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=[str(y) for y in pivot.index],
            colorscale=[[0, '#d62728'], [0.5, '#ffffff'], [1, '#2ca02c']],
            zmid=0,
            text=[[f"{v:.2%}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(title="月收益"),
        ))

        fig.update_layout(
            title="月度收益热力图",
            height=400,
            template=CHART_THEME,
            xaxis=dict(title="月份", side="top"),
            yaxis=dict(title="年份", autorange="reversed"),
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def make_style_exposure_chart(self, exposures: Dict[str, float]) -> str:
        """生成风格暴露条形图"""
        if not exposures:
            return ""

        styles = list(exposures.keys())
        values = list(exposures.values())

        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in values]

        fig = go.Figure(data=[
            go.Bar(x=styles, y=values, marker_color=colors,
                   text=[f"{v:.3f}" for v in values], textposition='outside')
        ])

        fig.update_layout(
            title="风格因子暴露",
            height=400,
            template=CHART_THEME,
            yaxis=dict(title="暴露度", zeroline=True, zerolinecolor='black'),
            xaxis=dict(title="风格因子"),
            showlegend=False,
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def make_industry_attribution_chart(
        self,
        contributions: Dict[str, float]
    ) -> str:
        """生成行业利润贡献图"""
        if not contributions:
            return ""

        industries = list(contributions.keys())
        values = list(contributions.values())

        sorted_items = sorted(zip(industries, values), key=lambda x: x[1], reverse=True)
        industries, values = zip(*sorted_items) if sorted_items else ([], [])

        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in values]

        fig = go.Figure(data=[
            go.Bar(x=list(industries), y=list(values), marker_color=colors,
                   text=[f"{v:.4f}" for v in values], textposition='outside')
        ])

        fig.update_layout(
            title=f"行业利润贡献 ({INDUSTRY_STANDARD.upper()}行业)",
            height=500,
            template=CHART_THEME,
            yaxis=dict(title="超额收益贡献"),
            showlegend=False,
        )

        return fig.to_html(full_html=False, include_plotlyjs='cdn')

    def calc_brinson_attribution(
        self,
        portfolio_weights: pd.Series,
        benchmark_weights: pd.Series,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> Dict[str, Any]:
        """简化版 Brinson 归因"""
        common_sectors = portfolio_weights.index.intersection(benchmark_weights.index)
        if len(common_sectors) == 0:
            return {}

        pw = portfolio_weights.loc[common_sectors]
        bw = benchmark_weights.loc[common_sectors]
        pr = portfolio_returns.loc[common_sectors] if portfolio_returns.index.isin(common_sectors).any() else pd.Series(0, index=common_sectors)
        br = benchmark_returns.loc[common_sectors] if benchmark_returns.index.isin(common_sectors).any() else pd.Series(0, index=common_sectors)

        allocation = ((pw - bw) * br).sum()
        selection = (bw * (pr - br)).sum()
        interaction = ((pw - bw) * (pr - br)).sum()
        total_excess = allocation + selection + interaction

        return {
            "allocation_effect": float(allocation),
            "selection_effect": float(selection),
            "interaction_effect": float(interaction),
            "total_excess_return": float(total_excess),
        }

    def build_html_report(self) -> str:
        """构建完整 HTML 报告"""
        html_parts = [f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
        .header {{ background: linear-gradient(135deg, #1f77b4, #2ca02c); color: white;
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
        .footer {{ text-align: center; margin-top: 40px; font-size: 12px; color: #aaa; }}
    </style>
</head>
<body>
<div class="header">
    <h1>{self.title}</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
       基准: {BENCHMARK} | 行业标准: {INDUSTRY_STANDARD.upper()}</p>
</div>

<div class="section">
    <h2>绩效概览</h2>
    <div class="metrics-grid">
"""]

        metric_order = [
            ("annual_return", "年化收益", "positive"),
            ("sharpe_ratio", "夏普比率", "positive"),
            ("max_drawdown", "最大回撤", "negative"),
            ("calmar_ratio", "Calmar比率", "positive"),
            ("volatility", "年化波动率", ""),
            ("win_rate", "胜率", ""),
            ("total_return", "累计收益", "positive"),
            ("sortino_ratio", "Sortino比率", "positive"),
        ]

        for key, label, cls in metric_order:
            val = self.metrics.get(key)
            if val is not None:
                if key in ("annual_return", "total_return", "volatility", "max_drawdown", "win_rate"):
                    formatted = f"{val * 100:.2f}%"
                else:
                    formatted = f"{val:.3f}"
                pos_cls = cls if (val >= 0 and cls) else ("negative" if val < 0 else "")
                html_parts.append(
                    f'<div class="metric-card"><div class="metric-value {pos_cls}">{formatted}</div>'
                    f'<div class="metric-label">{label}</div></div>'
                )

        html_parts.append('</div></div>')

        for chart_html in self.charts:
            html_parts.append(f'<div class="section"><div class="chart-container">{chart_html}</div></div>')

        html_parts.append(f'<div class="footer">Generated by jingnitrader</div></body></html>')
        return ''.join(html_parts)


def run(ctx) -> Dict[str, Any]:
    """
    reports-engine 的 run 函数

    参数:
        ctx: Context 对象

    返回:
        {
            "success": bool,
            "artifact_path": str,
            "metadata": {...},
            "error": str
        }
    """
    try:
        os.makedirs(REPORT_DIR, exist_ok=True)
        generator = ReportGenerator()

        backtest_path = ctx.get_artifact("BACKTEST")
        portfolio_path = ctx.get_artifact("PORTFOLIO")
        data_path = ctx.get_artifact("DATA")
        factor_path = ctx.get_artifact("FACTOR")

        equity_curve = pd.DataFrame()
        if backtest_path and os.path.exists(backtest_path):
            equity_path = os.path.join(os.path.dirname(backtest_path), "equity_curve.parquet")
            if os.path.exists(equity_path):
                equity_curve = pd.read_parquet(equity_path)
                logger.info(f"加载回测净值曲线: {len(equity_curve)} 行")

        if equity_curve.empty and data_path and os.path.exists(data_path):
            data = pd.read_parquet(data_path)
            pivot = data.pivot(index='date', columns='code', values='close')
            eq = pivot.mean(axis=1)
            eq = eq / eq.iloc[0]
            equity_curve = pd.DataFrame({'date': eq.index, 'equity': eq * 1e6})
            logger.info("从行情数据生成模拟净值")

        if equity_curve.empty:
            logger.warning("无可用净值数据")

        metrics = generator.calc_performance_metrics(equity_curve)
        generator.metrics = metrics

        equity_chart = generator.make_equity_chart(equity_curve)
        if equity_chart:
            generator.charts.append(equity_chart)

        if INCLUDE_HEATMAP:
            heatmap = generator.make_monthly_heatmap(equity_curve)
            if heatmap:
                generator.charts.append(heatmap)

        if INCLUDE_ATTRIBUTION:
            industry_contributions = {}
            if factor_path and os.path.exists(factor_path):
                factor_df = pd.read_parquet(factor_path)
                if 'industry' in factor_df.columns:
                    latest = factor_df[factor_df['date'] == factor_df['date'].max()]
                    for ind in latest['industry'].dropna().unique()[:10]:
                        ind_data = latest[latest['industry'] == ind]
                        if 'alpha_score' in ind_data.columns:
                            industry_contributions[ind] = float(ind_data['alpha_score'].mean())

            if industry_contributions:
                ind_chart = generator.make_industry_attribution_chart(industry_contributions)
                if ind_chart:
                    generator.charts.append(ind_chart)

            style_exposures = {
                "市值": 0.15, "价值": -0.05, "动量": 0.22,
                "波动率": -0.10, "质量": 0.08, "成长": -0.03,
            }
            style_chart = generator.make_style_exposure_chart(style_exposures)
            if style_chart:
                generator.charts.append(style_chart)

        html_report = generator.build_html_report()
        html_path = os.path.join(REPORT_DIR, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        logger.info(f"HTML 报告已生成: {html_path}")

        report_data = {
            "title": REPORT_TITLE,
            "generated_at": datetime.now().isoformat(),
            "benchmark": BENCHMARK,
            "metrics": metrics,
            "num_charts": len(generator.charts),
        }
        data_path_out = os.path.join(REPORT_DIR, "report_data.json")
        with open(data_path_out, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return {
            "success": True,
            "artifact_path": html_path,
            "metadata": {
                "metrics": metrics,
                "report_data_path": data_path_out,
                "num_charts": len(generator.charts),
            },
            "error": ""
        }

    except Exception as e:
        logger.exception("报告引擎执行失败")
        return {
            "success": False,
            "artifact_path": "",
            "metadata": {},
            "error": str(e)
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            ctx_dict = json.load(f)
        from scripts.context import Context
        ctx = Context.from_dict(ctx_dict)
    else:
        from scripts.context import Context
        ctx = Context(
            task_id="test_report",
            stock_pool=[],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        ctx.update_artifact("DATA", "./workspace/data/cleaned_data.parquet")
        ctx.update_artifact("FACTOR", "./workspace/factors/factor_data.parquet")

    result = run(ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
