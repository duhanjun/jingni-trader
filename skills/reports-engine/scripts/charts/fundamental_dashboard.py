"""
基本面仪表盘图表生成模块
生成PE/PB历史分位图、ROE趋势图、行业对比雷达图
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

try:
    from scripts.config import CHART_THEME
except ImportError:
    CHART_THEME = "plotly_white"

# A股颜色惯例
UP_COLOR = '#d62728'    # 红
DOWN_COLOR = '#2ca02c'  # 绿
NEUTRAL_COLOR = '#1f77b4'

# 指标中文标签映射
METRIC_LABELS = {
    'PE': '市盈率(PE)',
    'PB': '市净率(PB)',
    'PS': '市销率(PS)',
    'PCF': '市现率(PCF)',
    'ROE': 'ROE',
    'gross_margin': '毛利率',
    'net_margin': '净利率',
    'revenue_growth': '营收增速',
    'profit_growth': '利润增速',
    'debt_ratio': '资产负债率',
    'current_ratio': '流动比率',
}


class FundamentalDashboardGenerator:
    """基本面图表生成器"""

    def generate_valuation_gauge(self, metric: str, current_value: float,
                                 percentile: float, height: int = 250) -> go.Figure:
        """生成估值分位仪表盘（半圆形gauge）"""
        # 标准化percentile到0-100
        pct = percentile * 100 if percentile <= 1 else percentile
        pct = max(0, min(100, pct))

        label = METRIC_LABELS.get(metric.upper() if isinstance(metric, str) else '',
                                  metric if isinstance(metric, str) else '估值')

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': "%", 'font': {'size': 36}},
            title={'text': f"{label}<br><span style='font-size:14px;color:#888'>当前值: {current_value:.2f}</span>"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888"},
                'bar': {'color': NEUTRAL_COLOR},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ccc",
                'steps': [
                    {'range': [0, 30], 'color': '#2ca02c'},    # 低估 - 绿
                    {'range': [30, 70], 'color': '#ff9f43'},    # 合理 - 黄
                    {'range': [70, 100], 'color': '#d62728'},   # 高估 - 红
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.85,
                    'value': pct,
                }
            }
        ))

        # 估值结论
        if pct < 30:
            conclusion = "低估"
            color = DOWN_COLOR
        elif pct < 70:
            conclusion = "合理"
            color = '#ff9f43'
        else:
            conclusion = "高估"
            color = UP_COLOR

        fig.update_layout(
            title=f"{label}历史分位",
            height=height,
            template=CHART_THEME,
            annotations=[dict(
                text=f"<b style='color:{color}'>{conclusion}</b>",
                x=0.5, y=-0.15,
                xref='paper', yref='paper',
                showarrow=False,
                font=dict(size=16),
            )],
        )

        return fig

    def generate_roe_trend(self, df: pd.DataFrame, height: int = 300) -> go.Figure:
        """生成ROE/毛利率趋势柱状图"""
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        df = df.copy()

        # 确定时间列
        date_col = 'date' if 'date' in df.columns else (
            'period' if 'period' in df.columns else df.columns[0])
        if date_col == 'date':
            dates = pd.to_datetime(df[date_col])
        else:
            dates = df[date_col].astype(str)

        fig = go.Figure()

        # ROE柱状图
        if 'roe' in df.columns:
            roe = df['roe']
            colors = [UP_COLOR if v >= 15 else (NEUTRAL_COLOR if v >= 0 else DOWN_COLOR)
                      for v in roe]
            fig.add_trace(go.Bar(
                x=dates, y=roe,
                name='ROE(%)',
                marker_color=colors,
                text=[f"{v:.2f}%" for v in roe],
                textposition='outside',
            ))

        # 毛利率线图
        if 'gross_margin' in df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=df['gross_margin'],
                mode='lines+markers', name='毛利率(%)',
                line=dict(color='#ff7f0e', width=2),
                yaxis='y2',
            ))

        # 净利率线图
        if 'net_margin' in df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=df['net_margin'],
                mode='lines+markers', name='净利率(%)',
                line=dict(color='#9467bd', width=2),
                yaxis='y2',
            ))

        fig.update_layout(
            title="ROE / 利润率趋势",
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            yaxis=dict(title="ROE(%)", side="left"),
            yaxis2=dict(title="利润率(%)", side="right", overlaying="y"),
            xaxis=dict(title="期间"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='group',
        )

        return fig

    def generate_industry_radar(self, stock_values: Dict, industry_avg: Dict,
                                metrics: List[str] = None, height: int = 400) -> go.Figure:
        """生成行业对比雷达图"""
        if not stock_values or not industry_avg:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        if metrics is None:
            metrics = list(stock_values.keys())

        # 过滤有效指标
        valid_metrics = []
        for m in metrics:
            if m not in stock_values or m not in industry_avg:
                continue
            sv = stock_values[m]
            iv = industry_avg[m]
            if sv is None or iv is None:
                continue
            if isinstance(sv, float) and np.isnan(sv):
                continue
            if isinstance(iv, float) and np.isnan(iv):
                continue
            valid_metrics.append(m)

        if not valid_metrics:
            fig = go.Figure()
            fig.update_layout(title="无有效指标数据", template=CHART_THEME)
            return fig

        # 归一化到0-100：以最大值为100基准
        stock_normalized = []
        industry_normalized = []
        for m in valid_metrics:
            sv = float(stock_values[m])
            iv = float(industry_avg[m])
            max_val = max(abs(sv), abs(iv), 1e-6)
            stock_normalized.append(sv / max_val * 100 if max_val != 0 else 50)
            industry_normalized.append(iv / max_val * 100 if max_val != 0 else 50)

        labels = [METRIC_LABELS.get(m, m) for m in valid_metrics]

        fig = go.Figure()

        # 个股
        fig.add_trace(go.Scatterpolar(
            r=stock_normalized + [stock_normalized[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name='个股',
            line=dict(color=UP_COLOR, width=2),
            fillcolor='rgba(214,39,40,0.2)',
        ))

        # 行业均值
        fig.add_trace(go.Scatterpolar(
            r=industry_normalized + [industry_normalized[0]],
            theta=labels + [labels[0]],
            fill='toself',
            name='行业均值',
            line=dict(color=NEUTRAL_COLOR, width=2),
            fillcolor='rgba(31,119,180,0.2)',
        ))

        fig.update_layout(
            title="行业对比雷达图",
            height=height,
            template=CHART_THEME,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 110],
                    tickvals=[20, 40, 60, 80, 100],
                ),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        )

        return fig

    def generate_combined_dashboard(self, valuation_data: Dict,
                                    roe_data: pd.DataFrame,
                                    industry_data: Dict,
                                    height: int = 600) -> go.Figure:
        """生成综合基本面仪表盘"""
        # 判断可用组件
        has_valuation = bool(valuation_data)
        has_roe = roe_data is not None and not roe_data.empty
        has_industry = bool(industry_data and industry_data.get('stock')
                            and industry_data.get('industry_avg'))

        if not any([has_valuation, has_roe, has_industry]):
            fig = go.Figure()
            fig.update_layout(title="无基本面数据", template=CHART_THEME)
            return fig

        # 单组件情况直接调用对应生成器
        components = []
        if has_valuation:
            components.append('valuation')
        if has_roe:
            components.append('roe')
        if has_industry:
            components.append('industry')

        if len(components) == 1:
            if components[0] == 'valuation':
                metric, current, pct = self._extract_valuation(valuation_data)
                return self.generate_valuation_gauge(metric, current, pct, height)
            elif components[0] == 'roe':
                return self.generate_roe_trend(roe_data, height)
            else:
                return self.generate_industry_radar(
                    industry_data['stock'], industry_data['industry_avg'], height=height)

        # 多组件组合布局
        specs, rows = self._build_dashboard_specs(has_valuation, has_roe, has_industry)

        fig = make_subplots(
            rows=rows, cols=2,
            specs=specs,
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        col = 1

        # 估值gauge
        if has_valuation:
            metric, current, pct = self._extract_valuation(valuation_data)
            pct = pct * 100 if pct <= 1 else pct
            pct = max(0, min(100, pct))

            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': "%"},
                title={'text': f"{METRIC_LABELS.get(metric, metric)}分位"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': NEUTRAL_COLOR},
                    'steps': [
                        {'range': [0, 30], 'color': '#2ca02c'},
                        {'range': [30, 70], 'color': '#ff9f43'},
                        {'range': [70, 100], 'color': '#d62728'},
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.85,
                        'value': pct,
                    }
                }
            ), row=1, col=1)
            col = 2

        # ROE趋势
        if has_roe:
            roe_col = col if has_valuation else 1
            roe_row = 1
            self._add_roe_traces(fig, roe_data, roe_row, roe_col)
            col += 1

        # 行业雷达
        if has_industry:
            radar_row = rows  # 最后一行
            self._add_radar_traces(fig, industry_data, radar_row)

        fig.update_layout(
            title="基本面综合仪表盘",
            height=height,
            template=CHART_THEME,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1,
                        xanchor="center", x=0.5),
        )

        return fig

    # ---- 辅助方法 ----

    @staticmethod
    def _extract_valuation(valuation_data: Dict):
        """从valuation_data中提取第一个估值指标"""
        metric = list(valuation_data.keys())[0]
        vinfo = valuation_data[metric]
        if isinstance(vinfo, dict):
            current = vinfo.get('current', vinfo.get('value', 0))
            pct = vinfo.get('percentile', vinfo.get('pct', 0.5))
        else:
            current = float(vinfo)
            pct = 0.5
        return metric, float(current), float(pct)

    @staticmethod
    def _build_dashboard_specs(has_valuation: bool, has_roe: bool,
                               has_industry: bool):
        """构建综合仪表盘的specs布局"""
        if has_valuation and has_roe and has_industry:
            return [
                [{"type": "indicator"}, {"type": "xy"}],
                [{"type": "polar", "colspan": 2}, None],
            ], 2
        elif has_valuation and has_roe:
            return [[{"type": "indicator"}, {"type": "xy"}]], 1
        elif has_valuation and has_industry:
            return [[{"type": "indicator"}, {"type": "polar"}]], 1
        else:  # roe + industry
            return [[{"type": "xy"}, {"type": "polar"}]], 1

    @staticmethod
    def _add_roe_traces(fig: go.Figure, roe_data: pd.DataFrame, row: int, col: int):
        """向fig添加ROE趋势trace"""
        roe_df = roe_data.copy()
        date_col = 'date' if 'date' in roe_df.columns else (
            'period' if 'period' in roe_df.columns else roe_df.columns[0])
        if date_col == 'date':
            dates = pd.to_datetime(roe_df[date_col])
        else:
            dates = roe_df[date_col].astype(str)

        if 'roe' in roe_df.columns:
            colors = [UP_COLOR if v >= 15 else (NEUTRAL_COLOR if v >= 0 else DOWN_COLOR)
                      for v in roe_df['roe']]
            fig.add_trace(go.Bar(
                x=dates, y=roe_df['roe'],
                name='ROE(%)', marker_color=colors,
            ), row=row, col=col)

        if 'gross_margin' in roe_df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=roe_df['gross_margin'],
                mode='lines+markers', name='毛利率(%)',
                line=dict(color='#ff7f0e', width=2),
            ), row=row, col=col)

        if 'net_margin' in roe_df.columns:
            fig.add_trace(go.Scatter(
                x=dates, y=roe_df['net_margin'],
                mode='lines+markers', name='净利率(%)',
                line=dict(color='#9467bd', width=2),
            ), row=row, col=col)

    @staticmethod
    def _add_radar_traces(fig: go.Figure, industry_data: Dict, row: int):
        """向fig添加行业雷达trace"""
        stock_vals = industry_data['stock']
        industry_vals = industry_data['industry_avg']
        metrics_list = [m for m in stock_vals.keys()
                        if m in industry_vals
                        and stock_vals[m] is not None
                        and industry_vals[m] is not None]

        if not metrics_list:
            return

        # 归一化
        stock_norm = []
        ind_norm = []
        for m in metrics_list:
            sv = float(stock_vals[m])
            iv = float(industry_vals[m])
            max_val = max(abs(sv), abs(iv), 1e-6)
            stock_norm.append(sv / max_val * 100)
            ind_norm.append(iv / max_val * 100)

        labels = [METRIC_LABELS.get(m, m) for m in metrics_list]

        fig.add_trace(go.Scatterpolar(
            r=stock_norm + [stock_norm[0]],
            theta=labels + [labels[0]],
            fill='toself', name='个股',
            line=dict(color=UP_COLOR, width=2),
            fillcolor='rgba(214,39,40,0.2)',
        ), row=row, col=1)

        fig.add_trace(go.Scatterpolar(
            r=ind_norm + [ind_norm[0]],
            theta=labels + [labels[0]],
            fill='toself', name='行业均值',
            line=dict(color=NEUTRAL_COLOR, width=2),
            fillcolor='rgba(31,119,180,0.2)',
        ), row=row, col=1)
