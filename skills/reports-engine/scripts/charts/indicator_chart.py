"""
技术指标面板图生成模块
生成MACD、RSI、KDJ等技术指标子图
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict

try:
    from scripts.config import CHART_THEME
except ImportError:
    CHART_THEME = "plotly_white"

# A股颜色惯例
UP_COLOR = '#d62728'    # 红
DOWN_COLOR = '#2ca02c'  # 绿


class IndicatorChartGenerator:
    """技术指标图表生成器"""

    def generate_macd_panel(self, df: pd.DataFrame, height: int = 400) -> go.Figure:
        """生成MACD面板图（DIF线、DEA线、柱状图）"""
        if df is None or df.empty or 'close' not in df.columns:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        close = df['close']

        # MACD计算：DIF=EMA12-EMA26, DEA=EMA(DIF,9), MACD=2*(DIF-DEA)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        # 柱状图颜色：红正绿负
        hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in macd_hist]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
            subplot_titles=("DIF / DEA", "MACD柱状图"),
        )

        # DIF线
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=dif,
                mode='lines', name='DIF',
                line=dict(color='#1f77b4', width=1.5),
            ),
            row=1, col=1
        )

        # DEA线
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=dea,
                mode='lines', name='DEA',
                line=dict(color='#ff7f0e', width=1.5),
            ),
            row=1, col=1
        )

        # MACD柱状图
        fig.add_trace(
            go.Bar(
                x=df['date'], y=macd_hist,
                marker_color=hist_colors,
                name='MACD',
                showlegend=False,
            ),
            row=2, col=1
        )

        # 零轴
        fig.add_hline(y=0, line=dict(color='gray', width=0.5), row=2, col=1)

        fig.update_layout(
            title="MACD指标",
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_yaxes(title_text="DIF/DEA", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        return fig

    def generate_rsi_panel(self, df: pd.DataFrame, period: int = 14, height: int = 300) -> go.Figure:
        """生成RSI面板图，含30/70超买超卖线"""
        if df is None or df.empty or 'close' not in df.columns:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        close = df['close']

        # RSI计算（Wilder平滑法）
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50)

        fig = go.Figure()

        # RSI线
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=rsi,
                mode='lines', name=f'RSI{period}',
                line=dict(color='#1f77b4', width=1.5),
            )
        )

        # 超买线(70)
        fig.add_hline(
            y=70,
            line=dict(color=UP_COLOR, width=1, dash='dash'),
            annotation_text="超买 70",
            annotation_position="top left",
        )

        # 超卖线(30)
        fig.add_hline(
            y=30,
            line=dict(color=DOWN_COLOR, width=1, dash='dash'),
            annotation_text="超卖 30",
            annotation_position="bottom left",
        )

        # 中轴线(50)
        fig.add_hline(y=50, line=dict(color='gray', width=0.5, dash='dot'))

        # 超买超卖区域填充
        fig.add_hrect(y0=70, y1=100, fillcolor=UP_COLOR, opacity=0.05, line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor=DOWN_COLOR, opacity=0.05, line_width=0)

        fig.update_layout(
            title=f"RSI指标 (周期={period})",
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            yaxis=dict(title="RSI", range=[0, 100]),
            xaxis=dict(title="日期"),
        )

        return fig

    def generate_kdj_panel(self, df: pd.DataFrame, height: int = 300) -> go.Figure:
        """生成KDJ面板图"""
        if df is None or df.empty or not all(c in df.columns for c in ['high', 'low', 'close']):
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        n = 9
        low_min = df['low'].rolling(window=n, min_periods=1).min()
        high_max = df['high'].rolling(window=n, min_periods=1).max()

        # RSV计算
        rsv = (df['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        rsv = rsv.fillna(50)

        # KDJ平滑
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=df['date'], y=k, mode='lines', name='K',
                       line=dict(color='#1f77b4', width=1.5))
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=d, mode='lines', name='D',
                       line=dict(color='#ff7f0e', width=1.5))
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=j, mode='lines', name='J',
                       line=dict(color='#9467bd', width=1.5))
        )

        # 超买超卖参考线
        fig.add_hline(y=80, line=dict(color=UP_COLOR, width=0.8, dash='dash'),
                      annotation_text="超买 80", annotation_position="top left")
        fig.add_hline(y=20, line=dict(color=DOWN_COLOR, width=0.8, dash='dash'),
                      annotation_text="超卖 20", annotation_position="bottom left")
        fig.add_hline(y=50, line=dict(color='gray', width=0.5, dash='dot'))

        fig.update_layout(
            title=f"KDJ指标 (N={n})",
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            yaxis=dict(title="KDJ", range=[0, 100]),
            xaxis=dict(title="日期"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    def generate_combined_panel(self, df: pd.DataFrame, height: int = 800) -> go.Figure:
        """
        生成组合技术指标面板（3个子图垂直排列）
        子图1: MACD
        子图2: RSI
        子图3: KDJ 或 布林带
        """
        if df is None or df.empty or 'close' not in df.columns:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        close = df['close']

        # MACD计算
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)
        hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in macd_hist]

        # RSI计算
        period = 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50)

        # KDJ计算
        n = 9
        low_min = df['low'].rolling(window=n, min_periods=1).min() if 'low' in df.columns else close
        high_max = df['high'].rolling(window=n, min_periods=1).max() if 'high' in df.columns else close
        rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        rsv = rsv.fillna(50)
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=("MACD", "RSI(14)", "KDJ(9,3,3)"),
        )

        # MACD子图
        fig.add_trace(
            go.Scatter(x=df['date'], y=dif, mode='lines', name='DIF',
                       line=dict(color='#1f77b4', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=dea, mode='lines', name='DEA',
                       line=dict(color='#ff7f0e', width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=df['date'], y=macd_hist, marker_color=hist_colors,
                   name='MACD', showlegend=False),
            row=1, col=1
        )
        fig.add_hline(y=0, line=dict(color='gray', width=0.5), row=1, col=1)

        # RSI子图
        fig.add_trace(
            go.Scatter(x=df['date'], y=rsi, mode='lines', name='RSI',
                       line=dict(color='#1f77b4', width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=70, line=dict(color=UP_COLOR, width=1, dash='dash'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=DOWN_COLOR, width=1, dash='dash'), row=2, col=1)
        fig.add_hline(y=50, line=dict(color='gray', width=0.5, dash='dot'), row=2, col=1)

        # KDJ子图
        fig.add_trace(
            go.Scatter(x=df['date'], y=k, mode='lines', name='K',
                       line=dict(color='#1f77b4', width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=d, mode='lines', name='D',
                       line=dict(color='#ff7f0e', width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=j, mode='lines', name='J',
                       line=dict(color='#9467bd', width=1.5)),
            row=3, col=1
        )
        fig.add_hline(y=80, line=dict(color=UP_COLOR, width=0.8, dash='dash'), row=3, col=1)
        fig.add_hline(y=20, line=dict(color=DOWN_COLOR, width=0.8, dash='dash'), row=3, col=1)

        fig.update_layout(
            title="技术指标组合面板",
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig.update_yaxes(title_text="MACD", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="KDJ", range=[0, 100], row=3, col=1)
        fig.update_xaxes(title_text="日期", row=3, col=1)

        return fig
