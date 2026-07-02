import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional


class EquityCurvePlotter:
    """
    动态净值曲线生成器（Plotly）
    """

    def __init__(self, config):
        self.config = config

    def plot_interactive_equity_curve(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        绘制交互式净值曲线
        
        Args:
            portfolio_returns: 组合收益率序列
            benchmark_returns: 基准收益率序列
            output_path: 输出路径
        
        Returns:
            图表文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "equity_curve.html")
        
        portfolio_cum = (1 + portfolio_returns).cumprod()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_cum.index,
            y=portfolio_cum.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="#1f77b4", width=2)
        ))
        
        if benchmark_returns is not None:
            benchmark_cum = (1 + benchmark_returns).cumprod()
            fig.add_trace(go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values,
                mode="lines",
                name="Benchmark",
                line=dict(color="#ff7f0e", width=2, dash="dash")
            ))
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            template="plotly_white"
        )
        
        fig.write_html(output_path)
        
        return output_path

    def plot_drawdown(self, returns: pd.Series, output_path: Optional[str] = None) -> str:
        """
        绘制回撤曲线
        
        Args:
            returns: 收益率序列
            output_path: 输出路径
        
        Returns:
            图表文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "drawdown.html")
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            fill="tonexty",
            line=dict(color="#d62728", width=2)
        ))
        
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            template="plotly_white"
        )
        
        fig.write_html(output_path)
        
        return output_path
