from typing import Dict, Any, Optional
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class BacktestPlotter:
    """
    回测可视化工具
    """

    def __init__(self, config: Config):
        self.config = config

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制资金曲线

        Args:
            equity_curve: 资金曲线
            benchmark_curve: 基准曲线
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")

            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, equity_curve.values, label="Strategy", linewidth=2)

            if benchmark_curve is not None:
                plt.plot(benchmark_curve.index, benchmark_curve.values, label="Benchmark", linewidth=2, alpha=0.7)

            plt.title("Equity Curve", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Value", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Equity curve saved to: {save_path}")

            plt.close()

        except ImportError:
            print("matplotlib is not available")

    def plot_drawdown(
        self,
        equity_curve: pd.Series,
        save_path: Optional[str] = None,
    ):
        """
        绘制回撤曲线

        Args:
            equity_curve: 资金曲线
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")

            cumulative_max = equity_curve.cummax()
            drawdown = (equity_curve - cumulative_max) / cumulative_max

            plt.figure(figsize=(12, 6))
            plt.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
            plt.plot(drawdown.index, drawdown.values, label="Drawdown", linewidth=2, color="red")

            plt.title("Drawdown", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Drawdown", fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Drawdown plot saved to: {save_path}")

            plt.close()

        except ImportError:
            print("matplotlib is not available")

    def plot_interactive(
        self,
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制交互式图表

        Args:
            equity_curve: 资金曲线
            benchmark_curve: 基准曲线
            save_path: 保存路径
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Strategy",
                line=dict(width=2),
            ))

            if benchmark_curve is not None:
                fig.add_trace(go.Scatter(
                    x=benchmark_curve.index,
                    y=benchmark_curve.values,
                    mode="lines",
                    name="Benchmark",
                    line=dict(width=2),
                ))

            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode="x",
            )

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.write_html(save_path)
                print(f"Interactive plot saved to: {save_path}")

        except ImportError:
            print("plotly is not available")
