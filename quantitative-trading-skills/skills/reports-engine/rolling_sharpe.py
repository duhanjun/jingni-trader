import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class RollingSharpeHeatmap:
    """
    滚动夏普与月度热力图生成器
    """

    def __init__(self, config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)

    def plot_rolling_sharpe(
        self,
        returns: pd.Series,
        output_path: Optional[str] = None
    ) -> str:
        """
        绘制滚动夏普比率
        
        Args:
            returns: 收益率序列
            output_path: 输出路径
        
        Returns:
            图表路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "rolling_sharpe.png")
        
        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
        
        rolling_sharpe = returns.rolling(window=self.config.ROLLING_WINDOW).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() != 0 else 0
        )
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label="Rolling Sharpe", color="#1f77b4", linewidth=2)
        ax.axhline(y=rolling_sharpe.mean(), color="#ff7f0e", linestyle="--", label=f"Mean: {rolling_sharpe.mean():.2f}")
        ax.axhline(y=0, color="#333", linestyle="-", alpha=0.5)
        
        ax.set_title(f"Rolling Sharpe Ratio (Window: {self.config.ROLLING_WINDOW} days)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path

    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        output_path: Optional[str] = None
    ) -> str:
        """
        绘制月度收益热力图
        
        Args:
            returns: 收益率序列
            output_path: 输出路径
        
        Returns:
            图表路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "monthly_heatmap.png")
        
        df = returns.to_frame("return")
        df["year"] = df.index.year
        df["month"] = df.index.month
        
        monthly_returns = df.groupby(["year", "month"])["return"].sum().unstack()
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_returns.columns = month_names
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sns.heatmap(
            monthly_returns * 100,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            cbar_kws={"label": "Return (%)"}
        )
        
        ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        return output_path
