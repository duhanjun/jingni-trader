import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class IndustryExposureAnalyzer:
    """
    申万一级行业暴露分析器
    """

    def __init__(self, config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)

    def analyze_industry_exposure(
        self,
        industry_weights: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> dict:
        """
        分析行业暴露
        
        Args:
            industry_weights: 行业权重 DataFrame，列是申万一级行业名称
            output_path: 输出路径
        
        Returns:
            分析结果字典
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "industry_exposure.png")
        
        avg_weights = industry_weights.mean().sort_values(ascending=False)
        
        result = {
            "average_weights": avg_weights.to_dict(),
            "top_industries": avg_weights.head(5).to_dict(),
            "bottom_industries": avg_weights.tail(5).to_dict()
        }
        
        self._plot_industry_exposure(industry_weights, avg_weights, output_path)
        
        return result

    def _plot_industry_exposure(
        self,
        industry_weights: pd.DataFrame,
        avg_weights: pd.Series,
        output_path: str
    ):
        """
        绘制行业暴露图
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # 平均权重条形图
        colors = plt.cm.viridis(np.linspace(0, 1, len(avg_weights)))
        avg_weights.plot(kind="bar", ax=axes[0], color=colors)
        axes[0].set_title("Average Industry Weights", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Industry")
        axes[0].set_ylabel("Weight")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, alpha=0.3, axis="y")
        
        # 热力图
        if len(industry_weights) > 1:
            sns.heatmap(
                industry_weights.T,
                cmap="YlGnBu",
                ax=axes[1],
                cbar_kws={"label": "Weight"}
            )
            axes[1].set_title("Industry Weights Over Time", fontsize=14, fontweight="bold")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel("Industry")
        else:
            axes[1].text(0.5, 0.5, "Need multiple dates for heatmap", ha="center", va="center", fontsize=14)
            axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
