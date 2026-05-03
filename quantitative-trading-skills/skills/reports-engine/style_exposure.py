import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class StyleExposureAnalyzer:
    """
    A股风格暴露分析器（大盘/小盘/成长/价值）
    """

    def __init__(self, config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)

    def analyze_style_exposure(
        self,
        factor_exposures: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> dict:
        """
        分析风格暴露
        
        Args:
            factor_exposures: 因子暴露 DataFrame，列包含 ['size', 'value', 'growth']
            output_path: 输出路径
        
        Returns:
            分析结果字典
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "style_exposure.png")
        
        result = {
            "exposure_summary": factor_exposures.mean().to_dict(),
            "exposure_std": factor_exposures.std().to_dict()
        }
        
        self._plot_style_exposure(factor_exposures, output_path)
        
        return result

    def _plot_style_exposure(
        self,
        factor_exposures: pd.DataFrame,
        output_path: str
    ):
        """
        绘制风格暴露图
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 时间序列图
        for i, col in enumerate(factor_exposures.columns):
            axes[0].plot(factor_exposures.index, factor_exposures[col], 
                        label=col, linewidth=2, alpha=0.8)
        
        axes[0].axhline(y=0, color="#333", linestyle="-", alpha=0.5)
        axes[0].set_title("Style Factor Exposures Over Time", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Exposure")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图
        factor_exposures.boxplot(ax=axes[1])
        axes[1].axhline(y=0, color="#333", linestyle="-", alpha=0.5)
        axes[1].set_title("Style Factor Exposures Distribution", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Exposure")
        axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
