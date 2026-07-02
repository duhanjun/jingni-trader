import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class FactorDecayAnalyzer:
    """
    因子衰减分析器
    """

    def __init__(self, config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)

    def analyze_decay(
        self,
        ic_series: pd.Series,
        output_path: Optional[str] = None
    ) -> dict:
        """
        分析因子衰减
        
        Args:
            ic_series: IC序列
            output_path: 输出路径
        
        Returns:
            分析结果字典
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "factor_decay.png")
        
        decay_curve = []
        for lag in range(1, self.config.DECAY_PERIODS + 1):
            if lag < len(ic_series):
                decay_curve.append(ic_series.autocorr(lag))
            else:
                decay_curve.append(np.nan)
        
        decay_df = pd.DataFrame({
            "lag": list(range(1, self.config.DECAY_PERIODS + 1)),
            "autocorrelation": decay_curve
        })
        
        result = {
            "decay_curve": decay_df.to_dict(orient="list"),
            "half_life": self._calculate_half_life(decay_curve)
        }
        
        self._plot_decay(decay_df, output_path)
        
        return result

    def _calculate_half_life(self, decay_curve: list) -> Optional[int]:
        """
        计算半衰期
        """
        initial_value = decay_curve[0] if decay_curve and not np.isnan(decay_curve[0]) else 1
        half_value = initial_value / 2
        
        for i, val in enumerate(decay_curve):
            if not np.isnan(val) and val <= half_value:
                return i + 1
        
        return None

    def _plot_decay(self, decay_df: pd.DataFrame, output_path: str):
        """
        绘制衰减曲线
        """
        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
        
        ax.plot(decay_df["lag"], decay_df["autocorrelation"], 
               marker="o", linewidth=2, markersize=6, color="#1f77b4")
        ax.axhline(y=0, color="#333", linestyle="-", alpha=0.5)
        
        ax.set_title("Factor IC Decay", fontsize=14, fontweight="bold")
        ax.set_xlabel("Lag (Periods)")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
