import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class BrinsonAttribution:
    """
    Brinson归因分析器
    """

    def __init__(self, config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)

    def analyze(
        self,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        asset_returns: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> dict:
        """
        Brinson归因分析
        
        Args:
            portfolio_weights: 组合权重
            benchmark_weights: 基准权重
            portfolio_returns: 组合收益率
            benchmark_returns: 基准收益率
            asset_returns: 资产收益率
            output_path: 输出路径
        
        Returns:
            归因结果字典
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "brinson_attribution.png")
        
        allocation_effect, selection_effect, interaction_effect = [], [], []
        
        common_dates = portfolio_weights.index.intersection(benchmark_weights.index)
        
        for date in common_dates:
            pw = portfolio_weights.loc[date]
            bw = benchmark_weights.loc[date]
            ar = asset_returns.loc[date] if date in asset_returns.index else pd.Series(0, index=pw.index)
            
            common_assets = pw.index.intersection(bw.index)
            pw = pw.loc[common_assets]
            bw = bw.loc[common_assets]
            ar = ar.loc[common_assets]
            
            bm_return = (bw * ar).sum()
            
            alloc = ((pw - bw) * ar).sum()
            sel = (bw * (ar - bm_return)).sum()
            inter = ((pw - bw) * (ar - bm_return)).sum()
            
            allocation_effect.append(alloc)
            selection_effect.append(sel)
            interaction_effect.append(inter)
        
        result_df = pd.DataFrame({
            "allocation": allocation_effect,
            "selection": selection_effect,
            "interaction": interaction_effect
        }, index=common_dates)
        
        cumulative = result_df.cumsum()
        
        result = {
            "daily_attribution": result_df.to_dict(orient="index"),
            "cumulative_attribution": cumulative.iloc[-1].to_dict(),
            "total_excess_return": cumulative.iloc[-1].sum()
        }
        
        self._plot_attribution(cumulative, output_path)
        
        return result

    def _plot_attribution(self, cumulative: pd.DataFrame, output_path: str):
        """
        绘制归因结果图
        """
        fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
        
        cumulative.plot(ax=ax, linewidth=2)
        ax.set_title("Brinson Attribution - Cumulative Effects", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.legend(["Allocation Effect", "Selection Effect", "Interaction Effect"])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
