import os
import quantstats as qs
import pandas as pd
from typing import Optional


class QuantStatsReport:
    """
    QuantStats 报告生成器
    """

    def __init__(self, config):
        self.config = config
        qs.extend_pandas()

    def generate_report(
        self,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        output_path: Optional[str] = None,
        title: str = "Strategy Performance Report"
    ) -> str:
        """
        生成 quantstats HTML 报告
        
        Args:
            returns: 策略收益率序列
            benchmark: 基准收益率序列
            output_path: 输出文件路径
            title: 报告标题
        
        Returns:
            报告文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.config.REPORT_DIR, "quantstats_report.html")
        
        if benchmark is not None:
            qs.reports.html(
                returns,
                benchmark=benchmark,
                output=output_path,
                title=title,
                download_filename=os.path.basename(output_path)
            )
        else:
            qs.reports.html(
                returns,
                output=output_path,
                title=title,
                download_filename=os.path.basename(output_path)
            )
        
        return output_path

    def generate_key_metrics(self, returns: pd.Series, benchmark: Optional[pd.Series] = None) -> dict:
        """
        生成关键绩效指标
        
        Args:
            returns: 策略收益率序列
            benchmark: 基准收益率序列
        
        Returns:
            指标字典
        """
        metrics = {}
        
        metrics["cagr"] = qs.stats.cagr(returns)
        metrics["volatility"] = qs.stats.volatility(returns)
        metrics["sharpe"] = qs.stats.sharpe(returns, rf=self.config.RISK_FREE_RATE)
        metrics["sortino"] = qs.stats.sortino(returns, rf=self.config.RISK_FREE_RATE)
        metrics["max_drawdown"] = qs.stats.max_drawdown(returns)
        metrics["win_rate"] = qs.stats.win_rate(returns)
        metrics["profit_factor"] = qs.stats.profit_factor(returns)
        metrics["calmar"] = qs.stats.calmar(returns)
        
        if benchmark is not None:
            metrics["information_ratio"] = qs.stats.information_ratio(returns, benchmark)
            metrics["beta"] = qs.stats.greeks(returns, benchmark)["beta"]
            metrics["alpha"] = qs.stats.greeks(returns, benchmark)["alpha"]
        
        return metrics
