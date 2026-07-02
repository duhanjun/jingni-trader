import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def convert_to_serializable(obj):
    """转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

from config import Config, get_config
from quantstats_report import QuantStatsReport
from rolling_sharpe import RollingSharpeHeatmap
from style_exposure import StyleExposureAnalyzer
from industry_exposure import IndustryExposureAnalyzer
from brinson_attribution import BrinsonAttribution
from equity_curve import EquityCurvePlotter
from factor_decay import FactorDecayAnalyzer
from overfitting_warning import OverfittingWarning


class ReportsEngine:
    """
    A股量化策略报告引擎主类
    """

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        
        self.quantstats = QuantStatsReport(config)
        self.rolling_sharpe = RollingSharpeHeatmap(config)
        self.style_exposure = StyleExposureAnalyzer(config)
        self.industry_exposure = IndustryExposureAnalyzer(config)
        self.brinson = BrinsonAttribution(config)
        self.equity_curve = EquityCurvePlotter(config)
        self.factor_decay = FactorDecayAnalyzer(config)
        self.overfitting = OverfittingWarning(config)

    def generate_full_report(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_exposures: Optional[pd.DataFrame] = None,
        industry_weights: Optional[pd.DataFrame] = None,
        is_returns: Optional[pd.Series] = None,
        oos_returns: Optional[pd.Series] = None,
        ic_series: Optional[pd.Series] = None,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
        asset_returns: Optional[pd.DataFrame] = None,
        report_name: str = "strategy_report"
    ) -> Dict[str, Any]:
        """
        生成完整报告
        
        Args:
            portfolio_returns: 组合收益率
            benchmark_returns: 基准收益率
            factor_exposures: 因子暴露
            industry_weights: 行业权重
            is_returns: 样本内收益率
            oos_returns: 样本外收益率
            ic_series: IC序列
            portfolio_weights: 组合权重
            benchmark_weights: 基准权重
            asset_returns: 资产收益率
            report_name: 报告名称
        
        Returns:
            完整报告字典
        """
        report_dir = os.path.join(self.config.REPORT_DIR, report_name)
        os.makedirs(report_dir, exist_ok=True)
        
        report = {
            "report_name": report_name,
            "generated_at": pd.Timestamp.now().isoformat(),
            "sections": {}
        }
        
        # QuantStats报告
        report["sections"]["quantstats"] = {
            "report_path": self.quantstats.generate_report(
                portfolio_returns,
                benchmark=benchmark_returns,
                output_path=os.path.join(report_dir, "quantstats.html")
            ),
            "key_metrics": self.quantstats.generate_key_metrics(
                portfolio_returns,
                benchmark=benchmark_returns
            )
        }
        
        # 滚动夏普和月度热力图
        report["sections"]["rolling_sharpe"] = {
            "chart_path": self.rolling_sharpe.plot_rolling_sharpe(
                portfolio_returns,
                output_path=os.path.join(report_dir, "rolling_sharpe.png")
            )
        }
        report["sections"]["monthly_heatmap"] = {
            "chart_path": self.rolling_sharpe.plot_monthly_heatmap(
                portfolio_returns,
                output_path=os.path.join(report_dir, "monthly_heatmap.png")
            )
        }
        
        # 净值曲线
        report["sections"]["equity_curve"] = {
            "chart_path": self.equity_curve.plot_interactive_equity_curve(
                portfolio_returns,
                benchmark_returns=benchmark_returns,
                output_path=os.path.join(report_dir, "equity_curve.html")
            )
        }
        report["sections"]["drawdown"] = {
            "chart_path": self.equity_curve.plot_drawdown(
                portfolio_returns,
                output_path=os.path.join(report_dir, "drawdown.html")
            )
        }
        
        # 风格暴露（如果提供）
        if factor_exposures is not None:
            report["sections"]["style_exposure"] = self.style_exposure.analyze_style_exposure(
                factor_exposures,
                output_path=os.path.join(report_dir, "style_exposure.png")
            )
        
        # 行业暴露（如果提供）
        if industry_weights is not None:
            report["sections"]["industry_exposure"] = self.industry_exposure.analyze_industry_exposure(
                industry_weights,
                output_path=os.path.join(report_dir, "industry_exposure.png")
            )
        
        # Brinson归因（如果提供）
        if (portfolio_weights is not None and benchmark_weights is not None and 
            asset_returns is not None):
            report["sections"]["brinson_attribution"] = self.brinson.analyze(
                portfolio_weights,
                benchmark_weights,
                portfolio_returns,
                benchmark_returns if benchmark_returns is not None else pd.Series(),
                asset_returns,
                output_path=os.path.join(report_dir, "brinson_attribution.png")
            )
        
        # 因子衰减（如果提供）
        if ic_series is not None:
            report["sections"]["factor_decay"] = self.factor_decay.analyze_decay(
                ic_series,
                output_path=os.path.join(report_dir, "factor_decay.png")
            )
        
        # 过拟合检查（如果提供）
        if is_returns is not None and oos_returns is not None:
            report["sections"]["overfitting_check"] = self.overfitting.check_overfitting(
                is_returns,
                oos_returns
            )
        
        # 保存报告JSON
        self.save_report(report, os.path.join(report_dir, f"{report_name}.json"))
        
        return report

    def save_report(self, report: Dict[str, Any], output_path: str):
        """
        保存报告到文件
        
        Args:
            report: 报告字典
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        serializable_report = convert_to_serializable(report)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
