from typing import Dict, Any, Callable, Optional
import pandas as pd
import os
import sys

from config import Config, get_config
from base import BaseBacktestEngine
from adapters import BacktraderAdapter, gmAdapter
from rules import AShareTradingRules, FeeCalculator, SuspensionHandler
from performance import PerformanceMetrics
from overfitting import WalkForwardAnalyzer
from report import ReportGenerator
from visualization import BacktestPlotter


class BacktestEngine:
    """
    回测引擎主类
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

        self._adapter = self._create_adapter()
        self.rules = AShareTradingRules(self.config)
        self.suspension_handler = SuspensionHandler()
        self.metrics = PerformanceMetrics()
        self.walk_forward = WalkForwardAnalyzer(self.config)
        self.report_gen = ReportGenerator(self.config)
        self.plotter = BacktestPlotter(self.config)

    def _create_adapter(self) -> BaseBacktestEngine:
        """
        创建回测适配器

        Returns:
            回测适配器
        """
        engine_type = self.config.BACKTEST_ENGINE.lower()

        if engine_type == "backtrader":
            return BacktraderAdapter(self.config)
        elif engine_type == "gm":
            return gmAdapter(self.config)
        else:
            raise ValueError(f"Unsupported backtest engine: {engine_type}")

    def run(
        self,
        strategy: Callable,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        运行回测

        Args:
            strategy: 策略函数
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金

        Returns:
            回测结果
        """
        return self._adapter.run(
            strategy,
            start_date,
            end_date,
            initial_capital,
            **kwargs,
        )

    def get_trades(self) -> pd.DataFrame:
        """
        获取交易记录

        Returns:
            交易记录
        """
        return self._adapter.get_trades()

    def get_equity_curve(self) -> pd.Series:
        """
        获取资金曲线

        Returns:
            资金曲线
        """
        return self._adapter.get_equity_curve()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取绩效指标

        Returns:
            绩效指标
        """
        return self._adapter.get_performance_metrics()

    def generate_report(
        self,
        backtest_results: Dict[str, Any],
        walk_forward_results: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        生成回测报告

        Args:
            backtest_results: 回测结果
            walk_forward_results: Walk-Forward 分析结果

        Returns:
            报告
        """
        return self.report_gen.generate(backtest_results, walk_forward_results)

    def save_report(
        self,
        report: Dict[str, Any],
        filename: str,
        output_dir: str = None,
        formats: list = ["markdown", "json"],
    ) -> Dict[str, str]:
        """
        保存报告

        Args:
            report: 报告
            filename: 文件名
            output_dir: 输出目录
            formats: 输出格式列表

        Returns:
            文件路径字典
        """
        paths = {}

        if "markdown" in formats:
            paths["markdown"] = self.report_gen.save_markdown(report, filename, output_dir)

        if "json" in formats:
            paths["json"] = self.report_gen.save_json(report, filename, output_dir)

        return paths

    def plot_equity_curve(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ):
        """
        绘制资金曲线

        Args:
            results: 回测结果
            save_path: 保存路径
        """
        self.plotter.plot_equity_curve(
            results.get("equity_curve", pd.Series()),
            results.get("benchmark_curve", None),
            save_path,
        )

    def plot_drawdown(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ):
        """
        绘制回撤曲线

        Args:
            results: 回测结果
            save_path: 保存路径
        """
        self.plotter.plot_drawdown(
            results.get("equity_curve", pd.Series()),
            save_path,
        )

    def analyze_walk_forward(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000000,
    ) -> Dict[str, Any]:
        """
        执行 Walk-Forward 分析

        Args:
            strategy: 策略函数
            data: 行情数据
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金

        Returns:
            Walk-Forward 分析结果
        """
        return self.walk_forward.analyze(
            strategy,
            data,
            start_date,
            end_date,
            initial_capital,
        )
