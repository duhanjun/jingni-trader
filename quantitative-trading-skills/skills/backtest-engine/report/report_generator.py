from typing import Dict, Any
import pandas as pd
import json
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class ReportGenerator:
    """
    回测报告生成器
    """

    def __init__(self, config: Config):
        self.config = config

    def generate(
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
            报告 dict
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "backtest": backtest_results.get("performance_metrics", {}),
            "walk_forward": walk_forward_results if walk_forward_results else None,
        }

        return report

    def save_markdown(
        self,
        report: Dict[str, Any],
        filename: str,
        output_dir: str = None,
    ) -> str:
        """
        保存 Markdown 报告

        Args:
            report: 报告
            filename: 文件名
            output_dir: 输出目录

        Returns:
            文件路径
        """
        if output_dir is None:
            output_dir = self.config.REPORT_DIR

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.md")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._to_markdown(report))

        return filepath

    def save_json(
        self,
        report: Dict[str, Any],
        filename: str,
        output_dir: str = None,
    ) -> str:
        """
        保存 JSON 报告

        Args:
            report: 报告
            filename: 文件名
            output_dir: 输出目录

        Returns:
            文件路径
        """
        if output_dir is None:
            output_dir = self.config.REPORT_DIR

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{filename}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def _to_markdown(self, report: Dict[str, Any]) -> str:
        """
        转换为 Markdown

        Args:
            report: 报告

        Returns:
            Markdown 字符串
        """
        lines = []
        lines.append("# 回测报告")
        lines.append(f"\n生成时间: {report.get('generated_at', 'N/A')}")

        backtest = report.get("backtest", {})
        if backtest:
            lines.append("\n## 回测绩效指标")
            lines.append("\n| 指标 | 值 |")
            lines.append("|------|-----|")
            for key, value in backtest.items():
                if isinstance(value, float):
                    if "return" in key.lower() or "rate" in key.lower() or "ratio" in key.lower():
                        lines.append(f"| {key} | {value:.2%} |")
                    else:
                        lines.append(f"| {key} | {value:.4f} |")
                else:
                    lines.append(f"| {key} | {value} |")

        walk_forward = report.get("walk_forward")
        if walk_forward:
            lines.append("\n## Walk-Forward 分析")
            lines.append(f"\n- 总周期数: {walk_forward.get('total_periods', 0)}")
            lines.append(f"- 平均训练收益: {walk_forward.get('avg_train_return', 0):.2%}")
            lines.append(f"- 平均测试收益: {walk_forward.get('avg_test_return', 0):.2%}")
            lines.append(f"- 性能衰减: {walk_forward.get('performance_degradation', 0):.2%}")
            lines.append(f"- 是否过拟合: {'是' if walk_forward.get('is_overfitted', False) else '否'}")

        return "\n".join(lines)
