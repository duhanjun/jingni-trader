"""
报告引擎专属配置
"""
import os

REPORT_DIR = os.environ.get("REPORT_DIR", "./quant_workspace/reports")
REPORT_TITLE = os.environ.get("REPORT_TITLE", "A股量化策略绩效报告")
REPORT_FORMAT = os.environ.get("REPORT_FORMAT", "html")
INDUSTRY_STANDARD = os.environ.get("INDUSTRY_STANDARD", "sw")
BENCHMARK = os.environ.get("BENCHMARK", "000300.SH")
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", 0.03))
INCLUDE_TEARSHEET = os.environ.get("INCLUDE_TEARSHEET", "true").lower() == "true"
INCLUDE_HEATMAP = os.environ.get("INCLUDE_HEATMAP", "true").lower() == "true"
INCLUDE_ATTRIBUTION = os.environ.get("INCLUDE_ATTRIBUTION", "true").lower() == "true"
INCLUDE_TRADES = os.environ.get("INCLUDE_TRADES", "true").lower() == "true"
CHART_THEME = os.environ.get("CHART_THEME", "plotly_white")

os.makedirs(REPORT_DIR, exist_ok=True)