"""
基本面深度分析报告生成器（独立报告，借鉴 TradingAgents fundamentals_report 设计）

10 个章节：
1. 报告概览（基本面评分仪表盘）
2. 公司概况与业务分析
3. 行业分析与景气度
4. 财务报表分析（利润/资产负债/现金流）
5. 盈利能力分析（ROE/毛利率/净利率）
6. 成长性分析（营收/利润增速）
7. 估值分析（PE/PB/PS 分位）
8. 股东结构与资本运作（A股特色：十大股东/解禁/回购）
9. 基本面深度解读（LLM 分析师章节）
10. 风险提示

与 stock_analysis_report.py 的关系：
- 复用其估值/盈利/成长评分方法
- 独立渲染基本面相关章节，不包含技术面章节
"""
import os
import html as _html
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime
from jinja2 import Template

# 复用合并报告的基类方法（评分/格式化/基本面提取）
from .stock_analysis_report import StockAnalysisReportGenerator

logger = logging.getLogger("fundamental_report")


class FundamentalReportGenerator(StockAnalysisReportGenerator):
    """基本面深度分析报告生成器（独立报告）"""

    def generate(self,
                 stock_code: str,
                 stock_name: str,
                 fundamental_data: Dict,
                 ohlcv_data: Optional[pd.DataFrame] = None,
                 industry_data: Optional[Dict] = None,
                 shareholder_data: Optional[Dict] = None,
                 output_path: str = None,
                 llm_prompts: Optional[Dict] = None) -> str:
        """
        生成基本面深度分析报告

        参数:
            stock_code: 股票代码
            stock_name: 股票名称
            fundamental_data: 基本面数据（财务指标、估值指标等）
            ohlcv_data: OHLCV 数据（可选，用于提取当前价/市值）
            industry_data: 行业数据（行业景气度、竞争格局等）
            shareholder_data: 股东结构数据（A股特色：十大股东、解禁、回购）
            output_path: 输出文件路径
            llm_prompts: LLM 分析师 prompt

        返回:
            HTML 报告文件路径
        """
        logger.info(f"开始生成 {stock_name}({stock_code}) 基本面深度分析报告")

        # 防御性初始化
        fundamental_data = fundamental_data or {}
        industry_data = industry_data or {}
        shareholder_data = shareholder_data or {}

        # 基本面评分（满分100）
        valuation_score = self._score_valuation(fundamental_data)
        profitability_score = self._score_profitability(fundamental_data)
        growth_score = self._score_growth(fundamental_data)
        fundamental_total = float(min(100.0, max(0.0,
            valuation_score + profitability_score + growth_score
        )))
        rating = self._rating(fundamental_total)

        scores = {
            "fundamental": round(fundamental_total, 1),
            "rating": rating,
            "breakdown": {
                "valuation": round(valuation_score, 1),
                "profitability": round(profitability_score, 1),
                "growth": round(growth_score, 1),
            },
        }

        # 估值水平判定
        valuation_level = self._judge_valuation_level(fundamental_data)

        # 风险提示（基本面维度）
        risk_warnings = self._generate_fundamental_risk_warnings(
            fundamental_data, shareholder_data, industry_data
        )

        # K线行情图（TradingView lightweight-charts，仅保留基本K线和均线）
        kline_chart_html = ""
        if ohlcv_data is not None and len(ohlcv_data) > 0:
            kline_chart_html = self._safe_render_chart(
                self.kline_gen, "generate_tradingview_chart",
                ohlcv_data, stock_code=stock_code, stock_name=stock_name,
                show_support_resistance=False,
                fallback="K线图暂不可用"
            )

        # 当前价/市值
        current_price = 0.0
        market_cap = self._get_fundamental(fundamental_data, "market_cap", "total_market_cap", "总市值")
        if ohlcv_data is not None and len(ohlcv_data) > 0:
            try:
                current_price = float(ohlcv_data.iloc[-1]['close'])
            except Exception:
                pass

        data_date = self._get_data_date(ohlcv_data) if ohlcv_data is not None else "—"

        # 派生常用财务指标
        fin = self._extract_financial_metrics(fundamental_data)

        context = {
            "stock_code": _html.escape(str(stock_code)),
            "stock_name": _html.escape(str(stock_name)),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_date": data_date,
            "current_price": current_price,
            "market_cap": market_cap,
            "scores": scores,
            "valuation_level": valuation_level,
            "fundamental_data": fundamental_data,
            "industry_data": industry_data,
            "shareholder_data": shareholder_data,
            "fin": fin,
            "risk_warnings": risk_warnings,
            "kline_chart_html": kline_chart_html,
            "has_llm_prompts": bool(llm_prompts),
        }

        html_content = self._render_html(context)

        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"基本面报告已保存: {output_path}")
            return output_path
        return html_content

    # ================================================================
    # 估值水平判定
    # ================================================================

    def _judge_valuation_level(self, fundamental_data: Dict) -> str:
        """根据 PE/PB 分位判定估值水平：低估/合理/偏高/高估"""
        if not fundamental_data:
            return "—"

        pe_pct = self._get_fundamental(fundamental_data, "pe_percentile", "pe_pct", "pe_quantile")
        pb_pct = self._get_fundamental(fundamental_data, "pb_percentile", "pb_pct", "pb_quantile")

        pcts = []
        for pct in [pe_pct, pb_pct]:
            if pct is not None and pd.notna(pct):
                try:
                    pcts.append(float(pct))
                except Exception:
                    pass

        if not pcts:
            return "—"

        avg_pct = float(np.mean(pcts))
        if avg_pct < 0.2:
            return "低估"
        elif avg_pct < 0.4:
            return "合理偏低"
        elif avg_pct < 0.6:
            return "合理"
        elif avg_pct < 0.8:
            return "偏高"
        else:
            return "高估"

    # ================================================================
    # 财务指标提取（用于模板渲染）
    # ================================================================

    def _extract_financial_metrics(self, fundamental_data: Dict) -> Dict[str, Any]:
        """从 fundamental_data 中提取并归类财务指标，便于模板渲染"""
        fd = fundamental_data or {}
        return {
            # 估值类
            "pe_ttm": self._get_fundamental(fd, "pe_ttm", "pe", "pe_ratio"),
            "pb": self._get_fundamental(fd, "pb", "pb_ratio"),
            "ps_ttm": self._get_fundamental(fd, "ps_ttm", "ps", "ps_ratio"),
            "pe_percentile": self._get_fundamental(fd, "pe_percentile", "pe_pct", "pe_quantile"),
            "pb_percentile": self._get_fundamental(fd, "pb_percentile", "pb_pct", "pb_quantile"),
            "dv_ratio": self._get_fundamental(fd, "dv_ratio", "dividend_yield", "股息率"),
            # 盈利能力类
            "roe": self._get_fundamental(fd, "roe", "roe_ttm", "return_on_equity"),
            "roa": self._get_fundamental(fd, "roa", "return_on_assets"),
            "gross_margin": self._get_fundamental(fd, "gross_margin", "毛利率"),
            "net_margin": self._get_fundamental(fd, "net_margin", "净利率", "profit_margin"),
            # 成长性类
            "revenue_growth": self._get_fundamental(fd, "revenue_growth", "rev_growth", "营收增速"),
            "profit_growth": self._get_fundamental(fd, "profit_growth", "net_profit_growth", "利润增速"),
            # 规模类
            "revenue": self._get_fundamental(fd, "revenue", "total_revenue", "营业收入"),
            "net_profit": self._get_fundamental(fd, "net_profit", "净利润", "net_income"),
            "total_assets": self._get_fundamental(fd, "total_assets", "总资产"),
            "net_assets": self._get_fundamental(fd, "net_assets", "净资产", "shareholders_equity"),
            # 偿债能力
            "debt_ratio": self._get_fundamental(fd, "debt_ratio", "资产负债率"),
            "current_ratio": self._get_fundamental(fd, "current_ratio", "流动比率"),
            # 现金流
            "operating_cashflow": self._get_fundamental(fd, "operating_cashflow", "经营现金流", "cfo"),
            "free_cashflow": self._get_fundamental(fd, "free_cashflow", "fcf", "自由现金流"),
        }

    # ================================================================
    # 基本面风险提示
    # ================================================================

    def _generate_fundamental_risk_warnings(self,
                                            fundamental_data: Dict,
                                            shareholder_data: Dict,
                                            industry_data: Dict) -> List[str]:
        """生成基本面维度的风险提示"""
        warnings: List[str] = []
        fd = fundamental_data or {}
        sd = shareholder_data or {}
        idt = industry_data or {}

        # 1. 高估值风险
        pe_pct = self._get_fundamental(fd, "pe_percentile", "pe_pct", "pe_quantile")
        if pe_pct is not None and pd.notna(pe_pct):
            try:
                if float(pe_pct) > 0.8:
                    warnings.append(f"PE 处于历史 {float(pe_pct)*100:.0f}% 分位，估值偏高，回调风险较大")
                elif float(pe_pct) > 0.6:
                    warnings.append(f"PE 处于历史 {float(pe_pct)*100:.0f}% 分位，估值偏高")
            except Exception:
                pass

        # 2. 业绩下滑风险
        profit_growth = self._get_fundamental(fd, "profit_growth", "net_profit_growth", "利润增速")
        if profit_growth is not None and pd.notna(profit_growth):
            try:
                g = float(profit_growth)
                if abs(g) < 1:
                    g = g * 100
                if g < -10:
                    warnings.append(f"净利润同比下滑 {abs(g):.1f}%，业绩明显恶化")
                elif g < 0:
                    warnings.append(f"净利润同比下滑 {abs(g):.1f}%，需关注业绩可持续性")
            except Exception:
                pass

        # 3. 盈利能力恶化
        roe = self._get_fundamental(fd, "roe", "roe_ttm", "return_on_equity")
        if roe is not None and pd.notna(roe):
            try:
                r = float(roe)
                if abs(r) < 1:
                    r = r * 100
                if r < 0:
                    warnings.append(f"ROE 为 {r:.1f}%，处于亏损状态")
                elif r < 5:
                    warnings.append(f"ROE 仅 {r:.1f}%，盈利能力较弱")
            except Exception:
                pass

        # 4. 高负债风险
        debt_ratio = self._get_fundamental(fd, "debt_ratio", "资产负债率")
        if debt_ratio is not None and pd.notna(debt_ratio):
            try:
                d = float(debt_ratio)
                if abs(d) < 1:
                    d = d * 100
                if d > 70:
                    warnings.append(f"资产负债率 {d:.1f}%，负债水平较高，财务风险上升")
                elif d > 60:
                    warnings.append(f"资产负债率 {d:.1f}%，需关注偿债能力")
            except Exception:
                pass

        # 5. 解禁风险（A股特色）
        upcoming_unlock = sd.get("upcoming_unlock") if isinstance(sd, dict) else None
        if upcoming_unlock and isinstance(upcoming_unlock, dict):
            try:
                unlock_ratio = upcoming_unlock.get("unlock_ratio")
                if unlock_ratio and float(unlock_ratio) > 0.1:
                    warnings.append(
                        f"近期存在解禁，解禁比例约 {float(unlock_ratio)*100:.1f}%，"
                        f"解禁日期 {upcoming_unlock.get('unlock_date', '—')}"
                    )
            except Exception:
                pass

        # 6. 股东减持风险（A股特色）
        reduction = sd.get("shareholder_reduction") if isinstance(sd, dict) else None
        if reduction and isinstance(reduction, dict):
            try:
                red_pct = reduction.get("reduction_ratio")
                if red_pct and float(red_pct) > 0.02:
                    warnings.append(
                        f"大股东近期减持比例 {float(red_pct)*100:.2f}%，"
                        f"需关注管理层信心"
                    )
            except Exception:
                pass

        # 7. 行业景气度下行
        prosperity = idt.get("prosperity_trend") if isinstance(idt, dict) else None
        if prosperity == "down":
            warnings.append("行业景气度处于下行周期，整体需求走弱")

        if not warnings:
            warnings.append("未检测到明显基本面风险信号（基于现有数据）")

        return warnings

    # ================================================================
    # HTML 渲染（10章节基本面报告）
    # ================================================================

    def _render_html(self, context: Dict) -> str:
        template_str = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ stock_name }} ({{ stock_code }}) 基本面分析报告</title>
    <style>
        :root {
            --bg: #f0f2f5;
            --card-bg: #ffffff;
            --text: #1f2937;
            --text-muted: #6b7280;
            --text-light: #9ca3af;
            --border: #e5e7eb;
            --border-light: #f3f4f6;
            --primary: #0891b2;
            --primary-light: #cffafe;
            --success: #10b981;
            --success-light: #d1fae5;
            --danger: #ef4444;
            --danger-light: #fee2e2;
            --warning: #f59e0b;
            --warning-light: #fef3c7;
            --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
            --shadow-lg: 0 4px 12px rgba(0,0,0,0.08);
            --radius: 10px;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #111827;
                --card-bg: #1f2937;
                --text: #e5e7eb;
                --text-muted: #9ca3af;
                --text-light: #6b7280;
                --border: #374151;
                --border-light: #283242;
                --primary: #22d3ee;
                --primary-light: #0e3b45;
                --success: #34d399;
                --success-light: #1a3b2e;
                --danger: #f87171;
                --danger-light: #3b1a1a;
                --warning: #fbbf24;
                --warning-light: #3b3015;
                --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
                --shadow-lg: 0 4px 12px rgba(0,0,0,0.4);
            }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                         'Hiragino Sans GB', 'Microsoft YaHei', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px 20px 60px;
            font-size: 14px;
        }
        .report-header {
            background: linear-gradient(135deg, #134e4a 0%, #0891b2 50%, #06b6d4 100%);
            color: #fff;
            padding: 36px 32px;
            border-radius: var(--radius);
            margin-bottom: 24px;
            box-shadow: var(--shadow-lg);
        }
        .report-header h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; }
        .report-header .subtitle { font-size: 15px; opacity: 0.92; margin-bottom: 14px; }
        .report-header .meta { font-size: 12.5px; opacity: 0.78; display: flex; gap: 18px; flex-wrap: wrap; }
        .section {
            background: var(--card-bg);
            border-radius: var(--radius);
            padding: 24px 28px;
            margin-bottom: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-light);
        }
        .section-title {
            font-size: 17px; font-weight: 600; color: var(--text);
            margin-bottom: 18px; padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
            display: flex; align-items: center; gap: 8px;
        }
        .section-title::before {
            content: ''; display: inline-block;
            width: 4px; height: 18px;
            background: var(--primary); border-radius: 2px;
        }
        /* 仪表盘 */
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px; margin-bottom: 16px;
        }
        @media (max-width: 768px) { .dashboard { grid-template-columns: 1fr; } }
        .dash-card {
            background: var(--card-bg); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 20px; text-align: center;
        }
        .dash-card .label { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }
        .dash-card .value { font-size: 32px; font-weight: 700; line-height: 1.1; }
        .dash-card .sub { font-size: 12px; color: var(--text-muted); margin-top: 6px; }
        .value-high { color: var(--success); }
        .value-mid { color: var(--warning); }
        .value-low { color: var(--danger); }
        /* 指标卡片 */
        .indicator-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px;
        }
        .indicator-card {
            background: var(--bg); border-radius: 8px; padding: 14px 16px;
            border: 1px solid var(--border-light);
        }
        .indicator-card .ind-name { font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
        .indicator-card .ind-value { font-size: 22px; font-weight: 700; color: var(--text); }
        .indicator-card .ind-signal { font-size: 11.5px; margin-top: 4px; }
        /* 表格 */
        table { width: 100%; border-collapse: collapse; margin-top: 4px; font-size: 13px; }
        th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border-light); }
        th { background: var(--bg); font-weight: 600; color: var(--text-muted); font-size: 12px; }
        tbody tr:hover { background: var(--border-light); }
        .badge {
            display: inline-block; padding: 2px 10px;
            border-radius: 12px; font-size: 11.5px; font-weight: 500;
        }
        .badge-bullish { background: var(--success-light); color: var(--success); }
        .badge-bearish { background: var(--danger-light); color: var(--danger); }
        .badge-neutral { background: var(--border-light); color: var(--text-muted); }
        .badge-high { background: var(--danger-light); color: var(--danger); }
        .badge-medium { background: var(--warning-light); color: var(--warning); }
        .badge-low { background: var(--border-light); color: var(--text-muted); }
        /* 估值分位条 */
        .percentile-bar {
            height: 8px; background: var(--bg); border-radius: 4px;
            margin-top: 6px; position: relative; overflow: hidden;
        }
        .percentile-fill {
            height: 100%; border-radius: 4px;
            background: linear-gradient(90deg, var(--success) 0%, var(--warning) 50%, var(--danger) 100%);
        }
        .percentile-marker {
            position: absolute; top: -4px; width: 4px; height: 16px;
            background: var(--text); border-radius: 2px;
        }
        /* 风险提示 */
        .risk-section { border-left: 4px solid var(--warning); }
        .risk-list { list-style: none; padding: 0; }
        .risk-list li {
            padding: 10px 14px 10px 36px; margin-bottom: 8px;
            background: var(--bg); border-radius: 6px; position: relative;
        }
        .risk-list li::before {
            content: '⚠'; position: absolute; left: 12px; top: 50%;
            transform: translateY(-50%); color: var(--warning); font-size: 16px;
        }
        /* LLM 章节 */
        .llm-section { border-left: 4px solid #7c3aed; }
        .llm-section .section-title { color: #7c3aed; }
        .llm-placeholder {
            padding: 12px; background: #faf5ff; border: 1px dashed #c4b5fd;
            border-radius: 6px; color: #6b21a8; font-size: 13px;
        }
        /* LLM 分析师渲染内容 */
        .llm-analysis-header { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
        .llm-badge { display: inline-block; padding: 4px 12px; border-radius: 14px;
                     font-size: 12px; font-weight: 600; background: #ede9fe; color: #6b21a8; }
        .llm-badge-positive { background: #dcfce7; color: #15803d; }
        .llm-badge-negative { background: #fee2e2; color: #b91c1c; }
        .llm-analysis-body h4 { margin: 16px 0 6px 0; font-size: 14px; color: #4c1d95; font-weight: 600; }
        .llm-analysis-body p { margin: 0 0 8px 0; font-size: 13.5px; line-height: 1.7; color: var(--text); }
        .llm-tag-a { display: inline-block; padding: 1px 6px; border-radius: 8px;
                     font-size: 10px; font-weight: 500; background: #fef3c7; color: #92400e; margin-left: 4px; }
        /* 信息卡 */
        .info-card {
            background: var(--bg); padding: 14px 16px; border-radius: 8px;
            border: 1px solid var(--border-light); margin-bottom: 10px;
        }
        .info-card .info-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
        .info-card .info-value { font-size: 16px; font-weight: 700; }
        .chart-placeholder {
            padding: 40px; text-align: center; color: var(--text-light);
            background: var(--bg); border-radius: 8px; font-size: 13px;
        }
        .disclaimer {
            text-align: center; margin-top: 32px; padding: 20px;
            font-size: 12.5px; color: var(--text-light);
            border-top: 1px solid var(--border);
        }
        .disclaimer strong { color: var(--warning); }
        @media (max-width: 640px) {
            body { padding: 12px 8px 40px; font-size: 13px; }
            .report-header { padding: 24px 20px; }
            .report-header h1 { font-size: 20px; }
            .section { padding: 16px 14px; }
            .dash-card .value { font-size: 26px; }
            th, td { padding: 8px 10px; }
        }
    </style>
</head>
<body>

<!-- 报告标题 -->
<header class="report-header">
    <h1>{{ stock_name }} ({{ stock_code }})</h1>
    <div class="subtitle">📊 基本面深度分析报告</div>
    <div class="meta">
        <span>📅 生成时间: {{ generated_at }}</span>
        <span>📊 数据截止: {{ data_date }}</span>
        {% if current_price > 0 %}
        <span>💰 当前价: {{ _fmt_price(current_price) }} 元</span>
        {% endif %}
        {% if market_cap is not none %}
        <span>🏦 总市值: {{ _fmt_market_cap(market_cap) }}</span>
        {% endif %}
    </div>
</header>

<!-- 行情数据（K线趋势图 + 技术指标切换） -->
<section class="section">
    <div class="section-title">行情数据</div>
    <div class="chart-container">
    {% if kline_chart_html %}
        {{ kline_chart_html | safe }}
    {% else %}
        <div class="chart-placeholder">K线图暂不可用</div>
    {% endif %}
    </div>
</section>

<!-- 1. 报告概览（基本面评分仪表盘） -->
<section class="section">
    <div class="section-title">报告概览</div>
    <div class="dashboard">
        <div class="dash-card">
            <div class="label">基本面评分</div>
            <div class="value {{ 'value-high' if scores.fundamental >= 70 else ('value-mid' if scores.fundamental >= 50 else 'value-low') }}">{{ scores.fundamental }}</div>
            <div class="sub">满分 100 / {{ scores.rating }}</div>
        </div>
        <div class="dash-card">
            <div class="label">估值水平</div>
            <div class="value">
                {% if valuation_level in ['低估'] %}
                    <span class="value-high">{{ valuation_level }}</span>
                {% elif valuation_level in ['合理', '合理偏低'] %}
                    <span class="value-mid">{{ valuation_level }}</span>
                {% elif valuation_level in ['偏高', '高估'] %}
                    <span class="value-low">{{ valuation_level }}</span>
                {% else %}
                    <span style="color:var(--text-muted);">{{ valuation_level }}</span>
                {% endif %}
            </div>
            <div class="sub">基于 PE/PB 历史分位</div>
        </div>
        <div class="dash-card">
            <div class="label">投资评级</div>
            <div class="value" style="font-size:24px;">{{ scores.rating }}</div>
            <div class="sub">基于基本面综合评分</div>
        </div>
    </div>
    <div class="indicator-grid" style="margin-top:12px;">
        <div class="indicator-card">
            <div class="ind-name">估值水平 /35</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.valuation }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">盈利能力 /35</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.profitability }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">成长性 /30</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.growth }}</div>
        </div>
    </div>
</section>

<!-- 2. 公司概况与业务分析 -->
<section class="section">
    <div class="section-title">公司概况与业务分析</div>
    {% if fundamental_data %}
    <div class="indicator-grid">
        {% if fin.revenue is not none %}
        <div class="indicator-card">
            <div class="ind-name">营业收入 (TTM)</div>
            <div class="ind-value">{{ _fmt_market_cap(fin.revenue) }}</div>
        </div>
        {% endif %}
        {% if fin.net_profit is not none %}
        <div class="indicator-card">
            <div class="ind-name">净利润 (TTM)</div>
            <div class="ind-value">{{ _fmt_market_cap(fin.net_profit) }}</div>
        </div>
        {% endif %}
        {% if fin.total_assets is not none %}
        <div class="indicator-card">
            <div class="ind-name">总资产</div>
            <div class="ind-value">{{ _fmt_market_cap(fin.total_assets) }}</div>
        </div>
        {% endif %}
        {% if fin.net_assets is not none %}
        <div class="indicator-card">
            <div class="ind-name">净资产</div>
            <div class="ind-value">{{ _fmt_market_cap(fin.net_assets) }}</div>
        </div>
        {% endif %}
    </div>
    {% if fundamental_data.industry %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">所属行业</div>
        <div style="font-size:14px;">{{ fundamental_data.industry }}</div>
    </div>
    {% endif %}
    {% if fundamental_data.business_scope %}
    <div class="info-card">
        <div class="info-title">主营业务</div>
        <div style="font-size:13px;line-height:1.7;color:var(--text-muted);">{{ fundamental_data.business_scope }}</div>
    </div>
    {% endif %}
    {% if fundamental_data.main_products %}
    <div class="info-card">
        <div class="info-title">主要产品/服务</div>
        <div style="font-size:13px;line-height:1.7;color:var(--text-muted);">{{ fundamental_data.main_products }}</div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">公司概况数据暂缺</div>
    {% endif %}
</section>

<!-- 3. 行业分析与景气度 -->
<section class="section">
    <div class="section-title">行业分析与景气度</div>
    {% if industry_data and industry_data.has_data %}
    <div class="indicator-grid">
        {% if industry_data.prosperity_index is not none %}
        <div class="indicator-card">
            <div class="ind-name">行业景气度指数</div>
            <div class="ind-value">{{ _fmt_num(industry_data.prosperity_index, 1) }}</div>
            <div class="ind-signal">
                {% set pt = industry_data.prosperity_trend %}
                {% if pt == 'up' %}<span class="badge badge-bullish">景气上行</span>
                {% elif pt == 'down' %}<span class="badge badge-bearish">景气下行</span>
                {% else %}<span class="badge badge-neutral">平稳</span>{% endif %}
            </div>
        </div>
        {% endif %}
        {% if industry_data.market_share is not none %}
        <div class="indicator-card">
            <div class="ind-name">市场份额</div>
            <div class="ind-value">{{ _fmt_pct(industry_data.market_share) }}</div>
        </div>
        {% endif %}
        {% if industry_data.industry_growth is not none %}
        <div class="indicator-card">
            <div class="ind-name">行业增速</div>
            <div class="ind-value">{{ _fmt_pct(industry_data.industry_growth) }}</div>
        </div>
        {% endif %}
        {% if industry_data.competition_level %}
        <div class="indicator-card">
            <div class="ind-name">竞争格局</div>
            <div class="ind-value" style="font-size:18px;">{{ industry_data.competition_level }}</div>
        </div>
        {% endif %}
    </div>
    {% if industry_data.outlook %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">行业展望</div>
        <div style="font-size:13px;line-height:1.7;color:var(--text-muted);">{{ industry_data.outlook }}</div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">行业数据暂缺（需接入行业景气度数据源）</div>
    {% endif %}
</section>

<!-- 4. 财务报表分析 -->
<section class="section">
    <div class="section-title">财务报表分析</div>
    {% if fundamental_data %}
    <table>
        <thead>
            <tr>
                <th>财务指标</th><th>数值</th><th>类别</th>
            </tr>
        </thead>
        <tbody>
            {% if fin.revenue is not none %}
            <tr><td>营业收入</td><td>{{ _fmt_market_cap(fin.revenue) }}</td><td><span class="badge badge-neutral">利润表</span></td></tr>
            {% endif %}
            {% if fin.net_profit is not none %}
            <tr><td>净利润</td><td>{{ _fmt_market_cap(fin.net_profit) }}</td><td><span class="badge badge-neutral">利润表</span></td></tr>
            {% endif %}
            {% if fin.gross_margin is not none %}
            <tr><td>毛利率</td><td>{{ _fmt_pct(fin.gross_margin) }}</td><td><span class="badge badge-neutral">利润表</span></td></tr>
            {% endif %}
            {% if fin.net_margin is not none %}
            <tr><td>净利率</td><td>{{ _fmt_pct(fin.net_margin) }}</td><td><span class="badge badge-neutral">利润表</span></td></tr>
            {% endif %}
            {% if fin.total_assets is not none %}
            <tr><td>总资产</td><td>{{ _fmt_market_cap(fin.total_assets) }}</td><td><span class="badge badge-neutral">资产负债表</span></td></tr>
            {% endif %}
            {% if fin.net_assets is not none %}
            <tr><td>净资产</td><td>{{ _fmt_market_cap(fin.net_assets) }}</td><td><span class="badge badge-neutral">资产负债表</span></td></tr>
            {% endif %}
            {% if fin.debt_ratio is not none %}
            <tr><td>资产负债率</td><td>{{ _fmt_pct(fin.debt_ratio) }}</td><td><span class="badge badge-neutral">资产负债表</span></td></tr>
            {% endif %}
            {% if fin.current_ratio is not none %}
            <tr><td>流动比率</td><td>{{ _fmt_num(fin.current_ratio, 2) }}</td><td><span class="badge badge-neutral">资产负债表</span></td></tr>
            {% endif %}
            {% if fin.operating_cashflow is not none %}
            <tr><td>经营现金流</td><td>{{ _fmt_market_cap(fin.operating_cashflow) }}</td><td><span class="badge badge-neutral">现金流量表</span></td></tr>
            {% endif %}
            {% if fin.free_cashflow is not none %}
            <tr><td>自由现金流</td><td>{{ _fmt_market_cap(fin.free_cashflow) }}</td><td><span class="badge badge-neutral">现金流量表</span></td></tr>
            {% endif %}
        </tbody>
    </table>
    {% if fin.operating_cashflow is not none and fin.net_profit is not none and fin.net_profit > 0 %}
    {% set cfo_ratio = fin.operating_cashflow / fin.net_profit %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">盈利质量（经营现金流/净利润）</div>
        <div class="info-value {{ 'value-high' if cfo_ratio >= 1 else 'value-low' }}">
            {{ _fmt_num(cfo_ratio, 2) }}
            {% if cfo_ratio >= 1 %}
                <span class="badge badge-bullish" style="margin-left:8px;">现金流充足，盈利质量高</span>
            {% else %}
                <span class="badge badge-bearish" style="margin-left:8px;">现金流弱于账面利润</span>
            {% endif %}
        </div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">财务数据暂缺</div>
    {% endif %}
</section>

<!-- 5. 盈利能力分析 -->
<section class="section">
    <div class="section-title">盈利能力分析</div>
    {% if fundamental_data %}
    <div class="indicator-grid">
        {% if fin.roe is not none %}
        <div class="indicator-card">
            <div class="ind-name">ROE (TTM)</div>
            <div class="ind-value {{ 'value-high' if (fin.roe | float) >= 15 else ('value-mid' if (fin.roe | float) >= 10 else 'value-low') }}">
                {{ _fmt_pct(fin.roe) }}
            </div>
            <div class="ind-signal">
                {% set roe_v = fin.roe | float %}
                {% if roe_v >= 20 %}<span class="badge badge-bullish">优秀</span>
                {% elif roe_v >= 15 %}<span class="badge badge-bullish">良好</span>
                {% elif roe_v >= 10 %}<span class="badge badge-medium">中等</span>
                {% else %}<span class="badge badge-bearish">较弱</span>{% endif %}
            </div>
        </div>
        {% endif %}
        {% if fin.roa is not none %}
        <div class="indicator-card">
            <div class="ind-name">ROA</div>
            <div class="ind-value">{{ _fmt_pct(fin.roa) }}</div>
        </div>
        {% endif %}
        {% if fin.gross_margin is not none %}
        <div class="indicator-card">
            <div class="ind-name">毛利率</div>
            <div class="ind-value">{{ _fmt_pct(fin.gross_margin) }}</div>
            <div class="ind-signal">
                {% set gm = fin.gross_margin | float %}
                {% if gm >= 60 %}<span class="badge badge-bullish">高毛利</span>
                {% elif gm >= 40 %}<span class="badge badge-medium">中等</span>
                {% else %}<span class="badge badge-low">偏低</span>{% endif %}
            </div>
        </div>
        {% endif %}
        {% if fin.net_margin is not none %}
        <div class="indicator-card">
            <div class="ind-name">净利率</div>
            <div class="ind-value">{{ _fmt_pct(fin.net_margin) }}</div>
        </div>
        {% endif %}
    </div>
    {% else %}
    <div class="chart-placeholder">盈利能力数据暂缺</div>
    {% endif %}
</section>

<!-- 6. 成长性分析 -->
<section class="section">
    <div class="section-title">成长性分析</div>
    {% if fundamental_data %}
    <div class="indicator-grid">
        {% if fin.revenue_growth is not none %}
        <div class="indicator-card">
            <div class="ind-name">营收增速</div>
            <div class="ind-value {{ 'value-high' if (fin.revenue_growth | float) >= 15 else ('value-mid' if (fin.revenue_growth | float) >= 5 else 'value-low') }}">
                {{ _fmt_pct(fin.revenue_growth) }}
            </div>
            <div class="ind-signal">
                {% set rg = fin.revenue_growth | float %}
                {% if rg >= 30 %}<span class="badge badge-bullish">高成长</span>
                {% elif rg >= 15 %}<span class="badge badge-bullish">稳健成长</span>
                {% elif rg >= 5 %}<span class="badge badge-medium">温和增长</span>
                {% elif rg >= 0 %}<span class="badge badge-low">微增</span>
                {% else %}<span class="badge badge-bearish">负增长</span>{% endif %}
            </div>
        </div>
        {% endif %}
        {% if fin.profit_growth is not none %}
        <div class="indicator-card">
            <div class="ind-name">净利润增速</div>
            <div class="ind-value {{ 'value-high' if (fin.profit_growth | float) >= 15 else ('value-mid' if (fin.profit_growth | float) >= 5 else 'value-low') }}">
                {{ _fmt_pct(fin.profit_growth) }}
            </div>
            <div class="ind-signal">
                {% set pg = fin.profit_growth | float %}
                {% if pg >= 30 %}<span class="badge badge-bullish">高成长</span>
                {% elif pg >= 15 %}<span class="badge badge-bullish">稳健成长</span>
                {% elif pg >= 5 %}<span class="badge badge-medium">温和增长</span>
                {% elif pg >= 0 %}<span class="badge badge-low">微增</span>
                {% else %}<span class="badge badge-bearish">负增长</span>{% endif %}
            </div>
        </div>
        {% endif %}
    </div>
    {% if fin.revenue_growth is not none and fin.profit_growth is not none %}
    {% set rev_g = fin.revenue_growth | float %}
    {% set prof_g = fin.profit_growth | float %}
    {% if prof_g > rev_g * 1.2 and rev_g > 0 %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">利润增速 vs 营收增速</div>
        <div style="font-size:13px;color:var(--text-muted);">
            利润增速 <strong style="color:var(--success);">{{ _fmt_pct(prof_g) }}</strong>
            显著高于营收增速 <strong>{{ _fmt_pct(rev_g) }}</strong>，
            表明盈利能力提升或成本控制有效
        </div>
    </div>
    {% elif prof_g < rev_g * 0.5 and rev_g > 0 %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">利润增速 vs 营收增速</div>
        <div style="font-size:13px;color:var(--text-muted);">
            利润增速 <strong style="color:var(--danger);">{{ _fmt_pct(prof_g) }}</strong>
            明显低于营收增速 <strong>{{ _fmt_pct(rev_g) }}</strong>，
            需关注成本上升或毛利率下滑压力
        </div>
    </div>
    {% endif %}
    {% endif %}
    {% else %}
    <div class="chart-placeholder">成长性数据暂缺</div>
    {% endif %}
</section>

<!-- 7. 估值分析 -->
<section class="section">
    <div class="section-title">估值分析</div>
    {% if fundamental_data %}
    <div class="indicator-grid">
        {% if fin.pe_ttm is not none %}
        <div class="indicator-card">
            <div class="ind-name">PE (TTM)</div>
            <div class="ind-value">{{ _fmt_num(fin.pe_ttm, 1) }}</div>
            {% if fin.pe_percentile is not none %}
            <div class="ind-signal">历史分位: {{ _fmt_pct(fin.pe_percentile) }}</div>
            <div class="percentile-bar">
                <div class="percentile-fill" style="width:100%;"></div>
                {% set pe_pct = (fin.pe_percentile | float) * 100 %}
                <div class="percentile-marker" style="left:{{ pe_pct }}%;"></div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        {% if fin.pb is not none %}
        <div class="indicator-card">
            <div class="ind-name">PB</div>
            <div class="ind-value">{{ _fmt_num(fin.pb, 2) }}</div>
            {% if fin.pb_percentile is not none %}
            <div class="ind-signal">历史分位: {{ _fmt_pct(fin.pb_percentile) }}</div>
            <div class="percentile-bar">
                <div class="percentile-fill" style="width:100%;"></div>
                {% set pb_pct = (fin.pb_percentile | float) * 100 %}
                <div class="percentile-marker" style="left:{{ pb_pct }}%;"></div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        {% if fin.ps_ttm is not none %}
        <div class="indicator-card">
            <div class="ind-name">PS (TTM)</div>
            <div class="ind-value">{{ _fmt_num(fin.ps_ttm, 2) }}</div>
        </div>
        {% endif %}
        {% if fin.dv_ratio is not none %}
        <div class="indicator-card">
            <div class="ind-name">股息率</div>
            <div class="ind-value">{{ _fmt_pct(fin.dv_ratio) }}</div>
            <div class="ind-signal">
                {% set dv = fin.dv_ratio | float %}
                {% if dv >= 0.04 %}<span class="badge badge-bullish">高分红</span>
                {% elif dv >= 0.02 %}<span class="badge badge-medium">中等</span>
                {% else %}<span class="badge badge-low">低分红</span>{% endif %}
            </div>
        </div>
        {% endif %}
    </div>
    {% if fin.pe_percentile is not none or fin.pb_percentile is not none %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">估值综合判断</div>
        <div style="font-size:13px;color:var(--text-muted);">
            当前估值水平: <strong>{{ valuation_level }}</strong>
            <br>
            <span style="font-size:12px;color:var(--text-light);">
                （分位条左侧绿色=低估，中间黄色=合理，右侧红色=高估；竖线为当前所处位置）
            </span>
        </div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">估值数据暂缺</div>
    {% endif %}
</section>

<!-- 8. 股东结构与资本运作（A股特色） -->
<section class="section">
    <div class="section-title">股东结构与资本运作 <span class="badge badge-neutral" style="margin-left:8px;">A股特色</span></div>
    {% if shareholder_data and shareholder_data.has_data %}
    {% if shareholder_data.top_holders %}
    <h4 style="font-size:14px;margin-bottom:10px;color:var(--text);">十大股东</h4>
    <table>
        <thead>
            <tr><th>股东名称</th><th>持股比例</th><th>变动</th><th>性质</th></tr>
        </thead>
        <tbody>
        {% for h in shareholder_data.top_holders[:10] %}
            <tr>
                <td>{{ h.name | default('—', true) }}</td>
                <td><strong>{{ _fmt_pct(h.ratio) }}</strong></td>
                <td>
                    {% set chg = h.change %}
                    {% if chg is not none %}
                        {% if chg | float > 0 %}<span class="badge badge-bearish">增持 {{ _fmt_pct(chg) }}</span>
                        {% elif chg | float < 0 %}<span class="badge badge-bullish">减持 {{ _fmt_pct(chg) }}</span>
                        {% else %}<span class="badge badge-neutral">不变</span>{% endif %}
                    {% else %}
                        <span class="badge badge-neutral">—</span>
                    {% endif %}
                </td>
                <td>{{ h.type | default('—', true) }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% endif %}

    {% if shareholder_data.upcoming_unlock %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">近期解禁</div>
        <div class="info-value">
            解禁日期: {{ shareholder_data.upcoming_unlock.unlock_date | default('—', true) }} |
            解禁比例:
            {% if shareholder_data.upcoming_unlock.unlock_ratio is not none %}
                <span class="value-low">{{ _fmt_pct(shareholder_data.upcoming_unlock.unlock_ratio) }}</span>
            {% else %}—{% endif %}
        </div>
    </div>
    {% endif %}

    {% if shareholder_data.buyback %}
    <div class="info-card">
        <div class="info-title">股份回购</div>
        <div style="font-size:13px;color:var(--text-muted);">
            回购金额: {{ _fmt_market_cap(shareholder_data.buyback.amount) }} |
            回购价格区间: {{ shareholder_data.buyback.price_range | default('—', true) }}
        </div>
    </div>
    {% endif %}

    {% if shareholder_data.shareholder_reduction %}
    <div class="info-card">
        <div class="info-title">大股东减持</div>
        <div class="info-value">
            {% if shareholder_data.shareholder_reduction.reduction_ratio is not none %}
                减持比例: <span class="value-low">{{ _fmt_pct(shareholder_data.shareholder_reduction.reduction_ratio) }}</span>
            {% endif %}
        </div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">
        股东结构与资本运作数据暂缺（需接入 akshare 十大股东/解禁/回购接口）
    </div>
    {% endif %}
</section>

<!-- 10. LLM 分析师章节（基本面深度解读） -->
<section class="section llm-section">
    <div class="section-title">基本面深度解读</div>
<!--LLM_FUNDAMENTAL_ANALYSIS_PLACEHOLDER-->
</section>

<!-- 风险提示 -->
<section class="section risk-section">
    <div class="section-title">风险提示</div>
    <ul class="risk-list">
    {% for warning in risk_warnings %}
        <li>{{ warning }}</li>
    {% endfor %}
    </ul>
</section>

<!-- 免责声明 -->
<footer class="disclaimer">
    <strong>免责声明：</strong>以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。<br>
    <span style="font-size:11px;">Generated by jingnitrader · 基本面分析报告 · {{ generated_at }}</span>
</footer>

</body>
</html>
"""
        template = Template(template_str)
        render_context = dict(context)
        render_context["_fmt_num"] = self._fmt_num
        render_context["_fmt_price"] = self._fmt_price
        render_context["_fmt_pct"] = self._fmt_pct
        render_context["_fmt_date"] = self._fmt_date
        render_context["_fmt_market_cap"] = self._fmt_market_cap
        return template.render(**render_context)
