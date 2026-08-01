"""
技术面深度分析报告生成器（独立报告，借鉴 TradingAgents market_report 设计）

10 个章节：
1. 报告概览（决策仪表盘风格，借鉴 daily_stock_analysis）
2. 价格走势与K线形态（含涨跌停分析，A股特色）
3. 技术指标解读（MACD/RSI/KDJ/BOLL 逐项分析）
4. 均线系统与趋势（多周期共振）
5. 支撑阻力位
6. 量价关系
7. 资金面分析（A股特色：主力资金+北向资金）
8. 龙虎榜与大单（A股特色）
9. 技术面深度解读（LLM 分析师章节）
10. 风险提示

与 stock_analysis_report.py 的关系：
- 复用其评分/指标提取/格式化辅助方法
- 独立渲染技术面相关章节，不包含基本面章节
"""
import os
import html as _html
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime
from jinja2 import Template

# 复用合并报告的基类方法（评分/格式化/指标提取）
from .stock_analysis_report import StockAnalysisReportGenerator

logger = logging.getLogger("technical_report")


class TechnicalReportGenerator(StockAnalysisReportGenerator):
    """技术面深度分析报告生成器（独立报告）"""

    def generate(self,
                 stock_code: str,
                 stock_name: str,
                 ohlcv_data: pd.DataFrame,
                 technical_indicators: Dict,
                 pattern_results: Dict,
                 support_resistance: Dict,
                 multi_timeframe: Dict,
                 capital_flow: Optional[Dict] = None,
                 dragon_tiger: Optional[Dict] = None,
                 output_path: str = None,
                 llm_prompts: Optional[Dict] = None) -> str:
        """
        生成技术面深度分析报告

        参数:
            stock_code: 股票代码
            stock_name: 股票名称
            ohlcv_data: OHLCV数据
            technical_indicators: 技术指标数据 (MACD, RSI, KDJ, MA等)
            pattern_results: K线形态识别结果
            support_resistance: 支撑阻力位
            multi_timeframe: 多周期分析结果
            capital_flow: 资金面数据（主力资金流向、北向资金）A 股特色
            dragon_tiger: 龙虎榜数据（机构席位、营业部动向）A 股特色
            output_path: 输出文件路径
            llm_prompts: LLM 分析师 prompt

        返回:
            HTML 报告文件路径
        """
        logger.info(f"开始生成 {stock_name}({stock_code}) 技术面深度分析报告")

        # 防御性初始化
        technical_indicators = technical_indicators or {}
        pattern_results = pattern_results or {}
        support_resistance = support_resistance or {}
        multi_timeframe = multi_timeframe or {}
        capital_flow = capital_flow or {}
        dragon_tiger = dragon_tiger or {}

        # 注入量价指标
        enriched_indicators: Dict[str, Any] = dict(technical_indicators)
        volume_metrics = self._compute_volume_metrics(ohlcv_data)
        enriched_indicators.update(volume_metrics)

        # 技术面评分（满分100）
        trend_score = self._score_trend(multi_timeframe, enriched_indicators)
        indicator_score = self._score_indicator_signals(enriched_indicators, multi_timeframe)
        pattern_score = self._score_patterns(pattern_results)
        volume_score = self._score_volume_price(enriched_indicators)

        technical_total = float(min(100.0, max(0.0,
            trend_score + indicator_score + pattern_score + volume_score
        )))
        rating = self._rating(technical_total)

        scores = {
            "technical": round(technical_total, 1),
            "rating": rating,
            "breakdown": {
                "trend": round(trend_score, 1),
                "indicator": round(indicator_score, 1),
                "pattern": round(pattern_score, 1),
                "volume": round(volume_score, 1),
            },
        }

        # 风险提示（仅技术面维度）
        risk_warnings = self._generate_risk_warnings(
            enriched_indicators, pattern_results, multi_timeframe, support_resistance
        )

        # K线联动图（TradingView lightweight-charts）
        kline_chart_html = self._safe_render_chart(
            self.kline_gen, "generate_tradingview_chart",
            ohlcv_data, stock_code=stock_code, stock_name=stock_name,
            show_support_resistance=False,
            fallback="K线图暂不可用"
        )

        data_date = self._get_data_date(ohlcv_data)
        current_price = float(ohlcv_data.iloc[-1]['close']) if len(ohlcv_data) > 0 else 0.0

        # 涨跌停分析（A股特色）
        limit_analysis = self._analyze_price_limit(ohlcv_data)

        context = {
            "stock_code": _html.escape(str(stock_code)),
            "stock_name": _html.escape(str(stock_name)),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_date": data_date,
            "current_price": current_price,
            "scores": scores,
            "multi_timeframe": multi_timeframe,
            "technical_indicators": enriched_indicators,
            "pattern_results": pattern_results,
            "support_resistance": support_resistance,
            "risk_warnings": risk_warnings,
            "kline_chart_html": kline_chart_html,
            "trend_color": self._TREND_COLOR,
            "strength_color": self._STRENGTH_COLOR,
            "divergences": multi_timeframe.get("divergences", []),
            "resonance": multi_timeframe.get("resonance", {}),
            "tf_summary": multi_timeframe.get("summary", ""),
            "capital_flow": capital_flow,
            "dragon_tiger": dragon_tiger,
            "limit_analysis": limit_analysis,
            "has_llm_prompts": bool(llm_prompts),
        }

        html_content = self._render_html(context)

        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"技术面报告已保存: {output_path}")
            return output_path
        return html_content

    # ================================================================
    # A股特色：涨跌停分析
    # ================================================================

    def _analyze_price_limit(self, ohlcv_data: pd.DataFrame) -> Dict:
        """分析近期涨跌停情况（A股特色，ST股5%、主板10%、创业板/科创板20%）"""
        result = {
            "has_limit_data": False,
            "recent_limit_ups": 0,
            "recent_limit_downs": 0,
            "limit_type": "10%",  # 默认主板
            "near_limit": False,
        }
        if ohlcv_data is None or len(ohlcv_data) < 2:
            return result

        try:
            df = ohlcv_data.copy()
            if "date" in df.columns:
                df = df.sort_values("date").reset_index(drop=True)
            recent = df.tail(20).copy()

            # 判断涨跌停类型
            code = str(ohlcv_data["code"].iloc[0]) if "code" in ohlcv_data.columns else ""
            if code.startswith("300") or code.startswith("301") or code.startswith("688"):
                result["limit_type"] = "20%"
            elif "ST" in str(ohlcv_data.get("name", [""]).iloc[0] if "name" in ohlcv_data.columns else ""):
                result["limit_type"] = "5%"

            limit_pct = float(result["limit_type"].rstrip("%")) / 100.0

            # 统计近期涨跌停
            for _, row in recent.iterrows():
                if "pre_close" in row and row["pre_close"] and float(row["pre_close"]) > 0:
                    change = (float(row["close"]) - float(row["pre_close"])) / float(row["pre_close"])
                    if change >= limit_pct - 0.001:
                        result["recent_limit_ups"] += 1
                    elif change <= -limit_pct + 0.001:
                        result["recent_limit_downs"] += 1

            # 是否接近涨停（最近一日涨幅 > 8%）
            if len(recent) > 0:
                last = recent.iloc[-1]
                if "pre_close" in last and last["pre_close"] and float(last["pre_close"]) > 0:
                    last_change = (float(last["close"]) - float(last["pre_close"])) / float(last["pre_close"])
                    result["near_limit"] = last_change >= limit_pct - 0.02

            result["has_limit_data"] = True
        except Exception as e:
            logger.debug(f"涨跌停分析失败: {e}")

        return result

    # ================================================================
    # HTML 渲染（9章节技术面报告）
    # ================================================================

    def _render_html(self, context: Dict) -> str:
        template_str = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ stock_name }} ({{ stock_code }}) 技术面分析报告</title>
    <style>
        :root {
            --bg: #f0f2f5;
            --card-bg: #ffffff;
            --text: #1f2937;
            --text-muted: #6b7280;
            --text-light: #9ca3af;
            --border: #e5e7eb;
            --border-light: #f3f4f6;
            --primary: #3b82f6;
            --primary-light: #dbeafe;
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
                --primary: #60a5fa;
                --primary-light: #1e3a5f;
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
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #6366f1 100%);
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
        /* 决策仪表盘 */
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
        /* 表格 */
        table { width: 100%; border-collapse: collapse; margin-top: 4px; font-size: 13px; }
        th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border-light); }
        th { background: var(--bg); font-weight: 600; color: var(--text-muted); font-size: 12px; }
        tbody tr:hover { background: var(--border-light); }
        .trend-up { color: var(--danger); font-weight: 600; }
        .trend-down { color: var(--success); font-weight: 600; }
        .trend-neutral { color: var(--text-muted); font-weight: 500; }
        .trend-unknown { color: var(--text-light); }
        .strength-strong { color: var(--danger); font-weight: 600; }
        .strength-medium { color: var(--warning); font-weight: 500; }
        .strength-weak { color: var(--text-muted); }
        .badge {
            display: inline-block; padding: 2px 10px;
            border-radius: 12px; font-size: 11.5px; font-weight: 500;
        }
        .badge-bullish { background: var(--danger-light); color: var(--danger); }
        .badge-bearish { background: var(--success-light); color: var(--success); }
        .badge-neutral { background: var(--border-light); color: var(--text-muted); }
        .badge-high { background: var(--danger-light); color: var(--danger); }
        .badge-medium { background: var(--warning-light); color: var(--warning); }
        .badge-low { background: var(--border-light); color: var(--text-muted); }
        .resonance-box {
            margin-top: 14px; padding: 14px 18px;
            background: var(--bg); border-left: 4px solid var(--primary);
            border-radius: 6px; font-size: 13.5px;
        }
        .resonance-box.bullish { border-left-color: var(--danger); }
        .resonance-box.bearish { border-left-color: var(--success); }
        /* 指标网格 */
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
        /* 支撑阻力 */
        .levels-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        @media (max-width: 768px) { .levels-grid { grid-template-columns: 1fr; } }
        .levels-block h4 { font-size: 14px; margin-bottom: 10px; }
        .levels-block.resistance h4 { color: var(--danger); }
        .levels-block.support h4 { color: var(--success); }
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
        /* 图表 */
        .chart-container { width: 100%; overflow-x: auto; margin-top: 4px; }
        .chart-placeholder {
            padding: 40px; text-align: center; color: var(--text-light);
            background: var(--bg); border-radius: 8px; font-size: 13px;
        }
        /* 龙虎榜/资金面卡片 */
        .info-card {
            background: var(--bg); padding: 14px 16px; border-radius: 8px;
            border: 1px solid var(--border-light); margin-bottom: 10px;
        }
        .info-card .info-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
        .info-card .info-value { font-size: 16px; font-weight: 700; }
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
        /* 免责声明 */
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
    <div class="subtitle">📈 技术面深度分析报告</div>
    <div class="meta">
        <span>📅 生成时间: {{ generated_at }}</span>
        <span>📊 数据截止: {{ data_date }}</span>
        <span>💰 当前价: {{ _fmt_price(current_price) }}</span>
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

<!-- 1. 报告概览（决策仪表盘风格） -->
<section class="section">
    <div class="section-title">报告概览</div>
    <div class="dashboard">
        <div class="dash-card">
            <div class="label">技术面评分</div>
            <div class="value {{ 'value-high' if scores.technical >= 70 else ('value-mid' if scores.technical >= 50 else 'value-low') }}">{{ scores.technical }}</div>
            <div class="sub">满分 100 / {{ scores.rating }}</div>
        </div>
        <div class="dash-card">
            <div class="label">趋势方向</div>
            <div class="value">
                {% set daily_tf = (multi_timeframe.timeframes.get('daily', {}) if multi_timeframe.timeframes else {}) %}
                {% set daily_trend = daily_tf.trend if daily_tf.trend else '未知' %}
                <span class="{{ trend_color.get(daily_trend, 'trend-unknown') }}">{{ daily_trend }}</span>
            </div>
            <div class="sub">{{ '强度: ' ~ (daily_tf.strength if daily_tf.strength else '无') }}</div>
        </div>
        <div class="dash-card">
            <div class="label">风险等级</div>
            <div class="value">
                {% if scores.technical >= 70 %}
                    <span class="value-low">低</span>
                {% elif scores.technical >= 50 %}
                    <span class="value-mid">中</span>
                {% else %}
                    <span class="value-high">高</span>
                {% endif %}
            </div>
            <div class="sub">基于技术面评分</div>
        </div>
    </div>
    <div class="indicator-grid" style="margin-top:12px;">
        <div class="indicator-card">
            <div class="ind-name">趋势方向 /30</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.trend }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">指标信号 /30</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.indicator }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">K线形态 /20</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.pattern }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">量价配合 /20</div>
            <div class="ind-value" style="font-size:18px;">{{ scores.breakdown.volume }}</div>
        </div>
    </div>
</section>

<!-- 2. 价格走势与K线形态 -->
<section class="section">
    <div class="section-title">价格走势与K线形态</div>
    {% if pattern_results and (pattern_results.bullish_count or pattern_results.bearish_count) %}
    <div class="indicator-grid" style="margin-bottom:18px;">
        <div class="indicator-card">
            <div class="ind-name">看涨形态</div>
            <div class="ind-value" style="color:var(--danger);">{{ pattern_results.bullish_count | default(0) }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">看跌形态</div>
            <div class="ind-value" style="color:var(--success);">{{ pattern_results.bearish_count | default(0) }}</div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">主导信号</div>
            <div class="ind-value" style="font-size:18px;">
                {% if pattern_results.dominant_signal == 'bullish' %}
                    <span class="badge badge-bullish" style="font-size:14px;padding:4px 14px;">偏多</span>
                {% elif pattern_results.dominant_signal == 'bearish' %}
                    <span class="badge badge-bearish" style="font-size:14px;padding:4px 14px;">偏空</span>
                {% else %}
                    <span class="badge badge-neutral" style="font-size:14px;padding:4px 14px;">中性</span>
                {% endif %}
            </div>
        </div>
        {% if limit_analysis.has_limit_data %}
        <div class="indicator-card">
            <div class="ind-name">涨跌停类型</div>
            <div class="ind-value" style="font-size:18px;">{{ limit_analysis.limit_type }}</div>
            <div class="ind-signal">
                {% if limit_analysis.recent_limit_ups > 0 %}
                    <span class="badge badge-bullish">近期{{ limit_analysis.recent_limit_ups }}次涨停</span>
                {% endif %}
                {% if limit_analysis.recent_limit_downs > 0 %}
                    <span class="badge badge-bearish">近期{{ limit_analysis.recent_limit_downs }}次跌停</span>
                {% endif %}
                {% if limit_analysis.near_limit %}
                    <span class="badge badge-high">接近涨停</span>
                {% endif %}
            </div>
        </div>
        {% endif %}
    </div>
    {% set patterns = pattern_results.recent_patterns if pattern_results.recent_patterns else [] %}
    {% if patterns %}
    <table>
        <thead>
            <tr><th>日期</th><th>形态名称</th><th>方向</th><th>可靠度</th></tr>
        </thead>
        <tbody>
        {% for p in patterns[:15] %}
            <tr>
                <td>{{ _fmt_date(p.date) }}</td>
                <td>{{ p.chinese_name | default(p.pattern_name, true) }}</td>
                <td>
                    {% if p.signal_type == 'bullish' %}<span class="badge badge-bullish">看涨</span>
                    {% elif p.signal_type == 'bearish' %}<span class="badge badge-bearish">看跌</span>
                    {% else %}<span class="badge badge-neutral">中性</span>{% endif %}
                </td>
                <td>
                    {% set rel = (p.reliability | default('medium', true)) %}
                    <span class="badge badge-{{ rel }}">{{ {'high':'高','medium':'中','low':'低'}.get(rel, rel) }}</span>
                </td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% if patterns | length > 15 %}
    <div style="margin-top:8px;font-size:12px;color:var(--text-light);">仅展示最近 15 条形态，共 {{ patterns | length }} 条</div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">近期未检测到明显K线形态信号</div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">无K线形态数据</div>
    {% endif %}
</section>

<!-- 3. 技术指标解读 -->
<section class="section">
    <div class="section-title">技术指标解读</div>
    <div class="indicator-grid">
        <div class="indicator-card">
            <div class="ind-name">MACD DIF/DEA</div>
            <div class="ind-value">{{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'macd_dif'), 3) }} / {{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'macd_dea'), 3) }}</div>
            <div class="ind-signal">
                {% set dif = _ind_val(technical_indicators, multi_timeframe, 'macd_dif') %}
                {% set dea = _ind_val(technical_indicators, multi_timeframe, 'macd_dea') %}
                {% if dif is not none and dea is not none and dif > dea %}
                    <span class="badge badge-bullish">金叉</span>
                {% elif dif is not none and dea is not none and dif < dea %}
                    <span class="badge badge-bearish">死叉</span>
                {% else %}
                    <span class="badge badge-neutral">—</span>
                {% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">RSI (14)</div>
            <div class="ind-value">{{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'rsi'), 1) }}</div>
            <div class="ind-signal">
                {% set rsi_val = _ind_val(technical_indicators, multi_timeframe, 'rsi') %}
                {% if rsi_val is not none and rsi_val > 70 %}<span class="badge badge-bearish">超买</span>
                {% elif rsi_val is not none and rsi_val < 30 %}<span class="badge badge-bullish">超卖</span>
                {% else %}<span class="badge badge-neutral">中性</span>{% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">KDJ (K/D/J)</div>
            <div class="ind-value">{{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'kdj_k'), 1) }} / {{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'kdj_d'), 1) }} / {{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'kdj_j'), 1) }}</div>
            <div class="ind-signal">
                {% set j_val = _ind_val(technical_indicators, multi_timeframe, 'kdj_j') %}
                {% if j_val is not none and j_val > 100 %}<span class="badge badge-bearish">超买</span>
                {% elif j_val is not none and j_val < 0 %}<span class="badge badge-bullish">超卖</span>
                {% else %}<span class="badge badge-neutral">中性</span>{% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">布林带位置</div>
            <div class="ind-value">{{ _fmt_pct(_ind_val(technical_indicators, multi_timeframe, 'boll_position')) }}</div>
            <div class="ind-signal">
                {% set boll = _ind_val(technical_indicators, multi_timeframe, 'boll_position') %}
                {% if boll is not none and boll > 0.9 %}<span class="badge badge-bearish">触及上轨</span>
                {% elif boll is not none and boll < 0.1 %}<span class="badge badge-bullish">触及下轨</span>
                {% else %}<span class="badge badge-neutral">正常区间</span>{% endif %}
            </div>
        </div>
    </div>
</section>

<!-- 4. 均线系统与趋势 -->
<section class="section">
    <div class="section-title">均线系统与趋势（多周期共振）</div>
    <table>
        <thead>
            <tr>
                <th>周期</th><th>趋势</th><th>强度</th><th>收盘价</th>
                <th>MA5</th><th>MA20</th><th>MA60</th><th>关键信号</th>
            </tr>
        </thead>
        <tbody>
        {% for tf_key, tf_label in [('daily', '日线'), ('weekly', '周线'), ('monthly', '月线')] %}
            {% set tf = multi_timeframe.timeframes.get(tf_key, {}) if multi_timeframe.timeframes else {} %}
            {% set ind = tf.indicators if tf.indicators else {} %}
            <tr>
                <td><strong>{{ tf_label }}</strong></td>
                <td><span class="{{ trend_color.get(tf.trend, 'trend-unknown') }}">{{ tf.trend | default('未知', true) }}</span></td>
                <td><span class="{{ strength_color.get(tf.strength, 'strength-none') }}">{{ tf.strength | default('无', true) }}</span></td>
                <td>{{ _fmt_price(ind.close) }}</td>
                <td>{{ _fmt_price(ind.ma5) }}</td>
                <td>{{ _fmt_price(ind.ma20) }}</td>
                <td>{{ _fmt_price(ind.ma60) }}</td>
                <td>
                    {% for sig in (tf.signals if tf.signals else []) %}
                        <span class="badge {% if '金叉' in sig.type or '红柱' in sig.type or '超卖' in sig.type %}badge-bullish{% elif '死叉' in sig.type or '绿柱' in sig.type or '超买' in sig.type %}badge-bearish{% else %}badge-neutral{% endif %}" style="margin-bottom:3px;display:inline-block;">{{ sig.type }}</span>
                    {% else %}
                        <span style="color:var(--text-light);">—</span>
                    {% endfor %}
                </td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% if resonance %}
    <div class="resonance-box {{ 'bullish' if resonance.bullish else ('bearish' if resonance.bearish else '') }}">
        <strong>多周期共振：</strong>{{ resonance.description | default('无明确共振信号', true) }}
    </div>
    {% endif %}
    {% if divergences %}
    <div style="margin-top:14px;">
        <strong style="font-size:13px;color:var(--text-muted);">检测到的背离信号：</strong>
        <ul style="margin-top:8px;padding-left:20px;font-size:13px;">
        {% for div in divergences %}
            <li>{{ div.description | default(div.type ~ ' - ' ~ div.indicator, true) }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
</section>

<!-- 5. 支撑阻力位 -->
<section class="section">
    <div class="section-title">支撑阻力位</div>
    {% if support_resistance and (support_resistance.resistance or support_resistance.support) %}
    <div class="levels-grid">
        <div class="levels-block resistance">
            <h4>▲ 阻力位 (由近及远)</h4>
            <table>
                <thead><tr><th>价格</th><th>类型</th><th>强度</th><th>方法</th></tr></thead>
                <tbody>
                {% for r in support_resistance.resistance %}
                    <tr>
                        <td><strong>{{ _fmt_price(r.price) }}</strong></td>
                        <td>{{ r.type | default('—', true) }}</td>
                        <td><span class="badge badge-{{ {'很强':'high','强':'high','中':'medium','弱':'low'}.get(r.strength, 'medium') }}">{{ r.strength | default('中', true) }}</span></td>
                        <td>{{ r.method | default('—', true) }}</td>
                    </tr>
                {% else %}
                    <tr><td colspan="4" style="text-align:center;color:var(--text-light);">暂无阻力位</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        <div class="levels-block support">
            <h4>▼ 支撑位 (由近及远)</h4>
            <table>
                <thead><tr><th>价格</th><th>类型</th><th>强度</th><th>方法</th></tr></thead>
                <tbody>
                {% for s in support_resistance.support %}
                    <tr>
                        <td><strong>{{ _fmt_price(s.price) }}</strong></td>
                        <td>{{ s.type | default('—', true) }}</td>
                        <td><span class="badge badge-{{ {'很强':'high','强':'high','中':'medium','弱':'low'}.get(s.strength, 'medium') }}">{{ s.strength | default('中', true) }}</span></td>
                        <td>{{ s.method | default('—', true) }}</td>
                    </tr>
                {% else %}
                    <tr><td colspan="4" style="text-align:center;color:var(--text-light);">暂无支撑位</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% else %}
    <div class="chart-placeholder">暂无支撑阻力位数据</div>
    {% endif %}
</section>

<!-- 6. 量价关系 -->
<section class="section">
    <div class="section-title">量价关系</div>
    <div class="indicator-grid">
        <div class="indicator-card">
            <div class="ind-name">上涨/下跌日量比</div>
            <div class="ind-value">{{ _fmt_num(technical_indicators.up_down_volume_ratio, 2) }}</div>
            <div class="ind-signal">
                {% set vr = technical_indicators.up_down_volume_ratio %}
                {% if vr is not none and vr > 1.4 %}<span class="badge badge-bullish">放量上涨</span>
                {% elif vr is not none and vr < 0.8 %}<span class="badge badge-bearish">放量下跌</span>
                {% else %}<span class="badge badge-neutral">量价平衡</span>{% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">成交量趋势</div>
            <div class="ind-value" style="font-size:18px;">
                {% set vt = technical_indicators.volume_trend %}
                {% if vt == 'increasing' %}放量
                {% elif vt == 'decreasing' %}缩量
                {% elif vt == 'stable' %}平稳
                {% else %}未知{% endif %}
            </div>
        </div>
    </div>
</section>

<!-- 7. 资金面分析（A股特色） -->
<section class="section">
    <div class="section-title">资金面分析</div>
    {% if capital_flow %}
    <div class="indicator-grid">
        {% if capital_flow.main_net_inflow is not none %}
        <div class="indicator-card">
            <div class="ind-name">主力净流入</div>
            <div class="ind-value {{ 'value-high' if capital_flow.main_net_inflow > 0 else 'value-low' }}">
                {{ _fmt_market_cap(capital_flow.main_net_inflow) }}
            </div>
        </div>
        {% endif %}
        {% if capital_flow.north_net_inflow is not none %}
        <div class="indicator-card">
            <div class="ind-name">北向资金净流入</div>
            <div class="ind-value {{ 'value-high' if capital_flow.north_net_inflow > 0 else 'value-low' }}">
                {{ _fmt_market_cap(capital_flow.north_net_inflow) }}
            </div>
            <div class="ind-signal"><span class="badge badge-neutral">A股特色指标</span></div>
        </div>
        {% endif %}
        {% if capital_flow.main_net_inflow_5d is not none %}
        <div class="indicator-card">
            <div class="ind-name">5日主力净流入</div>
            <div class="ind-value {{ 'value-high' if capital_flow.main_net_inflow_5d > 0 else 'value-low' }}">
                {{ _fmt_market_cap(capital_flow.main_net_inflow_5d) }}
            </div>
        </div>
        {% endif %}
    </div>
    {% else %}
    <div class="chart-placeholder">资金面数据暂缺（需接入 akshare 资金流接口）</div>
    {% endif %}
</section>

<!-- 8. 龙虎榜与大单（A股特色） -->
<section class="section">
    <div class="section-title">龙虎榜与大单</div>
    {% if dragon_tiger and dragon_tiger.has_data %}
    <table>
        <thead>
            <tr><th>日期</th><th>上榜原因</th><th>净买入</th><th>买入额</th><th>卖出额</th></tr>
        </thead>
        <tbody>
        {% for item in dragon_tiger.records %}
            <tr>
                <td>{{ _fmt_date(item.date) }}</td>
                <td>{{ item.reason | default('—', true) }}</td>
                <td><strong class="{{ 'value-low' if item.net_buy > 0 else 'value-high' }}">{{ _fmt_market_cap(item.net_buy) }}</strong></td>
                <td>{{ _fmt_market_cap(item.buy_amount) }}</td>
                <td>{{ _fmt_market_cap(item.sell_amount) }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% if dragon_tiger.institutional_buy is not none %}
    <div class="info-card" style="margin-top:14px;">
        <div class="info-title">机构席位动向</div>
        <div class="info-value">
            机构净买入: <span class="{{ 'value-low' if dragon_tiger.institutional_buy > 0 else 'value-high' }}">
                {{ _fmt_market_cap(dragon_tiger.institutional_buy) }}
            </span>
        </div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">龙虎榜数据暂缺（需接入 akshare 龙虎榜接口，近5日无上榜记录也属正常）</div>
    {% endif %}
</section>

<!-- 9. 技术面深度解读（LLM 分析师章节） -->
<section class="section llm-section">
    <div class="section-title">技术面深度解读</div>
<!--LLM_TECHNICAL_ANALYSIS_PLACEHOLDER-->
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
    <span style="font-size:11px;">Generated by jingnitrader · 技术面分析报告 · {{ generated_at }}</span>
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
        render_context["_ind_val"] = self._tmpl_ind_val
        return template.render(**render_context)
