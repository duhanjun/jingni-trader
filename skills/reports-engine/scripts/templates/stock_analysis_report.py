"""
个股深度分析报告生成器
整合技术面、基本面、K线形态、支撑阻力位、多周期分析，
生成完整的HTML个股分析报告
"""
import os
import html as _html
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
from datetime import datetime
from jinja2 import Template

logger = logging.getLogger("stock_analysis_report")


class StockAnalysisReportGenerator:
    """个股深度分析报告生成器"""

    # 评分等级阈值
    _RATING_THRESHOLDS = [
        (80, "强烈推荐"),
        (70, "推荐"),
        (60, "中性偏多"),
        (50, "中性"),
        (40, "中性偏空"),
        (30, "回避"),
        (0, "强烈回避"),
    ]

    # 趋势/强度对应的颜色类
    _TREND_COLOR = {
        "上涨": "trend-up",
        "下跌": "trend-down",
        "震荡": "trend-neutral",
        "未知": "trend-unknown",
    }
    _STRENGTH_COLOR = {
        "强": "strength-strong",
        "中": "strength-medium",
        "弱": "strength-weak",
        "无": "strength-none",
    }

    def __init__(self):
        # 防御性导入: charts 模块可能在后续任务中创建
        self.kline_gen = None
        self.indicator_gen = None
        self.fundamental_gen = None
        try:
            from ..charts.kline_chart import KlineChartGenerator
            self.kline_gen = KlineChartGenerator()
        except Exception as e:
            logger.warning(f"KlineChartGenerator 不可用: {e}")
        try:
            from ..charts.indicator_chart import IndicatorChartGenerator
            self.indicator_gen = IndicatorChartGenerator()
        except Exception as e:
            logger.warning(f"IndicatorChartGenerator 不可用: {e}")
        try:
            from ..charts.fundamental_dashboard import FundamentalDashboardGenerator
            self.fundamental_gen = FundamentalDashboardGenerator()
        except Exception as e:
            logger.warning(f"FundamentalDashboardGenerator 不可用: {e}")

    # ================================================================
    # 公共入口
    # ================================================================

    def generate(self,
                 stock_code: str,
                 stock_name: str,
                 ohlcv_data: pd.DataFrame,
                 technical_indicators: Dict,
                 pattern_results: Dict,
                 support_resistance: Dict,
                 multi_timeframe: Dict,
                 fundamental_data: Optional[Dict] = None,
                 output_path: str = None,
                 llm_prompts: Optional[Dict] = None) -> str:
        """
        生成个股深度分析报告

        参数:
            stock_code: 股票代码
            stock_name: 股票名称
            ohlcv_data: OHLCV数据
            technical_indicators: 技术指标数据 (MACD, RSI, KDJ, MA等)
            pattern_results: K线形态识别结果
            support_resistance: 支撑阻力位
            multi_timeframe: 多周期分析结果
            fundamental_data: 基本面数据 (可选)
            output_path: 输出文件路径

        返回:
            HTML报告文件路径
        """
        logger.info(f"开始生成 {stock_name}({stock_code}) 深度分析报告")

        # 防御性初始化
        technical_indicators = technical_indicators or {}
        pattern_results = pattern_results or {}
        support_resistance = support_resistance or {}
        multi_timeframe = multi_timeframe or {}

        # 1. 将 ohlcv 的成交量指标注入 technical_indicators, 供量价评分使用
        enriched_indicators: Dict[str, Any] = dict(technical_indicators)
        volume_metrics = self._compute_volume_metrics(ohlcv_data)
        enriched_indicators.update(volume_metrics)

        # 2. 计算综合评分
        scores = self._calc_comprehensive_score(
            enriched_indicators, pattern_results, multi_timeframe, fundamental_data
        )

        # 3. 生成风险提示
        risk_warnings = self._generate_risk_warnings(
            enriched_indicators, pattern_results, multi_timeframe, support_resistance
        )

        # 4. 生成图表 (TradingView lightweight-charts)
        # K线+成交量+MACD+RSI+KDJ 五合一联动图
        kline_chart_html = self._safe_render_chart(
            self.kline_gen, "generate_tradingview_chart",
            ohlcv_data, stock_code=stock_code, stock_name=stock_name,
            support_resistance=support_resistance,
            fallback="K线图暂不可用"
        )
        # 技术指标图不再单独生成（已嵌入K线联动图）
        indicator_chart_html = ""
        fundamental_chart_html = ""
        if fundamental_data:
            # 从 fundamental_data 提取估值数据用于仪表盘
            valuation_dict = {}
            for k in ("pe_ttm", "pb", "ps_ttm", "dv_ratio"):
                if k in fundamental_data:
                    valuation_dict[k] = fundamental_data[k]
            # 估值分位数据
            for k in ("pe_percentile", "pb_percentile"):
                if k in fundamental_data:
                    valuation_dict[k] = fundamental_data[k]
            # 生成估值仪表盘（单组件）
            if valuation_dict:
                fundamental_chart_html = self._safe_render_chart(
                    self.fundamental_gen, "generate_combined_dashboard",
                    valuation_dict, None, {},
                    fallback="基本面图表暂不可用"
                )

        # 5. 提取数据截止日期
        data_date = self._get_data_date(ohlcv_data)

        # 6. 组装模板上下文
        context = {
            "stock_code": _html.escape(str(stock_code)),
            "stock_name": _html.escape(str(stock_name)),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_date": data_date,
            "scores": scores,
            "multi_timeframe": multi_timeframe,
            "technical_indicators": enriched_indicators,
            "pattern_results": pattern_results,
            "support_resistance": support_resistance,
            "fundamental_data": fundamental_data,
            "risk_warnings": risk_warnings,
            "kline_chart_html": kline_chart_html,
            "indicator_chart_html": indicator_chart_html,
            "fundamental_chart_html": fundamental_chart_html,
            "trend_color": self._TREND_COLOR,
            "strength_color": self._STRENGTH_COLOR,
            "divergences": multi_timeframe.get("divergences", []),
            "resonance": multi_timeframe.get("resonance", {}),
            "tf_summary": multi_timeframe.get("summary", ""),
            # LLM 分析师 prompt 标记（供 agent 端识别并替换占位符）
            "has_llm_prompts": bool(llm_prompts),
        }

        # 7. 渲染 HTML
        html_content = self._render_html(context)

        # 8. 写入文件
        if output_path:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"报告已保存: {output_path}")
            return output_path

        return html_content

    # ================================================================
    # 综合评分
    # ================================================================

    def _calc_comprehensive_score(self, technical_indicators: Dict,
                                  pattern_results: Dict,
                                  multi_timeframe: Dict,
                                  fundamental_data: Optional[Dict] = None) -> Dict:
        """
        计算综合评分

        技术面评分 (0-100):
        - 趋势方向 (30分): 多周期共振 + 均线排列
        - 指标信号 (30分): MACD/RSI/KDJ信号
        - K线形态 (20分): 近期看涨/看跌形态
        - 量价配合 (20分): 成交量变化

        基本面评分 (0-100):
        - 估值水平 (35分): PE/PB分位
        - 盈利能力 (35分): ROE/毛利率
        - 成长性 (30分): 营收/利润增速

        综合评分 = 技术面 * 0.5 + 基本面 * 0.5
        """
        # ── 技术面评分 ──────────────────────────────
        trend_score = self._score_trend(multi_timeframe, technical_indicators)
        indicator_score = self._score_indicator_signals(technical_indicators, multi_timeframe)
        pattern_score = self._score_patterns(pattern_results)
        volume_score = self._score_volume_price(technical_indicators)

        technical_total = trend_score + indicator_score + pattern_score + volume_score
        technical_total = float(min(100.0, max(0.0, technical_total)))

        # ── 基本面评分 ──────────────────────────────
        fundamental_total = None
        valuation_score = None
        profitability_score = None
        growth_score = None
        if fundamental_data:
            valuation_score = self._score_valuation(fundamental_data)
            profitability_score = self._score_profitability(fundamental_data)
            growth_score = self._score_growth(fundamental_data)
            fundamental_total = float(min(100.0, max(0.0,
                valuation_score + profitability_score + growth_score
            )))

        # ── 综合评分 ──────────────────────────────
        if fundamental_total is not None:
            comprehensive = technical_total * 0.5 + fundamental_total * 0.5
        else:
            comprehensive = technical_total
        comprehensive = float(min(100.0, max(0.0, comprehensive)))

        rating = self._rating(comprehensive)

        return {
            "technical": round(technical_total, 1),
            "fundamental": round(fundamental_total, 1) if fundamental_total is not None else None,
            "comprehensive": round(comprehensive, 1),
            "rating": rating,
            "has_fundamental": fundamental_total is not None,
            "breakdown": {
                "trend": round(trend_score, 1),
                "indicator": round(indicator_score, 1),
                "pattern": round(pattern_score, 1),
                "volume": round(volume_score, 1),
                "valuation": round(valuation_score, 1) if valuation_score is not None else None,
                "profitability": round(profitability_score, 1) if profitability_score is not None else None,
                "growth": round(growth_score, 1) if growth_score is not None else None,
            },
        }

    def _score_trend(self, multi_timeframe: Dict, technical_indicators: Dict) -> float:
        """趋势方向评分 (满分 30): 多周期共振 + 均线排列"""
        score = 0.0
        resonance = multi_timeframe.get("resonance", {}) if multi_timeframe else {}

        # 多周期共振 (0-22分)
        if resonance.get("all_bullish"):
            score += 22.0
        elif resonance.get("bullish"):
            score += 16.0
        elif resonance.get("all_bearish"):
            score += 0.0
        elif resonance.get("bearish"):
            score += 6.0
        else:
            score += 11.0  # 中性

        # 均线排列 (0-8分): MA5 > MA20 > MA60 为多头排列
        ma5 = self._get_indicator(technical_indicators, multi_timeframe, "ma5")
        ma20 = self._get_indicator(technical_indicators, multi_timeframe, "ma20")
        ma60 = self._get_indicator(technical_indicators, multi_timeframe, "ma60")
        if all(v is not None and pd.notna(v) for v in [ma5, ma20, ma60]):
            if ma5 > ma20 > ma60:
                score += 8.0
            elif ma5 < ma20 < ma60:
                score += 0.0
            elif ma5 > ma20:
                score += 4.0
            else:
                score += 2.0
        else:
            score += 4.0  # 数据缺失, 给中性分

        return min(30.0, score)

    def _score_indicator_signals(self, technical_indicators: Dict, multi_timeframe: Dict) -> float:
        """指标信号评分 (满分 30): MACD/RSI/KDJ 信号"""
        score = 0.0

        # MACD (0-12分)
        macd_dif = self._get_indicator(technical_indicators, multi_timeframe, "macd_dif")
        macd_dea = self._get_indicator(technical_indicators, multi_timeframe, "macd_dea")
        macd_hist = self._get_indicator(technical_indicators, multi_timeframe, "macd_hist")
        if macd_dif is not None and macd_dea is not None and pd.notna(macd_dif) and pd.notna(macd_dea):
            if macd_dif > macd_dea:
                score += 9.0  # 金叉状态
                if macd_dif > 0:
                    score += 3.0  # 零轴上方
            else:
                score += 2.0  # 死叉状态
                if macd_dif < 0:
                    score += 0.0
                else:
                    score += 1.0
        elif macd_hist is not None and pd.notna(macd_hist):
            score += 6.0 if macd_hist > 0 else 3.0
        else:
            score += 6.0  # 数据缺失, 中性

        # RSI (0-9分)
        rsi = self._get_indicator(technical_indicators, multi_timeframe, "rsi")
        if rsi is not None and pd.notna(rsi):
            rsi = float(rsi)
            if rsi < 30:
                score += 9.0  # 超卖, 反弹机会
            elif rsi < 45:
                score += 7.0
            elif rsi < 55:
                score += 5.0  # 中性
            elif rsi < 70:
                score += 3.0
            else:
                score += 1.0  # 超买, 回调风险
        else:
            score += 4.5

        # KDJ (0-9分)
        kdj_j = self._get_indicator(technical_indicators, multi_timeframe, "kdj_j")
        kdj_k = self._get_indicator(technical_indicators, multi_timeframe, "kdj_k")
        kdj_d = self._get_indicator(technical_indicators, multi_timeframe, "kdj_d")
        if kdj_k is not None and kdj_d is not None and pd.notna(kdj_k) and pd.notna(kdj_d):
            if kdj_k > kdj_d:
                score += 6.0  # 金叉
                if kdj_j is not None and pd.notna(kdj_j) and kdj_j < 20:
                    score += 3.0  # 低位金叉
                elif kdj_j is not None and pd.notna(kdj_j) and kdj_j > 100:
                    score += 0.0  # 高位, 虽金叉但风险高
                else:
                    score += 1.5
            else:
                score += 1.0  # 死叉
                if kdj_j is not None and pd.notna(kdj_j) and kdj_j > 100:
                    score += 0.0
                elif kdj_j is not None and pd.notna(kdj_j) and kdj_j < 0:
                    score += 3.0  # 超卖区死叉, 可能接近底部
                else:
                    score += 1.0
        else:
            score += 4.5

        return min(30.0, score)

    def _score_patterns(self, pattern_results: Dict) -> float:
        """K线形态评分 (满分 20): 近期看涨/看跌形态"""
        if not pattern_results:
            return 10.0  # 中性

        bullish_count = int(pattern_results.get("bullish_count", 0))
        bearish_count = int(pattern_results.get("bearish_count", 0))
        dominant = pattern_results.get("dominant_signal", "neutral")

        total = bullish_count + bearish_count
        if total == 0:
            return 10.0

        bull_ratio = bullish_count / total

        if dominant == "bullish":
            base = 16.0
        elif dominant == "bearish":
            base = 4.0
        else:
            base = 10.0

        # 根据比例微调
        adjustment = (bull_ratio - 0.5) * 8.0
        score = base + adjustment
        return float(min(20.0, max(0.0, score)))

    def _score_volume_price(self, technical_indicators: Dict) -> float:
        """量价配合评分 (满分 20): 成交量变化"""
        # 优先使用预计算的 up_down_volume_ratio
        ratio = technical_indicators.get("up_down_volume_ratio")
        if ratio is None or not pd.notna(ratio):
            return 10.0  # 中性

        ratio = float(ratio)
        # ratio > 1 表示上涨日成交量大于下跌日
        if ratio > 1.8:
            return 20.0
        elif ratio > 1.4:
            return 17.0
        elif ratio > 1.1:
            return 14.0
        elif ratio > 0.9:
            return 10.0
        elif ratio > 0.6:
            return 6.0
        else:
            return 3.0

    def _score_valuation(self, fundamental_data: Dict) -> float:
        """估值水平评分 (满分 35): PE/PB分位"""
        if not fundamental_data:
            return 17.5

        pe_pct = self._get_fundamental(fundamental_data, "pe_percentile", "pe_pct", "pe_quantile")
        pb_pct = self._get_fundamental(fundamental_data, "pb_percentile", "pb_pct", "pb_quantile")

        scores = []
        for pct in [pe_pct, pb_pct]:
            if pct is not None and pd.notna(pct):
                pct = float(pct)
                if pct < 0.2:
                    scores.append(17.5)
                elif pct < 0.4:
                    scores.append(14.0)
                elif pct < 0.6:
                    scores.append(10.5)
                elif pct < 0.8:
                    scores.append(7.0)
                else:
                    scores.append(3.5)

        if not scores:
            # 无分位数据时用 PE 绝对值粗略判断
            pe = self._get_fundamental(fundamental_data, "pe", "pe_ttm", "pe_ratio")
            if pe is not None and pd.notna(pe):
                pe = float(pe)
                if pe < 0:
                    return 5.0  # 亏损
                elif pe < 15:
                    return 28.0
                elif pe < 25:
                    return 21.0
                elif pe < 40:
                    return 14.0
                elif pe < 60:
                    return 7.0
                else:
                    return 3.5
            return 17.5

        return float(np.mean(scores))

    def _score_profitability(self, fundamental_data: Dict) -> float:
        """盈利能力评分 (满分 35): ROE/毛利率"""
        if not fundamental_data:
            return 17.5

        roe = self._get_fundamental(fundamental_data, "roe", "roe_ttm", "return_on_equity")
        gross_margin = self._get_fundamental(fundamental_data, "gross_margin", "毛利率")

        scores = []

        if roe is not None and pd.notna(roe):
            roe = float(roe)
            # ROE 以百分比或小数传入都能处理
            if abs(roe) < 1:
                roe = roe * 100
            if roe > 20:
                scores.append(22.0)
            elif roe > 15:
                scores.append(18.0)
            elif roe > 10:
                scores.append(14.0)
            elif roe > 5:
                scores.append(9.0)
            elif roe > 0:
                scores.append(5.0)
            else:
                scores.append(0.0)

        if gross_margin is not None and pd.notna(gross_margin):
            gm = float(gross_margin)
            if abs(gm) < 1:
                gm = gm * 100
            if gm > 60:
                scores.append(13.0)
            elif gm > 40:
                scores.append(10.0)
            elif gm > 25:
                scores.append(7.0)
            elif gm > 15:
                scores.append(4.0)
            else:
                scores.append(2.0)

        if not scores:
            return 17.5

        # ROE 权重高于毛利率
        if len(scores) == 2:
            return float(scores[0] * (22.0 / 35.0) + scores[1] * (13.0 / 35.0))
        # 单项时按比例还原到35分制
        if roe is not None and pd.notna(roe):
            return float(scores[0] * (35.0 / 22.0))
        return float(scores[0] * (35.0 / 13.0))

    def _score_growth(self, fundamental_data: Dict) -> float:
        """成长性评分 (满分 30): 营收/利润增速"""
        if not fundamental_data:
            return 15.0

        rev_growth = self._get_fundamental(fundamental_data, "revenue_growth", "rev_growth", "营收增速")
        profit_growth = self._get_fundamental(fundamental_data, "profit_growth", "net_profit_growth", "利润增速")

        scores = []

        for g in [rev_growth, profit_growth]:
            if g is not None and pd.notna(g):
                g = float(g)
                if abs(g) < 1:
                    g = g * 100
                if g > 30:
                    scores.append(15.0)
                elif g > 15:
                    scores.append(11.0)
                elif g > 5:
                    scores.append(8.0)
                elif g > 0:
                    scores.append(5.0)
                elif g > -10:
                    scores.append(3.0)
                else:
                    scores.append(0.0)

        if not scores:
            return 15.0

        return float(np.mean(scores))

    # ================================================================
    # 风险提示
    # ================================================================

    def _generate_risk_warnings(self, technical_indicators: Dict,
                                pattern_results: Dict,
                                multi_timeframe: Dict,
                                support_resistance: Dict) -> List[str]:
        """生成风险提示列表"""
        warnings: List[str] = []
        technical_indicators = technical_indicators or {}
        pattern_results = pattern_results or {}
        multi_timeframe = multi_timeframe or {}
        support_resistance = support_resistance or {}

        # 1. 多周期看空共振
        resonance = multi_timeframe.get("resonance", {})
        if resonance.get("all_bearish"):
            warnings.append("多周期共振看空，日/周/月线全部下跌，趋势明确转弱，建议谨慎")
        elif resonance.get("bearish"):
            warnings.append("日线与月线同步走弱，中期趋势偏空")

        # 2. 顶背离信号
        divergences = multi_timeframe.get("divergences", [])
        top_divs = [d for d in divergences if d.get("type") == "顶背离"]
        if top_divs:
            descs = "；".join(d.get("description", "") for d in top_divs[:3])
            warnings.append(f"检测到顶背离信号 ({descs})，价格创新高但指标未跟上，短期可能面临调整压力")

        # 3. RSI 超买
        rsi = self._get_indicator(technical_indicators, multi_timeframe, "rsi")
        if rsi is not None and pd.notna(rsi):
            if float(rsi) > 80:
                warnings.append(f"RSI={float(rsi):.1f}，处于严重超买区，回调风险较大")
            elif float(rsi) > 70:
                warnings.append(f"RSI={float(rsi):.1f}，处于超买区，注意短期回调")

        # 4. 布林带触及上轨
        boll_pos = self._get_indicator(technical_indicators, multi_timeframe, "boll_position")
        if boll_pos is not None and pd.notna(boll_pos):
            if float(boll_pos) > 0.95:
                warnings.append("价格触及布林带上轨，短期存在回归中轨的压力")

        # 5. KDJ 超买
        kdj_j = self._get_indicator(technical_indicators, multi_timeframe, "kdj_j")
        kdj_k = self._get_indicator(technical_indicators, multi_timeframe, "kdj_k")
        if kdj_j is not None and pd.notna(kdj_j) and float(kdj_j) > 100:
            warnings.append(f"KDJ的J值={float(kdj_j):.1f}，处于超买区，存在技术性回调风险")
        elif kdj_k is not None and pd.notna(kdj_k) and float(kdj_k) > 80:
            warnings.append(f"KDJ的K值={float(kdj_k):.1f}，处于高位，注意短期波动")

        # 6. MACD 死叉
        macd_dif = self._get_indicator(technical_indicators, multi_timeframe, "macd_dif")
        macd_dea = self._get_indicator(technical_indicators, multi_timeframe, "macd_dea")
        if macd_dif is not None and macd_dea is not None and pd.notna(macd_dif) and pd.notna(macd_dea):
            if macd_dif < macd_dea:
                if macd_dif < 0:
                    warnings.append("MACD死叉且处于零轴下方，趋势偏空")
                else:
                    warnings.append("MACD死叉，短期趋势转弱")

        # 7. K线形态偏空
        dominant = pattern_results.get("dominant_signal", "neutral")
        bearish_count = int(pattern_results.get("bearish_count", 0))
        if dominant == "bearish" and bearish_count > 0:
            warnings.append(f"近期K线形态偏空，检测到{bearish_count}个看跌形态，注意下跌风险")

        # 8. 接近阻力位
        current_price = support_resistance.get("current_price")
        nearest_resistance = support_resistance.get("nearest_resistance")
        if current_price and nearest_resistance and current_price > 0:
            distance = (nearest_resistance - current_price) / current_price
            if 0 < distance < 0.03:
                warnings.append(f"价格接近阻力位 {nearest_resistance:.2f} (距现价 {distance*100:.1f}%)，注意上方压力")

        # 9. 跌破支撑位
        nearest_support = support_resistance.get("nearest_support")
        if current_price and nearest_support and current_price > 0:
            if current_price < nearest_support:
                warnings.append(f"价格已跌破最近支撑位 {nearest_support:.2f}，下方支撑失效")

        # 10. 估值过高 (基本面)
        pe_pct = self._get_fundamental(technical_indicators, "pe_percentile", "pe_pct")
        if pe_pct is not None and pd.notna(pe_pct) and float(pe_pct) > 0.8:
            warnings.append(f"PE估值处于历史{float(pe_pct)*100:.0f}%分位，估值偏高，注意估值回归风险")

        # 11. 业绩下滑 (若有基本面数据传入 technical_indicators)
        profit_growth = self._get_fundamental(technical_indicators, "profit_growth", "net_profit_growth")
        if profit_growth is not None and pd.notna(profit_growth):
            pg = float(profit_growth)
            if abs(pg) < 1:
                pg = pg * 100
            if pg < -15:
                warnings.append(f"净利润增速={pg:.1f}%，业绩明显下滑，关注基本面恶化风险")

        # 12. 量价背离
        vol_ratio = technical_indicators.get("up_down_volume_ratio")
        if vol_ratio is not None and pd.notna(vol_ratio):
            # 价格上涨但下跌日成交量更大 → 量价背离
            daily_trend = multi_timeframe.get("timeframes", {}).get("daily", {}).get("trend", "")
            if daily_trend == "上涨" and float(vol_ratio) < 0.8:
                warnings.append("价格上涨但下跌日成交量显著放大，存在量价背离，上涨持续性存疑")

        if not warnings:
            warnings.append("暂无重大风险信号提示，但仍需关注市场整体环境变化")

        return warnings

    # ================================================================
    # HTML 渲染
    # ================================================================

    def _render_html(self, context: Dict) -> str:
        """使用Jinja2模板渲染HTML报告"""
        template_str = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ stock_name }} ({{ stock_code }}) 深度分析报告</title>
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
        .report-header h1 {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }
        .report-header .subtitle {
            font-size: 15px;
            opacity: 0.92;
            margin-bottom: 14px;
        }
        .report-header .meta {
            font-size: 12.5px;
            opacity: 0.78;
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
        }
        .report-header .meta span { display: inline-flex; align-items: center; }
        .section {
            background: var(--card-bg);
            border-radius: var(--radius);
            padding: 24px 28px;
            margin-bottom: 20px;
            box-shadow: var(--shadow);
            border: 1px solid var(--border-light);
        }
        .section-title {
            font-size: 17px;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 18px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .section-title::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 18px;
            background: var(--primary);
            border-radius: 2px;
        }
        /* 评分卡片 */
        .score-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }
        .score-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .score-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
        }
        .score-card.technical::before { background: var(--primary); }
        .score-card.fundamental::before { background: var(--success); }
        .score-card.comprehensive::before { background: var(--warning); }
        .score-card .score-label {
            font-size: 13px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }
        .score-card .score-value {
            font-size: 36px;
            font-weight: 700;
            line-height: 1.1;
        }
        .score-card .score-value.score-high { color: var(--success); }
        .score-card .score-value.score-mid { color: var(--warning); }
        .score-card .score-value.score-low { color: var(--danger); }
        .score-card .score-rating {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 6px;
        }
        .score-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px dashed var(--border);
        }
        .breakdown-item {
            text-align: center;
            font-size: 12px;
        }
        .breakdown-item .label { color: var(--text-muted); }
        .breakdown-item .value {
            font-size: 16px;
            font-weight: 600;
            color: var(--text);
            margin-top: 2px;
        }
        .breakdown-item .value.score-high { color: var(--success); }
        .breakdown-item .value.score-low { color: var(--danger); }
        /* 表格 */
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 4px;
            font-size: 13px;
        }
        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-light);
        }
        th {
            background: var(--bg);
            font-weight: 600;
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        tbody tr:hover { background: var(--border-light); }
        .trend-up { color: var(--danger); font-weight: 600; }
        .trend-down { color: var(--success); font-weight: 600; }
        .trend-neutral { color: var(--text-muted); font-weight: 500; }
        .trend-unknown { color: var(--text-light); }
        .strength-strong { color: var(--danger); font-weight: 600; }
        .strength-medium { color: var(--warning); font-weight: 500; }
        .strength-weak { color: var(--text-muted); }
        .strength-none { color: var(--text-light); }
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11.5px;
            font-weight: 500;
        }
        .badge-bullish { background: var(--danger-light); color: var(--danger); }
        .badge-bearish { background: var(--success-light); color: var(--success); }
        .badge-neutral { background: var(--border-light); color: var(--text-muted); }
        .badge-high { background: var(--danger-light); color: var(--danger); }
        .badge-medium { background: var(--warning-light); color: var(--warning); }
        .badge-low { background: var(--border-light); color: var(--text-muted); }
        .resonance-box {
            margin-top: 14px;
            padding: 14px 18px;
            background: var(--bg);
            border-left: 4px solid var(--primary);
            border-radius: 6px;
            font-size: 13.5px;
            color: var(--text);
        }
        .resonance-box.bullish { border-left-color: var(--danger); }
        .resonance-box.bearish { border-left-color: var(--success); }
        /* 指标网格 */
        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
        }
        .indicator-card {
            background: var(--bg);
            border-radius: 8px;
            padding: 14px 16px;
            border: 1px solid var(--border-light);
        }
        .indicator-card .ind-name {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .indicator-card .ind-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--text);
        }
        .indicator-card .ind-signal {
            font-size: 11.5px;
            margin-top: 4px;
        }
        /* 支撑阻力位 */
        .levels-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }
        @media (max-width: 768px) {
            .levels-grid { grid-template-columns: 1fr; }
        }
        .levels-block h4 {
            font-size: 14px;
            margin-bottom: 10px;
            color: var(--text);
        }
        .levels-block.resistance h4 { color: var(--danger); }
        .levels-block.support h4 { color: var(--success); }
        .levels-block .current-price {
            margin-top: 12px;
            padding: 10px 14px;
            background: var(--bg);
            border-radius: 6px;
            font-size: 13px;
            color: var(--text-muted);
        }
        .levels-block .current-price strong {
            color: var(--text);
            font-size: 16px;
        }
        /* 风险提示 */
        .risk-section { border-left: 4px solid var(--warning); }
        .risk-list {
            list-style: none;
            padding: 0;
        }
        .risk-list li {
            padding: 10px 14px 10px 36px;
            margin-bottom: 8px;
            background: var(--bg);
            border-radius: 6px;
            position: relative;
            font-size: 13.5px;
            color: var(--text);
        }
        .risk-list li::before {
            content: '⚠';
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--warning);
            font-size: 16px;
        }
        /* 图表容器 */
        .chart-container {
            width: 100%;
            overflow-x: auto;
            margin-top: 4px;
        }
        .chart-placeholder {
            padding: 40px;
            text-align: center;
            color: var(--text-light);
            background: var(--bg);
            border-radius: 8px;
            font-size: 13px;
        }
        /* 基本面网格 */
        .fundamental-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 14px;
        }
        .fund-card {
            background: var(--bg);
            padding: 14px 16px;
            border-radius: 8px;
            border: 1px solid var(--border-light);
        }
        .fund-card .fund-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .fund-card .fund-value {
            font-size: 20px;
            font-weight: 700;
            color: var(--text);
        }
        /* 摘要 */
        .summary-box {
            padding: 16px 20px;
            background: var(--bg);
            border-radius: 8px;
            font-size: 14px;
            color: var(--text);
            line-height: 1.8;
        }
        /* 免责声明 */
        .disclaimer {
            text-align: center;
            margin-top: 32px;
            padding: 20px;
            font-size: 12.5px;
            color: var(--text-light);
            border-top: 1px solid var(--border);
        }
        .disclaimer strong { color: var(--warning); }
        /* 响应式 */
        @media (max-width: 640px) {
            body { padding: 12px 8px 40px; font-size: 13px; }
            .report-header { padding: 24px 20px; }
            .report-header h1 { font-size: 20px; }
            .section { padding: 16px 14px; }
            .score-card .score-value { font-size: 28px; }
            th, td { padding: 8px 10px; }
        }
        /* LLM 分析师章节样式 */
        .llm-section { border-left: 4px solid #7c3aed; }
        .llm-section .section-title { color: #7c3aed; }
        .llm-analysis-header {
            display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;
        }
        .llm-badge {
            display: inline-block; padding: 4px 10px; border-radius: 4px;
            background: #f3f4f6; color: #555; font-size: 12px; font-weight: 600;
        }
        .llm-badge-positive { background: #dcfce7; color: #166534; }
        .llm-badge-negative { background: #fee2e2; color: #991b1b; }
        .llm-analysis-body h4 {
            margin: 14px 0 6px 0; font-size: 14px; color: #7c3aed;
            border-left: 3px solid #7c3aed; padding-left: 8px;
        }
        .llm-analysis-body p {
            margin: 0 0 8px 0; line-height: 1.7; color: #444; font-size: 14px;
        }
        .llm-placeholder {
            padding: 12px; background: #faf5ff; border: 1px dashed #c4b5fd;
            border-radius: 6px; color: #6b21a8; font-size: 13px;
        }
    </style>
</head>
<body>

<!-- 报告标题 -->
<header class="report-header">
    <h1>{{ stock_name }} ({{ stock_code }})</h1>
    <div class="subtitle">个股深度分析报告</div>
    <div class="meta">
        <span>📅 生成时间: {{ generated_at }}</span>
        <span>📊 数据截止: {{ data_date }}</span>
    </div>
</header>

<!-- 综合评分 -->
<section class="section">
    <div class="section-title">综合评分</div>
    <div class="score-grid">
        <div class="score-card technical">
            <div class="score-label">技术面评分</div>
            <div class="score-value {{ 'score-high' if scores.technical >= 70 else ('score-mid' if scores.technical >= 50 else 'score-low') }}">{{ scores.technical }}</div>
            <div class="score-rating">满分 100</div>
        </div>
        {% if scores.has_fundamental %}
        <div class="score-card fundamental">
            <div class="score-label">基本面评分</div>
            <div class="score-value {{ 'score-high' if scores.fundamental >= 70 else ('score-mid' if scores.fundamental >= 50 else 'score-low') }}">{{ scores.fundamental }}</div>
            <div class="score-rating">满分 100</div>
        </div>
        {% endif %}
        <div class="score-card comprehensive">
            <div class="score-label">综合评分</div>
            <div class="score-value {{ 'score-high' if scores.comprehensive >= 70 else ('score-mid' if scores.comprehensive >= 50 else 'score-low') }}">{{ scores.comprehensive }}</div>
            <div class="score-rating">{{ scores.rating }}</div>
        </div>
    </div>
    <div class="score-breakdown">
        <div class="breakdown-item">
            <div class="label">趋势方向 /30</div>
            <div class="value {{ 'score-high' if scores.breakdown.trend >= 20 else ('score-low' if scores.breakdown.trend < 10 else '') }}">{{ scores.breakdown.trend }}</div>
        </div>
        <div class="breakdown-item">
            <div class="label">指标信号 /30</div>
            <div class="value {{ 'score-high' if scores.breakdown.indicator >= 20 else ('score-low' if scores.breakdown.indicator < 10 else '') }}">{{ scores.breakdown.indicator }}</div>
        </div>
        <div class="breakdown-item">
            <div class="label">K线形态 /20</div>
            <div class="value {{ 'score-high' if scores.breakdown.pattern >= 14 else ('score-low' if scores.breakdown.pattern < 7 else '') }}">{{ scores.breakdown.pattern }}</div>
        </div>
        <div class="breakdown-item">
            <div class="label">量价配合 /20</div>
            <div class="value {{ 'score-high' if scores.breakdown.volume >= 14 else ('score-low' if scores.breakdown.volume < 7 else '') }}">{{ scores.breakdown.volume }}</div>
        </div>
        {% if scores.has_fundamental %}
        <div class="breakdown-item">
            <div class="label">估值水平 /35</div>
            <div class="value {{ 'score-high' if scores.breakdown.valuation >= 24 else ('score-low' if scores.breakdown.valuation < 12 else '') }}">{{ scores.breakdown.valuation }}</div>
        </div>
        <div class="breakdown-item">
            <div class="label">盈利能力 /35</div>
            <div class="value {{ 'score-high' if scores.breakdown.profitability >= 24 else ('score-low' if scores.breakdown.profitability < 12 else '') }}">{{ scores.breakdown.profitability }}</div>
        </div>
        <div class="breakdown-item">
            <div class="label">成长性 /30</div>
            <div class="value {{ 'score-high' if scores.breakdown.growth >= 20 else ('score-low' if scores.breakdown.growth < 10 else '') }}">{{ scores.breakdown.growth }}</div>
        </div>
        {% endif %}
    </div>
</section>

<!-- 多周期技术面分析 -->
<section class="section">
    <div class="section-title">多周期技术面分析</div>
    <table>
        <thead>
            <tr>
                <th>周期</th>
                <th>趋势</th>
                <th>强度</th>
                <th>收盘价</th>
                <th>MACD(DIF)</th>
                <th>RSI</th>
                <th>KDJ(K/D/J)</th>
                <th>关键信号</th>
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
                <td>{{ _fmt_num(ind.macd_dif, 3) }}</td>
                <td>{{ _fmt_num(ind.rsi, 1) }}</td>
                <td>{{ _fmt_num(ind.kdj_k, 1) }} / {{ _fmt_num(ind.kdj_d, 1) }} / {{ _fmt_num(ind.kdj_j, 1) }}</td>
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
    {% if tf_summary %}
    <div class="resonance-box" style="margin-top:10px;border-left-color:var(--text-muted);">
        <strong>综合摘要：</strong>{{ tf_summary }}
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

<!-- 技术指标信号汇总 -->
<section class="section">
    <div class="section-title">技术指标信号汇总</div>
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
                {% if rsi_val is not none and rsi_val > 70 %}
                    <span class="badge badge-bearish">超买</span>
                {% elif rsi_val is not none and rsi_val < 30 %}
                    <span class="badge badge-bullish">超卖</span>
                {% else %}
                    <span class="badge badge-neutral">中性</span>
                {% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">KDJ (K/D/J)</div>
            <div class="ind-value">{{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'kdj_k'), 1) }} / {{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'kdj_d'), 1) }} / {{ _fmt_num(_ind_val(technical_indicators, multi_timeframe, 'kdj_j'), 1) }}</div>
            <div class="ind-signal">
                {% set j_val = _ind_val(technical_indicators, multi_timeframe, 'kdj_j') %}
                {% if j_val is not none and j_val > 100 %}
                    <span class="badge badge-bearish">超买</span>
                {% elif j_val is not none and j_val < 0 %}
                    <span class="badge badge-bullish">超卖</span>
                {% else %}
                    <span class="badge badge-neutral">中性</span>
                {% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">布林带位置</div>
            <div class="ind-value">{{ _fmt_pct(_ind_val(technical_indicators, multi_timeframe, 'boll_position')) }}</div>
            <div class="ind-signal">
                {% set boll = _ind_val(technical_indicators, multi_timeframe, 'boll_position') %}
                {% if boll is not none and boll > 0.9 %}
                    <span class="badge badge-bearish">触及上轨</span>
                {% elif boll is not none and boll < 0.1 %}
                    <span class="badge badge-bullish">触及下轨</span>
                {% else %}
                    <span class="badge badge-neutral">正常区间</span>
                {% endif %}
            </div>
        </div>
        <div class="indicator-card">
            <div class="ind-name">MA5 / MA20 / MA60</div>
            <div class="ind-value" style="font-size:16px;">{{ _fmt_price(_ind_val(technical_indicators, multi_timeframe, 'ma5')) }} / {{ _fmt_price(_ind_val(technical_indicators, multi_timeframe, 'ma20')) }} / {{ _fmt_price(_ind_val(technical_indicators, multi_timeframe, 'ma60')) }}</div>
            <div class="ind-signal">
                {% set ma5 = _ind_val(technical_indicators, multi_timeframe, 'ma5') %}
                {% set ma20 = _ind_val(technical_indicators, multi_timeframe, 'ma20') %}
                {% set ma60 = _ind_val(technical_indicators, multi_timeframe, 'ma60') %}
                {% if ma5 is not none and ma20 is not none and ma60 is not none and ma5 > ma20 > ma60 %}
                    <span class="badge badge-bearish">多头排列</span>
                {% elif ma5 is not none and ma20 is not none and ma60 is not none and ma5 < ma20 < ma60 %}
                    <span class="badge badge-bullish">空头排列</span>
                {% else %}
                    <span class="badge badge-neutral">交织</span>
                {% endif %}
            </div>
        </div>
    </div>
</section>

<!-- LLM 技术面深度解读（占位符，agent 端用 LLM 生成内容替换） -->
<section class="section llm-section">
    <div class="section-title">LLM 技术面深度解读</div>
<!--LLM_TECHNICAL_ANALYSIS_PLACEHOLDER-->
</section>

<!-- K线形态识别结果 -->
<section class="section">
    <div class="section-title">K线形态识别结果</div>
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
    </div>
    {% set patterns = pattern_results.recent_patterns if pattern_results.recent_patterns else [] %}
    {% if patterns %}
    <table>
        <thead>
            <tr>
                <th>日期</th>
                <th>形态名称</th>
                <th>方向</th>
                <th>可靠度</th>
            </tr>
        </thead>
        <tbody>
        {% for p in patterns[:15] %}
            <tr>
                <td>{{ _fmt_date(p.date) }}</td>
                <td>{{ p.chinese_name | default(p.pattern_name, true) }}</td>
                <td>
                    {% if p.signal_type == 'bullish' %}
                        <span class="badge badge-bullish">看涨</span>
                    {% elif p.signal_type == 'bearish' %}
                        <span class="badge badge-bearish">看跌</span>
                    {% else %}
                        <span class="badge badge-neutral">中性</span>
                    {% endif %}
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

<!-- 支撑阻力位 -->
<section class="section">
    <div class="section-title">支撑阻力位</div>
    {% if support_resistance and (support_resistance.resistance or support_resistance.support) %}
    <div class="levels-grid">
        <div class="levels-block resistance">
            <h4>▲ 阻力位 (由近及远)</h4>
            <table>
                <thead>
                    <tr><th>价格</th><th>类型</th><th>强度</th><th>方法</th></tr>
                </thead>
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
                <thead>
                    <tr><th>价格</th><th>类型</th><th>强度</th><th>方法</th></tr>
                </thead>
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
    {% if support_resistance.current_price %}
    <div class="levels-block" style="margin-top:16px;">
        <div class="current-price">
            当前价格: <strong>{{ _fmt_price(support_resistance.current_price) }}</strong>
            {% if support_resistance.nearest_resistance %}
            ｜ 最近阻力: <strong style="color:var(--danger);">{{ _fmt_price(support_resistance.nearest_resistance) }}</strong>
            {% endif %}
            {% if support_resistance.nearest_support %}
            ｜ 最近支撑: <strong style="color:var(--success);">{{ _fmt_price(support_resistance.nearest_support) }}</strong>
            {% endif %}
        </div>
    </div>
    {% endif %}
    {% else %}
    <div class="chart-placeholder">暂无支撑阻力位数据</div>
    {% endif %}
</section>

<!-- 基本面概览 -->
{% if fundamental_data %}
<section class="section">
    <div class="section-title">基本面概览</div>
    <div class="fundamental-grid">
        <div class="fund-card">
            <div class="fund-label">市盈率 PE(TTM)</div>
            <div class="fund-value">{{ _fmt_num(_fund_val(fundamental_data, 'pe', 'pe_ttm', 'pe_ratio'), 1) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">市净率 PB</div>
            <div class="fund-value">{{ _fmt_num(_fund_val(fundamental_data, 'pb', 'pb_ratio'), 2) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">PE分位</div>
            <div class="fund-value">{{ _fmt_pct(_fund_val(fundamental_data, 'pe_percentile', 'pe_pct')) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">ROE</div>
            <div class="fund-value">{{ _fmt_pct(_fund_val(fundamental_data, 'roe', 'roe_ttm')) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">毛利率</div>
            <div class="fund-value">{{ _fmt_pct(_fund_val(fundamental_data, 'gross_margin')) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">营收增速</div>
            <div class="fund-value">{{ _fmt_pct(_fund_val(fundamental_data, 'revenue_growth', 'rev_growth')) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">净利润增速</div>
            <div class="fund-value">{{ _fmt_pct(_fund_val(fundamental_data, 'profit_growth', 'net_profit_growth')) }}</div>
        </div>
        <div class="fund-card">
            <div class="fund-label">总市值</div>
            <div class="fund-value">{{ _fmt_market_cap(_fund_val(fundamental_data, 'market_cap', 'total_mv')) }}</div>
        </div>
    </div>
    {% if fundamental_chart_html %}
    <div class="chart-container" style="margin-top:20px;">{{ fundamental_chart_html | safe }}</div>
    {% endif %}
</section>
{% endif %}

<!-- LLM 基本面深度解读（占位符，agent 端用 LLM 生成内容替换） -->
<section class="section llm-section">
    <div class="section-title">LLM 基本面深度解读</div>
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

<!-- 综合技术分析图（K线+成交量+MACD+RSI+KDJ 联动） -->
<section class="section">
    <div class="section-title">综合技术分析图</div>
    <div class="chart-container">
    {% if kline_chart_html %}
        {{ kline_chart_html | safe }}
    {% else %}
        <div class="chart-placeholder">K线图暂不可用</div>
    {% endif %}
    </div>
</section>

<!-- 免责声明 -->
<footer class="disclaimer">
    <strong>免责声明：</strong>以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。<br>
    <span style="font-size:11px;">Generated by jingnitrader · {{ generated_at }}</span>
</footer>

</body>
</html>
"""
        template = Template(template_str)
        # 注入格式化辅助函数到模板上下文
        render_context = dict(context)
        render_context["_fmt_num"] = self._fmt_num
        render_context["_fmt_price"] = self._fmt_price
        render_context["_fmt_pct"] = self._fmt_pct
        render_context["_fmt_date"] = self._fmt_date
        render_context["_fmt_market_cap"] = self._fmt_market_cap
        render_context["_ind_val"] = self._tmpl_ind_val
        render_context["_fund_val"] = self._tmpl_fund_val
        return template.render(**render_context)

    # ================================================================
    # 辅助方法
    # ================================================================

    def _rating(self, score: float) -> str:
        """根据综合评分返回评级"""
        for threshold, label in self._RATING_THRESHOLDS:
            if score >= threshold:
                return label
        return "强烈回避"

    def _get_indicator(self, technical_indicators: Dict,
                       multi_timeframe: Dict, key: str) -> Any:
        """
        从 technical_indicators 或 multi_timeframe.daily.indicators 中
        提取指标值, technical_indicators 优先
        """
        # 1. 在 technical_indicators 中查找 (支持嵌套 dict)
        if technical_indicators:
            # 直接键名匹配 (忽略大小写)
            for k, v in technical_indicators.items():
                if isinstance(k, str) and k.lower() == key.lower():
                    if not isinstance(v, dict):
                        return v
            # 嵌套分组查找: {"MACD": {"dif": ...}, "KDJ": {"k": ...}}
            group_map = {
                "macd_dif": ("MACD", ["dif", "diff", "DIF"]),
                "macd_dea": ("MACD", ["dea", "signal", "DEA"]),
                "macd_hist": ("MACD", ["hist", "bar", "HIST"]),
                "kdj_k": ("KDJ", ["k", "K"]),
                "kdj_d": ("KDJ", ["d", "D"]),
                "kdj_j": ("KDJ", ["j", "J"]),
                "ma5": ("MA", ["ma5", "MA5", "5"]),
                "ma10": ("MA", ["ma10", "MA10", "10"]),
                "ma20": ("MA", ["ma20", "MA20", "20"]),
                "ma60": ("MA", ["ma60", "MA60", "60"]),
            }
            if key in group_map:
                group_name, sub_keys = group_map[key]
                group = technical_indicators.get(group_name)
                if isinstance(group, dict):
                    for sk in sub_keys:
                        if sk in group:
                            return group[sk]
                # 也尝试小写的组名
                group = technical_indicators.get(group_name.lower())
                if isinstance(group, dict):
                    for sk in sub_keys:
                        if sk in group:
                            return group[sk]

        # 2. 回退到 multi_timeframe.daily.indicators
        daily_ind = (multi_timeframe or {}).get(
            "timeframes", {}
        ).get("daily", {}).get("indicators", {})
        if daily_ind:
            for k, v in daily_ind.items():
                if isinstance(k, str) and k.lower() == key.lower():
                    return v
        return None

    def _get_fundamental(self, fundamental_data: Dict, *keys) -> Any:
        """从基本面数据中查找值, 支持多个候选键名"""
        if not fundamental_data:
            return None
        lower_map = {}
        for k, v in fundamental_data.items():
            if isinstance(k, str):
                lower_map[k.lower()] = v
        for key in keys:
            k = key.lower() if isinstance(key, str) else key
            if k in lower_map:
                return lower_map[k]
        return None

    def _tmpl_ind_val(self, technical_indicators: Dict,
                      multi_timeframe: Dict, key: str) -> Any:
        """模板可调用的指标提取函数"""
        return self._get_indicator(technical_indicators, multi_timeframe, key)

    def _tmpl_fund_val(self, fundamental_data: Dict, *keys) -> Any:
        """模板可调用的基本面值提取函数"""
        return self._get_fundamental(fundamental_data, *keys)

    def _compute_volume_metrics(self, ohlcv_data: pd.DataFrame) -> Dict:
        """从 OHLCV 数据计算量价指标, 注入 technical_indicators 供评分使用"""
        metrics = {}
        if ohlcv_data is None or len(ohlcv_data) < 10:
            metrics["up_down_volume_ratio"] = np.nan
            metrics["volume_trend"] = "unknown"
            return metrics

        try:
            df = ohlcv_data.copy()
            if "date" in df.columns:
                df = df.sort_values("date").reset_index(drop=True)
            recent = df.tail(20).copy()
            close = recent["close"].astype(float)
            volume = recent["volume"].astype(float)
            returns = close.pct_change().dropna()
            vol = volume.iloc[1:].reset_index(drop=True)
            rets = returns.reset_index(drop=True)

            up_mask = rets > 0
            down_mask = rets < 0
            up_vol = vol[up_mask].mean() if up_mask.any() else 0.0
            down_vol = vol[down_mask].mean() if down_mask.any() else 0.0

            if down_vol > 0 and up_vol > 0:
                ratio = float(up_vol / down_vol)
            elif up_vol > 0:
                ratio = 2.0
            elif down_vol > 0:
                ratio = 0.3
            else:
                ratio = 1.0
            metrics["up_down_volume_ratio"] = ratio

            # 成交量趋势: 近5日均量 vs 前10日均量
            if len(volume) >= 15:
                recent_vol = volume.tail(5).mean()
                prev_vol = volume.iloc[-15:-5].mean()
                if prev_vol > 0:
                    vol_change = recent_vol / prev_vol
                    if vol_change > 1.2:
                        metrics["volume_trend"] = "increasing"
                    elif vol_change < 0.8:
                        metrics["volume_trend"] = "decreasing"
                    else:
                        metrics["volume_trend"] = "stable"
                else:
                    metrics["volume_trend"] = "stable"
            else:
                metrics["volume_trend"] = "unknown"
        except Exception as e:
            logger.debug(f"计算量价指标失败: {e}")
            metrics["up_down_volume_ratio"] = np.nan
            metrics["volume_trend"] = "unknown"

        return metrics

    def _get_data_date(self, ohlcv_data: pd.DataFrame) -> str:
        """从 OHLCV 数据提取最新日期"""
        if ohlcv_data is None or ohlcv_data.empty:
            return "—"
        try:
            if "date" in ohlcv_data.columns:
                last_date = ohlcv_data["date"].iloc[-1]
                return str(last_date)[:10]
            return "—"
        except Exception:
            return "—"

    def _safe_render_chart(self, chart_gen, method_name: str, *args,
                           fallback: str = "", **kwargs) -> str:
        """安全调用图表生成器, 失败时返回占位符"""
        if chart_gen is None:
            return ""
        method = getattr(chart_gen, method_name, None)
        if method is None:
            return ""
        try:
            result = method(*args, **kwargs)
            if isinstance(result, str):
                return result
            # Plotly Figure -> HTML 片段
            if hasattr(result, "to_html"):
                return result.to_html(full_html=False, include_plotlyjs="cdn")
            return str(result) if result else ""
        except Exception as e:
            logger.warning(f"图表生成失败 ({method_name}): {e}")
            return ""

    # ── 格式化辅助 ─────────────────────────────

    @staticmethod
    def _fmt_num(value: Any, decimals: int = 2) -> str:
        """格式化数值"""
        if value is None:
            return "—"
        try:
            v = float(value)
            if pd.isna(v) or not np.isfinite(v):
                return "—"
            return f"{v:.{decimals}f}"
        except Exception:
            return "—"

    @staticmethod
    def _fmt_price(value: Any) -> str:
        """格式化价格"""
        if value is None:
            return "—"
        try:
            v = float(value)
            if pd.isna(v) or not np.isfinite(v):
                return "—"
            return f"{v:.2f}"
        except Exception:
            return "—"

    @staticmethod
    def _fmt_pct(value: Any) -> str:
        """格式化百分比 (输入可为小数 0.15 或百分数 15.0)"""
        if value is None:
            return "—"
        try:
            v = float(value)
            if pd.isna(v) or not np.isfinite(v):
                return "—"
            if abs(v) < 1:
                v = v * 100
            return f"{v:.2f}%"
        except Exception:
            return "—"

    @staticmethod
    def _fmt_date(value: Any) -> str:
        """格式化日期"""
        if value is None:
            return "—"
        try:
            s = str(value)
            return s[:10] if len(s) >= 10 else s
        except Exception:
            return "—"

    @staticmethod
    def _fmt_market_cap(value: Any) -> str:
        """格式化市值 (输入为元)"""
        if value is None:
            return "—"
        try:
            v = float(value)
            if pd.isna(v) or not np.isfinite(v):
                return "—"
            if v >= 1e12:
                return f"{v / 1e12:.2f}万亿"
            elif v >= 1e8:
                return f"{v / 1e8:.2f}亿"
            elif v >= 1e4:
                return f"{v / 1e4:.2f}万"
            return f"{v:.2f}"
        except Exception:
            return "—"
