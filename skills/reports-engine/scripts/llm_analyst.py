"""
LLM 分析师模块（借鉴 TradingAgents 源码设计）

借鉴点：
1. 三层防幻觉：身份锚定 + 行情真值快照 + 禁止编造断言
   - 对应源码 resolve_instrument_identity() / get_verified_market_snapshot()
   - 对应源码 market_analyst.py 第 68-70 行 "Do not claim historical validation..."
2. 结构化输出 schema：Pydantic BaseModel，字段描述即输出指令
   - 对应源码 schemas.py 的 SentimentReport / PortfolioDecision
3. 预取数据入 prompt：不调用工具，LLM 只做解读
   - 对应源码各分析师的 system_message + 数据注入

设计差异（与源码）：
- 不用 LangChain / tool-calling（数据已由 factor-engine 预计算）
- LLM 调用在 reports-engine 内部完成（llm_client.py），skill 运行即 agent 调用
- Schema 用 dataclass 而非 Pydantic（减少依赖，agent 端可自由解析）
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum


# ================================================================
# 结构化输出 Schema
# 借鉴 TradingAgents schemas.py：结构化摘要字段 + 详细叙事字段
# ================================================================

class TrendDirection(str, Enum):
    """趋势方向（借鉴 SentimentBand 的枚举模式）"""
    BULLISH = "看涨"
    BEARISH = "看跌"
    SIDEWAYS = "震荡"


class ConfidenceLevel(str, Enum):
    """置信度等级"""
    HIGH = "高"
    MEDIUM = "中"
    LOW = "低"


class ValuationLevel(str, Enum):
    """估值水平"""
    UNDERVALUED = "低估"
    FAIR = "合理"
    PREMIUM = "偏高"
    OVERVALUED = "高估"


@dataclass
class TechnicalAnalysisReport:
    """
    技术面深度分析报告

    借鉴 SentimentReport 的设计：结构化摘要字段（机器可解析）
    + 详细叙事字段（人类可读），render() 方法渲染为 HTML。

    扩展 A 股特色字段：资金面/龙虎榜/涨跌停分析。
    """
    # 结构化摘要（机器可解析，便于后续回测/审计）
    trend_direction: str          # 整体趋势方向：看涨/看跌/震荡
    trend_confidence: str         # 趋势判断置信度：高/中/低
    technical_score: float        # 技术面综合评分 0-100

    # 详细叙事（人类可读）
    overall_assessment: str       # 整体技术面评估（1-2句概括）
    trend_analysis: str           # 多周期趋势分析（日/周/月线共振与背离）
    indicator_analysis: str       # 技术指标信号解读（MACD/RSI/KDJ/BOLL逐项分析）
    key_levels: str               # 关键价位分析（支撑/阻力的有效性及突破方向）
    risk_signals: str             # 风险信号（背离/超买超卖/量价异常）
    short_term_outlook: str       # 短期展望（1-2周方向判断及关键观察点）

    # A 股特色字段（可选；数据缺失时为空字符串）
    capital_flow_analysis: str = ""       # 资金面分析（主力资金/北向资金动向）
    dragon_tiger_analysis: str = ""       # 龙虎榜分析（机构席位、营业部动向）
    price_limit_analysis: str = ""        # 涨跌停分析（近期涨跌停、封板强度）

    def render_html(self) -> str:
        """渲染为报告章节 HTML（借鉴 render_sentiment_report 模式）

        输出可直接替换模板中的 <!--LLM_TECHNICAL_ANALYSIS_PLACEHOLDER-->。
        """
        import html as _html
        score_color = "positive" if self.technical_score >= 60 else (
            "negative" if self.technical_score < 40 else ""
        )
        # A 股特色章节（仅在 LLM 生成内容时渲染）
        optional_sections = []
        if self.capital_flow_analysis:
            optional_sections.append(
                f"<h4>资金面分析 <span class='llm-tag-a'>A股特色</span></h4>"
                f"<p>{_html.escape(self.capital_flow_analysis)}</p>"
            )
        if self.dragon_tiger_analysis:
            optional_sections.append(
                f"<h4>龙虎榜解读 <span class='llm-tag-a'>A股特色</span></h4>"
                f"<p>{_html.escape(self.dragon_tiger_analysis)}</p>"
            )
        if self.price_limit_analysis:
            optional_sections.append(
                f"<h4>涨跌停分析 <span class='llm-tag-a'>A股特色</span></h4>"
                f"<p>{_html.escape(self.price_limit_analysis)}</p>"
            )
        optional_html = "\n            ".join(optional_sections)
        return f"""
        <div class="llm-analysis-header">
            <span class="llm-badge llm-badge-{score_color}">
                技术评分 {self.technical_score:.0f}
            </span>
            <span class="llm-badge">趋势：{self.trend_direction}</span>
            <span class="llm-badge">置信度：{self.trend_confidence}</span>
        </div>
        <div class="llm-analysis-body">
            <h4>整体评估</h4>
            <p>{_html.escape(self.overall_assessment)}</p>
            <h4>多周期趋势分析</h4>
            <p>{_html.escape(self.trend_analysis)}</p>
            <h4>技术指标信号解读</h4>
            <p>{_html.escape(self.indicator_analysis)}</p>
            <h4>关键价位分析</h4>
            <p>{_html.escape(self.key_levels)}</p>
            <h4>风险信号</h4>
            <p>{_html.escape(self.risk_signals)}</p>
            <h4>短期展望</h4>
            <p>{_html.escape(self.short_term_outlook)}</p>
            {optional_html}
        </div>"""


@dataclass
class FundamentalAnalysisReport:
    """基本面深度分析报告

    扩展 A 股特色字段：股东结构、行业景气度、投资评级。
    """
    # 结构化摘要
    valuation_level: str          # 估值水平：低估/合理/偏高/高估
    fundamental_score: float      # 基本面综合评分 0-100
    investment_rating: str = "—"  # 投资评级：买入/增持/中性/减持/卖出

    # 详细叙事
    overall_assessment: str = ""       # 整体基本面评估
    valuation_analysis: str = ""       # 估值分析（PE/PB分位解读，与历史/同业对比）
    profitability_analysis: str = ""   # 盈利能力分析（ROE/毛利率拆解与质量评估）
    growth_analysis: str = ""          # 成长性分析（营收/利润增速趋势与可持续性）
    risk_factors: str = ""             # 风险因素（高估值/业绩下滑/行业拐点/竞争加剧）

    # 扩展章节
    industry_analysis: str = ""        # 行业分析与景气度
    financial_statement_analysis: str = ""  # 财务报表分析（三大报表关键科目）
    shareholder_analysis: str = ""     # 股东结构与资本运作（A股特色）

    def render_html(self) -> str:
        """渲染为报告章节 HTML

        输出可直接替换模板中的 <!--LLM_FUNDAMENTAL_ANALYSIS_PLACEHOLDER-->。
        """
        import html as _html
        score_color = "positive" if self.fundamental_score >= 60 else (
            "negative" if self.fundamental_score < 40 else ""
        )
        optional_sections = []
        if self.industry_analysis:
            optional_sections.append(
                f"<h4>行业分析与景气度</h4><p>{_html.escape(self.industry_analysis)}</p>"
            )
        if self.financial_statement_analysis:
            optional_sections.append(
                f"<h4>财务报表分析</h4><p>{_html.escape(self.financial_statement_analysis)}</p>"
            )
        if self.shareholder_analysis:
            optional_sections.append(
                f"<h4>股东结构与资本运作 <span class='llm-tag-a'>A股特色</span></h4>"
                f"<p>{_html.escape(self.shareholder_analysis)}</p>"
            )
        optional_html = "\n            ".join(optional_sections)
        return f"""
        <div class="llm-analysis-header">
            <span class="llm-badge llm-badge-{score_color}">
                基本面评分 {self.fundamental_score:.0f}
            </span>
            <span class="llm-badge">估值：{self.valuation_level}</span>
            <span class="llm-badge">评级：{_html.escape(self.investment_rating)}</span>
        </div>
        <div class="llm-analysis-body">
            <h4>整体评估</h4>
            <p>{_html.escape(self.overall_assessment)}</p>
            <h4>估值分析</h4>
            <p>{_html.escape(self.valuation_analysis)}</p>
            <h4>盈利能力分析</h4>
            <p>{_html.escape(self.profitability_analysis)}</p>
            <h4>成长性分析</h4>
            <p>{_html.escape(self.growth_analysis)}</p>
            <h4>风险因素</h4>
            <p>{_html.escape(self.risk_factors)}</p>
            {optional_html}
        </div>"""


# ================================================================
# Schema 的 JSON 描述（供 agent 端 LLM 结构化输出参考）
# ================================================================

TECHNICAL_REPORT_SCHEMA = {
    "name": "TechnicalAnalysisReport",
    "description": "技术面深度分析报告（含 A 股特色字段）",
    "fields": {
        "trend_direction": {
            "type": "string",
            "enum": ["看涨", "看跌", "震荡"],
            "description": "整体趋势方向"
        },
        "trend_confidence": {
            "type": "string",
            "enum": ["高", "中", "低"],
            "description": "趋势判断置信度"
        },
        "technical_score": {
            "type": "float",
            "range": [0, 100],
            "description": "技术面综合评分"
        },
        "overall_assessment": {"type": "string", "description": "整体技术面评估（1-2句概括）"},
        "trend_analysis": {"type": "string", "description": "多周期趋势分析（日/周/月线共振与背离）"},
        "indicator_analysis": {"type": "string", "description": "技术指标信号解读（MACD/RSI/KDJ/BOLL逐项分析）"},
        "key_levels": {"type": "string", "description": "关键价位分析（支撑/阻力的有效性及突破方向）"},
        "risk_signals": {"type": "string", "description": "风险信号（背离/超买超卖/量价异常）"},
        "short_term_outlook": {"type": "string", "description": "短期展望（1-2周方向判断及关键观察点）"},
        "capital_flow_analysis": {
            "type": "string",
            "optional": True,
            "description": "A 股特色：资金面分析（主力资金/北向资金动向）。数据缺失时填空字符串"
        },
        "dragon_tiger_analysis": {
            "type": "string",
            "optional": True,
            "description": "A 股特色：龙虎榜分析（机构席位、营业部动向）。数据缺失时填空字符串"
        },
        "price_limit_analysis": {
            "type": "string",
            "optional": True,
            "description": "A 股特色：涨跌停分析（近期涨跌停、封板强度）。数据缺失时填空字符串"
        },
    }
}

FUNDAMENTAL_REPORT_SCHEMA = {
    "name": "FundamentalAnalysisReport",
    "description": "基本面深度分析报告（含 A 股特色字段）",
    "fields": {
        "valuation_level": {
            "type": "string",
            "enum": ["低估", "合理", "偏高", "高估"],
            "description": "估值水平"
        },
        "fundamental_score": {
            "type": "float",
            "range": [0, 100],
            "description": "基本面综合评分"
        },
        "investment_rating": {
            "type": "string",
            "enum": ["买入", "增持", "中性", "减持", "卖出"],
            "description": "投资评级"
        },
        "overall_assessment": {"type": "string", "description": "整体基本面评估"},
        "valuation_analysis": {"type": "string", "description": "估值分析（PE/PB分位解读）"},
        "profitability_analysis": {"type": "string", "description": "盈利能力分析（ROE/毛利率拆解）"},
        "growth_analysis": {"type": "string", "description": "成长性分析（营收/利润增速趋势）"},
        "risk_factors": {"type": "string", "description": "风险因素"},
        "industry_analysis": {
            "type": "string",
            "optional": True,
            "description": "行业分析与景气度（行业增速、竞争格局、景气周期）"
        },
        "financial_statement_analysis": {
            "type": "string",
            "optional": True,
            "description": "财务报表分析（三大报表关键科目与盈利质量）"
        },
        "shareholder_analysis": {
            "type": "string",
            "optional": True,
            "description": "A 股特色：股东结构与资本运作（十大股东、解禁、回购）"
        },
    }
}


# ================================================================
# 因子名称 → 说明映射表（用于动态生成 prompt 中的指标参考）
# ================================================================

_TECHNICAL_FACTOR_DESCRIPTIONS = {
    # 均线
    "ma5": "5日均线，短期趋势参考",
    "ma10": "10日均线，短期趋势参考",
    "ma20": "20日均线，中期趋势参考（布林带中轨）",
    "ma60": "60日均线，中长期趋势参考（牛熊分界线）",
    # MACD
    "macd_dif": "MACD 快线（DIF），短期均线与长期均线差值",
    "macd_dea": "MACD 慢线（DEA），DIF 的移动平均",
    "macd_hist": "MACD 柱（DIF-DEA 的 2 倍），反映动量强弱与方向",
    # 动量指标
    "rsi_14": "RSI(14)，相对强弱指标，超买>70、超卖<30",
    "kdj_k": "KDJ 快速线 K，反映当前价格在近期区间的位置",
    "kdj_d": "KDJ 慢速线 D，K 的移动平均",
    "kdj_j": "KDJ J 值，3K-2D，领先于 K/D 的极端信号",
    "wr": "威廉指标 WR，超买>80、超卖<20",
    "cci": "商品通道指标 CCI，超买>100、超卖<-100",
    # 布林带
    "boll_ub": "布林带上轨，价格触及上轨意味短期偏强",
    "boll_mid": "布林带中轨（MA20），多空分界参考",
    "boll_lb": "布林带下轨，价格触及下轨意味短期偏弱",
    # 量价
    "volume_20d": "20日均量，衡量成交量相对水平",
    "volume_ratio": "量比，当日成交量与5日均量之比",
    "turnover_20d": "20日均换手率，反映交投活跃度",
    "turnover_change": "换手率变化，正值为活跃度增加",
    "obv": "能量潮 OBV，量价配合关系指标",
    # 资金面
    "main_net_inflow": "主力资金当日净流入（亿元），正值=流入，负值=流出",
    "main_net_inflow_5d": "5日主力资金累计净流入（亿元）",
    "north_net_inflow": "北向资金当日净流入（亿元），外资动向参考",
    # K线形态
    "cdll_pattern_count": "近期识别到的 K 线形态总数",
    "cdll_dominant_signal": "主导 K 线形态信号方向（看涨/看跌/中性）",
    # 龙虎榜
    "lhb_count_5d": "5日内龙虎榜上榜次数",
    "lhb_net_buy": "龙虎榜净买入额（亿元）",
}

_FUNDAMENTAL_FACTOR_DESCRIPTIONS = {
    "industry": "所属行业（申万/中信分类）",
    "name": "公司全称",
    "list_date": "上市日期",
    "total_share": "总股本（亿股）",
    "float_share": "流通股本（亿股）",
    "roe_ttm": "ROE(TTM)，净资产收益率，衡量股东回报效率",
    "roa_ttm": "ROA(TTM)，总资产收益率，衡量资产使用效率",
    "gross_margin_ttm": "毛利率(TTM)，衡量产品或服务的定价能力",
    "net_margin_ttm": "净利率(TTM)，衡量费用控制与最终盈利水平",
    "revenue_growth_yoy": "营收同比增速，衡量业务扩张速度",
    "profit_growth_yoy": "净利润同比增速，衡量盈利增长质量",
    "pe_ttm": "PE(TTM)，滚动市盈率，估值水平核心指标",
    "pb": "PB，市净率，资产端估值参考",
    "ps_ttm": "PS(TTM)，市销率，收入端估值参考",
    "dv_ratio": "股息率，衡量分红回报",
    "pe_percentile": "PE 历史分位值（%），当前 PE 在历史中的相对位置",
    "pb_percentile": "PB 历史分位值（%），当前 PB 在历史中的相对位置",
    "debt_ratio": "资产负债率（%），衡量杠杆水平",
    "current_ratio": "流动比率，衡量短期偿债能力",
    "ocf": "经营现金流（亿元），衡量盈利质量与现金流安全",
    "top_holder_change": "十大股东持股变动（增持/减持）",
    "holder_concentration": "股东集中度，衡量筹码集中程度",
}

# ================================================================
# 技术分析师
# 借鉴 TradingAgents market_analyst.py 的 system_message 设计
# ================================================================

class TechnicalAnalyst:
    """
    技术面 LLM 分析师

    借鉴 TradingAgents market_analyst 设计：
    - 身份锚定（resolve_instrument_identity）
    - 行情真值快照（get_verified_market_snapshot）
    - 禁止编造断言（market_analyst.py 第 68-70 行）
    - 指标分类说明（market_analyst.py 的指标列表）

    差异：不调用工具，数据已由 factor-engine 预计算并注入 prompt

    v2 动态 prompt：根据模板 factor_groups 动态生成系统提示词，
    而非使用硬编码的指标说明。
    """

    def prepare(
        self,
        ctx: Dict[str, Any],
        factor_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        准备 prompt，返回 {system_prompt, user_prompt, response_schema}

        参数:
            ctx: 包含 stock_code, stock_name, current_price, data_date,
                 technical_indicators, multi_timeframe, pattern_results,
                 support_resistance 的字典
            factor_groups: 模板配置中的因子分组列表（来自 technical.yaml），
                           用于动态生成指标参考说明。若为 None 则回退到硬编码。
        """
        identity = self._build_identity(ctx)
        snapshot = self._build_snapshot(ctx)

        # 动态生成指标参考（基于模板 factor_groups）
        if factor_groups:
            indicator_guide = self._build_factor_guide(factor_groups, _TECHNICAL_FACTOR_DESCRIPTIONS)
            group_analysis_hints = self._build_analysis_hints(factor_groups)
        else:
            indicator_guide = self._indicator_guide()
            group_analysis_hints = ""

        system_prompt = self._build_system_prompt(indicator_guide, group_analysis_hints, bool(factor_groups))

        user_prompt = f"""## 身份锚定
{identity}

## 行情真值（以此为唯一事实来源）
{snapshot}

## 需要分析的数据
请基于以上真值数据，输出结构化的技术面分析报告 JSON。"""

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_schema": TECHNICAL_REPORT_SCHEMA,
        }

    def _build_system_prompt(
        self,
        indicator_guide: str,
        group_analysis_hints: str,
        is_dynamic: bool,
    ) -> str:
        """构建 system prompt（动态版或静态版）"""
        return f"""你是一位 A 股技术分析师，专注于解读已有的技术指标数据。

## 核心规则（借鉴 TradingAgents 防幻觉设计）
1. 所有价格、涨跌幅、指标值必须以"行情真值"为准，不凭空编造任何数字
2. 如果某项数据缺失，诚实说明"数据暂缺"，不猜测填充
3. 不要声称"历史回测显示支撑有效"或"该价位已多次确认"——除非数据中明确提供了这些历史证据
4. 解读要通俗易懂，面向非专业投资者，避免过多术语堆砌
5. 明确标注每个判断的依据（如"日线MACD金叉表明短期趋势转强"）

## 技术指标参考（{'模板动态生成' if is_dynamic else '借鉴 TradingAgents market_analyst 指标说明'}）
{indicator_guide}
{group_analysis_hints}
## A 股特色分析要求
1. capital_flow_analysis：解读主力资金与北向资金动向。若 prompt 中提供了资金面数据，
   基于数据解读资金是否支持当前价格趋势；若数据缺失，填空字符串""
2. dragon_tiger_analysis：解读龙虎榜信号。若上榜，分析机构席位净买卖方向及含义；
   若未上榜或数据缺失，填空字符串""
3. price_limit_analysis：分析近期涨跌停情况及对后续走势的含义。若无涨跌停，填空字符串""

## 输出要求
输出一份结构化的技术面分析报告，严格遵循以下 JSON 格式：
{{
  "trend_direction": "看涨|看跌|震荡",
  "trend_confidence": "高|中|低",
  "technical_score": 0-100的数字,
  "overall_assessment": "整体技术面评估（1-2句概括）",
  "trend_analysis": "多周期趋势分析（日/周/月线共振与背离）",
  "indicator_analysis": "技术指标信号解读（MACD/RSI/KDJ/BOLL逐项分析）",
  "key_levels": "关键价位分析（支撑/阻力的有效性及突破方向）",
  "risk_signals": "风险信号（背离/超买超卖/量价异常）",
  "short_term_outlook": "短期展望（1-2周方向判断及关键观察点）",
  "capital_flow_analysis": "资金面分析，或空字符串",
  "dragon_tiger_analysis": "龙虎榜分析，或空字符串",
  "price_limit_analysis": "涨跌停分析，或空字符串"
}}

只输出 JSON，不要其他文字。"""

    def _build_identity(self, ctx: Dict) -> str:
        """构建身份锚定（借鉴 build_instrument_context）"""
        return (
            f"标的：{ctx.get('stock_name', '')}（{ctx.get('stock_code', '')}），"
            f"当前价 {ctx.get('current_price', '')} 元，"
            f"数据截止 {ctx.get('data_date', '')}"
        )

    def _build_snapshot(self, ctx: Dict) -> str:
        """构建行情真值快照（借鉴 get_verified_market_snapshot）"""
        lines = []

        # 多周期分析
        mtf = ctx.get("multi_timeframe", {}) or {}
        if mtf:
            lines.append("### 多周期分析")
            timeframes = mtf.get("timeframes", {}) or {}
            for key, label in [("daily", "日线"), ("weekly", "周线"), ("monthly", "月线")]:
                tf = timeframes.get(key, {}) or {}
                if tf:
                    ind = tf.get("indicators", {}) or {}
                    lines.append(
                        f"- {label}: 趋势={tf.get('trend', '未知')}, "
                        f"强度={tf.get('strength', '无')}, "
                        f"收盘={ind.get('close', '数据暂缺')}"
                    )
                    signals = tf.get("signals", []) or []
                    if signals:
                        sig_types = [s.get("type", "") for s in signals if isinstance(s, dict)]
                        if sig_types:
                            lines.append(f"  信号: {', '.join(sig_types)}")
            resonance = mtf.get("resonance", {}) or {}
            if resonance:
                lines.append(
                    f"- 多周期共振: {resonance.get('description', '无明确共振')}"
                )
            divergences = mtf.get("divergences", []) or []
            if divergences:
                lines.append("- 检测到的背离信号:")
                for div in divergences:
                    if isinstance(div, dict):
                        lines.append(f"  • {div.get('description', div.get('type', ''))}")

        # 技术指标
        ind = ctx.get("technical_indicators", {}) or {}
        if ind:
            lines.append("\n### 技术指标")
            lines.append(
                f"- MACD: DIF={ind.get('macd_dif', '数据暂缺')}, "
                f"DEA={ind.get('macd_dea', '数据暂缺')}, "
                f"柱={ind.get('macd_hist', '数据暂缺')}"
            )
            lines.append(f"- RSI(14): {ind.get('rsi', '数据暂缺')}")
            lines.append(
                f"- KDJ: K={ind.get('kdj_k', '数据暂缺')}, "
                f"D={ind.get('kdj_d', '数据暂缺')}, "
                f"J={ind.get('kdj_j', '数据暂缺')}"
            )
            lines.append(
                f"- BOLL: 上轨={ind.get('boll_ub', '数据暂缺')}, "
                f"中轨={ind.get('boll_mid', '数据暂缺')}, "
                f"下轨={ind.get('boll_lb', '数据暂缺')}"
            )
            lines.append(
                f"- 均线: MA5={ind.get('ma5', '数据暂缺')}, "
                f"MA10={ind.get('ma10', '数据暂缺')}, "
                f"MA20={ind.get('ma20', '数据暂缺')}, "
                f"MA60={ind.get('ma60', '数据暂缺')}"
            )

        # K线形态
        patterns = ctx.get("pattern_results", {}) or {}
        if patterns:
            lines.append("\n### K线形态")
            lines.append(f"- 主导信号: {patterns.get('dominant_signal', '无明确信号')}")
            lines.append(f"- 看涨形态数: {patterns.get('bullish_count', 0)}")
            lines.append(f"- 看跌形态数: {patterns.get('bearish_count', 0)}")
            recent = patterns.get("recent_patterns", []) or []
            if recent:
                lines.append("- 近期形态:")
                for p in recent[:5]:
                    if isinstance(p, dict):
                        lines.append(
                            f"  • {p.get('date', '')} {p.get('type', '')} "
                            f"({p.get('direction', '')})"
                        )

        # 支撑阻力
        sr = ctx.get("support_resistance", {}) or {}
        if sr:
            lines.append("\n### 支撑阻力位")
            supports = sr.get("supports", []) or []
            resistances = sr.get("resistances", []) or []
            if supports:
                lines.append(f"- 支撑位: {supports}")
            if resistances:
                lines.append(f"- 阻力位: {resistances}")

        # A 股特色：资金面数据
        cf = ctx.get("capital_flow", {}) or {}
        if cf:
            lines.append("\n### 资金面数据（A股特色）")
            if cf.get("main_net_inflow") is not None:
                lines.append(f"- 主力净流入: {cf.get('main_net_inflow')}")
            if cf.get("north_net_inflow") is not None:
                lines.append(f"- 北向资金净流入: {cf.get('north_net_inflow')}")
            if cf.get("main_net_inflow_5d") is not None:
                lines.append(f"- 5日主力净流入: {cf.get('main_net_inflow_5d')}")
            if not any(cf.get(k) is not None for k in
                       ["main_net_inflow", "north_net_inflow", "main_net_inflow_5d"]):
                lines.append("- （资金面数据暂缺）")

        # A 股特色：龙虎榜数据
        dt = ctx.get("dragon_tiger", {}) or {}
        if dt:
            lines.append("\n### 龙虎榜数据（A股特色）")
            if dt.get("has_data"):
                records = dt.get("records", []) or []
                if records:
                    lines.append(f"- 上榜次数: {len(records)}")
                    for r in records[:3]:
                        if isinstance(r, dict):
                            lines.append(
                                f"  • {r.get('date', '')} 净买入={r.get('net_buy', '—')} "
                                f"原因={r.get('reason', '—')}"
                            )
                if dt.get("institutional_buy") is not None:
                    lines.append(f"- 机构净买入: {dt.get('institutional_buy')}")
            else:
                lines.append("- 近期未上榜或数据暂缺")

        # A 股特色：涨跌停数据
        pl = ctx.get("limit_analysis", {}) or {}
        if pl:
            lines.append("\n### 涨跌停数据（A股特色）")
            if pl.get("has_limit_data"):
                lines.append(f"- 涨跌停类型: {pl.get('limit_type', '—')}")
                lines.append(f"- 近期涨停次数: {pl.get('recent_limit_ups', 0)}")
                lines.append(f"- 近期跌停次数: {pl.get('recent_limit_downs', 0)}")
                if pl.get("near_limit"):
                    lines.append("- 最近一日接近涨停")
            else:
                lines.append("- 涨跌停数据暂缺")

        return "\n".join(lines) if lines else "（行情真值数据暂缺）"

    def _indicator_guide(self) -> str:
        """借鉴 market_analyst.py 的指标分类说明（静态回退版本）"""
        return """- 均线 (MA5/MA10/MA20/MA60): 短中长期趋势判断，金叉死叉信号
  • MA5 上穿 MA10/MA20：短期转强；MA5 下穿：短期转弱
  • 价格站上 MA60：中期趋势偏多；跌破 MA60：中期转弱
- MACD (DIF/DEA/柱): 动量变化与趋势转折
  • DIF 上穿 DEA（金叉）：买入信号；DIF 下穿 DEA（死叉）：卖出信号
  • 红柱缩短：上涨动能减弱；绿柱缩短：下跌动能减弱
- RSI (14): 超买超卖判断
  • RSI > 70：超买，回调风险；RSI < 30：超卖，反弹机会
  • RSI 与价格背离：趋势反转预警
- KDJ (K/D/J): 超买超卖与金叉死叉
  • K/D > 80：超买；K/D < 20：超卖
  • K 上穿 D（金叉）：买入信号；K 下穿 D（死叉）：卖出信号
- BOLL (上中下轨): 价格运行区间与突破
  • 触及上轨：短期偏强；触及下轨：短期偏弱
  • 收口后突破：变盘信号
- 成交量: 量价配合关系
  • 放量上涨：多头强势；放量下跌：空头强势
  • 缩量上涨：上涨乏力；缩量下跌：下跌动能减弱"""

    def _build_factor_guide(
        self,
        factor_groups: List[Dict[str, Any]],
        descriptions: Dict[str, str],
    ) -> str:
        """根据模板 factor_groups 动态生成指标参考说明"""
        lines = []
        for group in factor_groups:
            group_id = group.get("id", "")
            title = group.get("title", group_id)
            factors = group.get("factors", [])
            lines.append(f"\n### {title}")
            for f in factors:
                desc = descriptions.get(f, f"技术指标 {f}")
                lines.append(f"- {f}: {desc}")
        return "\n".join(lines) if lines else "（无因子配置）"

    def _build_analysis_hints(
        self,
        factor_groups: List[Dict[str, Any]],
    ) -> str:
        """根据模板 factor_groups 生成分析要点提示"""
        hints = []
        for group in factor_groups:
            title = group.get("title", "")
            hint = group.get("analysis_hint", "")
            if hint:
                hints.append(f"- {title}：{hint}")
        if not hints:
            return ""
        return "\n## 分析要点提示（模板配置）\n" + "\n".join(hints) + "\n"


# ================================================================
# 基本面分析师
# 借鉴 TradingAgents fundamentals_analyst.py 的 system_message 设计
# ================================================================

class FundamentalsAnalyst:
    """
    基本面 LLM 分析师

    借鉴 TradingAgents fundamentals_analyst 设计：
    - 身份锚定
    - 财务数据真值快照（对应源码的 get_fundamentals/get_balance_sheet 等工具结果）
    - 禁止编造断言

    差异：不调用工具，数据已预计算并注入 prompt

    v2 动态 prompt：根据模板 factor_groups 动态生成指标参考说明。
    """

    def prepare(
        self,
        ctx: Dict[str, Any],
        factor_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        准备 prompt，返回 {system_prompt, user_prompt, response_schema}

        参数:
            ctx: 包含 stock_code, stock_name, current_price, data_date,
                 fundamental_data, industry_data, shareholder_data 的字典
            factor_groups: 模板配置中的因子分组列表（来自 fundamental.yaml），
                           用于动态生成指标参考说明。若为 None 则使用默认 prompt。
        """
        identity = self._build_identity(ctx)
        snapshot = self._build_fundamental_snapshot(ctx)

        # 动态生成因子参考说明
        if factor_groups:
            factor_guide = self._build_factor_guide(factor_groups, _FUNDAMENTAL_FACTOR_DESCRIPTIONS)
            group_analysis_hints = self._build_analysis_hints(factor_groups)
        else:
            factor_guide = ""
            group_analysis_hints = ""

        system_prompt = self._build_system_prompt(factor_guide, group_analysis_hints, bool(factor_groups))

        user_prompt = f"""## 身份锚定
{identity}

## 基本面数据（以此为唯一事实来源）
{snapshot}

## 需要分析的内容
请基于以上基本面数据，输出结构化的基本面分析报告 JSON。"""

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_schema": FUNDAMENTAL_REPORT_SCHEMA,
        }

    def _build_system_prompt(
        self,
        factor_guide: str,
        group_analysis_hints: str,
        is_dynamic: bool,
    ) -> str:
        """构建 system prompt（动态版或静态版）"""
        factor_section = ""
        if factor_guide:
            factor_section = f"\n## 基本面因子参考（模板动态生成）\n{factor_guide}\n"

        return f"""你是一位 A 股基本面分析师，专注于解读已有的财务数据。

## 核心规则（借鉴 TradingAgents 防幻觉设计）
1. 所有财务数据以提供的"基本面数据"为准，不凭空编造
2. 如果某项数据缺失，诚实说明"数据暂缺"，不猜测
3. 不要声称"行业地位领先"或"竞争优势明显"——除非数据中明确提供了相关证据
4. 解读要通俗易懂，面向非专业投资者
5. 提供具体、可操作的洞察，有证据支撑
{factor_section}{group_analysis_hints}
## 扩展章节要求
1. industry_analysis：基于行业数据解读景气度与竞争格局。数据缺失时填空字符串""
2. financial_statement_analysis：解读三大报表关键科目与盈利质量（经营现金流/净利润）。
   数据缺失时填空字符串""
3. shareholder_analysis：A 股特色，解读十大股东变动、解禁、回购等资本运作信号。
   数据缺失时填空字符串""
4. investment_rating：基于综合评分给出 5 档评级（买入/增持/中性/减持/卖出）
   - 买入：fundamental_score >= 80
   - 增持：fundamental_score >= 65
   - 中性：fundamental_score >= 50
   - 减持：fundamental_score >= 35
   - 卖出：fundamental_score < 35

## 输出要求
输出一份结构化的基本面分析报告，严格遵循以下 JSON 格式：
{{
  "valuation_level": "低估|合理|偏高|高估",
  "fundamental_score": 0-100的数字,
  "investment_rating": "买入|增持|中性|减持|卖出",
  "overall_assessment": "整体基本面评估",
  "valuation_analysis": "估值分析（PE/PB分位解读，与历史/同业对比）",
  "profitability_analysis": "盈利能力分析（ROE/毛利率拆解与质量评估）",
  "growth_analysis": "成长性分析（营收/利润增速趋势与可持续性）",
  "risk_factors": "风险因素（高估值/业绩下滑/行业拐点/竞争加剧）",
  "industry_analysis": "行业分析与景气度，或空字符串",
  "financial_statement_analysis": "财务报表分析，或空字符串",
  "shareholder_analysis": "股东结构与资本运作分析，或空字符串"
}}

只输出 JSON，不要其他文字。"""

    def _build_factor_guide(
        self,
        factor_groups: List[Dict[str, Any]],
        descriptions: Dict[str, str],
    ) -> str:
        """根据模板 factor_groups 动态生成因子参考说明"""
        lines = []
        for group in factor_groups:
            group_id = group.get("id", "")
            title = group.get("title", group_id)
            factors = group.get("factors", [])
            lines.append(f"\n### {title}")
            for f in factors:
                desc = descriptions.get(f, f"财务指标 {f}")
                lines.append(f"- {f}: {desc}")
        return "\n".join(lines) if lines else ""

    def _build_analysis_hints(
        self,
        factor_groups: List[Dict[str, Any]],
    ) -> str:
        """根据模板 factor_groups 生成分析要点提示"""
        hints = []
        for group in factor_groups:
            title = group.get("title", "")
            hint = group.get("analysis_hint", "")
            if hint:
                hints.append(f"- {title}：{hint}")
        if not hints:
            return ""
        return "\n## 分析要点提示（模板配置）\n" + "\n".join(hints) + "\n"

    def _build_identity(self, ctx: Dict) -> str:
        return (
            f"标的：{ctx.get('stock_name', '')}（{ctx.get('stock_code', '')}），"
            f"数据截止 {ctx.get('data_date', '')}"
        )

    def _build_fundamental_snapshot(self, ctx: Dict) -> str:
        """构建基本面数据真值快照（借鉴 get_fundamentals 工具结果注入）"""
        fundamental = ctx.get("fundamental_data", {}) or {}
        if not fundamental:
            return "（基本面数据暂缺，请基于已知信息说明数据缺失）"

        lines = []
        for label, keys in [
            ("市盈率 PE(TTM)", ["pe_ttm", "pe", "pe_ratio"]),
            ("市净率 PB", ["pb", "pb_ratio"]),
            ("PE 历史分位", ["pe_percentile", "pe_pct"]),
            ("PB 历史分位", ["pb_percentile", "pb_pct"]),
            ("ROE", ["roe", "roe_ttm"]),
            ("毛利率", ["gross_margin"]),
            ("净利率", ["net_margin"]),
            ("营收增速", ["revenue_growth", "rev_growth"]),
            ("净利润增速", ["profit_growth", "net_profit_growth"]),
            ("总市值(亿)", ["market_cap", "total_mv"]),
            ("PS(TTM)", ["ps_ttm", "ps"]),
            ("股息率", ["dv_ratio", "dividend_yield"]),
            ("ROA", ["roa"]),
            ("营业收入", ["revenue", "total_revenue"]),
            ("净利润", ["net_profit"]),
            ("总资产", ["total_assets"]),
            ("净资产", ["net_assets"]),
            ("资产负债率", ["debt_ratio"]),
            ("流动比率", ["current_ratio"]),
            ("经营现金流", ["operating_cashflow"]),
            ("自由现金流", ["free_cashflow"]),
        ]:
            val = None
            for k in keys:
                if k in fundamental and fundamental[k] is not None:
                    val = fundamental[k]
                    break
            if val is not None:
                lines.append(f"- {label}: {val}")

        # 行业数据
        idt = ctx.get("industry_data", {}) or {}
        if idt:
            lines.append("\n### 行业数据")
            if idt.get("prosperity_index") is not None:
                lines.append(f"- 行业景气度指数: {idt.get('prosperity_index')}")
            if idt.get("prosperity_trend"):
                lines.append(f"- 景气度趋势: {idt.get('prosperity_trend')}")
            if idt.get("industry_growth") is not None:
                lines.append(f"- 行业增速: {idt.get('industry_growth')}")
            if idt.get("market_share") is not None:
                lines.append(f"- 市场份额: {idt.get('market_share')}")
            if idt.get("competition_level"):
                lines.append(f"- 竞争格局: {idt.get('competition_level')}")
            if idt.get("outlook"):
                lines.append(f"- 行业展望: {idt.get('outlook')}")
            if not any(idt.get(k) for k in
                       ["prosperity_index", "prosperity_trend", "industry_growth",
                        "market_share", "competition_level", "outlook"]):
                lines.append("- （行业数据暂缺）")

        # 股东结构数据（A股特色）
        sd = ctx.get("shareholder_data", {}) or {}
        if sd:
            lines.append("\n### 股东结构与资本运作（A股特色）")
            if sd.get("has_data"):
                holders = sd.get("top_holders", []) or []
                if holders:
                    lines.append(f"- 十大股东数量: {len(holders)}")
                    for h in holders[:3]:
                        if isinstance(h, dict):
                            lines.append(
                                f"  • {h.get('name', '—')} 持股={h.get('ratio', '—')} "
                                f"变动={h.get('change', '—')}"
                            )
                if sd.get("upcoming_unlock"):
                    uu = sd["upcoming_unlock"]
                    lines.append(
                        f"- 近期解禁: 日期={uu.get('unlock_date', '—')} "
                        f"比例={uu.get('unlock_ratio', '—')}"
                    )
                if sd.get("buyback"):
                    bk = sd["buyback"]
                    lines.append(
                        f"- 股份回购: 金额={bk.get('amount', '—')} "
                        f"价格区间={bk.get('price_range', '—')}"
                    )
                if sd.get("shareholder_reduction"):
                    sr = sd["shareholder_reduction"]
                    lines.append(f"- 大股东减持: 比例={sr.get('reduction_ratio', '—')}")
            else:
                lines.append("- 股东结构数据暂缺")

        return "\n".join(lines) if lines else "（基本面数据暂缺）"
