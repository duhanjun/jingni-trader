"""
因子注册表

所有因子统一注册，携带元数据用于：
1. 报告模板选择渲染器
2. agent 根据 analysis_method 生成对应分析要素
3. 因子发现 → 策略构建时提供因子描述
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FactorMeta:
    """因子元数据"""
    name: str                # 因子名（parquet列名）
    category: str            # momentum/technical/financial/valuation/capital_flow/...
    description: str         # 中文描述
    direction: int           # 1=越大越看多, -1=越大越看空, 0=中性
    render_type: str         # 默认渲染器类型
    analysis_method: str     # percentile/signal/threshold/trend/direction/...
    industry_compare: bool   # 是否需要行业对比


# 因子注册表
FACTOR_REGISTRY: Dict[str, FactorMeta] = {
    # ── 动量/量价因子（已有，由 engine.compute_a_share_factors 计算）──
    "reversal_5d": FactorMeta("reversal_5d", "momentum", "5日反转", -1, "metric_grid", "threshold", False),
    "reversal_20d": FactorMeta("reversal_20d", "momentum", "20日反转", -1, "metric_grid", "threshold", False),
    "lncap": FactorMeta("lncap", "size", "对数市值", 0, "metric_grid", "threshold", False),
    "turnover_20d": FactorMeta("turnover_20d", "liquidity", "20日均换手率", 0, "metric_grid", "threshold", False),
    "turnover_change": FactorMeta("turnover_change", "liquidity", "换手率变化", 0, "metric_grid", "trend", False),
    "volatility_20d": FactorMeta("volatility_20d", "volatility", "20日波动率", -1, "metric_grid", "threshold", False),
    "volume_20d": FactorMeta("volume_20d", "volume", "20日均量", 0, "metric_grid", "threshold", False),
    "volume_ratio": FactorMeta("volume_ratio", "volume", "量比", 0, "metric_grid", "threshold", False),
    "money_flow_20d": FactorMeta("money_flow_20d", "money_flow", "20日累计资金流", 1, "metric_grid", "direction", False),

    # ── 技术因子（新增，由 technical_factors.py 计算）──
    "macd_dif": FactorMeta("macd_dif", "technical", "MACD DIF线", 1, "indicator_panel", "signal", False),
    "macd_dea": FactorMeta("macd_dea", "technical", "MACD DEA线", 1, "indicator_panel", "signal", False),
    "macd_hist": FactorMeta("macd_hist", "technical", "MACD柱", 1, "indicator_panel", "signal", False),
    "rsi_14": FactorMeta("rsi_14", "technical", "RSI(14)", 0, "indicator_panel", "signal", False),
    "kdj_k": FactorMeta("kdj_k", "technical", "KDJ K值", 0, "indicator_panel", "signal", False),
    "kdj_d": FactorMeta("kdj_d", "technical", "KDJ D值", 0, "indicator_panel", "signal", False),
    "kdj_j": FactorMeta("kdj_j", "technical", "KDJ J值", 0, "indicator_panel", "signal", False),
    "boll_ub": FactorMeta("boll_ub", "technical", "布林带上轨", 0, "band_chart", "threshold", False),
    "boll_mid": FactorMeta("boll_mid", "technical", "布林带中轨", 0, "band_chart", "threshold", False),
    "boll_lb": FactorMeta("boll_lb", "technical", "布林带下轨", 0, "band_chart", "threshold", False),
    "ma5": FactorMeta("ma5", "technical", "5日均线", 1, "indicator_panel", "signal", False),
    "ma10": FactorMeta("ma10", "technical", "10日均线", 1, "indicator_panel", "signal", False),
    "ma20": FactorMeta("ma20", "technical", "20日均线", 1, "indicator_panel", "signal", False),
    "ma60": FactorMeta("ma60", "technical", "60日均线", 1, "indicator_panel", "signal", False),
    "wr": FactorMeta("wr", "technical", "WR指标", 0, "indicator_panel", "signal", False),
    "cci": FactorMeta("cci", "technical", "CCI指标", 0, "indicator_panel", "signal", False),
    "obv": FactorMeta("obv", "technical", "OBV指标", 1, "indicator_panel", "direction", False),

    # ── 财务因子（新增，由 financial_factors.py 计算）──
    "roe_ttm": FactorMeta("roe_ttm", "financial", "ROE(TTM)", 1, "metric_grid", "industry_compare", True),
    "roa_ttm": FactorMeta("roa_ttm", "financial", "ROA(TTM)", 1, "metric_grid", "industry_compare", True),
    "gross_margin_ttm": FactorMeta("gross_margin_ttm", "financial", "毛利率(TTM)", 1, "metric_grid", "industry_compare", True),
    "net_margin_ttm": FactorMeta("net_margin_ttm", "financial", "净利率(TTM)", 1, "metric_grid", "industry_compare", True),
    "revenue_growth_yoy": FactorMeta("revenue_growth_yoy", "growth", "营收同比增速", 1, "trend_chart", "trend", False),
    "profit_growth_yoy": FactorMeta("profit_growth_yoy", "growth", "利润同比增速", 1, "trend_chart", "trend", False),
    "debt_ratio": FactorMeta("debt_ratio", "financial", "资产负债率", -1, "metric_grid", "threshold", False),
    "current_ratio": FactorMeta("current_ratio", "financial", "流动比率", 1, "metric_grid", "threshold", False),
    "ocf": FactorMeta("ocf", "financial", "经营现金流", 1, "metric_grid", "direction", False),

    # ── 估值因子（新增，由 valuation_factors.py 计算）──
    "pe_ttm": FactorMeta("pe_ttm", "valuation", "市盈率TTM", -1, "percentile_chart", "percentile", False),
    "pb": FactorMeta("pb", "valuation", "市净率", -1, "percentile_chart", "percentile", False),
    "ps_ttm": FactorMeta("ps_ttm", "valuation", "市销率TTM", -1, "percentile_chart", "percentile", False),
    "dv_ratio": FactorMeta("dv_ratio", "valuation", "股息率", 1, "metric_grid", "threshold", False),
    "pe_percentile": FactorMeta("pe_percentile", "valuation", "PE历史分位", -1, "percentile_chart", "percentile", False),
    "pb_percentile": FactorMeta("pb_percentile", "valuation", "PB历史分位", -1, "percentile_chart", "percentile", False),

    # ── 资金面因子（新增，由 capital_flow_factors.py 计算）──
    "main_net_inflow": FactorMeta("main_net_inflow", "capital_flow", "主力净流入", 1, "flow_table", "direction", False),
    "main_net_inflow_5d": FactorMeta("main_net_inflow_5d", "capital_flow", "5日主力净流入均值", 1, "flow_table", "direction", False),
    "north_net_inflow": FactorMeta("north_net_inflow", "capital_flow", "北向资金净流入", 1, "flow_table", "direction", False),

    # ── 龙虎榜因子（新增，由 dragon_tiger_factors.py 计算）──
    "lhb_count_5d": FactorMeta("lhb_count_5d", "dragon_tiger", "5日上榜次数", 1, "event_list", "event", False),
    "lhb_net_buy": FactorMeta("lhb_net_buy", "dragon_tiger", "龙虎榜净买入额", 1, "event_list", "event", False),

    # ── 股东结构因子（新增，由 shareholder_factors.py 计算）──
    "top_holder_change": FactorMeta("top_holder_change", "shareholder", "十大股东变动", 0, "holder_table", "change", False),
    "holder_concentration": FactorMeta("holder_concentration", "shareholder", "持股集中度", 0, "holder_table", "change", False),

    # ── 复合因子 ──
    "alpha_score": FactorMeta("alpha_score", "composite", "多因子融合Alpha", 1, "metric_grid", "threshold", False),
}


def get_factor_meta(name: str) -> Optional[FactorMeta]:
    """查询因子元数据"""
    return FACTOR_REGISTRY.get(name)


def get_factors_by_category(category: str) -> Dict[str, FactorMeta]:
    """按类别获取因子"""
    return {k: v for k, v in FACTOR_REGISTRY.items() if v.category == category}
