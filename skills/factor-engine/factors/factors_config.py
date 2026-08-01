"""
因子类别开关配置

控制哪些因子类别启用，声明依赖的数据产物。
factor-engine 的 run() 按此配置调度各类因子模块。
"""

FACTOR_CATEGORIES = {
    "momentum": {
        "enabled": True,
        "module": "engine.compute_a_share_factors",
        "requires": "price_data",
        "description": "动量/反转/量价/波动率/资金流代理因子（硬编码在 FactorEngine 中）",
    },
    "technical": {
        "enabled": True,
        "module": "factors.technical_factors",
        "requires": "price_data",
        "description": "技术指标因子：MACD/RSI/KDJ/BOLL/MA/WR/CCI/OBV",
    },
    "financial": {
        "enabled": True,
        "module": "factors.financial_factors",
        "requires": "FINANCIAL",
        "description": "财务因子：ROE/ROA/毛利率/净利率/增速/偿债能力",
    },
    "valuation": {
        "enabled": True,
        "module": "factors.valuation_factors",
        "requires": "FINANCIAL",
        "description": "估值因子：PE/PB/PS/股息率 + 历史分位",
    },
    "capital_flow": {
        "enabled": True,
        "module": "factors.capital_flow_factors",
        "requires": "CAPITAL_FLOW",
        "description": "资金面因子：主力净流入/北向资金",
    },
    "dragon_tiger": {
        "enabled": True,
        "module": "factors.dragon_tiger_factors",
        "requires": "DRAGON_TIGER",
        "description": "龙虎榜因子：上榜次数/净买入额",
    },
    "shareholder": {
        "enabled": True,
        "module": "factors.shareholder_factors",
        "requires": "SHAREHOLDER",
        "description": "股东结构因子：持股变动/集中度",
    },
}
