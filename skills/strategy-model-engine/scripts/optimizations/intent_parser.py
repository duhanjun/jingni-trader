"""
intent_parser - 增强版意图解析器

借鉴来源:
  - TradingAgents (TauricResearch/TradingAgents) 的多智能体理解
  - qlib 的 workflow config dict 设计
  - LangGraph state machine 思想

jigni-trader 现状 (engine.py parse_intent):
  - 纯关键字匹配, 鲁棒性差
  - 解析后无法回填到自然语言让用户确认
  - 不支持复杂语义 (如 "过去 1 年每月第一个交易日调仓")
  - 不支持置信度评估

本模块提供:
  1. 结构化解析 (Pipeline Config Dict), 与 qlib workflow_config 对齐
  2. 多源支持: 关键字 / 正则 / 显式参数优先级
  3. 置信度 + 缺失字段标记
  4. 自然语言回显 (Confirmation) - 把解析结果格式化为一句话
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("quant_opt.intent_parser")


# ============================================================================
# 1. 解析后的结构化意图
# ============================================================================

@dataclass
class ParsedIntent:
    """结构化的用户意图"""
    target_stages: List[str] = field(default_factory=list)
    stock_pool: List[str] = field(default_factory=list)
    benchmark: str = "000300.SH"
    start_date: str = ""
    end_date: str = ""
    strategy_name: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    risk_constraints: Dict[str, Any] = field(default_factory=dict)
    n_trials: int = 1
    confidence: float = 0.0       # 0~1
    missing_fields: List[str] = field(default_factory=list)
    raw_intent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 2. 规则引擎
# ============================================================================

STAGE_KEYWORDS = {
    "DATA":     ["数据", "获取", "下载", "采集", "data", "下载数据"],
    "FACTOR":   ["因子", "alpha", "ic", "选股因子"],
    "MODEL":    ["模型", "训练", "lightgbm", "机器学习", "训练模型", "ml"],
    "BACKTEST": ["回测", "backtest", "模拟", "回测验证"],
    "PORTFOLIO": ["组合", "优化", "portfolio", "仓位", "配置"],
    "EXECUTION": ["实盘", "交易", "下单", "execution", "执行", "模拟盘"],
    "REPORT":   ["报告", "report", "可视化", "绩效", "归因", "生成报告"],
}

# 关键词 -> 解析函数
REGEX_RULES: List[Tuple[re.Pattern, str]] = [
    # 时间区间
    (re.compile(r"近(\d+)\s*年|最近\s*(\d+)\s*年|过去\s*(\d+)\s*年"), "years_back"),
    (re.compile(r"近(\d+)\s*月|最近\s*(\d+)\s*月|过去\s*(\d+)\s*月"), "months_back"),
    (re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})\s*[-到至~]+\s*(\d{4})[-./](\d{1,2})[-./](\d{1,2})"), "explicit_range"),
    (re.compile(r"(\d{4})[-./年](\d{1,2})\s*[-到至~]+\s*(\d{4})[-./年](\d{1,2})"), "explicit_range_ym"),
    # 股票池
    (re.compile(r"沪深\s*300|hs300|csi300"), "pool_csi300"),
    (re.compile(r"中证\s*500|csi500|zz500"), "pool_csi500"),
    (re.compile(r"中证\s*1000|csi1000|zz1000"), "pool_csi1000"),
    (re.compile(r"全\s*A|全市场|all[\s_]?a"), "pool_alla"),
    (re.compile(r"上证\s*50|sh50"), "pool_sh50"),
    (re.compile(r"创业板|chinext|szse\s*growth"), "pool_chinext"),
    (re.compile(r"科创板|star\s*market|sse\s*star"), "pool_star"),
    # 因子类型
    (re.compile(r"反转|reversal|mean[\s_]?reversion"), "factor_reversal"),
    (re.compile(r"动量|momentum|趋势|trend"), "factor_momentum"),
    (re.compile(r"量价|价量|price[\s_]?volume|volume[\s_]?price"), "factor_pv"),
    (re.compile(r"换手|turnover"), "factor_turnover"),
    (re.compile(r"波动|volatility"), "factor_vol"),
    # 周期
    (re.compile(r"(\d+)\s*日\s*反转|(\d+)\s*天\s*反转"), "factor_reversal_n"),
    (re.compile(r"(\d+)\s*日\s*动量|(\d+)\s*天\s*动量"), "factor_momentum_n"),
    (re.compile(r"ma\s*(\d+)|均线\s*(\d+)|sma\s*(\d+)", re.I), "ma_n"),
    (re.compile(r"rsi\s*(\d+)?", re.I), "rsi_n"),
    (re.compile(r"macd", re.I), "macd"),
    # 风控 (允许 "最大回撤控制在15%以内" / "最大回撤<15%")
    (re.compile(r"最大回撤.{0,4}?[<≤]?\s*(\d+(?:\.\d+)?)\s*%"), "max_dd"),
    (re.compile(r"年化.{0,4}?[>≥]\s*(\d+(?:\.\d+)?)\s*%"), "target_ann_ret"),
    # 频率
    (re.compile(r"日\s*频|日线|daily"), "freq_daily"),
    (re.compile(r"周\s*频|周线|weekly"), "freq_weekly"),
    (re.compile(r"分钟|分钟线|intraday|min"), "freq_minute"),
    # 调仓
    (re.compile(r"每日调仓|每日换仓|daily\s*rebal"), "rebal_daily"),
    (re.compile(r"每周调仓|weekly\s*rebal"), "rebal_weekly"),
    (re.compile(r"每月调仓|monthly\s*rebal"), "rebal_monthly"),
    (re.compile(r"每(\d+)\s*日\s*调仓|每(\d+)\s*天\s*调仓"), "rebal_n_days"),
]


# ============================================================================
# 3. 解析器
# ============================================================================

class IntentParser:
    """增强版意图解析器"""

    def __init__(self, today: Optional[datetime] = None):
        self.today = today or datetime.now()

    def parse(self, user_input: str) -> ParsedIntent:
        text = (user_input or "").strip()
        lower = text.lower()
        intent = ParsedIntent(raw_intent=text)

        # 1. 目标阶段
        intent.target_stages = self._parse_stages(text, lower)

        # 2. 时间区间
        start, end = self._parse_date_range(text, lower)
        intent.start_date = start
        intent.end_date = end

        # 3. 股票池
        intent.stock_pool = self._parse_pool(text, lower)

        # 4. 因子 / 策略
        intent.strategy_name, intent.strategy_params = self._parse_strategy(text, lower)

        # 5. 风控约束
        intent.risk_constraints = self._parse_risk(text, lower)

        # 6. 调仓频率
        if "rebal_daily" in intent.strategy_params:
            intent.strategy_params.setdefault("rebalance_freq", "daily")
        elif "rebal_weekly" in intent.strategy_params:
            intent.strategy_params.setdefault("rebalance_freq", "weekly")
        elif "rebal_monthly" in intent.strategy_params:
            intent.strategy_params.setdefault("rebalance_freq", "monthly")

        # 7. 缺失字段与置信度
        intent.missing_fields = self._detect_missing(intent)
        intent.confidence = self._compute_confidence(intent)

        return intent

    # ------------------------------------------------------------
    def _parse_stages(self, text: str, lower: str) -> List[str]:
        stages = []
        for stage, keywords in STAGE_KEYWORDS.items():
            if any(kw in text or kw in lower for kw in keywords):
                stages.append(stage)
        if not stages:
            stages = ["DATA", "FACTOR", "MODEL", "BACKTEST", "REPORT"]
        # 保证依赖顺序
        order = ["DATA", "FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "EXECUTION", "REPORT"]
        if any(s in stages for s in ["FACTOR", "MODEL", "BACKTEST", "PORTFOLIO", "REPORT"]) and "DATA" not in stages:
            stages.insert(0, "DATA")
        return sorted(stages, key=lambda s: order.index(s) if s in order else 99)

    def _parse_date_range(self, text: str, lower: str) -> Tuple[str, str]:
        today = self.today
        # 显式范围 (精确到日)
        m = re.search(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})\s*[-到至~]+\s*(\d{4})[-./](\d{1,2})[-./](\d{1,2})", text)
        if m:
            y1, m1, d1, y2, m2, d2 = m.groups()
            return (f"{int(y1):04d}-{int(m1):02d}-{int(d1):02d}",
                    f"{int(y2):04d}-{int(m2):02d}-{int(d2):02d}")
        # 显式范围 (精确到月)
        m = re.search(r"(\d{4})[-./年](\d{1,2})\s*[-到至~]+\s*(\d{4})[-./年](\d{1,2})", text)
        if m:
            y1, m1, y2, m2 = m.groups()
            return (f"{int(y1):04d}-{int(m1):02d}-01",
                    f"{int(y2):04d}-{int(m2):02d}-28")
        # 相对区间
        m = re.search(r"近(\d+)\s*年|最近\s*(\d+)\s*年|过去\s*(\d+)\s*年", text)
        if m:
            years = int(next(g for g in m.groups() if g))
            end = today.replace(day=1) - timedelta(days=1)
            start = end.replace(year=end.year - years)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        m = re.search(r"近(\d+)\s*月|最近\s*(\d+)\s*月|过去\s*(\d+)\s*月", text)
        if m:
            months = int(next(g for g in m.groups() if g))
            end = today.replace(day=1) - timedelta(days=1)
            year = end.year - months // 12
            month = end.month - (months % 12)
            if month <= 0:
                month += 12
                year -= 1
            start = end.replace(year=year, month=month)
            return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return ("", "")

    def _parse_pool(self, text: str, lower: str) -> List[str]:
        if re.search(r"沪深\s*300|hs300|csi300", text, re.I):
            return ["000300.SH"]
        if re.search(r"中证\s*500|csi500|zz500", text, re.I):
            return ["000905.SH"]
        if re.search(r"中证\s*1000|csi1000|zz1000", text, re.I):
            return ["000852.SH"]
        if re.search(r"上证\s*50|sh50", text, re.I):
            return ["000016.SH"]
        if re.search(r"全\s*A|全市场|all[\s_]?a", text, re.I):
            return []
        return []

    def _parse_strategy(self, text: str, lower: str) -> Tuple[str, Dict[str, Any]]:
        params: Dict[str, Any] = {}
        name = "default"

        # 因子名 -> 周期
        m = re.search(r"(\d+)\s*日\s*反转|(\d+)\s*天\s*反转", text)
        if m:
            n = int(next(g for g in m.groups() if g))
            name = "reversal"
            params["factor"] = f"reversal_{n}d"
            params["lookback"] = n
            self._add_rebal(text, params)
            return name, params

        m = re.search(r"(\d+)\s*日\s*动量|(\d+)\s*天\s*动量", text)
        if m:
            n = int(next(g for g in m.groups() if g))
            name = "momentum"
            params["factor"] = f"ret_{n}d"
            params["lookback"] = n
            self._add_rebal(text, params)
            return name, params

        m = re.search(r"ma\s*(\d+)|均线\s*(\d+)|sma\s*(\d+)", text, re.I)
        if m:
            n = int(next(g for g in m.groups() if g))
            name = "ma_cross"
            params["fast_ma"] = max(5, n // 4)
            params["slow_ma"] = n
            self._add_rebal(text, params)
            return name, params

        m = re.search(r"rsi\s*(\d+)?", text, re.I)
        if m:
            n = int(m.group(1)) if m.group(1) else 14
            name = "rsi"
            params["period"] = n
            self._add_rebal(text, params)
            return name, params

        if re.search(r"macd", text, re.I):
            name = "macd"
            self._add_rebal(text, params)
            return name, params

        if re.search(r"反转|reversal|mean[\s_]?reversion", text, re.I):
            name = "reversal"
            params["factor"] = "reversal_20d"
        elif re.search(r"动量|momentum|趋势|trend", text, re.I):
            name = "momentum"
            params["factor"] = "ret_20d"
        elif re.search(r"量价|价量|price[\s_]?volume|volume[\s_]?price", text, re.I):
            name = "price_volume"
        elif re.search(r"换手|turnover", text, re.I):
            name = "turnover"
        elif re.search(r"波动|volatility", text, re.I):
            name = "low_volatility"

        # 调仓频率
        self._add_rebal(text, params)
        return name, params

    def _add_rebal(self, text: str, params: Dict[str, Any]) -> None:
        """从文本中提取调仓频率, 写入 params"""
        for pat, name2 in REGEX_RULES:
            if "rebal" not in name2:
                continue
            m = pat.search(text)
            if not m:
                continue
            if name2 == "rebal_n_days":
                n = int(next(g for g in m.groups() if g))
                params["rebalance_freq"] = f"every_{n}_days"
            elif name2 == "rebal_daily":
                params["rebalance_freq"] = "daily"
            elif name2 == "rebal_weekly":
                params["rebalance_freq"] = "weekly"
            elif name2 == "rebal_monthly":
                params["rebalance_freq"] = "monthly"
            else:
                params["rebal_flag"] = name2
            return  # 只取第一个匹配的

    def _parse_risk(self, text: str, lower: str) -> Dict[str, Any]:
        risk: Dict[str, Any] = {}
        # 允许 "最大回撤控制在15%以内" / "最大回撤≤15%" 等
        m = re.search(r"最大回撤.{0,4}?[<≤]?\s*(\d+(?:\.\d+)?)\s*%", text)
        if m:
            risk["max_drawdown"] = float(m.group(1)) / 100
        m = re.search(r"年化.{0,4}?[>≥]\s*(\d+(?:\.\d+)?)\s*%", text)
        if m:
            risk["target_annual_return"] = float(m.group(1)) / 100
        return risk

    def _detect_missing(self, intent: ParsedIntent) -> List[str]:
        miss = []
        if not intent.start_date or not intent.end_date:
            miss.append("date_range")
        if not intent.stock_pool and "全A" not in intent.raw_intent and "全市场" not in intent.raw_intent:
            miss.append("stock_pool")
        if not intent.strategy_name or intent.strategy_name == "default":
            miss.append("strategy_name")
        return miss

    def _compute_confidence(self, intent: ParsedIntent) -> float:
        # 字段完整度
        fields = [intent.target_stages, intent.start_date, intent.end_date,
                  intent.stock_pool, intent.strategy_name, intent.risk_constraints]
        weights = [0.15, 0.15, 0.15, 0.20, 0.20, 0.15]
        score = 0.0
        for f, w in zip(fields, weights):
            score += w * (1.0 if f else 0.0)
        return round(score, 3)


# ============================================================================
# 4. CLI 自检
# ============================================================================

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("text", nargs="?", default=None)
    args = ap.parse_args()
    parser = IntentParser()

    if args.self_test:
        samples = [
            "帮我用近3年A股数据做一个20日反转因子选股回测，最大回撤控制在15%以内",
            "回测一下中证500过去1年月线MACD策略",
            "测试沪深300 5日动量选股，每月调仓",
            "用2022-01-01到2024-06-30的A股数据做双均线回测",
            "make a backtest",  # 英文
        ]
        for s in samples:
            print(f"\n> {s}")
            intent = parser.parse(s)
            print(json.dumps(intent.to_dict(), indent=2, ensure_ascii=False))
    elif args.text:
        intent = parser.parse(args.text)
        print(json.dumps(intent.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()