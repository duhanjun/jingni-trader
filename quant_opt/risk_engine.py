"""
risk_engine - 事前风控引擎

借鉴来源:
  - NautilusTrader 的 Pre-trade risk checks
    https://nautilustrader.io/docs/latest/concepts/risk_engine/
  - nanobook (ricardofrantz/nanobook) 的 concentration / leverage / short checks
  - NOFX (NoFxAiOS/nofx) 的 Safe Mode: 连续失败 3 次自动熔断

jigni-trader 现状:
  scripts/config.py 中只有简单的风控阈值 (MAX_DAILY_LOSS_RATIO, MAX_SINGLE_STOCK_WEIGHT ...)
  实际并没有一个统一的 RiskEngine 来执行这些检查
  execution-monitor-engine 中也没有"事前校验"环节

本模块提供:
  1. 单笔订单层: 单票最大权重 / 单笔最大金额 / 价格上下限 / 涨跌停熔断
  2. 组合层: 杠杆上限 / 行业集中度 / 现金留存 / 换手率上限
  3. 时间序列层: 日亏损熔断 / 周亏损熔断 / 连续亏损熔断
  4. 健壮性: NaN/Inf 防护 / 数据新鲜度检查
  5. 决策记录: 详细解释每条规则被触发的原因, 便于审计
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scripts.config import (
        MAX_DAILY_LOSS_RATIO, MAX_SINGLE_STOCK_WEIGHT,
        A_SHARE_COMMISSION_RATE, A_SHARE_MIN_LOT,
    )
except Exception:
    # 当以独立模块运行时 (非 jingni-trader 根目录)
    MAX_DAILY_LOSS_RATIO = 0.03
    MAX_SINGLE_STOCK_WEIGHT = 0.10
    A_SHARE_COMMISSION_RATE = 0.00025
    A_SHARE_MIN_LOT = 100

logger = logging.getLogger("quant_opt.risk_engine")


class RiskLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    BLOCK = "block"  # 阻断


@dataclass
class RiskDecision:
    """单条风控决策"""
    rule: str
    level: RiskLevel
    passed: bool
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RiskReport:
    """总风控报告"""
    decisions: List[RiskDecision] = field(default_factory=list)
    n_blocked: int = 0
    n_warn: int = 0
    n_passed: int = 0

    def add(self, decision: RiskDecision):
        self.decisions.append(decision)
        if decision.passed:
            self.n_passed += 1
        elif decision.level == RiskLevel.BLOCK:
            self.n_blocked += 1
        else:
            self.n_warn += 1

    @property
    def blocked(self) -> bool:
        return self.n_blocked > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_blocked": self.n_blocked,
            "n_warn": self.n_warn,
            "n_passed": self.n_passed,
            "blocked": self.blocked,
            "decisions": [asdict(d) for d in self.decisions],
        }


# ============================================================================
# 1. 单笔订单检查
# ============================================================================

def check_single_order(
    order: Dict[str, Any],
    portfolio_value: float,
    holdings: Optional[Dict[str, float]] = None,
    max_weight: float = MAX_SINGLE_STOCK_WEIGHT,
    max_order_ratio: float = 0.02,
    price_limit_buffer: float = 0.0,
) -> RiskDecision:
    """
    单笔订单检查

    order: {code, side, price, shares, amount, is_limit_up?, is_limit_down?}
    """
    holdings = holdings or {}
    code = order.get("code")
    side = str(order.get("side", "")).lower()
    price = float(order.get("price", 0))
    shares = int(order.get("shares", 0))
    amount = float(order.get("amount", price * shares))

    if price <= 0 or shares <= 0:
        return RiskDecision(
            rule="single_order.invalid",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"price={price}, shares={shares} 无效",
        )

    # 1. 最小交易单位
    if shares % A_SHARE_MIN_LOT != 0:
        return RiskDecision(
            rule="single_order.lot_size",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"shares={shares} 不是 {A_SHARE_MIN_LOT} 的整数倍",
        )

    # 2. 价格非有限数
    if not np.isfinite(price):
        return RiskDecision(
            rule="single_order.nan_price",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"价格异常: {price}",
        )

    # 3. 涨跌停熔断
    if order.get("is_limit_up") and side == "buy":
        return RiskDecision(
            rule="single_order.limit_up",
            level=RiskLevel.BLOCK,
            passed=False,
            detail="涨停禁止买入",
        )
    if order.get("is_limit_down") and side == "sell":
        return RiskDecision(
            rule="single_order.limit_down",
            level=RiskLevel.BLOCK,
            passed=False,
            detail="跌停禁止卖出",
        )

    # 4. 单笔金额占比
    if portfolio_value > 0 and amount / portfolio_value > max_order_ratio:
        return RiskDecision(
            rule="single_order.amount_ratio",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"单笔金额 {amount/portfolio_value:.2%} 超过上限 {max_order_ratio:.2%}",
        )

    # 5. 单票最大权重 (买入后) - 仅在 amount_ratio 通过后才检查
    if portfolio_value > 0:
        current = holdings.get(code, 0.0) * price
        if side == "buy":
            post = (current + amount) / (portfolio_value + amount)
        elif side == "sell":
            denom = portfolio_value - amount
            post = ((current - amount) / denom) if denom > 0 else 0.0
        else:
            post = current / portfolio_value
        if post > max_weight:
            return RiskDecision(
                rule="single_order.single_weight",
                level=RiskLevel.BLOCK,
                passed=False,
                detail=f"单票 {code} 持仓权重 {post:.2%} 超过上限 {max_weight:.2%}",
            )

    # 6. 价格上限保护 (例: 偏离 VWAP > 5% 视为异常)
    vwap = order.get("vwap")
    if vwap and vwap > 0:
        deviation = abs(price - vwap) / vwap
        if deviation > 0.05:
            return RiskDecision(
                rule="single_order.price_deviation",
                level=RiskLevel.BLOCK,
                passed=False,
                detail=f"价格 {price} 偏离 VWAP {vwap} {deviation:.2%}",
            )

    return RiskDecision(
        rule="single_order.passed",
        level=RiskLevel.INFO,
        passed=True,
        detail=f"{side} {code} {shares}@{price:.2f} amount={amount:.0f}",
    )


# ============================================================================
# 2. 组合层检查
# ============================================================================

def check_portfolio(
    weights: pd.Series,
    target_weights: Optional[pd.Series] = None,
    industry_map: Optional[Dict[str, str]] = None,
    max_leverage: float = 1.0,
    max_industry_concentration: float = 0.30,
    cash_buffer: float = 0.05,
) -> List[RiskDecision]:
    """
    组合层风控检查
    weights: 当前各持仓权重 (sum <= 1, 现金为 1 - sum(weights))
    target_weights: 调仓后目标权重
    """
    decisions: List[RiskDecision] = []
    weights = weights.fillna(0.0)

    # 1. 杠杆检查
    gross = float(weights.abs().sum())
    if gross > max_leverage + 1e-6:
        decisions.append(RiskDecision(
            rule="portfolio.leverage",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"组合总敞口 {gross:.2%} 超过 {max_leverage:.2%}",
        ))

    # 2. 现金留存
    cash = 1.0 - float(weights.sum())
    if cash < -1e-6:
        decisions.append(RiskDecision(
            rule="portfolio.cash_negative",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"现金为负: {cash:.2%} (权重合计超过 100%)",
        ))
    if cash < cash_buffer:
        decisions.append(RiskDecision(
            rule="portfolio.cash_buffer",
            level=RiskLevel.WARN,
            passed=False,
            detail=f"现金留存 {cash:.2%} 低于缓冲 {cash_buffer:.2%}",
        ))

    # 3. 行业集中度
    if industry_map is not None and len(industry_map) > 0:
        ind_exp = weights.groupby(industry_map).sum()
        max_ind = float(ind_exp.max())
        if max_ind > max_industry_concentration:
            decisions.append(RiskDecision(
                rule="portfolio.industry_concentration",
                level=RiskLevel.BLOCK,
                passed=False,
                detail=f"最大行业暴露 {max_ind:.2%} 超过 {max_industry_concentration:.2%}",
            ))

    # 4. 换手率检查
    if target_weights is not None:
        delta = (target_weights.fillna(0.0) - weights).abs().sum() / 2
        if delta > 0.5:
            decisions.append(RiskDecision(
                rule="portfolio.turnover",
                level=RiskLevel.WARN,
                passed=False,
                detail=f"调仓换手率 {delta:.2%} 超过 50%",
            ))

    if not decisions:
        decisions.append(RiskDecision(
            rule="portfolio.passed",
            level=RiskLevel.INFO,
            passed=True,
            detail=f"组合检查通过: gross={gross:.2%}, cash={cash:.2%}",
        ))

    return decisions


# ============================================================================
# 3. 时序熔断
# ============================================================================

def check_daily_loss(equity_curve: pd.Series,
                     max_daily_loss: float = MAX_DAILY_LOSS_RATIO,
                     max_weekly_loss: float = 0.05,
                     max_consecutive_loss_days: int = 5) -> List[RiskDecision]:
    """日亏损 / 周亏损 / 连续亏损熔断"""
    decisions: List[RiskDecision] = []
    if len(equity_curve) < 2:
        return [RiskDecision("daily_loss.empty", RiskLevel.WARN, False, "equity 数据不足")]

    eq = equity_curve.dropna()
    daily_ret = eq.pct_change().dropna()

    # 日亏损
    worst_day = float(daily_ret.min())
    if worst_day < -max_daily_loss:
        decisions.append(RiskDecision(
            rule="circuit_breaker.daily_loss",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"单日亏损 {worst_day:.2%} 超过上限 {-max_daily_loss:.2%}",
        ))

    # 周亏损 (滚动 5 日累计)
    if len(daily_ret) >= 5:
        weekly = (1 + daily_ret).rolling(5).apply(np.prod, raw=True) - 1
        worst_week = float(weekly.min())
        if worst_week < -max_weekly_loss:
            decisions.append(RiskDecision(
                rule="circuit_breaker.weekly_loss",
                level=RiskLevel.BLOCK,
                passed=False,
                detail=f"5 日累计亏损 {worst_week:.2%} 超过上限 {-max_weekly_loss:.2%}",
            ))

    # 连续亏损
    if len(daily_ret) > 0:
        sign = (daily_ret < 0).astype(int)
        # 最长连续 1
        groups = (sign != sign.shift()).cumsum()
        max_streak = int(sign.groupby(groups).sum().max()) if len(sign) > 0 else 0
        if max_streak >= max_consecutive_loss_days:
            decisions.append(RiskDecision(
                rule="circuit_breaker.consecutive_loss",
                level=RiskLevel.BLOCK,
                passed=False,
                detail=f"连续亏损 {max_streak} 天, 上限 {max_consecutive_loss_days}",
            ))

    if not decisions:
        decisions.append(RiskDecision(
            rule="circuit_breaker.passed",
            level=RiskLevel.INFO,
            passed=True,
            detail=f"日最大亏损 {worst_day:.2%} (上限 {-max_daily_loss:.2%})",
        ))

    return decisions


# ============================================================================
# 4. 数据完整性
# ============================================================================

def check_data_freshness(data: pd.DataFrame,
                          max_age_days: int = 5) -> List[RiskDecision]:
    """检查最新数据时间, 避免用过期数据交易"""
    decisions: List[RiskDecision] = []
    if data.empty or 'date' not in data.columns:
        decisions.append(RiskDecision(
            rule="data.empty",
            level=RiskLevel.BLOCK,
            passed=False,
            detail="data 为空或无 date 列",
        ))
        return decisions

    latest = pd.to_datetime(data['date']).max()
    age = (datetime.now() - latest.to_pydatetime()).days if hasattr(latest, 'to_pydatetime') else \
          (pd.Timestamp.now() - latest).days
    if age > max_age_days:
        decisions.append(RiskDecision(
            rule="data.staleness",
            level=RiskLevel.BLOCK,
            passed=False,
            detail=f"数据最新日期 {latest.date()} 距今 {age} 天, 超过 {max_age_days} 天上限",
        ))

    # NaN/Inf
    for col in ['close', 'open', 'high', 'low', 'volume']:
        if col in data.columns:
            arr = data[col].values
            # 检查是否包含 NaN 或 Inf
            if pd.isna(arr).any() or np.isinf(arr[~pd.isna(arr)]).any():
                decisions.append(RiskDecision(
                    rule=f"data.nan_inf.{col}",
                    level=RiskLevel.BLOCK,
                    passed=False,
                    detail=f"列 {col} 包含 NaN/Inf",
                ))

    if not decisions:
        decisions.append(RiskDecision(
            rule="data.passed",
            level=RiskLevel.INFO,
            passed=True,
            detail=f"数据最新日期 {latest.date()}, {len(data)} 行",
        ))
    return decisions


# ============================================================================
# 5. 统一入口
# ============================================================================

class RiskEngine:
    """统一风控引擎, 执行多层检查并汇总报告"""

    def __init__(self,
                 max_daily_loss: float = MAX_DAILY_LOSS_RATIO,
                 max_single_weight: float = MAX_SINGLE_STOCK_WEIGHT,
                 max_industry_concentration: float = 0.30,
                 max_leverage: float = 1.0,
                 data_max_age_days: int = 5):
        self.max_daily_loss = max_daily_loss
        self.max_single_weight = max_single_weight
        self.max_industry_concentration = max_industry_concentration
        self.max_leverage = max_leverage
        self.data_max_age_days = data_max_age_days

    def pre_trade_check(self, order: Dict[str, Any],
                        portfolio_value: float,
                        holdings: Optional[Dict[str, float]] = None) -> RiskDecision:
        return check_single_order(
            order, portfolio_value, holdings,
            max_weight=self.max_single_weight,
        )

    def pre_batch_check(self, orders: List[Dict[str, Any]],
                        portfolio_value: float,
                        holdings: Optional[Dict[str, float]] = None) -> RiskReport:
        report = RiskReport()
        for o in orders:
            report.add(self.pre_trade_check(o, portfolio_value, holdings))
        return report

    def portfolio_check(self, weights: pd.Series,
                        target_weights: Optional[pd.Series] = None,
                        industry_map: Optional[Dict[str, str]] = None) -> RiskReport:
        report = RiskReport()
        for d in check_portfolio(
            weights, target_weights, industry_map,
            max_leverage=self.max_leverage,
            max_industry_concentration=self.max_industry_concentration,
        ):
            report.add(d)
        return report

    def circuit_breaker_check(self, equity_curve: pd.Series) -> RiskReport:
        report = RiskReport()
        for d in check_daily_loss(equity_curve, self.max_daily_loss):
            report.add(d)
        return report

    def data_freshness_check(self, data: pd.DataFrame) -> RiskReport:
        report = RiskReport()
        for d in check_data_freshness(data, self.data_max_age_days):
            report.add(d)
        return report

    def comprehensive_check(
        self,
        data: pd.DataFrame,
        orders: Optional[List[Dict[str, Any]]] = None,
        portfolio_value: Optional[float] = None,
        holdings: Optional[Dict[str, float]] = None,
        weights: Optional[pd.Series] = None,
        target_weights: Optional[pd.Series] = None,
        industry_map: Optional[Dict[str, str]] = None,
        equity_curve: Optional[pd.Series] = None,
    ) -> RiskReport:
        report = RiskReport()
        for d in check_data_freshness(data, self.data_max_age_days):
            report.add(d)
        if orders and portfolio_value:
            for d in self.pre_batch_check(orders, portfolio_value, holdings).decisions:
                report.add(d)
        if weights is not None:
            for d in check_portfolio(
                weights, target_weights, industry_map,
                max_leverage=self.max_leverage,
                max_industry_concentration=self.max_industry_concentration,
            ):
                report.add(d)
        if equity_curve is not None and len(equity_curve) > 0:
            for d in check_daily_loss(equity_curve, self.max_daily_loss):
                report.add(d)
        return report


# ============================================================================
# 6. CLI 自检
# ============================================================================

def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        engine = RiskEngine()
        # 正常订单
        d1 = engine.pre_trade_check(
            {"code": "600000.SH", "side": "buy", "price": 10.0, "shares": 100, "amount": 1000.0},
            portfolio_value=1_000_000,
        )
        # 100 手 = 10000 股, 10000*10 = 100000 = 10% 持仓, 触发权重熔断
        d2 = engine.pre_trade_check(
            {"code": "600000.SH", "side": "buy", "price": 10.0, "shares": 10000, "amount": 100000.0},
            portfolio_value=1_000_000,
        )
        # NaN price
        d3 = engine.pre_trade_check(
            {"code": "600001.SH", "side": "buy", "price": float("nan"), "shares": 100, "amount": 1000.0},
            portfolio_value=1_000_000,
        )
        print(json.dumps([asdict(d) for d in [d1, d2, d3]], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
