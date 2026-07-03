"""
多层风控引擎 (Multi-Layer Risk Engine)
====================================

借鉴项目：nautechsystems/nautilus_trader
    - 官方文档: https://nautilustrader.io/docs/latest/concepts/execution/
    - 架构：RiskEngine -> ExecutionEngine -> ExecutionClient
    - 关键设计：TradingState (ACTIVE / HALTED / REDUCING) + 预交易校验链 +
               Throttler + 拒绝事件带原因码

借鉴点：
    1. 引入 trading_state 状态机（jingni-trader 现有 CircuitBreaker
       只做单一检查，缺少全局交易状态）
    2. 把"日亏止损 / 单笔限额 / 频率 / 行业偏离"等多维约束
       整合到一个 RiskEngine 中，输出 OrderDenied 风格的原因
    3. 提供 HALT 状态 + 解除 HALT 的人工通道（kill switch）
    4. 所有拒绝事件写入审计日志（JSONL）

设计目标：
    - 完全保留与现有 CircuitBreaker.check_send_order 同形接口
    - 增加更多细粒度控制（instrument 级、portfolio 级）
    - 用 dataclass 表达 OrderDenied 事件，便于上层订阅
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =====================================================================
# 枚举与事件
# =====================================================================


class TradingState(str, Enum):
    """全局交易状态机"""
    ACTIVE = "ACTIVE"          # 正常
    REDUCING = "REDUCING"      # 仅允许减仓
    HALTED = "HALTED"          # 完全停止


class DenialReason(str, Enum):
    NONE = "NONE"
    STATE_HALTED = "STATE_HALTED"
    STATE_REDUCE_ONLY = "STATE_REDUCE_ONLY"
    DAILY_LOSS = "DAILY_LOSS"
    SINGLE_ORDER_SIZE = "SINGLE_ORDER_SIZE"
    FREQUENCY = "FREQUENCY"
    PER_INSTRUMENT_NOTIONAL = "PER_INSTRUMENT_NOTIONAL"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    REDUCE_ONLY_FAIL = "REDUCE_ONLY_FAIL"
    POSITION_LIMIT = "POSITION_LIMIT"
    INDUSTRY_DEVIATION = "INDUSTRY_DEVIATION"


@dataclass
class OrderDecision:
    """风控决策结果"""
    allowed: bool
    reason: DenialReason = DenialReason.NONE
    reason_detail: str = ""
    trading_state: TradingState = TradingState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trading_state"] = self.trading_state.value
        d["reason"] = self.reason.value
        return d


@dataclass
class AccountSnapshot:
    """风控引擎所见的账户状态（与 PaperExecutor 解耦）"""
    nav: float
    available_cash: float
    start_of_day_nav: float
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    industry_map: Dict[str, str] = field(default_factory=dict)
    benchmark_industry_weights: Dict[str, float] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)


# =====================================================================
# 节流器
# =====================================================================


class TokenBucketThrottler:
    """
    令牌桶节流器：用于限制下单/撤单频率
    借鉴 NautilusTrader 的 max_order_submit_rate / max_order_modify_rate
    """

    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None):
        self.rate = rate_per_sec
        self.capacity = capacity or max(1, int(rate_per_sec))
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def allow(self) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


# =====================================================================
# 主风控引擎
# =====================================================================


class MultiLayerRiskEngine:
    """
    多层风控引擎

    校验顺序（自上而下，与 NautilusTrader RiskEngine 一致）：
        1. trading_state（全局状态机）
        2. 标的有效性 / reduce_only
        3. 资金/数量
        4. 单笔最大金额
        5. 单标的最大持仓
        6. 日亏止损
        7. 行业偏离
        8. 频率节流
    """

    def __init__(
        self,
        max_daily_loss_ratio: float = 0.03,
        max_single_order_ratio: float = 0.02,
        max_single_stock_weight: float = 0.10,
        max_industry_deviation: float = 0.05,
        max_orders_per_sec: float = 5.0,
        max_notional_per_instrument: Optional[Dict[str, float]] = None,
        audit_log_path: Optional[str] = None,
    ):
        self.max_daily_loss_ratio = max_daily_loss_ratio
        self.max_single_order_ratio = max_single_order_ratio
        self.max_single_stock_weight = max_single_stock_weight
        self.max_industry_deviation = max_industry_deviation
        self.max_orders_per_sec = max_orders_per_sec
        self.max_notional_per_instrument = max_notional_per_instrument or {}

        self.trading_state: TradingState = TradingState.ACTIVE
        self.submit_throttler = TokenBucketThrottler(rate_per_sec=max_orders_per_sec)
        self.modify_throttler = TokenBucketThrottler(rate_per_sec=max_orders_per_sec * 2)

        self.audit_log_path = audit_log_path
        self._decisions: List[OrderDecision] = []
        self._denied_count: int = 0
        self._allowed_count: int = 0

    # -------- 状态机控制 --------

    def halt(self, reason: str = "manual"):
        """进入 HALTED 状态：拒绝所有新订单"""
        self.trading_state = TradingState.HALTED
        self._audit("HALT", "", reason)

    def reduce_only(self, reason: str = ""):
        """进入 REDUCING 状态：仅允许减仓"""
        self.trading_state = TradingState.REDUCING
        self._audit("REDUCE_ONLY", "", reason)

    def resume(self):
        """恢复正常 ACTIVE 状态"""
        self.trading_state = TradingState.ACTIVE
        self._audit("RESUME", "", "")

    # -------- 校验主入口 --------

    def check_order(
        self,
        code: str,
        side: str,
        volume: int,
        price: float,
        account: AccountSnapshot,
    ) -> OrderDecision:
        """
        校验单笔订单
        """
        order_value = abs(volume) * price

        # 1) 全局状态
        if self.trading_state == TradingState.HALTED:
            return self._deny(
                DenialReason.STATE_HALTED,
                f"交易已停止（HALTED），拒绝 {side} {code}",
                account,
            )
        if self.trading_state == TradingState.REDUCING and side == "buy":
            return self._deny(
                DenialReason.STATE_REDUCE_ONLY,
                f"REDUCING 状态仅允许卖出，拒绝买入 {code}",
                account,
            )

        # 2) 资金 / reduce_only
        if side == "buy":
            if order_value > account.available_cash:
                return self._deny(
                    DenialReason.INSUFFICIENT_CASH,
                    f"资金不足：需 {order_value:.0f}，可用 {account.available_cash:.0f}",
                    account,
                )
        else:  # sell
            current = account.positions.get(code, {}).get("volume", 0)
            if volume > current:
                return self._deny(
                    DenialReason.REDUCE_ONLY_FAIL,
                    f"减仓数量 {volume} 超过持仓 {current}",
                    account,
                )

        # 3) 单笔金额上限
        if order_value > account.nav * self.max_single_order_ratio:
            return self._deny(
                DenialReason.SINGLE_ORDER_SIZE,
                f"单笔金额 {order_value:.0f} 超过上限 {account.nav * self.max_single_order_ratio:.0f}",
                account,
            )

        # 4) 单标的最大持仓（按 NAV 占比）
        if side == "buy":
            future_pos = account.positions.get(code, {}).get("volume", 0) + volume
            future_value = future_pos * price
            if account.nav > 0 and future_value / account.nav > self.max_single_stock_weight:
                return self._deny(
                    DenialReason.POSITION_LIMIT,
                    f"单票 {code} 持仓将达 {future_value / account.nav:.1%}，"
                    f"超过上限 {self.max_single_stock_weight:.1%}",
                    account,
                )

        # 5) 单标的名义上限（如设置）
        if code in self.max_notional_per_instrument:
            if order_value > self.max_notional_per_instrument[code]:
                return self._deny(
                    DenialReason.PER_INSTRUMENT_NOTIONAL,
                    f"标的 {code} 单笔 {order_value:.0f} 超过 {self.max_notional_per_instrument[code]:.0f}",
                    account,
                )

        # 6) 日亏止损
        if account.start_of_day_nav > 0:
            daily_return = (account.nav - account.start_of_day_nav) / account.start_of_day_nav
            if daily_return <= -self.max_daily_loss_ratio:
                return self._deny(
                    DenialReason.DAILY_LOSS,
                    f"日亏 {daily_return:.2%} 超过阈值 {-self.max_daily_loss_ratio:.2%}",
                    account,
                )

        # 7) 行业偏离（仅 buy 时检查）
        if side == "buy" and account.industry_map and account.benchmark_industry_weights:
            ind = account.industry_map.get(code, "其他")
            cur_ind_w = self._calc_industry_weight(account, ind)
            bench_w = account.benchmark_industry_weights.get(ind, 0.0)
            new_total_nav = account.nav  # 简化：下单前后 NAV 视作不变
            new_ind_w = cur_ind_w + order_value / new_total_nav
            if abs(new_ind_w - bench_w) > self.max_industry_deviation:
                return self._deny(
                    DenialReason.INDUSTRY_DEVIATION,
                    f"行业 {ind} 偏离 {new_ind_w - bench_w:.1%} 超过阈值 "
                    f"{self.max_industry_deviation:.1%}",
                    account,
                )

        # 8) 频率节流
        if not self.submit_throttler.allow():
            return self._deny(
                DenialReason.FREQUENCY,
                f"下单频率超过 {self.max_orders_per_sec} 次/秒",
                account,
            )

        return self._allow(account)

    # -------- 工具方法 --------

    def _deny(self, reason: DenialReason, detail: str,
              account: AccountSnapshot) -> OrderDecision:
        self._denied_count += 1
        decision = OrderDecision(
            allowed=False, reason=reason, reason_detail=detail,
            trading_state=self.trading_state,
        )
        self._decisions.append(decision)
        self._audit("DENY", reason.value, detail)
        return decision

    def _allow(self, account: AccountSnapshot) -> OrderDecision:
        self._allowed_count += 1
        decision = OrderDecision(
            allowed=True, reason=DenialReason.NONE, reason_detail="",
            trading_state=self.trading_state,
        )
        self._decisions.append(decision)
        self._audit("ALLOW", "", "")
        return decision

    def _calc_industry_weight(self, account: AccountSnapshot, industry: str) -> float:
        if account.nav <= 0:
            return 0.0
        ind_value = 0.0
        for code, pos in account.positions.items():
            if account.industry_map.get(code) == industry:
                price = account.prices.get(code, pos.get("avg_cost", 0.0))
                ind_value += pos.get("volume", 0) * price
        return ind_value / account.nav

    def _audit(self, kind: str, reason: str, detail: str):
        if not self.audit_log_path:
            return
        os.makedirs(os.path.dirname(self.audit_log_path) or ".", exist_ok=True)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "kind": kind,
            "reason": reason,
            "detail": detail,
            "trading_state": self.trading_state.value,
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -------- 统计 --------

    def stats(self) -> Dict[str, Any]:
        return {
            "trading_state": self.trading_state.value,
            "allowed": self._allowed_count,
            "denied": self._denied_count,
            "denial_reasons": self._reason_breakdown(),
            "decisions_tail": [d.to_dict() for d in self._decisions[-10:]],
        }

    def _reason_breakdown(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d in self._decisions:
            if not d.allowed:
                out[d.reason.value] = out.get(d.reason.value, 0) + 1
        return out


__all__ = [
    "TradingState", "DenialReason", "OrderDecision", "AccountSnapshot",
    "TokenBucketThrottler", "MultiLayerRiskEngine",
]
