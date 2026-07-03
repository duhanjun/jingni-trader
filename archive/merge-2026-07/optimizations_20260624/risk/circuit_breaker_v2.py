"""
断路器 v2 —— 优化验证版

借鉴来源：
  - vn.py vnpy_riskmanager: 规则化风控引擎（ActiveOrderRule / DailyLimitRule / OrderSizeRule 等
    独立规则类，Cython 编译实现微秒级检查）
  - 生产级断路器最佳实践（ssystechsoftwares.com / nadcab.com）：
      1. 滞回（Hysteresis）：触发阈值与恢复阈值不同，避免在阈值附近反复抖动
      2. Fail-open 语义：自身异常时默认放行，避免风控 bug 拖垮整个系统
      3. 最小样本量：样本不足时不触发，避免小样本噪声误杀
      4. JSON 持久化：状态跨进程重启可恢复

相对 jingni-trader main 分支 execution-monitor-engine/engine.py CircuitBreaker 的改进点：
  1. 增加滞回机制：trip_threshold（如 -5%）与 recover_threshold（如 +0%）分离，
     避免旧版单阈值在临界点反复触发/恢复导致抖动
  2. 增加 fail-open 语义：check_send_order 内部异常时返回 allowed=True，
     避免风控自身 bug 阻断全部交易（旧版无异常保护）
  3. 增加最小样本量：rolling_window 内交易数 < min_samples 时不触发断路，
     避免小样本噪声误杀（旧版无此保护）
  4. 增加 JSON 状态持久化：save_state / load_state 支持跨进程恢复（旧版仅内存态）
  5. 增加多日滚动 PnL 评估：旧版仅看单日 daily_return，新版支持 N 日滚动 ROI
  6. 保留并修正旧版单日亏损检查逻辑（经复核旧版方向逻辑正确，但缺乏滞回保护）

注意：经仔细复核，main 分支 CircuitBreaker.check_send_order 中
  `checks["daily_loss"] = daily_return > -MAX_DAILY_LOSS_RATIO`
  逻辑方向实际是正确的（return 高于 -阈值 即未超亏 → 放行），
  本次不改动该方向，仅在其基础上增加滞回 / fail-open / 持久化 / 滚动评估。
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger("circuit_breaker_v2")


class CircuitBreakerV2:
    """带滞回 / fail-open / 持久化的风控断路器 v2。

    参数
    ----
    daily_loss_limit : float
        单日亏损触发阈值（正数，如 0.05 表示 -5% 触发断路）。
    recover_threshold : float
        恢复阈值（正数，如 0.0 表示 ROI 回正才恢复）。
        必须大于 -daily_loss_limit，否则无滞回效果。
    rolling_window : int
        滚动评估窗口天数。
    min_samples : int
        窗口内最小交易/样本数，不足则不触发断路（避免小样本噪声）。
    max_order_frequency : int
        每秒最大下单数。
    state_path : Optional[str]
        JSON 状态持久化路径，None 表示不持久化。
    """

    def __init__(
        self,
        daily_loss_limit: float = 0.05,
        recover_threshold: float = 0.0,
        rolling_window: int = 5,
        min_samples: int = 3,
        max_order_frequency: int = 5,
        state_path: Optional[str] = None,
    ):
        if recover_threshold <= -daily_loss_limit:
            raise ValueError(
                f"recover_threshold({recover_threshold}) 必须大于 -daily_loss_limit({-daily_loss_limit})，"
                f"否则无滞回效果"
            )
        self.daily_loss_limit = daily_loss_limit
        self.recover_threshold = recover_threshold
        self.rolling_window = rolling_window
        self.min_samples = min_samples
        self.max_order_frequency = max_order_frequency
        self.state_path = state_path

        # 运行时状态
        self.tripped: bool = False
        self.trip_reason: str = ""
        self.trip_time: Optional[float] = None
        self.last_order_times: List[float] = []
        self.daily_returns: List[float] = []  # 滚动窗口内的日收益序列

        if state_path and os.path.exists(state_path):
            self.load_state()

    # ------------------------------------------------------------------
    # 核心检查
    # ------------------------------------------------------------------

    def check_send_order(
        self,
        current_nav: float,
        start_of_day_nav: float,
        order_value: float,
        prices: Optional[Dict[str, float]] = None,
        positions_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """检查是否允许下单。

        Fail-open 设计：任何内部异常都返回 allowed=True，并记录 warning。
        借鉴生产级断路器最佳实践：风控自身故障不应阻断交易。
        """
        try:
            # 已断路状态下，检查是否满足恢复条件（滞回）
            if self.tripped:
                if self._should_recover(current_nav, start_of_day_nav):
                    self._recover()
                else:
                    return {
                        "allowed": False,
                        "reason": f"断路中（{self.trip_reason}），尚未满足恢复条件 "
                                  f"(需 ROI >= {self.recover_threshold:.2%})",
                    }

            # 单日亏损检查（保留旧版正确逻辑 + 滞回保护）
            daily_return = (
                (current_nav - start_of_day_nav) / start_of_day_nav
                if start_of_day_nav > 0 else 0.0
            )
            if daily_return < -self.daily_loss_limit:
                self._trip(
                    f"单日亏损 {daily_return:.2%} 超过阈值 -{self.daily_loss_limit:.2%}"
                )
                return {"allowed": False, "reason": self.trip_reason}

            # 滚动窗口多日亏损检查（新增，借鉴多日断路器设计）
            self.daily_returns.append(daily_return)
            if len(self.daily_returns) > self.rolling_window:
                self.daily_returns = self.daily_returns[-self.rolling_window:]
            if len(self.daily_returns) >= self.min_samples:
                rolling_roi = float(np.prod([1 + r for r in self.daily_returns]) - 1)
                if rolling_roi < -self.daily_loss_limit:
                    self._trip(
                        f"滚动 {len(self.daily_returns)} 日 ROI {rolling_roi:.2%} "
                        f"超过阈值 -{self.daily_loss_limit:.2%}"
                    )
                    return {"allowed": False, "reason": self.trip_reason}

            # 单笔金额检查
            if order_value > current_nav * 0.1:  # 单笔不超过 10% 净值
                return {
                    "allowed": False,
                    "reason": f"单笔金额 {order_value:.0f} 超过净值 10% 上限",
                }

            # 频率检查
            if not self._check_frequency():
                return {
                    "allowed": False,
                    "reason": f"订单频率超过每秒 {self.max_order_frequency} 次限制",
                }

            self.last_order_times.append(time.time())
            return {"allowed": True, "reason": ""}

        except Exception as exc:
            # Fail-open：风控自身异常时放行，避免 bug 阻断全部交易
            logger.warning(f"断路器检查异常，fail-open 放行: {exc}")
            return {"allowed": True, "reason": f"fail-open: {exc}"}

    # ------------------------------------------------------------------
    # 滞回状态机
    # ------------------------------------------------------------------

    def _should_recover(self, current_nav: float, start_of_day_nav: float) -> bool:
        """判断是否满足恢复条件（滞回：恢复阈值与触发阈值不同）。"""
        daily_return = (
            (current_nav - start_of_day_nav) / start_of_day_nav
            if start_of_day_nav > 0 else 0.0
        )
        # 单日恢复：当日收益 >= recover_threshold
        if daily_return >= self.recover_threshold:
            return True
        # 滚动恢复：窗口 ROI >= recover_threshold
        if len(self.daily_returns) >= self.min_samples:
            rolling_roi = float(np.prod([1 + r for r in self.daily_returns]) - 1)
            if rolling_roi >= self.recover_threshold:
                return True
        return False

    def _trip(self, reason: str):
        self.tripped = True
        self.trip_reason = reason
        self.trip_time = time.time()
        logger.warning(f"断路器触发: {reason}")
        if self.state_path:
            self.save_state()

    def _recover(self):
        logger.info(f"断路器恢复（已满足恢复阈值 {self.recover_threshold:.2%}）")
        self.tripped = False
        self.trip_reason = ""
        self.trip_time = None
        if self.state_path:
            self.save_state()

    def _check_frequency(self) -> bool:
        now = time.time()
        self.last_order_times = [t for t in self.last_order_times if now - t < 1.0]
        return len(self.last_order_times) < self.max_order_frequency

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_state(self):
        if not self.state_path:
            return
        try:
            state = {
                "tripped": self.tripped,
                "trip_reason": self.trip_reason,
                "trip_time": self.trip_time,
                "daily_returns": self.daily_returns,
                "saved_at": time.time(),
            }
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"断路器状态保存失败: {exc}")

    def load_state(self):
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.tripped = state.get("tripped", False)
            self.trip_reason = state.get("trip_reason", "")
            self.trip_time = state.get("trip_time")
            self.daily_returns = state.get("daily_returns", [])
            logger.info(f"断路器状态已加载: tripped={self.tripped}")
        except Exception as exc:
            logger.warning(f"断路器状态加载失败，使用默认状态: {exc}")