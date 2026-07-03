"""
多层风险控制框架

借鉴 best practices (https://blog.www2.ro/build-python-base-quant-hedge 与多份 A 股
实盘风控白皮书) 实现 3 层风控：

L1 头寸层：个股止损（ATR / 固定比例）
L2 组合层：单日亏损熔断、组合回撤熔断
L3 策略层：波动率目标仓位、动态杠杆

针对 jingni-trader 现有 RiskManager 的改进点：
1. 增加动态 ATR 止损（更稳健）
2. 增加组合回撤熔断 + 仓位缩放（drawdown-based de-risking）
3. 增加波动率目标（volatility targeting）仓位
4. 返回结构化分层风险状态便于上层决策
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RiskConfig:
    """风控配置（与 jingni-trader scripts/config.py 风格保持一致）"""
    # L1 个股层
    individual_stop_loss: float = 0.08        # 固定比例止损
    individual_take_profit: float = 0.20      # 固定比例止盈
    atr_stop_multiplier: float = 2.0          # ATR 止损倍数
    atr_period: int = 14

    # L2 组合层
    max_daily_loss_ratio: float = 0.03        # 单日最大亏损
    max_drawdown_threshold: float = 0.15      # 组合最大回撤熔断阈值
    drawdown_scale_start: float = 0.05        # 超过该回撤开始缩仓
    drawdown_scale_end: float = 0.15          # 缩仓至最低仓位
    min_position_scale: float = 0.3           # 最大缩仓时保留 30% 仓位

    # L3 策略层
    target_annual_vol: float = 0.10           # 目标年化波动率
    vol_lookback: int = 20                    # 波动率回看窗口（日）
    max_leverage: float = 1.0                 # 最大杠杆

    risk_free_rate: float = 0.03


class MultiLayerRiskManager:
    """
    三层风控管理器
    每个交易日依次执行 L1 -> L2 -> L3 检查
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()
        self._peak_nav: float = 0.0
        self._start_day_nav: float = 0.0
        self._current_day: Optional[pd.Timestamp] = None

    def reset_day(self, day: pd.Timestamp, current_nav: float) -> None:
        """每个交易日开始时调用"""
        self._current_day = day
        self._start_day_nav = current_nav
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav

    # ── L1 头寸层 ─────────────────────────────────────────
    def check_position_stop(self, entry_price: float, current_price: float,
                            high_at_entry: Optional[float] = None) -> Dict:
        """
        个股止损检查（固定比例 + 移动止盈）
        返回: {"stop": bool, "reason": str, "return": float}
        """
        if entry_price <= 0:
            return {"stop": False, "reason": "invalid_entry", "return": 0.0}
        ret = (current_price - entry_price) / entry_price
        # 固定止损
        if ret <= -self.cfg.individual_stop_loss:
            return {"stop": True, "reason": "stop_loss", "return": float(ret)}
        # 固定止盈
        if ret >= self.cfg.individual_take_profit:
            return {"stop": True, "reason": "take_profit", "return": float(ret)}
        # 移动止盈：曾经达到 1.5*止盈目标后回撤 50% 离场
        if high_at_entry is not None and high_at_entry > 0:
            high_ret = (high_at_entry - entry_price) / entry_price
            if high_ret >= self.cfg.individual_take_profit and \
               ret <= high_ret * 0.5:
                return {"stop": True, "reason": "trailing_stop", "return": float(ret)}
        return {"stop": False, "reason": "hold", "return": float(ret)}

    def atr_stop_price(self, entry_price: float, atr: float,
                       multiplier: Optional[float] = None) -> float:
        """ATR 动态止损价格（多头）"""
        m = multiplier if multiplier is not None else self.cfg.atr_stop_multiplier
        return entry_price - m * atr

    def compute_atr(self, high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, period: Optional[int] = None) -> np.ndarray:
        """平均真实波幅 (Wilder smoothing)"""
        p = period if period is not None else self.cfg.atr_period
        n = len(close)
        if n < 2:
            return np.zeros(n)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        tr[1:] = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ])
        atr = np.zeros(n)
        atr[:p] = np.nan
        if n < p:
            return atr
        atr[p - 1] = tr[:p].mean()
        for i in range(p, n):
            atr[i] = (atr[i - 1] * (p - 1) + tr[i]) / p
        return atr

    # ── L2 组合层 ─────────────────────────────────────────
    def check_daily_loss(self, current_nav: float) -> Dict:
        """单日亏损熔断"""
        if self._start_day_nav <= 0:
            return {"triggered": False, "daily_return": 0.0, "reason": "no_start_nav"}
        daily_ret = (current_nav - self._start_day_nav) / self._start_day_nav
        triggered = daily_ret <= -self.cfg.max_daily_loss_ratio
        return {
            "triggered": triggered,
            "daily_return": float(daily_ret),
            "threshold": self.cfg.max_daily_loss_ratio,
            "reason": "daily_loss_breach" if triggered else "ok",
        }

    def current_drawdown(self, current_nav: float) -> float:
        """当前组合回撤（负数）"""
        if self._peak_nav <= 0:
            return 0.0
        if current_nav > self._peak_nav:
            self._peak_nav = current_nav
        return float(current_nav / self._peak_nav - 1.0)

    def check_portfolio_drawdown(self, current_nav: float) -> Dict:
        """组合回撤熔断 + 缩仓"""
        dd = self.current_drawdown(current_nav)
        triggered = dd <= -self.cfg.max_drawdown_threshold
        scale = self.position_scale(dd)
        return {
            "triggered": triggered,
            "drawdown": float(dd),
            "threshold": self.cfg.max_drawdown_threshold,
            "position_scale": float(scale),
            "reason": "drawdown_breach" if triggered else "ok",
        }

    def position_scale(self, current_drawdown: float) -> float:
        """
        基于回撤的仓位缩放：
        dd >= -drawdown_scale_start: scale = 1.0
        dd <= -drawdown_scale_end:   scale = min_position_scale
        中间线性插值
        """
        if current_drawdown >= -self.cfg.drawdown_scale_start:
            return 1.0
        if current_drawdown <= -self.cfg.drawdown_scale_end:
            return self.cfg.min_position_scale
        # 线性: dd 在 [-start, -end] 区间内，scale 从 1.0 单调减到 min
        #   progress = (dd + start) / (end - start)，0 = 刚进入缩仓区, 1 = 触底
        span = self.cfg.drawdown_scale_end - self.cfg.drawdown_scale_start
        progress = (-current_drawdown - self.cfg.drawdown_scale_start) / span
        # progress ∈ [0, 1]
        return float(1.0 - (1.0 - self.cfg.min_position_scale) * progress)

    # ── L3 策略层 ─────────────────────────────────────────
    def volatility_target_leverage(self, recent_returns: np.ndarray,
                                    target_vol: Optional[float] = None,
                                    lookback: Optional[int] = None) -> float:
        """
        波动率目标：实际杠杆 = target_vol / realized_vol
        用于抑制高波动期风险敞口
        """
        tv = target_vol if target_vol is not None else self.cfg.target_annual_vol
        lb = lookback if lookback is not None else self.cfg.vol_lookback
        r = np.asarray(recent_returns, dtype=np.float64)
        r = r[-lb:]
        r = r[~np.isnan(r)]
        if r.size < 5:
            return 1.0
        realized_vol = float(np.std(r, ddof=1) * np.sqrt(252))
        if realized_vol < 1e-6:
            return self.cfg.max_leverage
        lev = tv / realized_vol
        return float(min(self.cfg.max_leverage, max(0.0, lev)))

    # ── 综合评估 ─────────────────────────────────────────
    def evaluate(self, current_nav: float, recent_returns: np.ndarray,
                 positions: Optional[pd.DataFrame] = None) -> Dict:
        """
        综合评估当前风控状态
        positions: DataFrame[code, entry_price, current_price, high_at_entry, atr]
        """
        l2_daily = self.check_daily_loss(current_nav)
        l2_dd = self.check_portfolio_drawdown(current_nav)
        l3_lev = self.volatility_target_leverage(recent_returns)

        l1_results = {}
        if positions is not None and not positions.empty:
            for _, row in positions.iterrows():
                l1_results[row["code"]] = self.check_position_stop(
                    entry_price=row["entry_price"],
                    current_price=row["current_price"],
                    high_at_entry=row.get("high_at_entry"),
                )

        triggered = l2_daily["triggered"] or l2_dd["triggered"]
        final_scale = l2_dd["position_scale"] * l3_lev
        return {
            "L1_position_stops": l1_results,
            "L2_daily_loss": l2_daily,
            "L2_drawdown": l2_dd,
            "L3_leverage": l3_lev,
            "any_triggered": triggered,
            "final_position_scale": float(final_scale),
        }
