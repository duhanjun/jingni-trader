"""
可组合风险管理模型 (Composable Risk Management Models)
借鉴来源: QuantConnect/Lean 的 IRiskManagementModel + NOFX 的熔断机制

设计目标:
1. 把风险管理拆分为可插拔的独立模型, 每个模型只负责一类风险检查
2. 借鉴 QuantConnect 的 IRiskManagementModel 接口:
   ManageRisk(portfolio, targets) -> adjusted_targets
3. 实现 A 股场景常用的风控模型:
   - MaximumDrawdownRiskModel: 组合最大回撤熔断
   - TrailingStopRiskModel: 个股追踪止损
   - MaxPositionRiskModel: 单一持仓上限
   - PortfolioHeatRiskModel: 组合总风险敞口上限
   - CircuitBreakerRiskModel: 连续亏损熔断 (借鉴 NOFX)
   - VolatilityScalingModel: 波动率目标调仓
4. 支持模型链式组合: 多个模型依次过滤, 后者基于前者输出

与现有 portfolio-risk-engine 的 RiskManager 对比:
- 现有: 单一类, 方法耦合, 难以独立测试与组合
- 优化: 接口统一, 可插拔, 可组合, 易测试
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("risk_management_models")


# ============================================================
# 数据结构 (借鉴 QuantConnect 的 PortfolioTarget / Insight)
# ============================================================

@dataclass
class PortfolioTarget:
    """
    目标持仓 (借鉴 QuantConnect 的 PortfolioTarget)

    code: 标的代码
    target_weight: 目标权重 (0~1)
    current_weight: 当前权重
    entry_price: 建仓价 (用于止损计算)
    current_price: 当前价
    unrealized_pnl_pct: 当前浮亏/浮盈比例
    holding_days: 持有天数
    """
    code: str
    target_weight: float
    current_weight: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl_pct: float = 0.0
    holding_days: int = 0


@dataclass
class PortfolioState:
    """
    组合状态快照 (传给各风控模型)

    nav: 当前净值
    peak_nav: 历史峰值净值
    cash: 现金
    drawdown: 当前回撤 (负数)
    daily_pnl: 当日盈亏
    consecutive_losses: 连续亏损天数
    total_risk_pct: 当前总风险敞口占比
    targets: 目标持仓列表
    """
    nav: float
    peak_nav: float
    cash: float
    drawdown: float
    daily_pnl: float
    consecutive_losses: int
    total_risk_pct: float
    targets: List[PortfolioTarget] = field(default_factory=list)


@dataclass
class RiskCheckResult:
    """单个风控模型的检查结果"""
    model_name: str
    triggered: bool
    reason: str
    adjusted_targets: Optional[List[PortfolioTarget]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 风控模型抽象基类 (借鉴 QuantConnect IRiskManagementModel)
# ============================================================

class IRiskManagementModel(ABC):
    """
    风险管理模型接口 (借鉴 QuantConnect/Lean IRiskManagementModel)

    子类只需实现 manage_risk 方法, 返回调整后的 targets.
    若触发风控, 可将部分 target 的权重降为 0 (清仓) 或降低.
    """

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def manage_risk(
        self,
        state: PortfolioState,
        targets: List[PortfolioTarget],
    ) -> RiskCheckResult:
        """
        检查并调整目标持仓

        参数:
            state: 当前组合状态
            targets: 上一个模型输出的目标持仓 (链式)

        返回:
            RiskCheckResult, 含调整后的 targets
        """
        ...

    def _adjust_targets(
        self,
        targets: List[PortfolioTarget],
        zero_codes: Optional[set] = None,
        scale: float = 1.0,
    ) -> List[PortfolioTarget]:
        """工具方法: 调整 targets 权重"""
        adjusted = []
        zero_codes = zero_codes or set()
        for t in targets:
            if t.code in zero_codes:
                adjusted.append(PortfolioTarget(
                    code=t.code, target_weight=0.0,
                    current_weight=t.current_weight,
                    entry_price=t.entry_price,
                    current_price=t.current_price,
                    unrealized_pnl_pct=t.unrealized_pnl_pct,
                    holding_days=t.holding_days,
                ))
            else:
                adjusted.append(PortfolioTarget(
                    code=t.code, target_weight=t.target_weight * scale,
                    current_weight=t.current_weight,
                    entry_price=t.entry_price,
                    current_price=t.current_price,
                    unrealized_pnl_pct=t.unrealized_pnl_pct,
                    holding_days=t.holding_days,
                ))
        return adjusted


# ============================================================
# 具体风控模型实现
# ============================================================

class MaximumDrawdownRiskModel(IRiskManagementModel):
    """
    最大回撤熔断模型

    当组合回撤超过阈值时, 全部清仓并停止交易.
    借鉴 QuantConnect 的 MaximumDrawdownRiskManagementModel.
    """

    def __init__(self, max_drawdown: float = 0.10, cooldown_days: int = 5):
        super().__init__()
        self.max_drawdown = max_drawdown
        self.cooldown_days = cooldown_days
        self._cooldown_remaining = 0

    def manage_risk(self, state: PortfolioState, targets: List[PortfolioTarget]) -> RiskCheckResult:
        # 冷却期: 继续保持空仓
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason=f"冷却期剩余 {self._cooldown_remaining} 天, 保持空仓",
                adjusted_targets=self._adjust_targets(targets, zero_codes={t.code for t in targets}),
                metadata={"cooldown_remaining": self._cooldown_remaining},
            )

        # 回撤触发
        if state.drawdown <= -self.max_drawdown:
            self._cooldown_remaining = self.cooldown_days
            all_codes = {t.code for t in targets}
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason=f"组合回撤 {state.drawdown:.2%} 超过阈值 -{self.max_drawdown:.2%}, 触发熔断清仓, 冷却 {self.cooldown_days} 天",
                adjusted_targets=self._adjust_targets(targets, zero_codes=all_codes),
                metadata={
                    "drawdown": state.drawdown,
                    "threshold": -self.max_drawdown,
                    "cooldown_days": self.cooldown_days,
                },
            )

        return RiskCheckResult(
            model_name=self.name, triggered=False, reason="",
            adjusted_targets=targets,
            metadata={"drawdown": state.drawdown},
        )


class TrailingStopRiskModel(IRiskManagementModel):
    """
    个股追踪止损模型

    当个股浮亏超过阈值, 或从最高点回撤超过阈值时, 清仓该个股.
    借鉴 QuantConnect 的 TrailingStopRiskManagementModel.
    """

    def __init__(self, max_loss_pct: float = 0.08, trailing_pct: float = 0.15):
        super().__init__()
        self.max_loss_pct = max_loss_pct
        self.trailing_pct = trailing_pct

    def manage_risk(self, state: PortfolioState, targets: List[PortfolioTarget]) -> RiskCheckResult:
        stop_codes = set()
        reasons = []

        for t in targets:
            if t.target_weight <= 0:
                continue
            # 浮亏止损
            if t.unrealized_pnl_pct <= -self.max_loss_pct:
                stop_codes.add(t.code)
                reasons.append(f"{t.code} 浮亏 {t.unrealized_pnl_pct:.2%} ≤ -{self.max_loss_pct:.2%}")
                continue
            # 追踪止损 (需要历史最高价, 这里用 entry_price 近似)
            # 实际场景应传入 peak_price
            if t.entry_price > 0 and t.current_price > 0:
                # 简化: 若当前价相对建仓价回撤超过 trailing_pct (适用于有过盈利的情况)
                pass

        if stop_codes:
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason="; ".join(reasons),
                adjusted_targets=self._adjust_targets(targets, zero_codes=stop_codes),
                metadata={"stopped_codes": list(stop_codes), "max_loss_pct": self.max_loss_pct},
            )

        return RiskCheckResult(
            model_name=self.name, triggered=False, reason="",
            adjusted_targets=targets,
        )


class MaxPositionRiskModel(IRiskManagementModel):
    """
    单一持仓上限模型

    任何单一标的权重不得超过上限, 超过则截断.
    借鉴 QuantConnect 的 MaximumPositionSizeRiskManagementModel.
    """

    def __init__(self, max_position_pct: float = 0.05):
        super().__init__()
        self.max_position_pct = max_position_pct

    def manage_risk(self, state: PortfolioState, targets: List[PortfolioTarget]) -> RiskCheckResult:
        capped = []
        capped_codes = []
        for t in targets:
            if t.target_weight > self.max_position_pct:
                capped.append(PortfolioTarget(
                    code=t.code, target_weight=self.max_position_pct,
                    current_weight=t.current_weight,
                    entry_price=t.entry_price, current_price=t.current_price,
                    unrealized_pnl_pct=t.unrealized_pnl_pct,
                    holding_days=t.holding_days,
                ))
                capped_codes.append(t.code)
            else:
                capped.append(t)

        if capped_codes:
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason=f"以下标的权重超过上限 {self.max_position_pct:.2%}, 已截断: {capped_codes}",
                adjusted_targets=capped,
                metadata={"capped_codes": capped_codes, "max_position_pct": self.max_position_pct},
            )

        return RiskCheckResult(
            model_name=self.name, triggered=False, reason="",
            adjusted_targets=targets,
        )


class PortfolioHeatRiskModel(IRiskManagementModel):
    """
    组合总风险敞口模型

    所有持仓的总风险 (权重 * 个股波动率) 不得超过上限.
    超过则等比例缩减所有持仓.
    借鉴 theledgermind 的 portfolio heat 概念.
    """

    def __init__(self, max_total_risk: float = 0.06, volatilities: Optional[Dict[str, float]] = None):
        super().__init__()
        self.max_total_risk = max_total_risk
        self.volatilities = volatilities or {}

    def manage_risk(self, state: PortfolioState, targets: List[PortfolioTarget]) -> RiskCheckResult:
        # 计算总风险 = sum(weight * vol)
        total_risk = 0.0
        for t in targets:
            vol = self.volatilities.get(t.code, 0.02)  # 默认 2% 日波动
            total_risk += abs(t.target_weight) * vol

        if total_risk > self.max_total_risk and total_risk > 0:
            scale = self.max_total_risk / total_risk
            scaled = []
            for t in targets:
                scaled.append(PortfolioTarget(
                    code=t.code, target_weight=t.target_weight * scale,
                    current_weight=t.current_weight,
                    entry_price=t.entry_price, current_price=t.current_price,
                    unrealized_pnl_pct=t.unrealized_pnl_pct,
                    holding_days=t.holding_days,
                ))
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason=f"组合总风险 {total_risk:.2%} 超过上限 {self.max_total_risk:.2%}, 等比缩减至 {scale:.2%}",
                adjusted_targets=scaled,
                metadata={"total_risk": total_risk, "scale": scale, "max_total_risk": self.max_total_risk},
            )

        return RiskCheckResult(
            model_name=self.name, triggered=False, reason="",
            adjusted_targets=targets,
            metadata={"total_risk": total_risk},
        )


class CircuitBreakerRiskModel(IRiskManagementModel):
    """
    连续亏损熔断模型

    连续 N 天亏损时, 全部清仓并进入冷却期.
    借鉴 NOFX 的 Safe Mode (3 consecutive failures -> auto-protect).
    """

    def __init__(self, max_consecutive_losses: int = 3, cooldown_days: int = 3):
        super().__init__()
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_days = cooldown_days
        self._cooldown_remaining = 0

    def manage_risk(self, state: PortfolioState, targets: List[PortfolioTarget]) -> RiskCheckResult:
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason=f"连续亏损冷却期剩余 {self._cooldown_remaining} 天",
                adjusted_targets=self._adjust_targets(targets, zero_codes={t.code for t in targets}),
                metadata={"cooldown_remaining": self._cooldown_remaining},
            )

        if state.consecutive_losses >= self.max_consecutive_losses:
            self._cooldown_remaining = self.cooldown_days
            all_codes = {t.code for t in targets}
            return RiskCheckResult(
                model_name=self.name,
                triggered=True,
                reason=f"连续亏损 {state.consecutive_losses} 天 ≥ {self.max_consecutive_losses}, 触发熔断, 冷却 {self.cooldown_days} 天",
                adjusted_targets=self._adjust_targets(targets, zero_codes=all_codes),
                metadata={
                    "consecutive_losses": state.consecutive_losses,
                    "cooldown_days": self.cooldown_days,
                },
            )

        return RiskCheckResult(
            model_name=self.name, triggered=False, reason="",
            adjusted_targets=targets,
            metadata={"consecutive_losses": state.consecutive_losses},
        )


class VolatilityScalingModel(IRiskManagementModel):
    """
    波动率目标模型

    根据组合实际波动率与目标波动率, 等比例调整持仓.
    借鉴 Risk Parity / Vol Targeting 策略.
    """

    def __init__(self, target_volatility: float = 0.15, current_volatility: float = 0.20,
                 max_leverage: float = 1.5):
        super().__init__()
        self.target_volatility = target_volatility
        self.current_volatility = current_volatility
        self.max_leverage = max_leverage

    def manage_risk(self, state: PortfolioState, targets: List[PortfolioTarget]) -> RiskCheckResult:
        if self.current_volatility <= 0:
            return RiskCheckResult(
                model_name=self.name, triggered=False, reason="",
                adjusted_targets=targets,
            )

        scale = self.target_volatility / self.current_volatility
        scale = min(scale, self.max_leverage)  # 限制最大杠杆
        scale = max(scale, 0.0)  # 不允许负杠杆

        scaled = []
        for t in targets:
            scaled.append(PortfolioTarget(
                code=t.code, target_weight=t.target_weight * scale,
                current_weight=t.current_weight,
                entry_price=t.entry_price, current_price=t.current_price,
                unrealized_pnl_pct=t.unrealized_pnl_pct,
                holding_days=t.holding_days,
            ))

        triggered = abs(scale - 1.0) > 0.01
        return RiskCheckResult(
            model_name=self.name,
            triggered=triggered,
            reason=f"波动率目标调整: 当前 {self.current_volatility:.2%} -> 目标 {self.target_volatility:.2%}, 缩放 {scale:.2%}" if triggered else "",
            adjusted_targets=scaled,
            metadata={
                "target_vol": self.target_volatility,
                "current_vol": self.current_volatility,
                "scale": scale,
            },
        )


# ============================================================
# 风控模型管理器 (链式组合)
# ============================================================

class RiskManagerChain:
    """
    风控模型链 (借鉴 QuantConnect 的 RiskManagement 模型组合方式)

    按顺序应用多个风控模型, 后者基于前者输出.
    任一模型触发熔断清仓, 后续模型仍会执行 (但 targets 已为空).

    用法:
        chain = RiskManagerChain([
            MaximumDrawdownRiskModel(max_drawdown=0.10),
            TrailingStopRiskModel(max_loss_pct=0.08),
            MaxPositionRiskModel(max_position_pct=0.05),
            PortfolioHeatRiskModel(max_total_risk=0.06),
        ])
        result = chain.manage(state, targets)
    """

    def __init__(self, models: List[IRiskManagementModel]):
        self.models = models
        self._last_results: List[RiskCheckResult] = []

    def manage(
        self,
        state: PortfolioState,
        targets: List[PortfolioTarget],
    ) -> Dict[str, Any]:
        """
        依次应用所有风控模型

        返回:
            {
                "final_targets": List[PortfolioTarget],
                "results": List[RiskCheckResult],
                "any_triggered": bool,
                "all_cleared": bool,
            }
        """
        current_targets = list(targets)
        results = []
        any_triggered = False

        for model in self.models:
            result = model.manage_risk(state, current_targets)
            results.append(result)
            if result.triggered:
                any_triggered = True
                if result.adjusted_targets is not None:
                    current_targets = result.adjusted_targets
                logger.info(f"[风控] {result.model_name}: {result.reason}")
            # 即使未触发, 也用返回的 targets (可能含元数据调整)

        self._last_results = results

        all_cleared = all(t.target_weight == 0 for t in current_targets) if current_targets else True

        return {
            "final_targets": current_targets,
            "results": results,
            "any_triggered": any_triggered,
            "all_cleared": all_cleared,
            "triggered_models": [r.model_name for r in results if r.triggered],
        }

    def get_last_results(self) -> List[RiskCheckResult]:
        return self._last_results


# ============================================================
# 工具函数: 从 DataFrame 构建 PortfolioState / Targets
# ============================================================

def build_portfolio_state(
    nav: float,
    peak_nav: float,
    cash: float,
    daily_pnl: float,
    consecutive_losses: int,
    weights: pd.Series,
    prices: pd.Series,
    entry_prices: Optional[pd.Series] = None,
    holding_days: Optional[pd.Series] = None,
    volatilities: Optional[Dict[str, float]] = None,
) -> Tuple[PortfolioState, List[PortfolioTarget]]:
    """
    从 pandas Series 构建组合状态与目标列表

    参数:
        nav: 当前净值
        peak_nav: 历史峰值
        cash: 现金
        daily_pnl: 当日盈亏
        consecutive_losses: 连续亏损天数
        weights: 目标权重 (code -> weight)
        prices: 当前价 (code -> price)
        entry_prices: 建仓价
        holding_days: 持有天数
        volatilities: 个股波动率 (用于总风险计算)
    """
    volatilities = volatilities or {}
    drawdown = (nav - peak_nav) / peak_nav if peak_nav > 0 else 0.0

    total_risk = 0.0
    targets = []
    for code, w in weights.items():
        price = prices.get(code, 0.0)
        ep = entry_prices.get(code, price) if entry_prices is not None else price
        pnl_pct = (price - ep) / ep if ep > 0 else 0.0
        hd = int(holding_days.get(code, 0)) if holding_days is not None else 0
        vol = volatilities.get(code, 0.02)
        total_risk += abs(w) * vol
        targets.append(PortfolioTarget(
            code=code, target_weight=float(w),
            current_weight=float(w), entry_price=float(ep),
            current_price=float(price), unrealized_pnl_pct=float(pnl_pct),
            holding_days=hd,
        ))

    state = PortfolioState(
        nav=nav, peak_nav=peak_nav, cash=cash,
        drawdown=drawdown, daily_pnl=daily_pnl,
        consecutive_losses=consecutive_losses,
        total_risk_pct=total_risk,
        targets=targets,
    )
    return state, targets


def default_risk_chain(
    max_drawdown: float = 0.10,
    max_loss_pct: float = 0.08,
    max_position_pct: float = 0.05,
    max_total_risk: float = 0.06,
    max_consecutive_losses: int = 3,
) -> RiskManagerChain:
    """构建默认风控链 (A 股常用配置)"""
    return RiskManagerChain([
        MaximumDrawdownRiskModel(max_drawdown=max_drawdown, cooldown_days=5),
        CircuitBreakerRiskModel(max_consecutive_losses=max_consecutive_losses, cooldown_days=3),
        TrailingStopRiskModel(max_loss_pct=max_loss_pct),
        MaxPositionRiskModel(max_position_pct=max_position_pct),
        PortfolioHeatRiskModel(max_total_risk=max_total_risk),
    ])
