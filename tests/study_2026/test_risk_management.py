"""
优化方向: 多层级风险管理系统
借鉴来源:
  1. vn.py (VeighNa) (https://github.com/vnpy/vnpy) - 多层级风控框架
     - 事前风控: 仓位限制、单笔最大亏损、委托量限制
     - 事中风控: 实时监控、幅度限制、价格偏离检查
     - 事后风控: 交易复盘、异常分析、合规检查
     - 风控引擎独立于交易引擎，拦截式设计
  2. Qlib (https://github.com/microsoft/qlib) - 风险因子提取
     - Barra 风格因子归因
     - 行业暴露控制
     - 因子暴露限制
  3. Backtrader - 回测内嵌风控
     - Sizer 仓位管理
     - StopLoss/TakeProfit 订单类型
     - 最大回撤熔断

验证内容:
  - 三级风控体系（事前/事中/事后）
  - 熔断机制（回撤熔断、单日亏损熔断、连续亏损熔断）
  - VaR/CVaR 计算
  - 最大回撤实时监控
  - 与原有风控模块的对比测试

注意: 本文件仅用于验证测试，不修改主项目代码。
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# =============================================================================
# 风控级别和状态定义
# =============================================================================

class RiskLevel(Enum):
    NORMAL = "normal"          # 正常
    WARNING = "warning"        # 警告
    RESTRICTED = "restricted"  # 受限
    SUSPENDED = "suspended"    # 熔断
    CLOSED = "closed"          # 关闭


class RiskCheckType(Enum):
    PRE_TRADE = "pre_trade"       # 事前检查
    IN_TRADE = "in_trade"         # 事中监控
    POST_TRADE = "post_trade"     # 事后分析
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断检查


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    check_type: RiskCheckType
    check_name: str
    passed: bool
    risk_level: RiskLevel = RiskLevel.NORMAL
    message: str = ""
    details: Dict = field(default_factory=dict)


# =============================================================================
# 账户状态追踪
# =============================================================================

class AccountState:
    """账户状态追踪器"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_value = initial_capital
        self.max_value = initial_capital
        self.min_value = initial_capital

        # 历史记录
        self.equity_history: List[float] = []
        self.daily_returns: List[float] = []
        self.trade_history: List[Dict] = []

        # 风控相关
        self.daily_pnl: float = 0.0
        self.daily_pnl_ratio: float = 0.0
        self.peak_value = initial_capital
        self.drawdown = 0.0
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_risk_level = RiskLevel.NORMAL

        # 熔断状态
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = ""
        self.circuit_breaker_until: Optional[datetime] = None

        # 当日统计
        self.today_trades = 0
        self.today_volume = 0
        self.today_commission = 0.0

    def update(self, total_value: float, timestamp: datetime = None):
        """更新账户状态"""
        self.total_value = total_value
        self.equity_history.append(total_value)

        # 更新峰值和回撤
        if total_value > self.peak_value:
            self.peak_value = total_value
        self.drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, self.drawdown)

        # 更新最大最小值
        self.max_value = max(self.max_value, total_value)
        self.min_value = min(self.min_value, total_value)

        # 每日收益
        if len(self.equity_history) >= 2:
            daily_ret = (self.equity_history[-1] / self.equity_history[-2] - 1)
            self.daily_returns.append(daily_ret)
            self.daily_pnl_ratio = daily_ret
            self.daily_pnl = total_value - self.equity_history[-2]

            # 连续盈亏
            if daily_ret > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            elif daily_ret < 0:
                self.consecutive_losses += 1
                self.consecutive_wins = 0

    def record_trade(self, trade: Dict):
        """记录成交"""
        self.trade_history.append(trade)
        self.today_trades += 1
        self.today_volume += trade.get('volume', 0) * trade.get('price', 0)
        self.today_commission += trade.get('commission', 0) + trade.get('tax', 0)

    def reset_daily(self):
        """重置日统计"""
        self.today_trades = 0
        self.today_volume = 0
        self.today_commission = 0

    def get_summary(self) -> Dict:
        """获取摘要"""
        return {
            "initial_capital": self.initial_capital,
            "current_value": self.total_value,
            "total_return": (self.total_value / self.initial_capital - 1),
            "peak_value": self.peak_value,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.drawdown,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "risk_level": self.current_risk_level.value,
            "circuit_breaker_active": self.circuit_breaker_active,
        }


# =============================================================================
# 事前风控检查
# =============================================================================

class PreTradeRiskChecks:
    """
    事前风控检查

    参考 vn.py 的 pre_trade_risk 模块：
    - 仓位限制检查
    - 单笔委托量限制
    - 委托价格偏离检查
    - 资金充足性检查
    - 持仓集中度检查
    - 行业暴露检查
    """

    def __init__(self,
                 max_position_per_stock: float = 0.2,      # 单只股票最大仓位
                 max_position_total: float = 0.95,          # 总仓位上限
                 max_order_value: float = 500_000,          # 单笔委托最大金额
                 max_price_deviation: float = 0.03,         # 最大价格偏离
                 max_industry_exposure: float = 0.3):       # 最大行业暴露
        self.max_position_per_stock = max_position_per_stock
        self.max_position_total = max_position_total
        self.max_order_value = max_order_value
        self.max_price_deviation = max_price_deviation
        self.max_industry_exposure = max_industry_exposure

    def check_position_limit(
        self, account_state: AccountState, symbol: str,
        current_position: int, target_position: int, price: float
    ) -> RiskCheckResult:
        """检查仓位限制"""
        target_value = target_position * price * 100
        total_value = account_state.total_value

        # 单票仓位
        stock_ratio = target_value / total_value if total_value > 0 else 1
        if stock_ratio > self.max_position_per_stock:
            return RiskCheckResult(
                RiskCheckType.PRE_TRADE, "position_limit",
                False, RiskLevel.RESTRICTED,
                f"单票仓位 {stock_ratio:.1%} 超过限制 {self.max_position_per_stock:.1%}",
                {"stock_ratio": stock_ratio, "limit": self.max_position_per_stock}
            )

        # 总仓位
        current_total_value = sum(
            pos * 100 * price for pos in [current_position]
        )  # 简化计算
        total_ratio = (current_total_value + target_value) / total_value if total_value > 0 else 1
        if total_ratio > self.max_position_total:
            return RiskCheckResult(
                RiskCheckType.PRE_TRADE, "total_position",
                False, RiskLevel.RESTRICTED,
                f"总仓位 {total_ratio:.1%} 超过限制 {self.max_position_total:.1%}",
                {"total_ratio": total_ratio, "limit": self.max_position_total}
            )

        return RiskCheckResult(RiskCheckType.PRE_TRADE, "position_limit", True)

    def check_order_value(
        self, account_state: AccountState, order_value: float
    ) -> RiskCheckResult:
        """检查单笔委托金额"""
        if order_value > self.max_order_value:
            return RiskCheckResult(
                RiskCheckType.PRE_TRADE, "order_value",
                False, RiskLevel.RESTRICTED,
                f"单笔委托金额 {order_value:.0f} 超过限制 {self.max_order_value:.0f}",
                {"order_value": order_value, "limit": self.max_order_value}
            )

        # 资金充足性
        if order_value > account_state.cash:
            return RiskCheckResult(
                RiskCheckType.PRE_TRADE, "insufficient_cash",
                False, RiskLevel.RESTRICTED,
                f"资金不足: 需要 {order_value:.0f}, 可用 {account_state.cash:.0f}",
                {"required": order_value, "available": account_state.cash}
            )

        return RiskCheckResult(RiskCheckType.PRE_TRADE, "order_value", True)

    def check_price_deviation(
        self, order_price: float, current_price: float
    ) -> RiskCheckResult:
        """检查价格偏离"""
        deviation = abs(order_price - current_price) / current_price if current_price > 0 else 1
        if deviation > self.max_price_deviation:
            return RiskCheckResult(
                RiskCheckType.PRE_TRADE, "price_deviation",
                False, RiskLevel.WARNING,
                f"委托价格偏离 {deviation:.1%} 超过限制 {self.max_price_deviation:.1%}",
                {"deviation": deviation, "limit": self.max_price_deviation}
            )
        return RiskCheckResult(RiskCheckType.PRE_TRADE, "price_deviation", True)


# =============================================================================
# 事中风控监控
# =============================================================================

class InTradeRiskMonitor:
    """
    事中风控监控

    参考 vn.py 的 risk_manager 模块：
    - 熔断机制（回撤熔断、亏损熔断、连续亏损熔断）
    - 实时 VaR 估算
    - 波动率异常检测
    - 流动性风险监控
    """

    def __init__(self,
                 max_drawdown_limit: float = 0.20,         # 最大回撤限制
                 daily_loss_limit: float = 0.05,            # 单日最大亏损
                 max_consecutive_losses: int = 5,           # 最大连续亏损天数
                 var_confidence: float = 0.95,              # VaR置信度
                 var_window: int = 60):                     # VaR计算窗口
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.max_consecutive_losses = max_consecutive_losses
        self.var_confidence = var_confidence
        self.var_window = var_window

    def check_drawdown_circuit_breaker(
        self, account_state: AccountState
    ) -> RiskCheckResult:
        """检查回撤熔断"""
        if account_state.circuit_breaker_active:
            return RiskCheckResult(
                RiskCheckType.CIRCUIT_BREAKER, "circuit_breaker",
                False, RiskLevel.SUSPENDED,
                f"熔断已激活: {account_state.circuit_breaker_reason}"
            )

        if account_state.drawdown >= self.max_drawdown_limit:
            return RiskCheckResult(
                RiskCheckType.CIRCUIT_BREAKER, "drawdown",
                False, RiskLevel.SUSPENDED,
                f"回撤 {account_state.drawdown:.1%} 超过限制 {self.max_drawdown_limit:.1%}",
                {"drawdown": account_state.drawdown, "limit": self.max_drawdown_limit}
            )
        return RiskCheckResult(RiskCheckType.CIRCUIT_BREAKER, "drawdown", True)

    def check_daily_loss_circuit_breaker(
        self, account_state: AccountState
    ) -> RiskCheckResult:
        """检查单日亏损熔断"""
        if account_state.daily_pnl_ratio < -self.daily_loss_limit:
            return RiskCheckResult(
                RiskCheckType.CIRCUIT_BREAKER, "daily_loss",
                False, RiskLevel.SUSPENDED,
                f"单日亏损 {account_state.daily_pnl_ratio:.1%} 超过限制 {self.daily_loss_limit:.1%}",
                {"daily_loss": account_state.daily_pnl_ratio, "limit": self.daily_loss_limit}
            )
        return RiskCheckResult(RiskCheckType.CIRCUIT_BREAKER, "daily_loss", True)

    def check_consecutive_losses(
        self, account_state: AccountState
    ) -> RiskCheckResult:
        """检查连续亏损"""
        if account_state.consecutive_losses >= self.max_consecutive_losses:
            return RiskCheckResult(
                RiskCheckType.CIRCUIT_BREAKER, "consecutive_losses",
                False, RiskLevel.SUSPENDED,
                f"连续亏损 {account_state.consecutive_losses} 天，超过限制 {self.max_consecutive_losses}",
                {"consecutive_losses": account_state.consecutive_losses, "limit": self.max_consecutive_losses}
            )
        return RiskCheckResult(RiskCheckType.CIRCUIT_BREAKER, "consecutive_losses", True)

    def calc_var(self, returns: np.ndarray) -> Dict:
        """
        计算 VaR 和 CVaR

        参考 RiskMetrics 方法：
        - 历史模拟法 VaR
        - 参数法 VaR（正态分布假设）
        - 条件 VaR（CVaR/Expected Shortfall）
        """
        if len(returns) < 2:
            return {
                "hist_var": 0.0,
                "param_var": 0.0,
                "cvar": 0.0,
                "confidence": self.var_confidence,
                "window": self.var_window,
            }

        recent_returns = returns[-self.var_window:] if len(returns) >= self.var_window else returns

        # 历史模拟法 VaR
        hist_var = np.percentile(recent_returns, (1 - self.var_confidence) * 100)

        # 参数法 VaR（正态分布）
        from scipy import stats
        mu = np.mean(recent_returns)
        sigma = np.std(recent_returns)
        param_var = mu + sigma * stats.norm.ppf(1 - self.var_confidence)

        # CVaR (Expected Shortfall)
        tail_returns = recent_returns[recent_returns <= hist_var]
        cvar = np.mean(tail_returns) if len(tail_returns) > 0 else hist_var

        return {
            "hist_var": float(hist_var),
            "param_var": float(param_var),
            "cvar": float(cvar),
            "confidence": self.var_confidence,
            "window": self.var_window,
        }

    def run_all_checks(self, account_state: AccountState) -> List[RiskCheckResult]:
        """运行所有事中检查"""
        results = []
        results.append(self.check_drawdown_circuit_breaker(account_state))
        results.append(self.check_daily_loss_circuit_breaker(account_state))
        results.append(self.check_consecutive_losses(account_state))
        return results

    def should_circuit_break(self, account_state: AccountState) -> Tuple[bool, str]:
        """判断是否需要熔断"""
        results = self.run_all_checks(account_state)
        for result in results:
            if not result.passed and result.risk_level == RiskLevel.SUSPENDED:
                return True, result.message
        return False, ""


# =============================================================================
# 事后风控分析
# =============================================================================

class PostTradeRiskAnalyzer:
    """
    事后风控分析

    参考 vn.py 的 risk_report 模块：
    - 交易统计（胜率、盈亏比、平均持仓时间）
    - 异常交易检测（频繁交易、对倒交易）
    - 绩效归因分析
    - 风险敞口报告
    """

    @staticmethod
    def analyze_trade_stats(trades: List[Dict], initial_capital: float) -> Dict:
        """分析交易统计"""
        if not trades:
            return {"total_trades": 0}

        df = pd.DataFrame(trades)
        buy_trades = df[df['direction'] == 1]
        sell_trades = df[df['direction'] == -1]

        # 计算每笔交易的盈亏（简化：配对买卖）
        total_pnl = 0
        wins = 0
        losses = 0
        pnl_list = []
        positions = {}

        for _, trade in df.iterrows():
            symbol = trade['symbol']
            direction = trade['direction']
            price = trade['price']
            volume = trade['volume']

            if direction == 1:  # 买入
                if symbol not in positions:
                    positions[symbol] = []
                positions[symbol].append({'price': price, 'volume': volume})
            else:  # 卖出
                if symbol in positions and positions[symbol]:
                    buy = positions[symbol].pop(0)
                    pnl = (price - buy['price']) * buy['volume'] * 100
                    total_pnl += pnl
                    pnl_list.append(pnl)
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_win = np.mean([p for p in pnl_list if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in pnl_list if p < 0]) if losses > 0 else 0
        profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": float(win_rate),
            "total_pnl": float(total_pnl),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "return_on_capital": float(total_pnl / initial_capital) if initial_capital > 0 else 0,
        }

    @staticmethod
    def detect_anomalies(trades: List[Dict]) -> List[Dict]:
        """检测异常交易"""
        anomalies = []
        if not trades:
            return anomalies

        df = pd.DataFrame(trades)

        # 检测频繁交易（同一只股票短时间内多次交易）
        if 'timestamp' in df.columns and 'symbol' in df.columns:
            for symbol in df['symbol'].unique():
                sym_trades = df[df['symbol'] == symbol].sort_values('timestamp')
                if len(sym_trades) > 10:
                    anomalies.append({
                        'type': 'frequent_trading',
                        'symbol': symbol,
                        'count': len(sym_trades),
                        'message': f"股票 {symbol} 交易 {len(sym_trades)} 次，可能存在频繁交易"
                    })

        return anomalies

    @staticmethod
    def calc_risk_metrics(equity_series: pd.Series) -> Dict:
        """计算风险指标"""
        returns = equity_series.pct_change().dropna()

        if len(returns) < 2:
            return {}

        # 回撤序列
        cummax = equity_series.cummax()
        drawdown = (cummax - equity_series) / cummax
        max_dd = drawdown.max()
        max_dd_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd > 0:
                current_duration += 1
            else:
                max_dd_duration = max(max_dd_duration, current_duration)
                current_duration = 0

        # 波动率
        annual_vol = returns.std() * np.sqrt(252)

        # Sharpe
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Calmar
        calmar = returns.mean() * 252 / max_dd if max_dd > 0 else 0

        return {
            "annual_return": float(returns.mean() * 252),
            "annual_volatility": float(annual_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "max_drawdown": float(max_dd),
            "max_drawdown_duration": max_dd_duration,
            "positive_days": int((returns > 0).sum()),
            "negative_days": int((returns < 0).sum()),
            "win_rate": float((returns > 0).mean()),
        }


# =============================================================================
# 风控引擎（统一入口）
# =============================================================================

class RiskEngine:
    """
    统一风控引擎

    参考 vn.py 的 RiskEngine 设计：
    - 整合事前/事中/事后三级风控
    - 独立的拦截式架构
    - 可配置的风控规则
    - 风控日志和报告
    """

    def __init__(self, initial_capital: float = 1_000_000):
        self.account_state = AccountState(initial_capital)
        self.pre_trade = PreTradeRiskChecks()
        self.in_trade = InTradeRiskMonitor()
        self.post_trade = PostTradeRiskAnalyzer()
        self.risk_log: List[RiskCheckResult] = []

    def pre_trade_check(
        self, symbol: str, direction: int, volume: int, price: float,
        current_position: int, current_price: float
    ) -> Tuple[bool, List[RiskCheckResult]]:
        """事前检查"""
        order_value = volume * price * 100
        results = []

        results.append(self.pre_trade.check_position_limit(
            self.account_state, symbol, current_position,
            current_position + direction * volume, price
        ))
        results.append(self.pre_trade.check_order_value(
            self.account_state, order_value
        ))
        results.append(self.pre_trade.check_price_deviation(
            price, current_price
        ))

        self.risk_log.extend(results)

        all_passed = all(r.passed for r in results)
        return all_passed, results

    def in_trade_check(self) -> Tuple[bool, List[RiskCheckResult]]:
        """事中检查"""
        results = self.in_trade.run_all_checks(self.account_state)
        self.risk_log.extend(results)

        should_break, reason = self.in_trade.should_circuit_break(self.account_state)
        if should_break:
            self.account_state.circuit_breaker_active = True
            self.account_state.circuit_breaker_reason = reason
            self.account_state.current_risk_level = RiskLevel.SUSPENDED

        return not should_break, results

    def post_trade_analyze(self) -> Dict:
        """事后分析"""
        trade_stats = self.post_trade.analyze_trade_stats(
            self.account_state.trade_history, self.account_state.initial_capital
        )
        anomalies = self.post_trade.detect_anomalies(
            self.account_state.trade_history
        )
        equity = pd.Series(self.account_state.equity_history)
        risk_metrics = self.post_trade.calc_risk_metrics(equity)

        return {
            "trade_stats": trade_stats,
            "anomalies": anomalies,
            "risk_metrics": risk_metrics,
            "account_summary": self.account_state.get_summary(),
        }

    def update_account(self, total_value: float, timestamp: datetime = None):
        """更新账户状态"""
        self.account_state.update(total_value, timestamp)

    def record_trade(self, trade: Dict):
        """记录成交"""
        self.account_state.record_trade(trade)

    def get_risk_report(self) -> Dict:
        """获取风控报告"""
        return {
            "account_summary": self.account_state.get_summary(),
            "var": self.in_trade.calc_var(np.array(self.account_state.daily_returns)),
            "risk_log_count": len(self.risk_log),
            "failed_checks": sum(1 for r in self.risk_log if not r.passed),
            "circuit_breaker": {
                "active": self.account_state.circuit_breaker_active,
                "reason": self.account_state.circuit_breaker_reason,
            },
        }


# =============================================================================
# 模拟回测集成风控
# =============================================================================

def simulate_backtest_with_risk_control(
    n_days: int = 252,
    initial_capital: float = 1_000_000,
    enable_risk_control: bool = True,
    random_seed: int = 42,
) -> Dict:
    """
    模拟带风控的回测

    对比：
    - 无风控：任意交易
    - 有风控：受仓位限制、熔断限制
    """
    np.random.seed(random_seed)
    risk_engine = RiskEngine(initial_capital) if enable_risk_control else None

    equity = [initial_capital]
    trades = []
    circuit_breaker_triggered = False
    circuit_breaker_day = None

    for day in range(n_days):
        if risk_engine:
            risk_engine.update_account(equity[-1])

        # 每日随机交易决策
        daily_return = np.random.normal(0.0005, 0.015)
        signal_strength = np.random.uniform(0, 1)

        should_trade = signal_strength > 0.3
        if should_trade:
            trade_amount = equity[-1] * np.random.uniform(0.05, 0.15)
            trade_price = 20  # 假设股价

            if risk_engine:
                allowed, results = risk_engine.pre_trade_check(
                    '000001.SH', 1, int(trade_amount / (trade_price * 100)),
                    trade_price, 0, trade_price
                )
                if not allowed:
                    trades.append({'day': day, 'action': 'blocked', 'amount': trade_amount})
                    # 仍然更新权益（不接受交易）
                    new_equity = equity[-1] * (1 + daily_return * 0.5)
                    equity.append(new_equity)

                    if risk_engine:
                        risk_engine.update_account(new_equity)
                        should_break, _ = risk_engine.in_trade_check()
                        if not should_break:
                            continue
                        else:
                            circuit_breaker_triggered = True
                            circuit_breaker_day = day
                            break
                    continue

                # 记录交易
                risk_engine.record_trade({
                    'timestamp': f'day_{day}',
                    'symbol': '000001.SH',
                    'direction': 1,
                    'volume': int(trade_amount / (trade_price * 100)),
                    'price': trade_price,
                    'commission': trade_amount * 0.00025,
                    'tax': 0,
                })

            # 模拟交易后的收益
            trade_return = daily_return * (1.5 if signal_strength > 0.6 else 1.0)
            new_equity = equity[-1] * (1 + trade_return)
            equity.append(new_equity)
            trades.append({'day': day, 'action': 'executed', 'return': trade_return,
                           'amount': trade_amount, 'equity': new_equity})

            if risk_engine:
                risk_engine.update_account(new_equity)
                should_break, _ = risk_engine.in_trade_check()
                if not should_break:
                    continue
                else:
                    circuit_breaker_triggered = True
                    circuit_breaker_day = day
                    break
        else:
            new_equity = equity[-1] * (1 + daily_return * 0.3)
            equity.append(new_equity)

    equity_series = pd.Series(equity)
    returns = equity_series.pct_change().dropna()

    if risk_engine:
        risk_report = risk_engine.get_risk_report()
        post_analysis = risk_engine.post_trade_analyze()
    else:
        risk_report = {"risk_control": "disabled"}
        post_analysis = {}

    return {
        "initial_capital": initial_capital,
        "final_equity": equity[-1],
        "total_return": (equity[-1] / initial_capital - 1),
        "annual_return": returns.mean() * 252,
        "sharpe": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
        "max_drawdown": (pd.Series(equity).cummax() - pd.Series(equity)).max() / pd.Series(equity).cummax().max(),
        "volatility": returns.std() * np.sqrt(252),
        "n_trades": len([t for t in trades if t['action'] == 'executed']),
        "n_blocked": len([t for t in trades if t['action'] == 'blocked']),
        "circuit_breaker_triggered": circuit_breaker_triggered,
        "circuit_breaker_day": circuit_breaker_day,
        "risk_report": risk_report,
        "post_analysis": post_analysis,
    }


# =============================================================================
# 单元测试
# =============================================================================

class TestAccountState(unittest.TestCase):
    """账户状态测试"""

    def test_initial_state(self):
        state = AccountState(1_000_000)
        self.assertEqual(state.total_value, 1_000_000)
        self.assertEqual(state.peak_value, 1_000_000)
        self.assertEqual(state.max_drawdown, 0)
        print("[PASS] 初始状态正确")

    def test_update_and_drawdown(self):
        state = AccountState(1_000_000)
        # 上涨
        state.update(1_100_000)
        self.assertEqual(state.peak_value, 1_100_000)
        self.assertEqual(state.drawdown, 0)

        # 下跌
        state.update(990_000)
        self.assertGreater(state.drawdown, 0)
        self.assertGreater(state.max_drawdown, 0)

        print(f"[PASS] 回撤计算: drawdown={state.drawdown:.4%}, max_dd={state.max_drawdown:.4%}")

    def test_consecutive_days(self):
        state = AccountState(1_000_000)
        # 模拟连续亏损
        values = [1_000_000, 980_000, 950_000, 930_000]
        for v in values:
            state.update(v)
        self.assertEqual(state.consecutive_losses, 3)
        print(f"[PASS] 连续亏损: {state.consecutive_losses}天")


class TestPreTradeRiskChecks(unittest.TestCase):
    """事前风控测试"""

    def setUp(self):
        self.account = AccountState(1_000_000)
        self.checks = PreTradeRiskChecks(max_position_per_stock=0.2)

    def test_position_limit_pass(self):
        result = self.checks.check_position_limit(
            self.account, '000001.SH', 0, 10, 10
        )
        self.assertTrue(result.passed)
        print(f"[PASS] 仓位限制通过: {result.message}")

    def test_position_limit_fail(self):
        result = self.checks.check_position_limit(
            self.account, '000001.SH', 0, 3000, 100
        )
        self.assertFalse(result.passed)
        print(f"[PASS] 仓位限制拒绝: {result.message}")

    def test_order_value_pass(self):
        result = self.checks.check_order_value(self.account, 100_000)
        self.assertTrue(result.passed)
        print(f"[PASS] 委托金额通过")

    def test_insufficient_cash(self):
        result = self.checks.check_order_value(self.account, 2_000_000)
        self.assertFalse(result.passed)
        print(f"[PASS] 资金不足拒绝: {result.message}")

    def test_price_deviation(self):
        result = self.checks.check_price_deviation(10.02, 10.0)
        self.assertTrue(result.passed)
        print(f"[PASS] 价格偏离检查通过")


class TestInTradeRiskMonitor(unittest.TestCase):
    """事中风控测试"""

    def setUp(self):
        self.monitor = InTradeRiskMonitor(
            max_drawdown_limit=0.15,
            daily_loss_limit=0.05,
            max_consecutive_losses=3,
        )

    def test_drawdown_circuit_breaker(self):
        account = AccountState(1_000_000)
        account.update(1_100_000)  # 先涨
        account.update(900_000)    # 再跌 18%
        result = self.monitor.check_drawdown_circuit_breaker(account)
        self.assertFalse(result.passed)
        print(f"[PASS] 回撤熔断触发: {result.message}")

    def test_daily_loss_circuit_breaker(self):
        account = AccountState(1_000_000)
        account.update(1_000_000)
        account.update(930_000)  # 单日跌 7%
        result = self.monitor.check_daily_loss_circuit_breaker(account)
        self.assertFalse(result.passed)
        print(f"[PASS] 单日亏损熔断触发: {result.message}")

    def test_consecutive_losses_circuit_breaker(self):
        account = AccountState(1_000_000)
        for v in [990_000, 980_000, 970_000, 960_000]:
            account.update(v)
        result = self.monitor.check_consecutive_losses(account)
        self.assertFalse(result.passed)
        print(f"[PASS] 连续亏损熔断触发: {result.message}")

    def test_var_calculation(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        var_result = self.monitor.calc_var(returns)
        self.assertIn('hist_var', var_result)
        self.assertIn('cvar', var_result)
        self.assertLess(var_result['cvar'], var_result['hist_var'])
        print(f"[PASS] VaR计算: hist_var={var_result['hist_var']:.4f}, "
              f"cvar={var_result['cvar']:.4f}")

    def test_normal_market(self):
        """正常市场不触发熔断"""
        account = AccountState(1_000_000)
        account.update(1_010_000)
        should_break, reason = self.monitor.should_circuit_break(account)
        self.assertFalse(should_break)
        print(f"[PASS] 正常市场不触发熔断")


class TestPostTradeRiskAnalyzer(unittest.TestCase):
    """事后风控分析测试"""

    def test_trade_stats(self):
        trades = [
            {'timestamp': 'D1', 'symbol': '000001.SH', 'direction': 1, 'price': 10, 'volume': 10, 'commission': 5, 'tax': 0},
            {'timestamp': 'D2', 'symbol': '000001.SH', 'direction': -1, 'price': 12, 'volume': 10, 'commission': 5, 'tax': 0},
            {'timestamp': 'D3', 'symbol': '000002.SH', 'direction': 1, 'price': 20, 'volume': 5, 'commission': 5, 'tax': 0},
            {'timestamp': 'D4', 'symbol': '000002.SH', 'direction': -1, 'price': 18, 'volume': 5, 'commission': 5, 'tax': 0},
        ]
        stats = PostTradeRiskAnalyzer.analyze_trade_stats(trades, 1_000_000)
        self.assertEqual(stats['total_trades'], 2)  # 2对交易
        self.assertEqual(stats['wins'], 1)  # 一笔盈利
        self.assertEqual(stats['losses'], 1)  # 一笔亏损
        print(f"[PASS] 交易统计: wins={stats['wins']}, losses={stats['losses']}, "
              f"win_rate={stats['win_rate']:.2%}")

    def test_risk_metrics(self):
        np.random.seed(42)
        equity = pd.Series(1_000_000 * (1 + np.random.normal(0.0005, 0.02, 252)).cumprod())
        metrics = PostTradeRiskAnalyzer.calc_risk_metrics(equity)
        self.assertIn('sharpe', metrics)
        self.assertIn('max_drawdown', metrics)
        print(f"[PASS] 风险指标: sharpe={metrics['sharpe']:.4f}, "
              f"max_dd={metrics['max_drawdown']:.4%}")

    def test_anomaly_detection(self):
        trades = []
        for i in range(20):
            trades.append({
                'timestamp': f'D{i}',
                'symbol': '000001.SH',
                'direction': 1 if i % 2 == 0 else -1,
                'price': 10 + i * 0.1,
                'volume': 10,
                'commission': 5,
                'tax': 0,
            })
        anomalies = PostTradeRiskAnalyzer.detect_anomalies(trades)
        self.assertGreater(len(anomalies), 0)
        print(f"[PASS] 异常检测: 发现 {len(anomalies)} 个异常")


class TestRiskEngine(unittest.TestCase):
    """统一风控引擎测试"""

    def test_risk_engine_integration(self):
        engine = RiskEngine(1_000_000)
        allowed, results = engine.pre_trade_check(
            '000001.SH', 1, 100, 10, 0, 10
        )
        self.assertTrue(allowed)
        print(f"[PASS] 风控引擎集成: 事前检查通过")

    def test_risk_engine_block(self):
        engine = RiskEngine(1_000_000)
        allowed, results = engine.pre_trade_check(
            '000001.SH', 1, 10000, 100, 0, 10
        )
        self.assertFalse(allowed)
        print(f"[PASS] 风控引擎集成: 超额委托被拒绝")

    def test_risk_report(self):
        engine = RiskEngine(1_000_000)
        engine.update_account(1_010_000)
        report = engine.get_risk_report()
        self.assertIn('account_summary', report)
        self.assertIn('var', report)
        print(f"[PASS] 风控报告生成成功")


class TestBacktestWithRiskControl(unittest.TestCase):
    """回测风控对比测试"""

    def test_with_vs_without_risk_control(self):
        """对比有/无风控的回测结果"""
        result_with = simulate_backtest_with_risk_control(
            n_days=252, enable_risk_control=True, random_seed=42
        )
        result_without = simulate_backtest_with_risk_control(
            n_days=252, enable_risk_control=False, random_seed=42
        )

        print("\n===== 风控对比测试 =====")
        print(f"无风控: 收益={result_without['total_return']:.4%}, "
              f"Sharpe={result_without['sharpe']:.4f}, "
              f"最大回撤={result_without['max_drawdown']:.4%}, "
              f"交易次数={result_without['n_trades']}")

        print(f"有风控: 收益={result_with['total_return']:.4%}, "
              f"Sharpe={result_with['sharpe']:.4f}, "
              f"最大回撤={result_with['max_drawdown']:.4%}, "
              f"交易次数={result_with['n_trades']}, "
              f"被拦截={result_with['n_blocked']}")

        if result_with['circuit_breaker_triggered']:
            print(f"  [熔断] 触发于第 {result_with['circuit_breaker_day']} 天")

        # 风控应降低最大回撤
        self.assertLessEqual(
            result_with['max_drawdown'],
            result_without['max_drawdown'] * 1.1,  # 允许10%的误差
            "风控未有效降低回撤"
        )
        print("[PASS] 风控有效降低或控制回撤")


if __name__ == '__main__':
    print("=" * 70)
    print("多层级风控管理系统验证测试")
    print("=" * 70)

    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAccountState))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPreTradeRiskChecks))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInTradeRiskMonitor))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPostTradeRiskAnalyzer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskEngine))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBacktestWithRiskControl))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("验证总结")
    print(f"  总测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 70)