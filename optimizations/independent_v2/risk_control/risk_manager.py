"""
风险控制模块

借鉴来源:
1. Kelly Criterion 仓位管理
   - 原始论文: Kelly, J. L. (1956). "A New Interpretation of Information Rate"
   - 实践建议: Never Trade Full Kelly (gov.capital 风控清单 Rule 3)
   - f* = (b*p - q) / b, 其中 p=胜率, b=盈亏比, q=1-p
   - 实战中常用半凯利 (half-Kelly) 降低破产风险

2. ATR 动态止损
   - Wilder, J. W. (1978). "New Concepts in Technical Trading Systems"
   - 核心思想: 用 ATR 衡量波动率，止损距离 = n × ATR
   - 优于固定百分比止损：适应不同波动率环境

3. 回撤断路器 (Drawdown Circuit Breaker)
   - 实践来源: 量化风控标准实践
   - 当组合回撤超过阈值时，自动降低仓位（如回撤>10%仓位减半）
   - 防止亏损加速，给策略恢复留出空间

设计思路:
- jingni-trader 现有 backtest-engine 仅内置 T+1、涨跌停、佣金、滑点，
  缺少仓位管理、动态止损、回撤控制等主动风控能力。
- 本模块提供可组合的风控组件，可嵌入回测或实盘流程。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd


# ---------- 1. 凯利公式仓位管理 ----------

class KellySizer:
    """基于凯利公式的仓位管理器。

    支持半凯利 (half-Kelly) 模式以降低破产风险。
    """

    def __init__(self, fraction: float = 0.5, max_weight: float = 0.25):
        """
        参数:
            fraction: 凯利比例，0.5=半凯利（推荐），1.0=全凯利（激进）
            max_weight: 单只股票最大权重上限
        """
        if not 0 < fraction <= 1.0:
            raise ValueError("fraction 必须在 (0, 1] 区间")
        self.fraction = fraction
        self.max_weight = max_weight

    def estimate_from_trades(self, trades: pd.DataFrame) -> Dict[str, float]:
        """从历史成交记录估计胜率与盈亏比。

        参数:
            trades: 含 pnl 列的成交记录

        返回:
            {"win_rate": float, "win_loss_ratio": float, "kelly_f": float}
        """
        if trades.empty or "pnl" not in trades.columns:
            return {"win_rate": 0.5, "win_loss_ratio": 1.0, "kelly_f": 0.0}

        pnls = trades["pnl"].dropna()
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        if len(pnls) == 0:
            return {"win_rate": 0.5, "win_loss_ratio": 1.0, "kelly_f": 0.0}

        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.5
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly_f = self.calc_kelly(win_rate, win_loss_ratio)
        return {
            "win_rate": float(win_rate),
            "win_loss_ratio": float(win_loss_ratio),
            "kelly_f": float(kelly_f),
        }

    def calc_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """计算凯利比例。

        f* = (b*p - q) / b
        其中 p=胜率, q=1-p, b=盈亏比
        """
        p = win_rate
        q = 1 - p
        b = win_loss_ratio
        if b <= 0:
            return 0.0
        kelly = (b * p - q) / b
        # 应用比例（半凯利）并限制上限
        kelly = kelly * self.fraction
        return max(0.0, min(kelly, self.max_weight))

    def size_position(
        self,
        capital: float,
        win_rate: float,
        win_loss_ratio: float,
        price: float,
    ) -> Dict[str, float]:
        """计算建议持仓。

        返回:
            {"weight": float, "shares": int, "amount": float}
        """
        weight = self.calc_kelly(win_rate, win_loss_ratio)
        amount = capital * weight
        shares = int(amount / price / 100) * 100 if price > 0 else 0  # A股100股整手
        return {
            "weight": float(weight),
            "shares": int(shares),
            "amount": float(shares * price),
        }


# ---------- 2. ATR 动态止损 ----------

class ATRStopLoss:
    """基于 ATR 的动态止损管理器。"""

    def __init__(self, atr_period: int = 14, atr_multiplier: float = 2.0):
        """
        参数:
            atr_period: ATR 计算周期
            atr_multiplier: 止损距离 = multiplier × ATR
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def calc_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.Series:
        """计算 ATR (Average True Range)。

        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = TR 的 N 日 EMA
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        return atr

    def calc_stop_price(
        self, entry_price: float, atr: float, direction: str = "long"
    ) -> float:
        """计算止损价。

        参数:
            entry_price: 入场价
            atr: 当前 ATR 值
            direction: 'long' 或 'short'

        返回:
            止损价
        """
        stop_distance = self.atr_multiplier * atr
        if direction == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def generate_stop_signals(
        self, data: pd.DataFrame, entry_dates: pd.DataFrame
    ) -> pd.DataFrame:
        """为每笔入场生成止损价序列。

        参数:
            data: OHLCV 数据 (code, date, high, low, close)
            entry_dates: 含 code, date, entry_price 列的入场记录

        返回:
            含 code, date, atr, stop_price 列的 DataFrame
        """
        df = data.sort_values(["code", "date"]).copy()
        # 逐股票计算 ATR，避免 groupby.apply 返回 DataFrame 的问题
        atr_values = pd.Series(index=df.index, dtype=float)
        for code, grp in df.groupby("code"):
            atr = self.calc_atr(grp["high"], grp["low"], grp["close"])
            atr_values.loc[grp.index] = atr.values
        df["atr"] = atr_values

        # 对每个入场记录，从入场日起追踪止损价
        results = []
        for _, entry in entry_dates.iterrows():
            code = entry["code"]
            entry_date = entry["date"]
            entry_price = entry["entry_price"]
            stock_df = df[(df["code"] == code) & (df["date"] >= entry_date)].copy()
            if stock_df.empty:
                continue
            # 初始止损价
            initial_atr = stock_df["atr"].iloc[0]
            if np.isnan(initial_atr):
                continue
            stop_price = self.calc_stop_price(entry_price, initial_atr, "long")
            # 向量化追踪：止损价只上移不下移（trailing stop）
            for _, row in stock_df.iterrows():
                if np.isnan(row["atr"]):
                    continue
                new_stop = row["close"] - self.atr_multiplier * row["atr"]
                stop_price = max(stop_price, new_stop)  # 只上移
                results.append({
                    "code": code,
                    "date": row["date"],
                    "atr": row["atr"],
                    "stop_price": stop_price,
                    "should_stop": row["close"] < stop_price,
                })
        return pd.DataFrame(results)


# ---------- 3. 回撤断路器 ----------

@dataclass
class DrawdownCircuitBreaker:
    """回撤断路器：当组合回撤超过阈值时自动降低仓位。

    规则:
    - 回撤 < warn_threshold: 正常仓位
    - warn_threshold <= 回撤 < reduce_threshold: 仓位 × reduce_factor
    - 回撤 >= reduce_threshold: 仓位 × emergency_factor（紧急降仓）
    - 回撤 >= halt_threshold: 清仓
    """
    warn_threshold: float = 0.05       # 5% 回撤预警
    reduce_threshold: float = 0.10     # 10% 回撤降仓
    halt_threshold: float = 0.20       # 20% 回撤清仓
    reduce_factor: float = 0.75        # 降仓后保留 75% 仓位
    emergency_factor: float = 0.50     # 紧急降仓后保留 50% 仓位

    def calc_drawdown(self, equity: pd.Series) -> pd.Series:
        """计算回撤序列。"""
        cum_max = equity.cummax()
        return (equity - cum_max) / cum_max

    def get_position_multiplier(self, current_drawdown: float) -> float:
        """根据当前回撤返回仓位乘数。

        参数:
            current_drawdown: 当前回撤值（负数，如 -0.08 表示 -8%）

        返回:
            仓位乘数 (0.0 ~ 1.0)
        """
        dd = abs(current_drawdown)
        if dd >= self.halt_threshold:
            return 0.0  # 清仓
        elif dd >= self.reduce_threshold:
            return self.emergency_factor
        elif dd >= self.warn_threshold:
            return self.reduce_factor
        else:
            return 1.0

    def apply_to_equity(
        self, equity: pd.Series, target_weights: pd.DataFrame
    ) -> pd.DataFrame:
        """将断路器应用于目标权重序列。

        参数:
            equity: 组合净值序列
            target_weights: 目标权重矩阵 (date × code)

        返回:
            调整后的权重矩阵
        """
        drawdown = self.calc_drawdown(equity)
        # 对每个日期计算仓位乘数
        multipliers = drawdown.apply(self.get_position_multiplier)
        # 广播到权重矩阵
        adjusted = target_weights.mul(multipliers, axis=0)
        return adjusted

    def get_status(self, current_drawdown: float) -> Dict[str, Any]:
        """获取当前风控状态。"""
        mult = self.get_position_multiplier(current_drawdown)
        dd = abs(current_drawdown)
        if dd >= self.halt_threshold:
            status = "HALT"
        elif dd >= self.reduce_threshold:
            status = "EMERGENCY_REDUCE"
        elif dd >= self.warn_threshold:
            status = "WARN_REDUCE"
        else:
            status = "NORMAL"
        return {
            "status": status,
            "drawdown": float(current_drawdown),
            "position_multiplier": float(mult),
            "warn_threshold": self.warn_threshold,
            "reduce_threshold": self.reduce_threshold,
            "halt_threshold": self.halt_threshold,
        }


# ---------- 4. 统一风控管理器 ----------

class RiskManager:
    """统一风控管理器，组合 Kelly + ATR + DrawdownBreaker。"""

    def __init__(
        self,
        kelly: Optional[KellySizer] = None,
        atr_stop: Optional[ATRStopLoss] = None,
        dd_breaker: Optional[DrawdownCircuitBreaker] = None,
    ):
        self.kelly = kelly or KellySizer()
        self.atr_stop = atr_stop or ATRStopLoss()
        self.dd_breaker = dd_breaker or DrawdownCircuitBreaker()

    def evaluate(
        self,
        equity: pd.Series,
        trades: pd.DataFrame,
        current_positions: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """综合风控评估。"""
        # 1. 凯利仓位建议
        kelly_stats = self.kelly.estimate_from_trades(trades)

        # 2. 当前回撤状态
        if len(equity) > 0:
            drawdown = self.dd_breaker.calc_drawdown(equity)
            current_dd = float(drawdown.iloc[-1])
        else:
            current_dd = 0.0
        dd_status = self.dd_breaker.get_status(current_dd)

        return {
            "kelly": kelly_stats,
            "drawdown": dd_status,
            "recommendation": self._build_recommendation(kelly_stats, dd_status),
        }

    def _build_recommendation(
        self, kelly_stats: Dict, dd_status: Dict
    ) -> str:
        """生成风控建议文本。"""
        parts = []
        if dd_status["status"] == "HALT":
            parts.append("回撤已超过清仓阈值，建议立即清仓")
        elif dd_status["status"] in ("EMERGENCY_REDUCE", "WARN_REDUCE"):
            parts.append(
                f"回撤 {abs(dd_status['drawdown'])*100:.1f}% 触发降仓，"
                f"建议仓位乘数 {dd_status['position_multiplier']:.2f}"
            )
        else:
            parts.append("回撤在正常范围内")

        kelly_f = kelly_stats["kelly_f"]
        if kelly_f > 0:
            parts.append(
                f"凯利建议仓位 {kelly_f*100:.1f}% "
                f"(胜率 {kelly_stats['win_rate']*100:.1f}%, "
                f"盈亏比 {kelly_stats['win_loss_ratio']:.2f})"
            )
        else:
            parts.append("历史数据不足以计算凯利仓位")

        return "；".join(parts)
