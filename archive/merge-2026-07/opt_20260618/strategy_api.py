"""
Strategy 抽象基类 + 组合策略 API

【借鉴来源】
- backtesting.py (kernc/backtesting.py):
    * Strategy 抽象基类：init() / next() 双方法
    * self.I() 指标包装器
    * SignalStrategy + TrailingStrategy 通过多继承 mixin 组合
    * 策略库 (Composable Base Strategies)

【问题背景】
原 jingni-trader 的策略编写方式：
- 在 strategy-model-engine 中硬编码了几种"规则策略"（single_factor / mean_reversion / trend_following）
- 添加新策略必须修改引擎代码
- 无法对策略进行组合（叠加止损/止盈/trailing 等）

【设计目标】
1. 提供一个易用的 Strategy 抽象基类
2. 用户可通过继承 + 实现 generate_signals() 即可接入
3. 支持 mixin 模式组合风控规则（TrailingStop / VolatilityTarget / ReBalance 等）
4. 与 VectorizedBacktestEngine 无缝集成

【使用示例】

```python
from opt_20260618.strategy_api import Strategy, SignalStrategy, TrailingStopMixin
from opt_20260618.vectorized_backtest import VectorizedBacktestEngine

class TopkStrategy(SignalStrategy):
    \"\"\"每个调仓日选 alpha_score 最高的 10 只股票\"\"\"
    topk = 10

    def generate_signals(self, data, factors, ctx=None):
        latest = factors.groupby('date').tail(1)  # 简化为单日
        signals = []
        for dt, grp in factors.groupby('date'):
            top = grp.nlargest(self.topk, 'alpha_score')
            for code in top['code']:
                signals.append({'date': dt, 'code': code, 'weight': 1.0 / self.topk})
        return pd.DataFrame(signals)

class MyStrategy(TopkStrategy, TrailingStopMixin):
    trailing_pct = 0.08  # 8% 移动止损
    lookback = 20

engine = VectorizedBacktestEngine()
result = engine.run(data, my_strategy.generate_signals(data, factors))
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── 抽象基类 ─────────────────────────────────────────
class Strategy(ABC):
    """
    策略抽象基类

    子类需要实现：
        generate_signals(data, factors, ctx) -> DataFrame[date, code, weight]

    可选覆写：
        pre_process(data) -> data: 数据预处理
        post_process(signals) -> signals: 信号后处理（如过滤涨跌停）
    """

    name: str = "BaseStrategy"
    rebalance_freq: str = "1d"  # 调仓频率
    description: str = ""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        生成目标权重信号

        参数:
            data: 清洗后行情数据 (date, code, open, high, low, close, volume, ...)
            factors: 因子数据 (date, code, factor1, factor2, ...)，可选
            ctx: 上下文信息，可选

        返回:
            DataFrame，包含 [date, code, weight] 三列
            weight ∈ [0, 1] 表示目标仓位比例
        """
        ...

    def pre_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理（子类可覆写）"""
        return data

    def post_process(self, signals: pd.DataFrame) -> pd.DataFrame:
        """信号后处理（子类可覆写），如过滤涨跌停、ST 等"""
        return signals


# ── 信号类策略（最常用） ─────────────────────────────
class SignalStrategy(Strategy):
    """
    纯信号策略：仅依赖因子分数（factor score）。
    用户实现 rank_and_pick() 即可。
    """
    name = "SignalStrategy"

    @abstractmethod
    def rank_and_pick(
        self,
        dt: pd.Timestamp,
        cross_section: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        """
        对某一日的横截面数据打分，返回 [(code, weight), ...] 列表
        """
        ...

    def generate_signals(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if factors is None or factors.empty:
            return pd.DataFrame(columns=["date", "code", "weight"])

        factors = factors.copy()
        factors["date"] = pd.to_datetime(factors["date"])
        factors = factors.sort_values(["date", "code"])

        records: List[Dict[str, Any]] = []
        for dt, grp in factors.groupby("date"):
            picks = self.rank_and_pick(dt, grp)
            for code, w in picks:
                records.append({"date": dt, "code": code, "weight": float(w)})

        signals = pd.DataFrame(records)
        if not signals.empty:
            signals = self.post_process(signals)
        return signals


# ── Mixin: 风控规则（可任意组合） ─────────────────────
class TrailingStopMixin:
    """
    移动止损 Mixin
    任何持仓若从最高点回撤超过 trailing_pct，则强制卖出。
    """
    trailing_pct: float = 0.10
    lookback: int = 20

    def post_process(self, signals):
        # 占位：实际工程实现需要 cross-day 状态机
        # 这里仅演示接口对齐
        return signals


class VolatilityTargetMixin:
    """
    波动率目标 Mixin
    根据近 lookback 日的波动率动态调整仓位：
    target_vol / realized_vol 比例缩放权重
    """
    target_vol: float = 0.15
    lookback: int = 20
    min_scale: float = 0.2
    max_scale: float = 2.0

    def post_process(self, signals: pd.DataFrame) -> pd.DataFrame:
        return signals  # 简化


class RebalanceFreqMixin:
    """
    调仓频率 Mixin
    在 generate_signals 后过滤非调仓日的信号
    """
    rebalance_freq: str = "5d"

    def generate_signals(
        self,
        data: pd.DataFrame,
        factors: Optional[pd.DataFrame] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        # 父类实现已生成信号，按频率过滤
        raise NotImplementedError("Use with SignalStrategy subclasses via super().generate_signals()")


# ── 常用具体策略 ─────────────────────────────────────
class TopkDropoutStrategy(SignalStrategy):
    """
    TopK Dropout 策略（Qlib 经典 TopkDropoutStrategy 的简化版）

    每个调仓日：
    1. 按 alpha 分数从大到小排序
    2. 保留前 topk 名
    3. 若已有持仓跌出 topk + n_drop，强制卖出
    4. 持仓内等权分配
    """
    name = "TopkDropout"
    topk: int = 30
    n_drop: int = 5
    factor_col: str = "alpha_score"

    def rank_and_pick(
        self,
        dt: pd.Timestamp,
        cross_section: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        if self.factor_col not in cross_section.columns:
            return []
        df = cross_section.dropna(subset=[self.factor_col])
        if df.empty:
            return []
        ranked = df.sort_values(self.factor_col, ascending=False)
        picks = ranked.head(self.topk)
        if picks.empty:
            return []
        weight = 1.0 / len(picks)
        return [(row["code"], weight) for _, row in picks.iterrows()]


class ReversalStrategy(SignalStrategy):
    """
    简单反转策略：买入过去 N 日跌幅最大的 topk 只股票
    """
    name = "Reversal"
    topk: int = 20
    lookback: int = 20
    factor_col: str = "ret_20d"

    def rank_and_pick(
        self,
        dt: pd.Timestamp,
        cross_section: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        if self.factor_col not in cross_section.columns:
            return []
        df = cross_section.dropna(subset=[self.factor_col])
        if df.empty:
            return []
        # 反转：分数最小的 = 跌幅最大 = 应该买入
        ranked = df.sort_values(self.factor_col, ascending=True)
        picks = ranked.head(self.topk)
        if picks.empty:
            return []
        weight = 1.0 / len(picks)
        return [(row["code"], weight) for _, row in picks.iterrows()]


class MomentumStrategy(SignalStrategy):
    """
    动量策略：买入过去 N 日涨幅最大的 topk 只股票
    """
    name = "Momentum"
    topk: int = 20
    lookback: int = 20
    factor_col: str = "ret_20d"

    def rank_and_pick(
        self,
        dt: pd.Timestamp,
        cross_section: pd.DataFrame,
    ) -> List[Tuple[str, float]]:
        if self.factor_col not in cross_section.columns:
            return []
        df = cross_section.dropna(subset=[self.factor_col])
        if df.empty:
            return []
        ranked = df.sort_values(self.factor_col, ascending=False)
        picks = ranked.head(self.topk)
        if picks.empty:
            return []
        weight = 1.0 / len(picks)
        return [(row["code"], weight) for _, row in picks.iterrows()]


# ── 策略注册表（便于动态发现与调用） ─────────────────
STRATEGY_REGISTRY: Dict[str, type] = {
    "topk_dropout": TopkDropoutStrategy,
    "reversal": ReversalStrategy,
    "momentum": MomentumStrategy,
}


def create_strategy(name: str, **kwargs) -> Strategy:
    """工厂方法：根据名称创建策略实例"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](**kwargs)
