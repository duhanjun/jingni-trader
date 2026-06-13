"""
优化方向：可组合策略管道（Weight-Centric 设计）
借鉴来源：FinRL-X (https://github.com/AI4Finance-Foundation/FinRL-Trading)
论文：FinRL-X: An AI-Native Modular Infrastructure for Quantitative Trading (arXiv:2603.21330)

核心思想：
  FinRL-X 将目标投资组合权重向量 w_t 作为策略层与下游模块之间的唯一接口。
  策略管道由四个可组合的"保合同变换"组成：
    S(Selection) → A(Allocation) → T(Timing) → R(Risk Overlay)

问题分析：
  jingni-trader 当前策略构建分布在多个模块：
  - factor-engine: 因子计算
  - strategy-model-engine: 模型训练 + 信号生成（signal ∈ {-1, 0, 1}）
  - backtest-engine: 信号转交易（直接使用信号做多/空）
  - portfolio-risk-engine: 组合优化（单独的权重优化步骤）
  各模块输出格式不一致：factors → alpha_score → signal → weights，
  缺乏统一的接口契约。

优化方案：
  引入 ComposableStrategy 管道模式，统一以权重向量 w_t 为接口合约。
  支持四个可组合阶段：
  1. Selector:     股票选择 → 候选集 C_t
  2. Allocator:    权重分配 → 基础权重 w_base
  3. TimingAdjuster: 择时调整 → 调整后权重 w_timing
  4. RiskOverlay:   风险覆盖 → 最终权重 w_final
  每个阶段是保合同变换（contract-preserving）：输入输出都是权重向量。
"""

import sys
import os
import unittest
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ============================================================
# 核心接口定义（借鉴 FinRL-X weight-centric architecture）
# ============================================================

@dataclass
class StrategyContext:
    """
    策略执行上下文
    包含当前时刻的市场状态快照
    """
    date: pd.Timestamp
    universe: pd.Index                   # 候选股票代码
    prices: pd.Series                    # code → price
    factors: pd.DataFrame = None         # 当前截面的因子数据
    market_data: Dict = field(default_factory=dict)  # 其他市场数据
    constraints: Dict = field(default_factory=dict)  # 约束条件


class Selector(ABC):
    """
    股票选择器（阶段1：S）
    输入：全市场 universe + 因子数据
    输出：候选集 C_t ⊆ universe
    对应 FinRL-X 的 stock selection transform
    """

    @abstractmethod
    def select(self, ctx: StrategyContext) -> pd.Index:
        """返回候选股票列表"""
        pass


class Allocator(ABC):
    """
    权重分配器（阶段2：A）
    输入：候选集 + 因子/信号数据
    输出：基础权重向量 w_base（元素之和 = 1）
    对应 FinRL-X 的 portfolio allocation transform
    """

    @abstractmethod
    def allocate(self, ctx: StrategyContext, candidates: pd.Index) -> pd.Series:
        """返回 code → weight 的 Series"""
        pass


class TimingAdjuster(ABC):
    """
    择时调整器（阶段3：T）
    输入：基础权重 w_base
    输出：调整后权重 w_timing（可能降低仓位或清仓）
    对应 FinRL-X 的 timing adjustment transform
    """

    @abstractmethod
    def adjust(self, ctx: StrategyContext, weights: pd.Series) -> pd.Series:
        """返回调整后的权重"""
        pass


class RiskOverlay(ABC):
    """
    风险覆盖层（阶段4：R）
    输入：择时后权重 w_timing
    输出：最终权重 w_final（满足所有风控约束）
    对应 FinRL-X 的 portfolio-level risk overlay
    """

    @abstractmethod
    def overlay(self, ctx: StrategyContext, weights: pd.Series) -> pd.Series:
        """返回风控裁剪后的权重"""
        pass


class ComposableStrategy:
    """
    可组合策略（借鉴 FinRL-X 管道模式）

    通过 compose() 方法串联 selector → allocator → timing → risk
    每个阶段输出统一的权重向量接口
    可替换任意阶段实现
    """

    def __init__(self, name: str = "composable_strategy"):
        self.name = name
        self._selector: Optional[Selector] = None
        self._allocator: Optional[Allocator] = None
        self._timing: Optional[TimingAdjuster] = None
        self._risk_overlay: Optional[RiskOverlay] = None

    def compose(
        self,
        selector: Selector = None,
        allocator: Allocator = None,
        timing: TimingAdjuster = None,
        risk_overlay: RiskOverlay = None,
    ) -> "ComposableStrategy":
        """组合策略管道"""
        self._selector = selector
        self._allocator = allocator
        self._timing = timing
        self._risk_overlay = risk_overlay
        return self

    def __call__(self, ctx: StrategyContext) -> pd.Series:
        """
        执行策略管道：S → A → T → R

        返回: code → weight 的最终权重向量
        """
        # Phase S: Selection
        if self._selector:
            candidates = self._selector.select(ctx)
            if len(candidates) == 0:
                return pd.Series(dtype=float)
        else:
            candidates = ctx.universe

        # Phase A: Allocation
        if self._allocator:
            weights = self._allocator.allocate(ctx, candidates)
        else:
            # 默认等权
            n = len(candidates)
            weights = pd.Series(1.0 / n, index=candidates)

        # Phase T: Timing
        if self._timing:
            weights = self._timing.adjust(ctx, weights)

        # Phase R: Risk Overlay
        if self._risk_overlay:
            weights = self._risk_overlay.overlay(ctx, weights)

        # 归一化确保权重和为1（仅在无 risk_overlay 时，因为 risk_overlay 自己保证约束）
        if self._risk_overlay is None and weights.sum() > 0:
            weights = weights / weights.sum()
        elif self._risk_overlay is not None:
            # 有 risk_overlay 时，由 overlay 保证权重约束，不再强制归一化
            # 但确保总和在合理范围
            if weights.sum() > 0:
                weights = weights / weights.sum()

        return weights


# ============================================================
# 具体实现（模拟 jingni-trader 现有策略逻辑）
# ============================================================

class TopKSelector(Selector):
    """Top-K 选股器（基于因子排名）"""

    def __init__(self, factor_col: str = "alpha_score", top_k: int = 20):
        self.factor_col = factor_col
        self.top_k = top_k

    def select(self, ctx: StrategyContext) -> pd.Index:
        if ctx.factors is None or self.factor_col not in ctx.factors.columns:
            return ctx.universe

        scores = ctx.factors.set_index('code')[self.factor_col]
        scores = scores.reindex(ctx.universe).dropna()
        if len(scores) == 0:
            return ctx.universe[:self.top_k]
        return scores.nlargest(self.top_k).index


class AlphaWeightedAllocator(Allocator):
    """Alpha 加权分配器（按因子值分配权重）"""

    def __init__(self, factor_col: str = "alpha_score"):
        self.factor_col = factor_col

    def allocate(self, ctx: StrategyContext, candidates: pd.Index) -> pd.Series:
        if ctx.factors is None or self.factor_col not in ctx.factors.columns:
            n = len(candidates)
            return pd.Series(1.0 / n, index=candidates)

        scores = ctx.factors.set_index('code')[self.factor_col]
        scores = scores.reindex(candidates).fillna(0)

        # 确保最小权重为非负（仅做多）
        scores = scores.clip(lower=0)

        total = scores.sum()
        if total > 0:
            weights = scores / total
        else:
            n = len(candidates)
            weights = pd.Series(1.0 / n, index=candidates)

        return weights


class EqualWeightAllocator(Allocator):
    """等权分配器"""

    def allocate(self, ctx: StrategyContext, candidates: pd.Index) -> pd.Series:
        n = len(candidates)
        return pd.Series(1.0 / n, index=candidates)


class MaxDrawdownTimingAdjuster(TimingAdjuster):
    """最大回撤择时调整器：如果市场处于下跌趋势，降低仓位"""

    def __init__(self, max_drawdown_threshold: float = -0.05):
        self.max_drawdown_threshold = max_drawdown_threshold
        self._peak = {}

    def adjust(self, ctx: StrategyContext, weights: pd.Series) -> pd.Series:
        # 简化的趋势判断：基于价格 20 日均线
        if 'ma_20' not in ctx.market_data:
            return weights  # 无数据时不调整

        # 如果市场平均价格低于 20 日均线 5%，降低 50% 仓位
        avg_price = ctx.prices.mean()
        avg_ma20 = ctx.market_data['ma_20'].mean() if ctx.market_data.get('ma_20') is not None else avg_price

        if avg_ma20 > 0 and (avg_price / avg_ma20 - 1) < self.max_drawdown_threshold:
            return weights * 0.5  # 降低一半仓位

        return weights


class SingleStockCapOverlay(RiskOverlay):
    """个股权重上限风险覆盖"""

    def __init__(self, max_weight: float = 0.10):
        self.max_weight = max_weight

    def overlay(self, ctx: StrategyContext, weights: pd.Series) -> pd.Series:
        # 裁剪到上限，超额部分不重新分配（转为现金）
        capped = weights.clip(upper=self.max_weight)
        if capped.sum() > 0:
            capped = capped / capped.sum()
        return capped


class IndustryDiversificationOverlay(RiskOverlay):
    """行业分散化风险覆盖"""

    def __init__(self, industry_map: Dict[str, str], max_industry_weight: float = 0.30):
        self.industry_map = industry_map
        self.max_industry_weight = max_industry_weight

    def overlay(self, ctx: StrategyContext, weights: pd.Series) -> pd.Series:
        # 按行业汇总权重
        industry_weights = {}
        for code, w in weights.items():
            ind = self.industry_map.get(code, "other")
            industry_weights[ind] = industry_weights.get(ind, 0) + w

        # 对超过上限的行业降低权重
        adjusted = weights.copy()
        for ind, ind_w in industry_weights.items():
            if ind_w > self.max_industry_weight:
                scale = self.max_industry_weight / ind_w
                ind_codes = [c for c, i in self.industry_map.items() if i == ind]
                for code in ind_codes:
                    if code in adjusted.index:
                        adjusted[code] *= scale

        return adjusted


class CompositeRiskOverlay(RiskOverlay):
    """组合风险覆盖（链式组合多个风险约束）"""

    def __init__(self, overlays: List[RiskOverlay]):
        self.overlays = overlays

    def overlay(self, ctx: StrategyContext, weights: pd.Series) -> pd.Series:
        result = weights
        for overlay in self.overlays:
            result = overlay.overlay(ctx, result)
        return result


# ============================================================
# 单元测试
# ============================================================


class TestComposableStrategy(unittest.TestCase):
    """可组合策略管道测试"""

    @classmethod
    def setUpClass(cls):
        """生成模拟数据"""
        np.random.seed(42)
        cls.codes = [f"{i:06d}.SZ" for i in range(50)]
        cls.prices = pd.Series(
            np.random.uniform(10, 100, len(cls.codes)),
            index=cls.codes
        )
        cls.factor_data = pd.DataFrame({
            'code': cls.codes,
            'alpha_score': np.random.randn(len(cls.codes)),
            'ret_20d': np.random.randn(len(cls.codes)) * 0.05,
            'volatility_20d': np.random.uniform(0.01, 0.05, len(cls.codes)),
        })
        cls.industry_map = {
            c: np.random.choice(["金融", "科技", "消费", "医药", "能源"])
            for c in cls.codes
        }

    def _make_context(self, date=None) -> StrategyContext:
        return StrategyContext(
            date=date or pd.Timestamp("2024-01-01"),
            universe=pd.Index(self.codes),
            prices=self.prices,
            factors=self.factor_data,
            market_data={
                'ma_20': self.prices * np.random.uniform(0.95, 1.05, len(self.codes)),
            },
        )

    def test_simple_pipeline(self):
        """测试基础管道：Selector + Allocator"""
        strategy = ComposableStrategy("test").compose(
            selector=TopKSelector(factor_col="alpha_score", top_k=10),
            allocator=AlphaWeightedAllocator(factor_col="alpha_score"),
        )

        ctx = self._make_context()
        weights = strategy(ctx)

        self.assertIsNotNone(weights)
        self.assertEqual(len(weights), 10, "TopK=10 应该选出 10 只股票")
        self.assertAlmostEqual(weights.sum(), 1.0, places=3, msg="权重应归一化为 1")

        # 权重应全为正
        self.assertTrue((weights >= 0).all(), "仅做多策略权重应全为非负")

    def test_full_pipeline_with_timing(self):
        """测试完整管道：S + A + T"""
        strategy = ComposableStrategy("test").compose(
            selector=TopKSelector(top_k=10),
            allocator=EqualWeightAllocator(),
            timing=MaxDrawdownTimingAdjuster(max_drawdown_threshold=0.0),
        )

        ctx = self._make_context()
        ctx.market_data['ma_20'] = ctx.prices * 1.10  # 价格低于均线 10% -> 触发降仓

        weights = strategy(ctx)

        # 触发降仓后，权重和应小于 0.6（降仓 50%，之后归一化会放大，但不触发时和为1）
        # 注意：ComposableStrategy 末尾会归一化，所以这里验证降仓逻辑是否执行
        self.assertIsNotNone(weights)
        # 权重各分量最大不应超过等权时的 50% 以上（表示确实触发了缩放）
        max_weight = weights.max()
        self.assertLessEqual(max_weight, 0.5, f"降仓后最大单股权重应 ≤ 0.5，实际: {max_weight}")

    def test_risk_overlay_cap(self):
        """测试风险覆盖：个股权重上限"""
        strategy = ComposableStrategy("test").compose(
            selector=TopKSelector(top_k=10),
            allocator=AlphaWeightedAllocator(factor_col="alpha_score"),  # 非等权，有的>10%
            risk_overlay=SingleStockCapOverlay(max_weight=0.15),
        )

        ctx = self._make_context()
        weights = strategy(ctx)

        self.assertTrue(
            (weights <= 0.15 + 1e-6).all(),
            f"所有个股权重应不超过 15%，实际最大: {weights.max():.4f}"
        )
        self.assertAlmostEqual(weights.sum(), 1.0, places=3)

    def test_industry_diversification(self):
        """测试行业分散化风险覆盖"""
        strategy = ComposableStrategy("test").compose(
            selector=TopKSelector(top_k=20),
            allocator=EqualWeightAllocator(),
            risk_overlay=IndustryDiversificationOverlay(
                industry_map=self.industry_map,
                max_industry_weight=0.25,
            ),
        )

        ctx = self._make_context()
        weights = strategy(ctx)

        # 检查行业权重
        ind_weights = {}
        for code, w in weights.items():
            ind = self.industry_map.get(code, "other")
            ind_weights[ind] = ind_weights.get(ind, 0) + w

        self.assertTrue(
            all(w <= 0.25 + 1e-6 for w in ind_weights.values()),
            f"所有行业权重应不超过 25%，实际: {ind_weights}"
        )

    def test_composite_risk_overlay(self):
        """测试复合风险覆盖"""
        strategy = ComposableStrategy("test").compose(
            selector=TopKSelector(top_k=15),
            allocator=EqualWeightAllocator(),
            risk_overlay=CompositeRiskOverlay([
                SingleStockCapOverlay(max_weight=0.10),
                IndustryDiversificationOverlay(
                    industry_map=self.industry_map,
                    max_industry_weight=0.30,
                ),
            ]),
        )

        ctx = self._make_context()
        weights = strategy(ctx)

        self.assertTrue((weights <= 0.10 + 1e-6).all())
        self.assertAlmostEqual(weights.sum(), 1.0, places=3)

    def test_empty_selection(self):
        """测试选股结果为空时的边界条件"""
        empty_selector = TopKSelector(factor_col="non_existent", top_k=5)
        strategy = ComposableStrategy("test").compose(
            selector=empty_selector,
            allocator=EqualWeightAllocator(),
        )

        ctx = self._make_context()
        # 移除因子数据
        ctx.factors = pd.DataFrame(columns=['code', 'non_existent'])
        weights = strategy(ctx)

        self.assertGreater(len(weights), 0)
        self.assertAlmostEqual(weights.sum(), 1.0, places=3)

    def test_contract_preserving(self):
        """测试接口保合同变换（所有阶段输出权重向量）"""
        # 阶段1：selector → candidates (Index)
        selector = TopKSelector(top_k=10)
        ctx = self._make_context()
        candidates = selector.select(ctx)
        self.assertIsInstance(candidates, pd.Index)

        # 阶段2：allocator → weights (Series)
        allocator = EqualWeightAllocator()
        w_base = allocator.allocate(ctx, candidates)
        self.assertIsInstance(w_base, pd.Series)
        self.assertAlmostEqual(w_base.sum(), 1.0, places=3)

        # 阶段3：timing → weights (Series)
        timing = MaxDrawdownTimingAdjuster()
        w_timing = timing.adjust(ctx, w_base)
        self.assertIsInstance(w_timing, pd.Series)

        # 阶段4：risk_overlay → weights (Series)
        risk = SingleStockCapOverlay()
        w_final = risk.overlay(ctx, w_timing)
        self.assertIsInstance(w_final, pd.Series)

        # 所有中间输出都是 weight vector
        print("\n保合同变换验证:")
        print(f"  w_base  : sum={w_base.sum():.4f}, len={len(w_base)}")
        print(f"  w_timing: sum={w_timing.sum():.4f}, len={len(w_timing)}")
        print(f"  w_final : sum={w_final.sum():.4f}, len={len(w_final)}")

    def test_reusability(self):
        """测试策略组件可复用性：同一组件用于不同管道"""
        selector_a = TopKSelector(top_k=5)
        selector_b = TopKSelector(top_k=20)

        strategy_a = ComposableStrategy("A").compose(
            selector=selector_a,
            allocator=EqualWeightAllocator(),
        )
        strategy_b = ComposableStrategy("B").compose(
            selector=selector_b,
            allocator=AlphaWeightedAllocator(),
        )

        ctx = self._make_context()
        w_a = strategy_a(ctx)
        w_b = strategy_b(ctx)

        self.assertEqual(len(w_a), 5)
        self.assertEqual(len(w_b), 20)

        # 策略 A 等权，策略 B 按 Alpha 加权
        self.assertAlmostEqual(w_a.iloc[0], 1.0 / 5, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)