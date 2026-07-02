"""
动态因子权重融合（Dynamic Factor Fusion）
===========================================

借鉴来源
--------
- AlphaForge 论文 (arXiv:2406.18394) 的两阶段因子融合框架
  * 生成-预测网络挖掘新因子
  * 组合模型根据因子近期表现动态调整权重
- qlib 默认的 IC 加权方法
- jingni-trader 原 factor-engine.factor_fusion（静态 IC_IR 加权）

设计目标
--------
解决 jingni-trader 当前 factor_fusion 的痛点：
1. 当前权重是"全样本 IC_IR 静态加权"（factor-engine/engine.py:345-364）
2. 没有考虑因子的"时序衰减"和"近期有效性"
3. 没有处理因子池中"死因子"（长期失效但仍占权重）

核心创新
--------
引入三种动态权重机制（可组合使用）：
1. EMA IC 加权：IC 越近期权重越大
2. Adaptive IC 加权：用自适应 IC（AlphaForge 思路）
   * 对每个因子维护 "fitness score" 序列（rolling IC + decay）
   * 在每一天只选择 top-K 高 fitness 的因子参与融合
3. Regime-aware 加权（可选扩展）：根据市场状态切换权重

接口
----
>>> from quant_opt_20260617.dynamic_factor_fusion import (
...     DynamicFactorFusion, FusionConfig, FusionMethod
... )
>>> fuser = DynamicFactorFusion(config)
>>> result_df = fuser.fuse(factor_df, ic_results)
>>> result_df[['code', 'date', 'alpha_score']].to_parquet(...)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class FusionMethod(str, Enum):
    """因子融合方法"""
    STATIC_IC_WEIGHTED = "static_ic_weighted"     # jingni-trader 现有方法（baseline）
    EMA_IC_WEIGHTED = "ema_ic_weighted"           # IC 时序衰减加权
    ADAPTIVE_TOPK = "adaptive_topk"               # AlphaForge 风格：top-K + 动态权重
    EQUAL_WEIGHT = "equal_weight"                 # 等权（基线）


@dataclass
class FusionConfig:
    """融合配置"""
    method: FusionMethod = FusionMethod.EMA_IC_WEIGHTED
    ema_halflife_days: int = 60           # IC 的指数衰减半衰期
    lookback_days: int = 252              # 滚动 IC 窗口（与 ema_halflife 配合使用）
    top_k: int = 10                       # adaptive_topk 保留的因子数
    min_weight_floor: float = 1e-3        # 权重下限（防 0 权）
    ic_floor: float = 0.01                # 因子 IC 绝对值下限（低于此视为死因子）
    forward_col: str = "ret_forward_5d"   # 评估用的前向收益列
    weight_smooth: bool = True            # 是否对权重做平滑（避免跳变）
    smooth_eps: float = 0.05              # 平滑系数（越大越接近静态）


# ============================================================
# 核心实现
# ============================================================

class DynamicFactorFusion:
    """
    动态因子权重融合

    输入
    ----
    factor_df: 包含多因子列的 DataFrame（code, date, [factor_1, factor_2, ...]）
    ic_results: IC 分析结果（参考 factor-engine.engine.py 的 ic_analysis 返回格式）
                {
                    "ret_forward_5d": [
                        {"factor": "f1", "ic_mean": 0.05, "ic_ir": 0.8, ...},
                        ...
                    ]
                }
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        if isinstance(self.config.method, str):
            self.config.method = FusionMethod(self.config.method)

    # --------------------------------------------------------
    # 工具方法
    # --------------------------------------------------------

    @staticmethod
    def _rank_by_date(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """对每个因子按 date 做截面百分位排名"""
        out = df[['code', 'date']].copy()
        for f in factor_cols:
            if f not in df.columns:
                continue
            out[f"{f}_rank"] = df.groupby('date')[f].transform(
                lambda x: x.rank(pct=True, na_option='keep')
            )
        return out

    @staticmethod
    def _calc_daily_ic(
        factor_df: pd.DataFrame,
        factor_col: str,
        forward_ret: pd.Series,
    ) -> pd.Series:
        """计算单因子的逐日 IC（Spearman）"""
        # 统一用 (code, date) 合并，确保行对齐（处理 dropna 后的非连续索引）
        merged = factor_df[['code', 'date', factor_col]].copy().reset_index(drop=True)
        fwd_df = pd.DataFrame({
            'code': factor_df['code'].values,
            'date': factor_df['date'].values,
            '_fwd': np.asarray(forward_ret).reshape(-1),
        })
        merged = merged.merge(fwd_df, on=['code', 'date'], how='left')
        merged = merged.dropna(subset=[factor_col, '_fwd'])

        daily_ic = {}
        for dt, group in merged.groupby('date'):
            if len(group) < 5:
                continue
            ic, _ = spearmanr(group[factor_col], group['_fwd'], nan_policy='omit')
            if not np.isnan(ic):
                daily_ic[pd.Timestamp(dt)] = float(ic)
        return pd.Series(daily_ic).sort_index()

    # --------------------------------------------------------
    # 静态 IC 加权（baseline，对照 jingni-trader 原实现）
    # --------------------------------------------------------

    def _static_ic_weights(
        self,
        factor_names: List[str],
        ic_results: Dict,
    ) -> Dict[str, float]:
        """原 jingni-trader 风格：全样本 IC_IR 静态加权"""
        weights = {f: 0.0 for f in factor_names}
        ic_list = ic_results.get(self.config.forward_col, [])
        ic_map = {item['factor']: item.get('ic_ir', 0) for item in ic_list}
        total = 0.0
        for f in factor_names:
            w = abs(ic_map.get(f, 0))
            weights[f] = w
            total += w
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            n = max(1, len(factor_names))
            weights = {k: 1.0 / n for k in weights}
        return weights

    # --------------------------------------------------------
    # EMA IC 加权
    # --------------------------------------------------------

    def _ema_ic_weights(
        self,
        factor_df: pd.DataFrame,
        factor_names: List[str],
        forward_returns: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, float], Dict[str, pd.Series]]:
        """
        用 EMA IC 计算每个因子的权重
        返回 (weights, ema_ic_per_factor)
        """
        ema_ic_per_factor: Dict[str, pd.Series] = {}

        if forward_returns is None or self.config.forward_col not in forward_returns.columns:
            # 没有 forward_returns 时降级为静态
            return self._static_ic_weights(factor_names, {}), ema_ic_per_factor

        # 准备 forward_ret Series，与 factor_df 严格对齐
        fwd_aligned = self._align_forward_returns(factor_df, forward_returns)

        halflife = self.config.ema_halflife_days
        lookback = self.config.lookback_days

        for f in factor_names:
            if f not in factor_df.columns:
                ema_ic_per_factor[f] = pd.Series(dtype=float)
                continue
            daily_ic = self._calc_daily_ic(factor_df, f, fwd_aligned)
            if len(daily_ic) < 5:
                ema_ic_per_factor[f] = pd.Series(dtype=float)
                continue
            # 截取最近 lookback 天
            daily_ic = daily_ic.iloc[-lookback:]
            # EMA 平滑
            ema_ic = daily_ic.ewm(halflife=halflife, adjust=False).mean()
            ema_ic_per_factor[f] = ema_ic

        # 每个因子用最新一天的 EMA IC 作为权重
        weights = {f: 0.0 for f in factor_names}
        for f, ema_series in ema_ic_per_factor.items():
            if ema_series.empty:
                continue
            # 取最后一个值；如果 NaN，置 0
            v = float(ema_series.iloc[-1])
            if np.isnan(v):
                v = 0.0
            # 死亡因子过滤
            if abs(v) < self.config.ic_floor:
                v = 0.0
            weights[f] = abs(v)  # 权重用绝对值（正负由因子方向决定）

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            n = max(1, sum(1 for v in weights.values() if v > 0) or len(factor_names))
            weights = {k: (1.0 / n if v > 0 else 0.0) for k, v in weights.items()}

        # 权重下限
        if self.config.min_weight_floor > 0:
            n_active = sum(1 for v in weights.values() if v > 0)
            if n_active > 0:
                for k in weights:
                    if weights[k] > 0 and weights[k] < self.config.min_weight_floor:
                        weights[k] = self.config.min_weight_floor
                # 重新归一化
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}

        # 平滑（混入均匀权重）
        if self.config.weight_smooth:
            eps = self.config.smooth_eps
            n = max(1, len(factor_names))
            uniform = {k: 1.0 / n for k in factor_names}
            weights = {k: (1 - eps) * weights.get(k, 0) + eps * uniform[k] for k in factor_names}
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return weights, ema_ic_per_factor

    def _align_forward_returns(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.Series:
        """把 forward_returns 与 factor_df 按 (code, date) 对齐，返回一个 RangeIndex Series"""
        fwd_sub = forward_returns[['code', 'date', self.config.forward_col]].copy()
        merged = factor_df[['code', 'date']].merge(
            fwd_sub, on=['code', 'date'], how='left'
        )
        return pd.Series(merged[self.config.forward_col].values, index=merged.index)

    # --------------------------------------------------------
    # Adaptive TopK (AlphaForge 风格)
    # --------------------------------------------------------

    def _adaptive_topk_weights(
        self,
        factor_df: pd.DataFrame,
        factor_names: List[str],
        forward_returns: Optional[pd.DataFrame],
    ) -> Tuple[Dict[str, float], Dict[str, pd.Series]]:
        """
        AlphaForge 风格：每个交易日选 fitness top-K
        fitness ≈ abs(rolling IC) + decay
        返回每个交易日的权重（不是单一权重）
        """
        if forward_returns is None or self.config.forward_col not in forward_returns.columns:
            return self._static_ic_weights(factor_names, {}), {}

        fwd_aligned = self._align_forward_returns(factor_df, forward_returns)
        halflife = self.config.ema_halflife_days

        # 计算所有因子的 EMA IC（按时序）
        fitness = {f: pd.Series(dtype=float) for f in factor_names}

        for f in factor_names:
            if f not in factor_df.columns:
                continue
            daily_ic = self._calc_daily_ic(factor_df, f, fwd_aligned)
            if len(daily_ic) >= 5:
                fitness[f] = daily_ic.ewm(halflife=halflife, adjust=False).mean().abs()

        if not any(len(v) > 0 for v in fitness.values()):
            return self._static_ic_weights(factor_names, {}), fitness

        # 把 fitness 对齐到统一的时间索引
        fitness_df = pd.DataFrame(fitness)
        # 对每个日期：取 top-K 因子（按 fitness），等权
        topk = min(self.config.top_k, len(factor_names))
        weights_per_date: Dict[str, Dict[pd.Timestamp, float]] = {
            f: {} for f in factor_names
        }

        for dt in fitness_df.index:
            row = fitness_df.loc[dt].dropna()
            if row.empty:
                continue
            # 过滤低于 ic_floor 的因子
            row = row[row >= self.config.ic_floor]
            if row.empty:
                continue
            top_factors = row.nlargest(topk).index.tolist()
            w = 1.0 / len(top_factors)
            for f in top_factors:
                weights_per_date[f][dt] = w

        # 对每个因子取最后一天的权重（用于与 baseline 公平比较）
        # 但实际上"逐日"权重才是 AlphaForge 的精髓
        final_weights: Dict[str, float] = {}
        for f in factor_names:
            if f in weights_per_date and weights_per_date[f]:
                # 用最近一个交易日的权重作为"代表"
                last_date = max(weights_per_date[f].keys())
                final_weights[f] = weights_per_date[f][last_date]
            else:
                final_weights[f] = 0.0

        # 把权重 dict 转成时间序列（返回全量，用于回测）
        weight_ts: Dict[str, pd.Series] = {}
        for f in factor_names:
            if f in weights_per_date and weights_per_date[f]:
                weight_ts[f] = pd.Series(weights_per_date[f]).sort_index()

        return final_weights, weight_ts

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------

    def fuse(
        self,
        factor_df: pd.DataFrame,
        ic_results: Optional[Dict] = None,
        forward_returns: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        执行因子融合

        返回
        ----
        DataFrame，列: code, date, alpha_score, [各因子权重序列]
        """
        factor_names = [
            c for c in factor_df.columns
            if c not in ('code', 'date', 'industry')
        ]

        if not factor_names:
            return pd.DataFrame(columns=['code', 'date', 'alpha_score'])

        # 1) 计算权重
        if self.config.method == FusionMethod.STATIC_IC_WEIGHTED:
            weights = self._static_ic_weights(
                factor_names, ic_results or {}
            )
        elif self.config.method == FusionMethod.EMA_IC_WEIGHTED:
            weights, _ = self._ema_ic_weights(
                factor_df, factor_names, forward_returns
            )
        elif self.config.method == FusionMethod.ADAPTIVE_TOPK:
            weights, _ = self._adaptive_topk_weights(
                factor_df, factor_names, forward_returns
            )
        elif self.config.method == FusionMethod.EQUAL_WEIGHT:
            n = len(factor_names)
            weights = {f: 1.0 / n for f in factor_names}
        else:
            raise ValueError(f"不支持的融合方法: {self.config.method}")

        # 2) 截面排名 + 加权求和
        rank_df = self._rank_by_date(factor_df, factor_names)
        rank_cols = [f"{f}_rank" for f in factor_names if f"{f}_rank" in rank_df.columns]

        rank_df['alpha_score'] = 0.0
        for f in factor_names:
            col = f"{f}_rank"
            if col not in rank_df.columns:
                continue
            w = weights.get(f, 0.0)
            if w > 0:
                rank_df['alpha_score'] = rank_df['alpha_score'] + w * rank_df[col]

        result = rank_df[['code', 'date', 'alpha_score']].copy()

        # 3) 附加权重元信息
        for f, w in weights.items():
            result.attrs[f'weight_{f}'] = w

        return result

    def compare_methods(
        self,
        factor_df: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        对比所有方法的 IC / Rank IC / 多空收益
        用于离线选择最优融合方法
        """
        if self.config.forward_col not in forward_returns.columns:
            forward_returns = forward_returns.copy()
            forward_returns[self.config.forward_col] = (
                forward_returns.groupby('code')['close'].transform(
                    lambda x: x.shift(-5) / x - 1
                )
            )

        results = []
        for method in FusionMethod:
            self.config.method = method
            try:
                fused = self.fuse(factor_df, forward_returns=forward_returns)
            except Exception as e:
                results.append({
                    "method": method.value,
                    "status": f"FAILED: {e}",
                    "rank_ic": np.nan,
                    "ic": np.nan,
                    "long_short_return": np.nan,
                })
                continue

            # 评估：每天把 alpha_score 分 5 组，取 top/bottom 的下期收益差
            merged = fused.merge(
                forward_returns[['code', 'date', self.config.forward_col]],
                on=['code', 'date'], how='inner'
            ).dropna()

            if merged.empty:
                results.append({
                    "method": method.value,
                    "status": "EMPTY",
                    "rank_ic": np.nan, "ic": np.nan, "long_short_return": np.nan,
                })
                continue

            # 逐日 Rank IC
            daily_ics = []
            daily_ls = []
            for dt, group in merged.groupby('date'):
                if len(group) < 10:
                    continue
                ic, _ = spearmanr(group['alpha_score'], group[self.config.forward_col], nan_policy='omit')
                if not np.isnan(ic):
                    daily_ics.append(ic)
                # long-short
                if len(group) >= 20:
                    sorted_g = group.sort_values('alpha_score')
                    bottom_q = sorted_g.head(len(group) // 5)
                    top_q = sorted_g.tail(len(group) // 5)
                    if not top_q.empty and not bottom_q.empty:
                        ls = top_q[self.config.forward_col].mean() - bottom_q[self.config.forward_col].mean()
                        daily_ls.append(ls)

            results.append({
                "method": method.value,
                "status": "OK",
                "rank_ic": float(np.mean(daily_ics)) if daily_ics else np.nan,
                "rank_ic_std": float(np.std(daily_ics)) if daily_ics else np.nan,
                "ic_ir": float(np.mean(daily_ics) / (np.std(daily_ics) + 1e-9)) if daily_ics else np.nan,
                "long_short_return": float(np.mean(daily_ls)) if daily_ls else np.nan,
                "n_days": len(daily_ics),
            })
            # 还原默认方法
            self.config.method = FusionMethod.EMA_IC_WEIGHTED

        return pd.DataFrame(results)
