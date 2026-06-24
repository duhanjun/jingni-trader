"""
风险引擎修复 (feat/quant-opt-20260624)

借鉴来源:
  - RQAlpha: FrontendValidator 盘前风控(已集成到 optimized_backtest.py)
  - PyPortfolioOpt HRPOpt 正确用法

针对 jingni-trader portfolio-risk-engine 的修复点:
  1. _optimize_hrp 原实现 `returns = pd.DataFrame()` 空表 → HRPOpt 必然失败/无意义
     修复: 传入真实历史收益率矩阵
  2. _optimize_cvar 原实现返回等权占位 → 实现基于历史收益的真实 CVaR 优化
  3. 风控未接入回测 → 通过 FrontendValidator(在 optimized_backtest.py) 接入
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from pypfopt import HRPOpt, EfficientFrontier, risk_models, expected_returns
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False


class RiskFixes:
    """组合优化修复"""

    def __init__(self):
        self.has_pypfopt = HAS_PYPFOPT

    def optimize_hrp_fixed(
        self, returns: pd.DataFrame
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        修复版分层风险平价 (HRP)

        原实现 bug:
            returns = pd.DataFrame()   # 空表!
            hrp = HRPOpt(returns)      # 无数据,优化无意义
            weights = hrp.optimize()

        修复: 传入真实历史收益率矩阵,HRPOpt 基于距离矩阵和层次聚类分配权重
        """
        if not self.has_pypfopt:
            n = returns.shape[1]
            w = pd.Series(1.0 / n, index=returns.columns)
            return w, {"method": "equal_weight", "note": "PyPortfolioOpt 未安装"}

        if returns.empty or returns.shape[1] < 2:
            n = max(returns.shape[1], 1)
            w = pd.Series(1.0 / n, index=returns.columns)
            return w, {"method": "equal_weight", "note": "收益率数据不足"}

        # 正确用法: HRPOpt 接收历史收益率 DataFrame (index=date, columns=asset)
        hrp = HRPOpt(returns)
        weights = hrp.optimize()
        cleaned = hrp.clean_weights()

        # 计算组合绩效
        try:
            perf = hrp.portfolio_performance()
            metrics = {
                "method": "hrp",
                "expected_return": float(perf[0]),
                "volatility": float(perf[1]),
                "sharpe_ratio": float(perf[2]),
            }
        except Exception:
            metrics = {"method": "hrp"}

        return pd.Series(cleaned), metrics

    def optimize_cvar_fixed(
        self,
        returns: pd.DataFrame,
        expected_rets: Optional[pd.Series] = None,
        confidence: float = 0.95,
        max_weight: float = 0.10,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        修复版 CVaR 优化

        原实现: 直接返回等权占位,注释"CVaR优化需要历史收益数据,当前返回等权组合"
        修复: 基于 EfficientSemivariance/EfficientCVaR(pypfopt) 实现真实 CVaR 最小化
        """
        if not self.has_pypfopt:
            n = returns.shape[1]
            return pd.Series(1.0 / n, index=returns.columns), {"method": "equal_weight"}

        try:
            from pypfopt import EfficientCVaR
        except ImportError:
            n = returns.shape[1]
            return pd.Series(1.0 / n, index=returns.columns), {"method": "equal_weight"}

        if expected_rets is None:
            expected_rets = expected_returns.mean_historical_return(returns, frequency=252)

        try:
            ec = EfficientCVaR(expected_rets, returns, weight_bounds=(0, max_weight))
            ec.min_cvar(s_market_neutral=False)
            cleaned = ec.clean_weights()
            weights = pd.Series(cleaned)
            return weights, {
                "method": "cvar",
                "confidence": confidence,
                "note": "基于历史收益的真实 CVaR 优化",
            }
        except Exception as e:
            n = returns.shape[1]
            return pd.Series(1.0 / n, index=returns.columns), {
                "method": "equal_weight",
                "note": f"CVaR 优化失败,回退等权: {e}",
            }

    def calc_var_cvar(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> Dict[str, float]:
        """历史模拟法 VaR/CVaR(与原实现一致,验证用)"""
        var = float(np.percentile(returns, (1 - confidence) * 100))
        cvar = float(returns[returns <= var].mean()) if (returns <= var).any() else var
        return {"VaR": var, "CVaR": cvar, "confidence": confidence}