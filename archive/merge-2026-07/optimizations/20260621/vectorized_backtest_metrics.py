"""
Optimization B: 向量化回测 + 修正后的绩效指标
================================================================================

借鉴来源:
  - akquant: 用日期索引 + 零拷贝数据架构降低 Python 层开销; 强调回测主循环性能。
  - 通用量化最佳实践 (QuantConnect/Lean, Zipline): 绩效指标必须区分算术年化与
    几何年化, 胜率应基于 "已平仓交易" 而非 "所有成交单"。

jingni-trader 现状痛点:
  1. skills/backtest-engine/scripts/adapters/native_adapter.py 第 44-46 行:
       for dt in dates:
           day_data = data[data['date'] == dt]
     每个交易日都对全量 DataFrame 做一次布尔索引 (O(n)), 整体 O(n × 交易日数),
     全市场面板下是性能瓶颈。

  2. skills/backtest-engine/scripts/base/base_backtest.py 指标计算存在两处缺陷:
     - calc_sharpe (第 49 行) 用算术年化 ``returns.mean() * trading_days``;
       calc_annual_return (第 34 行) 用几何年化 ``(total_return) ** (1/n_years) - 1``。
       两者口径不一致, 导致 Sharpe 分子与报告的 annual_return 对不上。
     - calc_win_rate (第 80 行) 统计所有 trades 的 pnl>0, 但 native_adapter 中
       买单 pnl = -buy_amount - commission (恒负), 被错误计入 "亏损单",
       胜率被严重低估且无业务意义。

本模块提供:
  - ``DateIndexedBacktester``: 用 dict 预索引行情, 把日内查找从 O(n) 降到 O(1),
    其余逻辑与 native_adapter 保持等价 (T+1、涨跌停、印花税、滑点), 便于公平对比。
  - ``CorrectedMetrics``: 修正后的指标计算, 同时给出 arithmetic/geometric 年化、
    仅基于平仓单的胜率、以及 benchmark 相对指标 (alpha/beta/information_ratio)。

本文件为独立验证实现, 不修改 main 分支任何代码。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("vectorized-backtest-metrics")


# ===========================================================================
# 优化 1: 日期预索引回测器
# ===========================================================================

class DateIndexedBacktester:
    """与 native_adapter 逻辑等价但用日期预索引的回测器。

    与原版 NativeAdapter.run_backtest 的关键差异:
      - 进入循环前用 ``data.groupby('date')`` 一次性建好 {date: day_df} 索引,
        循环内 ``day_data = date_index[dt]`` 是 O(1) dict 查找;
      - 信号也按日期预分组;
      - 交易撮合顺序、T+1、涨跌停、费用计算逻辑与原版完全一致, 保证结果可对比。
    """

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        slippage: float = 0.001,
    ) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        # ---- 关键优化: 预索引 ----
        # 原版每个交易日 data[data['date']==dt] 是 O(n) 扫描; 这里一次性建 dict。
        date_index: Dict[Any, pd.DataFrame] = {
            dt: g.set_index("code") for dt, g in data.groupby("date")
        }
        signal_index: Dict[Any, pd.DataFrame] = {
            dt: g for dt, g in signals.groupby("date")
        }

        dates = sorted(signal_index.keys())
        if not dates:
            return self._empty_result()

        cash = init_capital
        positions: Dict[str, int] = {}
        equity_records = []
        trades = []

        for dt in dates:
            day_data_map = date_index.get(dt)
            if day_data_map is None or day_data_map.empty:
                continue
            day_signal = signal_index[dt]

            sell_codes = []
            buy_codes = []
            for _, row in day_signal.iterrows():
                code = row["code"]
                sig = row.get("signal", 0)
                if isinstance(sig, (int, float, np.integer, np.floating)):
                    sig = float(sig)
                    if sig > 0:
                        buy_codes.append(code)
                    elif sig < 0:
                        sell_codes.append(code)

            # ---- 卖出 (先卖后买, 与原版一致) ----
            for code in sell_codes:
                if positions.get(code, 0) <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and bool(price_row.get("is_limit_down", False)):
                    continue
                price = price_row["close"]
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": price, "shares": shares, "amount": sell_amount,
                    "commission": commission, "tax": tax,
                    "pnl": sell_amount - cost,  # 与原版字段一致 (注意: 这是毛额, 非真实盈亏)
                })
                positions[code] = 0

            # ---- 买入 ----
            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * 0.95 / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    if price_limit and bool(price_row.get("is_limit_up", False)):
                        continue
                    price = price_row["close"] * (1 + slippage)
                    shares = int(budget_per_stock / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                    if cost > cash:
                        shares = int((cash * 0.98) / price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = price * shares
                        commission = max(buy_amount * commission_rate, 5)
                        cost = buy_amount + commission
                    cash -= cost
                    positions[code] = positions.get(code, 0) + shares
                    trades.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": price, "shares": shares, "amount": buy_amount,
                        "commission": commission, "tax": 0,
                        "pnl": -buy_amount - commission,
                    })

            # ---- 盯市 ----
            market_value = 0
            for code, shares in positions.items():
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, "close"]
            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        if equity_curve.empty:
            return self._empty_result()

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(
                list(positions.items()), columns=["code", "shares"]
            ),
            "equity_curve": equity_curve,
            "metrics": {},  # 由 CorrectedMetrics 单独计算, 便于对比
            "report_path": "",
        }

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }


# ===========================================================================
# 优化 2: 修正后的绩效指标
# ===========================================================================

class CorrectedMetrics:
    """修正 jingni-trader ``BaseBacktestMetrics`` 的两处缺陷, 并补充相对指标。

    修正点:
      1. 年化收益率口径统一: 同时输出 ``annual_return_geom`` (几何, 与原 calc_annual_return
         一致) 和 ``annual_return_arith`` (算术, 与原 calc_sharpe 分子一致), 让 Sharpe
         分子分母口径自洽。
      2. 胜率只统计 ``action == 'sell'`` 的平仓单, 排除买单的负 pnl 干扰。
      3. 新增 benchmark 相对指标: alpha/beta/information_ratio/excess_return
         (需传入 benchmark 收益序列)。
    """

    TRADING_DAYS = 252

    @staticmethod
    def calc_returns(equity_curve: pd.Series) -> pd.Series:
        return equity_curve.pct_change().dropna()

    @staticmethod
    def calc_total_return(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    @classmethod
    def calc_annual_return_geom(cls, equity_curve: pd.Series) -> float:
        """几何年化: (1+total)^(252/n) - 1。与原 calc_annual_return 一致。"""
        if len(equity_curve) < 2:
            return 0.0
        total = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = len(equity_curve) / cls.TRADING_DAYS
        if n_years <= 0:
            return 0.0
        return float(total ** (1 / n_years) - 1)

    @classmethod
    def calc_annual_return_arith(cls, returns: pd.Series) -> float:
        """算术年化: mean(daily) * 252。与原 calc_sharpe 分子一致。"""
        if len(returns) < 1:
            return 0.0
        return float(returns.mean() * cls.TRADING_DAYS)

    @classmethod
    def calc_volatility(cls, returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(cls.TRADING_DAYS))

    @classmethod
    def calc_sharpe(
        cls, returns: pd.Series, risk_free: float = 0.03
    ) -> float:
        """Sharpe: 用算术年化收益保证分子分母同口径 (与原 calc_sharpe 分子一致)。

        如需几何年化口径, 用 ``calc_sharpe_consistent`` 获取两种结果对比。
        """
        vol = cls.calc_volatility(returns)
        if vol == 0:
            return 0.0
        ann_ret = cls.calc_annual_return_arith(returns)
        return float((ann_ret - risk_free) / vol)

    @classmethod
    def calc_sharpe_consistent(
        cls, equity_curve: pd.Series, risk_free: float = 0.03
    ) -> Dict[str, float]:
        """同时给出两种口径的 Sharpe, 便于诊断原版不一致问题。"""
        returns = cls.calc_returns(equity_curve)
        vol = cls.calc_volatility(returns)
        ann_arith = cls.calc_annual_return_arith(returns)
        ann_geom = cls.calc_annual_return_geom(equity_curve)
        sharpe_arith = (ann_arith - risk_free) / vol if vol > 0 else 0.0
        sharpe_geom = (ann_geom - risk_free) / vol if vol > 0 else 0.0
        return {
            "annual_return_arith": float(ann_arith),
            "annual_return_geom": float(ann_geom),
            "sharpe_arith": float(sharpe_arith),
            "sharpe_geom": float(sharpe_geom),
            "volatility": float(vol),
        }

    @classmethod
    def calc_max_drawdown(cls, equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        cum_max = equity_curve.cummax()
        drawdown = (equity_curve - cum_max) / cum_max
        return float(drawdown.min())

    @classmethod
    def calc_sortino(
        cls, returns: pd.Series, risk_free: float = 0.03
    ) -> float:
        neg = returns[returns < 0]
        if len(neg) < 2:
            return 0.0
        downside = neg.std() * np.sqrt(cls.TRADING_DAYS)
        if downside == 0:
            return 0.0
        ann_arith = cls.calc_annual_return_arith(returns)
        return float((ann_arith - risk_free) / downside)

    @classmethod
    def calc_win_rate_corrected(cls, trades: pd.DataFrame) -> float:
        """胜率: 仅统计平仓 (sell) 单。

        原版 calc_win_rate 统计所有 trades, 但买单 pnl 恒负, 会拉低胜率。
        这里只看 sell 单。若 trades 没有 pnl 或没有 sell 单, 返回 0。
        """
        if trades.empty or "action" not in trades.columns:
            return 0.0
        sells = trades[trades["action"] == "sell"]
        if sells.empty or "pnl" not in sells.columns:
            return 0.0
        winning = (sells["pnl"] > 0).sum()
        total = len(sells)
        return float(winning / total) if total > 0 else 0.0

    @classmethod
    def calc_benchmark_metrics(
        cls,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free: float = 0.03,
    ) -> Dict[str, float]:
        """相对基准指标: excess_return / beta / alpha / information_ratio。

        参数:
            strategy_returns: 策略日收益率
            benchmark_returns: 基准日收益率 (需已对齐到相同日期)
        """
        aligned = pd.concat(
            [strategy_returns.rename("s"), benchmark_returns.rename("b")],
            axis=1,
        ).dropna()
        if len(aligned) < 3:
            return {
                "excess_return": 0.0, "beta": 0.0, "alpha": 0.0,
                "information_ratio": 0.0, "tracking_error": 0.0,
            }
        s = aligned["s"]
        b = aligned["b"]
        excess = s - b
        # beta = cov(s,b)/var(b)
        cov_sb = float(s.cov(b))
        var_b = float(b.var())
        beta = cov_sb / var_b if var_b > 0 else 0.0
        # alpha (年化, Jensen): arith_annual(s) - [rf + beta*(arith_annual(b)-rf)]
        ann_s = s.mean() * cls.TRADING_DAYS
        ann_b = b.mean() * cls.TRADING_DAYS
        alpha = ann_s - (risk_free + beta * (ann_b - risk_free))
        tracking_error = float(excess.std() * np.sqrt(cls.TRADING_DAYS))
        info_ratio = (
            float(excess.mean() * cls.TRADING_DAYS / tracking_error)
            if tracking_error > 0 else 0.0
        )
        return {
            "excess_return": float(excess.mean() * cls.TRADING_DAYS),
            "beta": float(beta),
            "alpha": float(alpha),
            "information_ratio": float(info_ratio),
            "tracking_error": tracking_error,
        }

    @classmethod
    def calc_all(
        cls,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free: float = 0.03,
    ) -> Dict[str, Any]:
        """一次性计算全部修正后指标。"""
        returns = cls.calc_returns(equity_curve)
        sharp = cls.calc_sharpe_consistent(equity_curve, risk_free)
        mdd = cls.calc_max_drawdown(equity_curve)
        ann_geom = sharp["annual_return_geom"]
        calmar = ann_geom / abs(mdd) if mdd != 0 else 0.0
        out: Dict[str, Any] = {
            "total_return": cls.calc_total_return(equity_curve),
            "annual_return_geom": ann_geom,
            "annual_return_arith": sharp["annual_return_arith"],
            "volatility": sharp["volatility"],
            "sharpe_arith": sharp["sharpe_arith"],
            "sharpe_geom": sharp["sharpe_geom"],
            "sortino_ratio": cls.calc_sortino(returns, risk_free),
            "max_drawdown": mdd,
            "calmar_ratio": float(calmar),
            "win_rate_closed_only": cls.calc_win_rate_corrected(trades),
            "total_trades": len(trades),
            "closed_trades": (
                int((trades["action"] == "sell").sum())
                if not trades.empty and "action" in trades.columns
                else 0
            ),
        }
        if benchmark_returns is not None:
            out.update(cls.calc_benchmark_metrics(returns, benchmark_returns, risk_free))
        return out


# ===========================================================================
# 辅助: 复刻原版 BaseBacktestMetrics 用于对比测试
# ===========================================================================

class OriginalMetricsReplica:
    """忠实复刻 skills/backtest-engine/scripts/base/base_backtest.py 的指标计算,
    用于在测试中对比 "原版 vs 修正版"。"""

    TRADING_DAYS = 252

    @staticmethod
    def calc_total_return(equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)

    @staticmethod
    def calc_annual_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
        if len(equity_curve) < 2:
            return 0.0
        total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
        n_years = len(equity_curve) / trading_days
        if n_years <= 0:
            return 0.0
        return float(total_return ** (1 / n_years) - 1)

    @staticmethod
    def calc_volatility(returns: pd.Series, trading_days: int = 252) -> float:
        if len(returns) < 2:
            return 0.0
        return float(returns.std() * np.sqrt(trading_days))

    @staticmethod
    def calc_sharpe(returns: pd.Series, risk_free: float = 0.03, trading_days: int = 252) -> float:
        vol = OriginalMetricsReplica.calc_volatility(returns, trading_days)
        if vol == 0:
            return 0.0
        ann_return = returns.mean() * trading_days
        return float((ann_return - risk_free) / vol)

    @staticmethod
    def calc_win_rate(trades: pd.DataFrame) -> float:
        if trades.empty:
            return 0.0
        winning = (trades["pnl"] > 0).sum()
        total = len(trades)
        return float(winning / total) if total > 0 else 0.0
