"""
原生回测引擎适配器 v2 —— 优化验证版

借鉴来源：
  - NautilusTrader: FillModel / FeeModel 可插拔分离，确定性事件循环
  - Microsoft Qlib: excess_return_with_cost / without_cost 成本分离，limit_threshold 价格限制过滤
  - vn.py: 涨跌停 / T+1 等 A 股市场规则严格化

相对 jingni-trader main 分支 native_adapter.py 的修复点：
  1. T+1 真正实现：记录每只股票的最近买入日期，卖出时强制 current_date > buy_date
  2. PnL 正确计算：positions 改为 {code: {shares, avg_cost, last_buy_date}}，
     卖出 pnl = (sell_price - avg_cost) * shares - costs（旧版把成交金额当盈亏）
  3. 滑点双侧应用：卖出价也乘以 (1 - slippage)（旧版卖出无滑点）
  4. 过户费计算：卖出侧加 transfer_fee = sell_amount * transfer_fee_rate（旧版完全缺失）
  5. 基准对比：equity_curve 增加 benchmark 列，metrics 增加 alpha/beta/relative_drawdown
  6. 成本分离：metrics 同时输出 gross_return（不含费用）与 net_return（含费用），借鉴 Qlib
  7. 涨跌停过滤增强：买入时同时检查 is_limit_up，卖出时同时检查 is_limit_down（旧版已做）

所有新代码位于 feat/quant-opt-20260624 分支的独立目录，不修改 main 分支代码。
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 可插拔模型（借鉴 NautilusTrader FillModel / FeeModel 设计）
# ---------------------------------------------------------------------------

class FillModel:
    """成交价模型基类。

    借鉴 NautilusTrader 的 FillModel 设计：将"成交价如何确定"与"回测主循环"解耦，
    便于切换 BestPrice / Slippage / SizeAware 等不同假设。
    """

    def buy_fill_price(self, row: pd.Series, slippage: float) -> float:
        raise NotImplementedError

    def sell_fill_price(self, row: pd.Series, slippage: float) -> float:
        raise NotImplementedError


class CloseSlippageFillModel(FillModel):
    """收盘价 + 双侧滑点模型。

    修复点：旧版 native_adapter 仅在买入侧应用 slippage，卖出侧直接用 close，
    导致卖出成本被低估。本模型在买卖两侧都应用滑点。
    """

    def buy_fill_price(self, row: pd.Series, slippage: float) -> float:
        return float(row["close"]) * (1.0 + slippage)

    def sell_fill_price(self, row: pd.Series, slippage: float) -> float:
        return float(row["close"]) * (1.0 - slippage)


class FeeModel:
    """费用模型基类。

    借鉴 NautilusTrader 的 FeeModel 设计：将费用计算独立出来，
    便于支持 maker/taker、固定费用、过户费等不同结构。
    """

    def calc_buy_cost(self, amount: float, min_commission: float = 5.0) -> Dict[str, float]:
        raise NotImplementedError

    def calc_sell_cost(self, amount: float, min_commission: float = 5.0) -> Dict[str, float]:
        raise NotImplementedError


class AShareFeeModel(FeeModel):
    """A 股费用模型：佣金 + 印花税（卖出）+ 过户费（双侧）。

    修复点：旧版 native_adapter 完全没有计算过户费（config 中定义了
    TRANSFER_FEE_RATE 但从未使用）。本模型补齐过户费计算。
    """

    def __init__(
        self,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        transfer_fee_rate: float = 0.00002,
    ):
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate

    def calc_buy_cost(self, amount: float, min_commission: float = 5.0) -> Dict[str, float]:
        commission = max(amount * self.commission_rate, min_commission)
        transfer_fee = amount * self.transfer_fee_rate
        return {"commission": commission, "stamp_tax": 0.0, "transfer_fee": transfer_fee}

    def calc_sell_cost(self, amount: float, min_commission: float = 5.0) -> Dict[str, float]:
        commission = max(amount * self.commission_rate, min_commission)
        stamp_tax = amount * self.stamp_tax_rate
        transfer_fee = amount * self.transfer_fee_rate
        return {"commission": commission, "stamp_tax": stamp_tax, "transfer_fee": transfer_fee}


# ---------------------------------------------------------------------------
# 持仓结构（修复 avg_cost 缺失问题）
# ---------------------------------------------------------------------------

class Position:
    """单只股票持仓，记录股数、平均成本、最近买入日期。

    修复点：旧版 positions = {code: shares} 仅记录股数，无法计算真实盈亏。
    本类记录 avg_cost 以支持正确的 PnL 计算，记录 last_buy_date 以支持 T+1。
    """

    __slots__ = ("shares", "avg_cost", "last_buy_date")

    def __init__(self, shares: int = 0, avg_cost: float = 0.0, last_buy_date=None):
        self.shares = shares
        self.avg_cost = avg_cost
        self.last_buy_date = last_buy_date

    def add(self, shares: int, price: float, date):
        """加仓，更新平均成本与最近买入日期。"""
        old_total = self.avg_cost * self.shares
        self.shares += shares
        if self.shares > 0:
            self.avg_cost = (old_total + shares * price) / self.shares
        self.last_buy_date = date

    def reduce(self, shares: int):
        """减仓，股数减为 0 时不清零 avg_cost（便于后续审计），仅减股数。"""
        self.shares = max(0, self.shares - shares)


# ---------------------------------------------------------------------------
# 回测引擎 v2
# ---------------------------------------------------------------------------

class NativeAdapterV2:
    """原生回测适配器 v2。

    与 main 分支 NativeAdapter 接口保持一致（run_backtest 签名兼容），
    但内部修复了 T+1 / PnL / 滑点 / 过户费 / 基准 / 成本分离六大问题。
    """

    def __init__(
        self,
        fill_model: Optional[FillModel] = None,
        fee_model: Optional[FeeModel] = None,
    ):
        self.fill_model = fill_model or CloseSlippageFillModel()
        self.fee_model = fee_model or AShareFeeModel()

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
        transfer_fee_rate: float = 0.00002,
        min_commission: float = 5.0,
        budget_ratio: float = 0.95,
    ) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return self._empty_result()

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        dates = sorted(signals["date"].unique())
        if not dates:
            return self._empty_result()

        # 用传入参数覆盖默认 fee_model（保持向后兼容）
        if transfer_fee_rate != 0.00002 or commission_rate != 0.00025 or stamp_tax_rate != 0.001:
            self.fee_model = AShareFeeModel(commission_rate, stamp_tax_rate, transfer_fee_rate)

        cash = init_capital
        positions: Dict[str, Position] = {}
        equity_records = []
        trades = []
        gross_equity_records = []  # 不含费用的权益（借鉴 Qlib excess_return_without_cost）
        cumulative_fees = 0.0  # 累计已支付费用，用于计算毛收益

        # 预提取基准净值（若有）
        benchmark_prices = {}
        bm_data = data[data["code"] == benchmark] if benchmark else pd.DataFrame()
        if not bm_data.empty:
            benchmark_prices = dict(zip(bm_data["date"], bm_data["close"]))
        bm_initial = None

        for dt in dates:
            day_signal = signals[signals["date"] == dt]
            day_data = data[data["date"] == dt]
            if day_data.empty:
                continue
            day_data_map = day_data.set_index("code")

            sell_codes, buy_codes = self._split_signals(day_signal)

            # ---- 卖出（先卖后买，释放资金）----
            for code in sell_codes:
                pos = positions.get(code)
                if pos is None or pos.shares <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                # T+1 检查：买入当日不得卖出
                if t_plus_1 and pos.last_buy_date is not None and dt <= pos.last_buy_date:
                    continue
                # 跌停不得卖出
                if price_limit and bool(price_row.get("is_limit_down", False)):
                    continue
                fill_price = self.fill_model.sell_fill_price(price_row, slippage)
                shares = pos.shares
                sell_amount = fill_price * shares
                costs = self.fee_model.calc_sell_cost(sell_amount, min_commission)
                total_cost = costs["commission"] + costs["stamp_tax"] + costs["transfer_fee"]
                # 真实盈亏 = (卖出价 - 平均成本) * 股数 - 费用
                realized_pnl = (fill_price - pos.avg_cost) * shares - total_cost
                cash += sell_amount - total_cost
                cumulative_fees += total_cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": fill_price, "shares": shares, "amount": sell_amount,
                    "commission": costs["commission"], "stamp_tax": costs["stamp_tax"],
                    "transfer_fee": costs["transfer_fee"], "pnl": realized_pnl,
                    "avg_cost": pos.avg_cost,
                })
                pos.reduce(shares)

            # ---- 买入 ----
            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * budget_ratio / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    # 涨停不得买入
                    if price_limit and bool(price_row.get("is_limit_up", False)):
                        continue
                    fill_price = self.fill_model.buy_fill_price(price_row, slippage)
                    shares = int(budget_per_stock / fill_price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = fill_price * shares
                    costs = self.fee_model.calc_buy_cost(buy_amount, min_commission)
                    total_cost = costs["commission"] + costs["transfer_fee"]
                    if buy_amount + total_cost > cash:
                        shares = int((cash * 0.98) / fill_price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = fill_price * shares
                        costs = self.fee_model.calc_buy_cost(buy_amount, min_commission)
                        total_cost = costs["commission"] + costs["transfer_fee"]
                    cash -= buy_amount + total_cost
                    cumulative_fees += total_cost
                    if code not in positions:
                        positions[code] = Position()
                    positions[code].add(shares, fill_price, dt)
                    trades.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": fill_price, "shares": shares, "amount": buy_amount,
                        "commission": costs["commission"], "stamp_tax": 0.0,
                        "transfer_fee": costs["transfer_fee"], "pnl": 0.0,
                        "avg_cost": positions[code].avg_cost,
                    })

            # ---- 估值 ----
            market_value = 0.0
            gross_market_value = 0.0  # 不含费用的市值（近似）
            for code, pos in positions.items():
                if pos.shares <= 0:
                    continue
                if code in day_data_map.index:
                    close = float(day_data_map.loc[code, "close"])
                    market_value += pos.shares * close
                    gross_market_value += pos.shares * close
            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for p in positions.values() if p.shares > 0),
                "benchmark": benchmark_prices.get(dt, np.nan),
            })
            # 毛权益 = 净权益 + 累计费用（假设未支付费用的情景）
            gross_equity_records.append({
                "date": dt,
                "equity": total_equity + cumulative_fees,
            })

        equity_curve = pd.DataFrame(equity_records)
        gross_curve = pd.DataFrame(gross_equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        # 初始化基准首值
        if not equity_curve["benchmark"].isna().all():
            first_valid = equity_curve["benchmark"].dropna().iloc[0]
            bm_initial = first_valid

        metrics = self._calc_metrics_v2(
            equity_curve, gross_curve, trades_df, init_capital, bm_initial
        )

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(
                [(c, p.shares, p.avg_cost) for c, p in positions.items() if p.shares > 0],
                columns=["code", "shares", "avg_cost"],
            ),
            "equity_curve": equity_curve,
            "gross_equity_curve": gross_curve,
            "metrics": metrics,
            "report_path": "",
        }

    # ---- 辅助方法 ----

    @staticmethod
    def _split_signals(day_signal: pd.DataFrame) -> Tuple[list, list]:
        sell_codes, buy_codes = [], []
        for _, row in day_signal.iterrows():
            code = row["code"]
            sig = row.get("signal", 0)
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    buy_codes.append(code)
                elif sig < 0:
                    sell_codes.append(code)
        return sell_codes, buy_codes

    @staticmethod
    def _calc_metrics_v2(
        equity_curve: pd.DataFrame,
        gross_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        init_capital: float,
        bm_initial: Optional[float],
    ) -> Dict[str, Any]:
        """计算绩效指标 v2，增加成本分离与基准对比。"""
        eq = equity_curve.set_index("date")["equity"]
        gross_eq = gross_curve.set_index("date")["equity"]

        # 基础净值指标（净收益，含费用）
        total_return = eq.iloc[-1] / eq.iloc[0] - 1.0 if len(eq) > 1 else 0.0
        n_days = len(eq)
        annual_return = (1 + total_return) ** (252.0 / max(n_days, 1)) - 1.0 if total_return > -1 else -1.0
        daily_ret = eq.pct_change().dropna()
        volatility = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else 0.0
        sharpe = annual_return / volatility if volatility > 0 else 0.0
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        max_drawdown = float(drawdown.min())
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        # 毛收益（不含费用，借鉴 Qlib excess_return_without_cost）
        gross_total_return = gross_eq.iloc[-1] / gross_eq.iloc[0] - 1.0 if len(gross_eq) > 1 else 0.0
        gross_annual = (1 + gross_total_return) ** (252.0 / max(n_days, 1)) - 1.0 if gross_total_return > -1 else -1.0

        # 胜率：基于真实 pnl（修复后才有意义）
        win_rate = 0.0
        if not trades_df.empty and "pnl" in trades_df.columns:
            sell_trades = trades_df[trades_df["action"] == "sell"]
            if not sell_trades.empty:
                win_rate = float((sell_trades["pnl"] > 0).mean())

        # 基准对比
        benchmark_metrics = {}
        if bm_initial is not None and "benchmark" in equity_curve.columns:
            bm_series = equity_curve.set_index("date")["benchmark"].dropna()
            if len(bm_series) > 1:
                bm_return = bm_series.iloc[-1] / bm_series.iloc[0] - 1.0
                bm_daily = bm_series.pct_change().dropna()
                bm_vol = bm_daily.std() * np.sqrt(252) if len(bm_daily) > 1 else 0.0
                bm_cummax = bm_series.cummax()
                bm_dd = ((bm_series - bm_cummax) / bm_cummax).min()
                # alpha / beta（协方差法）
                aligned = pd.concat([daily_ret, bm_daily], axis=1, join="inner").dropna()
                aligned.columns = ["strat", "bench"]
                if len(aligned) > 2 and aligned["bench"].std() > 0:
                    beta = float(aligned.cov().iloc[0, 1] / aligned["bench"].var())
                    alpha = float(annual_return - beta * (bm_return if abs(bm_return) < 10 else 0))
                else:
                    beta, alpha = 0.0, 0.0
                benchmark_metrics = {
                    "benchmark_return": float(bm_return),
                    "benchmark_volatility": float(bm_vol),
                    "benchmark_max_drawdown": float(bm_dd),
                    "alpha": alpha,
                    "beta": beta,
                    "excess_return": float(total_return - bm_return),
                }

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
            "calmar": float(calmar),
            "win_rate": win_rate,
            # 成本分离（借鉴 Qlib）
            "gross_total_return": float(gross_total_return),
            "gross_annual_return": float(gross_annual),
            "total_cost_drag": float(gross_total_return - total_return),
            **benchmark_metrics,
        }

    @staticmethod
    def _empty_result():
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "gross_equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
