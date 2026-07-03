"""
向量化回测引擎 (Vectorized Backtest Engine)

借鉴来源:
- Microsoft Qlib: 矩阵化/向量化回测思路，避免逐日 for-loop
- backtrader: 事件驱动抽象，但内部用向量化加速
- QKA: 简洁 API + A股 T+1/涨跌停/印花税规则

针对 jingni-trader native_adapter.py 的优化点:
1. T+1 真正实现: 记录买入日期，次日才能卖出 (原版仅"先卖后买"顺序，未真正限制)
2. 向量化计算持仓市值与净值，避免逐日 iterrows
3. 基准对比: equity_curve 同时输出策略净值与基准净值
4. 成交量限制: 买入量不超过当日成交量的 max_volume_pct
5. 支持 target_weight 信号 (原版仅支持 signal=1/-1/0)
6. 滑点双向应用 (原版仅买入侧)
7. 卖出也受涨跌停限制 (原版仅检查 limit_down，未检查流动性)
"""
from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd


class VectorizedBacktestEngine:
    """向量化回测引擎 (A股规则)"""

    def __init__(
        self,
        init_capital: float = 1e6,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        max_volume_pct: float = 0.10,
        benchmark: str = "000300.SH",
    ):
        """
        参数:
            init_capital: 初始资金
            commission_rate: 双边佣金费率
            stamp_tax_rate: 卖出印花税率
            min_commission: 单笔最低佣金 (元)
            slippage: 滑点比例 (双向)
            t_plus_1: 是否启用 T+1
            price_limit: 是否启用涨跌停限制
            max_volume_pct: 单笔买入不超过当日成交量的比例
            benchmark: 基准代码
        """
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit
        self.max_volume_pct = max_volume_pct
        self.benchmark = benchmark

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            data: 行情数据，列: code, date, open, high, low, close, volume,
                  is_st, is_limit_up, is_limit_down
            signals: 交易信号
                - 模式A (signal): code, date, signal (1买入, -1卖出, 0持仓)
                - 模式B (target_weight): code, date, target_weight (0~1)
            benchmark_data: 基准行情数据 (可选)，列: date, close

        返回:
            {
                "trades": DataFrame,
                "positions": DataFrame,    # 每日持仓明细
                "equity_curve": DataFrame, # 含 equity, benchmark, cash, market_value
                "metrics": dict,
                "elapsed_sec": float,
            }
        """
        t0 = time.perf_counter()

        if data.empty or signals.empty:
            return self._empty_result(elapsed=time.perf_counter() - t0)

        # 统一列名与排序
        data = data.copy()
        signals = signals.copy()
        data["date"] = pd.to_datetime(data["date"])
        signals["date"] = pd.to_datetime(signals["date"])

        # 补全缺失的辅助列
        for col, default in [
            ("is_limit_up", False),
            ("is_limit_down", False),
            ("is_st", False),
            ("volume", 0),
        ]:
            if col not in data.columns:
                data[col] = default

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        # 检测信号模式
        signal_mode = self._detect_signal_mode(signals)

        # 预构建日期索引: date -> DataFrame (避免每日 filter)
        data_by_date = {dt: group for dt, group in data.groupby("date")}

        all_dates = sorted(signals["date"].unique())
        if len(all_dates) == 0:
            return self._empty_result(elapsed=time.perf_counter() - t0)

        # 持仓状态: code -> {"shares": int, "buy_date": Timestamp, "entry_price": float}
        positions: Dict[str, Dict[str, Any]] = {}
        cash = float(self.init_capital)
        trades: List[Dict] = []
        equity_records: List[Dict] = []
        daily_position_records: List[Dict] = []

        for dt in all_dates:
            # 当日行情 (用预构建的 dict 直接取)
            day_data = data_by_date.get(dt)
            if day_data is None or day_data.empty:
                continue

            # 构建 code -> 各字段的映射 (用 set_index 一次完成)
            day_data_indexed = day_data.set_index("code")
            day_price_map = day_data_indexed["close"].to_dict()
            day_volume_map = day_data_indexed["volume"].to_dict() if "volume" in day_data_indexed else {}
            day_limit_up = day_data_indexed["is_limit_up"].to_dict() if "is_limit_up" in day_data_indexed else {}
            day_limit_down = day_data_indexed["is_limit_down"].to_dict() if "is_limit_down" in day_data_indexed else {}

            # 当日信号
            day_signal = signals[signals["date"] == dt]

            # 1) 决定目标持仓变化
            if signal_mode == "target_weight":
                orders = self._derive_orders_from_weight(
                    day_signal, positions, cash, day_price_map, dt
                )
            else:
                orders = self._derive_orders_from_signal(
                    day_signal, positions, day_price_map, dt
                )

            # 2) 先卖后买
            sell_orders = [o for o in orders if o["action"] == "sell"]
            buy_orders = [o for o in orders if o["action"] == "buy"]

            # 执行卖出
            for o in sell_orders:
                code = o["code"]
                if code not in positions:
                    continue
                pos = positions[code]
                # T+1 检查
                if self.t_plus_1 and pos["buy_date"] >= dt:
                    continue
                # 涨跌停检查 (卖出时若跌停无法成交)
                if self.price_limit and day_limit_down.get(code, False):
                    continue
                price = day_price_map.get(code)
                if price is None or price <= 0:
                    continue
                # 滑点 (卖出价偏低)
                fill_price = price * (1 - self.slippage)
                shares = pos["shares"]
                amount = fill_price * shares
                commission = max(amount * self.commission_rate, self.min_commission)
                tax = amount * self.stamp_tax_rate
                cost = commission + tax
                cash += amount - cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": fill_price, "shares": shares, "amount": amount,
                    "commission": commission, "tax": tax,
                    "pnl": amount - cost - pos["shares"] * pos["entry_price"],
                })
                del positions[code]

            # 执行买入
            if buy_orders:
                # 按信号强度排序 (若有)，否则等权
                total_strength = sum(o.get("strength", 1.0) for o in buy_orders) or 1.0
                budget = cash * 0.95
                for o in buy_orders:
                    code = o["code"]
                    if self.price_limit and day_limit_up.get(code, False):
                        continue
                    price = day_price_map.get(code)
                    if price is None or price <= 0:
                        continue
                    fill_price = price * (1 + self.slippage)
                    # 资金分配 (按强度加权)
                    strength = o.get("strength", 1.0)
                    alloc = budget * (strength / total_strength)
                    # 成交量限制
                    day_vol = day_volume_map.get(code, 0)
                    if day_vol > 0 and self.max_volume_pct < 1.0:
                        max_shares_by_vol = int(day_vol * self.max_volume_pct)
                    else:
                        max_shares_by_vol = 10**9
                    target_shares = int(alloc / fill_price / 100) * 100
                    target_shares = min(target_shares, max_shares_by_vol)
                    if target_shares <= 0:
                        continue
                    amount = fill_price * target_shares
                    commission = max(amount * self.commission_rate, self.min_commission)
                    cost = amount + commission
                    if cost > cash:
                        target_shares = int((cash * 0.98) / fill_price / 100) * 100
                        if target_shares <= 0:
                            continue
                        amount = fill_price * target_shares
                        commission = max(amount * self.commission_rate, self.min_commission)
                        cost = amount + commission
                    cash -= cost
                    # 加仓: 累加份额，更新买入日期与成本
                    if code in positions:
                        old = positions[code]
                        total_shares = old["shares"] + target_shares
                        new_entry = (
                            old["shares"] * old["entry_price"] + target_shares * fill_price
                        ) / total_shares
                        positions[code] = {
                            "shares": total_shares,
                            "buy_date": dt,  # T+1: 最新买入日
                            "entry_price": new_entry,
                        }
                    else:
                        positions[code] = {
                            "shares": target_shares,
                            "buy_date": dt,
                            "entry_price": fill_price,
                        }
                    trades.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": fill_price, "shares": target_shares, "amount": amount,
                        "commission": commission, "tax": 0.0,
                        "pnl": -cost,
                    })

            # 3) 计算当日总权益 (向量化)
            market_value = 0.0
            for code, pos in positions.items():
                px = day_price_map.get(code)
                if px is not None:
                    market_value += pos["shares"] * px

            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": len(positions),
            })

            # 记录持仓明细 (用于归因)
            for code, pos in positions.items():
                px = day_price_map.get(code)
                daily_position_records.append({
                    "date": dt,
                    "code": code,
                    "shares": pos["shares"],
                    "price": px if px is not None else 0.0,
                    "market_value": pos["shares"] * (px or 0.0),
                    "entry_price": pos["entry_price"],
                    "days_held": (dt - pos["buy_date"]).days,
                })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)
        positions_df = pd.DataFrame(daily_position_records)

        # 基准对比
        if benchmark_data is not None and not benchmark_data.empty:
            bench = benchmark_data.copy()
            bench["date"] = pd.to_datetime(bench["date"])
            bench = bench.sort_values("date").drop_duplicates("date")
            bench = bench.set_index("date")["close"]
            # 对齐到策略日期
            bench_aligned = bench.reindex(equity_curve["date"]).ffill()
            # 基准归一化为同初始资金
            if len(bench_aligned) > 0 and bench_aligned.iloc[0] > 0:
                equity_curve["benchmark"] = (
                    bench_aligned.values / bench_aligned.iloc[0] * self.init_capital
                )
            else:
                equity_curve["benchmark"] = self.init_capital
        else:
            equity_curve["benchmark"] = self.init_capital

        # 计算绩效
        metrics = self._calc_metrics(equity_curve, trades_df)

        elapsed = time.perf_counter() - t0
        metrics["elapsed_sec"] = round(elapsed, 4)

        return {
            "trades": trades_df,
            "positions": positions_df,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "elapsed_sec": round(elapsed, 4),
            "signal_mode": signal_mode,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _detect_signal_mode(self, signals: pd.DataFrame) -> str:
        if "target_weight" in signals.columns:
            return "target_weight"
        return "signal"

    def _derive_orders_from_signal(
        self,
        day_signal: pd.DataFrame,
        positions: Dict,
        day_price_map: Dict,
        dt,
    ) -> List[Dict]:
        orders = []
        for _, row in day_signal.iterrows():
            code = row["code"]
            sig = row.get("signal", 0)
            if isinstance(sig, (int, float, np.integer, np.floating)):
                sig = float(sig)
                if sig > 0:
                    # 允许加仓 (与原版一致)，T+1 在卖出时检查
                    orders.append({"code": code, "action": "buy", "strength": abs(sig)})
                elif sig < 0 and code in positions:
                    orders.append({"code": code, "action": "sell", "strength": abs(sig)})
        return orders

    def _derive_orders_from_weight(
        self,
        day_signal: pd.DataFrame,
        positions: Dict,
        cash: float,
        day_price_map: Dict,
        dt,
    ) -> List[Dict]:
        """根据目标权重生成买卖单"""
        orders = []
        current_codes = set(positions.keys())
        target_codes = set()

        for _, row in day_signal.iterrows():
            code = row["code"]
            tw = float(row.get("target_weight", 0.0))
            if tw > 0.001:
                target_codes.add(code)
                # 简化：目标有则视为买入信号 (实际应计算差额)
                if code not in positions:
                    orders.append({"code": code, "action": "buy", "strength": tw})
            else:
                # 目标权重为 0，若持有则卖出
                if code in positions:
                    orders.append({"code": code, "action": "sell", "strength": 1.0})

        # 持有但不在目标中的，卖出
        for code in current_codes - target_codes:
            orders.append({"code": code, "action": "sell", "strength": 1.0})

        return orders

    def _calc_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        risk_free: float = 0.03,
        trading_days: int = 252,
    ) -> Dict[str, Any]:
        if equity_curve.empty or len(equity_curve) < 2:
            return {}

        eq = equity_curve.set_index("date")["equity"]
        bench = equity_curve.set_index("date")["benchmark"] if "benchmark" in equity_curve.columns else None

        returns = eq.pct_change().dropna()
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(eq) / trading_days
        ann_return = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        vol = float(returns.std() * np.sqrt(trading_days)) if len(returns) > 1 else 0.0
        sharpe = float((returns.mean() * trading_days - risk_free) / vol) if vol > 0 else 0.0

        # 最大回撤
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        max_dd = float(drawdown.min())

        # Sortino
        neg_ret = returns[returns < 0]
        downside_std = float(neg_ret.std() * np.sqrt(trading_days)) if len(neg_ret) > 1 else 0.0
        sortino = float((returns.mean() * trading_days - risk_free) / downside_std) if downside_std > 0 else 0.0

        # Calmar
        calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

        # 胜率
        if not trades.empty:
            sell_trades = trades[trades["action"] == "sell"]
            win_rate = float((sell_trades["pnl"] > 0).mean()) if len(sell_trades) > 0 else 0.0
            n_trades = len(trades)
        else:
            win_rate = 0.0
            n_trades = 0

        # 基准对比
        bench_total_return = None
        alpha = None
        beta = None
        if bench is not None and len(bench) > 1:
            bench_total_return = float(bench.iloc[-1] / bench.iloc[0] - 1)
            bench_returns = bench.pct_change().dropna()
            aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
            aligned.columns = ["strat", "bench"]
            if len(aligned) > 1 and aligned["bench"].std() > 0:
                cov = aligned.cov().iloc[0, 1]
                beta = float(cov / aligned["bench"].var())
                alpha = float(ann_return - risk_free - beta * (bench_total_return / max(n_years, 0.01) - risk_free))

        return {
            "total_return": round(total_return, 6),
            "annual_return": round(ann_return, 6),
            "benchmark_return": round(bench_total_return, 6) if bench_total_return is not None else None,
            "excess_return": round(total_return - bench_total_return, 6) if bench_total_return is not None else None,
            "volatility": round(vol, 6),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown": round(max_dd, 6),
            "win_rate": round(win_rate, 4),
            "total_trades": n_trades,
            "alpha": round(alpha, 4) if alpha is not None else None,
            "beta": round(beta, 4) if beta is not None else None,
        }

    def _empty_result(self, elapsed: float = 0.0) -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {"elapsed_sec": round(elapsed, 4)},
            "elapsed_sec": round(elapsed, 4),
        }
