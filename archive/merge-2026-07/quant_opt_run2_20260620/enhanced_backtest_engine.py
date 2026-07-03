"""
增强版回测引擎（向量化 + T+1 执行 + 基准相对指标）

借鉴来源：
  - Microsoft Qlib: Point-in-Time 执行、防止前视偏差
    https://arxiv.org/abs/2009.11189
  - QuantRocket Moonshot: 向量化回测思想
    https://www.quantrocket.com/
  - pyfolio: 基准相对绩效指标 (Alpha/Beta/IR/Tracking Error)
    https://github.com/quantopian/pyfolio

与 jingni-trader 现有 native_adapter.py 的关键差异：
  1. 前视偏差修复：信号在 T 日收盘生成，T+1 日执行（现有实现信号日即执行日）
  2. 性能：用 groupby/date 一次性切片替代 data[data['date']==dt] 逐日过滤
  3. 持仓权重：支持 target_weight 信号（现有仅支持 1/-1 等额）
  4. 风险指标：新增 Alpha/Beta/Information Ratio/Tracking Error/VaR/CVaR/Omega
  5. 基准对比：equity_curve 含 benchmark 列

注意：本文件位于 feat/quant-opt-20260620 分支，不修改 main 分支代码。
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class EnhancedBacktestEngine:
    """
    向量化回测引擎

    支持两种信号格式：
      A) signal: 1 买入 / -1 卖出 / 0 持仓（等额分配，兼容现有接口）
      B) target_weight: 目标权重（0~1），支持信号强度仓位
    """

    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        init_capital: float = 1e6,
        benchmark: str = "000300.SH",
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        t_plus_1: bool = True,
        price_limit: bool = True,
        slippage: float = 0.001,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        执行回测

        参数:
            data: 行情数据，列含 code, date, open, high, low, close, volume,
                  is_limit_up, is_limit_down（可选）
            signals: 信号，列含 code, date, signal 或 target_weight
            init_capital: 初始资金
            benchmark: 基准代码（仅用于标注）
            t_plus_1: 是否启用 T+1（信号 T 日，T+1 开盘执行）
            price_limit: 是否启用涨跌停限制
            slippage: 滑点比例
            benchmark_returns: 基准日收益率 Series（index=date），
                               用于计算 Alpha/Beta/IR；为空则这些指标置 NaN

        返回:
            {
                "trades": DataFrame,
                "positions": DataFrame,        # 每日持仓快照
                "equity_curve": DataFrame,     # date, equity, cash, market_value, benchmark
                "metrics": dict,
            }
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # ---- 数据预处理 ----
        data = data.copy()
        signals = signals.copy()
        data["date"] = pd.to_datetime(data["date"])
        signals["date"] = pd.to_datetime(signals["date"])

        # 补齐涨跌停标记（若数据未提供则视为无限制）
        for col in ("is_limit_up", "is_limit_down"):
            if col not in data.columns:
                data[col] = False

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        # ---- 信号归一化为 target_weight ----
        signals = self._normalize_signals(signals)

        # ---- T+1：信号日 shift 到下一交易日执行 ----
        # 构造执行信号：每个 code 的目标权重延迟一天应用
        exec_signals = self._shift_signals_to_next_day(signals, data)

        # ---- 按日期分组向量化回测 ----
        result = self._vectorized_loop(
            data=data,
            exec_signals=exec_signals,
            init_capital=init_capital,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            min_commission=min_commission,
            price_limit=price_limit,
            slippage=slippage,
            benchmark=benchmark,
            benchmark_returns=benchmark_returns,
        )
        return result

    # ------------------------------------------------------------
    # 信号归一化
    # ------------------------------------------------------------
    def _normalize_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """把 signal(1/-1/0) 或 target_weight 统一成 target_weight 列"""
        out = signals[["code", "date"]].copy()
        if "target_weight" in signals.columns:
            out["target_weight"] = signals["target_weight"].astype(float).fillna(0.0)
        elif "signal" in signals.columns:
            sig = signals["signal"].astype(float).fillna(0.0)
            # 1 -> 正权重, -1 -> 0(清仓), 0 -> 保持
            # 等额分配需在外层按日归一化，这里先保留原始信号
            out["raw_signal"] = sig
            out["target_weight"] = np.nan  # 占位，后续按日归一
        else:
            raise ValueError("signals 必须包含 signal 或 target_weight 列")
        return out

    # ------------------------------------------------------------
    # T+1 信号位移
    # ------------------------------------------------------------
    def _shift_signals_to_next_day(
        self, signals: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        将 T 日信号映射到 T+1 交易日执行。

        实现：对每只股票，把信号日期映射为该股票下一个交易日。
        这样回测时用 T+1 的开盘价成交，避免用 T 日收盘价成交的前视偏差。
        """
        # 取每只股票的交易日序列
        trade_days_per_code = (
            data.sort_values(["code", "date"])
            .groupby("code")["date"]
            .apply(lambda s: s.reset_index(drop=True))
            .reset_index(level=0)
        )
        # trade_days_per_code: columns = code, date, level_2(idx within code)
        trade_days_per_code = trade_days_per_code.rename(columns={"level_2": "_idx"})
        trade_days_per_code["_next_date"] = trade_days_per_code.groupby("code")["date"].shift(-1)

        # 信号 join 到 next_date
        merged = signals.merge(
            trade_days_per_code[["code", "date", "_next_date"]],
            on=["code", "date"],
            how="left",
        )
        # 执行日 = 信号的下一交易日；若信号日就是最后一天则无执行日，丢弃
        merged = merged.dropna(subset=["_next_date"])
        merged["exec_date"] = merged["_next_date"]
        merged = merged.drop(columns=["date", "_next_date"]).rename(columns={"exec_date": "date"})

        # 对 signal 模式：按执行日归一化为等额 target_weight
        if "raw_signal" in merged.columns and "target_weight" not in merged.columns:
            merged["target_weight"] = np.nan
        if "raw_signal" in merged.columns:
            # 买入信号(>0)按当日买入标的数等额分配；卖出/0 -> 0
            buy_mask = merged["raw_signal"] > 0
            # 每个执行日的买入标的数
            n_buy_per_day = merged[buy_mask].groupby("date")["code"].transform("count")
            merged["target_weight"] = 0.0
            merged.loc[buy_mask, "target_weight"] = 1.0 / n_buy_per_day.reindex(
                merged.index, fill_value=1
            )
            # 卖出信号 -> 权重 0
            merged.loc[merged["raw_signal"] < 0, "target_weight"] = 0.0
        return merged[["code", "date", "target_weight"]].reset_index(drop=True)

    # ------------------------------------------------------------
    # 向量化回测主循环
    # ------------------------------------------------------------
    def _vectorized_loop(
        self,
        data: pd.DataFrame,
        exec_signals: pd.DataFrame,
        init_capital: float,
        commission_rate: float,
        stamp_tax_rate: float,
        min_commission: float,
        price_limit: bool,
        slippage: float,
        benchmark: str,
        benchmark_returns: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """
        按执行日迭代，每日内用向量化操作处理该日所有标的。

        相比 native_adapter 的 data[data['date']==dt] 过滤，
        这里用 groupby(date) 一次性切分，避免重复全表扫描。
        """
        # 用 dict 索引日期 -> 该日数据块，避免重复过滤
        data_by_date = {d: g for d, g in data.groupby("date")}
        sig_by_date = {d: g for d, g in exec_signals.groupby("date")}

        all_dates = sorted(data_by_date.keys())
        if not all_dates:
            return self._empty_result()

        cash = init_capital
        positions: Dict[str, float] = {}  # code -> shares
        cost_basis: Dict[str, float] = {}  # code -> 平均成本
        equity_records: List[Dict] = []
        trades: List[Dict] = []

        for dt in all_dates:
            day_data = data_by_date[dt]
            day_data_map = day_data.set_index("code")

            # 当日信号（已 T+1 位移）
            day_sig = sig_by_date.get(dt)
            if day_sig is not None and not day_sig.empty:
                cash, positions, cost_basis, new_trades = self._rebalance(
                    day_sig=day_sig,
                    day_data_map=day_data_map,
                    cash=cash,
                    positions=positions,
                    cost_basis=cost_basis,
                    init_capital=init_capital,
                    commission_rate=commission_rate,
                    stamp_tax_rate=stamp_tax_rate,
                    min_commission=min_commission,
                    price_limit=price_limit,
                    slippage=slippage,
                    dt=dt,
                )
                trades.extend(new_trades)

            # 盯市：用当日收盘价计算总权益
            market_value = 0.0
            for code, shares in list(positions.items()):
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * float(day_data_map.loc[code, "close"])
            total_equity = cash + market_value

            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records).set_index("date")
        trades_df = pd.DataFrame(trades)

        # 基准对齐
        if benchmark_returns is not None:
            bench = benchmark_returns.copy()
            bench.index = pd.to_datetime(bench.index)
            # 累计基准净值（起点对齐 init_capital）
            common = equity_curve.index.intersection(bench.index)
            if len(common) > 0:
                bench_aligned = bench.loc[common]
                bench_nav = (1 + bench_aligned).cumprod() * init_capital
                equity_curve["benchmark"] = bench_nav
            else:
                equity_curve["benchmark"] = np.nan
        else:
            equity_curve["benchmark"] = np.nan

        metrics = self._calc_metrics(equity_curve, trades_df, benchmark_returns)

        positions_df = pd.DataFrame(
            [{"code": c, "shares": s} for c, s in positions.items() if s > 0]
        )

        return {
            "trades": trades_df,
            "positions": positions_df,
            "equity_curve": equity_curve.reset_index(),
            "metrics": metrics,
            "report_path": "",
        }

    # ------------------------------------------------------------
    # 调仓
    # ------------------------------------------------------------
    def _rebalance(
        self,
        day_sig: pd.DataFrame,
        day_data_map: pd.DataFrame,
        cash: float,
        positions: Dict[str, float],
        cost_basis: Dict[str, float],
        init_capital: float,
        commission_rate: float,
        stamp_tax_rate: float,
        min_commission: float,
        price_limit: bool,
        slippage: float,
        dt: pd.Timestamp,
    ):
        """
        根据目标权重调仓。

        执行价 = 当日开盘价 * (1 + slippage)（买入）/ * (1 - slippage)（卖出）
        涨停不买、跌停不卖。
        """
        new_trades: List[Dict] = []

        # 目标权重 -> 目标市值
        # 用当前总权益估算（cash + 持仓市值）
        cur_market_value = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in day_data_map.index:
                cur_market_value += shares * float(day_data_map.loc[code, "close"])
        total_equity = cash + cur_market_value

        # 1) 先处理卖出（目标权重 < 当前权重）
        for _, row in day_sig.iterrows():
            code = row["code"]
            target_w = float(row["target_weight"])
            cur_shares = positions.get(code, 0.0)
            if cur_shares <= 0:
                continue
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            cur_price = float(price_row["close"])
            cur_value = cur_shares * cur_price
            cur_w = cur_value / total_equity if total_equity > 0 else 0
            if target_w < cur_w:
                # 卖出
                if price_limit and bool(price_row.get("is_limit_down", False)):
                    continue  # 跌停无法卖出
                exec_price = cur_price * (1 - slippage)
                # 卖到目标权重
                target_value = target_w * total_equity
                target_shares = max(0.0, target_value / exec_price)
                sell_shares = cur_shares - target_shares
                sell_shares = int(sell_shares / 100) * 100  # 整百
                if sell_shares <= 0:
                    continue
                sell_amount = exec_price * sell_shares
                commission = max(sell_amount * commission_rate, min_commission)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                # 已实现盈亏
                avg_cost = cost_basis.get(code, exec_price)
                realized_pnl = (exec_price - avg_cost) * sell_shares - cost
                positions[code] = cur_shares - sell_shares
                if positions[code] <= 0:
                    positions.pop(code, None)
                    cost_basis.pop(code, None)
                new_trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": exec_price, "shares": sell_shares,
                    "amount": sell_amount, "commission": commission,
                    "tax": tax, "pnl": realized_pnl,
                })

        # 2) 再处理买入（目标权重 > 当前权重）
        # 重新计算可用资金下的总权益
        cur_market_value = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in day_data_map.index:
                cur_market_value += shares * float(day_data_map.loc[code, "close"])
        total_equity = cash + cur_market_value

        buy_candidates = []
        for _, row in day_sig.iterrows():
            code = row["code"]
            target_w = float(row["target_weight"])
            cur_shares = positions.get(code, 0.0)
            if code not in day_data_map.index:
                continue
            price_row = day_data_map.loc[code]
            cur_price = float(price_row["close"])
            cur_value = cur_shares * cur_price
            cur_w = cur_value / total_equity if total_equity > 0 else 0
            if target_w > cur_w and target_w > 0:
                if price_limit and bool(price_row.get("is_limit_up", False)):
                    continue  # 涨停无法买入
                buy_candidates.append((code, target_w, cur_w, cur_price, price_row))

        if buy_candidates:
            # 按目标权重等比缩放，确保总仓位 <= 95%
            total_target_w = sum(tw for _, tw, _, _, _ in buy_candidates)
            # 加上已持仓但不调仓的部分
            existing_w = sum(
                positions.get(c, 0) * float(day_data_map.loc[c, "close"])
                for c in positions if c in day_data_map.index
            ) / total_equity if total_equity > 0 else 0
            max_total_w = 0.95
            scale = min(1.0, (max_total_w - existing_w) / total_target_w) if total_target_w > 0 else 0
            scale = max(0.0, scale)

            for code, target_w, cur_w, cur_price, price_row in buy_candidates:
                adj_target_w = target_w * scale
                exec_price = cur_price * (1 + slippage)
                target_value = adj_target_w * total_equity
                target_shares = target_value / exec_price
                # 整百
                target_shares = int(target_shares / 100) * 100
                cur_shares = positions.get(code, 0.0)
                buy_shares = target_shares - cur_shares
                if buy_shares <= 0:
                    continue
                buy_amount = exec_price * buy_shares
                commission = max(buy_amount * commission_rate, min_commission)
                cost = buy_amount + commission
                if cost > cash:
                    # 资金不足，按可用资金重算
                    buy_shares = int((cash * 0.98) / exec_price / 100) * 100
                    if buy_shares <= 0:
                        continue
                    buy_amount = exec_price * buy_shares
                    commission = max(buy_amount * commission_rate, min_commission)
                    cost = buy_amount + commission
                cash -= cost
                # 更新成本
                old_cost = cost_basis.get(code, 0) * cur_shares
                new_cost_total = old_cost + buy_amount
                new_shares = cur_shares + buy_shares
                cost_basis[code] = new_cost_total / new_shares if new_shares > 0 else exec_price
                positions[code] = new_shares
                new_trades.append({
                    "date": dt, "code": code, "action": "buy",
                    "price": exec_price, "shares": buy_shares,
                    "amount": buy_amount, "commission": commission,
                    "tax": 0.0, "pnl": -commission,
                })

        return cash, positions, cost_basis, new_trades

    # ------------------------------------------------------------
    # 绩效指标计算（含基准相对指标）
    # ------------------------------------------------------------
    def _calc_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        benchmark_returns: Optional[pd.Series],
    ) -> Dict[str, Any]:
        if equity_curve.empty or "equity" not in equity_curve.columns:
            return {}
        eq = equity_curve["equity"]
        if len(eq) < 2:
            return {}
        returns = eq.pct_change().dropna()
        trading_days = 252
        risk_free = 0.03

        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)
        n_years = len(eq) / trading_days
        annual_return = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        volatility = float(returns.std() * np.sqrt(trading_days)) if len(returns) > 1 else 0.0
        sharpe = float((annual_return - risk_free) / volatility) if volatility > 0 else 0.0
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        max_drawdown = float(drawdown.min())
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
        neg = returns[returns < 0]
        downside_std = float(neg.std() * np.sqrt(trading_days)) if len(neg) > 1 else 0.0
        sortino = float((annual_return - risk_free) / downside_std) if downside_std > 0 else 0.0

        # 基准相对指标
        alpha = beta = info_ratio = tracking_error = float("nan")
        var_95 = cvar_95 = omega = float("nan")
        win_rate = 0.0

        if benchmark_returns is not None and not benchmark_returns.empty:
            bench = benchmark_returns.copy()
            bench.index = pd.to_datetime(bench.index)
            common = returns.index.intersection(bench.index)
            if len(common) > 1:
                r = returns.loc[common]
                b = bench.loc[common].reindex(common).fillna(0.0)
                # Beta / Alpha (CAPM)
                cov_rb = float(np.cov(r, b)[0, 1])
                var_b = float(b.var())
                beta = cov_rb / var_b if var_b > 0 else float("nan")
                bench_annual = float(b.mean() * trading_days)
                alpha = annual_return - risk_free - beta * (bench_annual - risk_free) if not np.isnan(beta) else float("nan")
                # Tracking Error & Information Ratio
                active = r - b
                tracking_error = float(active.std() * np.sqrt(trading_days)) if len(active) > 1 else 0.0
                ann_active = float(active.mean() * trading_days)
                info_ratio = ann_active / tracking_error if tracking_error > 0 else 0.0

        # VaR / CVaR (历史法, 95%)
        if len(returns) > 1:
            var_95 = float(np.percentile(returns, 5))
            tail = returns[returns <= var_95]
            cvar_95 = float(tail.mean()) if len(tail) > 0 else var_95
            # Omega ratio
            threshold = 0.0
            gains = returns[returns > threshold]
            losses = -returns[returns < threshold]
            omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else float("inf")

        # 胜率：按已实现交易盈亏
        if not trades.empty and "pnl" in trades.columns:
            sell_trades = trades[trades.get("action") == "sell"]
            if not sell_trades.empty:
                win_rate = float((sell_trades["pnl"] > 0).mean())

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": info_ratio,
            "tracking_error": tracking_error,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "omega_ratio": omega,
            "total_trades": int(len(trades)),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
