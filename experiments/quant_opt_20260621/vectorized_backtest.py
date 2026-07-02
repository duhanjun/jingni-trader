"""
向量化回测引擎 - 优化验证模块

借鉴来源:
  - VectorBT / VectorBT PRO: 向量化回测思想, 用 NumPy 数组运算替代逐 bar Python 循环
  - NautilusTrader: 强调回测/实盘一致性, 严格避免前视偏差 (look-ahead bias)
  - Zipline-Reloaded: 信号 T 日生成 -> T+1 日开盘成交的标准事件模型

针对 jingni-trader 现有 native_adapter.py 的主要问题:
  1. 逐行 iterrows 遍历, 性能差 (O(N*M) Python 循环)
  2. 信号在当日 close 生成, 同时用当日 close 成交 -> 前视偏差
  3. t_plus_1 参数被传入但未实际使用
  4. 涨跌停 / 停牌过滤不完整
  5. win_rate 基于 buy/sell 单笔 pnl, 计算无意义

本模块实现:
  - 信号 T 日生成 -> T+1 日 open 成交 (消除前视偏差)
  - 严格 T+1: 当日买入不可当日卖出
  - 涨停拒绝买入, 跌停拒绝卖出, 停牌拒绝成交
  - 向量化计算每日持仓市值与现金
  - 等权调仓模式 (equal_weight) 与目标权重模式 (target_weight)

注意: 本文件仅用于优化验证, 不修改 main 分支任何代码。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """回测参数配置"""
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025      # 佣金费率 (双边)
    min_commission: float = 5.0           # 单笔最低佣金 (元)
    stamp_tax_rate: float = 0.001         # 印花税 (仅卖出)
    slippage: float = 0.001               # 滑点 (买入价上浮, 卖出价下浮)
    t_plus_1: bool = True                 # T+1 限制
    price_limit: bool = True              # 涨跌停限制
    reject_st: bool = True                # 停牌拒绝成交
    trade_on: Literal["next_open", "same_close"] = "next_open"  # 成交价选择


class VectorizedBacktester:
    """
    向量化回测引擎

    核心思路:
      1. 把信号按 (date, code) 对齐到行情, shift 一次得到 T+1 执行信号
      2. 用 next_open 作为成交价, 避免前视偏差
      3. 持仓市值 = 持仓股数 * 当日 close, 用矩阵乘法一次性算出
      4. 现金变化用向量化 cumsum 计算
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.cfg = config or BacktestConfig()

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        mode: Literal["equal_weight", "target_weight"] = "equal_weight",
        max_positions: int = 20,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            data: 行情数据, 必须含列 code, date, open, high, low, close, volume
                  可选列: is_limit_up, is_limit_down, is_st (0/1)
            signals: 信号数据, 必须含列 code, date
                     equal_weight 模式: signal 列 (1=买入, -1=卖出, 0=持有)
                     target_weight 模式: target_weight 列 (0~1)
            mode: 调仓模式
            max_positions: equal_weight 模式下最大持仓数

        返回:
            {
                "equity_curve": DataFrame[date, equity, cash, market_value, position_count],
                "trades": DataFrame[date, code, action, price, shares, amount, commission, tax, pnl],
                "metrics": dict,
                "execution_price": "next_open" | "same_close",
            }
        """
        cfg = self.cfg
        if data.empty or signals.empty:
            return self._empty_result()

        # 1. 数据预处理: 排序 + 透视为宽表 (index=date, columns=code)
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        signals = signals.copy()
        signals["date"] = pd.to_datetime(signals["date"])

        # 补全缺失的标志列
        for col, default in [("is_limit_up", 0), ("is_limit_down", 0), ("is_st", 0)]:
            if col not in data.columns:
                data[col] = default
            else:
                data[col] = data[col].fillna(0).astype(int)

        # 2. 构造成交价:
        #    - next_open: 信号 T 日生成 -> T+1 日 open 成交 (exec_price = 当日 open, 信号 shift 到下一日)
        #    - same_close: 信号 T 日生成 -> T 日 close 成交 (兼容旧行为, 有前视偏差)
        data = data.sort_values(["code", "date"]).reset_index(drop=True)
        if cfg.trade_on == "next_open":
            data["exec_price"] = data["open"]  # 执行日的开盘价 (信号已 shift 到该日)
        else:
            data["exec_price"] = data["close"]

        # 3. 信号对齐到行情, 信号 T 日生成 -> T+1 日成交
        # 用 merge 把 signal 附加到 data 上, 然后 shift(1) 得到 T+1 执行信号
        sig_col = "signal" if mode == "equal_weight" else "target_weight"
        if sig_col not in signals.columns:
            raise ValueError(f"signals 必须包含 {sig_col} 列 (mode={mode})")

        # 去重: 同一 (code, date) 只保留最后一条信号
        signals = signals.drop_duplicates(subset=["code", "date"], keep="last")

        merged = data.merge(
            signals[["code", "date", sig_col]],
            on=["code", "date"],
            how="left",
        )
        merged[sig_col] = merged[sig_col].fillna(0)
        # 关键: 信号 T 日生成, T+1 日开盘成交 -> 把信号 shift 到下一行
        # same_close 模式不 shift (兼容旧 native_adapter 行为, 但有前视偏差)
        if cfg.trade_on == "next_open":
            merged["exec_signal"] = merged.groupby("code")[sig_col].shift(1).fillna(0)
        else:
            merged["exec_signal"] = merged[sig_col]

        # 4. 透视为宽表, 便于向量化
        # exec_price_w: index=date, columns=code
        exec_price_w = merged.pivot(index="date", columns="code", values="exec_price")
        close_w = merged.pivot(index="date", columns="code", values="close")
        limit_up_w = merged.pivot(index="date", columns="code", values="is_limit_up").fillna(0)
        limit_down_w = merged.pivot(index="date", columns="code", values="is_limit_down").fillna(0)
        st_w = merged.pivot(index="date", columns="code", values="is_st").fillna(0)
        exec_signal_w = merged.pivot(index="date", columns="code", values="exec_signal").fillna(0)

        codes = list(exec_price_w.columns)
        dates = list(exec_price_w.index)
        n_codes = len(codes)
        n_dates = len(dates)

        # 5. 持仓矩阵 shares_w (date x code), 现金序列 cash
        shares_arr = np.zeros((n_dates, n_codes), dtype=np.float64)
        cash_arr = np.full(n_dates, cfg.init_capital, dtype=np.float64)
        # 记录当日是否新建仓 (用于 T+1 限制)
        bought_today = np.zeros((n_dates, n_codes), dtype=bool)

        trades_records: list[dict] = []

        # 6. 逐日调仓 (调仓逻辑必须按时间顺序, 但单日内可向量化)
        # 注: 调仓本身是 path-dependent (现金依赖前一日), 无法完全向量化
        #     但单日内的目标权重计算、过滤、成交可以批量处理
        for i, dt in enumerate(dates):
            if i == 0:
                # 第一日没有 exec_signal (因为 shift(1) 后第一日为 0)
                # 仅记录初始状态
                continue

            prev_cash = cash_arr[i - 1]
            prev_shares = shares_arr[i - 1].copy()
            # T+1 规则: 当日买入的股票不可当日卖出
            # 由于采用 "先卖后买" 顺序, 卖出时当日尚未买入, 自然满足 T+1
            # locked 仅在 target_weight 模式下用于防止同日先买后卖 (该模式下单只股票
            # 要么买要么卖, 不会触发, 保留作为安全防护)
            locked = np.zeros(n_codes, dtype=bool)

            exec_prices = exec_price_w.iloc[i].values.astype(np.float64)
            closes = close_w.iloc[i].values.astype(np.float64)
            sigs = exec_signal_w.iloc[i].values.astype(np.float64)
            lim_up = limit_up_w.iloc[i].values.astype(bool)
            lim_dn = limit_down_w.iloc[i].values.astype(bool)
            st_flag = st_w.iloc[i].values.astype(bool)

            # 无效价格 (停牌/NaN) 视为不可成交
            tradable = ~np.isnan(exec_prices) & (exec_prices > 0)
            if cfg.reject_st:
                tradable = tradable & ~st_flag

            new_shares = prev_shares.copy()
            new_bought_today = np.zeros(n_codes, dtype=bool)
            cash = prev_cash

            if mode == "equal_weight":
                # 卖出: signal == -1 且未锁定 且 可成交 (非跌停)
                sell_mask = (sigs < 0) & (prev_shares > 0) & ~locked & tradable & ~lim_dn
                if sell_mask.any():
                    sell_codes_idx = np.where(sell_mask)[0]
                    for idx in sell_codes_idx:
                        price = exec_prices[idx] * (1 - cfg.slippage)
                        shares = prev_shares[idx]
                        amount = price * shares
                        commission = max(amount * cfg.commission_rate, cfg.min_commission)
                        tax = amount * cfg.stamp_tax_rate
                        cash += amount - commission - tax
                        new_shares[idx] = 0
                        trades_records.append({
                            "date": dt, "code": codes[idx], "action": "sell",
                            "price": price, "shares": int(shares), "amount": amount,
                            "commission": commission, "tax": tax,
                            "pnl": amount - commission - tax,  # 卖出回收现金 (绝对值, 非盈亏)
                        })

                # 买入: signal == 1 且当前无持仓 且 可成交 (非涨停)
                buy_mask = (sigs > 0) & (prev_shares == 0) & tradable & ~lim_up
                buy_idx = np.where(buy_mask)[0]
                if len(buy_idx) > 0:
                    # 限制最大持仓数
                    current_hold = (new_shares > 0).sum()
                    available_slots = max(0, max_positions - current_hold)
                    if available_slots > 0:
                        buy_idx = buy_idx[:available_slots]
                        budget_per = cash * 0.95 / len(buy_idx)
                        for idx in buy_idx:
                            price = exec_prices[idx] * (1 + cfg.slippage)
                            # A股 100 股一手
                            shares = int(budget_per / price / 100) * 100
                            if shares <= 0:
                                continue
                            amount = price * shares
                            commission = max(amount * cfg.commission_rate, cfg.min_commission)
                            total_cost = amount + commission
                            if total_cost > cash:
                                shares = int((cash * 0.98) / price / 100) * 100
                                if shares <= 0:
                                    continue
                                amount = price * shares
                                commission = max(amount * cfg.commission_rate, cfg.min_commission)
                                total_cost = amount + commission
                            cash -= total_cost
                            new_shares[idx] = shares
                            new_bought_today[idx] = True
                            trades_records.append({
                                "date": dt, "code": codes[idx], "action": "buy",
                                "price": price, "shares": int(shares), "amount": amount,
                                "commission": commission, "tax": 0.0,
                                "pnl": -total_cost,  # 买入支出现金 (负值)
                            })

            elif mode == "target_weight":
                # 目标权重模式: 调整到 target_weight
                total_target = float(np.nansum(np.abs(sigs)))
                if total_target <= 0:
                    # 无信号, 仅持有
                    pass
                else:
                    # 计算每只股票目标市值
                    target_mv = (sigs / total_target) * cash  # 简化: 用当前现金分配
                    # 先卖后买
                    for idx in range(n_codes):
                        if not tradable[idx]:
                            continue
                        price = exec_prices[idx]
                        if np.isnan(price) or price <= 0:
                            continue
                        target_shares = int(target_mv[idx] / price / 100) * 100
                        current = new_shares[idx]
                        if target_shares > current and not lim_up[idx]:
                            # 买入
                            delta = target_shares - current
                            if cfg.t_plus_1 and locked[idx]:
                                continue
                            buy_price = price * (1 + cfg.slippage)
                            amount = buy_price * delta
                            commission = max(amount * cfg.commission_rate, cfg.min_commission)
                            if amount + commission > cash:
                                continue
                            cash -= amount + commission
                            new_shares[idx] += delta
                            new_bought_today[idx] = True
                            trades_records.append({
                                "date": dt, "code": codes[idx], "action": "buy",
                                "price": buy_price, "shares": int(delta), "amount": amount,
                                "commission": commission, "tax": 0.0, "pnl": -(amount + commission),
                            })
                        elif target_shares < current and not lim_dn[idx]:
                            # 卖出
                            if cfg.t_plus_1 and locked[idx]:
                                continue
                            delta = current - target_shares
                            sell_price = price * (1 - cfg.slippage)
                            amount = sell_price * delta
                            commission = max(amount * cfg.commission_rate, cfg.min_commission)
                            tax = amount * cfg.stamp_tax_rate
                            cash += amount - commission - tax
                            new_shares[idx] -= delta
                            trades_records.append({
                                "date": dt, "code": codes[idx], "action": "sell",
                                "price": sell_price, "shares": int(delta), "amount": amount,
                                "commission": commission, "tax": tax,
                                "pnl": amount - commission - tax,
                            })

            shares_arr[i] = new_shares
            cash_arr[i] = cash
            bought_today[i] = new_bought_today

        # 7. 向量化计算每日市值 (持仓股数 * 当日 close)
        close_arr = close_w.values.astype(np.float64)
        market_value_arr = np.nansum(shares_arr * close_arr, axis=1)
        equity_arr = cash_arr + market_value_arr
        position_count_arr = (shares_arr > 0).sum(axis=1)

        equity_curve = pd.DataFrame({
            "date": dates,
            "equity": equity_arr,
            "cash": cash_arr,
            "market_value": market_value_arr,
            "position_count": position_count_arr,
        })

        trades_df = pd.DataFrame(trades_records)

        # 8. 计算绩效指标
        metrics = self._calc_metrics(equity_curve, trades_df)

        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "metrics": metrics,
            "execution_price": cfg.trade_on,
            "n_codes": n_codes,
            "n_dates": n_dates,
        }

    # ---------------- 绩效指标计算 ----------------

    def _calc_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        risk_free: float = 0.03,
        trading_days: int = 252,
    ) -> Dict[str, Any]:
        """计算绩效指标 (修正版)"""
        if equity_curve.empty or len(equity_curve) < 2:
            return {}

        eq = equity_curve.set_index("date")["equity"].astype(float)
        returns = eq.pct_change().dropna()

        # 累计收益
        total_return = float(eq.iloc[-1] / eq.iloc[0] - 1)

        # 年化收益 (几何)
        n_years = len(eq) / trading_days
        annual_return = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        # 年化波动率
        volatility = float(returns.std() * np.sqrt(trading_days)) if len(returns) > 1 else 0.0

        # Sharpe (用几何年化收益, 与 annual_return 一致)
        sharpe = float((annual_return - risk_free) / volatility) if volatility > 0 else 0.0

        # Sortino (下行风险)
        neg_ret = returns[returns < 0]
        downside_std = float(neg_ret.std() * np.sqrt(trading_days)) if len(neg_ret) > 1 else 0.0
        sortino = float((annual_return - risk_free) / downside_std) if downside_std > 0 else 0.0

        # 最大回撤
        cummax = eq.cummax()
        drawdown = (eq - cummax) / cummax
        max_drawdown = float(drawdown.min())

        # Calmar
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

        # 胜率 (基于平仓交易: sell 的 pnl > 0 视为盈利)
        # 注: 这里 pnl 是卖出回收现金, 需要重新定义为 (卖出收入 - 买入成本)
        # 简化: 用日收益率正负比例作为日胜率
        daily_win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

        # 交易统计
        n_trades = len(trades)
        n_buys = int((trades["action"] == "buy").sum()) if n_trades > 0 else 0
        n_sells = int((trades["action"] == "sell").sum()) if n_trades > 0 else 0

        # 换手率 (粗略: 卖出金额 / 平均权益)
        avg_equity = float(eq.mean())
        sell_amount = float(trades.loc[trades["action"] == "sell", "amount"].sum()) if n_trades > 0 else 0.0
        turnover = float(sell_amount / avg_equity) if avg_equity > 0 else 0.0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "daily_win_rate": daily_win_rate,
            "n_trades": n_trades,
            "n_buys": n_buys,
            "n_sells": n_sells,
            "turnover": turnover,
            "n_days": len(eq),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "metrics": {},
            "execution_price": self.cfg.trade_on,
            "n_codes": 0,
            "n_dates": 0,
        }
