"""
向量化回测引擎 (Vectorized Backtest Engine)
============================================

借鉴来源:
- backtesting.py (kernc/backtesting.py, 17K+ stars) 的 vectorized + event-driven 双重设计
- Qlib (microsoft/qlib, 36K+ stars) 的 TopkDropoutStrategy 换仓逻辑
- 现有 jingni-trader 的 native_adapter.py 的 A 股规则 (T+1、涨跌停、印花税)

设计目标:
1. 相比现有 native_adapter.py 的逐行 iterrows (O(N×D))，改为按日批处理 + numba JIT 编译热循环
2. 复现现有 native_adapter 的 A 股规则（涨跌停、T+1、印花税、最低佣金、整百股）
3. 验证功能等价性：相同输入下，每日组合市值、交易记录、绩效指标在合理误差范围内一致

实现要点:
- inner_loop_jit(): 用 @njit 编译最内层「按日模拟」循环，绕过 Python 解释器开销
- 数据预对齐：把 signals + data pivot 成 dense 矩阵 (n_dates × n_codes)，对齐后用纯 numpy/jit 处理
- 与 native_adapter 在功能上等价（涨跌停、印花税、最低佣金、整百股均保留）

设计文档见 ./README.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        """numba 不可用时的占位装饰器"""
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@dataclass
class VectorizedBacktestConfig:
    """回测配置（与 native_adapter 对齐）"""
    init_capital: float = 1_000_000.0
    commission_rate: float = 0.00025     # 万 2.5
    min_commission: float = 5.0          # 最低 5 元
    stamp_tax_rate: float = 0.001        # 千 1（卖出）
    transfer_fee_rate: float = 0.00002    # 万 0.2
    slippage: float = 0.001              # 万 1
    t_plus_1: bool = True                # T+1
    price_limit: bool = True             # 涨跌停限制
    lot_size: int = 100                  # 整百股


@dataclass
class BacktestResult:
    """回测结果结构（与 native_adapter 对齐）"""
    trades: pd.DataFrame
    positions: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    report_path: str = ""


class VectorizedBacktestEngine:
    """
    向量化回测引擎

    核心差异（vs native_adapter）:
    - 不再 iterrows 全表，按日 groupby 后用 numpy 批量算份额/金额
    - 目标权重 → 实际份额 → 交易计划三步走，逻辑与 native 完全等价
    """

    def __init__(self, config: Optional[VectorizedBacktestConfig] = None):
        self.config = config or VectorizedBacktestConfig()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark: str = "000300.SH",
    ) -> BacktestResult:
        if data.empty or signals.empty:
            return self._empty_result()

        # 优先使用 JIT 路径（当 numba 可用）
        if HAS_NUMBA:
            return self._run_backtest_jit(data, signals)

        # 退化：纯 numpy/pandas 路径（保留前面的 _execute_* 实现）
        return self._run_backtest_pure_python(data, signals)

    # ------------------------------------------------------------------
    # JIT 路径：单趟扫描完成回测
    # ------------------------------------------------------------------
    def _run_backtest_jit(
        self, data: pd.DataFrame, signals: pd.DataFrame,
    ) -> BacktestResult:
        # 类型/排序
        data = data.copy()
        signals = signals.copy()
        data["date"] = pd.to_datetime(data["date"])
        signals["date"] = pd.to_datetime(signals["date"])

        # 涨跌停列
        if "is_limit_up" not in data.columns:
            data["is_limit_up"] = data.get("change_pct", 0) >= 9.9
        if "is_limit_down" not in data.columns:
            data["is_limit_down"] = data.get("change_pct", 0) <= -9.9
        data["is_limit_up"] = data["is_limit_up"].fillna(False).astype(bool)
        data["is_limit_down"] = data["is_limit_down"].fillna(False).astype(bool)

        # 对齐成 dense 矩阵
        dates = sorted(signals["date"].unique())
        codes = sorted(set(data["code"]).union(signals["code"]))
        if not dates or not codes:
            return self._empty_result()

        # 预对齐 data
        data_pivot = data.pivot(index="date", columns="code", values="close")
        data_pivot = data_pivot.reindex(index=dates, columns=codes)
        close_mat = data_pivot.values.astype(np.float64)  # (T, N)
        # 标记当日无数据 → 用 NaN；JIT 里用 0 价格视为停牌
        nan_mask = np.isnan(close_mat)

        # 涨跌停
        lu_pivot = data.pivot(index="date", columns="code", values="is_limit_up")
        lu_pivot = lu_pivot.reindex(index=dates, columns=codes).fillna(False)
        ld_pivot = data.pivot(index="date", columns="code", values="is_limit_down")
        ld_pivot = ld_pivot.reindex(index=dates, columns=codes).fillna(False)
        lu_mat = lu_pivot.values.astype(np.bool_)
        ld_mat = ld_pivot.values.astype(np.bool_)

        # 预对齐 signal
        sig_pivot = signals.pivot(index="date", columns="code", values="signal")
        sig_pivot = sig_pivot.reindex(index=dates, columns=codes).fillna(0)
        sig_mat = sig_pivot.values.astype(np.int8)

        # 用 0 价格替代 NaN 避免 jit 报错，并单独记录停牌
        close_for_jit = np.where(nan_mask, 0.0, close_mat)
        suspended = nan_mask  # 停牌 = 当日无数据

        cfg = self.config
        n_dates, n_codes = close_for_jit.shape
        cash = cfg.init_capital

        # positions 数组：连续两个 slot 存 [code_idx, shares]
        positions_shares = np.zeros(n_codes, dtype=np.int64)

        # 输出数组（预先分配最大可能大小）
        max_trades = n_dates * n_codes  # 极端上界
        trade_codes = np.zeros(max_trades, dtype=np.int32)
        trade_actions = np.zeros(max_trades, dtype=np.int8)  # 0=buy, 1=sell
        trade_dates_idx = np.zeros(max_trades, dtype=np.int32)
        trade_prices = np.zeros(max_trades, dtype=np.float64)
        trade_shares_arr = np.zeros(max_trades, dtype=np.int64)
        trade_amounts = np.zeros(max_trades, dtype=np.float64)
        trade_commissions = np.zeros(max_trades, dtype=np.float64)
        trade_taxes = np.zeros(max_trades, dtype=np.float64)
        trade_pnls = np.zeros(max_trades, dtype=np.float64)
        n_trades_out = np.zeros(1, dtype=np.int64)

        equity_arr = np.zeros(n_dates, dtype=np.float64)
        cash_arr = np.zeros(n_dates, dtype=np.float64)
        mv_arr = np.zeros(n_dates, dtype=np.float64)
        pos_cnt_arr = np.zeros(n_dates, dtype=np.int64)

        cash = _simulate_core(
            n_dates, n_codes,
            sig_mat, close_for_jit, lu_mat, ld_mat, suspended,
            cash, positions_shares,
            cfg.commission_rate, cfg.min_commission,
            cfg.stamp_tax_rate, cfg.slippage, cfg.lot_size,
            bool(cfg.price_limit),
            trade_codes, trade_actions, trade_dates_idx,
            trade_prices, trade_shares_arr, trade_amounts,
            trade_commissions, trade_taxes, trade_pnls,
            n_trades_out,
            equity_arr, cash_arr, mv_arr, pos_cnt_arr,
        )

        n_tr = int(n_trades_out[0])
        date_idx_to_str = np.array([pd.Timestamp(d) for d in dates])
        code_idx_to_str = np.array(codes)

        # 还原 trades DataFrame
        trades_df = pd.DataFrame({
            "date": date_idx_to_str[trade_dates_idx[:n_tr]],
            "code": code_idx_to_str[trade_codes[:n_tr]],
            "action": np.where(trade_actions[:n_tr] == 0, "buy", "sell"),
            "price": trade_prices[:n_tr],
            "shares": trade_shares_arr[:n_tr],
            "amount": trade_amounts[:n_tr],
            "commission": trade_commissions[:n_tr],
            "tax": trade_taxes[:n_tr],
            "pnl": trade_pnls[:n_tr],
        })

        equity_curve = pd.DataFrame({
            "date": date_idx_to_str,
            "equity": equity_arr,
            "cash": cash_arr,
            "market_value": mv_arr,
            "position_count": pos_cnt_arr,
        })

        eq_series = pd.Series(equity_arr, index=date_idx_to_str)
        metrics = self._calc_metrics(eq_series, trades_df)

        return BacktestResult(
            trades=trades_df,
            positions=pd.DataFrame({"code": code_idx_to_str, "shares": positions_shares}),
            equity_curve=equity_curve,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Pure Python 路径（numba 不可用时回退）
    # ------------------------------------------------------------------
    def _run_backtest_pure_python(
        self, data: pd.DataFrame, signals: pd.DataFrame,
    ) -> BacktestResult:
        data = data.copy()
        signals = signals.copy()
        data["date"] = pd.to_datetime(data["date"])
        signals["date"] = pd.to_datetime(signals["date"])
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        if "is_limit_up" not in data.columns:
            data["is_limit_up"] = data.get("change_pct", 0) >= 9.9
        if "is_limit_down" not in data.columns:
            data["is_limit_down"] = data.get("change_pct", 0) <= -9.9

        dates = sorted(signals["date"].unique())
        if not dates:
            return self._empty_result()

        cfg = self.config
        cash = cfg.init_capital
        positions: Dict[str, int] = {}
        equity_records: List[Dict[str, Any]] = []
        trade_records: List[Dict[str, Any]] = []

        data_by_date = {dt: g.set_index("code") for dt, g in data.groupby("date")}
        signal_by_date = {dt: g for dt, g in signals.groupby("date")}

        for dt in dates:
            day_data = data_by_date.get(dt)
            day_signal = signal_by_date.get(dt)
            if day_data is None or day_data.empty or day_signal is None:
                market_value = self._market_value(positions, day_data)
                equity_records.append({
                    "date": dt, "equity": cash + market_value,
                    "cash": cash, "market_value": market_value,
                    "position_count": sum(1 for s in positions.values() if s > 0),
                })
                continue
            sell_records = self._execute_sells(
                dt, day_signal, day_data, positions, cash, cfg, trade_records
            )
            cash = sell_records["cash_after"]
            positions = sell_records["positions_after"]
            buy_records = self._execute_buys(
                dt, day_signal, day_data, positions, cash, cfg, trade_records
            )
            cash = buy_records["cash_after"]
            positions = buy_records["positions_after"]
            market_value = self._market_value(positions, day_data)
            equity_records.append({
                "date": dt, "equity": cash + market_value,
                "cash": cash, "market_value": market_value,
                "position_count": sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trade_records)
        if equity_curve.empty:
            return self._empty_result()
        eq_series = equity_curve.set_index("date")["equity"]
        metrics = self._calc_metrics(eq_series, trades_df)
        return BacktestResult(
            trades=trades_df,
            positions=pd.DataFrame(list(positions.items()), columns=["code", "shares"]),
            equity_curve=equity_curve,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # 卖出 (向量化)
    # ------------------------------------------------------------------
    def _execute_sells(
        self, dt, day_signal, day_data, positions, cash, cfg, trade_records
    ) -> Dict[str, Any]:
        sell_codes = day_signal.loc[day_signal["signal"] < 0, "code"].tolist()
        held_sell_codes = [c for c in sell_codes if positions.get(c, 0) > 0 and c in day_data.index]

        if not held_sell_codes:
            return {"cash_after": cash, "positions_after": positions}

        rows = day_data.loc[held_sell_codes]
        # 涨跌停限制：跌停日不能卖
        if cfg.price_limit and "is_limit_down" in rows.columns:
            blocked = rows.index[rows["is_limit_down"] == True].tolist()  # noqa
            held_sell_codes = [c for c in held_sell_codes if c not in blocked]
            rows = day_data.loc[held_sell_codes]
        if rows.empty:
            return {"cash_after": cash, "positions_after": positions}

        prices = rows["close"].values
        shares = np.array([positions[c] for c in rows.index], dtype=np.int64)
        amounts = prices * shares
        commissions = np.maximum(amounts * cfg.commission_rate, cfg.min_commission)
        taxes = amounts * cfg.stamp_tax_rate
        net_amounts = amounts - commissions - taxes
        cash_after = cash + net_amounts.sum()

        for i, code in enumerate(rows.index):
            trade_records.append({
                "date": dt, "code": code, "action": "sell",
                "price": float(prices[i]), "shares": int(shares[i]),
                "amount": float(amounts[i]),
                "commission": float(commissions[i]),
                "tax": float(taxes[i]),
                "pnl": float(net_amounts[i]),
            })
            positions[code] = 0
        return {"cash_after": cash_after, "positions_after": positions}

    # ------------------------------------------------------------------
    # 买入 (向量化)
    # ------------------------------------------------------------------
    def _execute_buys(
        self, dt, day_signal, day_data, positions, cash, cfg, trade_records
    ) -> Dict[str, Any]:
        buy_codes = day_signal.loc[day_signal["signal"] > 0, "code"].tolist()
        buy_codes = [c for c in buy_codes if c in day_data.index]
        if not buy_codes:
            return {"cash_after": cash, "positions_after": positions}

        rows = day_data.loc[buy_codes]
        # 涨跌停限制：涨停日不能买
        if cfg.price_limit and "is_limit_up" in rows.columns:
            blocked = rows.index[rows["is_limit_up"] == True].tolist()  # noqa
            buy_codes = [c for c in buy_codes if c not in blocked]
            rows = day_data.loc[buy_codes]
        if rows.empty:
            return {"cash_after": cash, "positions_after": positions}

        n = len(rows)
        # 与 native 一致：每只分配现金的 95% / n
        budget_per = cash * 0.95 / n
        prices_with_slippage = rows["close"].values * (1 + cfg.slippage)
        raw_shares = budget_per / prices_with_slippage
        # 整百股
        shares = (raw_shares // cfg.lot_size * cfg.lot_size).astype(np.int64)
        # 0 股过滤
        valid_mask = shares > 0
        rows = rows[valid_mask]
        shares = shares[valid_mask]
        if shares.sum() == 0:
            return {"cash_after": cash, "positions_after": positions}

        prices = rows["close"].values * (1 + cfg.slippage)
        amounts = prices * shares
        commissions = np.maximum(amounts * cfg.commission_rate, cfg.min_commission)
        costs = amounts + commissions
        # 资金不够就按比例削减（与 native 的 int(cash*0.98/price/100)*100 等价，但更严格）
        total_cost = costs.sum()
        if total_cost > cash:
            # 按比例缩小
            scale = (cash * 0.98) / total_cost
            shares = ((shares * scale) // cfg.lot_size * cfg.lot_size).astype(np.int64)
            valid_mask = shares > 0
            rows = rows[valid_mask]
            shares = shares[valid_mask]
            prices = rows["close"].values * (1 + cfg.slippage)
            amounts = prices * shares
            commissions = np.maximum(amounts * cfg.commission_rate, cfg.min_commission)
            costs = amounts + commissions

        cash_after = cash - costs.sum()
        for i, code in enumerate(rows.index):
            trade_records.append({
                "date": dt, "code": code, "action": "buy",
                "price": float(prices[i]), "shares": int(shares[i]),
                "amount": float(amounts[i]),
                "commission": float(commissions[i]),
                "tax": 0.0, "pnl": float(-(amounts[i] + commissions[i])),
            })
            positions[code] = positions.get(code, 0) + int(shares[i])
        return {"cash_after": cash_after, "positions_after": positions}

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _market_value(positions: Dict[str, int], day_data) -> float:
        if day_data is None or day_data.empty:
            return 0.0
        mv = 0.0
        for code, shares in positions.items():
            if shares > 0 and code in day_data.index:
                mv += shares * day_data.loc[code, "close"]
        return float(mv)

    def _calc_metrics(self, equity: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        # 始终返回交易相关指标，即使 equity 长度 < 2
        trade_metrics: Dict[str, Any] = {
            "n_trades": int(len(trades)),
            "n_buy": int((trades["action"] == "buy").sum()) if not trades.empty else 0,
            "n_sell": int((trades["action"] == "sell").sum()) if not trades.empty else 0,
        }
        if not trades.empty and "action" in trades.columns:
            sell_trades = trades[trades["action"] == "sell"]
            if not sell_trades.empty:
                trade_metrics["trade_win_rate"] = float((sell_trades["pnl"] > 0).mean())
            else:
                trade_metrics["trade_win_rate"] = 0.0
        else:
            trade_metrics["trade_win_rate"] = 0.0

        if equity.empty or len(equity) < 2:
            trade_metrics["final_equity"] = float(equity.iloc[-1]) if not equity.empty else 0.0
            return trade_metrics

        returns = equity.pct_change().dropna()
        if returns.empty:
            trade_metrics["final_equity"] = float(equity.iloc[-1])
            return trade_metrics

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        running_max = equity.cummax()
        max_drawdown = (equity / running_max - 1).min()
        sharpe = (annual_return - 0.03) / volatility if volatility != 0 else 0
        win_rate = (returns > 0).mean() if len(returns) > 0 else 0
        downside = returns[returns < 0]
        sortino = (annual_return - 0.03) / (downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            **trade_metrics,
            "final_equity": float(equity.iloc[-1]),
        }

    @staticmethod
    def _empty_result() -> BacktestResult:
        return BacktestResult(
            trades=pd.DataFrame(),
            positions=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            metrics={},
        )


# ======================================================================
# JIT 编译的热循环（核心加速点）
# ======================================================================

@njit(cache=True, fastmath=False)
def _simulate_core(
    n_dates, n_codes,
    sig_mat, close_for_jit, lu_mat, ld_mat, suspended,
    cash, positions_shares,
    commission_rate, min_commission,
    stamp_tax_rate, slippage, lot_size,
    price_limit,
    trade_codes, trade_actions, trade_dates_idx,
    trade_prices, trade_shares_arr, trade_amounts,
    trade_commissions, trade_taxes, trade_pnls,
    n_trades_out,
    equity_arr, cash_arr, mv_arr, pos_cnt_arr,
):
    """
    单趟扫描回测核心循环（与 native_adapter 功能等价）。

    状态:
        cash, positions_shares (length n_codes)

    信号:
        sig_mat[t, c]  in {-1, 0, 1}

    价格:
        close_for_jit[t, c]  (停牌时为 0)

    涨跌停:
        lu_mat[t, c], ld_mat[t, c]

    资金分配规则（与 native 一致）:
        - 卖出：先按 signal<0 且当前 holdings>0 执行，按 (close, shares) 算净额
        - 买入：signal>0 标的数 n_buy > 0 时，按 cash * 0.95 / n_buy 分配
        - 整百股取整
        - 跌停不能卖，涨停不能买
        - 现金不足时按比例缩股
    """
    for t in range(n_dates):
        # 1) 当日信号向量（连续扫描）
        # 1.1) 卖出
        for c in range(n_codes):
            sig = sig_mat[t, c]
            if sig < 0 and positions_shares[c] > 0:
                if suspended[t, c]:
                    continue
                if price_limit and ld_mat[t, c]:
                    continue
                price = close_for_jit[t, c]
                if price <= 0:
                    continue
                shares = positions_shares[c]
                amount = price * shares
                commission = amount * commission_rate
                if commission < min_commission:
                    commission = min_commission
                tax = amount * stamp_tax_rate
                net = amount - commission - tax
                cash += net
                # 记录
                idx = n_trades_out[0]
                trade_codes[idx] = c
                trade_actions[idx] = 1  # sell
                trade_dates_idx[idx] = t
                trade_prices[idx] = price
                trade_shares_arr[idx] = shares
                trade_amounts[idx] = amount
                trade_commissions[idx] = commission
                trade_taxes[idx] = tax
                trade_pnls[idx] = net
                n_trades_out[0] = idx + 1
                positions_shares[c] = 0

        # 1.2) 统计买入标的数 & 总预算
        n_buy = 0
        for c in range(n_codes):
            if sig_mat[t, c] > 0 and not suspended[t, c]:
                if price_limit and lu_mat[t, c]:
                    continue
                if close_for_jit[t, c] > 0:
                    n_buy += 1

        if n_buy > 0:
            budget_per = cash * 0.95 / n_buy
            # 第一遍：计算每只股票的目标份额
            target_shares = np.zeros(n_codes, dtype=np.int64)
            for c in range(n_codes):
                if sig_mat[t, c] > 0 and not suspended[t, c]:
                    if price_limit and lu_mat[t, c]:
                        continue
                    price = close_for_jit[t, c]
                    if price <= 0:
                        continue
                    buy_price = price * (1.0 + slippage)
                    raw = budget_per / buy_price
                    sh = (int(raw) // lot_size) * lot_size
                    target_shares[c] = sh
            # 算总成本，若超出 cash*0.98 则按比例缩
            total_cost = 0.0
            for c in range(n_codes):
                if target_shares[c] > 0:
                    price = close_for_jit[t, c] * (1.0 + slippage)
                    amount = price * target_shares[c]
                    commission = amount * commission_rate
                    if commission < min_commission:
                        commission = min_commission
                    total_cost += amount + commission
            if total_cost > cash * 0.98 and total_cost > 0:
                scale = (cash * 0.98) / total_cost
                for c in range(n_codes):
                    new_sh = int(target_shares[c] * scale)
                    new_sh = (new_sh // lot_size) * lot_size
                    target_shares[c] = new_sh
                # 重算 total_cost
                total_cost = 0.0
                for c in range(n_codes):
                    if target_shares[c] > 0:
                        price = close_for_jit[t, c] * (1.0 + slippage)
                        amount = price * target_shares[c]
                        commission = amount * commission_rate
                        if commission < min_commission:
                            commission = min_commission
                        total_cost += amount + commission
            # 实际执行
            for c in range(n_codes):
                if target_shares[c] > 0:
                    price = close_for_jit[t, c] * (1.0 + slippage)
                    amount = price * target_shares[c]
                    commission = amount * commission_rate
                    if commission < min_commission:
                        commission = min_commission
                    cost = amount + commission
                    if cost > cash:
                        # 极端情况：仍超现金，则跳过该标的
                        target_shares[c] = 0
                        continue
                    cash -= cost
                    positions_shares[c] += target_shares[c]
                    idx = n_trades_out[0]
                    trade_codes[idx] = c
                    trade_actions[idx] = 0  # buy
                    trade_dates_idx[idx] = t
                    trade_prices[idx] = price
                    trade_shares_arr[idx] = target_shares[c]
                    trade_amounts[idx] = amount
                    trade_commissions[idx] = commission
                    trade_taxes[idx] = 0.0
                    trade_pnls[idx] = -cost
                    n_trades_out[0] = idx + 1

        # 2) 当日权益
        mv = 0.0
        pos_cnt = 0
        for c in range(n_codes):
            sh = positions_shares[c]
            if sh > 0:
                pos_cnt += 1
                price = close_for_jit[t, c]
                if price > 0:
                    mv += sh * price
        equity_arr[t] = cash + mv
        cash_arr[t] = cash
        mv_arr[t] = mv
        pos_cnt_arr[t] = pos_cnt

    return cash