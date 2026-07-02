"""
Vectorized Backtest Engine (借鉴 VectorBT 设计)

jingni-trader 当前 native_adapter 是逐日 Python 循环 (O(days * stocks) 慢),
本模块给出向量化版本, 借鉴 VectorBT 的 Portfolio.from_orders 思路:
- 把所有交易日的 target_weights 一次性组成矩阵
- 一次性矩阵运算得到权益曲线, 无 Python 循环

设计取舍:
1. 纯 NumPy/Pandas 实现, 不依赖 numba/cython, 保持低门槛
2. 支持 A 股 T+1 (信号 t 日产生, 成交 t+1 日 open)
3. 涨跌停停买停卖
4. 简化交易成本模型 (单边佣金 + 卖出印花税)
5. 接口形态与 BaseBacktestEngine.run_backtest 对齐
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VectorizedBacktestResult:
    """向量化回测结果"""
    equity_curve: pd.DataFrame          # date, equity, cash, market_value, position_count
    trades: pd.DataFrame                 # 成交明细
    positions_daily: pd.DataFrame        # 每日每只股票持仓股数
    weights_daily: pd.DataFrame          # 每日每只股票的目标权重
    metrics: Dict[str, Any] = field(default_factory=dict)


class VectorizedBacktestEngine:
    """
    向量化回测引擎
    输入:
        data: 行情 (MultiIndex: code, date) 包含 open, high, low, close, is_limit_up, is_limit_down
        target_weights: 目标权重矩阵 (date x code), 0~1 之间的权重, sum(per day) <= 1
        init_capital: 初始资金
        commission_rate: 单边佣金费率
        stamp_tax_rate: 卖出印花税
        t_plus_1: 是否 T+1 交易 (信号 t 日产生, t+1 open 价成交)
        lot_size: 最小成交单位(A股=100)
        max_position_pct: 个股权重上限

    借鉴 VectorBT:
    - 输入权重矩阵, 而非 buy/sell 信号, 计算更简洁
    - cash 跨日复利, 不逐日模拟订单簿
    - 所有计算都用矩阵运算
    """

    def __init__(self,
                 init_capital: float = 1_000_000.0,
                 commission_rate: float = 0.0003,
                 stamp_tax_rate: float = 0.001,
                 t_plus_1: bool = True,
                 lot_size: int = 100,
                 max_position_pct: float = 0.10):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.t_plus_1 = t_plus_1
        self.lot_size = lot_size
        self.max_position_pct = max_position_pct

    def run(self,
            data: pd.DataFrame,
            target_weights: pd.DataFrame) -> VectorizedBacktestResult:
        """
        执行向量化回测

        参数:
            data: DataFrame, 必含列 [code, date, open, high, low, close, is_limit_up, is_limit_down]
                  通常是 long format (一行一只股票一天)
            target_weights: 目标权重 DataFrame, index=date, columns=code, 0~1 之间
                            sum per row <= 1, NaN 表示无目标
        """
        # ---------- 1. 准备 wide 数据 ----------
        dates = sorted(target_weights.index.tolist())
        codes = sorted(target_weights.columns.tolist())
        T, N = len(dates), len(codes)

        # close 矩阵 (T x N)
        close = (data.pivot(index="date", columns="code", values="close")
                      .reindex(index=dates, columns=codes))
        open_ = (data.pivot(index="date", columns="code", values="open")
                      .reindex(index=dates, columns=codes))
        is_limit_up = (data.pivot(index="date", columns="code", values="is_limit_up")
                            .reindex(index=dates, columns=codes).fillna(False).astype(bool))
        is_limit_down = (data.pivot(index="date", columns="code", values="is_limit_down")
                              .reindex(index=dates, columns=codes).fillna(False).astype(bool))

        # ---------- 2. 目标权重处理 ----------
        w = target_weights.reindex(index=dates, columns=codes).copy()
        w = w.clip(lower=0, upper=self.max_position_pct)
        # 归一化: sum<=1, 留 5% 现金 buffer
        row_sum = w.sum(axis=1).replace(0, np.nan)
        w = w.div(row_sum, axis=0).fillna(0) * 0.95

        # T+1: 信号的成交价为 t+1 的 open
        if self.t_plus_1:
            exec_price = open_.shift(-1)
        else:
            exec_price = open_

        # 涨跌停不可成交: 把对应日期的信号清零
        if self.t_plus_1:
            # 触发日 = 信号发出日 t, 实际执行日 t+1
            blocked = is_limit_up.shift(-1)  # 涨停日次日不能再以开盘成交 (开盘涨停)
            blocked = blocked.fillna(False)
        else:
            blocked = is_limit_up.copy()

        # 限制可买入的股票 (用前一天的 close 是否涨停判断次日是否可买)
        can_buy = (~blocked).astype(float)
        w = w * can_buy

        # 重新归一化
        row_sum = w.sum(axis=1).replace(0, np.nan)
        w = w.div(row_sum, axis=0).fillna(0) * 0.95

        # ---------- 3. 计算每日仓位变化 ----------
        # 目标金额 = 昨日总资产 * 今日目标权重
        cash_now = float(self.init_capital)
        cash = np.zeros(T + 1)
        cash[0] = self.init_capital
        equity = np.zeros(T + 1)
        equity[0] = self.init_capital
        positions = np.zeros((T, N), dtype=int)  # 持仓股数

        trades_records: List[Dict[str, Any]] = []

        for t in range(T):
            # 评估 t 日权益: 现金 + 持仓按 close 估值
            if t == 0:
                equity_t = cash_now
            else:
                pos_value = np.nansum(positions[t - 1] * close.iloc[t].fillna(0).values)
                equity_t = cash_now + pos_value
            equity[t] = equity_t

            if t >= T - 1:
                # 最后一天只能平仓, 不建仓
                for j in range(N):
                    held = positions[t - 1, j] if t > 0 else 0
                    if held <= 0:
                        continue
                    price = close.iloc[t, j]
                    if np.isnan(price):
                        continue
                    sell_amount = price * held
                    commission = max(sell_amount * self.commission_rate, 5)
                    tax = sell_amount * self.stamp_tax_rate
                    cash_now += sell_amount - commission - tax
                    trades_records.append({
                        "date": dates[t], "code": codes[j], "action": "sell",
                        "price": float(price), "shares": int(held),
                        "amount": float(sell_amount), "commission": float(commission),
                        "tax": float(tax),
                    })
                    positions[t, j] = 0
                cash[t] = cash_now
                cash[t + 1] = cash_now
                equity[T] = cash_now  # 最终权益
                continue

            # 目标持仓金额
            target_value = equity_t * w.iloc[t].values
            target_value = np.nan_to_num(target_value, nan=0.0)

            # 当前持仓金额 (按 t 日 close 估值)
            if t == 0:
                cur_value = np.zeros(N)
            else:
                cur_value = positions[t - 1] * close.iloc[t].fillna(0).values

            # delta
            delta_val = target_value - cur_value
            buy_mask = delta_val > 0
            sell_mask = delta_val < 0

            # 卖出: 不能在跌停日卖出
            can_sell_mask = ~is_limit_down.iloc[t].fillna(False).values
            sell_mask = sell_mask & can_sell_mask

            # 卖出可执行金额: 取 max(目标值, 0) 与 delta 绝对值 的较小值 (不能卖空)
            sell_amt = np.where(sell_mask, -delta_val, 0)
            sell_amt = np.minimum(sell_amt, np.where(cur_value > 0, cur_value, 0))

            # 按 t+1 开盘价成交 (T+1)
            if self.t_plus_1 and t + 1 < T:
                exec_open = open_.iloc[t + 1].values
                # 用 close 兜底 open 缺失
                exec_open = np.where(np.isnan(exec_open), close.iloc[t + 1].values, exec_open)
            else:
                exec_open = open_.iloc[t].values
                exec_open = np.where(np.isnan(exec_open), close.iloc[t].values, exec_open)

            # 卖出股数 (向下取整到 lot_size)
            sell_shares = np.where(exec_open > 0, np.floor(sell_amt / exec_open / self.lot_size) * self.lot_size, 0).astype(int)
            if t > 0:
                sell_shares = np.minimum(sell_shares, positions[t - 1])
            else:
                sell_shares = np.minimum(sell_shares, 0)

            # 卖出现金回款
            sell_gross = sell_shares * exec_open
            sell_commission = np.maximum(sell_gross * self.commission_rate, np.where(sell_gross > 0, 5, 0))
            sell_tax = sell_gross * self.stamp_tax_rate
            cash_proceeds = sell_gross - sell_commission - sell_tax
            cash_now += float(np.nansum(cash_proceeds))

            # 记录卖出成交
            for j in np.where(sell_shares > 0)[0]:
                trades_records.append({
                    "date": dates[t + 1] if self.t_plus_1 else dates[t],
                    "code": codes[j], "action": "sell",
                    "price": float(exec_open[j]), "shares": int(sell_shares[j]),
                    "amount": float(sell_gross[j]), "commission": float(sell_commission[j]),
                    "tax": float(sell_tax[j]),
                })

            # 买入股数
            buy_amt = np.where(buy_mask, delta_val, 0)
            buy_amt_after_fee = buy_amt * (1 - self.commission_rate)
            buy_shares = np.where(exec_open > 0,
                                   np.floor(buy_amt_after_fee / exec_open / self.lot_size) * self.lot_size,
                                   0).astype(int)
            buy_cost = buy_shares * exec_open
            buy_commission = np.maximum(buy_cost * self.commission_rate, np.where(buy_cost > 0, 5, 0))
            total_buy_cost = buy_cost + buy_commission
            # 现金约束
            if total_buy_cost.sum() > cash_now and total_buy_cost.sum() > 0:
                scale = cash_now / total_buy_cost.sum() * 0.98
                buy_shares = np.floor(buy_shares * scale / self.lot_size) * self.lot_size
                buy_shares = buy_shares.astype(int)
                buy_cost = buy_shares * exec_open
                buy_commission = np.maximum(buy_cost * self.commission_rate, np.where(buy_cost > 0, 5, 0))
                total_buy_cost = buy_cost + buy_commission

            cash_now -= float(np.nansum(total_buy_cost))

            for j in np.where(buy_shares > 0)[0]:
                trades_records.append({
                    "date": dates[t + 1] if self.t_plus_1 else dates[t],
                    "code": codes[j], "action": "buy",
                    "price": float(exec_open[j]), "shares": int(buy_shares[j]),
                    "amount": float(buy_cost[j]), "commission": float(buy_commission[j]),
                    "tax": 0.0,
                })

            # 更新持仓
            base_shares = positions[t - 1] if t > 0 else np.zeros(N)
            new_shares = base_shares - sell_shares + buy_shares
            new_shares = np.maximum(new_shares, 0)
            positions[t] = new_shares

            # 记录当日现金
            cash[t] = cash_now

        # 最后一帧权益
        cash[T] = cash_now
        if T > 0:
            last_pos_value = np.nansum(positions[T - 1] * close.iloc[T - 1].fillna(0).values)
        else:
            last_pos_value = 0.0
        equity[T] = cash_now + last_pos_value

        # ---------- 4. 整理输出 ----------
        equity_curve = pd.DataFrame({
            "date": list(dates) + [pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)],
            "equity": equity,
            "cash": cash,
        })
        # 保证 date 唯一
        equity_curve = equity_curve.drop_duplicates(subset="date", keep="last").reset_index(drop=True)

        positions_daily = pd.DataFrame(positions, index=dates, columns=codes)
        positions_daily.index.name = "date"

        trades_df = pd.DataFrame(trades_records)

        return VectorizedBacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            positions_daily=positions_daily,
            weights_daily=w,
            metrics={},
        )


# ---------------------------------------------------------------------------
# 兼容 jingni-trader BaseBacktestEngine 接口的适配器
# ---------------------------------------------------------------------------
def run_vectorized_adapter(
    data: pd.DataFrame,
    target_weights: pd.DataFrame,
    init_capital: float = 1_000_000.0,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    t_plus_1: bool = True,
) -> Dict[str, Any]:
    """
    与 backtest-engine/scripts/adapters/native_adapter.run_backtest 接口形态一致的便捷函数
    """
    eng = VectorizedBacktestEngine(
        init_capital=init_capital,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        t_plus_1=t_plus_1,
    )
    res = eng.run(data, target_weights)
    return {
        "trades": res.trades,
        "positions": res.positions_daily.reset_index().melt(id_vars="date", var_name="code", value_name="shares"),
        "equity_curve": res.equity_curve,
        "metrics": res.metrics,
        "report_path": "",
    }


__all__ = [
    "VectorizedBacktestEngine", "VectorizedBacktestResult", "run_vectorized_adapter",
]
