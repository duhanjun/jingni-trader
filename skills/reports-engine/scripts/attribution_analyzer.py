"""
绩效归因分析器

从 execution-monitor-engine 的 ledger.jsonl / trade_log.json 产物中
提取成交记录，执行 round-trip 归组、盈亏归因、执行质量分析。
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger("attribution-analyzer")


# ============================================================================
# RoundTrip 数据类
# ============================================================================

@dataclass
class RoundTrip:
    """一次完整的买卖闭环"""
    code: str = ""
    buy_date: str = ""
    sell_date: str = ""
    buy_price: float = 0.0
    sell_price: float = 0.0
    shares: int = 0
    gross_pnl: float = 0.0          # 毛盈亏（不含费用）
    commission: float = 0.0          # 总佣金
    stamp_tax: float = 0.0           # 总印花税
    slippage_cost: float = 0.0       # 总滑点成本
    net_pnl: float = 0.0             # 净盈亏
    return_pct: float = 0.0          # 收益率
    holding_days: int = 0            # 持仓天数
    is_win: bool = False             # 是否盈利


# ============================================================================
# AttributionAnalyzer
# ============================================================================

class AttributionAnalyzer:
    """绩效归因分析器

    从 ledger.jsonl 加载成交记录，通过 FIFO 归组构建 round-trip，
    提供交易统计、按标的归因、执行质量分析、净值序列、压力期表现等。

    使用方式::

        analyzer = AttributionAnalyzer(ledger_path)
        analyzer.load()
        analyzer.build_round_trips()
        stats = analyzer.get_round_trip_stats()
    """

    def __init__(
        self,
        ledger_path: str,
        trade_log_path: Optional[str] = None,
        init_capital: float = 1_000_000.0,
    ):
        self.ledger_path = Path(ledger_path)
        self.trade_log_path = Path(trade_log_path) if trade_log_path else None
        self.init_capital = init_capital
        self.records: List[dict] = []
        self.round_trips: List[RoundTrip] = []
        self._nav_series: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """加载 ledger.jsonl，解析为 records 列表。

        损坏行自动跳过（与 paper_ledger.read_paper_trades 行为一致）。
        返回 True 表示成功加载到至少一条记录。
        """
        if not self.ledger_path.exists():
            logger.warning(f"ledger 文件不存在: {self.ledger_path}")
            return False

        records: List[dict] = []
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(
                        f"ledger 损坏行已跳过: {self.ledger_path}:{lineno} - {e}"
                    )
                    continue

        self.records = sorted(records, key=lambda r: r.get("trade_date", ""))
        return len(self.records) > 0

    # ------------------------------------------------------------------
    # Round-Trip 归组（FIFO 匹配）
    # ------------------------------------------------------------------

    def build_round_trips(self) -> List[RoundTrip]:
        """将零散成交记录归组为完整买卖闭环。

        按标的分组后按时间排序，使用 FIFO 匹配买入和卖出。
        支持部分卖出（一次卖出匹配多笔买入，或一次买入分多次卖出）。
        """
        from collections import defaultdict, deque

        by_code: Dict[str, list] = defaultdict(list)
        for rec in self.records:
            if rec.get("confirmed", True):
                code = rec.get("code", "")
                if code:
                    by_code[code].append(rec)

        round_trips: List[RoundTrip] = []

        for code, trades in by_code.items():
            trades.sort(key=lambda t: t.get("trade_date", ""))
            buy_queue: deque = deque()  # (buy_record, remaining_shares)

            for trade in trades:
                if trade.get("side") == "buy":
                    buy_queue.append((trade, trade.get("shares", 0)))
                    continue

                # 卖出：按 FIFO 匹配买入
                sell_shares = trade.get("shares", 0)
                while sell_shares > 0 and buy_queue:
                    buy_rec, buy_shares = buy_queue[0]
                    matched = min(buy_shares, sell_shares)
                    if matched <= 0:
                        buy_queue.popleft()
                        continue

                    buy_price = buy_rec.get("price", 0)
                    sell_price = trade.get("price", 0)
                    gross_pnl = (sell_price - buy_price) * matched

                    # 费用按比例分摊
                    buy_total_shares = buy_rec.get("shares", 1) or 1
                    sell_total_shares = trade.get("shares", 1) or 1
                    buy_comm = buy_rec.get("commission", 0) * (matched / buy_total_shares)
                    buy_slip = buy_rec.get("slippage_cost", 0) * (matched / buy_total_shares)
                    sell_comm = trade.get("commission", 0) * (matched / sell_total_shares)
                    sell_stamp = trade.get("stamp_tax", 0) * (matched / sell_total_shares)
                    sell_slip = trade.get("slippage_cost", 0) * (matched / sell_total_shares)

                    total_comm = buy_comm + sell_comm
                    total_stamp = sell_stamp
                    total_slip = buy_slip + sell_slip
                    net_pnl = gross_pnl - total_comm - total_stamp - total_slip

                    buy_date = buy_rec.get("trade_date", "")
                    sell_date = trade.get("trade_date", "")
                    holding_days = 1
                    if buy_date and sell_date:
                        try:
                            holding_days = max(
                                (
                                    datetime.strptime(sell_date, "%Y-%m-%d") -
                                    datetime.strptime(buy_date, "%Y-%m-%d")
                                ).days,
                                1,
                            )
                        except (ValueError, TypeError):
                            holding_days = 1

                    cost_basis = buy_price * matched
                    return_pct = (net_pnl / cost_basis) if cost_basis > 0 else 0.0

                    rt = RoundTrip(
                        code=code,
                        buy_date=buy_date,
                        sell_date=sell_date,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        shares=matched,
                        gross_pnl=gross_pnl,
                        commission=total_comm,
                        stamp_tax=total_stamp,
                        slippage_cost=total_slip,
                        net_pnl=net_pnl,
                        return_pct=return_pct,
                        holding_days=holding_days,
                        is_win=net_pnl > 0,
                    )
                    round_trips.append(rt)

                    # 更新队列
                    buy_shares -= matched
                    sell_shares -= matched
                    if buy_shares <= 0:
                        buy_queue.popleft()
                    else:
                        buy_queue[0] = (buy_rec, buy_shares)

        self.round_trips = round_trips
        return round_trips

    # ------------------------------------------------------------------
    # 交易统计概览
    # ------------------------------------------------------------------

    def get_transaction_stats(self) -> dict:
        """交易统计概览：总成交笔数、买卖比、佣金/印花税/滑点汇总。"""
        if not self.records:
            return {}

        buys = [r for r in self.records if r.get("side") == "buy"]
        sells = [r for r in self.records if r.get("side") == "sell"]
        total_commission = sum(r.get("commission", 0) for r in self.records)
        total_stamp_tax = sum(r.get("stamp_tax", 0) for r in self.records)
        total_slippage = sum(r.get("slippage_cost", 0) for r in self.records)

        return {
            "total_trades": len(self.records),
            "total_buys": len(buys),
            "total_sells": len(sells),
            "total_commission": round(total_commission, 2),
            "total_stamp_tax": round(total_stamp_tax, 2),
            "total_slippage": round(total_slippage, 2),
            "total_cost": round(total_commission + total_stamp_tax + total_slippage, 2),
            "unique_stocks": len(set(r.get("code", "") for r in self.records if r.get("code"))),
        }

    # ------------------------------------------------------------------
    # Round-Trip 统计
    # ------------------------------------------------------------------

    def get_round_trip_stats(self) -> dict:
        """Round-trip 统计：胜率、盈亏比、平均持仓天数等。"""
        if not self.round_trips:
            return {}

        net_pnls = [rt.net_pnl for rt in self.round_trips]
        returns = [rt.return_pct for rt in self.round_trips]
        holding_days = [rt.holding_days for rt in self.round_trips]
        wins = [rt for rt in self.round_trips if rt.is_win]
        losses = [rt for rt in self.round_trips if not rt.is_win]

        total_net_pnl = sum(net_pnls)
        avg_win = np.mean([w.net_pnl for w in wins]) if wins else 0
        avg_loss = np.mean([l.net_pnl for l in losses]) if losses else 0

        gross_profit = sum(w.net_pnl for w in wins)
        gross_loss = abs(sum(l.net_pnl for l in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        return {
            "total_round_trips": len(self.round_trips),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / len(self.round_trips), 4) if self.round_trips else 0,
            "total_net_pnl": round(total_net_pnl, 2),
            "avg_return_pct": round(np.mean(returns) * 100, 2) if returns else 0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_holding_days": round(np.mean(holding_days), 1) if holding_days else 0,
        }

    # ------------------------------------------------------------------
    # 按标的归因
    # ------------------------------------------------------------------

    def get_pnl_by_stock(self) -> pd.DataFrame:
        """按标的归因：每只股票的总盈亏、交易次数、胜率、平均收益率。"""
        if not self.round_trips:
            return pd.DataFrame()

        data = []
        for code in sorted(set(rt.code for rt in self.round_trips)):
            stock_rts = [rt for rt in self.round_trips if rt.code == code]
            total_pnl = sum(rt.net_pnl for rt in stock_rts)
            wins = sum(1 for rt in stock_rts if rt.is_win)
            data.append({
                "code": code,
                "total_pnl": round(total_pnl, 2),
                "trade_count": len(stock_rts),
                "win_count": wins,
                "win_rate": round(wins / len(stock_rts) * 100, 1) if stock_rts else 0,
                "avg_return_pct": round(np.mean([rt.return_pct for rt in stock_rts]) * 100, 2),
            })

        df = pd.DataFrame(data)
        return df.sort_values("total_pnl", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 执行质量分析
    # ------------------------------------------------------------------

    def get_execution_quality(self) -> dict:
        """执行质量分析：费用占比(bps)、滑点占比(bps)、成交规模。"""
        if not self.records:
            return {}

        total_cost = sum(
            r.get("commission", 0) + r.get("stamp_tax", 0) + r.get("slippage_cost", 0)
            for r in self.records
        )
        total_turnover = sum(
            r.get("price", 0) * r.get("shares", 0) for r in self.records
        )
        total_slippage = sum(r.get("slippage_cost", 0) for r in self.records)

        all_volumes = [r.get("price", 0) * r.get("shares", 0) for r in self.records]

        return {
            "total_turnover": round(total_turnover, 2),
            "total_cost": round(total_cost, 2),
            "cost_ratio_bps": round(total_cost / total_turnover * 10000, 2) if total_turnover > 0 else 0,
            "avg_trade_size": round(np.mean(all_volumes), 2) if all_volumes else 0,
            "max_trade_size": round(max(all_volumes), 2) if all_volumes else 0,
            "slippage_ratio_bps": round(total_slippage / total_turnover * 10000, 2) if total_turnover > 0 else 0,
        }

    # ------------------------------------------------------------------
    # 净值序列
    # ------------------------------------------------------------------

    def get_nav_series(self) -> pd.Series:
        """从 ledger 提取每日净值序列（按日期取最后一条记录的 nav_after）。"""
        if self._nav_series is not None:
            return self._nav_series

        if not self.records:
            return pd.Series(dtype=float)

        nav_by_date: Dict[str, float] = {}
        for rec in self.records:
            date_str = rec.get("trade_date", "")
            nav = rec.get("nav_after")
            if date_str and nav is not None:
                nav_by_date[date_str] = float(nav)

        if not nav_by_date:
            return pd.Series(dtype=float)

        df = pd.DataFrame({
            "date": list(nav_by_date.keys()),
            "nav": list(nav_by_date.values()),
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        self._nav_series = df["nav"]
        return self._nav_series

    # ------------------------------------------------------------------
    # A 股压力期表现
    # ------------------------------------------------------------------

    def get_stress_period_performance(self) -> dict:
        """A 股特定压力期表现（若交易记录覆盖对应时间段）。"""
        stress_periods = {
            "2015年股灾": ("2015-06-12", "2015-08-26"),
            "2016年熔断": ("2016-01-04", "2016-01-28"),
            "2018年熊市": ("2018-01-24", "2018-10-18"),
            "2020年疫情": ("2020-01-20", "2020-03-23"),
            "2024年初下跌": ("2024-01-02", "2024-02-05"),
        }

        nav = self.get_nav_series()
        if nav.empty:
            return {}

        results = {}
        for name, (start, end) in stress_periods.items():
            try:
                period_nav = nav.loc[start:end]
            except (KeyError, TypeError):
                continue
            if len(period_nav) < 2:
                continue
            first_val = float(period_nav.iloc[0])
            last_val = float(period_nav.iloc[-1])
            if first_val <= 0:
                continue
            period_return = last_val / first_val - 1
            period_mdd = float((period_nav / period_nav.cummax() - 1).min())
            results[name] = {
                "return_pct": round(period_return * 100, 2),
                "max_drawdown_pct": round(period_mdd * 100, 2),
            }

        return results

    # ------------------------------------------------------------------
    # 连胜/连败统计
    # ------------------------------------------------------------------

    def get_consecutive_stats(self) -> dict:
        """最大连胜/连败次数。"""
        if not self.round_trips:
            return {}

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for rt in self.round_trips:
            if rt.is_win:
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }
