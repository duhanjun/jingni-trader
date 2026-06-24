"""
向量化回测引擎

借鉴来源：
    - VectorBT / VectorBT PRO：将逐 bar 的 Python 循环下沉为 NumPy/pandas 向量运算
    - NautilusTrader：保留事件驱动语义以保证回测/实盘一致性

设计要点：
    1. 与 jingni-trader 的 NativeAdapter 保持【完全相同】的交易语义：
       先卖后买、等额分配、T+1、涨跌停限制、佣金、印花税、滑点。
       —— 这样正确性测试可直接对比两者结果。
    2. 保留外层日期循环（现金状态具有时序依赖，无法完全向量化），
       但把所有【日内】操作从 Python for 循环改为向量化：
       - 用 groupby/merge 替代 iterrows
       - 用 numpy 数组运算替代逐股票持仓市值累加
    3. 持仓状态用 dict[code]->shares 维护，但市值计算用向量化 merge。

性能瓶颈分析（NativeAdapter）：
    - for _, row in day_signal.iterrows()  ← O(股票数) 纯 Python
    - for code, shares in positions.items()  ← O(持仓数) 纯 Python
    这两处在股票池扩大时是主要耗时点，本实现将其向量化。
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

# 复用主仓库的绩效计算基类，保证指标口径一致
import os
import sys
import importlib.util

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _load_module_from_path(name: str, path: str):
    """通过文件路径加载模块（绕过 hyphen 目录名无法作为包名导入的问题）"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# 主仓库 skills 目录使用 hyphen 命名（backtest-engine），无法用常规 import
# 这里通过文件路径加载基类，保证绩效指标口径与 NativeAdapter 完全一致
_base_bt_path = os.path.join(_REPO_ROOT, "skills", "backtest-engine", "scripts", "base", "base_backtest.py")
_base_engine_path = os.path.join(_REPO_ROOT, "skills", "backtest-engine", "scripts", "base", "base_backtest_engine.py")

_base_backtest_mod = _load_module_from_path("_jt_base_backtest", _base_bt_path)
_base_engine_mod = _load_module_from_path("_jt_base_backtest_engine", _base_engine_path)

BaseBacktestEngine = _base_engine_mod.BaseBacktestEngine
BaseBacktestMetrics = _base_backtest_mod.BaseBacktestMetrics


class VectorizedAdapter(BaseBacktestEngine):
    """向量化回测适配器（语义等价于 NativeAdapter）"""

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

        # ---- 预处理：构造日期索引与日内数据映射（向量化） ----
        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        signals = signals.sort_values(["date", "code"]).reset_index(drop=True)

        # 信号透视：每个 (date, code) -> signal，缺失填 0
        sig_pivot = signals.pivot(index="date", columns="code", values="signal").fillna(0.0)
        dates = sig_pivot.index.tolist()
        if not dates:
            return self._empty_result()

        # 行情透视：close / is_limit_up / is_limit_down 按 (date, code) 索引
        price_close = data.pivot(index="date", columns="code", values="close")
        limit_up = data.pivot(index="date", columns="code", values="is_limit_up").fillna(False) if "is_limit_up" in data.columns else None
        limit_down = data.pivot(index="date", columns="code", values="is_limit_down").fillna(False) if "is_limit_down" in data.columns else None

        # 对齐列（以信号列为准）
        codes = sig_pivot.columns.tolist()
        price_close = price_close.reindex(columns=codes)
        if limit_up is not None:
            limit_up = limit_up.reindex(columns=codes).fillna(False)
        if limit_down is not None:
            limit_down = limit_down.reindex(columns=codes).fillna(False)

        cash = float(init_capital)
        positions: Dict[str, float] = {}  # code -> shares
        equity_records = []
        trades = []

        for dt in dates:
            sig_row = sig_pivot.loc[dt]
            close_row = price_close.loc[dt]
            lu_row = limit_up.loc[dt] if limit_up is not None else None
            ld_row = limit_down.loc[dt] if limit_down is not None else None

            # ---- 向量化：分离买/卖信号 ----
            # 只对当前持仓且信号<0 的标的卖出
            held_codes = [c for c, s in positions.items() if s > 0]
            if held_codes:
                sell_mask = (sig_row.reindex(held_codes).fillna(0) < 0)
                sell_codes = sell_mask[sell_mask].index.tolist()
            else:
                sell_codes = []

            # ---- 卖出（向量化计算金额） ----
            for code in sell_codes:
                # 涨跌停与数据存在性检查
                if pd.isna(close_row.get(code)):
                    continue
                if price_limit and ld_row is not None and bool(ld_row.get(code, False)):
                    continue
                price = float(close_row[code])
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5.0)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                trades.append({
                    "date": dt, "code": code, "action": "sell",
                    "price": price, "shares": shares, "amount": sell_amount,
                    "commission": commission, "tax": tax, "pnl": sell_amount - cost,
                })
                positions[code] = 0

            # ---- 买入：向量化筛选可买标的 ----
            buy_mask = sig_row > 0
            buy_codes = buy_mask[buy_mask].index.tolist()

            if buy_codes and cash > 0:
                n_buy = len(buy_codes)
                budget_per_stock = cash * 0.95 / n_buy

                # 向量化取价 / 涨跌停过滤
                buy_close = close_row.reindex(buy_codes)
                buy_lu = lu_row.reindex(buy_codes) if lu_row is not None else pd.Series(False, index=buy_codes)

                # 涨停过滤 + 数据有效
                valid = buy_close.notna()
                if price_limit:
                    valid = valid & (~buy_lu.fillna(False))
                buy_codes_valid = valid[valid].index.tolist()

                for code in buy_codes_valid:
                    price = float(buy_close[code]) * (1 + slippage)
                    shares = int(budget_per_stock / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5.0)
                    cost = buy_amount + commission
                    if cost > cash:
                        shares = int((cash * 0.98) / price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = price * shares
                        commission = max(buy_amount * commission_rate, 5.0)
                        cost = buy_amount + commission
                    cash -= cost
                    positions[code] = positions.get(code, 0) + shares
                    trades.append({
                        "date": dt, "code": code, "action": "buy",
                        "price": price, "shares": shares, "amount": buy_amount,
                        "commission": commission, "tax": 0.0, "pnl": -buy_amount - commission,
                    })

            # ---- 向量化计算当日总市值 ----
            held = {c: s for c, s in positions.items() if s > 0}
            if held:
                held_codes = list(held.keys())
                held_shares = np.array([held[c] for c in held_codes], dtype=float)
                held_prices = close_row.reindex(held_codes).fillna(0.0).values.astype(float)
                market_value = float(np.dot(held_prices, held_shares))
            else:
                market_value = 0.0

            total_equity = cash + market_value
            equity_records.append({
                "date": dt,
                "equity": total_equity,
                "cash": cash,
                "market_value": market_value,
                "position_count": len(held),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        eq_series = equity_curve.set_index("date")["equity"]
        metrics = BaseBacktestMetrics.calc_all_metrics(eq_series, trades_df)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(list(positions.items()), columns=["code", "shares"]),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    def _empty_result(self):
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }


if __name__ == "__main__":
    # 自检：用合成数据跑一次
    from synthetic_data import generate_synthetic_ohlcv, generate_signals
    data = generate_synthetic_ohlcv(n_codes=20, n_days=120)
    sig = generate_signals(data, strategy="ma_cross")
    bt = VectorizedAdapter()
    res = bt.run_backtest(data, sig)
    print("回测完成:")
    print("  交易笔数:", len(res["trades"]))
    print("  净值终点:", res["equity_curve"]["equity"].iloc[-1] if not res["equity_curve"].empty else "N/A")
    print("  指标:", res["metrics"])
