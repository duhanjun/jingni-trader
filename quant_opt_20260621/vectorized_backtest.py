"""
向量化回测引擎（Vectorized Backtest Adapter）

借鉴来源：
- VectorBT (https://vectorbt.dev) ：将策略 / 持仓 / 现金表示为多维数组，
  用 numpy 向量化代替 Python 逐 bar 循环，性能提升 100-1000x。
- jingni-trader 既有 native_adapter.py：保留 A 股 T+1、涨跌停、印花税等
  市场规则，但用矩阵运算替换 pandas iterrows / DataFrame 过滤。

核心优化点（对照 native_adapter.py）：
1. 预先把 data / signals pivot 为 (date × code) 矩阵，避免每个交易日
   重复执行 `df[df['date']==dt]` 这种 O(n) 全表扫描。
2. positions / cash 用 numpy 数组维护，按 code 索引；买卖预算分配、
   涨跌停过滤、市值计算全部向量化。
3. 保留 T+1（买入当日不可卖）与涨跌停限制，确保结果与 native 一致。

兼容性：实现 BaseBacktestEngine.run_backtest 同名方法，返回结构一致，
可作为 backtest-engine 的新 adapter 注册到 config.BACKTEST_BACKEND。
"""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import pandas as pd


class VectorizedBacktestAdapter:
    """
    向量化回测适配器

    与 native_adapter.NativeAdapter 接口完全一致，但内部使用 numpy
    矩阵运算，避免 Python 逐行循环。
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

        # ---- 1. 预处理：构建 (date × code) 矩阵 ----
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        # 统一 date 类型，避免 datetime vs str 比较出错
        data['date'] = pd.to_datetime(data['date'])
        signals['date'] = pd.to_datetime(signals['date'])

        # 仅保留同时出现在 data 和 signals 中的 (date, code) 范围
        # 但允许某些 (date, code) 在 data 中存在而 signals 中无信号（视为 signal=0）
        all_dates = sorted(data['date'].unique())
        all_codes = sorted(data['code'].unique())

        if not all_dates:
            return self._empty_result()

        code_to_idx: Dict[str, int] = {c: i for i, c in enumerate(all_codes)}
        n_codes = len(all_codes)
        n_dates = len(all_dates)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # pivot close / limit flags —— 一次性构建，O(n log n)
        close_mat = self._pivot(data, 'close', all_dates, all_codes, date_to_idx, code_to_idx)
        is_limit_up_mat = self._pivot_bool(data, 'is_limit_up', all_dates, all_codes, date_to_idx, code_to_idx)
        is_limit_down_mat = self._pivot_bool(data, 'is_limit_down', all_dates, all_codes, date_to_idx, code_to_idx)

        # pivot signals -> signal_mat (1 buy, -1 sell, 0 hold)
        sig_mat = np.zeros((n_dates, n_codes), dtype=np.int8)
        # 向量化构建：用 pivot 代替逐行 iterrows
        sig_pivot = signals.copy()
        sig_pivot['date'] = pd.to_datetime(sig_pivot['date'])
        # 仅保留在 data 范围内的 (date, code)
        sig_pivot = sig_pivot[
            sig_pivot['date'].isin(date_to_idx) & sig_pivot['code'].isin(code_to_idx)
        ]
        if not sig_pivot.empty:
            sig_pivot['_didx'] = sig_pivot['date'].map(date_to_idx)
            sig_pivot['_cidx'] = sig_pivot['code'].map(code_to_idx)
            sig_val = sig_pivot['signal'].values
            sig_d = sig_pivot['_didx'].values.astype(np.int64)
            sig_c = sig_pivot['_cidx'].values.astype(np.int64)
            # 向量化赋值
            valid_mask = np.isfinite(sig_val.astype(float))
            for i in np.where(valid_mask)[0]:
                v = float(sig_val[i])
                if v > 0:
                    sig_mat[sig_d[i], sig_c[i]] = 1
                elif v < 0:
                    sig_mat[sig_d[i], sig_c[i]] = -1

        # ---- 2. 主循环：逐日推进，但日内全部向量化 ----
        cash = float(init_capital)
        positions = np.zeros(n_codes, dtype=np.float64)        # 持仓股数
        buy_today = np.zeros(n_codes, dtype=bool)              # T+1：今日买入标记
        equity_records: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []

        for t in range(n_dates):
            buy_today[:] = False
            close_t = close_mat[t]
            limit_up_t = is_limit_up_mat[t]
            limit_down_t = is_limit_down_mat[t]
            sig_t = sig_mat[t]

            # ---- 2a. 卖出（先卖后买，释放资金）----
            sell_mask = (sig_t == -1) & (positions > 0)
            if price_limit:
                # 涨跌停限制：跌停无法卖出
                sell_mask = sell_mask & (~limit_down_t)
            # 用 np.nan 表示缺失行情的标的，卖出时跳过
            sellable = sell_mask & (~np.isnan(close_t))
            if sellable.any():
                shares = positions[sellable].copy()
                prices = close_t[sellable]
                amounts = shares * prices
                commissions = np.maximum(amounts * commission_rate, 5.0)
                taxes = amounts * stamp_tax_rate
                net = amounts - commissions - taxes
                cash += float(net.sum())
                # 记录每笔成交
                sell_codes_idx = np.where(sellable)[0]
                for k in sell_codes_idx:
                    trades.append({
                        'date': all_dates[t],
                        'code': all_codes[k],
                        'action': 'sell',
                        'price': float(prices[k]),
                        'shares': float(shares[k]),
                        'amount': float(amounts[k]),
                        'commission': float(commissions[k]),
                        'tax': float(taxes[k]),
                        'pnl': float(amounts[k] - commissions[k] - taxes[k]),
                    })
                positions[sellable] = 0.0

            # ---- 2b. 买入（等权分配可用资金）----
            # 与 native_adapter 行为一致：budget_per_stock 基于当日初始 cash 计算，
            # 但逐标的检查 cost > cash（现金随买入递减），不足时降级或跳过。
            # 此处保留 per-day 内的逐标的循环（n_buy 通常远小于 n_codes），
            # 既保证与 native 行为完全一致，又避免了 native 中每日 DataFrame 过滤的开销。
            buy_mask = (sig_t == 1)
            if price_limit:
                buy_mask = buy_mask & (~limit_up_t)
            buy_mask = buy_mask & (~np.isnan(close_t))

            buy_codes_idx = np.where(buy_mask)[0]
            n_buy = len(buy_codes_idx)
            if n_buy > 0:
                budget_per_stock = cash * 0.95 / n_buy
                for code_idx in buy_codes_idx:
                    price = float(close_t[code_idx]) * (1 + slippage)
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
                    positions[code_idx] += shares
                    buy_today[code_idx] = True
                    trades.append({
                        'date': all_dates[t],
                        'code': all_codes[code_idx],
                        'action': 'buy',
                        'price': float(price),
                        'shares': float(shares),
                        'amount': float(buy_amount),
                        'commission': float(commission),
                        'tax': 0.0,
                        'pnl': float(-buy_amount - commission),
                    })

            # ---- 2c. 当日市值与净值 ----
            held = positions > 0
            market_value = 0.0
            if held.any():
                held_close = close_t[held]
                held_shares = positions[held]
                # 缺失行情的标的用前一日收盘价（向前填充）—— 简化处理：
                # 若当日无行情，则该标的市值按 0 计入（与 native 行为接近：
                # native 在 day_data_map 找不到 code 时直接跳过）
                valid_prices = held_close[~np.isnan(held_close)]
                valid_shares = held_shares[~np.isnan(held_close)]
                market_value = float((valid_prices * valid_shares).sum())

            total_equity = cash + market_value
            equity_records.append({
                'date': all_dates[t],
                'equity': total_equity,
                'cash': cash,
                'market_value': market_value,
                'position_count': int((positions > 0).sum()),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        # ---- 3. 计算绩效（复用既有 BaseBacktestMetrics 公式，自包含实现）----
        from .backtest_engine_compat import calc_all_metrics_compat
        eq_series = equity_curve.set_index('date')['equity']
        metrics = calc_all_metrics_compat(eq_series, trades_df)

        positions_df = pd.DataFrame({
            'code': all_codes,
            'shares': positions,
        })
        positions_df = positions_df[positions_df['shares'] > 0].reset_index(drop=True)

        return {
            "trades": trades_df,
            "positions": positions_df,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    # ------------------------------------------------------------------
    # 辅助：pivot 构建矩阵（向量化版本，避免逐行 itertuples）
    # ------------------------------------------------------------------
    @staticmethod
    def _pivot(
        df: pd.DataFrame, col: str,
        all_dates: List[Any], all_codes: List[str],
        date_to_idx: Dict[Any, int], code_to_idx: Dict[str, int],
    ) -> np.ndarray:
        """将 df[col] pivot 为 (n_dates × n_codes) 矩阵，缺失填 NaN"""
        mat = np.full((len(all_dates), len(all_codes)), np.nan, dtype=np.float64)
        sub = df[['date', 'code', col]].dropna(subset=[col])
        if sub.empty:
            return mat
        # 向量化：用 map 把 date/code 转为矩阵索引，然后一次性赋值
        d_idx = sub['date'].map(date_to_idx).values
        c_idx = sub['code'].map(code_to_idx).values
        vals = sub[col].values
        valid = (~pd.isna(d_idx)) & (~pd.isna(c_idx))
        if valid.any():
            mat[d_idx[valid].astype(int), c_idx[valid].astype(int)] = vals[valid]
        return mat

    @staticmethod
    def _pivot_bool(
        df: pd.DataFrame, col: str,
        all_dates: List[Any], all_codes: List[str],
        date_to_idx: Dict[Any, int], code_to_idx: Dict[str, int],
    ) -> np.ndarray:
        """将 df[col] pivot 为 (n_dates × n_codes) 布尔矩阵，缺失填 False"""
        mat = np.zeros((len(all_dates), len(all_codes)), dtype=bool)
        if col not in df.columns:
            return mat
        sub = df[['date', 'code', col]].dropna(subset=[col])
        if sub.empty:
            return mat
        d_idx = sub['date'].map(date_to_idx).values
        c_idx = sub['code'].map(code_to_idx).values
        vals = sub[col].values
        valid = (~pd.isna(d_idx)) & (~pd.isna(c_idx))
        if valid.any():
            mat[d_idx[valid].astype(int), c_idx[valid].astype(int)] = vals[valid]
        return mat

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
