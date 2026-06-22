"""
向量化回测引擎 (Vectorized Backtest Engine)

借鉴来源:
  - VectorBT: 向量化范式，用 NumPy 数组运算替代逐 bar 的 Python 循环
  - NautilusTrader: 回测/实盘行为一致性理念（T+1、滑点、涨跌停的严格处理）

优化点:
  1. 消除 native_adapter 中的 iterrows() 逐行循环，改用按日期的向量化 NumPy 运算
  2. 严格实现 A 股 T+1 规则（原实现 t_plus_1 参数存在但未真正生效）
  3. 滑点同时作用于买入和卖出（原实现仅作用于买入）
  4. 停牌过滤（close 为 NaN 或 volume==0 视为停牌）

该模块为独立实现，不修改 main 分支的 native_adapter.py。
接口与 BaseBacktestEngine.run_backtest 保持兼容。
"""
from typing import Dict, Any
import time
import numpy as np
import pandas as pd


class VectorizedBacktestEngine:
    """向量化回测引擎（与 NativeAdapter 接口兼容）"""

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

        t0 = time.perf_counter()

        # ── 数据预处理：构建 (date, code) -> 行情的宽表索引 ──
        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        # 统一日期类型
        data['date'] = pd.to_datetime(data['date'])
        signals['date'] = pd.to_datetime(signals['date'])

        all_dates = sorted(signals['date'].unique())
        if len(all_dates) == 0:
            return self._empty_result()

        # 构建 date -> 该日行情 dict(code -> row) 的查找结构
        # 用 pivot 一次性构建宽表，避免逐日 set_index 的重复开销
        price_close = data.pivot_table(index='date', columns='code', values='close')
        volume_wide = data.pivot_table(index='date', columns='code', values='volume') if 'volume' in data.columns else None

        # 涨跌停标记宽表（若存在）
        limit_up_wide = None
        limit_down_wide = None
        if price_limit and 'is_limit_up' in data.columns:
            limit_up_wide = data.pivot_table(index='date', columns='code', values='is_limit_up', aggfunc='max').fillna(0).astype(bool)
        if price_limit and 'is_limit_down' in data.columns:
            limit_down_wide = data.pivot_table(index='date', columns='code', values='is_limit_down', aggfunc='max').fillna(0).astype(bool)

        # 信号宽表：1 买入, -1 卖出, 0 无
        sig_wide = signals.pivot_table(index='date', columns='code', values='signal', aggfunc='first').fillna(0)

        # 对齐到回测日期序列（仅保留有信号的日期）
        sig_wide = sig_wide.reindex(all_dates)

        codes = list(sig_wide.columns)
        n_codes = len(codes)
        if n_codes == 0:
            return self._empty_result()

        # 关键：将所有行情宽表的列对齐到 codes（信号中的股票池），
        # 行对齐到 all_dates，确保 close_arr / sig_arr 维度一致
        price_close = price_close.reindex(index=all_dates, columns=codes)
        if volume_wide is not None:
            volume_wide = volume_wide.reindex(index=all_dates, columns=codes)
        if limit_up_wide is not None:
            limit_up_wide = limit_up_wide.reindex(index=all_dates, columns=codes).fillna(False)
        if limit_down_wide is not None:
            limit_down_wide = limit_down_wide.reindex(index=all_dates, columns=codes).fillna(False)

        # 转 numpy 加速
        sig_arr = sig_wide.values.astype(float)            # (T, N)
        close_arr = price_close.values.astype(float)       # (T, N)
        # 停牌：close 为 NaN 或 volume==0
        suspended = np.isnan(close_arr)
        if volume_wide is not None:
            vol_arr = volume_wide.values.astype(float)
            suspended = suspended | (vol_arr == 0)

        limit_up_arr = limit_up_wide.values if limit_up_wide is not None else np.zeros_like(suspended, dtype=bool)
        limit_down_arr = limit_down_wide.values if limit_down_wide is not None else np.zeros_like(suspended, dtype=bool)

        # ── 回测主循环（按日，但日内全部向量化）──
        cash = float(init_capital)
        holdings = np.zeros(n_codes, dtype=np.float64)      # 当前持仓股数
        # T+1: 记录每只股票最近一次买入所在的日期索引，t_plus_1 时当日买入不可卖
        buy_day = np.full(n_codes, -1, dtype=np.int64)

        equity_records = []
        trades = []

        for t_idx, dt in enumerate(all_dates):
            sig_t = sig_arr[t_idx]
            close_t = close_arr[t_idx]
            susp_t = suspended[t_idx]
            lu_t = limit_up_arr[t_idx]
            ld_t = limit_down_arr[t_idx]

            # 可交易掩码：未停牌
            tradable = ~susp_t

            # ── 卖出：signal < 0 ──
            sell_mask = (sig_t < 0) & (holdings > 0) & tradable
            if price_limit:
                # 涨跌停限制：跌停时无法卖出
                sell_mask = sell_mask & (~ld_t)
            if t_plus_1:
                # 当日买入的不可卖出
                sell_mask = sell_mask & (buy_day < t_idx)

            if sell_mask.any():
                # 卖出全部持仓
                sell_shares = holdings[sell_mask].copy()
                # 卖出价：close 扣减滑点
                sell_prices = close_t[sell_mask] * (1.0 - slippage)
                sell_amounts = sell_shares * sell_prices
                commissions = np.maximum(sell_amounts * commission_rate, 5.0)
                taxes = sell_amounts * stamp_tax_rate
                net_cash_in = sell_amounts - commissions - taxes
                cash += float(net_cash_in.sum())

                # 记录成交
                sell_codes_idx = np.where(sell_mask)[0]
                for k, ci in enumerate(sell_codes_idx):
                    trades.append({
                        'date': dt, 'code': codes[ci], 'action': 'sell',
                        'price': float(sell_prices[k]), 'shares': float(sell_shares[k]),
                        'amount': float(sell_amounts[k]), 'commission': float(commissions[k]),
                        'tax': float(taxes[k]), 'pnl': float(sell_amounts[k] - commissions[k] - taxes[k]),
                    })
                holdings[sell_mask] = 0.0

            # ── 买入：signal > 0 ──
            buy_mask = (sig_t > 0) & tradable
            if price_limit:
                # 涨停时无法买入
                buy_mask = buy_mask & (~lu_t)
            # 过滤 close 为 NaN（已由 tradable 处理）

            if buy_mask.any():
                n_buy = int(buy_mask.sum())
                # 等权分配：可用资金的 95% 平均分配
                budget_per = cash * 0.95 / n_buy
                buy_prices = close_t[buy_mask] * (1.0 + slippage)
                # 整 100 股下单
                buy_shares = np.floor(budget_per / buy_prices / 100.0) * 100.0
                # 资金不足时二次调整
                too_expensive = buy_shares <= 0
                if too_expensive.any():
                    alt_budget = cash * 0.98 / n_buy
                    alt_shares = np.floor(alt_budget / buy_prices / 100.0) * 100.0
                    buy_shares = np.where(too_expensive, alt_shares, buy_shares)

                valid = buy_shares > 0
                buy_amounts = buy_shares * buy_prices
                commissions = np.maximum(buy_amounts * commission_rate, 5.0)
                total_cost = float((buy_amounts + commissions).sum())

                # 若总成本超出现金，按比例缩减（保持等权）
                if total_cost > cash and total_cost > 0:
                    scale = cash * 0.98 / total_cost
                    buy_shares = np.floor(buy_shares * scale / 100.0) * 100.0
                    valid = buy_shares > 0
                    buy_amounts = buy_shares * buy_prices
                    commissions = np.maximum(buy_amounts * commission_rate, 5.0)
                    total_cost = float((buy_amounts + commissions).sum())

                cash -= total_cost

                buy_codes_idx = np.where(buy_mask)[0]
                for k, ci in enumerate(buy_codes_idx):
                    if valid[k]:
                        holdings[ci] += buy_shares[k]
                        buy_day[ci] = t_idx  # 记录买入日（T+1 用）
                        trades.append({
                            'date': dt, 'code': codes[ci], 'action': 'buy',
                            'price': float(buy_prices[k]), 'shares': float(buy_shares[k]),
                            'amount': float(buy_amounts[k]), 'commission': float(commissions[k]),
                            'tax': 0.0, 'pnl': float(-buy_amounts[k] - commissions[k]),
                        })

            # ── 估值 ──
            # 用当日 close 估值持仓（停牌的用上次估值，这里简化为 close 为 NaN 时取 0）
            valid_hold = holdings > 0
            if valid_hold.any():
                vals = np.where(susp_t, 0.0, close_t) * holdings
                market_value = float(vals.sum())
            else:
                market_value = 0.0

            total_equity = cash + market_value
            equity_records.append({
                'date': dt,
                'equity': total_equity,
                'cash': cash,
                'market_value': market_value,
                'position_count': int((holdings > 0).sum()),
            })

        elapsed = time.perf_counter() - t0

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        eq_series = equity_curve.set_index('date')['equity']
        metrics = self._calc_metrics(eq_series, trades_df)
        metrics['engine'] = 'vectorized'
        metrics['backtest_elapsed_sec'] = round(elapsed, 4)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame({'code': codes, 'shares': holdings}),
            "equity_curve": equity_curve,
            "metrics": metrics,
            "report_path": "",
        }

    @staticmethod
    def _calc_metrics(equity_curve: pd.Series, trades: pd.DataFrame) -> Dict[str, Any]:
        """绩效指标计算（与 BaseBacktestMetrics 对齐）"""
        from datetime import datetime
        if len(equity_curve) < 2:
            return {}
        returns = equity_curve.pct_change().dropna()
        total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        n_years = len(equity_curve) / 252.0
        annual_return = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
        volatility = float(returns.std() * np.sqrt(252))
        sharpe = float((returns.mean() * 252 - 0.03) / volatility) if volatility > 0 else 0.0
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = float(drawdown.min())
        calmar = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
        neg = returns[returns < 0]
        downside_std = float(neg.std() * np.sqrt(252)) if len(neg) >= 2 else 0.0
        sortino = float((returns.mean() * 252 - 0.03) / downside_std) if downside_std > 0 else 0.0
        win_rate = float((trades['pnl'] > 0).mean()) if not trades.empty and 'pnl' in trades.columns else 0.0
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "total_trades": int(len(trades)),
            "calculation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @staticmethod
    def _empty_result():
        return {
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "metrics": {},
            "report_path": "",
        }
