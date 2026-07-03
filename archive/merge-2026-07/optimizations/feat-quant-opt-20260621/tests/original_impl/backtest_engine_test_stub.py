"""
原版 NativeAdapter 的测试 stub

直接复制 /workspace/skills/backtest-engine/scripts/adapters/native_adapter.py 的逻辑，
仅将相对导入改为绝对导入，用于性能与正确性对比测试。

不修改原版代码，仅用于 benchmark。
"""
import sys
import os
from typing import Dict, Any
import pandas as pd
import numpy as np

# 添加原版 backtest-engine 路径
# 从 tests/original_impl/ 回到 /workspace 需要 4 级 ..
BACKTEST_ENGINE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "skills", "backtest-engine", "scripts"
))
sys.path.insert(0, BACKTEST_ENGINE_PATH)

from base.base_backtest import BaseBacktestMetrics  # noqa: E402


class NativeAdapterStub:
    """
    原版 NativeAdapter 的复刻 (用于对比测试)
    逻辑与 /workspace/skills/backtest-engine/scripts/adapters/native_adapter.py 完全一致
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

        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(signals['date'].unique())
        if not dates:
            return self._empty_result()

        cash = init_capital
        positions = {}
        equity_records = []
        trades = []

        for dt in dates:
            day_signal = signals[signals['date'] == dt]
            day_data = data[data['date'] == dt]

            if day_data.empty:
                continue

            day_data_map = day_data.set_index('code')

            sell_codes = []
            buy_codes = []
            for _, row in day_signal.iterrows():
                code = row['code']
                sig = row.get('signal', 0)
                if isinstance(sig, (int, float, np.integer, np.floating)):
                    sig = float(sig)
                    if sig > 0:
                        buy_codes.append(code)
                    elif sig < 0:
                        sell_codes.append(code)

            for code in sell_codes:
                if code not in positions or positions[code] <= 0:
                    continue
                if code not in day_data_map.index:
                    continue
                price_row = day_data_map.loc[code]
                if price_limit and price_row.get('is_limit_down', False):
                    continue
                price = price_row['close']
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * commission_rate, 5)
                tax = sell_amount * stamp_tax_rate
                cost = commission + tax
                cash += sell_amount - cost
                trades.append({
                    'date': dt, 'code': code, 'action': 'sell',
                    'price': price, 'shares': shares, 'amount': sell_amount,
                    'commission': commission, 'tax': tax, 'pnl': sell_amount - cost,
                })
                positions[code] = 0

            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * 0.95 / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    if price_limit and price_row.get('is_limit_up', False):
                        continue
                    price = price_row['close'] * (1 + slippage)
                    shares = int(budget_per_stock / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * commission_rate, 5)
                    cost = buy_amount + commission
                    if cost > cash:
                        shares = int((cash * 0.98) / price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = price * shares
                        commission = max(buy_amount * commission_rate, 5)
                        cost = buy_amount + commission
                    cash -= cost
                    positions[code] = positions.get(code, 0) + shares
                    trades.append({
                        'date': dt, 'code': code, 'action': 'buy',
                        'price': price, 'shares': shares, 'amount': buy_amount,
                        'commission': commission, 'tax': 0, 'pnl': -buy_amount - commission,
                    })

            market_value = 0
            for code, shares in list(positions.items()):
                if shares <= 0:
                    continue
                if code in day_data_map.index:
                    market_value += shares * day_data_map.loc[code, 'close']
            total_equity = cash + market_value

            equity_records.append({
                'date': dt,
                'equity': total_equity,
                'cash': cash,
                'market_value': market_value,
                'position_count': sum(1 for s in positions.values() if s > 0),
            })

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        eq_series = equity_curve.set_index('date')['equity']
        metrics = BaseBacktestMetrics.calc_all_metrics(eq_series, trades_df)

        return {
            "trades": trades_df,
            "positions": pd.DataFrame(list(positions.items()), columns=['code', 'shares']),
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
