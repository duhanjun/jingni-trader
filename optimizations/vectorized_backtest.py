"""
向量化回测引擎

借鉴来源：
- VectorBT 的向量化回测思想（用 NumPy 矩阵运算替代 Python for-loop）
- NautilusTrader 的确定性事件驱动思想（保留 T+1、涨跌停、印花税等 A 股规则）

对照 jingni-trader 现有实现：
- skills/backtest-engine/scripts/adapters/native_adapter.py
  使用 Python for-loop 逐日遍历日期，逐股票处理信号，性能受限

本模块的核心改进：
1. 将"逐日循环"改为"按日期分组的向量化运算"
2. 将持仓/现金状态用 DataFrame 表示，避免 dict 逐键查找
3. 信号对齐用 merge 一次完成，避免循环内 day_data_map.loc[code]
4. 保留 A 股 T+1、涨跌停、印花税、佣金、滑点等业务规则
"""
from typing import Dict, Any, Optional
import time
import numpy as np
import pandas as pd


class VectorizedBacktester:
    """向量化回测引擎（A股规则）"""

    def __init__(
        self,
        init_capital: float = 1e6,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        cash_buffer: float = 0.95,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit
        self.cash_buffer = cash_buffer

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        执行向量化回测

        参数:
            data: 日线数据，至少包含 date, code, open, high, low, close,
                  以及 is_limit_up, is_limit_down（可选）
            signals: 信号数据，包含 date, code, signal
                     signal > 0 买入，signal < 0 卖出

        返回:
            {
                "equity_curve": DataFrame,
                "trades": DataFrame,
                "metrics": dict,
            }
        """
        if data.empty or signals.empty:
            return self._empty_result()

        # ---- 1. 数据预处理（向量化）----
        # 避免不必要的 copy，只在必要时修改
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
        if not pd.api.types.is_datetime64_any_dtype(signals['date']):
            signals = signals.copy()
            signals['date'] = pd.to_datetime(signals['date'])

        # 补全涨跌停标记（若缺失）- 只在缺失时才操作
        if 'is_limit_up' not in data.columns:
            data = data.copy()
            data['is_limit_up'] = False
        if 'is_limit_down' not in data.columns:
            data = data.copy()
            data['is_limit_down'] = False
        if data['is_limit_up'].dtype != bool:
            data = data.copy()
            data['is_limit_up'] = data['is_limit_up'].fillna(False).astype(bool)
            data['is_limit_down'] = data['is_limit_down'].fillna(False).astype(bool)

        # ---- 2. 信号对齐（一次 merge 完成，替代循环内查找）----
        # 信号方向：1=买入, -1=卖出, 0=持有
        sig_dir = np.sign(signals['signal'].fillna(0).values).astype(int)
        active_signals = signals.iloc[np.where(sig_dir != 0)[0]][['date', 'code']].copy()
        active_signals['sig_dir'] = sig_dir[sig_dir != 0]

        # 将信号 merge 到行情数据上
        merged = data.merge(
            active_signals,
            on=['date', 'code'],
            how='left',
        )
        merged['sig_dir'] = merged['sig_dir'].fillna(0).astype(int)

        # 按日期排序
        merged = merged.sort_values(['date', 'code']).reset_index(drop=True)

        # ---- 3. 逐日结算（保留必要的循环，但每日内部向量化）----
        # 注意：回测本质上是路径依赖的（cash/positions 状态依赖前一日），
        # 完全向量化需要牺牲业务规则的真实性（如 T+1、涨跌停、资金约束）。
        # 这里采用"日外循环 + 日内向量化"的混合策略：
        #   - 日外循环：维护 cash/positions 状态机（无法避免）
        #   - 日内向量化：用 numpy 数组处理当日所有股票的买卖/估值
        # 优化：用 groupby 一次性分组，避免逐日 filter
        dates = sorted(merged['date'].unique())
        cash = float(self.init_capital)
        positions: Dict[str, int] = {}  # code -> shares
        # T+1: 记录当日买入的股票，当日不能卖
        bought_today: Dict[str, bool] = {}

        equity_records = []
        trades = []

        # 预先按日期分组（避免重复 filter）
        # 用 dict 缓存每个日期对应的 DataFrame
        date_groups = {dt: g for dt, g in merged.groupby('date')}

        for dt in dates:
            day = date_groups.get(dt)
            if day is None or day.empty:
                continue

            # 直接用 numpy 数组（避免 pandas 开销）
            codes = day['code'].values
            closes = day['close'].values
            limit_up = day['is_limit_up'].values
            limit_down = day['is_limit_down'].values
            sigs = day['sig_dir'].values

            # 当日收盘价映射（用 dict，对于 < 1000 只股票比 numpy 索引快）
            code_to_close = dict(zip(codes, closes))

            # ---- 3.1 卖出（先卖后买，释放资金）----
            sell_mask = sigs < 0
            if sell_mask.any():
                for i in np.where(sell_mask)[0]:
                    code = codes[i]
                    # T+1 检查
                    if self.t_plus_1 and bought_today.get(code, False):
                        continue
                    shares = positions.get(code, 0)
                    if shares <= 0:
                        continue
                    # 涨跌停检查（卖出时若跌停无法成交）
                    if self.price_limit and limit_down[i]:
                        continue
                    price = closes[i]
                    sell_amount = price * shares
                    commission = max(sell_amount * self.commission_rate, self.min_commission)
                    tax = sell_amount * self.stamp_tax_rate
                    cash += sell_amount - commission - tax
                    trades.append({
                        'date': dt, 'code': code, 'action': 'sell',
                        'price': price, 'shares': int(shares),
                        'amount': sell_amount, 'commission': commission,
                        'tax': tax,
                    })
                    positions[code] = 0

            # ---- 3.2 买入（向量化预算分配）----
            buy_mask = sigs > 0
            if buy_mask.any():
                buy_indices = np.where(buy_mask)[0]
                # 过滤涨停（无法买入）和已在持仓中的（可选）
                valid_mask = np.ones(len(buy_indices), dtype=bool)
                for k, idx in enumerate(buy_indices):
                    if self.price_limit and limit_up[idx]:
                        valid_mask[k] = False
                buy_indices = buy_indices[valid_mask]

                if len(buy_indices) > 0:
                    n_buy = len(buy_indices)
                    budget_per_stock = cash * self.cash_buffer / n_buy
                    # 向量化计算股数（向下取整到 100 股）
                    buy_prices = closes[buy_indices] * (1 + self.slippage)
                    shares_arr = (budget_per_stock / buy_prices / 100).astype(int) * 100

                    for k, idx in enumerate(buy_indices):
                        shares = int(shares_arr[k])
                        if shares <= 0:
                            continue
                        code = codes[idx]
                        price = buy_prices[k]
                        buy_amount = price * shares
                        commission = max(buy_amount * self.commission_rate, self.min_commission)
                        cost = buy_amount + commission
                        if cost > cash:
                            # 资金不足，按可用资金重新计算
                            shares = int((cash * 0.98) / price / 100) * 100
                            if shares <= 0:
                                continue
                            buy_amount = price * shares
                            commission = max(buy_amount * self.commission_rate, self.min_commission)
                            cost = buy_amount + commission
                        cash -= cost
                        positions[code] = positions.get(code, 0) + shares
                        bought_today[code] = True
                        trades.append({
                            'date': dt, 'code': code, 'action': 'buy',
                            'price': price, 'shares': shares,
                            'amount': buy_amount, 'commission': commission,
                            'tax': 0.0,
                        })

            # ---- 3.3 估值（用 dict 直接查找，避免 numpy 索引开销）----
            mv = 0.0
            pos_count = 0
            for code, shares in positions.items():
                if shares <= 0:
                    continue
                price = code_to_close.get(code)
                if price is not None:
                    mv += shares * price
                    pos_count += 1

            equity_records.append({
                'date': dt,
                'equity': cash + mv,
                'cash': cash,
                'market_value': mv,
                'position_count': pos_count,
            })

            # T+1: 日终清空当日买入标记
            bought_today = {}

        equity_curve = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(trades)

        if equity_curve.empty:
            return self._empty_result()

        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "positions": pd.DataFrame(
                list(positions.items()), columns=['code', 'shares']
            ) if positions else pd.DataFrame(columns=['code', 'shares']),
            "metrics": {},  # 由 enhanced_metrics 模块计算
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "positions": pd.DataFrame(columns=['code', 'shares']),
            "metrics": {},
        }


# ------------------------------------------------------------------
# 对照基准：复刻 jingni-trader 现有 native_adapter 的循环实现
# 用于性能对比和正确性验证
# ------------------------------------------------------------------
class LoopBacktester:
    """循环式回测（对照基准，复刻 jingni-trader 现有实现风格）"""

    def __init__(
        self,
        init_capital: float = 1e6,
        commission_rate: float = 0.00025,
        stamp_tax_rate: float = 0.001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        t_plus_1: bool = True,
        price_limit: bool = True,
        cash_buffer: float = 0.95,
    ):
        self.init_capital = init_capital
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.t_plus_1 = t_plus_1
        self.price_limit = price_limit
        self.cash_buffer = cash_buffer

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, Any]:
        if data.empty or signals.empty:
            return {"equity_curve": pd.DataFrame(), "trades": pd.DataFrame(), "metrics": {}}

        data = data.copy()
        signals = signals.copy()
        data['date'] = pd.to_datetime(data['date'])
        signals['date'] = pd.to_datetime(signals['date'])

        if 'is_limit_up' not in data.columns:
            data['is_limit_up'] = False
        if 'is_limit_down' not in data.columns:
            data['is_limit_down'] = False

        data = data.sort_values(['date', 'code']).reset_index(drop=True)
        signals = signals.sort_values(['date', 'code']).reset_index(drop=True)

        dates = sorted(signals['date'].unique())
        cash = float(self.init_capital)
        positions: Dict[str, int] = {}
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
                if self.price_limit and price_row.get('is_limit_down', False):
                    continue
                price = price_row['close']
                shares = positions[code]
                sell_amount = price * shares
                commission = max(sell_amount * self.commission_rate, self.min_commission)
                tax = sell_amount * self.stamp_tax_rate
                cash += sell_amount - commission - tax
                trades.append({
                    'date': dt, 'code': code, 'action': 'sell',
                    'price': price, 'shares': int(shares),
                    'amount': sell_amount, 'commission': commission, 'tax': tax,
                })
                positions[code] = 0

            if buy_codes:
                n_buy = len(buy_codes)
                budget_per_stock = cash * self.cash_buffer / n_buy
                for code in buy_codes:
                    if code not in day_data_map.index:
                        continue
                    price_row = day_data_map.loc[code]
                    if self.price_limit and price_row.get('is_limit_up', False):
                        continue
                    price = price_row['close'] * (1 + self.slippage)
                    shares = int(budget_per_stock / price / 100) * 100
                    if shares <= 0:
                        continue
                    buy_amount = price * shares
                    commission = max(buy_amount * self.commission_rate, self.min_commission)
                    cost = buy_amount + commission
                    if cost > cash:
                        shares = int((cash * 0.98) / price / 100) * 100
                        if shares <= 0:
                            continue
                        buy_amount = price * shares
                        commission = max(buy_amount * self.commission_rate, self.min_commission)
                        cost = buy_amount + commission
                    cash -= cost
                    positions[code] = positions.get(code, 0) + shares
                    trades.append({
                        'date': dt, 'code': code, 'action': 'buy',
                        'price': price, 'shares': int(shares),
                        'amount': buy_amount, 'commission': commission, 'tax': 0.0,
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
        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "positions": pd.DataFrame(
                list(positions.items()), columns=['code', 'shares']
            ) if positions else pd.DataFrame(columns=['code', 'shares']),
            "metrics": {},
        }


def benchmark_backtest(
    data: pd.DataFrame,
    signals: pd.DataFrame,
    runs: int = 3,
) -> Dict[str, Any]:
    """对比向量化回测 vs 循环回测的性能"""
    # 正确性验证
    vb = VectorizedBacktester()
    lb = LoopBacktester()

    vb_result = vb.run(data, signals)
    lb_result = lb.run(data, signals)

    # 性能测试
    vb_times = []
    lb_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        vb.run(data, signals)
        vb_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lb.run(data, signals)
        lb_times.append(time.perf_counter() - t0)

    return {
        "vectorized": {
            "median_time": float(np.median(vb_times)),
            "min_time": float(np.min(vb_times)),
            "times": vb_times,
            "final_equity": float(vb_result['equity_curve']['equity'].iloc[-1])
                if not vb_result['equity_curve'].empty else 0.0,
            "n_trades": len(vb_result['trades']),
        },
        "loop": {
            "median_time": float(np.median(lb_times)),
            "min_time": float(np.min(lb_times)),
            "times": lb_times,
            "final_equity": float(lb_result['equity_curve']['equity'].iloc[-1])
                if not lb_result['equity_curve'].empty else 0.0,
            "n_trades": len(lb_result['trades']),
        },
        "speedup": float(np.median(lb_times) / np.median(vb_times)) if np.median(vb_times) > 0 else 0,
        "equity_match": abs(
            float(vb_result['equity_curve']['equity'].iloc[-1] if not vb_result['equity_curve'].empty else 0) -
            float(lb_result['equity_curve']['equity'].iloc[-1] if not lb_result['equity_curve'].empty else 0)
        ) < 1e-6,
    }
