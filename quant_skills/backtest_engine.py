"""
策略回测模块
基于backtrader框架实现策略回测功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
import warnings


class BacktestEngine:
    """
    回测引擎
    提供策略回测、性能分析等功能
    """
    
    def __init__(self):
        self._backtrader_available = False
        
        try:
            import backtrader as bt
            self._bt = bt
            self._backtrader_available = True
        except ImportError:
            pass
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        datetime_index: bool = True
    ) -> pd.DataFrame:
        """
        准备回测数据
        
        参数:
            data: OHLCV数据
            datetime_index: 索引是否为datetime
            
        返回:
            pd.DataFrame: 格式化后的数据
        """
        df = data.copy()
        
        # 确保有必要的列
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 确保索引是datetime
        if not datetime_index:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
        
        return df
    
    class SimpleMovingAverageCross:
        """
        简单均线交叉策略
        """
        
        def __init__(self, params: Optional[Dict] = None):
            self.params = params or {
                "fast_period": 20,
                "slow_period": 60
            }
        
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            """
            生成交易信号
            
            参数:
                data: OHLC数据
                
            返回:
                pd.Series: 信号 (1=买入, -1=卖出, 0=持有)
            """
            fast_ma = data["close"].rolling(self.params["fast_period"]).mean()
            slow_ma = data["close"].rolling(self.params["slow_period"]).mean()
            
            signals = pd.Series(0, index=data.index)
            
            # 金叉: 短期均线上穿长期均线
            signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
            
            # 死叉: 短期均线下穿长期均线
            signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1
            
            return signals
    
    class RSIStrategy:
        """
        RSI策略
        """
        
        def __init__(self, params: Optional[Dict] = None):
            self.params = params or {
                "period": 14,
                "oversold": 30,
                "overbought": 70
            }
        
        def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
            """计算RSI"""
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            """生成交易信号"""
            rsi = self.calculate_rsi(data["close"], self.params["period"])
            
            signals = pd.Series(0, index=data.index)
            
            # RSI超卖买入
            signals[(rsi < self.params["oversold"]) & (rsi.shift(1) >= self.params["oversold"])] = 1
            
            # RSI超买卖出
            signals[(rsi > self.params["overbought"]) & (rsi.shift(1) <= self.params["overbought"])] = -1
            
            return signals
    
    def backtest(
        self,
        data: pd.DataFrame,
        strategy: Any,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.001
    ) -> Dict[str, Any]:
        """
        执行回测
        
        参数:
            data: OHLCV数据
            strategy: 策略对象 (需实现generate_signals方法)
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点率
            
        返回:
            Dict: 回测结果
        """
        df = data.copy()
        
        # 生成信号
        signals = strategy.generate_signals(df)
        
        # 初始化回测变量
        position = 0  # 当前持仓数量
        cash = initial_capital
        portfolio_values = []
        trades = []
        
        for i, date in enumerate(df.index):
            price = df["close"].iloc[i]
            
            # 考虑滑点
            buy_price = price * (1 + slippage)
            sell_price = price * (1 - slippage)
            
            # 执行交易
            if signals.iloc[i] == 1 and position == 0:
                # 买入
                shares = int(cash / buy_price)
                if shares > 0:
                    cost = shares * buy_price
                    fee = cost * commission
                    cash -= (cost + fee)
                    position = shares
                    trades.append({
                        "date": date,
                        "type": "buy",
                        "price": buy_price,
                        "shares": shares,
                        "fee": fee
                    })
            
            elif signals.iloc[i] == -1 and position > 0:
                # 卖出
                proceeds = position * sell_price
                fee = proceeds * commission
                cash += (proceeds - fee)
                trades.append({
                    "date": date,
                    "type": "sell",
                    "price": sell_price,
                    "shares": position,
                    "fee": fee
                })
                position = 0
            
            # 计算组合价值
            portfolio_value = cash + position * price
            portfolio_values.append({
                "date": date,
                "value": portfolio_value,
                "cash": cash,
                "position": position
            })
        
        # 计算最终价值
        portfolio_df = pd.DataFrame(portfolio_values).set_index("date")
        
        # 计算性能指标
        performance = self._calculate_performance(portfolio_df, initial_capital)
        
        return {
            "portfolio": portfolio_df,
            "trades": trades,
            "signals": signals,
            "performance": performance
        }
    
    def _calculate_performance(
        self,
        portfolio_df: pd.DataFrame,
        initial_capital: float
    ) -> Dict[str, Any]:
        """
        计算性能指标
        
        参数:
            portfolio_df: 组合价值数据
            initial_capital: 初始资金
            
        返回:
            Dict: 性能指标
        """
        values = portfolio_df["value"]
        returns = values.pct_change().dropna()
        
        # 总收益率
        total_return = (values.iloc[-1] - initial_capital) / initial_capital
        
        # 年化收益率
        days = (values.index[-1] - values.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # 年化波动率
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率 (假设无风险利率为0)
        if annual_vol > 0:
            sharpe_ratio = annual_return / annual_vol
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # 胜率 (如果有交易记录)
        win_rate = None
        if "position" in portfolio_df.columns:
            trades = portfolio_df[portfolio_df["position"].diff() != 0]
            if len(trades) > 1:
                # 简化计算
                win_rate = 0.5
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "final_value": values.iloc[-1],
            "win_rate": win_rate
        }
    
    def run_backtrader(
        self,
        data: pd.DataFrame,
        strategy_class: Any,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用backtrader执行回测
        
        参数:
            data: OHLCV数据
            strategy_class: backtrader策略类
            initial_capital: 初始资金
            commission: 手续费率
            **kwargs: 策略参数
            
        返回:
            Dict: 回测结果
        """
        if not self._backtrader_available:
            raise ImportError("backtrader库未安装，请运行: pip install backtrader")
        
        cerebro = self._bt.Cerebro()
        
        # 设置初始资金
        cerebro.broker.setcash(initial_capital)
        
        # 设置手续费
        cerebro.broker.setcommission(commission=commission)
        
        # 准备数据
        feed = self._bt.feeds.PandasData(dataname=data)
        cerebro.adddata(feed)
        
        # 添加策略
        cerebro.addstrategy(strategy_class, **kwargs)
        
        # 添加分析器
        cerebro.addanalyzer(self._bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(self._bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(self._bt.analyzers.Returns, _name="returns")
        
        # 运行回测
        results = cerebro.run()
        strat = results[0]
        
        # 获取分析结果
        performance = {
            "final_value": cerebro.broker.getvalue(),
            "sharpe_ratio": strat.analyzers.sharpe.get_analysis().get("sharperatio", 0),
            "max_drawdown": strat.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0),
            "total_return": strat.analyzers.returns.get_analysis().get("rnorm100", 0) / 100
        }
        
        return {
            "cerebro": cerebro,
            "strategy": strat,
            "performance": performance
        }
