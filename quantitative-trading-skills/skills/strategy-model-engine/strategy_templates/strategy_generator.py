import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class StrategyTemplateGenerator:
    """
    策略模板生成器
    """
    
    def __init__(self, config):
        self.config = config
    
    def generate_trend_following(self, **kwargs) -> Dict[str, Any]:
        """
        生成趋势跟踪策略模板
        
        Args:
            **kwargs: 策略参数覆盖
        
        Returns:
            策略模板字典
        """
        template = {
            "name": "trend_following",
            "description": "双均线趋势跟踪策略",
            "indicators": [
                {
                    "name": "MA_FAST",
                    "type": "sma",
                    "period": kwargs.get("ma_fast_period", 5)
                },
                {
                    "name": "MA_SLOW",
                    "type": "sma",
                    "period": kwargs.get("ma_slow_period", 20)
                },
                {
                    "name": "BOLLINGER_MID",
                    "type": "sma",
                    "period": kwargs.get("bollinger_period", 20)
                },
                {
                    "name": "BOLLINGER_UPPER",
                    "type": "bollinger_upper",
                    "period": kwargs.get("bollinger_period", 20),
                    "std_dev": kwargs.get("bollinger_std", 2)
                },
                {
                    "name": "BOLLINGER_LOWER",
                    "type": "bollinger_lower",
                    "period": kwargs.get("bollinger_period", 20),
                    "std_dev": kwargs.get("bollinger_std", 2)
                },
                {
                    "name": "DONCHIAN_HIGH",
                    "type": "donchian_high",
                    "period": kwargs.get("donchian_period", 20)
                },
                {
                    "name": "DONCHIAN_LOW",
                    "type": "donchian_low",
                    "period": kwargs.get("donchian_period", 20)
                }
            ],
            "signals": [
                {
                    "name": "MA_CROSS_UP",
                    "condition": "MA_FAST > MA_SLOW and MA_FAST.shift(1) <= MA_SLOW.shift(1)",
                    "action": "BUY"
                },
                {
                    "name": "MA_CROSS_DOWN",
                    "condition": "MA_FAST < MA_SLOW and MA_FAST.shift(1) >= MA_SLOW.shift(1)",
                    "action": "SELL"
                },
                {
                    "name": "BOLLINGER_BREAKOUT_UP",
                    "condition": "close > BOLLINGER_UPPER",
                    "action": "BUY"
                },
                {
                    "name": "BOLLINGER_BREAKOUT_DOWN",
                    "condition": "close < BOLLINGER_LOWER",
                    "action": "SELL"
                },
                {
                    "name": "DONCHIAN_BREAKOUT_UP",
                    "condition": "close > DONCHIAN_HIGH",
                    "action": "BUY"
                },
                {
                    "name": "DONCHIAN_BREAKOUT_DOWN",
                    "condition": "close < DONCHIAN_LOW",
                    "action": "SELL"
                }
            ],
            "parameters": {
                "ma_fast_period": kwargs.get("ma_fast_period", 5),
                "ma_slow_period": kwargs.get("ma_slow_period", 20),
                "bollinger_period": kwargs.get("bollinger_period", 20),
                "bollinger_std": kwargs.get("bollinger_std", 2),
                "donchian_period": kwargs.get("donchian_period", 20)
            },
            "risk_management": {
                "stop_loss": kwargs.get("stop_loss", 0.02),
                "take_profit": kwargs.get("take_profit", 0.05),
                "position_size": kwargs.get("position_size", 0.1)
            }
        }
        return template
    
    def generate_mean_reversion(self, **kwargs) -> Dict[str, Any]:
        """
        生成均值回归策略模板
        
        Args:
            **kwargs: 策略参数覆盖
        
        Returns:
            策略模板字典
        """
        template = {
            "name": "mean_reversion",
            "description": "基于RSI和布林带的均值回归策略",
            "indicators": [
                {
                    "name": "RSI",
                    "type": "rsi",
                    "period": kwargs.get("rsi_period", 14)
                },
                {
                    "name": "BOLLINGER_MID",
                    "type": "sma",
                    "period": kwargs.get("bollinger_period", 20)
                },
                {
                    "name": "BOLLINGER_UPPER",
                    "type": "bollinger_upper",
                    "period": kwargs.get("bollinger_period", 20),
                    "std_dev": kwargs.get("bollinger_std", 2)
                },
                {
                    "name": "BOLLINGER_LOWER",
                    "type": "bollinger_lower",
                    "period": kwargs.get("bollinger_period", 20),
                    "std_dev": kwargs.get("bollinger_std", 2)
                }
            ],
            "signals": [
                {
                    "name": "RSI_OVERSOLD",
                    "condition": "RSI < {}".format(kwargs.get("rsi_oversold", 30)),
                    "action": "BUY"
                },
                {
                    "name": "RSI_OVERBOUGHT",
                    "condition": "RSI > {}".format(kwargs.get("rsi_overbought", 70)),
                    "action": "SELL"
                },
                {
                    "name": "BOLLINGER_MEAN_REVERT_UP",
                    "condition": "close < BOLLINGER_LOWER",
                    "action": "BUY"
                },
                {
                    "name": "BOLLINGER_MEAN_REVERT_DOWN",
                    "condition": "close > BOLLINGER_UPPER",
                    "action": "SELL"
                }
            ],
            "parameters": {
                "rsi_period": kwargs.get("rsi_period", 14),
                "rsi_oversold": kwargs.get("rsi_oversold", 30),
                "rsi_overbought": kwargs.get("rsi_overbought", 70),
                "bollinger_period": kwargs.get("bollinger_period", 20),
                "bollinger_std": kwargs.get("bollinger_std", 2)
            },
            "risk_management": {
                "stop_loss": kwargs.get("stop_loss", 0.02),
                "take_profit": kwargs.get("take_profit", 0.04),
                "position_size": kwargs.get("position_size", 0.1)
            }
        }
        return template
    
    def generate_pair_trading(self, **kwargs) -> Dict[str, Any]:
        """
        生成配对交易策略模板
        
        Args:
            **kwargs: 策略参数覆盖
        
        Returns:
            策略模板字典
        """
        template = {
            "name": "pair_trading",
            "description": "基于协整检验的配对交易策略",
            "indicators": [
                {
                    "name": "SPREAD",
                    "type": "spread",
                    "method": kwargs.get("spread_method", "price_ratio")
                },
                {
                    "name": "SPREAD_MA",
                    "type": "sma",
                    "period": kwargs.get("spread_ma_period", 20),
                    "source": "SPREAD"
                },
                {
                    "name": "SPREAD_STD",
                    "type": "std",
                    "period": kwargs.get("spread_std_period", 20),
                    "source": "SPREAD"
                },
                {
                    "name": "Z_SCORE",
                    "type": "z_score",
                    "period": kwargs.get("z_score_period", 20)
                }
            ],
            "signals": [
                {
                    "name": "SPREAD_DIVERGENCE_LOW",
                    "condition": "Z_SCORE < -{}".format(kwargs.get("z_score_threshold", 2)),
                    "action": "BUY_SPREAD"
                },
                {
                    "name": "SPREAD_DIVERGENCE_HIGH",
                    "condition": "Z_SCORE > {}".format(kwargs.get("z_score_threshold", 2)),
                    "action": "SELL_SPREAD"
                },
                {
                    "name": "SPREAD_CONVERGENCE",
                    "condition": "abs(Z_SCORE) < {}".format(kwargs.get("z_score_close", 0.5)),
                    "action": "CLOSE"
                }
            ],
            "parameters": {
                "spread_method": kwargs.get("spread_method", "price_ratio"),
                "spread_ma_period": kwargs.get("spread_ma_period", 20),
                "spread_std_period": kwargs.get("spread_std_period", 20),
                "z_score_period": kwargs.get("z_score_period", 20),
                "z_score_threshold": kwargs.get("z_score_threshold", 2),
                "z_score_close": kwargs.get("z_score_close", 0.5),
                "cointegration_pvalue": kwargs.get("cointegration_pvalue", 0.05)
            },
            "risk_management": {
                "stop_loss": kwargs.get("stop_loss", 0.03),
                "take_profit": kwargs.get("take_profit", 0.05),
                "position_ratio": kwargs.get("position_ratio", 1.0)
            }
        }
        return template
    
    def generate_template(self, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """
        生成指定类型的策略模板
        
        Args:
            strategy_type: 策略类型 ('trend_following', 'mean_reversion', 'pair_trading')
            **kwargs: 策略参数覆盖
        
        Returns:
            策略模板字典
        """
        if strategy_type == "trend_following":
            return self.generate_trend_following(**kwargs)
        elif strategy_type == "mean_reversion":
            return self.generate_mean_reversion(**kwargs)
        elif strategy_type == "pair_trading":
            return self.generate_pair_trading(**kwargs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
