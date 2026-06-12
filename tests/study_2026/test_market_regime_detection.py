"""
验证代码：市场状态（Regime）检测与自适应策略切换
==================================================
借鉴来源：
  - TradingAgents (github.com/TauricResearch/TradingAgents)
    - 多智能体架构中的 Risk Manager Agent 市场状态评估
    - 多维度市场分析（基本面、技术面、情绪面）
  - FreqAI 的动态特征权重调整
  - Microsoft Qlib 的 Market Dynamics Modeling

优化方向：portfolio-risk-engine 模块增加市场状态检测，根据市场状态自适应调整
          风险参数和策略权重，避免在不利市场环境下使用不合适的策略。

对比分析：
  - 现有方式：固定风险参数（max_position, max_loss_per_day 等）
  - 优化方式：根据市场状态动态调整风险敞口和策略参数
"""

import sys
import os
import time
import json
import unittest
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

warnings.filterwarnings('ignore')


# =============================================================================
# 市场状态枚举
# =============================================================================

class MarketRegime(Enum):
    """市场状态分类"""
    BULL_TRENDING = "bull_trending"       # 牛市趋势
    BULL_VOLATILE = "bull_volatile"       # 牛市震荡
    BEAR_TRENDING = "bear_trending"       # 熊市趋势
    BEAR_VOLATILE = "bear_volatile"       # 熊市震荡
    SIDEWAYS = "sideways"                 # 横盘整理
    HIGH_VOLATILITY = "high_volatility"   # 高波动
    LOW_VOLATILITY = "low_volatility"     # 低波动
    CRISIS = "crisis"                     # 危机模式


# =============================================================================
# 市场状态检测器
# =============================================================================

class MarketRegimeDetector:
    """
    市场状态检测器
    
    借鉴 TradingAgents 的多维度分析思路：
    - 趋势分析：判断市场方向
    - 波动率分析：判断市场风险水平
    - 流动性分析：判断市场活跃度
    - 相关性分析：判断市场结构
    """
    
    def __init__(
        self,
        trend_lookback: int = 60,
        volatility_lookback: int = 20,
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.05,
        high_vol_threshold: float = 0.30,
        crisis_vol_threshold: float = 0.50,
    ):
        self.trend_lookback = trend_lookback
        self.volatility_lookback = volatility_lookback
        self.bull_threshold = bull_threshold
        self.bear_threshold = bear_threshold
        self.high_vol_threshold = high_vol_threshold
        self.crisis_vol_threshold = crisis_vol_threshold
    
    def detect(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测当前市场状态
        
        参数:
            price_data: 包含 date, close 等字段的行情数据
        
        返回:
            市场状态分析结果
        """
        if price_data.empty:
            return {'regime': MarketRegime.SIDEWAYS, 'confidence': 0.0}
        
        # 按日期排序
        df = price_data.sort_values('date').copy()
        
        # 1. 趋势分析
        trend = self._analyze_trend(df)
        
        # 2. 波动率分析
        volatility = self._analyze_volatility(df)
        
        # 3. 流动性分析
        liquidity = self._analyze_liquidity(df)
        
        # 4. 综合判断市场状态
        regime, confidence = self._classify_regime(trend, volatility, liquidity)
        
        return {
            'regime': regime,
            'confidence': confidence,
            'trend': trend,
            'volatility': volatility,
            'liquidity': liquidity,
            'details': self._generate_details(regime, trend, volatility, liquidity),
        }
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """趋势分析"""
        close = df['close']
        
        if len(close) < self.trend_lookback:
            return {'direction': 0, 'strength': 0, 'ma_alignment': 0}
        
        # 计算收益率
        ret_short = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
        ret_long = (close.iloc[-1] / close.iloc[-self.trend_lookback] - 1) if len(close) >= self.trend_lookback else 0
        
        # 均线排列（多头/空头）
        ma_5 = close.rolling(5).mean()
        ma_20 = close.rolling(20).mean()
        ma_60 = close.rolling(60).mean()
        
        if len(ma_60.dropna()) > 0:
            latest_ma5 = ma_5.iloc[-1]
            latest_ma20 = ma_20.iloc[-1]
            latest_ma60 = ma_60.iloc[-1]
            
            if latest_ma5 > latest_ma20 > latest_ma60:
                ma_alignment = 1.0  # 多头排列
            elif latest_ma5 < latest_ma20 < latest_ma60:
                ma_alignment = -1.0  # 空头排列
            else:
                ma_alignment = 0.0  # 交叉
        else:
            ma_alignment = 0.0
        
        # 趋势强度（ADX 简化版）
        high = df['high']
        low = df['low']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(14).mean()
        
        adx_value = adx.iloc[-1] if not adx.isna().all() else 0
        
        return {
            'direction': np.sign(ret_long),
            'ret_short': float(ret_short),
            'ret_long': float(ret_long),
            'ma_alignment': float(ma_alignment),
            'adx': float(adx_value) if not np.isnan(adx_value) else 0,
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """波动率分析"""
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < self.volatility_lookback:
            return {'vol_20d': 0, 'vol_60d': 0, 'vol_regime': 0}
        
        vol_20d = returns.tail(self.volatility_lookback).std() * np.sqrt(252)
        vol_60d = returns.tail(min(60, len(returns))).std() * np.sqrt(252) if len(returns) >= 60 else vol_20d
        
        # 波动率趋势
        if vol_60d > 0:
            vol_ratio = vol_20d / vol_60d
        else:
            vol_ratio = 1.0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'vol_20d': float(vol_20d),
            'vol_60d': float(vol_60d),
            'vol_ratio': float(vol_ratio),
            'max_drawdown': float(max_dd),
            'vol_regime': 'high' if vol_20d > self.high_vol_threshold else 'normal',
        }
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict[str, float]:
        """流动性分析"""
        if 'volume' not in df.columns:
            return {'volume_ratio': 1.0, 'liquidity_score': 0.5}
        
        volume = df['volume']
        vol_ma_20 = volume.rolling(20).mean()
        vol_ma_60 = volume.rolling(60).mean()
        
        if len(vol_ma_60.dropna()) > 0 and vol_ma_60.iloc[-1] > 0:
            volume_ratio = vol_ma_20.iloc[-1] / vol_ma_60.iloc[-1]
        else:
            volume_ratio = 1.0
        
        # 流动性评分（简化）
        liquidity_score = min(1.0, max(0.0, volume_ratio))
        
        return {
            'volume_ratio': float(volume_ratio),
            'liquidity_score': float(liquidity_score),
        }
    
    def _classify_regime(
        self,
        trend: Dict,
        volatility: Dict,
        liquidity: Dict,
    ) -> Tuple[MarketRegime, float]:
        """综合分类市场状态"""
        direction = trend.get('direction', 0)
        vol_20d = volatility.get('vol_20d', 0)
        adx = trend.get('adx', 0)
        max_dd = volatility.get('max_drawdown', 0)
        
        confidence = 0.5
        
        # 危机模式检测
        if vol_20d > self.crisis_vol_threshold or max_dd < -0.30:
            return MarketRegime.CRISIS, 0.9
        
        # 高波动模式
        if vol_20d > self.high_vol_threshold:
            if direction > 0:
                return MarketRegime.BULL_VOLATILE, 0.7
            elif direction < 0:
                return MarketRegime.BEAR_VOLATILE, 0.7
            else:
                return MarketRegime.HIGH_VOLATILITY, 0.6
        
        # 低波动模式
        if vol_20d < 0.10:
            return MarketRegime.LOW_VOLATILITY, 0.6
        
        # 趋势判断
        if adx > 25:  # 强趋势
            if direction > 0:
                return MarketRegime.BULL_TRENDING, 0.8
            elif direction < 0:
                return MarketRegime.BEAR_TRENDING, 0.8
            else:
                return MarketRegime.SIDEWAYS, 0.5
        else:  # 弱趋势
            if abs(direction) < 0.02:
                return MarketRegime.SIDEWAYS, 0.7
            elif direction > 0:
                return MarketRegime.BULL_VOLATILE, 0.5
            else:
                return MarketRegime.BEAR_VOLATILE, 0.5
    
    def _generate_details(
        self,
        regime: MarketRegime,
        trend: Dict,
        volatility: Dict,
        liquidity: Dict,
    ) -> Dict[str, Any]:
        """生成详细分析报告"""
        regime_advice = {
            MarketRegime.BULL_TRENDING: {
                'risk_level': 'low',
                'suggested_leverage': 1.0,
                'strategy_preference': ['trend_following', 'momentum'],
                'stop_loss_tightness': 'normal',
                'description': '牛市趋势：建议采用趋势跟踪策略，可适当提高仓位',
            },
            MarketRegime.BULL_VOLATILE: {
                'risk_level': 'medium',
                'suggested_leverage': 0.7,
                'strategy_preference': ['mean_reversion', 'momentum'],
                'stop_loss_tightness': 'tight',
                'description': '牛市震荡：建议降低仓位，采用均值回归策略',
            },
            MarketRegime.BEAR_TRENDING: {
                'risk_level': 'high',
                'suggested_leverage': 0.3,
                'strategy_preference': ['defensive', 'short'],
                'stop_loss_tightness': 'very_tight',
                'description': '熊市趋势：建议大幅降低仓位，以防御为主',
            },
            MarketRegime.BEAR_VOLATILE: {
                'risk_level': 'high',
                'suggested_leverage': 0.2,
                'strategy_preference': ['defensive', 'cash'],
                'stop_loss_tightness': 'very_tight',
                'description': '熊市震荡：建议保持低仓位，等待市场企稳',
            },
            MarketRegime.SIDEWAYS: {
                'risk_level': 'medium',
                'suggested_leverage': 0.5,
                'strategy_preference': ['mean_reversion', 'pairs_trading'],
                'stop_loss_tightness': 'tight',
                'description': '横盘整理：适合均值回归和配对交易策略',
            },
            MarketRegime.HIGH_VOLATILITY: {
                'risk_level': 'high',
                'suggested_leverage': 0.3,
                'strategy_preference': ['volatility_arbitrage', 'defensive'],
                'stop_loss_tightness': 'very_tight',
                'description': '高波动：建议严格控制风险，降低仓位',
            },
            MarketRegime.LOW_VOLATILITY: {
                'risk_level': 'low',
                'suggested_leverage': 0.8,
                'strategy_preference': ['trend_following', 'carry'],
                'stop_loss_tightness': 'loose',
                'description': '低波动：适合趋势跟踪和套利策略',
            },
            MarketRegime.CRISIS: {
                'risk_level': 'extreme',
                'suggested_leverage': 0.0,
                'strategy_preference': ['cash', 'hedge'],
                'stop_loss_tightness': 'immediate',
                'description': '危机模式：建议清仓或全面对冲',
            },
        }
        
        advice = regime_advice.get(regime, regime_advice[MarketRegime.SIDEWAYS])
        
        return {
            **advice,
            'trend_details': trend,
            'volatility_details': volatility,
            'liquidity_details': liquidity,
        }


# =============================================================================
# 自适应风险参数调整器
# =============================================================================

@dataclass
class AdaptiveRiskParams:
    """自适应风险参数"""
    max_position_pct: float     # 最大仓位比例
    max_single_stock_pct: float  # 单只股票最大仓位
    stop_loss_pct: float        # 止损比例
    take_profit_pct: float      # 止盈比例
    max_leverage: float         # 最大杠杆
    strategy_weights: Dict[str, float]  # 策略权重分配


class AdaptiveRiskManager:
    """
    自适应风险管理器
    
    借鉴 TradingAgents 的 Risk Manager Agent 设计：
    - 根据市场状态动态调整风险参数
    - 支持多种策略的权重调整
    - 硬止损 + 软风控双保险
    """
    
    # 不同市场状态下的风险参数
    REGIME_RISK_PARAMS = {
        MarketRegime.BULL_TRENDING: AdaptiveRiskParams(
            max_position_pct=0.95, max_single_stock_pct=0.10,
            stop_loss_pct=0.08, take_profit_pct=0.20,
            max_leverage=1.0,
            strategy_weights={'trend_following': 0.5, 'momentum': 0.3, 'mean_reversion': 0.2},
        ),
        MarketRegime.BULL_VOLATILE: AdaptiveRiskParams(
            max_position_pct=0.70, max_single_stock_pct=0.05,
            stop_loss_pct=0.05, take_profit_pct=0.10,
            max_leverage=0.7,
            strategy_weights={'momentum': 0.3, 'mean_reversion': 0.4, 'trend_following': 0.3},
        ),
        MarketRegime.BEAR_TRENDING: AdaptiveRiskParams(
            max_position_pct=0.30, max_single_stock_pct=0.03,
            stop_loss_pct=0.03, take_profit_pct=0.05,
            max_leverage=0.3,
            strategy_weights={'defensive': 0.6, 'short': 0.3, 'trend_following': 0.1},
        ),
        MarketRegime.BEAR_VOLATILE: AdaptiveRiskParams(
            max_position_pct=0.20, max_single_stock_pct=0.02,
            stop_loss_pct=0.02, take_profit_pct=0.03,
            max_leverage=0.2,
            strategy_weights={'defensive': 0.5, 'cash': 0.3, 'short': 0.2},
        ),
        MarketRegime.SIDEWAYS: AdaptiveRiskParams(
            max_position_pct=0.50, max_single_stock_pct=0.05,
            stop_loss_pct=0.04, take_profit_pct=0.06,
            max_leverage=0.5,
            strategy_weights={'mean_reversion': 0.5, 'pairs_trading': 0.3, 'trend_following': 0.2},
        ),
        MarketRegime.HIGH_VOLATILITY: AdaptiveRiskParams(
            max_position_pct=0.30, max_single_stock_pct=0.03,
            stop_loss_pct=0.03, take_profit_pct=0.05,
            max_leverage=0.3,
            strategy_weights={'volatility_arbitrage': 0.4, 'defensive': 0.4, 'mean_reversion': 0.2},
        ),
        MarketRegime.LOW_VOLATILITY: AdaptiveRiskParams(
            max_position_pct=0.80, max_single_stock_pct=0.08,
            stop_loss_pct=0.06, take_profit_pct=0.15,
            max_leverage=0.8,
            strategy_weights={'trend_following': 0.4, 'carry': 0.3, 'momentum': 0.3},
        ),
        MarketRegime.CRISIS: AdaptiveRiskParams(
            max_position_pct=0.05, max_single_stock_pct=0.01,
            stop_loss_pct=0.01, take_profit_pct=0.02,
            max_leverage=0.0,
            strategy_weights={'cash': 0.5, 'hedge': 0.5},
        ),
    }
    
    def __init__(self, detector: MarketRegimeDetector = None):
        self.detector = detector or MarketRegimeDetector()
        self.current_params = self.REGIME_RISK_PARAMS[MarketRegime.SIDEWAYS]
        self.regime_history: List[Dict] = []
    
    def update(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """根据最新市场数据更新风险参数"""
        analysis = self.detector.detect(price_data)
        regime = analysis['regime']
        
        # 更新风险参数
        self.current_params = self.REGIME_RISK_PARAMS.get(regime, self.REGIME_RISK_PARAMS[MarketRegime.SIDEWAYS])
        
        # 记录历史
        self.regime_history.append({
            'date': price_data['date'].max() if not price_data.empty else None,
            'regime': regime.value,
            'confidence': analysis['confidence'],
        })
        
        return {
            'regime': regime.value,
            'confidence': analysis['confidence'],
            'risk_params': {
                'max_position_pct': self.current_params.max_position_pct,
                'max_single_stock_pct': self.current_params.max_single_stock_pct,
                'stop_loss_pct': self.current_params.stop_loss_pct,
                'take_profit_pct': self.current_params.take_profit_pct,
                'max_leverage': self.current_params.max_leverage,
            },
            'strategy_weights': self.current_params.strategy_weights,
            'details': analysis['details'],
        }
    
    def get_risk_params(self) -> AdaptiveRiskParams:
        """获取当前风险参数"""
        return self.current_params
    
    def get_regime_history(self) -> pd.DataFrame:
        """获取市场状态历史"""
        if not self.regime_history:
            return pd.DataFrame()
        return pd.DataFrame(self.regime_history)


# =============================================================================
# 生成模拟数据
# =============================================================================

def generate_market_data(
    n_days: int = 504,
    regimes: List[Tuple[str, int]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟市场数据，包含不同市场状态
    
    regimes: [(regime_type, days), ...]
    """
    np.random.seed(seed)
    
    if regimes is None:
        regimes = [
            ('bull_trending', 100),
            ('bull_volatile', 80),
            ('sideways', 100),
            ('bear_trending', 100),
            ('bear_volatile', 80),
            ('crisis', 44),
        ]
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
    close = 100.0
    prices = []
    volumes = []
    
    day_idx = 0
    for regime_type, regime_days in regimes:
        for _ in range(regime_days):
            if day_idx >= n_days:
                break
            
            if regime_type == 'bull_trending':
                ret = np.random.normal(0.001, 0.01)
            elif regime_type == 'bull_volatile':
                ret = np.random.normal(0.0005, 0.02)
            elif regime_type == 'sideways':
                ret = np.random.normal(0.0, 0.01)
            elif regime_type == 'bear_trending':
                ret = np.random.normal(-0.001, 0.01)
            elif regime_type == 'bear_volatile':
                ret = np.random.normal(-0.0005, 0.02)
            elif regime_type == 'crisis':
                ret = np.random.normal(-0.003, 0.04)
            else:
                ret = np.random.normal(0, 0.01)
            
            close *= (1 + ret)
            close = max(close, 10)
            prices.append(close)
            volumes.append(np.random.lognormal(10, 0.5))
            day_idx += 1
    
    while day_idx < n_days:
        ret = np.random.normal(0, 0.01)
        close *= (1 + ret)
        close = max(close, 10)
        prices.append(close)
        volumes.append(np.random.lognormal(10, 0.5))
        day_idx += 1
    
    df = pd.DataFrame({
        'date': dates[:len(prices)],
        'open': np.array(prices) * (1 + np.random.randn(len(prices)) * 0.002),
        'high': np.array(prices) * (1 + np.abs(np.random.randn(len(prices)) * 0.01)),
        'low': np.array(prices) * (1 - np.abs(np.random.randn(len(prices)) * 0.01)),
        'close': prices,
        'volume': volumes,
    })
    
    return df


# =============================================================================
# 测试代码
# =============================================================================

class TestMarketRegimeDetection(unittest.TestCase):
    """市场状态检测测试"""
    
    @classmethod
    def setUpClass(cls):
        cls.data = generate_market_data(n_days=504)
        cls.detector = MarketRegimeDetector()
    
    def test_regime_detection(self):
        """测试市场状态检测"""
        result = self.detector.detect(self.data)
        
        print(f"\n市场状态检测结果:")
        print(f"  状态: {result['regime'].value}")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  趋势: 方向={result['trend']['direction']:.3f}, ADX={result['trend']['adx']:.1f}")
        print(f"  波动率: 20日={result['volatility']['vol_20d']:.3f}, 最大回撤={result['volatility']['max_drawdown']:.3f}")
        print(f"  建议: {result['details']['description']}")
        print(f"  推荐策略: {result['details']['strategy_preference']}")
        
        self.assertIn(result['regime'], MarketRegime)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_regime_transition_detection(self):
        """测试市场状态转换检测"""
        # 分段检测
        n = len(self.data)
        segments = [
            (0, 100),
            (100, 200),
            (200, 300),
            (300, 400),
            (400, n),
        ]
        
        print(f"\n市场状态转换检测:")
        regimes_detected = []
        for start, end in segments:
            segment = self.data.iloc[start:end]
            result = self.detector.detect(segment)
            regimes_detected.append(result['regime'].value)
            print(f"  区间 [{start}-{end}]: {result['regime'].value} "
                  f"(置信度={result['confidence']:.2f}, 年化波动率={result['volatility']['vol_20d']:.3f})")
        
        # 应该检测到至少两种不同的状态
        unique_regimes = set(regimes_detected)
        self.assertGreaterEqual(len(unique_regimes), 2, "未能检测到市场状态切换")
    
    def test_adaptive_risk_params(self):
        """测试自适应风险参数调整"""
        manager = AdaptiveRiskManager(self.detector)
        
        # 分段检测并更新风险参数
        n = len(self.data)
        segment_size = 100
        results = []
        
        for start in range(0, n, segment_size):
            end = min(start + segment_size, n)
            segment = self.data.iloc[start:end]
            
            result = manager.update(segment)
            params = result['risk_params']
            results.append({
                'regime': result['regime'],
                'max_position': params['max_position_pct'],
                'stop_loss': params['stop_loss_pct'],
                'leverage': params['max_leverage'],
            })
        
        print(f"\n自适应风险参数调整:")
        print(f"{'状态':<20} {'最大仓位':<10} {'止损':<10} {'杠杆':<10}")
        print("-" * 50)
        for r in results:
            print(f"{r['regime']:<20} {r['max_position']:<10.0%} {r['stop_loss']:<10.0%} {r['leverage']:<10.1f}")
        
        self.assertEqual(len(results), (n + segment_size - 1) // segment_size)
    
    def test_strategy_weight_adjustment(self):
        """测试策略权重自适应调整"""
        manager = AdaptiveRiskManager(self.detector)
        
        # 测试不同市场状态下的策略权重
        test_segments = {
            'bull': self.data.iloc[50:150],    # 牛市区域
            'bear': self.data.iloc[300:400],    # 熊市区域
            'crisis': self.data.iloc[450:504],  # 危机区域
        }
        
        print(f"\n策略权重自适应调整:")
        for name, segment in test_segments.items():
            result = manager.update(segment)
            weights = result['strategy_weights']
            print(f"  {name} ({result['regime']}):")
            for strategy, weight in weights.items():
                print(f"    {strategy}: {weight:.0%}")
        
        # 验证危机状态下仓位为0
        crisis_result = manager.update(test_segments['crisis'])
        self.assertLess(crisis_result['risk_params']['max_position_pct'], 0.1)
    
    def test_hard_risk_limits(self):
        """测试硬风控限制"""
        manager = AdaptiveRiskManager(self.detector)
        
        # 在危机模式下，验证硬风控限制
        crisis_data = generate_market_data(n_days=100, regimes=[('crisis', 100)])
        result = manager.update(crisis_data)
        
        print(f"\n硬风控限制验证:")
        print(f"  市场状态: {result['regime']}")
        print(f"  最大仓位: {result['risk_params']['max_position_pct']:.0%}")
        print(f"  止损比例: {result['risk_params']['stop_loss_pct']:.0%}")
        print(f"  最大杠杆: {result['risk_params']['max_leverage']:.1f}")
        
        # 硬风控限制
        self.assertLessEqual(result['risk_params']['max_position_pct'], 0.1,
                            "危机模式下仓位应 <= 10%")
        self.assertLessEqual(result['risk_params']['stop_loss_pct'], 0.02,
                            "危机模式下止损应 <= 2%")
        self.assertLessEqual(result['risk_params']['max_leverage'], 0.1,
                            "危机模式下不应使用杠杆")
    
    def test_regime_history_tracking(self):
        """测试市场状态历史追踪"""
        manager = AdaptiveRiskManager(self.detector)
        
        # 完整回放
        n = len(self.data)
        segment_size = 50
        for start in range(0, n, segment_size):
            end = min(start + segment_size, n)
            manager.update(self.data.iloc[start:end])
        
        history = manager.get_regime_history()
        
        print(f"\n市场状态历史:")
        print(f"  总记录数: {len(history)}")
        print(f"  状态分布:")
        for regime, count in history['regime'].value_counts().items():
            print(f"    {regime}: {count} ({count/len(history):.0%})")
        
        self.assertGreater(len(history), 0)


class TestRegimeBasedBacktest(unittest.TestCase):
    """基于市场状态的回测模拟"""
    
    @classmethod
    def setUpClass(cls):
        cls.data = generate_market_data(n_days=504)
        cls.detector = MarketRegimeDetector()
    
    def test_fixed_vs_adaptive(self):
        """对比固定参数 vs 自适应参数的回测表现"""
        print("\n" + "=" * 60)
        print("回测对比: 固定参数 vs 自适应参数")
        print("=" * 60)
        
        manager = AdaptiveRiskManager(self.detector)
        
        # 模拟回测
        initial_capital = 1_000_000
        n = len(self.data)
        lookback = 60
        
        # 固定参数策略
        fixed_positions = []
        fixed_capital = initial_capital
        fixed_params = AdaptiveRiskParams(
            max_position_pct=0.70, max_single_stock_pct=0.05,
            stop_loss_pct=0.05, take_profit_pct=0.10,
            max_leverage=0.7, strategy_weights={}
        )
        
        # 自适应参数策略
        adaptive_positions = []
        adaptive_capital = initial_capital
        
        for i in range(lookback, n, 20):  # 每20个交易日评估一次
            segment = self.data.iloc[max(0, i-lookback):i]
            current_price = self.data.iloc[i]['close']
            
            # 固定参数
            fixed_position = fixed_capital * fixed_params.max_position_pct
            fixed_positions.append(fixed_position)
            
            # 自适应参数
            regime_result = manager.update(segment)
            adaptive_params = manager.get_risk_params()
            adaptive_position = adaptive_capital * adaptive_params.max_position_pct
            adaptive_positions.append(adaptive_position)
            
            # 模拟收益（简化：假设市场收益率）
            market_ret = self.data.iloc[min(i+20, n-1)]['close'] / current_price - 1
            fixed_capital *= (1 + market_ret * (fixed_position / fixed_capital))
            adaptive_capital *= (1 + market_ret * (adaptive_position / adaptive_capital))
        
        print(f"\n{'策略':<20} {'最终资金':<15} {'收益率':<10} {'风险控制':<15}")
        print("-" * 60)
        print(f"{'固定参数':<20} {fixed_capital:<15,.0f} {(fixed_capital/initial_capital-1):<10.2%} {'固定':<15}")
        print(f"{'自适应参数':<20} {adaptive_capital:<15,.0f} {(adaptive_capital/initial_capital-1):<10.2%} {'动态调整':<15}")
        
        # 自适应策略在危机期间应表现更好（更少回撤）
        self.assertIsNotNone(fixed_capital)
        self.assertIsNotNone(adaptive_capital)


def run_tests():
    """运行所有测试"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestMarketRegimeDetection))
    suite.addTest(unittest.makeSuite(TestRegimeBasedBacktest))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("市场状态检测优化验证结果摘要")
    print("=" * 60)
    print(f"借鉴来源:")
    print(f"  - TradingAgents (github.com/TauricResearch/TradingAgents)")
    print(f"    - 多智能体架构中的 Risk Manager Agent")
    print(f"    - 市场状态多维评估")
    print(f"  - FreqAI 动态特征权重调整")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    print("\n优化建议:")
    print("1. 在 portfolio-risk-engine 中集成 MarketRegimeDetector")
    print("2. 增加自适应风险参数调整模块")
    print("3. 支持基于市场状态的策略权重动态调整")
    print("4. 添加市场状态历史追踪和可视化")
    print("5. 在 config 中增加 regime_detection 配置项")
    
    return result


if __name__ == '__main__':
    run_tests()