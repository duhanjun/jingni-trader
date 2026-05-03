import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import unittest

from engine import PortfolioRiskEngine
from config import get_config


class TestPortfolioRiskEngine(unittest.TestCase):
    """投资组合风险管理引擎测试"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.dates = pd.date_range(start='2023-01-01', periods=252)
        self.assets = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '600519.SH']
        self.returns = pd.DataFrame(
            np.random.normal(0, 0.02, (252, 5)),
            index=self.dates,
            columns=self.assets
        )
        
        self.config = get_config()
        self.engine = PortfolioRiskEngine(self.config)

    def test_config_initialization(self):
        """测试配置初始化"""
        self.assertEqual(self.config.MAX_SINGLE_WEIGHT, 0.10)
        self.assertEqual(self.config.CONFIDENCE_LEVEL, 0.95)
        self.assertEqual(self.config.RISK_FREE_RATE, 0.03)

    def test_covariance_estimation(self):
        """测试协方差矩阵估计"""
        # 测试历史协方差
        cov_historical = self.engine.estimate_covariance(self.returns, method='historical')
        self.assertEqual(cov_historical.shape, (5, 5))
        self.assertEqual(list(cov_historical.columns), self.assets)
        
        # 测试 EWMA 协方差
        cov_ewma = self.engine.estimate_covariance(self.returns, method='ewma')
        self.assertEqual(cov_ewma.shape, (5, 5))
        self.assertEqual(list(cov_ewma.columns), self.assets)

    def test_optimization_min_variance(self):
        """测试最小方差优化"""
        result = self.engine.optimize_portfolio(
            self.returns,
            method='min_variance'
        )
        
        weights = result['weights']
        self.assertEqual(len(weights), 5)
        
        # 检查权重和是否为1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)
        
        # 检查单个权重不超过限制
        for w in weights.values():
            self.assertLessEqual(w, self.config.MAX_SINGLE_WEIGHT + 1e-8)

    def test_optimization_max_sharpe(self):
        """测试最大夏普比率优化"""
        result = self.engine.optimize_portfolio(
            self.returns,
            method='max_sharpe'
        )
        
        weights = result['weights']
        performance = result['performance']
        
        self.assertEqual(len(weights), 5)
        self.assertIn('volatility', performance)
        self.assertIn('sharpe_ratio', performance)
        
        # 检查权重和是否为1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_optimization_risk_parity(self):
        """测试风险平价优化"""
        result = self.engine.optimize_portfolio(
            self.returns,
            method='risk_parity'
        )
        
        weights = result['weights']
        self.assertEqual(len(weights), 5)
        
        # 检查权重和是否为1
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)

    def test_var_calculation(self):
        """测试 VaR 计算"""
        # 使用等权重
        weights = {asset: 0.2 for asset in self.assets}
        
        # 测试历史模拟法
        var_result_hist = self.engine.calculate_var(
            self.returns, weights, method='historical'
        )
        self.assertIn('var', var_result_hist)
        self.assertIn('cvar', var_result_hist)
        self.assertLess(var_result_hist['var'], 0)  # VaR 应该是负数（亏损）
        
        # 测试参数法
        var_result_param = self.engine.calculate_var(
            self.returns, weights, method='parametric'
        )
        self.assertIn('var', var_result_param)
        self.assertIn('cvar', var_result_param)

    def test_constraint_checking(self):
        """测试约束检查"""
        # 创建一些权重
        weights = {
            '000001.SZ': 0.15,  # 超过限制
            '000002.SZ': 0.20,  # 超过限制
            '600000.SH': 0.25,
            '600036.SH': 0.20,
            '600519.SH': 0.20
        }
        
        result = self.engine.constraint_handler.check_single_stock_constraint(weights)
        self.assertFalse(result['valid'])
        self.assertEqual(len(result['violations']), 2)

    def test_constraint_application(self):
        """测试约束应用"""
        # 创建一些权重
        weights = {
            '000001.SZ': 0.15,
            '000002.SZ': 0.20,
            '600000.SH': 0.25,
            '600036.SH': 0.20,
            '600519.SH': 0.20
        }
        
        adjusted_weights = self.engine.constraint_handler.apply_single_stock_constraint(weights)
        
        # 检查调整后的权重不超过限制
        for w in adjusted_weights.values():
            self.assertLessEqual(w, self.config.MAX_SINGLE_WEIGHT + 1e-8)
        
        # 检查权重和是否为1
        self.assertAlmostEqual(sum(adjusted_weights.values()), 1.0, places=4)

    def test_analyze_risk(self):
        """测试全面风险分析"""
        result = self.engine.analyze_risk(self.returns)
        
        self.assertIn('weights', result)
        self.assertIn('covariance_matrix', result)
        self.assertIn('var', result)
        self.assertIn('constraints', result)
        
        # 检查约束是否满足
        self.assertTrue(result['constraints']['all_valid'])

    def test_stop_loss_check(self):
        """测试止损检查"""
        # 组合收益率
        portfolio_returns = pd.Series(
            np.random.normal(0, 0.02, 252),
            index=self.dates
        )
        
        # 测试不触发止损
        result = self.engine.check_stop_loss(portfolio_returns)
        self.assertIn('portfolio', result)
        
        # 测试触发止损
        portfolio_returns.iloc[-1] = -0.05  # 单日回撤5%
        result = self.engine.check_stop_loss(portfolio_returns)
        self.assertTrue(result['portfolio']['triggered'])

    def test_risk_monitoring(self):
        """测试风险监控"""
        positions = {
            '000001.SZ': 1000000,
            '000002.SZ': 1000000,
            '600000.SH': 1000000,
            '600036.SH': 1000000,
            '600519.SH': 1000000
        }
        equity = 10000000
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        
        result = self.engine.monitor_risk(positions, equity, returns)
        
        self.assertTrue(result['all_ok'])
        self.assertIn('position', result)
        self.assertIn('margin', result)
        self.assertIn('profit', result)


if __name__ == '__main__':
    unittest.main()
