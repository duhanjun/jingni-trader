import sys
import os
import pandas as pd
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import AShareFactorEngine
from config import get_config


def generate_test_data():
    """
    生成测试数据
    """
    codes = [f"00000{i}.SZ" for i in range(1, 6)]
    dates = pd.date_range(start="2024-01-01", end="2024-03-31", freq="B")
    
    data = []
    for code in codes:
        base_price = 10 + np.random.rand() * 10
        for date in dates:
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)
            data.append({
                "code": code,
                "date": date,
                "open": base_price * (1 + np.random.normal(0, 0.01)),
                "high": base_price * (1 + abs(np.random.normal(0, 0.02))),
                "low": base_price * (1 - abs(np.random.normal(0, 0.02))),
                "close": base_price,
                "volume": np.random.randint(100000, 1000000),
                "amount": base_price * np.random.randint(100000, 1000000),
                "turnover_rate": np.random.uniform(0.01, 0.1)
            })
    
    return pd.DataFrame(data)


def generate_forward_returns(price_data, periods=5):
    """
    生成未来收益
    """
    df = price_data.copy().sort_values(["code", "date"])
    df["forward_return"] = df.groupby("code")["close"].pct_change(periods).shift(-periods)
    return df[["code", "date", "forward_return"]].dropna()


class TestAShareFactorEngine(unittest.TestCase):
    """
    A股因子引擎测试
    """

    @classmethod
    def setUpClass(cls):
        cls.config = get_config()
        cls.engine = AShareFactorEngine(cls.config)
        cls.price_data = generate_test_data()
        cls.forward_returns = generate_forward_returns(cls.price_data)

    def test_compute_factors(self):
        """
        测试因子计算
        """
        factors = self.engine.compute_factors(
            self.price_data,
            factor_list=["momentum_1m", "lncap", "turnover"]
        )
        
        self.assertFalse(factors.empty)
        self.assertIn("code", factors.columns)
        self.assertIn("date", factors.columns)
        self.assertIn("momentum_1m", factors.columns)
        self.assertIn("lncap", factors.columns)
        self.assertIn("turnover", factors.columns)

    def test_analyze_ic(self):
        """
        测试IC分析
        """
        factors = self.engine.compute_factors(self.price_data)
        ic_report = self.engine.analyze_ic(factors, self.forward_returns)
        
        self.assertIn("ic_series", ic_report)
        self.assertIn("ic_mean", ic_report)
        self.assertIn("icir", ic_report)
        self.assertFalse(ic_report["ic_series"].empty)

    def test_analyze_correlation(self):
        """
        测试相关性分析
        """
        factors = self.engine.compute_factors(self.price_data)
        corr_report = self.engine.analyze_correlation(factors)
        
        self.assertIn("correlation_matrix", corr_report)
        self.assertIn("redundant_pairs", corr_report)
        self.assertFalse(corr_report["correlation_matrix"].empty)

    def test_combine_factors(self):
        """
        测试因子融合
        """
        factors = self.engine.compute_factors(self.price_data)
        combined = self.engine.combine_factors(factors, method="equal")
        
        self.assertFalse(combined.empty)
        self.assertIn("combined_alpha", combined.columns)

    def test_save_and_load_factors(self):
        """
        测试因子存储和加载
        """
        factors = self.engine.compute_factors(self.price_data)
        save_path = self.engine.save_factors(factors, "test_factors.parquet")
        self.assertTrue(os.path.exists(save_path))
        
        loaded = self.engine.load_factors("test_factors.parquet")
        self.assertFalse(loaded.empty)
        self.assertEqual(len(factors), len(loaded))

    def test_full_workflow(self):
        """
        测试完整工作流
        """
        result = self.engine.full_workflow(
            self.price_data,
            self.forward_returns,
            save_filename="workflow_test.parquet"
        )
        
        self.assertIn("factors", result)
        self.assertIn("ic_report", result)
        self.assertIn("corr_report", result)
        self.assertIn("combined_alpha", result)


if __name__ == "__main__":
    unittest.main()
