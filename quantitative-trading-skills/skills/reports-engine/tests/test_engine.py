"""
reports-engine 测试
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from engine import ReportsEngine
from config import get_config


def test_basic_functionality():
    """测试基本功能"""
    print("测试 reports-engine 基本功能...")
    
    # 创建配置
    config = get_config(REPORT_DIR="./test_reports")
    
    # 创建引擎
    engine = ReportsEngine(config)
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    portfolio_returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),
        index=dates,
        name="portfolio"
    )
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)),
        index=dates,
        name="benchmark"
    )
    
    # 模拟因子暴露
    factor_exposures = pd.DataFrame(
        np.random.normal(0, 0.5, (len(dates), 3)),
        index=dates,
        columns=["size", "value", "growth"]
    )
    
    # 模拟行业权重
    industries = ["银行", "非银金融", "医药", "电子", "计算机", "食品饮料"]
    industry_weights = pd.DataFrame(
        np.random.dirichlet(np.ones(6), len(dates)),
        index=dates,
        columns=industries
    )
    
    # 模拟IC序列
    ic_series = pd.Series(
        np.random.normal(0.05, 0.1, len(dates)),
        index=dates
    )
    
    # 拆分样本内外
    split_date = "2023-07-01"
    is_returns = portfolio_returns.loc[:split_date]
    oos_returns = portfolio_returns.loc[split_date:]
    
    # 生成报告
    print("生成完整报告...")
    report = engine.generate_full_report(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        factor_exposures=factor_exposures,
        industry_weights=industry_weights,
        is_returns=is_returns,
        oos_returns=oos_returns,
        ic_series=ic_series,
        report_name="test_report"
    )
    
    print(f"报告生成完成: {report['report_name']}")
    print(f"关键指标: {report['sections']['quantstats']['key_metrics']}")
    print("测试通过！")
    
    return report


if __name__ == "__main__":
    test_basic_functionality()
