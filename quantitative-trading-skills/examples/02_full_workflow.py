#!/usr/bin/env python3
"""
完整量化工作流程示例
演示如何使用多个Skill进行完整的量化投研
"""

import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
SKILLS_PATH = os.path.join(PROJECT_ROOT, 'skills')

print("="*60)
print("量化交易Skill套件 - 完整工作流程示例")
print("="*60)

print("\n[阶段1] 数据获取与预处理...")
sys.path.insert(0, os.path.join(SKILLS_PATH, 'a-share-data-engine'))
try:
    from a_share_data_engine import AShareDataEngine, get_config
    
    config = get_config(DATA_BACKEND="baostock")
    data_engine = AShareDataEngine(config)
    print("✅ 数据引擎初始化成功")
    
    stock_pool = ["000001.SZ", "600000.SH", "600519.SH"]
    
    # 生成模拟数据
    dates = pd.date_range(start='2023-01-01', periods=252)
    dfs = []
    for code in stock_pool:
        np.random.seed(hash(code) % 1000)
        base_price = 10 + np.random.randn() * 5
        close_prices = base_price + np.cumsum(np.random.randn(252) * 0.02)
        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': close_prices * (1 + np.random.randn(252)*0.01),
            'high': close_prices * (1 + np.random.randn(252)*0.01),
            'low': close_prices * (1 - np.random.randn(252)*0.01),
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, 252)
        })
        dfs.append(df)
    price_data = pd.concat(dfs, ignore_index=True)
    print(f"✅ 准备数据: {len(price_data)} 条")
    
except Exception as e:
    print(f"⚠️  数据引擎: {e}")
    print("使用模拟数据")
    # 生成模拟数据
    dates = pd.date_range(start='2023-01-01', periods=252)
    stock_pool = ["000001.SZ", "600000.SH", "600519.SH"]
    dfs = []
    for code in stock_pool:
        np.random.seed(hash(code) % 1000)
        base_price = 10 + np.random.randn() * 5
        close_prices = base_price + np.cumsum(np.random.randn(252) * 0.02)
        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': close_prices * (1 + np.random.randn(252)*0.01),
            'high': close_prices * (1 + np.random.randn(252)*0.01),
            'low': close_prices * (1 - np.random.randn(252)*0.01),
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, 252)
        })
        dfs.append(df)
    price_data = pd.concat(dfs, ignore_index=True)
    print(f"✅ 使用模拟数据: {len(price_data)} 条")

print("\n[阶段2] 因子构建与分析...")
sys.path.insert(0, os.path.join(SKILLS_PATH, 'a-share-factor-engine'))
try:
    from a_share_factor_engine import AShareFactorEngine
    from a_share_factor_engine.config import get_config
    
    factor_config = get_config()
    factor_engine = AShareFactorEngine(factor_config)
    print("✅ 因子引擎初始化成功")
    
    print("📊 计算因子...")
    # 简单计算几个因子
    factors = price_data.pivot(index='date', columns='code', values='close').pct_change(20)
    factors = factors.stack().reset_index(name='momentum')
    print(f"✅ 因子计算完成: {len(factors)} 条")
    
except Exception as e:
    print(f"⚠️  因子引擎: {e}")
    print("跳过因子构建")

print("\n[阶段3] 组合优化...")
sys.path.insert(0, os.path.join(SKILLS_PATH, 'portfolio-risk-engine'))
try:
    from portfolio_risk_engine import PortfolioRiskEngine
    from portfolio_risk_engine.config import get_config
    
    risk_config = get_config()
    risk_engine = PortfolioRiskEngine(risk_config)
    print("✅ 风险引擎初始化成功")
    
    # 计算收益率矩阵
    returns_df = price_data.pivot(index='date', columns='code', values='close').pct_change().dropna()
    
    # 简单等权组合
    weights = {code: 1.0/len(stock_pool) for code in stock_pool}
    
    print("📊 组合权重:")
    for code, w in weights.items():
        print(f"  {code}: {w:.2%}")
    
    # 计算组合收益
    portfolio_return = (returns_df * pd.Series(weights)).sum(axis=1)
    cum_return = (1 + portfolio_return).cumprod()
    print(f"\n组合累积收益: {((cum_return.iloc[-1] - 1)*100):.2f}%")
    
except Exception as e:
    print(f"⚠️  风险引擎: {e}")
    print("跳过组合优化")

print("\n[阶段4] 模拟交易...")
sys.path.insert(0, os.path.join(SKILLS_PATH, 'execution-monitor-engine'))
try:
    from execution_monitor_engine import ExecutionEngine
    from execution_monitor_engine.config import get_config
    
    exec_config = get_config(EXECUTION_BACKEND="sim", PAPER_TRADE="true")
    exec_engine = ExecutionEngine(exec_config)
    print("✅ 执行引擎初始化成功")
    
    exec_engine.initialize_account(initial_capital=1000000)
    print("💼 模拟账户初始化成功")
    
    # 简单买入第一个股票
    first_code = stock_pool[0]
    print(f"📤 模拟买入 {first_code}")
    # 在实际环境中会有交易执行逻辑
    
except Exception as e:
    print(f"⚠️  执行引擎: {e}")
    print("跳过模拟交易")

print("\n" + "="*60)
print("🎉 完整工作流程示例完成！")
print("下一步:")
print("  - 阅读 README.md 了解项目总览")
print("  - 查看各个Skill的独立README了解详细功能")
print("  - 运行 tests/ 目录下的测试验证系统功能")
print("="*60)
