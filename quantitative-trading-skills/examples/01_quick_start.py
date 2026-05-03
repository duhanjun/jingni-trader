#!/usr/bin/env python3
"""
快速入门示例：5分钟上手量化交易
演示如何获取数据并进行简单分析
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 添加skill路径
SKILLS_PATH = os.path.join(PROJECT_ROOT, 'skills')

print("="*60)
print("量化交易Skill套件 - 快速入门示例")
print("="*60)

print("\n步骤1: 初始化数据引擎...")
sys.path.insert(0, os.path.join(SKILLS_PATH, 'a-share-data-engine'))
try:
    from a_share_data_engine import AShareDataEngine, get_config
    
    config = get_config(DATA_BACKEND="baostock")
    data_engine = AShareDataEngine(config)
    print("✅ 数据引擎初始化成功！")
except Exception as e:
    print(f"⚠️  数据引擎初始化: {e}")
    print("使用模拟数据演示...")
    data_engine = None

print("\n步骤2: 获取或生成示例数据...")
if data_engine:
    try:
        stock_pool = ["000001.SZ"]
        df = data_engine.get_daily(
            codes=stock_pool,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        print(f"✅ 获取到 {len(df)} 条数据")
        print(df.head())
    except Exception as e:
        print(f"⚠️  获取数据: {e}")
        print("使用模拟数据...")
        # 创建模拟数据
        dates = pd.date_range(start='2024-01-01', periods=252)
        np.random.seed(42)
        close_prices = 10 + np.cumsum(np.random.randn(252) * 0.02)
        df = pd.DataFrame({
            'date': dates,
            'code': '000001.SZ',
            'open': close_prices * (1 + np.random.randn(252)*0.01),
            'high': close_prices * (1 + np.random.randn(252)*0.01),
            'low': close_prices * (1 - np.random.randn(252)*0.01),
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, 252)
        })
else:
    # 创建模拟数据
    dates = pd.date_range(start='2024-01-01', periods=252)
    np.random.seed(42)
    close_prices = 10 + np.cumsum(np.random.randn(252) * 0.02)
    df = pd.DataFrame({
        'date': dates,
        'code': '000001.SZ',
        'open': close_prices * (1 + np.random.randn(252)*0.01),
        'high': close_prices * (1 + np.random.randn(252)*0.01),
        'low': close_prices * (1 - np.random.randn(252)*0.01),
        'close': close_prices,
        'volume': np.random.randint(100000, 1000000, 252)
    })
    print(f"✅ 使用模拟数据 {len(df)} 条")
    print(df.head())

print("\n步骤3: 计算技术指标...")
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma20'] = df['close'].rolling(window=20).mean()

print("\n步骤4: 生成交易信号...")
df['signal'] = 0
df.loc[df['close'] > df['ma5'], 'signal'] = 1
df.loc[df['close'] < df['ma5'], 'signal'] = -1

print("\n步骤5: 计算策略收益...")
df['return'] = df['close'].pct_change()
df['strategy_return'] = df['signal'].shift(1) * df['return']

df['cum_return'] = (1 + df['return']).cumprod()
df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()

print("\n策略回测结果:")
print(f"持有期收益率: {((df['cum_return'].iloc[-1] - 1)*100):.2f}%")
print(f"策略收益率: {((df['cum_strategy_return'].iloc[-1] - 1)*100):.2f}%")

print("\n" + "="*60)
print("🎉 快速入门示例完成！")
print("下一步: 阅读 examples/02_full_workflow.py 查看完整工作流程")
print("="*60)
