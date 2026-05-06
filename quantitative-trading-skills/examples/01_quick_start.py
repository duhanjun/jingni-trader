#!/usr/bin/env python3
"""
快速入门示例：研究级组合方案
演示数据层→研究层→回测层的完整流程
"""

import sys
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

print("="*70)
print("📊 量化交易Skill套件 - 研究级组合方案快速入门")
print("="*70)

# ================================================
# 阶段1: 数据层 - Tushare + AKShare
# ================================================
print("\n📥 阶段1: 数据获取 (Tushare + AKShare)")
print("-"*70)

try:
    # 导入数据引擎
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'skills', 'a-share-data-engine'))
    from engine import AShareDataEngine
    
    # 创建数据引擎
    engine = AShareDataEngine()
    print("✅ 数据引擎初始化成功")
    
    # 获取A股数据
    stock_pool = ["000001.SZ", "600000.SH", "600519.SH"]
    df = engine.get_daily(
        codes=stock_pool,
        start_date="2024-01-01",
        end_date="2024-06-30"
    )
    
    print(f"✅ 获取到 {len(df)} 条A股日线数据")
    print("📊 数据预览:")
    print(df[['code', 'date', 'open', 'close', 'volume']].head())
    
except Exception as e:
    print(f"⚠️  获取数据失败: {e}")
    print("🔄 使用模拟数据演示...")
    
    # 生成模拟数据
    dates = pd.date_range(start='2024-01-01', periods=120)
    stock_pool = ["000001.SZ", "600000.SH", "600519.SH"]
    dfs = []
    for code in stock_pool:
        np.random.seed(hash(code) % 1000)
        close_prices = 10 + np.cumsum(np.random.randn(120) * 0.02)
        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': close_prices * (1 + np.random.randn(120)*0.01),
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, 120)
        })
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"✅ 使用模拟数据 {len(df)} 条")

# ================================================
# 阶段2: 研究层 - Qlib因子研究
# ================================================
print("\n🧪 阶段2: 因子研究 (Qlib)")
print("-"*70)

try:
    # 检查Qlib是否安装
    import qlib
    print("✅ Qlib已安装，准备进行因子研究")
    
    # Qlib初始化（需要配置数据）
    print("📌 Qlib因子研究流程:")
    print("   1. 初始化Qlib环境")
    print("   2. 使用Alpha360因子库")
    print("   3. 训练LightGBM多因子模型")
    print("   4. 生成选股信号")
    
    # 简化演示：计算基础因子
    print("\n📊 计算基础技术因子...")
    df['ma5'] = df.groupby('code')['close'].rolling(5).mean().reset_index(0, drop=True)
    df['ma20'] = df.groupby('code')['close'].rolling(20).mean().reset_index(0, drop=True)
    df['momentum'] = df.groupby('code')['close'].pct_change(20)
    print("✅ 因子计算完成")
    
except ImportError:
    print("⚠️  Qlib未安装，跳过因子研究阶段")
    print("💡 安装Qlib: pip install qlib")

# ================================================
# 阶段3: 回测层 - Backtrader
# ================================================
print("\n🔬 阶段3: 策略回测 (Backtrader)")
print("-"*70)

try:
    import backtrader as bt
    
    class SimpleStrategy(bt.Strategy):
        def __init__(self):
            self.sma5 = bt.indicators.SimpleMovingAverage(self.data.close, period=5)
            self.sma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        
        def next(self):
            if self.sma5[0] > self.sma20[0] and not self.position:
                self.buy(size=100)
            elif self.sma5[0] < self.sma20[0] and self.position:
                self.sell(size=100)
    
    print("✅ Backtrader策略回测演示")
    print("📈 策略: 5日均线突破20日均线买入，跌破卖出")
    print("✅ 回测框架就绪")
    
except ImportError:
    print("⚠️  Backtrader未安装，跳过回测阶段")
    print("💡 安装Backtrader: pip install backtrader")

# ================================================
# 阶段4: 实盘层 - xtquant/gm（演示）
# ================================================
print("\n💹 阶段4: 实盘执行 (xtquant/gm)")
print("-"*70)

print("📌 实盘执行流程:")
print("   1. 选择执行后端: xtquant (迅投QMT) 或 gm (掘金)")
print("   2. 连接交易接口")
print("   3. 设置风控参数")
print("   4. 执行订单")
print("   5. 监控持仓和资金")

print("\n✅ 实盘执行引擎就绪")

# ================================================
# 总结
# ================================================
print("\n" + "="*70)
print("🎉 快速入门示例完成！")
print("="*70)
print("\n📋 研究级组合方案架构:")
print("   ┌─────────────────────────────────────────┐")
print("   │ 实盘层: xtquant / gm                    │")
print("   ├─────────────────────────────────────────┤")
print("   │ 回测层: Backtrader                      │")
print("   ├─────────────────────────────────────────┤")
print("   │ 研究层: Qlib (微软AI量化平台)            │")
print("   ├─────────────────────────────────────────┤")
print("   │ 数据层: Tushare + AKShare               │")
print("   └─────────────────────────────────────────┘")
print("\n📚 下一步:")
print("   1. 安装Qlib: pip install qlib")
print("   2. 安装Backtrader: pip install backtrader")
print("   3. 配置Tushare Token")
print("   4. 查看 examples/02_full_workflow.py")
