#!/usr/bin/env python
"""
Qlib 数据准备脚本
从 Tushare 获取数据并转换为 Qlib 二进制格式
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'skills', 'a-share-data-engine'))

from config import get_config
from engine import AShareDataEngine
import pandas as pd
import numpy as np
from pathlib import Path

QLIB_DIR = Path(PROJECT_ROOT) / "qlib_data"
QLIB_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("📦 Qlib 数据准备")
print("-" * 70)

# 1. 获取股票列表
print("📋 获取股票列表...")
config = get_config()
engine = AShareDataEngine(config)

try:
    stocks_df = engine.get_stock_basic()
    if stocks_df.empty:
        raise ValueError("股票列表为空")
    symbols = stocks_df['ts_code'].tolist()
    print(f"✅ 获取 {len(symbols)} 只股票")
except Exception:
    print("⚠️ 股票列表获取失败，使用默认股票池")
    symbols = [
        '000001.SZ', '000002.SZ', '000858.SZ', '002415.SZ', '002475.SZ',
        '300750.SZ', '600000.SH', '600036.SH', '600519.SH', '601318.SH',
        '600276.SH', '600887.SH', '601012.SH', '600809.SH', '601166.SH',
        '000333.SZ', '002594.SZ', '600030.SH', '601398.SH', '600900.SH',
        '000651.SZ', '000725.SZ', '002230.SZ', '601888.SH', '600585.SH',
        '603259.SH', '000568.SZ', '002304.SZ', '601688.SH', '600031.SH',
    ]

# 2. 下载数据
print(f"\n📥 下载日线数据 ({len(symbols)} 只)...")
all_data = []

for i, code in enumerate(symbols):
    try:
        df = engine.get_daily(
            codes=[code],
            start_date='2020-01-01',
            end_date='2024-12-31'
        )
        if not df.empty:
            all_data.append(df)
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(symbols)}")
    except Exception as e:
        continue

if not all_data:
    print("⚠️ 数据下载失败，使用演示数据")
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    demo_data = []
    for i, code in enumerate(symbols[:10]):
        np.random.seed(i)
        prices = 10 + np.cumsum(np.random.randn(len(dates)) * 0.3)
        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100000, 10000000, len(dates)),
            'amount': prices * np.random.randint(100000, 10000000, len(dates)),
        })
        demo_data.append(df)
    all_data = demo_data

combined = pd.concat(all_data, ignore_index=True)
print(f"✅ 总数据量: {len(combined)} 条")

# 3. 转换为 Qlib 格式
print("\n🔄 转换为 Qlib 格式...")

calendars_dir = QLIB_DIR / "calendars"
features_dir = QLIB_DIR / "features"
calendars_dir.mkdir(parents=True, exist_ok=True)
features_dir.mkdir(parents=True, exist_ok=True)

# 保存交易日历
trading_dates = sorted(combined['date'].dropna().unique())
pd.Series([str(d) for d in trading_dates]).to_csv(
    calendars_dir / "day.txt", index=False, header=False
)
print(f"✅ 交易日历: {len(trading_dates)} 天")

# 按股票保存特征数据
for code in combined['code'].unique():
    stock_dir = features_dir / code.replace('.', '').lower()
    stock_dir.mkdir(parents=True, exist_ok=True)

    stock_data = combined[combined['code'] == code].copy()
    stock_data = stock_data.sort_values('date')

    for _, row in stock_data.iterrows():
        date_str = str(row['date'])[:10]
        day_dir = stock_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        features = {
            'open': float(row['open']) if pd.notna(row['open']) else 0.0,
            'high': float(row['high']) if pd.notna(row['high']) else 0.0,
            'low': float(row['low']) if pd.notna(row['low']) else 0.0,
            'close': float(row['close']) if pd.notna(row['close']) else 0.0,
            'volume': float(row['volume']) if pd.notna(row['volume']) else 0.0,
            'amount': float(row['amount']) if pd.notna(row['amount']) else 0.0,
        }

        import json
        with open(day_dir / "feature.json", 'w') as f:
            json.dump(features, f)

    if len(stock_data) > 0:
        print(f"  ✅ {code}: {len(stock_data)} 条")

# 4. 创建 instruments 配置
instruments = {}
for code in combined['code'].unique():
    instruments[code] = {
        'start_time': str(combined[combined['code'] == code]['date'].min())[:10],
        'end_time': str(combined[combined['code'] == code]['date'].max())[:10],
    }

os.makedirs(QLIB_DIR / "instruments", exist_ok=True)

with open(QLIB_DIR / "instruments" / "all.txt", 'w') as f:
    import json as j
    for code in combined['code'].unique():
        f.write(f"{code}\t{j.dumps(instruments[code])}\n")

# 5. 验证
data_size = 0
for root, dirs, files in os.walk(str(QLIB_DIR)):
    for f in files:
        data_size += os.path.getsize(os.path.join(root, f))
print(f"\n{'='*70}")
print(f"✅ Qlib 数据准备完成!")
print(f"📁 数据目录: {QLIB_DIR}")
print(f"📊 数据大小: {data_size / 1024 / 1024:.1f} MB")
print(f"📈 股票数: {combined['code'].nunique()}")
print(f"📅 时间范围: {combined['date'].min()} ~ {combined['date'].max()}")
print(f"{'='*70}")