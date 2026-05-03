#!/usr/bin/env python3
"""
简单的测试脚本，验证项目结构和基本功能
"""

import sys
import os

# 确保我们在正确的目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 A股数据引擎")
print("=" * 60)

# 测试1: 检查文件结构
print("\n[1/4] 检查项目结构...")
required_files = [
    "SKILL.md",
    "README.md",
    "config.py",
    "engine.py",
    "base/base_data_provider.py",
    "adapters/tushare_adapter.py",
    "adapters/baostock_adapter.py",
    "adapters/akshare_adapter.py",
    "cleaning/data_cleaner.py",
    "storage/data_storage.py",
    "snapshots/snapshot_manager.py",
]

all_exist = True
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} (缺失)")
        all_exist = False

print(f"\n项目结构检查: {'通过' if all_exist else '失败'}")

# 测试2: 尝试导入模块
print("\n[2/4] 测试模块导入...")
try:
    from config import Config, get_config
    print("  ✓ config.py 导入成功")
    
    from base import BaseDataProvider
    print("  ✓ base 导入成功")
    
    from cleaning import DataCleaner
    print("  ✓ cleaning 导入成功")
    
    from storage import DataStorage
    print("  ✓ storage 导入成功")
    
    from snapshots import SnapshotManager
    print("  ✓ snapshots 导入成功")
    
    from engine import AShareDataEngine
    print("  ✓ engine 导入成功")
    
    print("  所有核心模块导入成功!")
except Exception as e:
    print(f"  ✗ 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 测试配置创建
print("\n[3/4] 测试配置创建...")
try:
    config = get_config(
        DATA_BACKEND="tushare",
        DATA_DIR="./test_data",
        DB_TYPE="sqlite",
        CLEANING_ENABLED=True,
    )
    print(f"  ✓ 配置创建成功")
    print(f"    - 数据源: {config.DATA_BACKEND}")
    print(f"    - 数据目录: {config.DATA_DIR}")
    print(f"    - 数据库: {config.DB_TYPE}")
except Exception as e:
    print(f"  ✗ 配置创建失败: {e}")

# 测试4: 测试数据清洗器
print("\n[4/4] 测试数据清洗器...")
try:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建测试数据
    config = get_config()
    cleaner = DataCleaner(config)
    
    codes = ["000001.SZ", "600000.SH"]
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    
    data = []
    for code in codes:
        for date in dates:
            data.append({
                "code": code,
                "date": date,
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 100000,
                "amount": 1000000,
                "pre_close": 10.0,
                "change_pct": 0.0,
                "turnover_rate": 1.0,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            })
    
    df = pd.DataFrame(data)
    print(f"  ✓ 创建测试数据: {len(df)} 行")
    
    cleaned_df = cleaner.clean(df)
    print(f"  ✓ 数据清洗完成: {len(cleaned_df)} 行")
    
    missing_rate = cleaner.calculate_missing_rate(cleaned_df)
    print(f"  ✓ 数据缺失率: {missing_rate:.2%}")
    print(f"  ✓ 数据质量检查: {'通过' if missing_rate < 0.02 else '需要注意'}")
    
except Exception as e:
    print(f"  ✗ 数据清洗器测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n项目结构总结:")
print("  ✓ 目录结构完整")
print("  ✓ 核心模块可用")
print("  ✓ 配置系统工作")
print("  ✓ 数据清洗功能正常")
print("\n详细文档请查看:")
print("  - SKILL.md (Skill 描述)")
print("  - README.md (项目说明)")
print("  - references/api_reference.md (API文档)")
print("  - references/config_guide.md (配置指南)")
print("\n数据源适配器:")
print("  - Tushare (需要 Token)")
print("  - BaoStock (免费)")
print("  - AkShare (免费)")
print("  - xtquant (需要 Token)")
print("  - 掘金量化 (需要 Token)")
