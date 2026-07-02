import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config
from cleaning import DataCleaner


def create_test_data():
    """创建测试数据"""
    codes = ["000001.SZ", "600000.SH"]
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    
    data = []
    for code in codes:
        for date in dates:
            data.append({
                "code": code,
                "date": date,
                "open": 10.0 + np.random.randn(),
                "high": 11.0 + np.random.randn(),
                "low": 9.0 + np.random.randn(),
                "close": 10.5 + np.random.randn(),
                "volume": 100000 + np.random.randint(0, 50000),
                "amount": 1000000 + np.random.randint(0, 500000),
                "pre_close": 10.0,
                "change_pct": np.random.uniform(-5, 5),
                "turnover_rate": np.random.uniform(0, 5),
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            })
    
    # 加入一些缺失值和特殊情况
    df = pd.DataFrame(data)
    df.loc[5, "volume"] = 0  # 停牌
    df.loc[10, "change_pct"] = 10.0  # 涨停
    df.loc[15, "change_pct"] = -10.0  # 跌停
    
    return df


def test_data_cleaner():
    """测试数据清洗器"""
    print("=" * 50)
    print("测试数据清洗器")
    print("=" * 50)
    
    config = Config(FILTER_ST=False, FILTER_NEW_STOCK_DAYS=0)
    cleaner = DataCleaner(config)
    
    df = create_test_data()
    print(f"\n原始数据: {len(df)} 行")
    
    cleaned_df = cleaner.clean(df)
    print(f"清洗后数据: {len(cleaned_df)} 行")
    
    # 检查字段
    required_fields = ["code", "date", "open", "high", "low", "close", "volume", "is_suspended"]
    for field in required_fields:
        assert field in cleaned_df.columns, f"缺少字段: {field}"
    print("✓ 字段检查通过")
    
    # 检查停牌标记
    suspended_count = (cleaned_df["volume"] == 0).sum()
    assert "is_suspended" in cleaned_df.columns
    print(f"✓ 停牌标记: {suspended_count} 条")
    
    # 检查缺失率
    missing_rate = cleaner.calculate_missing_rate(cleaned_df)
    print(f"✓ 数据缺失率: {missing_rate:.2%}")
    assert missing_rate < 0.02, f"缺失率过高: {missing_rate:.2%}"
    
    print("\n✓ 数据清洗器测试通过")
    return True


def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("测试配置")
    print("=" * 50)
    
    config = Config(
        DATA_BACKEND="tushare",
        DATA_DIR="./test_data",
        DB_TYPE="sqlite",
        CLEANING_ENABLED=True,
    )
    
    assert config.DATA_BACKEND == "tushare"
    assert config.DATA_DIR == "./test_data"
    assert config.DB_TYPE == "sqlite"
    assert config.CLEANING_ENABLED == True
    
    print("✓ 配置测试通过")
    return True


if __name__ == "__main__":
    try:
        test_config()
        test_data_cleaner()
        print("\n" + "=" * 50)
        print("所有测试通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
