import pandas as pd
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import Config, get_config
from engine import AShareDataEngine


def test_config_creation():
    """测试配置创建"""
    print("=" * 50)
    print("测试引擎配置")
    print("=" * 50)
    
    config = get_config(
        DATA_BACKEND="xtquant",  # 使用占位适配器，无外部依赖
        DATA_DIR="./test_data",
    )
    
    assert config.DATA_BACKEND == "xtquant"
    assert config.DATA_DIR == "./test_data"
    print("✓ 配置创建成功")


def test_engine_creation():
    """测试引擎创建"""
    print("\n" + "=" * 50)
    print("测试引擎创建")
    print("=" * 50)
    
    # 使用占位适配器测试
    config = get_config(DATA_BACKEND="xtquant")
    engine = AShareDataEngine(config)
    assert engine is not None
    print("✓ 引擎创建成功")


def test_quality_check():
    """测试数据质量检查"""
    print("\n" + "=" * 50)
    print("测试数据质量检查")
    print("=" * 50)
    
    config = get_config(DATA_BACKEND="xtquant")
    engine = AShareDataEngine(config)
    
    # 创建模拟数据
    df = pd.DataFrame({
        "code": ["000001.SZ"] * 10,
        "date": pd.date_range("2024-01-01", periods=10),
        "open": [10.0] * 10,
        "high": [11.0] * 10,
        "low": [9.0] * 10,
        "close": [10.5] * 10,
        "volume": [100000] * 10,
        "amount": [1000000] * 10,
    })
    
    quality = engine.check_data_quality(df)
    print(f"数据质量报告:")
    print(f"  - 总行数: {quality['total_rows']}")
    print(f"  - 唯一代码数: {quality['unique_codes']}")
    print(f"  - 数据缺失率: {quality.get('missing_rate', 'N/A')}")
    print(f"  - 通过缺失检查: {quality.get('pass_missing_check', 'N/A')}")
    
    print("✓ 数据质量检查完成")


if __name__ == "__main__":
    try:
        test_config_creation()
        test_engine_creation()
        test_quality_check()
        print("\n" + "=" * 50)
        print("所有测试通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
