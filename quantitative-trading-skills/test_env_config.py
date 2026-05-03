
#!/usr/bin/env python3
"""
环境配置模块测试脚本
验证 env_config.py 的各项功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径，确保能导入模块
sys.path.insert(0, str(Path(__file__).parent))

from env_config import get_env_config, reset_env_config


def test_basic_functions():
    """测试基础功能"""
    print("=" * 60)
    print("测试 1: 基础功能测试")
    print("=" * 60)
    
    # 重置配置
    reset_env_config()
    
    # 设置测试环境变量
    os.environ["TEST_STR"] = "hello world"
    os.environ["TEST_INT"] = "42"
    os.environ["TEST_FLOAT"] = "3.14"
    os.environ["TEST_BOOL_TRUE"] = "true"
    os.environ["TEST_BOOL_FALSE"] = "false"
    os.environ["TEST_BOOL_YES"] = "yes"
    os.environ["TEST_BOOL_NO"] = "no"
    
    config = get_env_config()
    
    # 测试字符串
    assert config.get_str("TEST_STR") == "hello world"
    assert config.get_str("NON_EXISTENT", "default") == "default"
    print("✓ 字符串读取测试通过")
    
    # 测试整数
    assert config.get_int("TEST_INT") == 42
    assert config.get_int("NON_EXISTENT", 100) == 100
    assert config.get_int("TEST_FLOAT") is None  # 不能转换为整数
    print("✓ 整数读取测试通过")
    
    # 测试浮点数
    assert config.get_float("TEST_FLOAT") == 3.14
    assert config.get_float("TEST_INT") == 42.0
    assert config.get_float("NON_EXISTENT", 2.5) == 2.5
    print("✓ 浮点数读取测试通过")
    
    # 测试布尔值
    assert config.get_bool("TEST_BOOL_TRUE") is True
    assert config.get_bool("TEST_BOOL_FALSE") is False
    assert config.get_bool("TEST_BOOL_YES") is True
    assert config.get_bool("TEST_BOOL_NO") is False
    assert config.get_bool("NON_EXISTENT", True) is True
    print("✓ 布尔值读取测试通过")
    
    # 清理测试环境变量
    for key in list(os.environ.keys()):
        if key.startswith("TEST_"):
            del os.environ[key]
    
    print("✓ 所有基础功能测试通过！\n")
    return True


def test_env_file_loading():
    """测试 .env 文件加载"""
    print("=" * 60)
    print("测试 2: .env 文件加载测试")
    print("=" * 60)
    
    reset_env_config()
    
    # 创建临时 .env 文件
    test_env_content = """
# 测试配置
TEST_ENV_KEY1=value1
TEST_ENV_KEY2=value2
# 注释行应该被忽略
TEST_ENV_KEY3=value3
"""
    test_env_path = Path(".test_env")
    with open(test_env_path, "w", encoding="utf-8") as f:
        f.write(test_env_content)
    
    # 加载测试配置
    config = get_env_config(env_file=str(test_env_path))
    
    assert config.get_str("TEST_ENV_KEY1") == "value1"
    assert config.get_str("TEST_ENV_KEY2") == "value2"
    assert config.get_str("TEST_ENV_KEY3") == "value3"
    print("✓ .env 文件加载测试通过")
    
    # 清理
    test_env_path.unlink()
    for key in list(os.environ.keys()):
        if key.startswith("TEST_ENV_"):
            del os.environ[key]
    
    print("✓ .env 文件加载测试通过！\n")
    return True


def test_require_function():
    """测试 require_str 函数"""
    print("=" * 60)
    print("测试 3: require_str 函数测试")
    print("=" * 60)
    
    reset_env_config()
    
    config = get_env_config()
    
    # 测试已存在的变量
    os.environ["REQUIRED_KEY"] = "secret_value"
    assert config.require_str("REQUIRED_KEY") == "secret_value"
    print("✓ 现有变量获取测试通过")
    
    # 测试缺失变量抛出异常
    try:
        config.require_str("NON_EXISTENT_KEY", "这是一个必需的变量")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "NON_EXISTENT_KEY" in str(e)
        assert "这是一个必需的变量" in str(e)
        print("✓ 缺失变量异常测试通过")
    
    # 清理
    del os.environ["REQUIRED_KEY"]
    
    print("✓ require_str 函数测试通过！\n")
    return True


def test_list_all():
    """测试 list_all 函数隐藏敏感信息"""
    print("=" * 60)
    print("测试 4: 敏感信息隐藏测试")
    print("=" * 60)
    
    reset_env_config()
    
    # 设置敏感和非敏感变量
    os.environ["TUSHARE_TOKEN"] = "secret_tushare_token"
    os.environ["GM_TOKEN"] = "secret_gm_token"
    os.environ["DB_PASSWORD"] = "secret_db_password"
    os.environ["DB_USER"] = "db_user"
    os.environ["DATA_DIR"] = "./data"
    os.environ["REPORT_DIR"] = "./reports"
    
    config = get_env_config()
    config_list = config.list_all()
    
    # 检查敏感信息被隐藏
    assert config_list["TUSHARE_TOKEN"] == "***"
    assert config_list["GM_TOKEN"] == "***"
    assert config_list["DB_PASSWORD"] == "***"
    assert config_list["DB_USER"] == "***"
    
    # 检查非敏感信息正常显示
    assert config_list["DATA_DIR"] == "./data"
    assert config_list["REPORT_DIR"] == "./reports"
    
    print("✓ 敏感信息隐藏测试通过")
    
    # 清理
    for key in ["TUSHARE_TOKEN", "GM_TOKEN", "DB_PASSWORD", "DB_USER", "DATA_DIR", "REPORT_DIR"]:
        if key in os.environ:
            del os.environ[key]
    
    print("✓ 敏感信息隐藏测试通过！\n")
    return True


def test_singleton():
    """测试单例模式"""
    print("=" * 60)
    print("测试 5: 单例模式测试")
    print("=" * 60)
    
    reset_env_config()
    
    config1 = get_env_config()
    config2 = get_env_config()
    
    assert config1 is config2
    print("✓ 单例模式测试通过")
    
    reset_env_config()
    
    config3 = get_env_config()
    assert config1 is not config3
    print("✓ 重置单例测试通过")
    
    print("✓ 单例模式测试通过！\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("环境配置模块测试")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_functions,
        test_env_file_loading,
        test_require_function,
        test_list_all,
        test_singleton,
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败！")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

