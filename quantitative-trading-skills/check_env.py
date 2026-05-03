
#!/usr/bin/env python3
"""
环境变量检查脚本
验证所有必需的环境变量是否已正确配置
"""

import sys
from pathlib import Path
from env_config import get_env_config


def check_environment() -> bool:
    """
    检查环境变量配置
    
    Returns:
        bool: 所有检查通过返回 True，否则返回 False
    """
    print("=" * 60)
    print("量化交易技能套件 - 环境变量检查")
    print("=" * 60)
    
    config = get_env_config()
    all_passed = True
    
    # 检查 .env 文件
    env_path = Path(".env")
    if env_path.exists():
        print("✓ .env 文件存在")
    else:
        print("⚠ .env 文件不存在，将使用环境变量或默认值")
        print(f"  提示：可以从 .env.example 复制创建 .env 文件")
    
    print()
    
    # 数据源配置检查
    print("--- 数据源配置检查 ---")
    
    # Tushare Token
    tushare_token = config.get_str("TUSHARE_TOKEN")
    if tushare_token:
        print("✓ TUSHARE_TOKEN 已配置")
    else:
        print("✗ TUSHARE_TOKEN 未配置 (使用 Tushare 数据源时必需)")
        print("  获取地址：https://tushare.pro/register")
        all_passed = False
    
    # XCSC Tushare Token (可选)
    xcsc_token = config.get_str("XCSC_TUSHARE_TOKEN")
    if xcsc_token:
        print("✓ XCSC_TUSHARE_TOKEN 已配置")
    else:
        print("ℹ XCSC_TUSHARE_TOKEN 未配置 (可选)")
    
    # 掘金 Token (可选)
    gm_token = config.get_str("GM_TOKEN")
    if gm_token:
        print("✓ GM_TOKEN 已配置")
    else:
        print("ℹ GM_TOKEN 未配置 (可选，用于掘金数据源和回测)")
    
    print()
    
    # 数据库配置检查
    print("--- 数据库配置检查 ---")
    db_type = config.get_str("DB_TYPE", "sqlite")
    print(f"✓ 数据库类型：{db_type}")
    
    if db_type == "sqlite":
        print("✓ SQLite 无需额外配置")
    else:
        db_host = config.get_str("DB_HOST", "localhost")
        db_port = config.get_int("DB_PORT", 3306 if db_type == "mysql" else 5432)
        db_name = config.get_str("DB_NAME")
        db_user = config.get_str("DB_USER")
        db_password = config.get_str("DB_PASSWORD")
        
        print(f"✓ DB_HOST：{db_host}")
        print(f"✓ DB_PORT：{db_port}")
        
        if db_name:
            print(f"✓ DB_NAME 已配置")
        else:
            print(f"✗ DB_NAME 未配置")
            all_passed = False
        
        if db_user:
            print(f"✓ DB_USER 已配置")
        else:
            print(f"✗ DB_USER 未配置")
            all_passed = False
        
        if db_password:
            print(f"✓ DB_PASSWORD 已配置")
        else:
            print(f"✗ DB_PASSWORD 未配置")
            all_passed = False
    
    print()
    
    # 存储配置检查
    print("--- 存储配置检查 ---")
    data_dir = config.get_str("DATA_DIR", "./data")
    factor_dir = config.get_str("FACTOR_DATA_DIR", "./factor_data")
    report_dir = config.get_str("REPORT_DIR", "./reports")
    
    print(f"✓ DATA_DIR：{data_dir}")
    print(f"✓ FACTOR_DATA_DIR：{factor_dir}")
    print(f"✓ REPORT_DIR：{report_dir}")
    
    # 检查目录是否存在或可创建
    for dir_path in [data_dir, factor_dir, report_dir]:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✓ 目录 {dir_path} 可访问")
        except Exception as e:
            print(f"✗ 目录 {dir_path} 无法创建：{e}")
            all_passed = False
    
    print()
    
    # 应用配置
    print("--- 应用配置 ---")
    data_backend = config.get_str("DATA_BACKEND", "tushare")
    backtest_engine = config.get_str("BACKTEST_ENGINE", "rqalpha")
    ta_library = config.get_str("TA_LIBRARY", "pandas-ta")
    plot_backend = config.get_str("PLOT_BACKEND", "matplotlib")
    log_level = config.get_str("LOG_LEVEL", "INFO")
    
    print(f"✓ DATA_BACKEND：{data_backend}")
    print(f"✓ BACKTEST_ENGINE：{backtest_engine}")
    print(f"✓ TA_LIBRARY：{ta_library}")
    print(f"✓ PLOT_BACKEND：{plot_backend}")
    print(f"✓ LOG_LEVEL：{log_level}")
    
    print()
    
    # 当前配置摘要（隐去敏感信息）
    print("--- 当前配置摘要 ---")
    config_summary = config.list_all()
    for key, value in config_summary.items():
        print(f"{key} = {value}")
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✓ 所有检查通过！")
    else:
        print("⚠ 部分配置缺失，请根据提示补充")
    
    print("=" * 60)
    
    return all_passed


def main():
    """主函数"""
    try:
        success = check_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n检查被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n检查过程中出错：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

