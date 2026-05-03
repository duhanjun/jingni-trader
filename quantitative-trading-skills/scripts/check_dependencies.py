#!/usr/bin/env python3
import sys
import subprocess
import argparse
import os
from typing import List, Tuple, Dict

REQUIRED_PYTHON_VERSION = (3, 9)

REQUIRED_PACKAGES = [
    ("pandas", ">=2.0.0"),
    ("numpy", ">=1.24.0"),
    ("sqlalchemy", ">=2.0.0"),
    ("tushare", ">=1.2.60"),
    ("baostock", ">=0.8.8"),
    ("akshare", ">=1.11.0"),
    ("pyarrow", ">=14.0.0"),
    ("fastparquet", ">=2023.10.0"),
    ("scipy", ">=1.10.0"),
    ("scikit-learn", ">=1.3.0"),
    ("matplotlib", ">=3.7.0"),
    ("plotly", ">=5.14.0"),
    ("quantstats", ">=0.0.59"),
    ("empyrical", ">=0.5.5"),
    ("pypfopt", ">=0.5.4"),
    ("cvxpy", ">=1.3.0"),
    ("lightgbm", ">=4.0.0"),
    ("catboost", ">=1.2.0"),
    ("optuna", ">=3.0.0"),
    ("mlflow", ">=2.0.0"),
    ("pandas-ta", ">=0.3.0"),
    ("ta-lib", ">=0.4.28"),
]

OPTIONAL_PACKAGES = {
    "xtquant": {
        "description": "迅投 QMT 量化交易系统",
        "packages": [("xtquant", "*")],
        "install_guide": "https://www.thinktrader.net/"
    },
    "gm": {
        "description": "掘金量化平台",
        "packages": [("gm", "*")],
        "install_guide": "https://www.myquant.cn/"
    },
    "vnpy": {
        "description": "VN.py 量化交易框架",
        "packages": [("vnpy", "*")],
        "install_guide": "https://www.vnpy.com/"
    },
    "rqalpha": {
        "description": "RQAlpha 回测引擎",
        "packages": [("rqalpha", ">=4.0.0")],
        "install_guide": "https://www.ricequant.com/"
    },
    "backtrader": {
        "description": "Backtrader 回测框架",
        "packages": [("backtrader", ">=1.9.78")],
        "install_guide": "https://www.backtrader.com/"
    }
}

def check_python_version() -> Tuple[bool, str]:
    current_version = sys.version_info[:2]
    if current_version >= REQUIRED_PYTHON_VERSION:
        return True, f"Python {current_version[0]}.{current_version[1]}"
    return False, f"Python {current_version[0]}.{current_version[1]} (需要 >= {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]})"

def get_package_version(package_name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except:
        try:
            module = __import__(package_name.replace('-', '_'))
            if hasattr(module, '__version__'):
                return module.__version__
        except:
            pass
    return None

def check_package(package_name: str, version_spec: str) -> Tuple[bool, str]:
    installed_version = get_package_version(package_name)
    if installed_version:
        return True, f"{package_name} {installed_version}"
    return False, f"{package_name} (未安装)"

def install_package(package: str):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_required_packages():
    print("正在安装必需包...")
    print("=" * 50)
    for package, version_spec in REQUIRED_PACKAGES:
        package_spec = f"{package}{version_spec}" if version_spec != "*" else package
        print(f"安装 {package_spec}...")
        if install_package(package_spec):
            print(f"✓ {package} 安装成功")
        else:
            print(f"✗ {package} 安装失败")
    print("=" * 50)

def install_optional_packages(selection: List[str]):
    for name in selection:
        if name in OPTIONAL_PACKAGES:
            print(f"\n正在安装 {OPTIONAL_PACKAGES[name]['description']}...")
            print("=" * 50)
            for package, version_spec in OPTIONAL_PACKAGES[name]['packages']:
                package_spec = f"{package}{version_spec}" if version_spec != "*" else package
                print(f"安装 {package_spec}...")
                if install_package(package_spec):
                    print(f"✓ {package} 安装成功")
                else:
                    print(f"✗ {package} 安装失败")
            print("=" * 50)

def check_env_vars():
    env_vars = [
        ("TUSHARE_TOKEN", "Tushare API Token (必需)"),
        ("GM_TOKEN", "掘金量化 API Token (可选)"),
        ("XTQUANT_TOKEN", "迅投 QMT API Token (可选)")
    ]
    
    print("\n环境变量检查:")
    print("=" * 50)
    for var_name, description in env_vars:
        value = os.environ.get(var_name)
        if value:
            status = "✓ 已设置"
        else:
            status = "✗ 未设置"
        print(f"{status:10} {var_name:20} - {description}")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="检查和安装依赖包")
    parser.add_argument("--install", action="store_true", help="安装所有必需包")
    parser.add_argument("--optional", nargs="*", choices=OPTIONAL_PACKAGES.keys(), 
                        help=f"安装可选包: {', '.join(OPTIONAL_PACKAGES.keys())}")
    parser.add_argument("--check-env", action="store_true", help="检查环境变量")
    parser.add_argument("--all", action="store_true", help="安装所有包（必需 + 可选）")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("  依赖检查工具")
    print("=" * 50)
    print()
    
    print("Python 版本检查:")
    print("-" * 50)
    ok, msg = check_python_version()
    prefix = "✓" if ok else "✗"
    print(f"{prefix} {msg}")
    
    if not ok:
        print("\n错误: Python 版本过低，请升级到 Python 3.9 或更高版本")
        sys.exit(1)
    
    print("\n必需包检查:")
    print("-" * 50)
    all_ok = True
    for package, version_spec in REQUIRED_PACKAGES:
        ok, msg = check_package(package, version_spec)
        prefix = "✓" if ok else "✗"
        print(f"{prefix} {msg}")
        if not ok:
            all_ok = False
    
    if args.install or args.all:
        install_required_packages()
    
    print("\n可选包检查:")
    print("-" * 50)
    for name, info in OPTIONAL_PACKAGES.items():
        all_installed = True
        installed_list = []
        for package, _ in info["packages"]:
            ok, _ = check_package(package, "*")
            if ok:
                installed_list.append(package)
            else:
                all_installed = False
        if all_installed:
            print(f"✓ {info['description']} (已安装)")
        else:
            print(f"- {info['description']} (可选)")
    
    if args.optional:
        install_optional_packages(args.optional)
    
    if args.all:
        install_optional_packages(list(OPTIONAL_PACKAGES.keys()))
    
    if args.check_env:
        check_env_vars()
    
    print("\n" + "=" * 50)
    if all_ok:
        print("  所有必需依赖已满足！")
    else:
        print("  部分依赖缺失，运行以下命令安装:")
        print("  python scripts/check_dependencies.py --install")
    print("=" * 50)

if __name__ == "__main__":
    main()

