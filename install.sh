#!/bin/bash
# 量化交易系统 - 一键安装脚本
# 检测OpenClaw环境、自动安装依赖、导入全部Skill、配置环境变量

set -e

PROJECT_NAME="JingniTrader"
INSTALL_DIR="${HOME}/.openclaw/skills"

echo "=========================================="
echo "  量化交易系统 - 一键安装脚本"
echo "=========================================="

# 检测Python环境
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "错误: 未找到 python3，请先安装 Python 3.8+"
        exit 1
    fi
    echo "✓ Python 版本: $(python3 --version)"
}

# 检测pip
check_pip() {
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        echo "错误: 未找到 pip，请先安装 pip"
        exit 1
    fi
    echo "✓ Pip 已安装"
}

# 创建OpenClaw技能目录
setup_skills_dir() {
    mkdir -p "${INSTALL_DIR}"
    echo "✓ 技能目录: ${INSTALL_DIR}"
}

# 安装核心依赖
install_dependencies() {
    echo ""
    echo "正在安装核心依赖..."
    
    pip3 install --upgrade pip -q
    
    # 数据采集依赖
    pip3 install tushare baostock akshare xtquant pandas numpy -q
    
    # 因子计算依赖
    pip3 install pandas-ta talib scipy statsmodels -q
    
    # 机器学习依赖
    pip3 install lightgbm catboost scikit-learn optuna mlflow -q
    
    # 回测依赖
    pip3 install rqalpha backtrader -q
    
    # 组合优化依赖
    pip3 install PyPortfolioOpt riskfolio-lib -q
    
    # 实盘执行依赖（xtquant 需手动安装，仅限券商渠道）
    
    # 报告可视化依赖
    pip3 install quantstats plotly matplotlib seaborn -q
    
    echo "✓ 核心依赖安装完成"
}

# 安装项目到Python路径
install_project() {
    echo ""
    echo "正在安装项目..."
    pip3 install -e . -q
    echo "✓ 项目安装完成"
}

# 复制Skill到OpenClaw目录
install_skills() {
    echo ""
    echo "正在安装 Skills..."
    
    # 复制所有Skill
    for skill_dir in jingnitrader a-share-data-engine a-share-factor-engine \
                     strategy-model-engine backtest-engine portfolio-risk-engine \
                     execution-monitor-engine reports-engine; do
        if [ -d "${skill_dir}" ]; then
            cp -r "${skill_dir}" "${INSTALL_DIR}/"
            echo "  ✓ ${skill_dir}"
        fi
    done
    
    echo "✓ Skills 安装完成"
}

# 创建配置文件示例
create_config_example() {
    echo ""
    echo "正在创建配置文件..."
    
    CONFIG_DIR="${HOME}/.quant-trading"
    mkdir -p "${CONFIG_DIR}"
    
    cat > "${CONFIG_DIR}/config.yaml.example" << 'EOF'
# 量化交易系统配置文件示例

# 数据源配置
data_source:
  default: "tushare"  # tushare, baostock, akshare, xtquant, gm
  tushare:
    token: "your_tushare_token_here"
  baostock:
    username: ""
    password: ""

# 回测配置
backtest:
  default: "rqalpha"  # rqalpha, backtrader, gm
  commission: 0.0003
  slippage: 0.0001

# 执行配置
execution:
  default: "xtquant"  # xtquant, gm
  mode: "simulate"    # simulate, paper, live

# 因子配置
factors:
  market_neutral: true
  industry_neutral: true

# 风控配置
risk:
  max_position: 0.05
  max_loss_per_day: 0.02
  var_confidence: 0.95
EOF
    
    echo "✓ 配置文件示例: ${CONFIG_DIR}/config.yaml.example"
}

# 主函数
main() {
    check_python
    check_pip
    setup_skills_dir
    install_dependencies
    install_project
    install_skills
    create_config_example
    
    echo ""
    echo "=========================================="
    echo "  安装完成！"
    echo "=========================================="
    echo ""
    echo "下一步："
    echo "  1. 复制配置文件: cp ~/.quant-trading/config.yaml.example ~/.quant-trading/config.yaml"
    echo "  2. 编辑配置文件，填入你的API Token"
    echo "  3. 运行示例: python3 -m jingnitrader.scripts.engine"
    echo ""
}

main "$@"
