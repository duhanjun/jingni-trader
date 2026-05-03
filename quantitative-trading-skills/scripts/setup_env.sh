#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env"

echo "=========================================="
echo "  环境变量配置向导"
echo "=========================================="
echo ""

create_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        cat > "$ENV_FILE" << 'EOF'
# 量化交易 Skill 套件 - 环境变量配置
# ========================================

# Tushare Pro API Token (必需)
# 获取地址: https://tushare.pro/register
TUSHARE_TOKEN=

# 掘金量化 API Token (可选)
# 获取地址: https://www.myquant.cn/
GM_TOKEN=

# 迅投 QMT API Token (可选)
# 获取地址: https://www.thinktrader.net/
XTQUANT_TOKEN=

# 数据存储目录 (可选，默认 ./data)
DATA_DIR=./data

# 日志级别 (可选: DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
EOF
        echo "✓ 创建环境变量配置文件: $ENV_FILE"
        echo ""
    else
        echo "✓ 环境变量配置文件已存在: $ENV_FILE"
        echo ""
    fi
}

display_config_guide() {
    cat << 'EOF'
环境变量说明
==========================================

1. TUSHARE_TOKEN (必需)
   - 用于获取A股历史数据、财务数据等
   - 免费注册: https://tushare.pro/register
   - 注册后在个人中心获取 Token

2. GM_TOKEN (可选)
   - 用于掘金量化平台的实时数据和回测
   - 免费注册: https://www.myquant.cn/
   - 适用于需要实时行情或实盘交易的场景

3. XTQUANT_TOKEN (可选)
   - 用于迅投 QMT 量化交易系统
   - 获取地址: https://www.thinktrader.net/

配置步骤
==========================================

1. 打开 $ENV_FILE 文件
2. 将获取到的 Token 填入对应位置
3. 保存文件
4. 运行以下命令加载环境变量:

   export $(grep -v '^#' .env | xargs)

或者将以下内容添加到 ~/.bashrc 或 ~/.zshrc:

   export TUSHARE_TOKEN="你的Token"
   export GM_TOKEN="你的Token"

验证配置
==========================================

配置完成后，可运行:

   python scripts/check_dependencies.py --check-env

EOF
}

main() {
    create_env_file
    display_config_guide
}

main "$@"

