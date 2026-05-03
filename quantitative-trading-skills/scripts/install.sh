#!/bin/bash

set -e

OPENCLAW_URL="https://github.com/openclaw/openclaw/releases/latest"
INSTALL_DIR="$HOME/.openclaw"

echo "=========================================="
echo "  OpenClaw 安装脚本"
echo "=========================================="
echo ""

check_openclaw_installed() {
    if command -v openclaw &> /dev/null; then
        echo "✓ OpenClaw 已安装"
        openclaw --version
        return 0
    else
        return 1
    fi
}

detect_os() {
    OS_TYPE=$(uname -s)
    case $OS_TYPE in
        Linux*)
            echo "linux"
            ;;
        Darwin*)
            echo "macos"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

install_openclaw_linux() {
    echo "检测到 Linux 系统"
    echo ""
    
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -sL"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -qO-"
    else
        echo "✗ 错误: 未找到 curl 或 wget，请先安装它们"
        exit 1
    fi
    
    echo "正在获取最新版本信息..."
    
    LATEST_RELEASE=$(curl -s https://api.github.com/repos/openclaw/openclaw/releases/latest | grep "tag_name" | cut -d '"' -f 4)
    
    if [ -z "$LATEST_RELEASE" ]; then
        echo "✗ 错误: 无法获取最新版本信息"
        exit 1
    fi
    
    echo "最新版本: $LATEST_RELEASE"
    echo ""
    
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            BINARY_URL="https://github.com/openclaw/openclaw/releases/download/${LATEST_RELEASE}/openclaw-linux-amd64"
            ;;
        aarch64|arm64)
            BINARY_URL="https://github.com/openclaw/openclaw/releases/download/${LATEST_RELEASE}/openclaw-linux-arm64"
            ;;
        *)
            echo "✗ 不支持的架构: $ARCH"
            exit 1
            ;;
    esac
    
    echo "正在下载 OpenClaw..."
    mkdir -p "$INSTALL_DIR"
    $DOWNLOAD_CMD "$BINARY_URL" -o "$INSTALL_DIR/openclaw"
    chmod +x "$INSTALL_DIR/openclaw"
    
    add_to_path
}

install_openclaw_macos() {
    echo "检测到 macOS 系统"
    echo ""
    
    if command -v brew &> /dev/null; then
        echo "使用 Homebrew 安装..."
        brew install openclaw/tap/openclaw || true
        if check_openclaw_installed; then
            return
        fi
    fi
    
    if command -v curl &> /dev/null; then
        DOWNLOAD_CMD="curl -sL"
    elif command -v wget &> /dev/null; then
        DOWNLOAD_CMD="wget -qO-"
    else
        echo "✗ 错误: 未找到 curl 或 wget，请先安装它们"
        exit 1
    fi
    
    echo "正在获取最新版本信息..."
    
    LATEST_RELEASE=$(curl -s https://api.github.com/repos/openclaw/openclaw/releases/latest | grep "tag_name" | cut -d '"' -f 4)
    
    if [ -z "$LATEST_RELEASE" ]; then
        echo "✗ 错误: 无法获取最新版本信息"
        exit 1
    fi
    
    echo "最新版本: $LATEST_RELEASE"
    echo ""
    
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            BINARY_URL="https://github.com/openclaw/openclaw/releases/download/${LATEST_RELEASE}/openclaw-darwin-amd64"
            ;;
        arm64)
            BINARY_URL="https://github.com/openclaw/openclaw/releases/download/${LATEST_RELEASE}/openclaw-darwin-arm64"
            ;;
        *)
            echo "✗ 不支持的架构: $ARCH"
            exit 1
            ;;
    esac
    
    echo "正在下载 OpenClaw..."
    mkdir -p "$INSTALL_DIR"
    $DOWNLOAD_CMD "$BINARY_URL" -o "$INSTALL_DIR/openclaw"
    chmod +x "$INSTALL_DIR/openclaw"
    
    add_to_path
}

add_to_path() {
    SHELL_CONFIG=""
    
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi
    
    if [ -n "$SHELL_CONFIG" ]; then
        if ! grep -q "export PATH=\"\$HOME/.openclaw:\$PATH\"" "$SHELL_CONFIG"; then
            echo "" >> "$SHELL_CONFIG"
            echo "# OpenClaw" >> "$SHELL_CONFIG"
            echo "export PATH=\"\$HOME/.openclaw:\$PATH\"" >> "$SHELL_CONFIG"
            echo "✓ 已将 OpenClaw 添加到 PATH"
        fi
    fi
    
    export PATH="$INSTALL_DIR:$PATH"
}

main() {
    if check_openclaw_installed; then
        echo ""
        echo "OpenClaw 已经安装，无需再次安装。"
        echo ""
        echo "如需更新，请运行:"
        echo "  openclaw update"
        echo ""
        exit 0
    fi
    
    OS=$(detect_os)
    
    case $OS in
        linux)
            install_openclaw_linux
            ;;
        macos)
            install_openclaw_macos
            ;;
        *)
            echo "✗ 不支持的操作系统: $OS"
            echo ""
            echo "请访问 $OPENCLAW_URL 手动下载安装"
            exit 1
            ;;
    esac
    
    if check_openclaw_installed; then
        echo ""
        echo "=========================================="
        echo "  OpenClaw 安装成功！"
        echo "=========================================="
        echo ""
        echo "请运行以下命令使环境变量生效："
        echo "  source ~/.bashrc  # 或 source ~/.zshrc"
        echo ""
        echo "然后验证安装："
        echo "  openclaw --version"
        echo ""
    else
        echo ""
        echo "✗ OpenClaw 安装失败"
        echo ""
        echo "请访问 $OPENCLAW_URL 查看更多安装选项"
        exit 1
    fi
}

main "$@"

