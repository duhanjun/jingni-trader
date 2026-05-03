#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_SRC="$PROJECT_DIR/skills"
OPENCLAW_SKILLS_DIR="$HOME/.openclaw/skills"

echo "=========================================="
echo "  Skill 导入脚本"
echo "=========================================="
echo ""

check_openclaw_installed() {
    if command -v openclaw &> /dev/null; then
        return 0
    else
        echo "✗ OpenClaw 未安装，请先运行 install.sh"
        exit 1
    fi
}

validate_skills() {
    if [ ! -d "$SKILLS_SRC" ]; then
        echo "✗ 错误: 找不到 skills 目录: $SKILLS_SRC"
        exit 1
    fi
    
    SKILL_COUNT=$(find "$SKILLS_SRC" -maxdepth 1 -type d ! -name "$(basename "$SKILLS_SRC")" | wc -l)
    
    if [ "$SKILL_COUNT" -eq 0 ]; then
        echo "✗ 错误: 在 skills 目录中未找到任何 Skill"
        exit 1
    fi
    
    echo "✓ 找到 $SKILL_COUNT 个 Skill"
    echo ""
}

list_skills() {
    echo "找到的 Skill:"
    echo ""
    for skill_dir in "$SKILLS_SRC"/*/; do
        skill_name=$(basename "$skill_dir")
        if [ -f "${skill_dir}SKILL.md" ]; then
            echo "  - $skill_name ✓"
        else
            echo "  - $skill_name ⚠ (缺少 SKILL.md)"
        fi
    done
    echo ""
}

import_skills() {
    mkdir -p "$OPENCLAW_SKILLS_DIR"
    
    echo "正在导入 Skill 到: $OPENCLAW_SKILLS_DIR"
    echo ""
    
    IMPORT_COUNT=0
    SKIP_COUNT=0
    
    for skill_dir in "$SKILLS_SRC"/*/; do
        skill_name=$(basename "$skill_dir")
        
        if [ "$skill_name" = "." ] || [ "$skill_name" = ".." ]; then
            continue
        fi
        
        DEST_DIR="$OPENCLAW_SKILLS_DIR/$skill_name"
        
        if [ -d "$DEST_DIR" ]; then
            read -p "Skill '$skill_name' 已存在，是否覆盖？(y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$DEST_DIR"
            else
                echo "跳过 $skill_name"
                SKIP_COUNT=$((SKIP_COUNT + 1))
                continue
            fi
        fi
        
        echo "导入 $skill_name..."
        cp -r "$skill_dir" "$DEST_DIR"
        IMPORT_COUNT=$((IMPORT_COUNT + 1))
    done
    
    echo ""
    echo "=========================================="
    echo "  导入完成！"
    echo "=========================================="
    echo ""
    echo "导入成功: $IMPORT_COUNT 个"
    echo "跳过: $SKIP_COUNT 个"
    echo ""
    echo "可使用以下命令验证导入:"
    echo "  openclaw skills list"
    echo ""
}

main() {
    check_openclaw_installed
    validate_skills
    list_skills
    
    read -p "是否继续导入？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "取消导入"
        exit 0
    fi
    
    import_skills
}

main "$@"

