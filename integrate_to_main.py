#!/usr/bin/env python3
"""
将 feat/consolidated-all-branches 分支的代码提取到 skills/quant-optimizations/ 下。
不修改 main 现有的任何文件。
修复所有绝对 import 路径。
"""

import os
import re
import subprocess

WORKSPACE = "/workspace"
SOURCE_BRANCH = "feat/consolidated-all-branches"
TARGET_BASE = "skills/quant-optimizations"

# 需要提取的顶层目录列表（从整合分支）
DIRS_TO_EXTRACT = [
    "quant_opt_experiments",
    "quant_opt_20260615_trae",
    "quant_opt_20260616",
    "quant_opt_20260616_core",
    "quant_opt_20260617",
    "quant_opt_20260617_r2",
    "quant_opt_20260618",
    "quant_opt_20260618_r3",
    "quant_opt_20260619",
    "quant_opt_20260619_m3",
    "quant_opt_20260623_r2",
    "quant_opt_20260624",
    "optimizations_20260616",
    "optimizations_20260621_r2",
    "optimizations_20260622_v2",
    "optimizations_20260624",
    "research",                    # 包含 quant-opt-20260616
    "research_20260617",
    "skills_quant_opt_20260618",
    "skills_backtest_opt_20260624",
    "docs_20260624",
    "reports_20260617_agent_m3",
    "reports_20260618",
    "tests_20260624",
]

# import 修复映射：原前缀 → 新前缀
# from <原前缀>.xxx → from skills.quant_optimizations.<原前缀>.xxx
IMPORT_PREFIXES = [
    "quant_opt_experiments",
    "quant_opt_20260615_trae",
    "quant_opt_20260616",
    "quant_opt_20260616_core",
    "quant_opt_20260617",
    "quant_opt_20260617_r2",
    "quant_opt_20260618",
    "quant_opt_20260618_r3",
    "quant_opt_20260619",
    "quant_opt_20260619_m3",
    "quant_opt_20260623_r2",
    "quant_opt_20260624",
    "optimizations_20260616",
    "optimizations_20260621_r2",
    "optimizations_20260622_v2",
    "optimizations_20260624",
    "research_20260617",
    "skills_quant_opt_20260618",
    "skills_backtest_opt_20260624",
    "tests_20260624",
]

# research/ 特殊处理：research/quant-opt-20260616/ 下的代码
# from research.quant_opt_20260616 → 不存在这种 import（之前确认过 0 个）


def run_git(*args):
    result = subprocess.run(
        ["git"] + list(args),
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def list_branch_files(branch, path_prefix):
    """列出分支中指定前缀下的所有文件"""
    rc, out, _ = run_git("ls-tree", "-r", "--name-only", branch, "--", path_prefix)
    if rc != 0:
        return []
    return [f for f in out.split("\n") if f.strip()]


def get_file_content(branch, filepath):
    """获取分支中文件内容"""
    rc, out, _ = run_git("show", f"{branch}:{filepath}")
    if rc != 0:
        return None
    return out


def fix_imports(content):
    """修复 Python 文件中的绝对 import 路径"""
    if content is None:
        return content

    for prefix in IMPORT_PREFIXES:
        # from <prefix>.xxx import ...
        content = re.sub(
            rf'\bfrom\s+{re.escape(prefix)}\.',
            f'from {TARGET_BASE.replace("/", ".")}.{prefix}.',
            content
        )
        # import <prefix>.xxx
        content = re.sub(
            rf'\bimport\s+{re.escape(prefix)}\.',
            f'import {TARGET_BASE.replace("/", ".")}.{prefix}.',
            content
        )

    return content


def main():
    print("=" * 60)
    print("Integrating optimization code into skills/quant-optimizations/")
    print("=" * 60)

    # 创建目标目录
    target_full = os.path.join(WORKSPACE, TARGET_BASE)
    os.makedirs(target_full, exist_ok=True)

    # 创建 __init__.py
    init_path = os.path.join(target_full, "__init__.py")
    with open(init_path, "w") as f:
        f.write('"""jingni-trader 量化优化实验代码集合\n\n整合自 15 个 feat/quant-opt-* 分支的优化验证代码。\n"""\n')

    total_files = 0

    for source_dir in DIRS_TO_EXTRACT:
        print(f"\nProcessing: {source_dir}/")

        # 列出整合分支中该目录下的所有文件
        files = list_branch_files(SOURCE_BRANCH, source_dir)
        if not files:
            print(f"  SKIP: No files found in {source_dir}/")
            continue

        written = 0
        for filepath in files:
            # 计算目标路径：skills/quant-optimizations/<原路径>
            target_path = os.path.join(TARGET_BASE, filepath)
            full_path = os.path.join(WORKSPACE, target_path)

            # 获取文件内容
            content = get_file_content(SOURCE_BRANCH, filepath)
            if content is None:
                continue

            # 修复 Python 文件的 import
            if filepath.endswith(".py"):
                content = fix_imports(content)

            # 写入文件
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            written += 1

        total_files += written
        print(f"  Written: {written} files")

    # git add
    print(f"\nTotal files written: {total_files}")
    run_git("add", TARGET_BASE)

    # 提交
    rc, _, err = run_git(
        "commit", "-m",
        "feat: integrate 15 quant-opt branches into skills/quant-optimizations/\n\n"
        "Consolidate optimization code from 15 feat/quant-opt-* branches into\n"
        "skills/quant-optimizations/ directory. Each branch's code is kept in\n"
        "its own subdirectory with import paths fixed.\n\n"
        "Main branch's existing files and directories are not modified."
    )

    if rc == 0:
        print("COMMITTED successfully")
    else:
        print(f"COMMIT FAILED: {err}")

    # 验证 main 现有文件未被修改
    print("\n=== Verifying main files unchanged ===")
    rc, out, _ = run_git("diff", "--name-only", "main", "HEAD", "--", "engine.py", "run_bond_etf_ma20.py", "test_engine_v3.py", "README.md", "SKILL.md", "install.sh", "scripts/", "references/", "skills/backtest-engine/", "skills/data-engine/", "skills/factor-engine/", "skills/strategy-model-engine/", "skills/portfolio-risk-engine/", "skills/reports-engine/", "skills/execution-monitor-engine/", "skills/__init__.py")
    if out:
        print(f"WARNING: main files modified: {out}")
    else:
        print("OK: No main files modified")


if __name__ == "__main__":
    main()
