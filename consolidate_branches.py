#!/usr/bin/env python3
"""
整合脚本：将 15 个 feat/quant-opt-* 分支的新增代码提取到 feat/consolidated-all-branches 分支。
策略：精选+重命名+合并，保留所有重复各自独立。

对每个分支：
1. 列出相对 main 的新增文件
2. 按映射表重命名冲突的顶层目录
3. 修复 Python import 语句中的目录名引用
4. 写入文件并 git add
"""

import os
import re
import subprocess
import json

WORKSPACE = "/workspace"

# 定义要整合的 15 个分支及其目录重命名映射
# 格式: (branch_name, {原顶层目录: 新顶层目录, ...})
# 如果原顶层目录不在映射中，则保持不变
BRANCH_CONFIG = [
    # === 建议合并的 10 个 ===
    (
        "feat/quant-opt-20260615",
        {
            # quant_opt_experiments/ 已经唯一，不需要重命名
        }
    ),
    (
        "feat/quant-opt-20260615-trae",
        {
            "quant_opt": "quant_opt_20260615_trae",
        }
    ),
    (
        "feat/quant-opt-20260616-trae",
        {
            # research/quant-opt-20260616/ 已经唯一
        }
    ),
    (
        "feat/quant-opt-20260617",
        {
            "quant_opt": "quant_opt_20260617",
        }
    ),
    (
        "feat/quant-opt-20260617-r2",
        {
            "quant_opt_20260617": "quant_opt_20260617_r2",
        }
    ),
    (
        "feat/quant-opt-20260618-r3",
        {
            "quant_opt": "quant_opt_20260618_r3",
        }
    ),
    (
        "feat/quant-opt-20260619-m3",
        {
            "quant_opt": "quant_opt_20260619_m3",
        }
    ),
    (
        "feat/quant-opt-20260621-r2",
        {
            "optimizations": "optimizations_20260621_r2",
        }
    ),
    (
        "feat/quant-opt-20260622-v2",
        {
            "optimizations": "optimizations_20260622_v2",
        }
    ),
    (
        "feat/quant-opt-20260623-r2",
        {
            "quant_opt": "quant_opt_20260623_r2",
        }
    ),
    # === 需人工复核的 5 个 ===
    (
        "feat/quant-opt-20260616",
        {
            "optimizations": "optimizations_20260616",
            "quant_opt": "quant_opt_20260616_core",
            # quant_opt_20260616/ 已经唯一
        }
    ),
    (
        "feat/quant-opt-20260617-agent-m3",
        {
            "reports": "reports_20260617_agent_m3",
            # research_20260617/ 已经唯一
        }
    ),
    (
        "feat/quant-opt-20260618",
        {
            "reports": "reports_20260618",
            # quant_opt_20260618/ 已经唯一
            # skills/quant_opt_20260618/ 需要特殊处理
        }
    ),
    (
        "feat/quant-opt-20260619",
        {
            "quant_opt": "quant_opt_20260619",
            # quant_opt_20260619/ 已经唯一（与上面的重命名不冲突，因为原目录名不同）
        }
    ),
    (
        "feat/quant-opt-20260624",
        {
            "docs": "docs_20260624",
            "optimizations": "optimizations_20260624",
            "tests": "tests_20260624",
            # quant_opt_20260624/ 已经唯一
            # skills/backtest-engine/scripts/optimizations/ 需要特殊处理
        }
    ),
]

# 特殊路径映射：某些分支的文件在 skills/ 或其他深层目录下
# 格式: (branch_name, 原路径前缀, 新路径前缀)
SPECIAL_PATH_MAP = [
    ("feat/quant-opt-20260618", "skills/quant_opt_20260618", "skills_quant_opt_20260618"),
    ("feat/quant-opt-20260618", "reports/quant_opt_20260618_report.md", "reports_20260618/quant_opt_20260618_report.md"),
    ("feat/quant-opt-20260624", "skills/backtest-engine/scripts/optimizations", "skills_backtest_opt_20260624"),
]


def run_git(*args, cwd=WORKSPACE):
    """执行 git 命令"""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def get_new_files(branch):
    """获取分支相对 main 的新增文件列表"""
    rc, out, _ = run_git("diff", "--name-only", "--diff-filter=A", "main", branch)
    if rc != 0:
        print(f"  WARNING: Failed to list files for {branch}")
        return []
    return [f for f in out.split("\n") if f.strip()]


def get_file_content(branch, filepath):
    """获取分支中文件的内容"""
    rc, out, _ = run_git("show", f"{branch}:{filepath}")
    if rc != 0:
        return None
    return out


def compute_target_path(branch, original_path, rename_map):
    """计算文件的目标路径（应用重命名映射）"""
    # 先检查特殊路径映射
    for spec_branch, orig_prefix, new_prefix in SPECIAL_PATH_MAP:
        if branch == spec_branch and original_path.startswith(orig_prefix):
            return new_prefix + original_path[len(orig_prefix):]

    # 检查顶层目录重命名
    parts = original_path.split("/")
    if len(parts) >= 1:
        top_dir = parts[0]
        if top_dir in rename_map:
            parts[0] = rename_map[top_dir]
            return "/".join(parts)

    # 无需重命名
    return original_path


def fix_imports(content, rename_map, branch):
    """修复 Python 文件中的 import 语句"""
    if content is None:
        return content

    # 构建替换规则：from orig_dir. → from new_dir.
    replacements = []
    for orig_dir, new_dir in rename_map.items():
        replacements.append((orig_dir, new_dir))

    # 特殊路径的 import 修复
    for spec_branch, orig_prefix, new_prefix in SPECIAL_PATH_MAP:
        if branch == spec_branch:
            # 将路径中的 / 替换为 . 用于 import
            orig_import = orig_prefix.replace("/", ".")
            new_import = new_prefix.replace("/", ".")
            replacements.append((orig_import, new_import))

    for orig, new in replacements:
        # from orig.xxx import ...
        content = re.sub(
            rf'\bfrom\s+{re.escape(orig)}\b',
            f'from {new}',
            content
        )
        # import orig.xxx
        content = re.sub(
            rf'\bimport\s+{re.escape(orig)}\b',
            f'import {new}',
            content
        )

    return content


def process_branch(branch, rename_map, commit=True):
    """处理单个分支：提取文件、重命名、修复 import、写入"""
    print(f"\n{'='*60}")
    print(f"Processing: {branch}")
    print(f"{'='*60}")

    # 获取新增文件列表
    new_files = get_new_files(branch)
    print(f"  New files: {len(new_files)}")

    if not new_files:
        print(f"  SKIP: No new files found")
        return 0, 0

    written = 0
    skipped = 0

    for filepath in new_files:
        # 计算目标路径
        target_path = compute_target_path(branch, filepath, rename_map)

        # 获取文件内容
        content = get_file_content(branch, filepath)
        if content is None:
            print(f"  SKIP (content not found): {filepath}")
            skipped += 1
            continue

        # 修复 Python 文件中的 import
        if filepath.endswith(".py"):
            content = fix_imports(content, rename_map, branch)

        # 写入文件
        full_path = os.path.join(WORKSPACE, target_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        written += 1

    print(f"  Written: {written}, Skipped: {skipped}")

    # Git add
    if commit and written > 0:
        # 获取所有目标路径的顶层目录
        top_dirs = set()
        for filepath in new_files:
            target = compute_target_path(branch, filepath, rename_map)
            top_dirs.add(target.split("/")[0])

        for td in top_dirs:
            run_git("add", td)

        # Commit
        short_name = branch.replace("feat/quant-opt-", "")
        rc, _, err = run_git(
            "commit", "-m",
            f"feat(consolidate): merge {branch}\n\n"
            f"Integrate optimization code from {branch}.\n"
            f"Directory renames applied to avoid conflicts.\n"
            f"Files: {written} added"
        )
        if rc == 0:
            print(f"  COMMITTED: {written} files")
        else:
            print(f"  COMMIT FAILED: {err}")

    return written, skipped


def main():
    print("=" * 60)
    print("jingni-trader Branch Consolidation Script")
    print("=" * 60)

    # 确认当前分支
    rc, branch, _ = run_git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "feat/consolidated-all-branches":
        print(f"ERROR: Expected to be on feat/consolidated-all-branches, but on {branch}")
        return

    total_written = 0
    total_skipped = 0
    results = []

    for branch_name, rename_map in BRANCH_CONFIG:
        written, skipped = process_branch(branch_name, rename_map)
        total_written += written
        total_skipped += skipped
        results.append((branch_name, written, skipped))

    # 汇总
    print("\n" + "=" * 60)
    print("CONSOLIDATION SUMMARY")
    print("=" * 60)
    for branch_name, written, skipped in results:
        print(f"  {branch_name}: {written} files written, {skipped} skipped")
    print(f"\n  TOTAL: {total_written} files written, {total_skipped} skipped")

    # 生成目录索引
    generate_index(results)


def generate_index(results):
    """生成整合版本的目录索引文件"""
    index_content = """# jingni-trader 整合版本目录索引

本分支整合了 15 个 feat/quant-opt-* 分支的优化代码。
每个分支的代码位于独立的目录中，互不冲突。

## 目录结构

| 目录 | 来源分支 | 主要优化方向 |
|------|---------|------------|
"""

    dir_info = [
        ("quant_opt_experiments/", "feat/quant-opt-20260615", "因子表达式引擎+矢量化回测(48.9x)+IC稳定性+Walk-Forward"),
        ("quant_opt_20260615_trae/", "feat/quant-opt-20260615-trae", "因子表达式引擎(AST沙箱)+向量化回测(3.2x)+Brinson归因"),
        ("research/quant-opt-20260616/", "feat/quant-opt-20260616-trae", "因子表达式(32算子)+TopK Dropout策略+Walk-Forward"),
        ("quant_opt_20260617/", "feat/quant-opt-20260617", "向量化IC(HAC 17.97x)+回测(Numba JIT)+因子表达式(17算子)"),
        ("quant_opt_20260617_r2/", "feat/quant-opt-20260617-r2", "向量化回测(19.7-32.9x)+WFO+Alpha158(44因子)+PIT"),
        ("quant_opt_20260618_r3/", "feat/quant-opt-20260618-r3", "Walk-Forward(过拟合检测)+因子DSL+前视偏差检测器(4类)"),
        ("quant_opt_20260619_m3/", "feat/quant-opt-20260619-m3", "扩展绩效指标(14个)+因子表达式+A股T+1回测"),
        ("optimizations_20260621_r2/", "feat/quant-opt-20260621-r2", "IC向量化(9.92x)+回测向量化(2.37x)+Bug复现"),
        ("optimizations_20260622_v2/", "feat/quant-opt-20260622-v2", "IC(6.2x)/中性化(15.7x)/回测(12.7x)+22扩展指标"),
        ("quant_opt_20260623_r2/", "feat/quant-opt-20260623-r2", "向量化回测(T+1修复)+因子表达式+IC/中性化"),
        ("optimizations_20260616/", "feat/quant-opt-20260616", "事件驱动回测+因子表达式+Walk-Forward"),
        ("quant_opt_20260616_core/", "feat/quant-opt-20260616", "动态加权IC-IR+向量化回测(7.4x)+PIT适配器"),
        ("quant_opt_20260616/", "feat/quant-opt-20260616", "因子表达式+绩效指标+Walk-Forward"),
        ("research_20260617/", "feat/quant-opt-20260617-agent-m3", "向量化回测(5.0x)+因子IC/IR+Walk-Forward"),
        ("reports_20260617_agent_m3/", "feat/quant-opt-20260617-agent-m3", "验证报告"),
        ("quant_opt_20260618/", "feat/quant-opt-20260618", "因子DSL(13算子+ALPHA158)+IC Decay+分位组合+bootstrap"),
        ("skills_quant_opt_20260618/", "feat/quant-opt-20260618", "IC Decay+分位分析+向量化回测+Walk-Forward"),
        ("reports_20260618/", "feat/quant-opt-20260618", "验证报告"),
        ("quant_opt_20260619/", "feat/quant-opt-20260619", "PIT+CPCV+记录器+YAML验证"),
        ("quant_opt_20260619_extra/", "feat/quant-opt-20260619", "多层风控引擎+意图解析器+向量化回测"),
        ("docs_20260624/", "feat/quant-opt-20260624", "优化报告"),
        ("optimizations_20260624/", "feat/quant-opt-20260624", "回测v2+风险v2+因子v2+Walk-Forward"),
        ("quant_opt_20260624/", "feat/quant-opt-20260624", "因子筛选+交易所模拟+验证"),
        ("tests_20260624/", "feat/quant-opt-20260624", "测试"),
        ("skills_backtest_opt_20260624/", "feat/quant-opt-20260624", "backtest-engine优化扩展"),
    ]

    for dir_name, source, desc in dir_info:
        index_content += f"| `{dir_name}` | {source} | {desc} |\n"

    index_content += """
## 测试方法

每个目录通常包含独立的测试套件，可通过以下方式运行：

```bash
# 运行单个目录的测试
python -m pytest <目录名>/tests/ -v

# 或使用目录自带的运行器
python <目录名>/tests/run_all.py
```

## 注意事项

- 各目录代码独立，存在功能重复（如多个因子表达式引擎实现），这是有意为之
- 部分目录的 import 路径已从原始分支调整（如 `quant_opt` → `quant_opt_20260617`）
- 0624 分支的 `quant_opt_20260624/tests/` 存在已知坏代码（API 不匹配），运行测试会失败
"""

    index_path = os.path.join(WORKSPACE, "CONSOLIDATED_INDEX.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_content)

    run_git("add", "CONSOLIDATED_INDEX.md")
    run_git("commit", "-m", "docs: add consolidated version directory index")


if __name__ == "__main__":
    main()
