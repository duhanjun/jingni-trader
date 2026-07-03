"""
统一测试运行器

运行方式:
    python optimizations/independent_v2/run_all_tests.py

输出:
    - 控制台测试结果
    - JSON 格式的测试摘要（供报告生成使用）
"""
from __future__ import annotations

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

TEST_FILES = [
    "optimizations/independent_v2/vectorized_backtest/test_vectorized_vs_native.py",
    "optimizations/independent_v2/alpha158_factors/test_alpha158.py",
    "optimizations/independent_v2/risk_control/test_risk_control.py",
]


def run_all_tests() -> dict:
    """运行所有测试并返回结构化结果。"""
    results = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "branch": "feat/quant-opt-20260621",
        "test_modules": [],
        "total_passed": 0,
        "total_failed": 0,
        "total_errors": 0,
        "all_passed": True,
    }

    for test_file in TEST_FILES:
        print(f"\n{'='*60}")
        print(f"运行: {test_file}")
        print(f"{'='*60}")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.stderr:
            print("STDERR:", proc.stderr[-500:])

        # 解析 pytest 输出
        passed = proc.stdout.count(" PASSED")
        failed = proc.stdout.count(" FAILED")
        errors = proc.stdout.count(" ERROR")

        results["test_modules"].append({
            "file": test_file,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "exit_code": proc.returncode,
        })
        results["total_passed"] += passed
        results["total_failed"] += failed
        results["total_errors"] += errors
        if failed > 0 or errors > 0:
            results["all_passed"] = False

    return results


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\n{'='*60}")
    print("测试摘要")
    print(f"{'='*60}")
    print(f"总通过: {results['total_passed']}")
    print(f"总失败: {results['total_failed']}")
    print(f"总错误: {results['total_errors']}")
    print(f"全部通过: {results['all_passed']}")

    # 保存 JSON 摘要
    summary_path = ROOT / "optimizations" / "independent_v2" / "test_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n测试摘要已保存至: {summary_path}")

    sys.exit(0 if results["all_passed"] else 1)
