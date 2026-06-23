"""
验证测试总运行器

依次运行三个优化模块的验证测试，汇总结果并生成 JSON 报告。
用法: python -m experiments.run_all_tests
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime

# 确保能 import experiments 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.vectorized_neutralization import run_tests as test_neutralization
from experiments.expression_factor_framework import run_tests as test_expression
from experiments.point_in_time_validator import run_tests as test_pit


def main() -> dict:
    print("=" * 70)
    print("jingni-trader 优化验证测试")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"分支: feat/quant-opt-20260623")
    print("=" * 70)

    all_results = {
        "run_at": datetime.now().isoformat(),
        "branch": "feat/quant-opt-20260623",
        "modules": [],
    }

    for label, runner in [
        ("1. 向量化因子中性化", test_neutralization),
        ("2. 表达式驱动因子框架", test_expression),
        ("3. Point-in-Time 防泄漏检测器", test_pit),
    ]:
        print(f"\n{'#' * 60}")
        print(f"# {label}")
        print(f"{'#' * 60}")
        try:
            res = runner()
            all_results["modules"].append(res)
        except Exception as e:
            import traceback
            print(f"模块执行异常: {e}")
            traceback.print_exc()
            all_results["modules"].append({
                "optimization": label,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    summary_lines = []
    for m in all_results["modules"]:
        name = m.get("optimization", "unknown")
        # 判定通过状态
        passed = True
        if "correctness" in m and isinstance(m["correctness"], dict):
            passed = passed and m["correctness"].get("passed", False)
        if "boundary" in m and isinstance(m["boundary"], dict):
            passed = passed and m["boundary"].get("all_passed", m["boundary"].get("passed", True))
        if "extensibility" in m and isinstance(m["extensibility"], dict):
            passed = passed and m["extensibility"].get("passed", True)
        if "performance" in m and isinstance(m["performance"], dict):
            passed = passed and m["performance"].get("passed", True)
        status = "PASS" if passed else "FAIL"
        summary_lines.append(f"  [{status}] {name}")
    print("\n".join(summary_lines))

    # 保存 JSON 结果
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n测试结果 JSON 已保存: {json_path}")

    return all_results


if __name__ == "__main__":
    main()
