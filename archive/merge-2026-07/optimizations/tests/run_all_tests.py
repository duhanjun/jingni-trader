"""
测试运行器：执行所有验证测试并生成报告

运行方式:
    python -m optimizations.tests.run_all_tests
或:
    python optimizations/tests/run_all_tests.py
"""
import sys
import os
import time
import json
import io
import contextlib
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from optimizations.tests.test_correctness import run_all_correctness_tests
from optimizations.tests.test_performance import run_all_performance_tests
from optimizations.tests.test_boundary import run_all_boundary_tests


def run_all_tests():
    """运行全部测试并汇总结果"""
    print("#" * 70)
    print("# jingni-trader 量化优化验证测试")
    print(f"# 分支: feat/quant-opt-20260621")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    # 捕获输出用于报告
    output_buffer = io.StringIO()

    overall_start = time.perf_counter()

    results = {}
    with contextlib.redirect_stdout(output_buffer):
        print("#" * 70)
        print("# jingni-trader 量化优化验证测试")
        print(f"# 分支: feat/quant-opt-20260621")
        print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#" * 70)

        for name, test_module_run in [
            ("correctness", run_all_correctness_tests),
            ("performance", run_all_performance_tests),
            ("boundary", run_all_boundary_tests),
        ]:
            print(f"\n{'#' * 70}")
            print(f"# 测试套件: {name}")
            print(f"{'#' * 70}")
            p, f = test_module_run()
            results[name] = {"passed": p, "failed": f}

    overall_time = time.perf_counter() - overall_start
    output = output_buffer.getvalue()

    # 打印完整输出
    print(output)

    # 汇总
    total_passed = sum(r["passed"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values())

    print("#" * 70)
    print(f"# 总计: {total_passed} 通过, {total_failed} 失败, 耗时 {overall_time:.2f}s")
    print("#" * 70)

    # 保存测试输出到文件
    report_dir = os.path.join(os.path.dirname(__file__), "..", "test_reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(report_dir, f"test_output_{timestamp}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    summary_path = os.path.join(report_dir, f"test_summary_{timestamp}.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "branch": "feat/quant-opt-20260621",
        "total_time_seconds": round(overall_time, 2),
        "total_passed": total_passed,
        "total_failed": total_failed,
        "suites": results,
        "output_file": output_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n测试输出已保存: {output_path}")
    print(f"测试摘要已保存: {summary_path}")

    return total_passed, total_failed, output_path, summary_path


if __name__ == "__main__":
    run_all_tests()
