"""
统一运行所有验证测试
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """依次运行 4 个测试模块"""
    print("\n" + "#" * 70)
    print("# jingni-trader 量化交易优化验证 (2026-06-18)")
    print("# 分支: feat/quant-opt-20260618")
    print(f"# 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70 + "\n")

    overall_t0 = time.time()
    all_results = {}

    # 测试 1
    from tests.test_vectorized_backtest import main as test1
    print("\n>>> 运行测试 1: 向量化回测引擎验证")
    rc1 = test1()
    all_results["test1_vectorized_backtest"] = "PASS" if rc1 == 0 else "FAIL"

    # 测试 2
    from tests.test_strategy_api import main as test2
    print("\n>>> 运行测试 2: Strategy API 验证")
    rc2 = test2()
    all_results["test2_strategy_api"] = "PASS" if rc2 == 0 else "FAIL"

    # 测试 3
    from tests.test_pit_guard import main as test3
    print("\n>>> 运行测试 3: PIT Guard + Purged CV 验证")
    rc3 = test3()
    all_results["test3_pit_guard"] = "PASS" if rc3 == 0 else "FAIL"

    # 测试 4
    from tests.test_stability import main as test4
    print("\n>>> 运行测试 4: 多源稳健性测试验证")
    rc4 = test4()
    all_results["test4_stability"] = "PASS" if rc4 == 0 else "FAIL"

    overall_t1 = time.time()

    # 汇总
    print("\n" + "#" * 70)
    print("# 全部测试汇总")
    print("#" * 70)
    for name, status in all_results.items():
        print(f"  {name}: {status}")
    total_pass = sum(1 for v in all_results.values() if v == "PASS")
    total = len(all_results)
    print(f"\n  通过: {total_pass}/{total}")
    print(f"  总耗时: {overall_t1 - overall_t0:.2f} 秒")

    # 保存结果到 JSON
    result_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_results.json"
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "branch": "feat/quant-opt-20260618",
            "results": all_results,
            "total_pass": total_pass,
            "total": total,
            "elapsed_seconds": overall_t1 - overall_t0,
        }, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {result_path}")

    return 0 if all(v == "PASS" for v in all_results.values()) else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
