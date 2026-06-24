"""
Master Test Runner
==================
一键运行所有 quant_opt_20260618 验证测试，并输出汇总报告。
"""
import sys
import os
import time
import importlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, ROOT)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def run_test_file(path: str) -> dict:
    """通过 importlib 加载并运行测试文件"""
    t0 = time.perf_counter()
    name = "test_mod_" + os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        success = True
        error = None
    except Exception as e:
        import traceback
        success = False
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return {
        "name": os.path.basename(path),
        "path": path,
        "elapsed_sec": time.perf_counter() - t0,
        "success": success,
        "error": error,
    }


def main():
    import importlib.util
    test_files = sorted(
        f for f in os.listdir(TEST_DIR)
        if f.startswith("test_") and f.endswith(".py") and not f.startswith("_")
    )
    print("=" * 80)
    print(" jingni-trader 量化优化验证套件 (feat/quant-opt-20260618)")
    print("=" * 80)
    print(f"测试根目录: {TEST_DIR}")
    print(f"测试文件: {len(test_files)}")
    print()

    results = []
    for tf in test_files:
        path = os.path.join(TEST_DIR, tf)
        print(f"▶ 正在运行 {tf} ...")
        result = run_test_file(path)
        results.append(result)
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        print(f"  {status}  ({result['elapsed_sec']:.1f}s)")
        if not result["success"]:
            print(f"  错误: {result['error']}")
        print()

    print("=" * 80)
    print(" 汇总")
    print("=" * 80)
    n_pass = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_pass
    print(f"通过: {n_pass}/{len(results)}    失败: {n_fail}")
    print(f"总耗时: {sum(r['elapsed_sec'] for r in results):.1f}s")
    if n_fail == 0:
        print("\n  ✓ 全部测试通过")
    else:
        print(f"\n  ✗ 有 {n_fail} 个测试失败, 请检查上述输出")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())