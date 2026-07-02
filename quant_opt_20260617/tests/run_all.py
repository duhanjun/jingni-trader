#!/usr/bin/env python3
"""
统一测试运行器: 一次跑完所有 quant_opt_20260617 模块的测试
"""
import os
import sys
import json
import time
import subprocess

ROOT = "/workspace"
sys.path.insert(0, ROOT)

TESTS = [
    ("backtest_engine", "quant_opt_20260617.tests.test_backtest_engine"),
    ("wfo", "quant_opt_20260617.tests.test_wfo"),
    ("factor_lib", "quant_opt_20260617.tests.test_factor_lib"),
    ("integration", "quant_opt_20260617.tests.test_integration"),
]


def main():
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + os.path.join(ROOT, "quant_opt_20260617")

    results = {}
    overall_start = time.perf_counter()
    for name, module in TESTS:
        print(f"\n{'='*70}")
        print(f"Running: {name}")
        print(f"{'='*70}")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module.replace(".", "/").replace("__init__.py", "").rstrip("/")],
                # 用 python -m 跑 module 需要 __main__ 模块，改用直接 runpy
                env=env, capture_output=True, text=True, timeout=300,
            )
        except Exception as e:
            results[name] = {"error": str(e), "duration_sec": time.perf_counter() - t0}
            continue

        # 上面 -m 方式可能路径不对，改回直接调用文件
        # 简化处理：直接 subprocess 跑对应的测试文件
        file_path = module.replace(".", "/") + ".py"
        proc = subprocess.run(
            [sys.executable, file_path],
            env=env, capture_output=True, text=True, timeout=300, cwd=ROOT,
        )
        duration = time.perf_counter() - t0
        results[name] = {
            "returncode": proc.returncode,
            "duration_sec": duration,
            "stdout_tail": proc.stdout[-1500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-1500:] if proc.stderr else "",
        }
        print(proc.stdout[-1000:])
        if proc.returncode != 0:
            print("STDERR:", proc.stderr[-500:])

    overall_duration = time.perf_counter() - overall_start
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_duration_sec": overall_duration,
        "results": results,
    }
    out_path = "/workspace/quant_opt_20260617/reports/all_tests_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"OVERALL: {overall_duration:.2f}s, results saved to {out_path}")
    print(f"{'='*70}")
    n_pass = sum(1 for r in results.values() if r.get("returncode") == 0)
    n_fail = sum(1 for r in results.values() if r.get("returncode") != 0)
    print(f"PASS: {n_pass}, FAIL: {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
