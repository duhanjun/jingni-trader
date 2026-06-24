"""
统一测试运行器 - 串联三个模块的测试, 输出可对比的结果
"""
import sys
import os
import time
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_subprocess(label: str, script: str) -> dict:
    """在子进程运行测试, 隔离环境, 防止相互干扰"""
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, env={**os.environ},
    )
    elapsed = time.time() - t0
    return {
        "label": label,
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "elapsed_sec": elapsed,
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    results = []
    for label, fname in [
        ("factor_expression", "test_factor_expression.py"),
        ("extended_metrics",  "test_extended_metrics.py"),
        ("vectorized_backtest", "test_vectorized_backtest.py"),
    ]:
        r = run_subprocess(label, os.path.join(here, fname))
        results.append(r)
        status = "PASS" if r["returncode"] == 0 else "FAIL"
        print(f"\n========== {label}: {status} (elapsed {r['elapsed_sec']:.2f}s) ==========")
        print(r["stdout"][-3000:])  # 末尾输出
        if r["returncode"] != 0:
            print("STDERR:", r["stderr"][-2000:])

    # 汇总
    print("\n" + "=" * 60)
    print("TEST RUN SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in results:
        ok = r["returncode"] == 0
        all_pass = all_pass and ok
        print(f"  [{'OK' if ok else 'FAIL'}] {r['label']:<25s} {r['elapsed_sec']:6.2f}s")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())