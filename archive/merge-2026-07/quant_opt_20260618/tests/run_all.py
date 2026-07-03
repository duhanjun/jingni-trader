"""
集成测试运行器
==============
依次执行 PIT / DSL / WFA 三个模块的所有测试，输出汇总报告。
"""
import sys
import os
import time
import subprocess
from datetime import datetime


def run_pytest_files(test_dir: str) -> dict:
    """用 pytest 运行指定目录下所有 test_*.py 文件"""
    test_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.startswith("test_") and f.endswith(".py")
    ]
    test_files.sort()

    results = {}
    for tf in test_files:
        name = os.path.basename(tf).replace("test_", "").replace(".py", "")
        print(f"\n{'=' * 70}")
        print(f"Running: {name}")
        print(f"{'=' * 70}")
        t0 = time.time()
        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", tf, "-v", "--tb=short",
                 "--no-header", "-q"],
                capture_output=True, text=True, timeout=300,
            )
            elapsed = time.time() - t0
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr[:1000])
            results[name] = {
                "elapsed": elapsed,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT on {tf}")
            results[name] = {"elapsed": 300, "returncode": -1, "stdout": "", "stderr": "timeout"}
        except Exception as e:
            print(f"ERROR on {tf}: {e}")
            results[name] = {"elapsed": 0, "returncode": -1, "stdout": "", "stderr": str(e)}

    return results


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(os.path.dirname(here), "tests")

    print("=" * 70)
    print(f"jingni-trader Quant Optimization Validation Suite")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Test dir: {test_dir}")
    print("=" * 70)

    results = run_pytest_files(test_dir)

    # 汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_pass = 0
    total_fail = 0
    for name, r in results.items():
        # 解析 pytest 输出中的 passed/failed 数
        passed = 0
        failed = 0
        for line in r["stdout"].split("\n"):
            if "passed" in line and "failed" in line:
                import re
                m = re.search(r"(\d+) passed", line)
                if m:
                    passed = int(m.group(1))
                m = re.search(r"(\d+) failed", line)
                if m:
                    failed = int(m.group(1))
                break
        total_pass += passed
        total_fail += failed
        status = "OK" if r["returncode"] == 0 else "FAIL"
        print(f"  [{status:4s}] {name:25s}  passed={passed:3d}  failed={failed:3d}  "
              f"elapsed={r['elapsed']:.2f}s")
    print(f"\n  TOTAL: {total_pass} passed, {total_fail} failed")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
