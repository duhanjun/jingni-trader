"""
Run all quant_opt tests and write a summary report.

For ease of use, this script wraps `pytest` and additionally prints a summary.
Test cases are designed to be runnable both as:
  - pytest  (fixture-based)
  - this script (lightweight wrapper)
"""
import subprocess
import sys
import time
from pathlib import Path

_OPT = Path(__file__).resolve().parent.parent


def main() -> int:
    print("=" * 78)
    print("quant_opt 验证测试套件 (委托给 pytest)")
    print("=" * 78)
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(_OPT / "tests"), "-v", "--tb=short"],
        cwd=_OPT.parent,
    )
    elapsed = time.time() - start
    print("=" * 78)
    print(f"用时: {elapsed:.2f}s  ->  退出码: {proc.returncode}")
    print("=" * 78)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
