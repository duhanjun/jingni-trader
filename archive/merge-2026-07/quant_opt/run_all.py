"""
run_all.py
==========

执行 quant_opt 下所有测试 + 性能基准测试。
"""
from __future__ import annotations

import sys
import os
import time
import subprocess
import json
from pathlib import Path


TESTS = [
    "tests/test_metrics.py",
    "tests/test_ic_analysis.py",
    "tests/test_backtest.py",
    "tests/test_optimizer.py",
]

ROOT = Path(__file__).parent.parent


def run_one(script: str) -> dict:
    t0 = time.perf_counter()
    result = subprocess.run(
        ["python3", str(ROOT / "quant_opt" / script)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    elapsed = time.perf_counter() - t0
    return {
        "script": script,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    results = []
    summary = []
    print("=" * 60)
    print("Running quant_opt test suite")
    print("=" * 60)
    for s in TESTS:
        print(f"\n>>> {s}")
        r = run_one(s)
        ok = r["returncode"] == 0
        print(f"    exit={r['returncode']}  time={r['elapsed']:.2f}s")
        print(r["stdout"][-800:] if r["stdout"] else "(no stdout)")
        if not ok:
            print("STDERR:", r["stderr"][-400:])
        results.append(r)
        summary.append((s, ok, r["elapsed"]))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for s, ok, t in summary:
        print(f"  [{'OK' if ok else 'FAIL'}]  {s:<35}  {t:.2f}s")
    total = sum(t for _, _, t in summary)
    print(f"\nTotal time: {total:.2f}s")
    print(f"Passed: {sum(1 for _, ok, _ in summary if ok)}/{len(summary)}")

    # 写到 results
    out_dir = ROOT / "quant_opt" / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump([{
            "script": r["script"],
            "elapsed": r["elapsed"],
            "returncode": r["returncode"],
            "passed": r["returncode"] == 0,
        } for r in results], f, indent=2)
    print(f"\nResults saved to {out_dir / 'test_results.json'}")

    return 0 if all(ok for _, ok, _ in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
