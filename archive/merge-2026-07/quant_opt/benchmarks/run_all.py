"""Run all benchmarks and tests, save JSON outputs to reports/."""
import contextlib
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from quant_opt.benchmarks.bench_ic import main as run_ic
from quant_opt.benchmarks.bench_engine import main as run_eng
from quant_opt.benchmarks.demo_pipeline import main as run_demo
from quant_opt.tests.test_quant_opt import run_all as run_tests


def main():
    reports_dir = os.path.join(ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 60)
    print("  jingni-trader quant_opt validation suite")
    print("=" * 60)

    # 1) IC benchmark
    print("\n[1/4] IC benchmark (vectorized vs loop)...")
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ic_result = run_ic()
    print(f"  done in {time.time() - t0:.1f}s, speedup={ic_result['speedup_x']:.1f}x")
    with open(os.path.join(reports_dir, "bench_ic.json"), "w") as f:
        json.dump(ic_result, f, indent=2)

    # 2) Engine benchmark
    print("\n[2/4] Expression engine benchmark...")
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng_result = run_eng()
    print(f"  done in {time.time() - t0:.1f}s")
    with open(os.path.join(reports_dir, "bench_engine.json"), "w") as f:
        json.dump(eng_result, f, indent=2)

    # 3) End-to-end demo
    print("\n[3/4] End-to-end pipeline demo...")
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        demo_result = run_demo()
    print(f"  done in {time.time() - t0:.1f}s, best_factor={demo_result['best_factor']}")
    with open(os.path.join(reports_dir, "demo_pipeline.json"), "w") as f:
        json.dump(demo_result, f, indent=2)

    # 4) Test suite
    print("\n[4/4] Test suite...")
    t0 = time.time()
    p, total, results = run_tests()
    print(f"  done in {time.time() - t0:.1f}s, {p}/{total} passed")
    test_results = [{"name": n, "passed": ok, "message": m} for n, ok, m in results]
    test_summary = {"passed": p, "total": total}
    with open(os.path.join(reports_dir, "test_results.json"), "w") as f:
        json.dump({"summary": test_summary, "tests": test_results}, f, indent=2)

    # Aggregate summary
    summary = {
        "ic_benchmark": ic_result,
        "engine_benchmark": eng_result,
        "pipeline_demo": demo_result,
        "tests": test_summary,
    }
    with open(os.path.join(reports_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nAll results saved to", reports_dir)
    return summary


if __name__ == "__main__":
    main()
