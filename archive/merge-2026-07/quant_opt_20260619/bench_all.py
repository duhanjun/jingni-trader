"""
最终汇总：运行所有 benchmark 并把输出保存到 /workspace/quant_opt_20260619/reports/
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
os.makedirs(OUT_DIR, exist_ok=True)


def run_and_capture(label, script_path):
    print(f"\n========== {label} ==========")
    r = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        capture_output=True, text=True
    )
    out = r.stdout
    if r.returncode != 0:
        out += "\n[STDERR]\n" + r.stderr
    out_path = os.path.join(OUT_DIR, f"{label}.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(out)
    print(f"[saved -> {out_path}]")
    return out


def main():
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    for label, script in [
        ("bench_pit", "bench_pit.py"),
        ("bench_cpcv", "bench_cpcv.py"),
        ("bench_recorders", "bench_recorders.py"),
    ]:
        run_and_capture(label, os.path.join(bench_dir, script))


if __name__ == "__main__":
    main()
