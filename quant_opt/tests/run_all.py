"""
run_all.py - 一次性运行所有单元测试

不依赖 pytest, 直接用 unittest.
"""
import sys
import os
import unittest
import time

# 添加 workspace 根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern="test_*.py",
        top_level_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    t0 = time.time()
    result = runner.run(suite)
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"耗时: {elapsed:.2f}s")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"{'='*60}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
