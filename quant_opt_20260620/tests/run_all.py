"""
统一测试运行器：执行全部验证测试
"""
import sys
import os
import unittest
import time

sys.path.insert(0, "/workspace")


def run_all():
    test_dir = os.path.join(os.path.dirname(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    start = time.perf_counter()
    result = runner.run(suite)
    elapsed = time.perf_counter() - start
    print(f"\n=== 全部测试耗时: {elapsed:.2f}s ===")
    return result


if __name__ == "__main__":
    res = run_all()
    sys.exit(0 if res.wasSuccessful() else 1)
