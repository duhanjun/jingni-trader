"""
测试入口: 全部运行 ``python -m quant_opt.tests.run_all`` 即可
"""

from .test_factor_expr_engine import run as run_factor
from .test_dynamic_weighting import run as run_dynamic
from .test_vectorized_backtest import run as run_vec
from .test_pit_adapter import run as run_pit
from .test_integration import run as run_integration


def run_all():
    results = {}
    for name, fn in [
        ("factor_expr_engine", run_factor),
        ("dynamic_weighting", run_dynamic),
        ("vectorized_backtest", run_vec),
        ("pit_adapter", run_pit),
        ("integration", run_integration),
    ]:
        print(f"\n=== {name} ===")
        results[name] = fn()
    return results


if __name__ == "__main__":
    run_all()
