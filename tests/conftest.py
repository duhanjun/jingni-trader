"""pytest 统一路径处理（GAP-5）。

将项目根目录加入 sys.path，使 tests/ 下的测试脚本能直接
`import run_bond_etf_ma20 as m`（根级模块）或通过相对路径
解析 master engine 所需的 `scripts` 包，无需硬编码 /workspace。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
