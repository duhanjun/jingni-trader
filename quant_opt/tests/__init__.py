"""
Validation test suite for jingni-trader optimization modules
============================================================

执行:  python -m pytest quant_opt/tests -v
或:  python quant_opt/tests/run_all.py
"""
import sys
from pathlib import Path

# Make quant_opt importable as top-level packages
_THIS = Path(__file__).resolve().parent
_OPT = _THIS.parent
sys.path.insert(0, str(_OPT))
