#!/usr/bin/env python3
"""Standalone test runner (avoids ``-m`` package issues)."""
import os
import sys

# Add the parent of the quant_opt package to sys.path so that
# ``import quant_opt`` works regardless of where this script is invoked.
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)            # .../quant_opt
PARENT = os.path.dirname(PKG_DIR)          # ...
sys.path.insert(0, PARENT)

# Also expose the package itself so ``tests`` can do relative imports.
sys.path.insert(0, PKG_DIR)

from tests import run_all

if __name__ == "__main__":
    n_pass, n_total, _ = run_all(verbose=True)
    sys.exit(0 if n_pass == n_total else 1)
