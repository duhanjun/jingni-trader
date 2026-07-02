"""
Quant Optimization Package
===========================

This package contains optimizations for jingni-trader, inspired by:

* Microsoft **Qlib** (Expression Engine, Data Layer, Alpha158 factor set)
* **AKQuant** (Polars-driven factor DSL, Walk-forward Validation)
* **VeighNa vnpy.alpha** (dataset / model / strategy / lab layered design)

The package is fully **additive** — it lives under ``quant_opt/`` and does
not modify any file in ``main``.  Each module can be imported
independently and is exercised by ``quant_opt/tests/``.
"""
__version__ = "0.1.0"
