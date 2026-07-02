"""
向量化原生回测引擎
==================

借鉴自 SimTradeLab (https://github.com/kay-ou/SimTradeLab) 的设计思想：
通过纯 pandas/numpy 的批量运算替代逐日循环，将 1-5 年的日频回测从
秒级压缩到毫秒级。

jingni-trader 现有 ``native_adapter.py`` 用 ``for dt in dates`` 逐日
iterrows，单次回测 5 年 3000 标的可能耗时 30-90 秒。向量化版用
``groupby('date')`` 一次性完成买卖信号匹配和资金分配，期望提速
10-50x。

references
----------
- SimTradeLab: https://pypi.org/project/simtradelab/
- Qlib backtest: https://qlib.readthedocs.io/en/latest/component/backtest.html
"""

from .vectorized_engine import VectorizedBacktester

__all__ = ["VectorizedBacktester"]