"""
测试套件：正确性、性能、边界测试

包含三类测试：
1. test_correctness.py : 新模块输出与原 jingni-trader 实现的等价性验证
2. test_performance.py : 性能对比基准（Polars vs pandas, 向量化回测 vs 原生）
3. test_edge_cases.py  : 边界条件（空数据、单股票、缺失列等）
"""
