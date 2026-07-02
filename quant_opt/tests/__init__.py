"""
tests - 单元测试

覆盖:
  test_alpha_engine       - 表达式引擎正确性
  test_metrics            - 指标计算与已知答案对比
  test_walk_forward       - 滚动前向验证
  test_risk_engine        - 各种风控规则触发
  test_vectorized_bt      - 向量化回测结果合理性
  test_intent_parser      - 各种自然语言样本

运行: python -m pytest quant_opt/tests/ -v
或:   python quant_opt/tests/run_all.py
"""
