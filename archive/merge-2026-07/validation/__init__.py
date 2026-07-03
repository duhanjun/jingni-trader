"""
jingni-trader 优化验证模块

本目录包含针对 jingni-trader 项目的优化验证代码。
所有代码仅作为离线验证，不会修改主分支的任何业务逻辑。

优化方向（借鉴 2026 年活跃量化开源项目）:
    1. vectorized_factor    向量化因子计算（VectorBT / qlib）
    2. purged_cv            Purged K-Fold + Walk-Forward（VectorBT / AKQuant / AFML）
    3. metrics              综合指标库 50+ 项（VectorBT.stats / quantstats）
    4. pipeline             数据预处理 Pipeline 防泄漏（AKQuant / sklearn）
    5. parameterized        参数化扫描装饰器（VectorBT @parameterized）
"""
__version__ = "0.1.0"
