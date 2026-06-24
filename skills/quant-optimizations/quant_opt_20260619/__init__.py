"""
quant_opt_20260619 - 量化交易优化验证工具集

本包借鉴自以下开源项目的设计思想：

1. Microsoft Qlib (44K stars, AI-oriented quant platform)
   - YAML 驱动的 qrun 工作流
   - Point-in-Time 数据库
   - RecordTemp 可插拔记录器 (SignalRecord / SigAnaRecord / PortAnaRecord)
   - QlibRecorder / MLflowExpManager 实验管理

2. KunQuant (compiler for factor expressions)
   - 编译型因子计算，相比 Pandas 提升 170x
   - 为后续可选加速后端预留接口

3. Marcos López de Prado《Advances in Financial Machine Learning》(Chapter 7)
   - Combinatorial Purged Cross-Validation (CPCV) 多路径交叉验证
   - Embargo 隔离期，防止标签泄漏

针对 jingni-trader 的优化方向：
- pit: PIT 数据校验器，避免 look-ahead bias
- cpcv: 组合式 purged K 折 + embargo 交叉验证
- recorders: 借鉴 Qlib RecordTemp 的可插拔结果记录器
- workflow: YAML 驱动的 stage 配置

所有代码均位于 feat/quant-opt-20260619 分支的独立目录，
不直接修改 main 分支任何文件。
"""
__version__ = "0.1.0"