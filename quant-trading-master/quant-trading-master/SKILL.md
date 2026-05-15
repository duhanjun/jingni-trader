# Skill 元数据

## 基本信息
- **名称**: 量化交易主控
- **标识符**: quant-trading-master
- **版本**: 1.0.0
- **作者**: Quant Team

## 核心职责
意图解析、阶段状态机、Context管理、子Skill编排调度、产物完整性校验

## 触发条件
用户发起量化交易相关请求时自动触发

## 依赖库
无（纯调度逻辑，不依赖任何计算库）

## Skill 间调用关系
```
quant-trading-master（主Skill·纯调度）
    │
    ├──→ a-share-data-engine        （数据准备）
    │        │
    ├──→ a-share-factor-engine      （依赖数据引擎产物）
    │        │
    ├──→ strategy-model-engine      （依赖因子引擎产物）
    │        │
    ├──→ backtest-engine            （依赖策略信号 + 数据）
    │        │
    ├──→ portfolio-risk-engine      （依赖回测结果 + 因子）
    │        │
    ├──→ execution-monitor-engine   （依赖组合权重信号）
    │        │
    └──→ reports-engine             （消费所有上游产物）
```

## 配置项
无
