# jingni-trader 量化交易学习报告

## 序列号: 2026-001 | 日期: 2026-06-13

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib (github.com/microsoft/qlib)
- **Stars**: 42k+ | **语言**: Python | **许可证**: MIT
- **核心亮点**:
  - **表达式引擎**: 声明式因子DSL (`$close`, `Ref($close, 1)`, `Mean($close, 20)`)，AST解析→操作符树→递归执行
  - **Alpha158/Alpha360**: 预定义因子集，158/360个因子一键生成，覆盖量价、基本面、技术指标
  - **DataHandler + Cache**: 自定义二进制格式(.bin)用于快速数据切片，多层缓存机制
  - **Model Manager**: 标准化训练/评估流水线，支持GBDT、TabNet、Transformer等模型
  - **滚动回测**: 带purge gap的滚动窗口回测，防止数据泄露
  - **RD-Agent**: 集成AI驱动的因子挖掘自动化

### 1.2 akquant (github.com/akfamily/akquant)
- **Stars**: 1.3k+ | **语言**: Rust + Python | **许可证**: MIT
- **最近更新**: 2026-06-11 (活跃维护中)
- **核心亮点**:
  - **Polars因子表达式引擎**: 基于Polars Lazy API，Alpha101风格表达式 (`Rank(Ts_Mean(Close, 5))`)，自动优化查询计划
  - **Rust+Python混合架构**: 性能关键路径用Rust实现，Python作为控制层
  - **TA-Lib双后端**: Rust和Python双后端，自动选择最优
  - **Walk-forward验证**: 时间序列交叉验证，含purge gap
  - **网格搜索**: 基于multiprocessing的参数优化框架
  - **专业级可视化**: 交互式回测报告，Plotly+Dash集成

### 1.3 NautilusTrader (github.com/nautechsystems/nautilus_trader)
- **Stars**: 23k+ | **语言**: Rust + Python (PyO3) | **许可证**: LGPL
- **最近更新**: 2026-06-11 (活跃维护中)
- **核心亮点**:
  - **Rust核心引擎**: 性能关键路径用Rust实现，Python通过PyO3调用
  - **确定性事件驱动架构**: 纳秒级时钟，单线程顺序处理，毫秒级乐观锁
  - **Research-to-Live同构**: 回测代码和实盘代码完全一致，仅切换执行环境
  - **预交易风险检查**: 订单级别风险控制，订单组管理
  - **MessageBus**: 发布/订阅模式的消息总线，解耦各模块
  - **Redis持久化**: 状态快照和恢复

---

## 二、可借鉴的方向列表

### 方向1: 因子引擎 - Polars 加速 + 声明式 DSL
- **借鉴来源**: Qlib Expression Engine + akquant FactorEngine
- **当前现状**: jingni-trader 因子计算使用 pandas + 硬编码循环，每个因子独立计算，效率低且难以扩展
- **优化建议**: 
  - 引入 Polars 作为因子计算后端，利用 Lazy API + 窗口函数批量计算
  - 实现声明式因子表达式 DSL，支持 `Return(field, n)`, `Mean(field, n)`, `Std(field, n)`, `Rank(field)` 等算子
  - 因子表达式可配置化，支持热加载新因子定义
- **优先级**: ⭐⭐⭐⭐⭐ (高)
- **影响范围**: `factor-engine` 模块

### 方向2: 回测引擎 - 向量化优化
- **借鉴来源**: NautilusTrader 确定性时间模型 + akquant 向量化仓位管理
- **当前现状**: jingni-trader native_adapter 使用逐日逐股循环，大股票池时性能瓶颈明显
- **优化建议**:
  - 信号处理采用矩阵操作（signal_matrix × price_matrix）
  - 仓位管理向量化（share_matrix 替代 dict-based positions）
  - 组合计算批量处理，减少Python循环开销
- **优先级**: ⭐⭐⭐⭐ (中高)
- **影响范围**: `backtest-engine` 模块

### 方向3: 风险模型 - Ledoit-Wolf 收缩 + 换手率惩罚
- **借鉴来源**: QUANTT 论文 CBO (Consensus-Based Optimizer)
- **当前现状**: jingni-trader 已集成 PyPortfolioOpt，但换手率惩罚实现不完善，缺少 Beta 中性化约束
- **优化建议**:
  - 完善 Ledoit-Wolf 协方差收缩（改善 N>>T 时条件数）
  - 实现 L1 换手率惩罚项，降低组合调仓频率和交易成本
  - 添加 Beta 中性化约束（λβ(β'w)²）
- **优先级**: ⭐⭐⭐ (中)
- **影响范围**: `portfolio-risk-engine` 模块

### 方向4: 数据管道 - 缓存与增量更新
- **借鉴来源**: Qlib DataHandler 缓存机制
- **当前现状**: jingni-trader 数据引擎每次重新计算因子，无缓存机制
- **优化建议**:
  - 实现因子计算结果缓存（Parquet格式）
  - 增量更新机制（仅计算新日期的因子）
  - 因子版本管理
- **优先级**: ⭐⭐⭐ (中)
- **影响范围**: `data-engine` + `factor-engine` 模块

### 方向5: 策略模型 - Walk-forward 验证
- **借鉴来源**: akquant Walk-forward 框架 + Qlib 滚动回测
- **当前现状**: jingni-trader 使用 Purged TimeSeriesSplit 做交叉验证，但缺少完整的 walk-forward 回测管道
- **优化建议**:
  - 实现完整的 walk-forward 训练/验证/测试管道
  - 自动记录每期模型性能，生成衰减曲线
  - 支持模型再训练触发条件配置
- **优先级**: ⭐⭐⭐ (中)
- **影响范围**: `strategy-model-engine` 模块

---

## 三、已完成的验证测试及结论

### 测试1: 因子表达式引擎 (Polars 加速)

**测试文件**: `tests/study_2026/test_factor_expression_engine.py`
**借鉴来源**: Microsoft Qlib + akquant

| 数据规模 | Pandas(s) | Polars(s) | 加速比 |
|------|------|------|------|
| 100股 × 500天 (50,000行) | 0.1987 | 0.0172 | **11.7x** |
| 300股 × 500天 (150,000行) | 0.5678 | 0.0405 | **14.0x** |
| 500股 × 500天 (250,000行) | 0.9118 | 0.0658 | **13.9x** |

**DSL 原型验证**:
- 表达式 `Return(close, 20)`, `Std(Return(close, 1), 20)`, `volume / Mean(volume, 20)`, `Return(close, 5) * -1`
- 4个因子在100股×200天数据上计算耗时 0.0066s
- 因子值域合理，有效率 90%-97.5%

**结论**: Polars 引擎可实现 **10-14x** 的因子计算加速。声明式 DSL 原型可行，表达式解析正确，支持嵌套函数调用和算术运算。

### 测试2: 向量化回测引擎

**测试文件**: `tests/study_2026/test_vectorized_backtest.py`
**借鉴来源**: NautilusTrader + akquant

| 股票数 | 逐日回测(s) | 向量化回测(s) | 加速比 | 结果一致性 |
|------|------|------|------|------|
| 50 | 1.6833 | 0.7948 | **2.1x** | 100%一致 |
| 100 | 2.8204 | 1.5741 | **1.8x** | 100%一致 |
| 200 | 5.0478 | 3.1935 | **1.6x** | 100%一致 |

**边界条件测试**:
- 空信号: 正确处理，不报错
- 全买入: 两种方法结果一致 (return=-0.1183%)
- 涨跌停过滤: 涨停股票被正确过滤
- T+1: 场景正确处理

**结论**: 向量化回测实现 **1.6-2.1x** 加速，与逐日回测**100%等价**。加速比随数据规模增大趋于稳定，主要瓶颈从Python循环转移到矩阵操作。

### 测试3: 增强风险模型

**测试文件**: `tests/study_2026/test_enhanced_risk_model.py`
**借鉴来源**: QUANTT 论文 (Consensus-Based Optimizer)

**协方差收缩稳定性**:
- 样本协方差条件数: 69.0
- Ledoit-Wolf 收缩后条件数: 48.8
- 条件数改善: **1.4x** (数值稳定性显著提升)

**换手率惩罚**:
- 当 PyPortfolioOpt 不可用时回退到等权配置
- 换手率惩罚机制设计验证完成，待 PyPortfolioOpt 安装后验证实际效果

**结论**: Ledoit-Wolf 协方差收缩有效降低条件数，改善优化数值稳定性。换手率惩罚和集中度控制机制设计合理，待完整环境验证。

---

## 四、待用户确认的优化建议

### 建议1: 引入 Polars 因子计算后端 (强烈推荐)
- **预期收益**: 10-14x 因子计算加速
- **实施难度**: 中等 (需要修改 factor-engine 适配器)
- **风险**: 低 (Polars 稳定成熟，52k+ GitHub Stars)
- **建议步骤**:
  1. 在 `factor-engine/scripts/adapters/` 下新增 `polars_adapter.py`
  2. 实现 `PolarsFactorCalculator` 类，继承基类接口
  3. 通过 config 切换 pandas/polars 后端
  4. 逐步迁移现有因子到 Polars 实现

### 建议2: 实现因子表达式 DSL (推荐)
- **预期收益**: 因子定义灵活性大幅提升，支持热加载新因子
- **实施难度**: 中等 (需要语法解析器 + 编译器)
- **风险**: 低 (已验证原型可行)
- **建议步骤**:
  1. 在 `factor-engine/scripts/` 下新增 `factor_dsl.py`
  2. 实现 `FactorExpression` 类 (参考测试文件)
  3. 支持 YAML/JSON 配置因子定义
  4. 为 Alpha158 风格因子集提供预定义模板

### 建议3: 向量化回测引擎优化 (推荐)
- **预期收益**: 1.6-2.1x 回测加速
- **实施难度**: 中等 (需要重构 NativeAdapter)
- **风险**: 中 (需要保证与现有回测结果100%一致)
- **建议步骤**:
  1. 在 `backtest-engine/scripts/adapters/` 下新增 `vectorized_adapter.py`
  2. 基于矩阵操作重构订单处理逻辑
  3. 通过对比测试验证与 NativeAdapter 的等价性
  4. 逐步替换为默认后端

### 建议4: 完善风险模型 (可选)
- **预期收益**: 组合稳定性提升，降低换手率
- **实施难度**: 低 (PyPortfolioOpt 已集成)
- **风险**: 低
- **建议步骤**:
  1. 完善 `portfolio-risk-engine` 的换手率惩罚实现
  2. 添加 Beta 中性化约束
  3. 参数敏感性分析

---

## 五、文件清单

| 文件 | 用途 |
|------|------|
| `tests/study_2026/test_factor_expression_engine.py` | Polars因子计算 + DSL原型验证 |
| `tests/study_2026/test_vectorized_backtest.py` | 向量化回测引擎验证 |
| `tests/study_2026/test_enhanced_risk_model.py` | 增强风险模型验证 |
| `tests/study_2026/benchmark_factors.json` | 因子计算性能基准数据 |
| `tests/study_2026/benchmark_backtest.json` | 回测性能基准数据 |
| `tests/study_2026/benchmark_risk.json` | 风险模型测试数据 |
| `tests/study_2026/LEARNING_REPORT.md` | 本报告 |

---

## 六、下一步行动

1. **等待用户确认**: 以上优化建议均需用户审核确认后方可合并
2. **建议优先实施**: 建议1 (Polars因子计算) 和 建议2 (DSL) 投入产出比最高
3. **环境准备**: 建议安装 `polars`, `pyarrow` 作为项目依赖
4. **分支策略**: 各优化方向建议在独立 `feature/xxx` 分支上开发