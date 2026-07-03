# jingni-trader 量化优化验证报告

> **执行日期**: 2026-06-24
> **分支**: `feat/quant-opt-20260624`
> **执行人**: 自动化学习与优化流程
> **测试结果**: 42/42 通过

---

## 一、学习项目清单及核心亮点

本次联网调研覆盖 GitHub Trending、Awesome Quant、arXiv、东方证券研报等来源，筛选出以下高价值开源项目深入学习：

### 项目 1：微软 Qlib（microsoft/qlib，15k+ Star）

**定位**：AI 导向量化投资平台，工程化标杆。

**核心亮点**：
- **6 层数据栈**：L1 采集 → L2 `.bin` 二进制存储 → L3 Provider → L4 四级缓存 → L5 DataHandlerLP → L6 Dataset，专为金融科学计算设计
- **表达式引擎**（`qlib/data/ops.py`）：字符串即因子，`Ref($close, 60) / $close` 可解析、可序列化、可缓存；算子实现 `get_extended_window_size()` 自动处理滚动边界
- **qrun + YAML 工作流**：`init_instance_by_config({class, module_path, kwargs})` 实现配置即代码
- **Exchange 防前视**（`qlib/backtest/exchange.py`）：`deal_price=("$open","$close")` 元组语义、`limit_threshold` 涨跌停校验、`check_stock_suspended` 停牌判定、`_clip_amount_by_volume` 成交量容量裁剪、`impact_cost` 线性冲击成本
- **Model 抽象基类**：`fit/predict` 统一接口，`DataHandlerLP.DK_L/DK_I/DK_R` 三态数据隔离防泄露
- **Alpha158/Alpha360**：因子族配置化，`get_feature_config()` 返回 `{group: [expr_list]}`

### 项目 2：QuantaAlpha（arXiv:2602.07085v3，2026-05）

**定位**：LLM 驱动的进化式 Alpha 因子挖掘框架。

**核心亮点**：
- **轨迹级进化**：进化对象从"因子公式"升级为"完整研究轨迹"（假设→因子→代码→回测→反馈）
- **多智能体协作**：假设生成 / 因子实现 / 迭代优化 / 因子筛选四类智能体
- **三重约束因子筛选**：Rank IC 显著 + 低冗余(相关<0.7) + 容量达标，贪心 RankIC 降序入库
- **AST 符号化**：因子符号化为 AST，支持一致性校验、复杂度约束、结构化去重
- **lineage 追踪**：`StrategyTrajectory.parent_ids` 记录因子进化谱系，可审计
- **A 股增强方向**（东方证券研报）：需加行业/市值中性化 IC，剥离系统性风格暴露

### 项目 3：其他参考项目

| 项目 | Star | 借鉴点 |
|------|------|--------|
| vn.py | 23k+ | 中文量化框架成熟度参考 |
| Backtrader | 10k+ | 事件驱动回测架构（已停更，仅参考） |
| Freqtrade | 25k+ | FreqAI ML 优化集成模式 |
| NautilusTrader | - | 机构级生产执行引擎设计 |

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码（7 引擎架构），识别出以下可借鉴改进点（按优先级）：

### 高优先级 + 低难度（快速落地）

| 编号 | 改进点 | 借鉴来源 | 现有缺陷 |
|------|--------|----------|----------|
| B1 | Exchange 涨跌停/停牌/成交量校验 | Qlib Exchange | native_adapter 仅检查 is_limit_up/down，无停牌 close=None 判定，无成交量容量 |
| B2 | deal_price 防前视语义 | Qlib Exchange | 现有用收盘价同时决策和成交，存在前视偏差 |
| B3 | 线性冲击成本模型 | Qlib Exchange | 现有仅固定滑点，无冲击成本 |
| B4 | T+1 显式强制 | Qlib Account | 现有靠买卖顺序"巧合"实现，未跟踪买入日期 |
| F1 | 表达式引擎 + 算子注册表 | Qlib ops.py | 因子硬编码在 compute_a_share_factors()，不可序列化/缓存/审计 |
| F2 | 因子族配置化 | Qlib Alpha158 | 因子无法插拔配置 |
| F5 | 三重约束因子筛选 | QuantaAlpha | 现有仅相关性去冗余，无 IC 门槛、无容量约束 |
| M1 | Model 抽象基类 | Qlib Model | 现有无统一模型接口 |

### 高优先级 + 中难度（核心能力）

| 编号 | 改进点 | 借鉴来源 |
|------|--------|----------|
| D1 | `.bin` 时序扁平文件存储 | Qlib L2 |
| D2 | PIT Provider 防前视 | Qlib L3 |
| D3 | 四级缓存机制 | Qlib L4 |
| F5+ | 行业/市值中性化 IC | 东方证券研报 |
| M4 | RollingGen 滚动窗口训练 | Qlib task |
| P1 | StructuredCovEstimator | Qlib riskmodel |
| C1 | qrun + YAML 工作流编排 | Qlib cli |

### 中优先级（增强能力，第二阶段）

| 编号 | 改进点 | 借鉴来源 |
|------|--------|----------|
| F3 | LLM 驱动因子挖掘管线 | QuantaAlpha |
| F4 | 因子 AST 符号化 | QuantaAlpha |
| F6 | 多样化规划初始化 | QuantaAlpha |
| F7 | 因子 lineage 追踪 | QuantaAlpha |
| B5 | NestedExecutor 多级嵌套执行 | Qlib |
| C3 | 轨迹级进化编排 | QuantaAlpha |

---

## 三、本次已完成的验证测试

本次针对 **3 个高价值优化点** 编写验证代码并测试，全部位于 `optimizations/20260624/` 独立目录，未修改 main 分支任何代码。

### 优化点 1：防前视 Exchange 回测交易所

**文件**: [exchange.py](file:///workspace/optimizations/20260624/exchange.py)

**借鉴来源**: Qlib `qlib/backtest/exchange.py`

**解决的现有缺陷**（对照 [native_adapter.py](file:///workspace/skills/backtest-engine/scripts/adapters/native_adapter.py)）：

| 缺陷 | 现有实现 | 优化后实现 |
|------|----------|------------|
| 前视偏差 | `price = price_row['close']` 收盘价决策+成交 | `deal_price="$open"` 次日开盘成交 |
| T+1 未强制 | 靠买卖顺序巧合，无买入日跟踪 | `Lot` 批次池 + `available_date` 显式约束 |
| 停牌未检查 | 仅判断 code 是否在行情中 | `check_stock_suspended()` 检查 close is None |
| 无成交量容量 | 可买超当日成交量 | `clip_amount_by_volume()` 按 10% 裁剪 |
| 无冲击成本 | 仅固定滑点 0.1% | 线性 `impact_cost_rate` 按成交额计 |
| 过户费缺失 | 配置有 TRANSFER_FEE_RATE 但未用 | `transfer_fee_rate` 计入买卖成本 |

**测试覆盖**（[test_optimizations.py](file:///workspace/optimizations/20260624/test_optimizations.py) 测试组 1-3）：
- ✓ 涨停拒绝买入 / 跌停拒绝卖出 / 停牌拒绝交易
- ✓ T+1 当日买入不可卖 / T+1 次日可卖
- ✓ 成交量容量裁剪 / 冲击成本计算 / 卖出含印花税 / 最低手续费 5 元
- ✓ 开盘价 vs 收盘价成交净值不同（证明 deal_price 语义生效）
- ✓ 空数据不崩溃 / 资金不足拒绝买入 / 全停牌无成交

### 优化点 2：因子表达式引擎 + 算子注册表

**文件**: [factor_expression.py](file:///workspace/optimizations/20260624/factor_expression.py)

**借鉴来源**: Qlib `qlib/data/ops.py` + `qlib/contrib/data/handler.py`

**解决的现有缺陷**（对照 [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py) `compute_a_share_factors()`）：

| 缺陷 | 现有实现 | 优化后实现 |
|------|----------|------------|
| 因子硬编码 | Python 函数逐个手写 | 字符串表达式 `Ref($close,20)/$close` 可解析 |
| 不可序列化 | 无法保存/传输因子定义 | `__str__` 序列化往返 |
| 不可缓存 | 每次重算 | AST 缓存 `_ast_cache` |
| 无边界声明 | 滚动窗口边界靠人工 | `get_extended_window_size()` 自动累加 |
| 因子不可插拔 | 改因子需改代码 | `FactorFamily.alpha158_lite()` 配置化 |
| 无算子注册 | 算子散落各处 | `OperatorRegistry` 集中注册 + 运行时扩展 |

**实现的算子集**：Ref / Mean / Std / Max / Min / Sum / Var / Rank / Delta / Abs / Sign / Neg / Add / Sub / Mul / Div / Corr / Cov / Slope（共 18 个，覆盖 Qlib 核心算子）

**测试覆盖**（测试组 4-9）：
- ✓ 解析字段引用 / 常量 / 函数调用 / 二元运算 / 嵌套表达式 / 序列化往返
- ✓ Ref / Mean / 二元运算(收益率) / Std 波动率 计算结果与硬编码一致
- ✓ Ref 回看窗口 / 嵌套回看窗口累加（Mean(Ref($close,5),20) = 25 天）
- ✓ 因子族批量计算（25 因子）+ 元信息含回看窗口
- ✓ 性能：25 因子 × 50 股 × 120 日 < 10s
- ✓ 未知字段报错 / 无效表达式报错 / 自定义算子注册

### 优化点 3：三重约束因子筛选器

**文件**: [factor_screener.py](file:///workspace/optimizations/20260624/factor_screener.py)

**借鉴来源**: QuantaAlpha 论文 Section 3.4 + 东方证券研报 A 股增强方向

**解决的现有缺陷**（对照 [factor-engine/engine.py](file:///workspace/skills/factor-engine/engine.py) `correlation_analysis()`）：

| 缺陷 | 现有实现 | 优化后实现 |
|------|----------|------------|
| 无预测力门槛 | 仅相关性去冗余 | Rank IC + IC_IR + t_stat 三重显著性检验 |
| 无容量约束 | 无 | 换手率 + 流动性容量评分 |
| 去冗余策略粗糙 | 按名称长度取舍 | 贪心 RankIC 降序，池化相关性 < 0.7 才入库 |
| 无中性化 IC | 无 | 行业/市值中性化 IC，剥离系统性风格暴露 |
| 无谱系追踪 | 无 | `FactorLineage` 记录因子来源与谱系 |
| 相关性计算缺陷 | 截面均值相关性（洗掉信号） | 池化相关性（保留个股差异） |

**三重约束流程**：
1. **Rank IC 门槛**：|IC均值| ≥ min_rank_ic 且 |IC_IR| ≥ min_ic_ir 且 |t_stat| ≥ min_ic_t_stat
2. **容量门槛**：换手率 ≤ max_turnover 且 流动性 ≥ min_liquidity
3. **低冗余贪心去重**：按 IC_IR 降序，与池中因子池化相关性 < max_correlation 才入库

**测试覆盖**（测试组 10-13）：
- ✓ 噪声因子被 IC 过滤（现有方法不过滤）
- ✓ 有效因子通过 IC
- ✓ 冗余因子对去重（高相关对至多一个入库）
- ✓ 入库因子数量合理 / 谱系记录存在
- ✓ 中性化 IC 计算（A 股增强）
- ✓ 与现有方法对比（三重约束过滤噪声，现有方法不过滤）
- ✓ 单因子筛选不崩溃 / 全不达标返回空

---

## 四、测试结果汇总

```
============================================================
jingni-trader 优化验证测试
分支: feat/quant-opt-20260624
日期: 2026-06-24
============================================================
测试汇总: 42/42 通过, 0 失败
============================================================
```

| 测试组 | 测试数 | 通过 | 覆盖类型 |
|--------|--------|------|----------|
| Exchange 正确性 | 9 | 9 | 涨跌停/停牌/T+1/容量/冲击成本/费用 |
| Exchange 防前视 | 2 | 2 | deal_price 语义/性能 |
| Exchange 边界 | 3 | 3 | 空数据/资金不足/全停牌 |
| 表达式解析 | 6 | 6 | 字段/常量/函数/二元/嵌套/序列化 |
| 因子计算正确性 | 4 | 4 | Ref/Mean/收益率/波动率 与硬编码对比 |
| 回看窗口声明 | 2 | 2 | 单层/嵌套累加 |
| 因子族批量计算 | 2 | 2 | 25 因子/元信息 |
| 表达式性能 | 1 | 1 | 25 因子 × 50 股 × 120 日 |
| 表达式边界 | 3 | 3 | 未知字段/无效表达式/自定义算子 |
| 三重约束筛选 | 5 | 5 | 噪声过滤/有效通过/冗余去重/数量/谱系 |
| 中性化 IC | 1 | 1 | A 股增强 |
| 与现有方法对比 | 2 | 2 | 噪声过滤对比/报告生成 |
| 筛选器边界 | 2 | 2 | 单因子/全不达标 |
| **合计** | **42** | **42** | |

---

## 五、对比分析

### 5.1 回测引擎对比

| 维度 | 现有 native_adapter | 优化后 Exchange |
|------|---------------------|-----------------|
| 前视偏差 | ❌ 收盘价决策+成交 | ✅ deal_price="$open" 次日开盘 |
| T+1 | ⚠️ 靠买卖顺序巧合 | ✅ Lot 批次池显式约束 |
| 停牌 | ❌ 未检查 close=None | ✅ check_stock_suspended |
| 成交量容量 | ❌ 无限制 | ✅ 10% 裁剪 |
| 冲击成本 | ❌ 仅固定滑点 | ✅ 线性 impact_cost |
| 过户费 | ❌ 配置未用 | ✅ 计入成本 |
| 最低手续费 | ✅ max(,5) | ✅ max(,5) |

### 5.2 因子定义方式对比

| 维度 | 现有 compute_a_share_factors | 优化后表达式引擎 |
|------|------------------------------|------------------|
| 定义方式 | Python 硬编码 | 字符串表达式 |
| 可序列化 | ❌ | ✅ __str__ 往返 |
| 可缓存 | ❌ | ✅ AST 缓存 |
| 可配置 | ❌ 改代码 | ✅ YAML/字典配置 |
| 边界处理 | 人工 | ✅ get_extended_window_size |
| 可审计 | ❌ | ✅ AST 元信息 |
| 算子扩展 | 改代码 | ✅ OperatorRegistry.register |

### 5.3 因子筛选对比

| 维度 | 现有 correlation_analysis | 优化后三重约束筛选 |
|------|---------------------------|---------------------|
| IC 门槛 | ❌ | ✅ IC + IC_IR + t_stat |
| 容量约束 | ❌ | ✅ 换手率 + 流动性 |
| 去冗余策略 | 按名称长度 | 贪心 RankIC 降序 |
| 相关性计算 | 截面均值（洗信号） | 池化（保留差异） |
| 中性化 IC | ❌ | ✅ 行业/市值中性化 |
| 谱系追踪 | ❌ | ✅ FactorLineage |
| 噪声过滤 | ❌ 不过滤 | ✅ 过滤 |

---

## 六、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260624` 分支验证通过，**等待用户确认后方可合并到 main**：

### 建议合并的优化（高置信度）

1. **防前视 Exchange 类** → 替换 native_adapter 的核心交易逻辑
   - 风险：低（纯新增独立文件，现有 native_adapter 未修改）
   - 收益：修复前视偏差这一关键正确性问题

2. **因子表达式引擎** → 作为 factor-engine 的可选后端
   - 风险：低（独立模块，可渐进集成）
   - 收益：因子可配置化、可缓存、可审计

3. **三重约束因子筛选器** → 替换 correlation_analysis
   - 风险：低（独立模块）
   - 收益：过滤无效因子，提升因子库质量

### 后续可探索方向（需进一步验证）

| 方向 | 借鉴来源 | 预估难度 |
|------|----------|----------|
| `.bin` 时序存储替代 Parquet | Qlib L2 | 中 |
| PIT Provider 防前视（基本面数据） | Qlib L3 | 中 |
| 四级缓存机制 | Qlib L4 | 中 |
| qrun + YAML 工作流编排 | Qlib cli | 中 |
| RollingGen 滚动窗口训练 | Qlib task | 中 |
| StructuredCovEstimator 风险模型 | Qlib riskmodel | 高 |
| LLM 驱动因子挖掘 | QuantaAlpha | 高 |
| 因子 AST 符号化去重 | QuantaAlpha | 高 |

### 重要约束遵守说明

- ✅ 所有新代码位于 `optimizations/20260624/` 独立目录，未修改 main 分支任何代码
- ✅ 已创建 `feat/quant-opt-20260624` 分支并推送（仅 push，未 merge）
- ⏳ **等待用户确认后方可执行 git merge / PR 合入 main**

---

## 七、文件清单

```
optimizations/20260624/
├── exchange.py              # 防前视 Exchange 回测交易所（借鉴 Qlib）
├── factor_expression.py     # 因子表达式引擎 + 算子注册表（借鉴 Qlib）
├── factor_screener.py       # 三重约束因子筛选器（借鉴 QuantaAlpha）
├── test_optimizations.py    # 验证测试（42 项，全部通过）
├── test_results.json        # 测试结果 JSON
└── REPORT.md                # 本报告
```
