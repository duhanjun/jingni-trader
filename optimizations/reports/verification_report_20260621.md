# jingni-trader 量化优化验证报告

> **执行日期**: 2026-06-21
> **分支**: `feat/quant-opt-20260621` (基于 main, 未合并)
> **执行人**: 自动化学习与优化流程

---

## 一、学习项目清单及核心亮点

本次联网调研了 GitHub、arXiv、PyPI 上的近期活跃量化交易开源项目，挑选以下 3 个最具借鉴价值的项目深入学习：

### 1. Microsoft Qlib (17.5K+ Stars)
- **仓库**: https://github.com/microsoft/qlib
- **定位**: AI 导向的量化投资全流程平台
- **核心亮点**:
  - **表达式引擎 (Expression Engine)**: 声明式因子定义，字符串表达式 → AST → 计算树，无需写 Python 即可定义任意复杂因子
  - **Alpha158 / Alpha360 因子库**: 158 个经历史回测验证的标准化因子，覆盖趋势/反转/量价/波动/流动性/市值六大维度
  - **算子体系**: ElemOperator(一元) / PairOperator(二元) / Rolling(滚动窗口) 分层设计，支持嵌套
  - **Walk-Forward 验证**: 滚动训练-测试，防止过拟合
  - **与 RD-Agent 集成**: LLM 驱动的自动因子挖掘与模型优化闭环

### 2. AKQuant (1.5K+ Stars)
- **仓库**: https://github.com/akfamily/akquant
- **定位**: Rust 内核 + Python 接口的高性能量化框架
- **核心亮点**:
  - **向量化回测**: Rust 零拷贝数据架构，回测性能显著优于纯 Python
  - **Walk-Forward Validation**: 内置滚动训练框架，无缝集成 PyTorch/sklearn
  - **因子表达式引擎**: Polars 驱动，支持 `Rank(Ts_Mean(Close, 5))` 等 Alpha101 风格公式
  - **参数优化**: 内置多进程网格搜索
  - **专业级风控**: 订单流管理与即时风控模块

### 3. alfa.rs / alfars (Rust + Python)
- **仓库**: https://github.com/EthanNOV56/alfa.rs
- **定位**: 高性能量化工作流，Rust 核心 + Python 绑定
- **核心亮点**:
  - **AST 表达式系统**: 自定义因子计算的表达式构建器
  - **惰性求值**: Polars 风格的延迟计算与查询优化
  - **遗传编程因子挖掘**: 自动发现高性能因子表达式
  - **因子库管理**: 搜索、缓存、版本化
  - **8-10x 加速**: Rust 并行计算

### 其他参考项目
- **vectorbt**: 极致向量化回测库，target_weight 模式
- **FactorEngine (arXiv:2603.16365)**: LLM 引导的程序级因子挖掘 + 贝叶斯优化
- **Microsoft RD-Agent**: 自动化因子挖掘与模型联合优化

---

## 二、jingni-trader 现有代码问题分析

通过逐文件阅读 jingni-trader 全部源码，识别出以下关键问题（按严重程度排序）：

### 致命问题 (P0)
| 问题 | 位置 | 影响 |
|------|------|------|
| sys.modules 操控切换 scripts 命名空间 | engine.py:170-173 | 模块状态丢失、单例失效、无法并行、测试复杂 |
| T+1 未实现（参数传入但从未使用） | native_adapter.py:27 | 当日买入当日可卖，违反 A 股规则，回测不可信 |
| RQAlpha/Backtrader 适配器是桩代码 | rqalpha_adapter.py:98, backtrader_adapter.py:43-47 | SKILL.md 默认 rqalpha 但实际不可用 |
| HRP 优化传入空 DataFrame | portfolio-risk-engine/engine.py:146 | 组合优化报错或返回无意义结果 |

### 高严重度 (P1)
| 问题 | 位置 | 影响 |
|------|------|------|
| 主 config 与子 config 参数不一致（SLIPPAGE 差 10 倍） | scripts/config.py vs 各子 config | 用户修改主 config 不生效 |
| 因子无注册机制，全部硬编码 | factor-engine/engine.py:48-117 | 无法动态扩展因子 |
| PandasTa/Talib calculator 加载但从未调用 | factor-engine/engine.py:31-46 | 死代码，资源浪费 |
| 过户费未计算 | native_adapter.py | 回测成本低估 |
| 涨跌停判断未区分板块（ST/科创/北交） | data-engine/engine.py:551 | 一刀切 9.9% 不准确 |
| Tushare 串行获取，MAX_WORKERS 定义但未用 | tushare_adapter.py:120 | 全市场获取极慢 |
| 前复权实现错误（返回后复权） | tushare_adapter.py:166-177 | 数据错误 |
| CVaR/Barra 归因是桩代码 | portfolio-risk-engine/engine.py:152-162, 294-306 | 功能名存实亡 |
| 风格暴露硬编码假数据 | reports-engine/engine.py:441-444 | 报告不反映真实持仓 |

### 中严重度 (P2)
| 问题 | 位置 | 影响 |
|------|------|------|
| IC 计算按日 Python 循环，性能差 | factor-engine/engine.py:250 | 大规模计算慢 |
| 无向量化回测选项 | backtest-engine | 全市场回测分钟级 |
| pnl 计算错误（现金流非盈亏） | native_adapter.py:83,115 | 胜率失真 |
| 无基准对比 | native_adapter.py | 无法评估超额收益 |
| 无数据缓存机制 | data-engine | 重复获取浪费配额 |
| 实盘模式未实现 | execution-monitor-engine/engine.py:329-332 | 仅模拟可用 |

---

## 三、可借鉴的方向列表

基于开源项目学习成果，结合 jingni-trader 现有问题，确定以下可借鉴优化方向：

| # | 优化方向 | 借鉴来源 | 解决的 jingni-trader 问题 | 优先级 |
|---|---------|---------|-------------------------|--------|
| 1 | **因子表达式引擎 + 注册机制** | Qlib Expression Engine + alfa.rs AST | 因子硬编码、无注册机制、calculator 未使用 | P0 |
| 2 | **向量化回测引擎 + T+1** | AKQuant + vectorbt | T+1 未实现、过户费缺失、性能差、无基准 | P0 |
| 3 | 模块化包架构（消除 sys.modules 操控） | Qlib 包结构 | sys.modules 致命缺陷 | P0 |
| 4 | 统一配置继承体系 | Qlib config 模式 | 主/子 config 不一致 | P1 |
| 5 | 数据缓存 + 增量更新 | Qlib 数据管理 | 无缓存、串行获取 | P1 |
| 6 | Walk-Forward 验证 | AKQuant / Qlib | config 有定义但代码未实现 | P1 |
| 7 | 板块涨跌停精确判断 | 通用 A 股规则 | 一刀切 9.9% | P1 |
| 8 | 完善风控断路器（回撤控制） | 通用风控模型 | 无回撤断路器 | P2 |
| 9 | QuantStats TearSheet 集成 | quantstats | 报告引擎未调用 | P2 |
| 10 | LLM 因子挖掘集成 | RD-Agent / FactorEngine | 仅关键词意图解析 | P3 |

本次验证聚焦于 **方向 1 和方向 2**（最高优先级、可独立验证、价值最大）。

---

## 四、已完成的验证测试及结论

### 4.1 优化点一：因子表达式引擎 + 注册机制

#### 借鉴来源
- Microsoft Qlib Expression Engine（声明式因子定义、AST 解析、算子体系）
- alfa.rs（AST 表达式系统、因子库管理）

#### 实现文件
- `optimizations/factor_expr/factor_registry.py` — 装饰器式因子注册中心（线程安全单例）
- `optimizations/factor_expr/expression_engine.py` — AST 表达式引擎（解析器 + 23 个算子）
- `optimizations/factor_expr/builtin_factors.py` — 14 个内置因子（7 大类别）
- `optimizations/factor_expr/test_factor_expr.py` — 37 个测试用例

#### 核心设计
1. **声明式因子定义**: 用字符串表达式描述因子，如 `MA($close, 20) - MA($close, 5)`，无需写 Python
2. **AST 解析器**: 递归下降解析器，支持字段引用(`$close`)、常量、函数调用、中缀运算、括号分组、一元负号
3. **算子体系** (参考 Qlib 分层):
   - 滚动算子: `Ref, MA, EMA, STD, VAR, SUM, MAX, MIN, QUANTILE, CORR, WMA`
   - 时序算子: `Delta, ROC, RSI`
   - 截面算子: `CSRank, CSZScore`（按日期分组）
   - 一元算子: `Log, Abs, Sign, Sqrt, Neg`
   - 二元算子: `Add, Sub, Mul, Div, Greater, Less`
4. **因子注册机制**: `@register_factor` 装饰器，携带元数据（名称/类别/方向/依赖字段/窗口/描述）
5. **向量化计算**: 全程 pandas rolling/groupby，无 Python 逐日循环
6. **解析缓存**: 同一表达式只解析一次

#### 测试结果（37/37 通过）

| 测试类别 | 测试数 | 结果 | 说明 |
|---------|--------|------|------|
| 表达式解析 | 8 | ✅ 全过 | 字段/常量/中缀/嵌套/一元负号/缓存/未知算子/语法错误 |
| 计算正确性 | 11 | ✅ 全过 | MA/Ref/ROC/Delta/STD/CSRank/CSZScore/CORR/RSI/复合/Log-Abs 与 pandas 手写一致 |
| 因子注册 | 8 | ✅ 全过 | 内置注册/元数据/分类过滤/单算/批算/自定义/方向校验/错误处理 |
| 边界条件 | 7 | ✅ 全过 | 单标的/大窗口/空数据/缺字段/除零/负窗口/NaN 传播 |
| 性能对比 | 3 | ✅ 全过 | 见下方性能数据 |

**性能数据**:
- 表达式引擎 vs 硬编码 pandas: **2.33x 开销**（含解析+AST 遍历，可接受，< 5x 阈值）
- 批量计算 10 个因子（120日×50股）: **210ms**
- 复合因子 `CSRank(-ROC($close,20)) + CSRank(-STD(ROC($close,1),20)) + CSRank(MA($turnover_rate,20))`（250日×30股）: **38.5ms**

#### 结论
✅ **验证通过**。因子表达式引擎可完全替代 jingni-trader 硬编码因子方式，提供声明式、可扩展、向量化计算的因子库，性能开销可接受（2.33x）。注册机制支持运行时动态扩展，解决了 calculator 加载未使用的问题。

---

### 4.2 优化点二：向量化回测引擎 + T+1

#### 借鉴来源
- AKQuant（Rust+Python 向量化回测）
- vectorbt（target_weight 模式向量化回测）

#### 实现文件
- `optimizations/vectorized_backtest/vectorized_engine.py` — 向量化回测引擎
- `optimizations/vectorized_backtest/test_vectorized_backtest.py` — 27 个测试用例

#### 核心设计
1. **向量化计算**: pandas/numpy 矩阵运算，无 Python 逐日循环（target_weight 模式）
2. **T+1 严格执行**:
   - target_weight 模式: 信号延迟一日执行，`actual_weights[T] = target_weights[T-1]`
   - signal 模式: 信号 shift(1) + buy_date 跟踪，买入次日才能卖
3. **完整 A 股费用模型** (`CostModel`):
   - 佣金: 万2.5，最低5元（双向）
   - 印花税: 千1（仅卖出）
   - **过户费: 万0.2（双向）** ← 修复 native_adapter 缺失
   - 滑点: 万1
4. **板块涨跌停精确判断** (`detect_board` + `LIMIT_TABLE`):
   - 沪深主板 10% / ST 5% / 科创板 20% / 创业板 20% / 北交所 30%
   - 修复 native_adapter 一刀切 9.9% 的问题
5. **两种回测模式**:
   - `run_target_weight`: 目标权重模式（推荐，极快）
   - `run_signal`: 买卖信号模式（兼容 native_adapter 接口）
6. **基准对比**: 自动计算超额收益、信息比率、跟踪误差
7. **完整绩效指标**: total_return / annual_return / volatility / sharpe / max_drawdown / calmar / sortino / win_rate / IR / tracking_error / turnover / 偏度峰度

#### 测试结果（27/27 通过）

| 测试类别 | 测试数 | 结果 | 说明 |
|---------|--------|------|------|
| 费用模型 | 4 | ✅ 全过 | 买入/卖出/卖出高于买入/过户费包含 |
| 板块识别 | 4 | ✅ 全过 | 主板/科创/创业/北交所 |
| T+1 执行 | 2 | ✅ 全过 | 信号延迟一日/同日买卖被阻止 |
| 回测核心 | 7 | ✅ 全过 | 买入持有/等权/全空仓/涨停限买/跌停限卖/基准/换手率 |
| 绩效指标 | 4 | ✅ 全过 | 已知收益/最大回撤/带基准/空净值 |
| 边界条件 | 5 | ✅ 全过 | 单日/单股/极端权重/NaN价格/索引不对齐 |
| 性能对比 | 1 | ✅ 全过 | 见下方性能数据 |

**性能数据**:
- 向量化 vs 逐日循环（250日×100股）: **30.8x 加速**（6.6ms vs 203.5ms）
- 端到端回测（250日×30股，月度调仓）: **7.6ms**

#### 关键修复对比

| 问题 | native_adapter (现状) | 向量化引擎 (优化后) |
|------|----------------------|-------------------|
| T+1 | 参数传入但未使用，当日买当日可卖 | 严格执行，信号延迟一日 + buy_date 跟踪 |
| 过户费 | 未计算 | 万0.2 双向计算 |
| 涨跌停 | 一刀切 9.9% | 按板块区分（10%/5%/20%/30%） |
| 成交价 | 当日收盘价（实盘不可实现） | 次日开盘价（符合实盘） |
| 性能 | 逐日 Python 循环 | 向量化矩阵运算，30.8x 加速 |
| 基准对比 | 无 | 超额收益/IR/跟踪误差 |
| pnl | 现金流非盈亏 | 基于持仓市值变化 |

#### 结论
✅ **验证通过**。向量化回测引擎修复了 native_adapter 的全部致命问题（T+1/过户费/涨跌停/成交价），性能提升 30.8x，并新增基准对比与完整绩效指标。可作为 native_adapter 的替代方案。

---

### 4.3 端到端集成验证

运行 `python -m optimizations.demo` 完成全流程验证：

```
因子表达式计算 (38.5ms) → Top-5 月度调仓信号 → 向量化回测 (7.6ms) → 绩效报告
```

- 14 个内置因子注册，7 大类别
- 复合因子表达式计算: 38.5ms
- 250 交易日 × 30 只股票回测: 7.6ms
- 全流程总耗时: ~46ms

---

## 五、待用户确认的优化建议

以下优化方向已通过验证，**等待用户确认后方可合并到 main 分支**：

### 建议一：将因子表达式引擎集成到 factor-engine（P0）
- **改动范围**: `skills/factor-engine/engine.py` + 新增 `scripts/expression/` 目录
- **内容**: 用表达式引擎替代 `compute_a_share_factors` 硬编码，保留向后兼容
- **风险**: 低（新增模块，不破坏现有接口）
- **预期收益**: 因子可声明式扩展，calculator 不再闲置

### 建议二：将向量化回测引擎作为 native_adapter 的替代（P0）
- **改动范围**: `skills/backtest-engine/scripts/adapters/` 新增 `vectorized_adapter.py`
- **内容**: 新增向量化适配器，config 默认切换为 `vectorized`，native_adapter 保留为备选
- **风险**: 中（需验证与现有 Context 接口兼容）
- **预期收益**: T+1 修复、30x 性能提升、费用模型完整

### 建议三：消除 sys.modules 操控（P0，未本次验证）
- **改动范围**: 全项目结构调整
- **内容**: 将子引擎改为独立 Python 包或使用相对导入
- **风险**: 高（影响所有引擎的导入方式）
- **预期收益**: 消除致命架构缺陷，支持并行执行
- **状态**: 需单独评估，建议作为下一阶段重点

### 建议四：统一配置继承体系（P1，未本次验证）
- **改动范围**: 各子引擎 `scripts/config.py`
- **内容**: 子 config 从主 config 继承，消除 10+ 处参数冲突
- **风险**: 中
- **预期收益**: 配置一致性，用户修改主 config 生效

---

## 六、测试运行方式

```bash
# 运行全部测试
cd /workspace
python -m pytest optimizations/ -v

# 单独运行因子表达式测试
python -m optimizations.factor_expr.test_factor_expr

# 单独运行向量化回测测试
python -m optimizations.vectorized_backtest.test_vectorized_backtest

# 运行端到端演示
python -m optimizations.demo
```

**测试统计**: 64 个测试用例，全部通过（37 因子表达式 + 27 向量化回测）

---

## 七、文件清单

```
optimizations/
├── __init__.py                              # 包入口
├── demo.py                                  # 端到端演示
├── factor_expr/                             # 因子表达式引擎
│   ├── __init__.py
│   ├── factor_registry.py                   # 因子注册中心
│   ├── expression_engine.py                 # AST 表达式引擎
│   ├── builtin_factors.py                   # 14 个内置因子
│   └── test_factor_expr.py                  # 37 个测试
├── vectorized_backtest/                     # 向量化回测引擎
│   ├── __init__.py
│   ├── vectorized_engine.py                 # 回测引擎 + 费用模型
│   └── test_vectorized_backtest.py          # 27 个测试
└── reports/
    └── verification_report_20260621.md      # 本报告
```

---

## 八、约束遵守说明

- ✅ 所有新代码位于 `feat/quant-opt-20260621` 分支的 `optimizations/` 目录，**未修改 main 分支任何代码**
- ✅ 未执行任何 `git merge` 操作
- ✅ 分支已推送到 GitHub 远程仓库（仅 push，不合并）
- ✅ 等待用户确认后方可合并到 main

---

*报告生成时间: 2026-06-21*
*分支: feat/quant-opt-20260621*
*测试结果: 64/64 通过*
