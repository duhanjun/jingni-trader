# 量化交易开源项目学习与 jingni-trader 优化验证报告

> **执行日期**: 2026-06-20
> **分支**: `feat/quant-opt-20260620`
> **状态**: 验证完成，待用户确认是否合并到 main

---

## 一、学习项目清单及核心亮点

### 1. Microsoft Qlib (36.5k★) — AI 量化研究工业级框架
- **仓库**: https://github.com/microsoft/qlib
- **论文**: https://arxiv.org/abs/2009.11189
- **核心亮点**:
  - **Point-in-Time (PIT) 数据库**: 文件式存储 `(date, period, value, _next)`，按发布时间组织财务数据，确保回测时每个时间点只能拿到当时已公开的信息，从根本上防止未来数据泄露（look-ahead bias）
  - **声明式因子表达式引擎**: 递归下降解析器将 `MA($close,20)-MA($close,5)` 等表达式解析为 AST，通过 `ElemOperator`/`PairOperator`/`Rolling` 算子树执行，新增因子无需写 Python 代码
  - **Model Zoo**: 内置 20+ SOTA 模型（LightGBM/XGBoost/LSTM/GRU/Transformer/GATs/TFT）
  - **RD-Agent**: LLM 驱动的自动因子挖掘与模型优化
  - **滚动验证 (Walk-forward)**: 时序合适的训练/验证切分，避免随机切分导致的泄露

### 2. AKQuant (Rust+Python, 2026 新发布) — 高性能混合框架
- **仓库**: https://pypi.org/project/akquant/
- **核心亮点**:
  - **Rust 内核 + Python 接口**: 零开销抽象与 Zero-Copy 数据架构，显著降低 Python 层开销
  - **Polars 驱动的因子表达式引擎**: 支持 `Rank(Ts_Mean(Close,5))` 等 Alpha101 风格公式，自动并行计算
  - **Walk-forward Validation 框架**: 内置滚动训练验证
  - **多进程网格搜索**: 策略参数并行优化
  - **基准对比报告**: 自动计算 Alpha/Beta/信息比率/跟踪误差

### 3. Freqtrade + FreqAI (44k★) — AI 驱动量化框架
- **仓库**: https://github.com/freqtrade/freqtrade
- **核心亮点**:
  - **FreqAI ML 管道**: 在策略逻辑内直接训练 LightGBM/XGBoost 预测模型
  - **Optuna 超参搜索**: NSGA-III 采样器 + Sharpe 比率损失函数
  - **Walk-Forward Optimization (WFO)**: 滚动重训练/验证方法论，鲁棒模型验证
  - **回测与实盘一致性**: Hyperopt 复用同一回测模块，保证结果可复现

### 4. 其他参考项目
- **QuantConnect Lean** (15.5k★): Alpha Streams 模块化架构
- **vn.py** (27.8k★): A 股实盘交易瑞士军刀
- **pyfolio / Alphalens** (Quantopian 遗产): 基准相对绩效指标、因子 IC 分析

---

## 二、可借鉴的方向列表

对照 jingni-trader 现有代码（`skills/backtest-engine/scripts/adapters/native_adapter.py`、`skills/factor-engine/scripts/adapters/pandas_ta_calculator.py`、`skills/backtest-engine/scripts/base/base_backtest.py`），识别出以下可借鉴方向：

| # | 借鉴方向 | 借鉴来源 | jingni-trader 现有问题 | 优先级 |
|---|---------|---------|----------------------|--------|
| 1 | 声明式因子表达式引擎 | Qlib / AKQuant | `pandas_ta_calculator.py` 用 if/elif 硬编码每个因子，新增因子需改源码；逐股票 `for code in unique()` 循环慢 | 高 |
| 2 | T+1 执行防前视偏差 | Qlib PIT | `native_adapter.py` 信号日即执行日（用当日 close 成交），存在前视偏差 | 高 |
| 3 | 向量化回测 | Moonshot / AKQuant | `native_adapter.py` 用 `data[data['date']==dt]` 逐日过滤，O(n·m) 性能差 | 高 |
| 4 | 基准相对风险指标 | pyfolio | `base_backtest.py` 缺 Alpha/Beta/信息比率/跟踪误差/VaR/CVaR/Omega | 中 |
| 5 | 因子预处理流水线 | Qlib / Alphalens | factor-engine 仅计算因子值，无缩尾/标准化/中性化 | 中 |
| 6 | 因子 IC 分析 | Alphalens | 无 IC/Rank IC/ICIR/IC 衰减/分组收益分析 | 中 |
| 7 | target_weight 仓位信号 | Qlib | `native_adapter.py` 仅支持 1/-1 等额信号，不支持信号强度仓位 | 中 |
| 8 | Point-in-Time 财务数据 | Qlib PIT DB | `base_data_provider.get_financial` 未处理财报修订 | 低（架构改动大） |

---

## 三、已完成的验证测试及结论

### 3.1 验证代码结构

```
quant_opt_run2_20260620/            # 本次运行独立目录（与远程已有工作隔离）
├── __init__.py
├── factor_expression_engine.py     # 优化点 1: 声明式因子表达式引擎
├── enhanced_backtest_engine.py     # 优化点 2/3/4/7: T+1+向量化+基准指标+权重信号
├── factor_analysis.py              # 优化点 5/6: 预处理 + IC 分析
├── REPORT.md                       # 本报告
└── tests/
    ├── __init__.py
    ├── synthetic_data.py           # 合成数据生成器
    └── test_optimizations.py       # 28 项测试
```

### 3.2 三个优化模块说明

#### 模块 A: 因子表达式引擎 (`factor_expression_engine.py`)
- **借鉴**: Qlib Expression Engine 的 AST 解析 + 算子树设计
- **实现**:
  - 递归下降解析器：表达式字符串 → AST（`Field`/`Constant`/`BinOp`/`UnaryOp`/`Rolling`/`CrossSection`）
  - 算子：算术(`+-*/**`)、一元(`Abs/Log/Sign/Neg`)、滚动(`MA/STD/MAX/MIN/SUM/Ref/Delta/WMA`)、截面(`Rank/ZScore`)
  - 预置 10 个 Alpha101 风格因子（动量/反转/均线偏离/波动率/量价/布林位置）
- **对比现有**: 替代 `pandas_ta_calculator._calc_single` 的逐股票 if/elif 链

#### 模块 B: 增强回测引擎 (`enhanced_backtest_engine.py`)
- **借鉴**: Qlib PIT 防泄露 + Moonshot 向量化 + pyfolio 基准指标
- **实现**:
  - **T+1 执行**: 信号 T 日生成，映射到 T+1 交易日开盘成交，杜绝前视偏差
  - **向量化切片**: `groupby(date)` 一次性建索引，替代逐日 `data[data['date']==dt]` 全表扫描
  - **target_weight 信号**: 支持信号强度仓位（现有仅 1/-1 等额）
  - **新增 7 项风险指标**: Alpha/Beta(_CAPM)/Information Ratio/Tracking Error/VaR_95/CVaR_95/Omega
  - 涨跌停拒绝、滑点、佣金、印花税、T+1 整百手规则齐全
- **对比现有**: 修复 `native_adapter` 信号日即执行日前视偏差；新增基准相对指标

#### 模块 C: 因子分析与预处理 (`factor_analysis.py`)
- **借鉴**: Alphalens IC 分析 + Qlib 因子预处理
- **实现**:
  - **预处理**: Winsorize（缩尾）、Standardize（截面 z-score）、Neutralize（行业+市值中性化，最小二乘残差）
  - **IC 分析**: 前向收益计算、IC/Rank IC（手写 spearman 免 scipy 依赖）、ICIR、IC 正比例、IC 衰减
  - **分组收益**: 按分位数分组计算各组前向收益均值，验证因子单调性
- **对比现有**: factor-engine 完全无预处理与 IC 分析能力

### 3.3 测试结果（28/28 通过）

```
=== 测试汇总: 28/28 通过, 0 失败 ===
```

| 测试组 | 测试项 | 结果 | 关键数据 |
|--------|--------|------|---------|
| 1. 因子表达式正确性 | MA/Ref/复合/Rank/预置因子 | 5/5 ✓ | 10 个预置因子全部可解析 |
| 2. 因子表达式性能 | vs 逐股票循环 | 1/1 ✓ | **加速 6.82x**（36ms vs 245ms，50股票×250天×3因子） |
| 3. 因子表达式边界 | 空数据/单股票/未知字段/语法错误 | 4/4 ✓ | 异常路径正确处理 |
| 4. 回测正确性 | 结构/净值/资金守恒/T+1/指标 | 5/5 ✓ | T+1 验证: 信号 01-17 → 成交 01-18 |
| 5. 回测 vs native | 独立可用性 | 1/1 ✓ | native 因包依赖不可导入，增强引擎独立运行 |
| 6. 回测边界 | 空数据/无信号/全涨停/权重信号 | 4/4 ✓ | 全涨停买入被拒、target_weight 模式工作 |
| 7. 因子预处理 | 缩尾/标准化/中性化/流水线 | 4/4 ✓ | 中性化: 相关性 0.993 → -0.054（降 95%） |
| 8. 因子 IC | 前向收益/IC 正/衰减/分组单调 | 4/4 ✓ | IC=0.98（强预测力因子）、Q5>Q1 单调 |

### 3.4 关键性能数据

- **因子计算加速**: 表达式引擎 36ms vs 逐股票循环 245ms，**6.82 倍加速**（50 股票 × 250 天 × 3 因子）
- **T+1 前视偏差修复**: 信号日 2023-01-17 → 成交日 2023-01-18，验证执行延迟正确
- **中性化效果**: 因子-市值相关性从 0.993 降至 -0.054（降低 95%）
- **IC 分析精度**: 强预测力因子 IC=0.9818，ICIR=147.48，5 日 IC 高于 1 日和 20 日（符合预期衰减）

---

## 四、待用户确认的优化建议

以下优化方案已在 `feat/quant-opt-20260620` 分支验证通过，**未合并到 main**，等待用户确认：

### 建议 1（高优先级）: 用因子表达式引擎替换 pandas_ta_calculator 的硬编码链
- **改动范围**: `skills/factor-engine/scripts/adapters/`
- **风险**: 低（新引擎与现有 `BaseFactorCalculator` 接口兼容，可并存）
- **收益**: 新增因子从"改源码"变为"写表达式字符串"；计算性能提升 ~7x

### 建议 2（高优先级）: 回测引擎引入 T+1 执行与向量化
- **改动范围**: `skills/backtest-engine/scripts/adapters/native_adapter.py`
- **风险**: 中（T+1 会改变回测结果，需重新校准既有策略预期）
- **收益**: 消除前视偏差，回测更贴近实盘；性能提升

### 建议 3（中优先级）: 扩展风险指标体系
- **改动范围**: `skills/backtest-engine/scripts/base/base_backtest.py`
- **风险**: 低（纯新增，不破坏现有指标）
- **收益**: 补齐 Alpha/Beta/IR/跟踪误差/VaR/CVaR/Omega，绩效评估更专业

### 建议 4（中优先级）: 集成因子预处理与 IC 分析
- **改动范围**: `skills/factor-engine/` 新增 analysis 子模块
- **风险**: 低（新增能力，不改现有流程）
- **收益**: 因子上线前可做 IC 筛选与中性化，提升因子库质量

### 建议 5（低优先级，长期）: 引入 Point-in-Time 财务数据存储
- **改动范围**: `skills/data-engine/` 架构级改动
- **风险**: 高（需重构数据存储格式）
- **收益**: 从根本解决财务数据回测泄露问题

---

## 五、合规说明

- ✅ 所有新代码位于 `feat/quant-opt-20260620` 分支独立目录 `quant_opt_20260620/`
- ✅ 未修改 main 分支任何代码
- ✅ 未执行 git merge
- ✅ 分支已推送至 GitHub 远程仓库（仅 push，不合并）
- ⏳ 等待用户确认后方可执行 merge / PR 合入 main

---

## 六、复现方式

```bash
# 切换到优化分支
git checkout feat/quant-opt-20260620

# 安装依赖
pip install numpy pandas

# 运行全部 28 项验证测试
python -m quant_opt_run2_20260620.tests.test_optimizations
```
