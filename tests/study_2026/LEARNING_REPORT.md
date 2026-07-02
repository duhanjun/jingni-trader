# jingni-trader 量化交易学习报告 #1

> 日期: 2026-06-13
> 序号: #1
> 当前分支: feature/quant-stream-inspired

---

## 一、学习项目清单及核心亮点

### 1.1 Microsoft Qlib
- **仓库**: [https://github.com/microsoft/qlib](https://github.com/microsoft/qlib)
- **Stars**: ~44,000
- **最新提交**: 2026-04 (活跃维护中)
- **语言**: Python

**核心亮点**:

| 特性 | 说明 |
|------|------|
| Alpha158 因子集 | 内置 158 个标准化因子，覆盖 K线、价格、成交量、换手率等类别 |
| 二进制数据引擎 | 使用 Parquet/Feather 格式存储，10x 快速数据加载 |
| 表达式 DSL | 因子通过表达式定义，支持嵌套组合，如 `Ref($close, -5) / $close - 1` |
| Walk-forward 训练 | 内置 Rolling Trainer，支持 expanding/sliding window |
| Point-in-Time 数据库 | 防止 look-ahead bias，确保训练集不包含未来信息 |
| RD-Agent | 自动化因子挖掘，通过强化学习搜索有效因子组合 |
| 模型 Zoo | 集成 LightGBM、CatBoost、XGBoost、LSTM、GRU、Transformer 等 |

### 1.2 QUANTAXIS
- **仓库**: [https://github.com/yutiansut/QUANTAXIS](https://github.com/yutiansut/QUANTAXIS)
- **Stars**: ~25,000
- **最新提交**: 2026-02
- **语言**: Python + Rust

**核心亮点**:

| 特性 | 说明 |
|------|------|
| Python+Rust 混合架构 | 性能关键路径用 Rust 实现（QARSBridge），Python 自动降级 |
| QIFI 协议 | 统一账户/持仓/订单模型，跨券商标准化 |
| 零拷贝数据桥 | Rust ↔ Python 之间使用 PyO3 实现零拷贝数据传输 |
| 事件驱动微服务 | Pub/Sub 架构处理实时行情，解耦数据流和处理流 |
| 因子表达式引擎 | 支持白名单校验的自定义因子表达式 |
| Clickhouse 数据存储 | 大规模时序数据的实时分析 |

### 1.3 AKQuant
- **仓库**: [https://github.com/akfamily/akquant](https://github.com/akfamily/akquant)
- **语言**: Rust + Python
- **定位**: 新一代量化平台

**核心亮点**:

| 特性 | 说明 |
|------|------|
| Rust 核心 | 计算密集型模块用 Rust 编写，Python 提供上层接口 |
| Walk-forward Validation | 内置完整的滚动验证框架，集成 PyTorch/Scikit-learn |
| TA-Lib 指标 | Rust 实现的 150+ 技术指标 |
| Zero-Copy 架构 | 跨语言数据传输零开销 |

---

## 二、可借鉴的方向列表

基于对上述项目的研究和对 jingni-trader 代码的全面审查，识别出以下优化方向：

### 2.1 回测引擎优化（已验证 ✓）
- **借鉴来源**: Qlib 数据驱动回测设计
- **问题**: jingni-trader 的 `native_adapter.py` 使用 `for dt in dates` 逐日 Python 循环
- **方案**: 引入 numpy 向量化批量计算路径
- **验证结果**: 平均加速 7.40x，结果完全一致
- **影响模块**: `backtest-engine`

### 2.2 因子定义系统扩展（已验证 ✓）
- **借鉴来源**: Qlib Alpha158 + QUANTAXIS 因子表达式
- **问题**: 因子硬编码在 `compute_a_share_factors()` 中，新增因子需改核心代码。仅 ~15 个因子
- **方案**: 因子注册表 + 表达式 DSL + 安全白名单
- **验证结果**: 20+ 内置因子定义，表达式与手动计算偏差为 0，1 行代码注册自定义因子
- **影响模块**: `factor-engine`

### 2.3 Walk-forward 验证框架（已验证 ✓）
- **借鉴来源**: Qlib Rolling Trainer + AKQuant Walk-forward + Lopez de Prado
- **问题**: 当前 `_prepare_data.py` 中的 `prepare_time_series_data()` 只有一次静态 train_test split
- **方案**: PurgedWalkForwardCV 支持 sliding/expanding window + purge gap + embargo
- **验证结果**: 正确生成 7 个滚动折叠，信息泄露检测有效，IC 标准差可衡量模型稳定性
- **影响模块**: `strategy-model-engine`

### 2.4 数据管道优化（待验证）
- **借鉴来源**: Qlib 二进制数据格式 + 增量更新
- **问题**: jingni-trader 使用 Parquet 但无增量缓存，每次重新拉取全量数据
- **方案**: 参考 Qlib 的 D.instrument 存量+增量双层缓存
- **影响模块**: `data-engine`

### 2.5 组合风险模型增强（待验证）
- **借鉴来源**: Qlib 的组合优化模块 + AKQuant 的多目标优化
- **问题**: jingni-trader 目前仅集成 PyPortfolioOpt，风险模型较简单
- **方案**: 引入 CVaR 约束、行业中性约束、换手率限制，支持多目标优化
- **影响模块**: `portfolio-risk-engine`

---

## 三、已完成的验证测试及结论

### 3.1 向量化回测性能对比

**测试文件**: `tests/study_2026/test_vectorized_backtest.py`

**测试内容**:
1. 正确性验证: 向量化 vs 逐日循环，结果一致性
2. 性能对比: 4 组不同数据规模（50/100/200/300 只股票 × 252/500/756 天）
3. 边界条件: 空数据、单日单股、全卖出信号

**测试结果**:

| 数据规模 | 向量化耗时 | 循环耗时 | 加速比 | 一致性 |
|----------|-----------|----------|--------|--------|
| 50股 × 252天 | 0.010s | 0.063s | 6.29x | ✓ |
| 100股 × 500天 | 0.025s | 0.196s | 7.75x | ✓ |
| 200股 × 500天 | 0.046s | 0.338s | 7.32x | ✓ |
| 300股 × 756天 | 0.089s | 0.730s | 8.22x | ✓ |

**结论**: 向量化路径在保证 100% 结果一致的前提下，平均性能提升 7.40x。在 300 只股票 × 3 年数据规模下，加速比达 8.22x。建议作为可选回测路径引入。

### 3.2 表达式因子定义系统

**测试文件**: `tests/study_2026/test_expression_factor.py`

**测试内容**:
1. 因子注册表完整性（20 个内置因子，8 个分类）
2. momentum_1d / reversal_5d / volatility_20d 与手动计算对比
3. 自定义因子注册（1 行代码）
4. 安全白名单校验
5. 批量因子计算性能

**测试结果**:

| 因子 | 与手动计算的最大偏差 | 状态 |
|------|---------------------|------|
| momentum_1d | 0.0000000000 | ✓ |
| reversal_5d | 0.0000000000 | ✓ |
| volatility_20d | 0.0000000000 | ✓ |
| custom_mom_vol_ratio | - | ✓ (900 有效值) |

**结论**: 表达式引擎计算完全正确，1 行代码即可注册新因子。白名单机制有效拦截 `exec()` 等非法函数。建议引入因子注册表 + 表达式机制替代当前的硬编码方式。

### 3.3 Walk-forward 验证框架

**测试文件**: `tests/study_2026/test_walkforward_validation.py`

**测试内容**:
1. Sliding/Expanding window 分割生成
2. 信息泄露检测（有/无 purge gap）
3. Walk-forward vs 静态 Split 的 IC/MSE 对比
4. 性能对比

**测试结果**:

```
Sliding Window: 7 folds generated
  Fold 1: Train [2021-01-01 ~ 2021-12-20] → Test [2021-12-28 ~ 2022-03-24]
  ...
  Fold 7: Train [2022-06-15 ~ 2023-06-01] → Test [2023-06-09 ~ 2023-09-05]

Expanding Window: 7 folds generated
  Fold 1: Train [2021-01-01 ~ 2021-12-13] → Test [2021-12-21 ~ 2022-03-17]
  ...
  Fold 7: Train [2021-01-01 ~ 2023-05-25] → Test [2023-06-02 ~ 2023-08-29]

Walk-forward 验证 (5 folds):
  Fold 1: IC=0.0003  MSE=0.000011  R²=-0.0140
  Fold 2: IC=-0.0246 MSE=0.000012  R²=-0.0104
  Fold 3: IC=-0.0090 MSE=0.000011  R²=-0.0033
  Fold 4: IC=-0.0045 MSE=0.000011  R²=-0.0013
  Fold 5: IC=0.0068  MSE=0.000010  R²=-0.0002

  平均 IC: -0.0062 ± 0.0106
  静态 IC: 0.0124

  信息泄露检测:
    有 purge gap → SAFE ✓
    无 purge gap → HIGH 风险 ✓ (正确检测)
```

**结论**: Walk-forward 框架提供 5-7 次独立验证，比静态 split 更稳健。IC 标准差可作为模型稳定性指标。建议替代当前的 `prepare_time_series_data()` 中的单次 split。

---

## 四、待用户确认的优化建议

### 优先级 HIGH

1. **引入向量化回测路径** (`backtest-engine`)
   - 在 `native_adapter.py` 中新增 `run_vectorized()` 方法
   - 保持现有 `for dt in dates` 作为 fallback
   - 预期收益: 5-8x 性能提升

2. **因子注册表 + 表达式机制** (`factor-engine`)
   - 在 `base_factor.py` 中新增 `FactorRegistry` 类和 `ExpressionEngine`
   - 现有 `compute_a_share_factors()` 作为注册表初始化输入
   - 预期收益: 因子可扩展性大幅提升，便于社区贡献

3. **PurgedWalkForwardCV** (`strategy-model-engine`)
   - 在 `_prepare_data.py` 中新增 `walkforward_split()` 函数
   - 模型训练支持 `mode='walkforward'`
   - 预期收益: 模型评估更可靠，避免前视偏差

### 优先级 MEDIUM

4. **数据双层缓存** (`data-engine`)
   - 参考 Qlib 的 D.instrument 设计
   - 存量数据 + 增量更新，减少重复拉取

5. **组合优化增强** (`portfolio-risk-engine`)
   - CVaR 约束、行业中性、换手率限制

### 优先级 LOW

6. **Rust 核心模块** (`backtest-engine` / `factor-engine`)
   - 参考 QUANTAXIS 的 QARSBridge，将热点计算用 Rust 重写
   - 长期优化，短期用 numpy 向量化即可

---

## 五、测试文件清单

| 文件 | 行数 | 测试通过 | 借鉴来源 |
|------|------|----------|----------|
| `tests/study_2026/test_vectorized_backtest.py` | ~310 | 全部 ✓ | Qlib |
| `tests/study_2026/test_expression_factor.py` | ~425 | 全部 ✓ | Qlib + QUANTAXIS |
| `tests/study_2026/test_walkforward_validation.py` | ~580 | 全部 ✓ | Qlib + AKQuant + Lopez de Prado |
| `tests/study_2026/LEARNING_REPORT.md` | 本文件 | - | - |

---

**重要提醒**: 
- 所有优化代码位于 `tests/study_2026/` 独立测试目录中，未修改主代码
- 在用户明确确认优化方案之前，**不会执行** git commit / push / merge
- 待用户确认后，需将验证通过的代码迁移到对应模块中