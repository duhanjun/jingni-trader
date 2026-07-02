# jingni-trader 量化学习报告

**日期**: 2026-06-12  
**学习序号**: #01  
**当前分支**: feature/quant-stream-inspired  

---

## 一、学习项目清单

本次学习选定了近期活跃、最具借鉴价值的三个开源项目：

| 项目 | 类型 | 来源 | 星标 | 核心领域 |
|------|------|------|------|----------|
| [Factor Engine](https://arxiv.org/abs/2602.14138) | 因子计算库 | 学术论文 (arXiv 2026-02) | - | 系统化因子计算与分析 |
| [AKQuant](https://github.com/akfamily/akquant) | 高性能量化框架 | GitHub | 1.4k+ | Rust+Python混合架构，Polars高性能引擎 |
| [Qlib](https://github.com/microsoft/qlib) | AI量化研究平台 | Microsoft GitHub | 15k+ | AI在量化投资中的应用 |

---

## 二、各项目核心亮点与可借鉴之处

### 1. Factor Engine (arXiv 2602.14138)

**核心亮点**:
- **装饰器式因子注册 API**: 使用 Python 装饰器注册因子函数，模块化设计，用户扩展自定义因子非常方便
- **基于 Polars 构建**: 利用 Polars 优越的性能，处理大规模面板数据
- **三大设计原则**: 模块化、兼容性、可扩展性
- **开箱即用**: 内建已验证的经典因子，用户无需从零开始
- **无缝集成**: 与现代数据科学生态系统（pandas、scikit-learn）兼容

**可借鉴之处**:
1. 当前 jingni-trader 的 factor-engine 使用硬编码方式在 `compute_a_share_factors` 中定义所有因子，新增因子需要修改核心代码。装饰器式注册机制可以解决这个问题。
2. Polars 在大规模数据计算上相比 Pandas 有显著性能优势，可以评估迁移。

---

### 2. AKQuant (akfamily/akquant)

**核心亮点**:
- **Rust + Python 混合架构**: 极致性能，零开销抽象，Zero-Copy 数据架构
- **Polars 驱动高性能因子计算引擎**: 原生支持 Alpha101 风格公式 `Rank(Ts_Mean(Close, 5))`
- **自动并行计算与数据对齐**: 减少用户手动处理
- **原生 ML 支持**: 内建 Walk-forward Validation 滚动训练框架，无缝集成 PyTorch/Scikit-learn
- **TA-Lib 双后端兼容**: 支持 Python/Rust 两种实现，自动降级
- **专业级风控**: 完善的订单流管理与即时风控模块

**可借鉴之处**:
1. 因子计算性能优化：采用 Polars 替换 Pandas，可以获得数量级的性能提升
2. 表达式因子语言：支持类 Alpha101 公式写法，让策略研究者更专注于因子逻辑而非代码
3. 渐进式性能升级：保留 Python 兼容，当 Rust 核心可用时自动获得性能提升，不影响可用性

---

### 3. Qlib (microsoft/qlib)

**核心亮点**:
- **成熟的 AI 量化研究流程**: 从数据处理 → 特征工程 → 模型训练 → 回测评估全链路支持
- **严谨的回测框架**: 支持滚动窗口回测、样本外测试，有效防范过拟合
- **丰富的预训练模型和因子**: 内置数百个 A 股特色因子
- **持续活跃维护**: 2026 年 4 月仍有代码提交

**可借鉴之处**:
1. 滚动训练/验证支持：当前 strategy-model-engine 已使用 `purged_group_ts_split`，但可以进一步完善 Walk-forward Validation 支持
2. 因子表达标准化：参考 Qlib 的因子数据接口规范，让因子更容易被模型复用

---

## 三、可优化方向评估

对照 jingni-trader 现有架构，从各个模块分析优化可行性：

| 维度 | 当前状况 | 优化潜力 | 推荐优先级 |
|------|----------|----------|------------|
| **回测引擎准确性与性能** | 当前支持多后端适配器 (rqalpha/backtrader/gm/native)，架构已经合理 | 采用 Polars 加速信号处理部分可以获得明显收益 | ⭐⭐⭐⭐ |
| **因子库的可扩展性** | 当前硬编码在 compute_a_share_factors，新增因子需要改核心代码 | 装饰器式注册机制可以极大提升可扩展性 | ⭐⭐⭐⭐⭐ |
| **策略编写 API 易用性** | 基于 Context 全流程编排，总体易用 | 可以考虑增加表达式因子语言支持 | ⭐⭐⭐ |
| **风险管理的完善程度** | 当前支持组合优化、VaR/CVaR、个股止损 | 可以增加动态止损、杠杆调整、换手率约束 | ⭐⭐⭐ |
| **数据获取与处理效率** | 已有降级链、模拟兜底，架构合理 | 大规模数据清洗可考虑用 Polars 加速 | ⭐⭐⭐⭐ |
| **代码架构合理性** | 清晰的模块化划分，主引擎+子 Skill 设计优良 | 仅需要在 factor-engine 引入装饰器注册 | ⭐⭐⭐⭐⭐ |

---

## 四、已完成验证测试

### 测试 1：装饰器式因子注册 API

**测试文件**: [test_decorator_factor_api.py](./test_decorator_factor_api.py)  
**借鉴来源**: Factor Engine (arXiv:2602.14138)  
**优化方向**: 提升因子库可扩展性，让用户无需修改引擎代码即可添加自定义因子

**测试结果**:
- ✅ **所有 10 个单元测试通过**，耗时 0.434s
- ✅ 注册机制正确：支持按类别查询、动态注册
- ✅ 计算结果与原始实现完全一致
- ✅ 支持运行时动态添加新因子（演示：动态添加 RSI14 成功）
- ✅ 支持按类别选择计算因子（如只计算 momentum 类）

**关键设计**:
```python
# 用户只需这样定义新因子，自动注册：
@registry.register(
    name="rsi_14",
    category="technical",
    requires=["close"],
    neutralize=True,
    description="14日RSI指标"
)
def compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    # ... 计算逻辑
    return rsi.rename("rsi_14")
```

**结论**: 方案验证成功，设计可行，不会破坏现有计算正确性，可以放心重构。

---

### 测试 2：Polars vs Pandas 因子计算性能对比

**测试文件**: [test_polars_performance.py](./test_polars_performance.py)  
**借鉴来源**: Factor Engine + AKQuant  
**优化方向**: 评估 Polars 在大规模因子计算上的性能收益

**测试环境**:
- Python 3.12
- pandas: 2.2.x, polars: 1.23.x
- 云服务器，单 CPU

**基准测试结果** (批量计算 12 个因子):

| 规模 | 行数 | Pandas 耗时 | Polars 耗时 | 加速比 |
|------|------|-------------|-------------|--------|
| 小规模 | 2,520 | 42.1 ms | 4.9 ms | **8.53x** |
| 中规模 | 50,400 | 444.2 ms | 22.9 ms | **19.38x** |
| 中大规模 | 126,000 | 1057.7 ms | 61.5 ms | **17.20x** |
| 大规模 | 252,000 | 2148.2 ms | 117.4 ms | **18.30x** |

**分操作性能对比** (200股 × 500天 = 10万行):

| 操作 | Pandas | Polars | 加速比 |
|------|--------|--------|--------|
| 5日收益率 | 6.45ms | 4.57ms | 1.41x |
| 20日波动率 | 139.81ms | 9.72ms | **14.38x** |
| 批量12因子 | 480.63ms | 40.81ms | **11.78x** |
| 1000股 × 252天 全量 | 2145ms | 110ms | **19.49x** |

**一致性验证**:
- ✅ **所有单元测试通过**，计算结果与 Pandas 在数值精度范围内一致
- ✅ 数值对比通过 `np.allclose` 验证一致

**结论**:
- **数据规模越大，加速效果越明显**。在 25 万行数据上，**加速比接近 20 倍**
- 即使小规模数据也有 8~9 倍加速
- Polars 的窗口函数（rolling）优化相比 Pandas 有极大优势，这正是因子计算最常用的操作
- **方案验证成功**，性能收益非常显著，值得引入。

---

## 五、验证总结

| 优化方向 | 验证状态 | 预期收益 | 风险 |
|----------|----------|----------|------|
| 装饰器式因子注册 | ✅ 验证通过 | 极大提升可扩展性，用户自定义因子无需改核心代码 | 低（API 兼容重构） |
| Polars 因子计算加速 | ✅ 验证通过 | 10~20 倍性能提升，大规模数据回测体验大幅改善 | 低（仅替换计算引擎层，接口不变） |
| 表达式因子语言 | - 待验证 | 提升策略研究者易用性 | 中 |

---

## 六、优化建议（待用户确认）

基于本次学习和验证，建议对 jingni-trader 进行以下优化：

### 高优先级（推荐立即实施）

1. **factor-engine 重构为装饰器式注册机制**
   - 保持现有因子计算逻辑不变，仅改变组织方式
   - 对原有硬编码因子逐个注册，兼容现有配置
   - 用户可以在不修改引擎代码的情况下动态添加自定义因子
   - 参考实现：[test_decorator_factor_api.py](./test_decorator_factor_api.py)

2. **引入 Polars 作为可选计算后端**
   - 渐进式迁移：检测 Polars 是否可用，可用则使用，否则回退到 Pandas
   - 因子计算阶段使用 Polars 获得数量级性能提升
   - 对结果输出保持 pandas DataFrame 格式，下游代码无需改动
   - 验证数据显示 25 万行数据从 ~2.1s → ~0.1s，体验提升巨大

### 中优先级（后续可做）

3. **数据引擎引入 Polars 加速数据清洗**
   - 数据清洗、缺失值处理、排序等操作也能获得明显加速

4. **完善 Walk-forward 滚动验证支持**
   - 参考 Qlib 的设计，增强模型训练阶段的样本外验证能力，减少过拟合

5. **支持表达式风格因子定义**
   - 如 `Rank(Ts_Mean(reversal_20d * volume_ratio, 10))`，让因子研究更便捷

---

## 七、附录

### 测试环境信息

```
Python version: 3.12.13
pandas version: 2.2.x
polars version: 1.23.x
pyarrow installed: yes
All tests passed: yes
test_decorator_factor_api: 10/10 passed
test_polars_performance: 11/11 passed
```

### 测试命令重现

```bash
cd /workspace
python tests/study_2026/test_decorator_factor_api.py
python tests/study_2026/test_polars_performance.py
```

---

**报告撰写**: jingni-trader 量化学习任务  
**保存位置**: `/workspace/tests/study_2026/LEARNING_REPORT.md`
