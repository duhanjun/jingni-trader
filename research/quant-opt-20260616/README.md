# 量化交易开源项目学习与 jingni-trader 优化建议（2026-06-16）

> 本报告由 `feat/quant-opt-20260616` 分支自动生成。
> 自动化任务执行者：MiniMax-M3  ·  执行日期：2026-06-16

## 一、调研方法

通过联网搜索（GitHub、arXiv、QuantConnect、JoinQuant、BigQuant 等），
结合 GitHub 仓库与官方文档的精读，挑选 3 个最有借鉴价值的开源项目深入研究：

| 项目 | 关注点 | 文档与代码参考 |
|------|--------|----------------|
| **Microsoft Qlib** | AI-Quant 平台、Alpha158/Alpha360 数据集、Expression Engine、Rolling 训练、TopK 策略 | github.com/microsoft/qlib |
| **AKQuant** | Rust+Python 混合、三轴语义、Polars 因子引擎、Walk-Forward 校验 | akquant.akfamily.xyz |
| **VeighNa Alpha Lab** | vnpy 内部的 alpha 研究框架、参考 Qlib 实现 | github.com/vnpy/vnpy |

此外，参考学习了 VectorBT、QSTrader、Zipline、Backtesting.py 等的轻量化设计。

## 二、核心亮点与可借鉴之处

### 1. Microsoft Qlib
- **Alpha158 / Alpha360 因子数据集**：把全部 alpha 因子统一成可复现的数据集，新人不再各自写因子。
- **Expression Engine（DSL）**：用类似 `Rank(Mean($close, 5))` 的字符串公式描述因子，
  编译为 AST 解释执行，免去重复写 Python 算子的成本。
- **Data Handler / Data Loader / Data Processor 分离**：数据拉取、转换、缓存三层清晰，
  适合多数据源并存的 A 股场景。
- **TopkDropoutStrategy**：相比纯 TopK，强制每期换出 n_dropout 只股票再补入 n_dropout 只榜外股票，
  避免组合过度集中且保持调仓纪律。
- **Estimator / Qrun 配置文件**：把所有超参集中到 YAML，qrun 一行跑完整个实验。

### 2. AKQuant
- **三轴语义（time / calendar / data feed）**：明确把时钟、交易日历、数据源分开建模，
  避免不同市场混用。
- **Polars 因子引擎**：基于表达式 + Polars 算子，性能比 pandas 高 5-10 倍。
- **Walk-Forward 子模块**：直接内置在 backtest 中，不是事后外挂。
- **RiskConfig / StrategyConfig 层级合并**：默认配置 + 用户覆写，避免长字典传参。

### 3. VeighNa Alpha Lab
- **集成 Qlib 的 Alpha158**：让 vnpy 用户零成本用上 Qlib 因子。
- **策略模板 + 信号类分层**：策略只负责把信号转成下单，组合管理交给独立模块。

## 三、jingni-trader 现状的改进空间

| 维度 | 现状 | 借鉴来源 | 改进建议 |
|------|------|----------|----------|
| 因子库可扩展性 | `factor-engine` 是 hardcoded 因子，新因子必须改引擎代码 | Qlib Expression Engine / AKQuant DSL | ✅ 已实现 DSL |
| 因子可复现性 | 各因子分散实现，无法一行写出复合 alpha | Qlib Alpha158 风格 | ✅ 已通过 DSL 表达复合公式 |
| 策略组合方式 | `portfolio-risk-engine` 仅做权重优化，缺 TopK 类组合策略 | Qlib TopkDropout | ✅ 已实现 TopKDropout |
| 模型验证方式 | `strategy-model-engine` 缺标准 walk-forward 框架 | Qlib RollingGen / AKQuant walk_forward | ✅ 已实现 RollingSplit + Runner |
| 回测-训练-验证一致性 | 各自独立模块，没有共享切片语义 | Qlib `qrun` 配置驱动 | 建议增加 `qrun` 式 YAML |
| 数据缓存粒度 | 单一 cache 目录 | Qlib 多层 cache | 建议按 (provider, freq) 分桶 |
| 实盘接口 | execution-monitor-engine 已有雏形 | AKQuant OrderGateway | 待细化 |

## 四、本次实施的三项优化（已验证）

详见 [`OPTIMIZATION_REPORT.md`](./OPTIMIZATION_REPORT.md)。

| # | 优化点 | 借鉴来源 | 验证结果 |
|---|--------|----------|----------|
| 1 | **因子表达式引擎 (DSL)** | Qlib `qlib.data.ops` + AKQuant `FactorEngine` | 11 单测 + 8 集成测全过 |
| 2 | **Top-K Dropout 策略** | Qlib `qlib.contrib.strategy.TopkDropoutStrategy` | 10 单测全过 |
| 3 | **Walk-Forward 滚动验证** | Qlib `qlib.contrib.rolling` + AKQuant `walk_forward` | 9 单测全过 |

总计：**38 个测试用例全部通过**。

## 五、待用户确认的后续优化建议

以下方向在本次任务中**未实施**，需要用户确认后再继续：

1. **集成 DSL 到 `factor-engine`**：在 `skills/factor-engine` 中新增 `expressions.py`，
   复用本次实现的 DSL，让 12 个 hardcoded 因子可被 1 行公式表达。
2. **Top-K Dropout 接入 `portfolio-risk-engine`**：把 `TopKDropoutStrategy` 接到组合优化器，
   并在 Context 中增加 `top_k` / `n_dropout` 字段。
3. **Walk-Forward 接入 `strategy-model-engine` / `backtest-engine`**：
   抽取 `RollingSplit` 为可独立 import 的工具，供两引擎共用。
4. **多数据源抽象**：参考 Qlib Data Handler / Loader / Processor 三层抽象，
   拆分 `data-engine` 内部模块。
5. **qrun 式 YAML 驱动**：用一个 yaml 描述一次完整研究流程（数据 → 因子 → 训练 → 回测 → 报告）。

## 六、仓库与分支信息

- **分支**：`feat/quant-opt-20260616`（基于 main）
- **改动范围**：仅在 `research/quant-opt-20260616/` 目录下新增代码，
  **未修改** main 分支上任何已有文件。
- **是否合并**：❌ 等待用户确认（按用户要求禁止自动 merge）

## 七、复现方式

```bash
# 切到新分支
git fetch origin
git checkout feat/quant-opt-20260616

# 安装依赖
pip install numpy pandas scipy pytest

# 跑全部 38 个测试
python3 -m pytest research/quant-opt-20260616/ -v

# 跑单个模块
python3 -m pytest research/quant-opt-20260616/factor_expression_engine/ -v
python3 -m pytest research/quant-opt-20260616/topk_dropout_strategy/ -v
python3 -m pytest research/quant-opt-20260616/walk_forward_validation/ -v
python3 -m pytest research/quant-opt-20260616/tests/ -v
```
