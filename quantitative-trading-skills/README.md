
# 量化交易Skill套件

这是一个完整的量化交易Skill套件，专为A股市场优化，基于专业量化交易机构的系统架构设计。

## 项目结构

```
quantitative-trading-skills/
├── skills/                              # Skill集合
│   ├── a-share-data-engine/            # A股数据引擎
│   ├── a-share-factor-engine/          # 因子构建引擎
│   ├── strategy-model-engine/          # 策略模型引擎
│   ├── backtest-engine/                # 回测引擎
│   ├── portfolio-risk-engine/          # 组合与风险管理
│   ├── execution-monitor-engine/       # 执行监控引擎
│   ├── reports-engine/                 # 报告引擎
│   └── quant-trading-master/           # 主Skill（协调中枢）
├── examples/                           # 示例脚本
├── scripts/                            # 自动化脚本
│   ├── install.sh                      # OpenClaw 安装脚本
│   ├── import_skills.sh                # Skill 导入脚本
│   ├── setup_env.sh                    # 环境变量配置向导
│   └── check_dependencies.py           # 依赖检查与安装工具
├── requirements.txt                    # Python 依赖列表
├── QuickStart.md                       # 新手5分钟入门指南
├── Example_Workflow.md                 # 完整量化工作流程示例
├── LICENSE_NOTICE.md                   # 许可与合规提示
└── README.md                           # 项目说明
```

## 包含的 Skill

| Skill 名称 | 功能描述 |
|-----------|---------|
| a-share-data-engine | A股数据获取、清洗和存储 |
| a-share-factor-engine | 因子构建、因子分析和相关性分析 |
| strategy-model-engine | 策略建模、机器学习模型训练和优化 |
| backtest-engine | 策略回测、绩效分析和可视化 |
| portfolio-risk-engine | 投资组合优化、风险分析和管理 |
| execution-monitor-engine | 交易执行、模拟交易和风险监控 |
| reports-engine | 绩效报告、分析图表和归因分析 |
| quant-trading-master | 主Skill，协调各子Skill完成完整量化流程 |

## 快速开始

新用户请先阅读：
- [QuickStart.md](QuickStart.md) - 5分钟快速入门指南
- [Example_Workflow.md](Example_Workflow.md) - 完整量化工作流程示例

### 1. 环境准备

确保你的系统已安装：
- Python 3.9 或更高版本
- Git (可选)

### 2. 一键安装流程

```bash
# 1. 进入项目目录
cd quantitative-trading-skills

# 2. 安装 OpenClaw (如果还没有安装)
chmod +x scripts/install.sh
./scripts/install.sh

# 3. 检查并安装 Python 依赖
python scripts/check_dependencies.py --install

# 4. 配置环境变量
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
# 编辑生成的 .env 文件，填入你的 Token

# 5. 导入 Skills 到 OpenClaw
chmod +x scripts/import_skills.sh
./scripts/import_skills.sh
```

### 3. 验证安装

```bash
# 检查环境变量配置
python scripts/check_dependencies.py --check-env

# 查看已安装的 Skills
openclaw skills list
```

## 主Skill功能

quant-trading-master是整个系统的协调中枢，负责：

1. **解析用户意图** - 判断当前处于量化投研的哪个阶段
2. **状态机管理** - 管理7个阶段的完整工作流
3. **子Skill调度** - 按顺序调用各子Skill完成工作
4. **上下文管理** - 维护任务状态和数据流转
5. **里程碑检查** - 每个阶段完成后验证结果完整性
6. **断点续传** - 支持从检查点恢复执行

## 阶段状态机

```
数据获取 → 因子构建 → 模型训练 → 回测验证 → 组合优化 → 模拟／实盘 → 绩效报告
```

### 分支逻辑

- 回测失败 → 返回因子调优
- 模型过拟合 → 样本外再验证

## 使用示例

### 基础使用

```python
from skills.quant-trading-master.scripts.main_workflow import run, QuantTradingContext

# 创建上下文
ctx = QuantTradingContext(
    task_id="my_task_001",
    session_id="my_session_001",
    stock_pool=["000001.SZ", "000002.SZ", "600000.SH"],
    time_range={"start_date": "2020-01-01", "end_date": "2024-01-01"},
    config={"data_backend": "tushare", "backtest_backend": "rqalpha"}
)

# 运行主流程
result = run(ctx)
```

### 单独使用某个 Skill

每个 Skill 都可以单独使用，具体请参考各 Skill 目录下的 README.md。

## 技术栈

- **Python 3.9+**
- **数据处理**: pandas, numpy, pyarrow
- **数据源**: tushare, baostock, akshare
- **量化分析**: ta-lib, pandas-ta, scipy
- **机器学习**: scikit-learn, lightgbm, catboost, optuna, mlflow
- **回测**: backtrader, rqalpha, quantstats, empyrical
- **优化**: pypfopt, cvxpy
- **可视化**: matplotlib, plotly

## 环境变量

| 变量名 | 说明 | 是否必需 |
|-------|------|---------|
| `TUSHARE_TOKEN` | Tushare Pro API Token | 是 |
| `GM_TOKEN` | 掘金量化 API Token | 否 |
| `XTQUANT_TOKEN` | 迅投 QMT API Token | 否 |
| `DATA_DIR` | 数据存储目录（默认：./data） | 否 |
| `LOG_LEVEL` | 日志级别（DEBUG/INFO/WARNING/ERROR） | 否 |

## 脚本说明

### scripts/install.sh
检测并自动安装 OpenClaw，支持 Linux 和 macOS 系统。

### scripts/import_skills.sh
将项目中的所有 Skill 导入到 OpenClaw 的技能目录中。

### scripts/setup_env.sh
创建环境变量配置文件并提供配置向导。

### scripts/check_dependencies.py
检查 Python 版本和必需包，支持选择性安装可选库：
```bash
# 仅检查依赖
python scripts/check_dependencies.py

# 安装所有必需包
python scripts/check_dependencies.py --install

# 安装特定可选包
python scripts/check_dependencies.py --optional backtrader rqalpha

# 安装所有包（必需 + 可选）
python scripts/check_dependencies.py --all

# 检查环境变量配置
python scripts/check_dependencies.py --check-env
```

## 许可与合规

请务必阅读 [LICENSE_NOTICE.md](LICENSE_NOTICE.md) 了解：
- 项目使用许可
- 第三方库的许可和使用限制
- 风险提示和免责声明

## 注意事项

1. 量化交易存在高风险，请谨慎使用
2. 请先在模拟环境充分验证策略
3. 本项目仅供学习和研究使用
4. 请确保遵守当地相关法律法规

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过 GitHub Issues 联系我们。
