# PRD：绩效归因报告（复盘环节）

> 文档版本：v1.1
> 创建日期：2026-08-03
> 关联方案：绩效归因报告（复盘）解决方案
> 状态：v1.0 开发完成

---

## 第 1 章 概述

### 1.1 背景

系统已实现"研究→决策→执行→复盘"闭环中的前三个环节：
- **研究**：量化策略路径（DATA → FACTOR → MODEL → BACKTEST）和主观个股分析路径（DATA → FACTOR → REPORT）
- **决策**：策略回测报告 + 股票分析报告
- **执行**：模拟盘/实盘交易（execution-monitor-engine），已生成 `ledger.jsonl`、`trade_log.json` 等产物

**复盘环节（绩效归因报告）完全缺失**。用户完成实盘/模拟交易后，无法回答"赚/亏的钱到底来自哪里？"这一核心问题。

当前系统已有可复用的基础能力：
- `ReportGenerator.calc_brinson_attribution()`：Brinson 归因分解（配置效应/选择效应/交互效应）
- `ReportGenerator.make_style_exposure_chart()`：风格因子暴露条形图
- `ReportGenerator.make_industry_attribution_chart()`：行业利润贡献图
- `execution-monitor-engine/scripts/paper_ledger.py`：`PaperTradeRecordV1` 追加式交易账本（含 execution_id / trade_date / code / side / shares / price / commission / stamp_tax / slippage_cost / nav_after 等字段）
- `reports-engine/engine.py` 的 `run()` 函数已有两分支路由（回测报告 / 模板报告），需新增第三分支

### 1.2 目标

在"Y型分叉+尾部合并"架构的复盘环节，实现第三类报告——**绩效归因报告**，回答"实盘盈亏来自哪里？"这一核心问题。通过意图识别触发，自动读取 EXECUTION 阶段产物，生成包含交易统计、盈亏归因、执行质量、风格暴露、压力期表现等内容的完整 HTML 报告。

### 1.3 范围

**本 PRD 范围**：
- 意图解析扩展：新增"绩效复盘"意图识别与路由
- 归因分析器：实现 `AttributionAnalyzer`（成交记录解析、round-trip 归组、盈亏归因、执行质量统计）
- 报告模板：新增 `attribution_report.py` 模板及 HTML 生成
- 报告路由：`reports-engine/engine.py` 中新增第三分支（优先级最高）
- LLM 深度解读：复用现有 LLM 注入机制生成绩效归因文字解读

**不在范围**：
- 因子引擎、回测引擎、执行引擎的修改
- 前端 UI 或 Web 界面开发
- 数据源或数据采集层的变更

---

## 第 2 章 需求清单

### 2.1 功能需求

| 需求 ID | 需求描述 | 优先级 |
|---------|---------|--------|
| AR-001 | 新增 `ATTRIBUTION_KEYWORDS` 关键词集合（绩效归因/归因分析/复盘/实盘报告/盈亏分析/执行报告/交易复盘/绩效复盘/attribution） | P0 |
| AR-002 | `MasterEngine.parse_intent()` 新增 `_is_attribution_intent()` 方法，识别绩效复盘意图 | P0 |
| AR-003 | 绩效复盘意图触发时，路由到 `["DATA", "FACTOR", "EXECUTION", "REPORT"]` 阶段路径 | P0 |
| AR-004 | 实现 `AttributionAnalyzer` 类，从 `ledger.jsonl` 解析成交记录并重建账户状态 | P0 |
| AR-005 | 实现 round-trip 分析法：将零散成交记录归组为完整买卖闭环（买入→卖出），计算每笔闭环的盈亏/收益率/持仓天数 | P0 |
| AR-006 | 实现执行质量分析：滑点成本、佣金/印花税占比、成交价格偏离度 | P0 |
| AR-007 | 实现交易统计概览：总成交笔数、胜率、盈亏比、平均持仓天数、最大连胜/连败 | P0 |
| AR-008 | 实现按标的归因：每只股票的总盈亏、收益率、交易次数、胜率 | P0 |
| AR-009 | 新增 `attribution_report.py` 模板，复用 `ReportGenerator` 现有图表能力 | P0 |
| AR-010 | `reports-engine/engine.py` 的 `run()` 新增第三分支：`report_intent == "attribution"` 时路由到 `_run_attribution_report()` | P0 |
| AR-011 | 报告含 A 股特定压力期表现（2015 股灾、2016 熔断、2020 疫情、2024 年初），若交易记录覆盖对应时间段则自动标注 | P1 |
| AR-012 | 复用现有 LLM 注入机制，生成绩效归因文字解读（含 `<!--LLM_ATTRIBUTION_PLACEHOLDER-->` 占位符） | P1 |
| AR-013 | `_detect_report_template()` 扩展：识别 `report_intent == "attribution"` 时返回 `"attribution"` | P0 |
| AR-014 | 集成到 `scripts/archive.py` 自动归档归因报告 | P0 |

### 2.2 非功能需求

| 需求 ID | 需求描述 |
|---------|---------|
| AR-NFR-001 | 归因报告生成耗时 ≤ 5 秒（1000 条成交记录以内） |
| AR-NFR-002 | 无 EXECUTION 产物时，返回友好错误提示而非崩溃 |
| AR-NFR-003 | 现有回测报告和模板报告路径不受影响，路由优先级正确（attribution > backtest > template） |
| AR-NFR-004 | 489 条现有回归测试 100% 通过 |
| AR-NFR-005 | `ledger.jsonl` 损坏行自动跳过（复用 `read_paper_trades` 的 skip 逻辑） |
| AR-NFR-006 | 报告为独立 HTML 文件，所有图表内嵌（plotly CDN），无需外部依赖 |

---

## 第 3 章 技术决策

### 3.1 意图解析扩展

#### 3.1.1 关键词集合

在 `engine.py` 的 `MasterEngine` 类中新增：

```python
ATTRIBUTION_KEYWORDS = {
    "绩效归因", "归因分析", "复盘", "实盘报告", "盈亏分析",
    "执行报告", "交易复盘", "绩效复盘", "attribution",
}
```

#### 3.1.2 意图识别方法

```python
def _is_attribution_intent(self, user_input: str) -> bool:
    """判断用户是否触发绩效复盘意图"""
    if not user_input:
        return False
    return any(kw in user_input for kw in ATTRIBUTION_KEYWORDS)
```

#### 3.1.3 parse_intent() 修改

在 `parse_intent()` 中，`strategy_required` 判断之后插入：

```python
# 绩效复盘意图：最高优先级，独立路由
if self._is_attribution_intent(user_input):
    ctx.metadata["report_intent"] = "attribution"
    target_stages = ["DATA", "FACTOR", "EXECUTION", "REPORT"]
    logger.info(f"检测到绩效复盘意图，路由到复盘路径: {' → '.join(target_stages)}")
```

路由优先级：**绩效复盘 > 量化策略 > 个股分析**。

### 3.2 报告路由逻辑

`reports-engine/engine.py` 的 `run()` 函数修改为三分支：

```python
def run(ctx) -> Dict[str, Any]:
    # 优先级 1: 绩效复盘意图 → 绩效归因报告
    meta = getattr(ctx, 'metadata', {}) or {}
    if meta.get("report_intent") == "attribution":
        return _run_attribution_report(ctx)

    # 优先级 2: 有 BACKTEST 产物 → 回测绩效报告
    backtest_path = ctx.get_artifact("BACKTEST") if hasattr(ctx, 'get_artifact') else None
    has_backtest = backtest_path and os.path.exists(backtest_path)
    if has_backtest:
        # ... 现有回测报告逻辑 ...

    # 优先级 3: 默认 → 模板化个股分析报告
    return _run_template_report(ctx)
```

### 3.3 归因分析器

#### 3.3.1 新增文件

`skills/reports-engine/scripts/attribution_analyzer.py`

```python
"""
绩效归因分析器

从 execution-monitor-engine 的 ledger.jsonl / trade_log.json 产物中
提取成交记录，执行 round-trip 归组、盈亏归因、执行质量分析。
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date

import pandas as pd
import numpy as np

logger = logging.getLogger("attribution-analyzer")


class RoundTrip:
    """一次完整的买卖闭环"""
    code: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    shares: int
    gross_pnl: float          # 毛盈亏（不含费用）
    commission: float          # 总佣金
    stamp_tax: float           # 总印花税
    slippage_cost: float       # 总滑点成本
    net_pnl: float             # 净盈亏
    return_pct: float          # 收益率
    holding_days: int          # 持仓天数
    is_win: bool               # 是否盈利


class AttributionAnalyzer:
    """绩效归因分析器"""

    def __init__(self, ledger_path: str, trade_log_path: Optional[str] = None,
                 init_capital: float = 1_000_000.0):
        self.ledger_path = Path(ledger_path)
        self.trade_log_path = Path(trade_log_path) if trade_log_path else None
        self.init_capital = init_capital
        self.records: List[dict] = []
        self.round_trips: List[RoundTrip] = []
        self._nav_series: Optional[pd.Series] = None

    def load(self) -> bool:
        """加载 ledger.jsonl，解析为 records 列表"""
        if not self.ledger_path.exists():
            logger.warning(f"ledger 文件不存在: {self.ledger_path}")
            return False

        records = []
        with self.ledger_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                except json.JSONDecodeError:
                    logger.warning(f"ledger 损坏行已跳过")
                    continue

        self.records = sorted(records, key=lambda r: r.get("trade_date", ""))
        return len(self.records) > 0

    def build_round_trips(self) -> List[RoundTrip]:
        """将零散成交记录归组为完整买卖闭环"""
        # 按标的分组，按时间排序，FIFO 匹配买卖
        from collections import defaultdict, deque

        by_code: Dict[str, list] = defaultdict(list)
        for rec in self.records:
            if rec.get("confirmed", True):
                by_code[rec["code"]].append(rec)

        round_trips = []
        for code, trades in by_code.items():
            trades.sort(key=lambda t: t["trade_date"])
            buy_queue = deque()  # (buy_record, remaining_shares)

            for trade in trades:
                if trade["side"] == "buy":
                    buy_queue.append((trade, trade["shares"]))
                else:
                    # 卖出：按 FIFO 匹配买入
                    sell_shares = trade["shares"]
                    while sell_shares > 0 and buy_queue:
                        buy_rec, buy_shares = buy_queue[0]
                        matched = min(buy_shares, sell_shares)

                        # 计算这笔闭环的盈亏
                        gross_pnl = (trade["price"] - buy_rec["price"]) * matched
                        # 费用按比例分摊
                        ratio = matched / buy_rec["shares"]
                        buy_comm = buy_rec.get("commission", 0) * ratio
                        buy_slip = buy_rec.get("slippage_cost", 0) * ratio
                        sell_ratio = matched / trade["shares"]
                        sell_comm = trade.get("commission", 0) * sell_ratio
                        sell_stamp = trade.get("stamp_tax", 0) * sell_ratio
                        sell_slip = trade.get("slippage_cost", 0) * sell_ratio

                        total_comm = buy_comm + sell_comm
                        total_stamp = sell_stamp
                        total_slip = buy_slip + sell_slip
                        net_pnl = gross_pnl - total_comm - total_stamp - total_slip

                        buy_date = buy_rec["trade_date"]
                        sell_date = trade["trade_date"]
                        holding_days = (
                            datetime.strptime(sell_date, "%Y-%m-%d") -
                            datetime.strptime(buy_date, "%Y-%m-%d")
                        ).days

                        rt = RoundTrip()
                        rt.code = code
                        rt.buy_date = buy_date
                        rt.sell_date = sell_date
                        rt.buy_price = buy_rec["price"]
                        rt.sell_price = trade["price"]
                        rt.shares = matched
                        rt.gross_pnl = gross_pnl
                        rt.commission = total_comm
                        rt.stamp_tax = total_stamp
                        rt.slippage_cost = total_slip
                        rt.net_pnl = net_pnl
                        rt.return_pct = net_pnl / (buy_rec["price"] * matched) if buy_rec["price"] > 0 else 0
                        rt.holding_days = max(holding_days, 1)
                        rt.is_win = net_pnl > 0

                        round_trips.append(rt)

                        # 更新队列
                        buy_shares -= matched
                        sell_shares -= matched
                        if buy_shares <= 0:
                            buy_queue.popleft()
                        else:
                            buy_queue[0] = (buy_rec, buy_shares)

        self.round_trips = round_trips
        return round_trips

    def get_transaction_stats(self) -> dict:
        """交易统计概览"""
        if not self.records:
            return {}

        buys = [r for r in self.records if r["side"] == "buy"]
        sells = [r for r in self.records if r["side"] == "sell"]
        total_commission = sum(r.get("commission", 0) for r in self.records)
        total_stamp_tax = sum(r.get("stamp_tax", 0) for r in self.records)
        total_slippage = sum(r.get("slippage_cost", 0) for r in self.records)

        return {
            "total_trades": len(self.records),
            "total_buys": len(buys),
            "total_sells": len(sells),
            "total_commission": round(total_commission, 2),
            "total_stamp_tax": round(total_stamp_tax, 2),
            "total_slippage": round(total_slippage, 2),
            "total_cost": round(total_commission + total_stamp_tax + total_slippage, 2),
            "unique_stocks": len(set(r["code"] for r in self.records)),
        }

    def get_round_trip_stats(self) -> dict:
        """Round-trip 统计"""
        if not self.round_trips:
            return {}

        net_pnls = [rt.net_pnl for rt in self.round_trips]
        returns = [rt.return_pct for rt in self.round_trips]
        holding_days = [rt.holding_days for rt in self.round_trips]
        wins = [rt for rt in self.round_trips if rt.is_win]
        losses = [rt for rt in self.round_trips if not rt.is_win]

        total_net_pnl = sum(net_pnls)
        avg_win = np.mean([w.net_pnl for w in wins]) if wins else 0
        avg_loss = np.mean([l.net_pnl for l in losses]) if losses else 0

        return {
            "total_round_trips": len(self.round_trips),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(self.round_trips) if self.round_trips else 0,
            "total_net_pnl": round(total_net_pnl, 2),
            "avg_return": round(np.mean(returns) * 100, 2) if returns else 0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": abs(sum(w.net_pnl for w in wins) / sum(l.net_pnl for l in losses)) if losses and sum(l.net_pnl for l in losses) != 0 else 0,
            "avg_holding_days": round(np.mean(holding_days), 1) if holding_days else 0,
        }

    def get_pnl_by_stock(self) -> pd.DataFrame:
        """按标的归因：每只股票的总盈亏"""
        if not self.round_trips:
            return pd.DataFrame()

        data = []
        for code in sorted(set(rt.code for rt in self.round_trips)):
            stock_rts = [rt for rt in self.round_trips if rt.code == code]
            total_pnl = sum(rt.net_pnl for rt in stock_rts)
            wins = sum(1 for rt in stock_rts if rt.is_win)
            data.append({
                "code": code,
                "total_pnl": round(total_pnl, 2),
                "trade_count": len(stock_rts),
                "win_count": wins,
                "win_rate": round(wins / len(stock_rts) * 100, 1) if stock_rts else 0,
                "avg_return": round(np.mean([rt.return_pct for rt in stock_rts]) * 100, 2),
            })

        df = pd.DataFrame(data)
        return df.sort_values("total_pnl", ascending=False)

    def get_execution_quality(self) -> dict:
        """执行质量分析"""
        if not self.records:
            return {}

        total_cost = sum(
            r.get("commission", 0) + r.get("stamp_tax", 0) + r.get("slippage_cost", 0)
            for r in self.records
        )
        total_turnover = sum(
            r["price"] * r["shares"] for r in self.records
        )

        buy_volumes = [r["price"] * r["shares"] for r in self.records if r["side"] == "buy"]
        sell_volumes = [r["price"] * r["shares"] for r in self.records if r["side"] == "sell"]

        return {
            "total_turnover": round(total_turnover, 2),
            "total_cost": round(total_cost, 2),
            "cost_ratio_bps": round(total_cost / total_turnover * 10000, 2) if total_turnover > 0 else 0,
            "avg_trade_size": round(np.mean(buy_volumes + sell_volumes), 2) if buy_volumes or sell_volumes else 0,
            "max_trade_size": round(max(buy_volumes + sell_volumes), 2) if buy_volumes or sell_volumes else 0,
            "slippage_ratio_bps": round(
                sum(r.get("slippage_cost", 0) for r in self.records) / total_turnover * 10000, 2
            ) if total_turnover > 0 else 0,
        }

    def get_nav_series(self) -> pd.Series:
        """从 ledger 提取每日净值序列"""
        if self._nav_series is not None:
            return self._nav_series

        if not self.records:
            return pd.Series(dtype=float)

        # 按日期取最后一条记录的 nav_after
        nav_by_date = {}
        for rec in self.records:
            date_str = rec.get("trade_date", "")
            nav = rec.get("nav_after")
            if date_str and nav is not None:
                nav_by_date[date_str] = nav

        if not nav_by_date:
            return pd.Series(dtype=float)

        df = pd.DataFrame(
            {"date": list(nav_by_date.keys()), "nav": list(nav_by_date.values())}
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")

        self._nav_series = df["nav"]
        return self._nav_series

    def get_stress_period_performance(self) -> dict:
        """A 股特定压力期表现（若交易记录覆盖对应时间段）"""
        stress_periods = {
            "2015年股灾": ("2015-06-12", "2015-08-26"),
            "2016年熔断": ("2016-01-04", "2016-01-28"),
            "2018年熊市": ("2018-01-24", "2018-10-18"),
            "2020年疫情": ("2020-01-20", "2020-03-23"),
            "2024年初下跌": ("2024-01-02", "2024-02-05"),
        }

        nav = self.get_nav_series()
        if nav.empty:
            return {}

        results = {}
        for name, (start, end) in stress_periods.items():
            period_nav = nav.loc[start:end]
            if len(period_nav) < 2:
                continue
            period_return = float(period_nav.iloc[-1] / period_nav.iloc[0] - 1)
            period_mdd = float((period_nav / period_nav.cummax() - 1).min())
            results[name] = {
                "return": round(period_return * 100, 2),
                "max_drawdown": round(period_mdd * 100, 2),
            }

        return results

    def get_consecutive_stats(self) -> dict:
        """最大连胜/连败"""
        if not self.round_trips:
            return {}

        max_win_streak = 0
        max_loss_streak = 0
        current_win = 0
        current_loss = 0

        for rt in self.round_trips:
            if rt.is_win:
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }
```

### 3.4 报告模板

#### 3.4.1 新增文件

`skills/reports-engine/scripts/templates/attribution_report.py`

```python
"""
绩效归因报告模板

生成完整的 HTML 绩效归因报告，包含：
1. 交易统计概览
2. Round-Trip 盈亏归因
3. 按标的盈亏明细
4. 净值曲线 + 回撤
5. 执行质量分析
6. A 股压力期表现
7. LLM 深度解读
"""
```

#### 3.4.2 报告内容结构

| 章节 | 内容 | 数据来源 | 图表类型 |
|------|------|---------|---------|
| 行情数据 | 净值曲线 + 回撤子图 | `get_nav_series()` | Plotly 双面板折线图 |
| 报告概览 | 核心指标卡片（总收益/胜率/盈亏比/最大回撤/夏普比/交易天数） | `get_round_trip_stats()` + `get_nav_series()` | HTML 指标卡片 |
| 交易统计 | 总成交笔数、买卖比、佣金/印花税/滑点汇总、涉及标的数 | `get_transaction_stats()` | HTML 表格 |
| Round-Trip 分析 | 闭环交易列表（标的/买入日/卖出日/持仓天数/收益率/净盈亏）、盈利/亏损分布 | `build_round_trips()` | Plotly 散点图 + HTML 表格 |
| 按标的归因 | 每只股票的总盈亏、交易次数、胜率、平均收益率 | `get_pnl_by_stock()` | Plotly 瀑布图 / 条形图 |
| 执行质量 | 总成交额、费用占比(bps)、滑点占比(bps)、平均/最大单笔规模 | `get_execution_quality()` | HTML 表格 |
| 压力期表现 | A 股 5 个特定压力期的收益与回撤 | `get_stress_period_performance()` | HTML 表格 |
| 深度解读 | LLM 生成的绩效归因文字分析 | LLM 注入 | 富文本 HTML |
| 风险提示 | 风险提示与免责声明 | 静态文本 | HTML |

#### 3.4.3 复用现有图表能力

直接复用 `ReportGenerator` 的方法：
- `make_equity_chart()`：净值曲线 + 回撤（传入从 ledger 重建的净值序列）
- `make_monthly_heatmap()`：月度收益热力图
- `calc_performance_metrics()`：全面绩效指标计算

### 3.5 reports-engine 路由集成

#### 3.5.1 `_run_attribution_report()` 函数

```python
def _run_attribution_report(ctx) -> Dict[str, Any]:
    """
    绩效归因报告生成路径

    流程：
    1. 读取 EXECUTION 产物（ledger.jsonl / trade_log.json）
    2. AttributionAnalyzer 解析和归因
    3. ReportGenerator 生成图表
    4. 渲染 HTML 报告（含 LLM 占位符）
    5. 调用 LLM 生成深度解读（可选）
    6. 替换占位符 → 输出最终报告
    """
    import os
    from scripts.attribution_analyzer import AttributionAnalyzer
    from datetime import datetime

    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. 定位 EXECUTION 产物
    execution_artifact = ctx.get_artifact("EXECUTION")
    if not execution_artifact:
        return {
            "success": False, "artifact_path": "", "metadata": {},
            "error": "未找到 EXECUTION 产物，无法生成绩效归因报告。请先执行模拟/实盘交易。"
        }

    # ledger.jsonl 位于 execution 目录下
    execution_dir = os.path.dirname(execution_artifact)
    ledger_path = os.path.join(execution_dir, "ledger.jsonl")
    trade_log_path = os.path.join(execution_dir, "trade_log.json")

    if not os.path.exists(ledger_path):
        return {
            "success": False, "artifact_path": "", "metadata": {},
            "error": f"ledger 文件不存在: {ledger_path}"
        }

    # 2. 初始化分析器
    analyzer = AttributionAnalyzer(ledger_path, trade_log_path)
    if not analyzer.load():
        return {
            "success": False, "artifact_path": "", "metadata": {},
            "error": "ledger 文件为空或无法解析"
        }

    analyzer.build_round_trips()

    # 3. 提取分析数据
    tx_stats = analyzer.get_transaction_stats()
    rt_stats = analyzer.get_round_trip_stats()
    pnl_by_stock = analyzer.get_pnl_by_stock()
    exec_quality = analyzer.get_execution_quality()
    stress_perf = analyzer.get_stress_period_performance()
    consecutive = analyzer.get_consecutive_stats()

    # 4. 生成图表
    generator = ReportGenerator(title="绩效归因报告")
    charts = []

    # 净值曲线
    nav_series = analyzer.get_nav_series()
    if not nav_series.empty:
        equity_curve = pd.DataFrame({
            "date": nav_series.index,
            "equity": nav_series.values
        })
        equity_chart = generator.make_equity_chart(equity_curve)
        if equity_chart:
            charts.append(equity_chart)

        # 月度热力图
        if INCLUDE_HEATMAP:
            heatmap = generator.make_monthly_heatmap(equity_curve)
            if heatmap:
                charts.append(heatmap)

        # 计算绩效指标
        metrics = generator.calc_performance_metrics(equity_curve)
    else:
        metrics = {}

    # 按标的盈亏图
    if not pnl_by_stock.empty:
        pnl_chart = _make_pnl_by_stock_chart(pnl_by_stock)
        if pnl_chart:
            charts.append(pnl_chart)

    # Round-trip 散点图
    if analyzer.round_trips:
        rt_chart = _make_round_trip_scatter(analyzer.round_trips)
        if rt_chart:
            charts.append(rt_chart)

    # 5. 构建 HTML
    html = _build_attribution_html(
        metrics=metrics,
        tx_stats=tx_stats,
        rt_stats=rt_stats,
        pnl_by_stock=pnl_by_stock,
        exec_quality=exec_quality,
        stress_perf=stress_perf,
        consecutive=consecutive,
        charts=charts,
        round_trips=analyzer.round_trips,
    )

    # 6. 写入文件
    html_path = os.path.join(REPORT_DIR, "attribution_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"绩效归因报告已生成: {html_path}")

    # 7. 准备 LLM prompt（供 agent 调用）
    llm_prompts = {
        "attribution": {
            "user_prompt": _build_attribution_llm_prompt(rt_stats, pnl_by_stock, exec_quality, stress_perf),
        }
    }

    # 8. 尝试调用 LLM（若已配置）
    llm_status = "skipped"
    llm_responses = {}
    try:
        from scripts.llm_client import generate_analysis, is_available
        if is_available():
            logger.info("开始调用 LLM 生成绩效归因解读...")
            resp = generate_analysis(llm_prompts["attribution"])
            if resp:
                llm_responses["attribution"] = resp
                llm_status = "success"
            else:
                llm_status = "failed"
    except Exception as e:
        logger.warning(f"LLM 调用异常: {e}")
        llm_status = "failed"

    # 9. 替换占位符
    _inject_attribution_analysis(html_path, llm_responses, llm_prompts)

    # 10. 落盘 report_data.json
    report_data = {
        "report_type": "attribution",
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics,
        "tx_stats": tx_stats,
        "rt_stats": rt_stats,
        "llm_status": llm_status,
    }
    data_path_out = os.path.join(REPORT_DIR, "report_data.json")
    with open(data_path_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

    return {
        "success": True,
        "artifact_path": html_path,
        "metadata": {
            "report_type": "attribution",
            "llm_prompts": llm_prompts,
            "llm_status": llm_status,
            "metrics": metrics,
            "report_data_path": data_path_out,
        },
        "error": ""
    }
```

### 3.6 报告产物结构

```
workspace/reports/
├── attribution_report.html     # 绩效归因报告（独立 HTML）
├── report_data.json            # 报告元数据
└── ...                         # 其他报告（technical_report.html, fundamental_report.html, report.html）
```

### 3.7 数据依赖

| 数据项 | 来源文件 | 必需 | 说明 |
|--------|---------|------|------|
| 成交记录 | `ledger.jsonl` (EXECUTION 产物) | 是 | 每行一条 `PaperTradeRecordV1` JSON |
| 交易日志 | `trade_log.json` (EXECUTION 产物) | 否 | 补充订单状态信息 |
| 账户状态 | replay_ledger 重建 | 自动 | 从 ledger 顺序重放重建 |
| 净值序列 | 从 ledger 中 `nav_after` 提取 | 自动 | 按日期取最后一条记录 |

---

## 第 4 章 验收标准

### 4.1 可验证验收标准

| CR ID | 验收标准 | 验证方法 |
|-------|---------|---------|
| CR-1 | 输入含"复盘"关键词时，`parse_intent()` 正确识别并路由到 `["DATA", "FACTOR", "EXECUTION", "REPORT"]` | 单元测试 |
| CR-2 | 有 ledger.jsonl 时，`_run_attribution_report()` 生成 `attribution_report.html` | 文件存在性检查 |
| CR-3 | 无 EXECUTION 产物时，返回友好错误提示（不崩溃） | 边界测试 |
| CR-4 | Round-trip 分析正确归组：买入→卖出形成闭环，FIFO 匹配 | 单元测试（构造已知买卖序列，验证归组结果） |
| CR-5 | 报告 HTML 包含 7 个完整章节（净值曲线/报告概览/交易统计/Round-Trip/按标的归因/执行质量/压力期表现） | HTML 内容检查 |
| CR-6 | 现有回测报告路径不受影响：有 BACKTEST 产物时仍生成回测报告 | 回归测试 |
| CR-7 | 现有模板报告路径不受影响：无 BACKTEST 且无 attribution 意图时仍生成模板报告 | 回归测试 |
| CR-8 | 489 条现有回归测试 100% 通过 | pytest 全量回归 |
| CR-9 | `ledger.jsonl` 损坏行自动跳过不中断 | 损坏数据注入测试 |
| CR-10 | 报告含 LLM 深度解读（若配置了 QUANT_LLM_API_KEY），否则用规则模板兜底 | 集成测试 |
| CR-11 | archive 自动包含 `attribution_report.html` | archive 目录检查 |

### 4.2 量化评估维度

| 维度 | 现状 | 目标 | 验证方法 |
|------|------|------|---------|
| 报告类型数 | 2（回测报告 + 模板报告） | 3（+ 绩效归因报告） | 文件存在性检查 |
| 意图路由分支 | 2（量化策略 / 个股分析） | 3（+ 绩效复盘） | 单元测试 |
| 归因分析维度 | 0 | 7 维度（交易统计/round-trip/按标的/执行质量/净值/压力期/连胜连败） | 报告章节数 |
| 报告生成耗时 | N/A | ≤ 5 秒（1000 条记录） | pytest 计时 |

---

## 第 5 章 兼容性与回滚

### 5.1 兼容层

| 兼容项 | 机制 |
|--------|------|
| 报告路由优先级 | 绩效复盘 > 回测报告 > 模板报告，不影响现有两条路径 |
| 现有 `run()` 分支 | 仅新增 if 分支，不修改现有回测/模板逻辑 |
| 无 EXECUTION 产物 | 返回友好错误 `{"success": False, "error": "..."}`，不崩溃 |
| `ledger.jsonl` 损坏行 | 复用 `read_paper_trades` 的 skip + warning 逻辑 |
| LLM 未配置 | 降级到规则模板兜底，与现有 `_build_fallback_*` 模式一致 |
| 现有 `parse_intent()` | 仅新增 `if` 判断，不修改现有逻辑分支 |

### 5.2 回滚计划

| 触发条件 | 回滚动作 | 验证 |
|---------|---------|------|
| 归因分析器加载 ledger 失败 | 返回错误提示，不生成报告 | 单元测试 |
| Round-trip 归组异常 | 跳过归组分析章节，仍生成其他章节 | 边界测试 |
| 报告生成耗时 > 10 秒 | 跳过月度热力图和压力期分析（降级输出） | 性能测试 |
| LLM 调用失败 | 使用规则模板兜底 | 现有机制已验证 |

---

## 第 6 章 测试策略

遵循项目硬约束"关键路径 100% 行覆盖，非关键路径 ≥ 80%"。

### 6.1 L2 单元测试

新增 `tests/reports_engine/test_attribution_analyzer.py`：

| 测试用例 | 描述 |
|---------|------|
| `test_load_ledger_basic` | 正常加载 ledger.jsonl |
| `test_load_ledger_empty` | 空文件 / 不存在的文件 |
| `test_load_ledger_corrupted_lines` | 损坏行自动跳过 |
| `test_build_round_trips_fifo` | FIFO 匹配买卖闭环 |
| `test_build_round_trips_partial_sell` | 部分卖出后剩余持仓 |
| `test_build_round_trips_no_sells` | 仅有买入无卖出 |
| `test_round_trip_pnl_calculation` | 盈亏计算正确性（含费用） |
| `test_get_transaction_stats` | 交易统计概览 |
| `test_get_round_trip_stats` | Round-trip 统计（胜率/盈亏比等） |
| `test_get_pnl_by_stock` | 按标的归因 |
| `test_get_execution_quality` | 执行质量分析 |
| `test_get_nav_series` | 净值序列提取 |
| `test_get_stress_period_performance` | 压力期表现（含覆盖/不覆盖边界） |
| `test_get_consecutive_stats` | 连胜/连败统计 |

### 6.2 L2 单元测试（意图解析）

扩展现有 `tests/test_engine.py`：

| 测试用例 | 描述 |
|---------|------|
| `test_parse_attribution_intent` | "复盘"关键词触发绩效复盘意图 |
| `test_parse_attribution_intent_variants` | 多种关键词变体（归因分析/实盘报告/交易复盘等） |
| `test_parse_attribution_not_triggered` | 普通输入不触发误识别 |

### 6.3 L3 集成测试

新增 `tests/reports_engine/test_attribution_report.py`：

| 测试用例 | 描述 |
|---------|------|
| `test_end_to_end_attribution_report` | 端到端：从 ledger.jsonl 到 attribution_report.html |
| `test_attribution_report_no_ledger` | 无 ledger 时返回友好错误 |
| `test_attribution_report_archive` | archive 自动包含归因报告 |
| `test_llm_fallback` | LLM 不可用时规则模板兜底 |

### 6.4 测试 marker

新增 `@pytest.mark.skill_reports_engine`（已有），归因相关测试放 `tests/reports_engine/` 目录下。

---

## 第 7 章 实施顺序与依赖

### 7.1 任务清单

| ID | 任务 | 文件 | 依赖 |
|----|------|------|------|
| T1 | 新增 `ATTRIBUTION_KEYWORDS` + `_is_attribution_intent()` | `engine.py` | - |
| T2 | `parse_intent()` 新增绩效复盘路由分支 | `engine.py` | T1 |
| T3 | 实现 `AttributionAnalyzer` 类（加载/解析/FIFO 归组） | `skills/reports-engine/scripts/attribution_analyzer.py`（新文件） | - |
| T4 | 实现 `get_transaction_stats` / `get_round_trip_stats` | `skills/reports-engine/scripts/attribution_analyzer.py` | T3 |
| T5 | 实现 `get_pnl_by_stock` / `get_execution_quality` | `skills/reports-engine/scripts/attribution_analyzer.py` | T3 |
| T6 | 实现 `get_nav_series` / `get_stress_period_performance` / `get_consecutive_stats` | `skills/reports-engine/scripts/attribution_analyzer.py` | T3 |
| T7 | 新增 `attribution_report.py` 模板 + HTML 构建 | `skills/reports-engine/scripts/templates/attribution_report.py`（新文件） | T4, T5, T6 |
| T8 | 实现 `_run_attribution_report()` 函数 | `skills/reports-engine/engine.py` | T7 |
| T9 | `run()` 新增第三分支路由 | `skills/reports-engine/engine.py` | T8 |
| T10 | 实现 LLM prompt 构建 + 注入逻辑 | `skills/reports-engine/engine.py` | T8 |
| T11 | L2 单元测试（AttributionAnalyzer） | `tests/reports_engine/test_attribution_analyzer.py`（新文件） | T3-T6 |
| T12 | L2 单元测试（意图解析） | `tests/test_engine.py` | T2 |
| T13 | L3 集成测试 | `tests/reports_engine/test_attribution_report.py`（新文件） | T9 |
| T14 | 集成到 archive 自动归档 | `skills/reports-engine/engine.py`（复用现有归档逻辑） | T9 |
| T15 | 文档更新（SKILL.md / README.md） | 2 个文件 | T9 |

### 7.2 依赖图

```
T1 (关键词) ──→ T2 (意图解析) ──→ T12 (意图测试)
T3 (分析器-加载) ──┬─→ T4 (交易统计) ──┬─→ T7 (模板) ──→ T8 (_run_attribution) ──→ T9 (路由)
                   ├─→ T5 (标的归因) ──┤                                      ├─→ T10 (LLM注入)
                   └─→ T6 (净值/压力) ─┘                                      ├─→ T14 (archive)
                                                                              └─→ T15 (文档)
T11 (分析器测试) ←─ T3, T4, T5, T6
T13 (集成测试) ←─ T9
```

### 7.3 实施顺序

1. **Phase 1（意图解析）**：T1 → T2（可并行）→ T12
2. **Phase 2（归因分析器）**：T3 → T4 + T5 + T6（可并行）→ T11
3. **Phase 3（报告生成）**：T7 → T8 → T9 → T10
4. **Phase 4（集成与测试）**：T13 → T14
5. **Phase 5（文档）**：T15

---

## 第 8 章 并行开发协调机制

### 8.1 文件负责人划分

| 文件 | 负责人角色 | 协作标注 |
|------|----------|---------|
| `engine.py`（主调度器） | 主调度器负责人 | ⚠️ 协作文件（仅 `parse_intent` 方法新增分支） |
| `skills/reports-engine/scripts/attribution_analyzer.py` | 归因分析器实现者 | - |
| `skills/reports-engine/scripts/templates/attribution_report.py` | 报告模板实现者 | - |
| `skills/reports-engine/engine.py` | 报告引擎负责人 | ⚠️ 协作文件（`run()` 新增分支 + `_run_attribution_report()`） |

### 8.2 sys.path 隔离

每个测试文件独立 conftest.py 清理 sys.path，遵循项目硬约束。

### 8.3 分支策略

- 独立 feature 分支：`attribution-report`
- 合并顺序：独立功能，无跨分支依赖，可随时合并
- 遵循项目约定：合并前全量回归测试通过

---

## 第 9 章 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| `ledger.jsonl` 格式与 `PaperTradeRecordV1` schema 不一致 | 低 | 高 | `load()` 使用 `json.loads` 宽松解析，损坏行 skip |
| 大量成交记录（>10000 条）导致 FIFO 归组耗时 | 中 | 低 | 按标的分组处理，每个标的独立 FIFO 队列 |
| 部分卖出后剩余持仓无法归组 | 高 | 低 | 未匹配的买入保留在队列中，不参与归组统计 |
| 同时存在多个 execution 目录 | 低 | 中 | 取 `ctx.get_artifact("EXECUTION")` 返回的最新路径 |
| 归因报告与回测报告路由冲突 | 低 | 高 | 归因报告优先级最高，语义明确不冲突 |
| LLM 调用超时 | 中 | 低 | 10 秒超时 + 规则模板兜底 |
| 净值序列缺失（ledger 无 nav_after 字段） | 低 | 中 | 跳过净值曲线和月度热力图章节 |

---

## 第 10 章 附录

### 10.1 环境变量

本 PRD 不新增环境变量。绩效归因报告由意图识别触发，无需额外配置开关。

### 10.2 Frozen Core 保护

本 PRD 不触碰 PRD v1.2 定义的 6 项 Frozen Core 路径：
- `real_broker`
- `risk`
- `schemas/order`
- `schemas/execution_report`
- `engine.py`（仅新增方法，不修改现有路径逻辑）
- `portfolio-risk-engine/scripts/cost.py`

### 10.3 与同类项目方案对比

| 特性 | pyfolio | QuantConnect | backtrader | jingni-trader（本 PRD） |
|------|---------|-------------|------------|------------------------|
| Round-trip 分析 | 内置 `round_trips` | TradeBuilder | 需自定义 Analyzer | FIFO 归组，支持部分卖出 |
| Brinson 归因 | 不支持 | 不支持 | 不支持 | 复用 `ReportGenerator.calc_brinson_attribution()` |
| 执行质量 | 滑点/佣金独立统计 | 交易成本分析 | 需自定义 | 费用占比 + 滑点 bps + 成交规模 |
| 压力期表现 | 需自定义 | 支持自定义周期 | 需自定义 | 内置 5 个 A 股特定压力期 |
| LLM 解读 | 不支持 | 不支持 | 不支持 | 复用现有 LLM 注入机制 |
| 语言 | 英文 | 英文 | 英文 | 中文报告 |

### 10.4 与其他报告的关系

| 报告类型 | 路由条件 | 回答的问题 | 数据依赖 |
|---------|---------|-----------|---------|
| 策略回测报告 | 有 BACKTEST 产物 | 策略理论上能赚钱吗？ | 历史行情 + 因子 + 模型 + 回测成交 |
| 股票分析报告 | 默认（无 BACKTEST 且无 attribution 意图） | 这只股票现在值得买吗？ | 个股行情 + 技术指标 + 财报 |
| **绩效归因报告** | `report_intent == "attribution"` | 实盘盈亏来自哪里？ | ledger.jsonl + trade_log.json |

### 10.5 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `engine.py` | 修改 | 新增 `ATTRIBUTION_KEYWORDS` + `_is_attribution_intent()` + `parse_intent()` 分支 |
| `skills/reports-engine/engine.py` | 修改 | `run()` 新增第三分支 + `_run_attribution_report()` + `_inject_attribution_analysis()` |
| `skills/reports-engine/scripts/attribution_analyzer.py` | 新增 | `AttributionAnalyzer` 类 + `RoundTrip` 数据类 |
| `skills/reports-engine/scripts/templates/attribution_report.py` | 新增 | 归因报告 HTML 模板 + 图表函数 |
| `tests/reports_engine/test_attribution_analyzer.py` | 新增 | L2 单元测试（14 项） |
| `tests/reports_engine/test_attribution_report.py` | 新增 | L3 集成测试（4 项） |
| `tests/test_engine.py` | 修改 | 新增意图解析测试（3 项） |
| `skills/reports-engine/SKILL.md` | 修改 | 新增归因报告类型说明 |
| `README.md` | 修改 | 更新"复盘"环节状态为已完成 |

---

**文档结束。状态：待开发。**