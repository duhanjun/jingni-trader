"""reports-engine L2 单元测试：attribution_analyzer。

覆盖 AttributionAnalyzer 的核心功能：
- load() 加载 ledger.jsonl（含损坏行跳过、空文件处理）
- build_round_trips() FIFO 归组（含部分卖出、无卖出边界）
- 净盈亏计算（net_pnl = gross_pnl - commission - stamp_tax - slippage_cost）
- get_transaction_stats() 交易统计概览
- get_round_trip_stats() round-trip 胜率/盈亏比/持仓天数
- get_pnl_by_stock() 按标的归因（按 total_pnl 降序）
- get_execution_quality() 执行质量（成交额/成本 bps/滑点 bps）
- get_nav_series() 净值序列
- get_stress_period_performance() A 股压力期表现
- get_consecutive_stats() 连胜/连败统计
"""
from __future__ import annotations

import os
import sys
import json
import importlib.util as ilu

import pytest
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANALYZER_PATH = os.path.join(ROOT, "skills", "reports-engine", "scripts", "attribution_analyzer.py")


def _load_analyzer():
    """显式加载 attribution_analyzer.py 为独立模块。

    使用 importlib 规避 scripts 包名冲突（多个 skill 各有 scripts 目录）。
    """
    spec = ilu.spec_from_file_location("attribution_analyzer_test", ANALYZER_PATH)
    mod = ilu.module_from_spec(spec)
    sys.modules["attribution_analyzer_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_ledger(tmp_path, records):
    """将记录列表写入 tmp_path/ledger.jsonl（JSONL 格式）。"""
    ledger_file = tmp_path / "ledger.jsonl"
    with ledger_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(ledger_file)


def _make_record(
    execution_id,
    trade_date,
    code,
    side,
    shares,
    price,
    commission=5.0,
    stamp_tax=0.0,
    slippage_cost=0.0,
    position_after_shares=0,
    cash_after=1_000_000.0,
    nav_after=1_000_000.0,
    confirmed=True,
):
    """构造单条成交记录（字段与 execution-monitor-engine ledger 对齐）。"""
    return {
        "execution_id": execution_id,
        "trade_date": trade_date,
        "code": code,
        "side": side,
        "shares": shares,
        "price": price,
        "commission": commission,
        "stamp_tax": stamp_tax,
        "slippage_cost": slippage_cost,
        "position_after_shares": position_after_shares,
        "cash_after": cash_after,
        "nav_after": nav_after,
        "confirmed": confirmed,
        "created_at": f"{trade_date}T10:00:00",
    }


@pytest.mark.skill_reports_engine
@pytest.mark.unit
class TestAttributionAnalyzer:
    """AttributionAnalyzer L2 单元测试。"""

    # ------------------------------------------------------------------
    # 1. load()
    # ------------------------------------------------------------------

    def test_load_ledger_basic(self, tmp_path):
        """加载含 2+ 条记录的 ledger.jsonl，返回 True 且记录数正确。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         position_after_shares=100, nav_after=1_000_000.0),
            _make_record("exec_002", "2024-01-16", "000001.SZ", "sell", 100, 11.0,
                         stamp_tax=1.0, position_after_shares=0, nav_after=1_000_089.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        ok = analyzer.load()

        assert ok is True
        assert len(analyzer.records) == 2
        # 记录按 trade_date 升序排列
        assert analyzer.records[0]["execution_id"] == "exec_001"
        assert analyzer.records[1]["execution_id"] == "exec_002"

    def test_load_ledger_empty(self, tmp_path):
        """空文件 / 不存在的文件均返回 False。"""
        mod = _load_analyzer()

        # 不存在的文件
        missing_path = str(tmp_path / "not_exist.jsonl")
        analyzer_missing = mod.AttributionAnalyzer(missing_path)
        assert analyzer_missing.load() is False
        assert analyzer_missing.records == []

        # 空文件
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        analyzer_empty = mod.AttributionAnalyzer(str(empty_file))
        assert analyzer_empty.load() is False
        assert analyzer_empty.records == []

    def test_load_ledger_corrupted_lines(self, tmp_path):
        """混合有效行与损坏 JSON 行，损坏行被跳过。"""
        mod = _load_analyzer()
        ledger_file = tmp_path / "ledger.jsonl"
        # 第 1 行有效、第 2 行损坏、第 3 行有效、第 4 行损坏
        lines = [
            json.dumps(_make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0)),
            "{bad json line",
            json.dumps(_make_record("exec_002", "2024-01-16", "000001.SZ", "sell", 100, 11.0,
                                    stamp_tax=1.0)),
            "not a json at all",
        ]
        ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        analyzer = mod.AttributionAnalyzer(str(ledger_file))
        ok = analyzer.load()

        assert ok is True
        # 仅 2 条有效记录（损坏行已跳过）
        assert len(analyzer.records) == 2
        exec_ids = [r["execution_id"] for r in analyzer.records]
        assert "exec_001" in exec_ids
        assert "exec_002" in exec_ids

    # ------------------------------------------------------------------
    # 2. build_round_trips()
    # ------------------------------------------------------------------

    def test_build_round_trips_fifo(self, tmp_path):
        """买 100 股再卖 100 股 → 1 个 round-trip，买卖价正确。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         commission=5.0, slippage_cost=0.5),
            _make_record("exec_002", "2024-01-20", "000001.SZ", "sell", 100, 11.0,
                         commission=5.0, stamp_tax=1.0, slippage_cost=0.5),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        rts = analyzer.build_round_trips()

        assert len(rts) == 1
        rt = rts[0]
        assert rt.code == "000001.SZ"
        assert rt.buy_price == 10.0
        assert rt.sell_price == 11.0
        assert rt.shares == 100
        assert rt.buy_date == "2024-01-15"
        assert rt.sell_date == "2024-01-20"

    def test_build_round_trips_partial_sell(self, tmp_path):
        """买 200 股，分两次卖 100 股 → 2 个 round-trip。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 200, 10.0,
                         commission=10.0, slippage_cost=1.0),
            _make_record("exec_002", "2024-01-20", "000001.SZ", "sell", 100, 11.0,
                         commission=5.0, stamp_tax=1.0, slippage_cost=0.5),
            _make_record("exec_003", "2024-01-25", "000001.SZ", "sell", 100, 12.0,
                         commission=5.0, stamp_tax=1.2, slippage_cost=0.5),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        rts = analyzer.build_round_trips()

        assert len(rts) == 2
        # 两次卖出对应同一笔买入，买入价应都为 10.0
        assert rts[0].buy_price == 10.0
        assert rts[0].sell_price == 11.0
        assert rts[0].shares == 100
        assert rts[1].buy_price == 10.0
        assert rts[1].sell_price == 12.0
        assert rts[1].shares == 100

    def test_build_round_trips_no_sells(self, tmp_path):
        """仅有买入、无卖出 → 0 个 round-trip。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0),
            _make_record("exec_002", "2024-01-16", "600000.SH", "buy", 200, 20.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        rts = analyzer.build_round_trips()

        assert len(rts) == 0

    # ------------------------------------------------------------------
    # 3. 净盈亏计算
    # ------------------------------------------------------------------

    def test_round_trip_pnl_calculation(self, tmp_path):
        """验证 net_pnl = gross_pnl - commission - stamp_tax - slippage_cost。

        买入 10 元 × 100 股，卖出 11 元 × 100 股：
        - gross_pnl = (11-10)*100 = 100
        - commission = 5(买) + 5(卖) = 10
        - stamp_tax = 1(卖)
        - slippage = 0
        - net_pnl = 100 - 10 - 1 - 0 = 89
        """
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         commission=5.0, stamp_tax=0.0, slippage_cost=0.0),
            _make_record("exec_002", "2024-01-20", "000001.SZ", "sell", 100, 11.0,
                         commission=5.0, stamp_tax=1.0, slippage_cost=0.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        rts = analyzer.build_round_trips()

        assert len(rts) == 1
        rt = rts[0]
        # 毛盈亏
        assert rt.gross_pnl == pytest.approx(100.0)
        # 佣金：买入 5 + 卖出 5 = 10
        assert rt.commission == pytest.approx(10.0)
        # 印花税：仅卖出 1
        assert rt.stamp_tax == pytest.approx(1.0)
        # 滑点：0
        assert rt.slippage_cost == pytest.approx(0.0)
        # 净盈亏
        assert rt.net_pnl == pytest.approx(89.0)
        # 应判定为盈利
        assert rt.is_win is True

    # ------------------------------------------------------------------
    # 4. get_transaction_stats()
    # ------------------------------------------------------------------

    def test_get_transaction_stats(self, tmp_path):
        """验证 total_trades / total_buys / total_sells / total_commission 等字段。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         commission=5.0, stamp_tax=0.0, slippage_cost=0.5),
            _make_record("exec_002", "2024-01-16", "600000.SH", "buy", 200, 20.0,
                         commission=8.0, stamp_tax=0.0, slippage_cost=1.0),
            _make_record("exec_003", "2024-01-20", "000001.SZ", "sell", 100, 11.0,
                         commission=5.0, stamp_tax=1.0, slippage_cost=0.5),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        stats = analyzer.get_transaction_stats()

        assert stats["total_trades"] == 3
        assert stats["total_buys"] == 2
        assert stats["total_sells"] == 1
        # 佣金合计：5 + 8 + 5 = 18
        assert stats["total_commission"] == pytest.approx(18.0)
        # 印花税合计：0 + 0 + 1 = 1
        assert stats["total_stamp_tax"] == pytest.approx(1.0)
        # 滑点合计：0.5 + 1.0 + 0.5 = 2.0
        assert stats["total_slippage"] == pytest.approx(2.0)
        # 总成本 = 18 + 1 + 2 = 21
        assert stats["total_cost"] == pytest.approx(21.0)
        # 涉及 2 只股票
        assert stats["unique_stocks"] == 2

    # ------------------------------------------------------------------
    # 5. get_round_trip_stats()
    # ------------------------------------------------------------------

    def test_get_round_trip_stats(self, tmp_path):
        """已知胜负的 round-trip，验证 win_rate / profit_factor / avg_holding_days。

        构造 2 个 round-trip：1 胜 1 负
        - RT1（胜）：买入 10，卖出 12，100 股 → gross 200
        - RT2（负）：买入 10，卖出 8，100 股 → gross -200
        """
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         commission=5.0),
            _make_record("exec_002", "2024-01-20", "000001.SZ", "sell", 100, 12.0,
                         commission=5.0, stamp_tax=1.0),
            _make_record("exec_003", "2024-02-01", "000001.SZ", "buy", 100, 10.0,
                         commission=5.0),
            _make_record("exec_004", "2024-02-10", "000001.SZ", "sell", 100, 8.0,
                         commission=5.0, stamp_tax=1.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        analyzer.build_round_trips()
        stats = analyzer.get_round_trip_stats()

        assert stats["total_round_trips"] == 2
        assert stats["win_count"] == 1
        assert stats["loss_count"] == 1
        # 胜率 = 1/2 = 0.5
        assert stats["win_rate"] == pytest.approx(0.5)
        # RT1 净盈亏 = (12-10)*100 - 5 - 5 - 1 = 189
        # RT2 净盈亏 = (8-10)*100 - 5 - 5 - 1 = -211
        # gross_profit = 189，gross_loss = 211
        # profit_factor = 189 / 211 ≈ 0.90
        assert stats["profit_factor"] == pytest.approx(189 / 211, abs=0.01)
        # 持仓天数：RT1=5, RT2=9, 平均=7.0
        assert stats["avg_holding_days"] == pytest.approx(7.0)

    # ------------------------------------------------------------------
    # 6. get_pnl_by_stock()
    # ------------------------------------------------------------------

    def test_get_pnl_by_stock(self, tmp_path):
        """多只股票，DataFrame 按 total_pnl 降序排列。

        股票 A：买入 10 卖出 11 → 盈利
        股票 B：买入 10 卖出 9.5 → 亏损
        股票 C：买入 10 卖出 10.3 → 小幅盈利
        """
        mod = _load_analyzer()
        records = [
            # 股票 A
            _make_record("a1", "2024-01-15", "000001.SZ", "buy", 100, 10.0),
            _make_record("a2", "2024-01-20", "000001.SZ", "sell", 100, 11.0, stamp_tax=1.0),
            # 股票 B
            _make_record("b1", "2024-01-15", "600000.SH", "buy", 100, 10.0),
            _make_record("b2", "2024-01-20", "600000.SH", "sell", 100, 9.5, stamp_tax=1.0),
            # 股票 C
            _make_record("c1", "2024-01-15", "300001.SZ", "buy", 100, 10.0),
            _make_record("c2", "2024-01-20", "300001.SZ", "sell", 100, 10.3, stamp_tax=1.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        analyzer.build_round_trips()
        df = analyzer.get_pnl_by_stock()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        # 降序：A(净盈亏最高) > C(小盈利) > B(亏损)
        assert df.iloc[0]["code"] == "000001.SZ"
        assert df.iloc[1]["code"] == "300001.SZ"
        assert df.iloc[2]["code"] == "600000.SH"
        # total_pnl 单调递减
        pnls = df["total_pnl"].tolist()
        assert pnls[0] > pnls[1] > pnls[2]

    # ------------------------------------------------------------------
    # 7. get_execution_quality()
    # ------------------------------------------------------------------

    def test_get_execution_quality(self, tmp_path):
        """验证 total_turnover / cost_ratio_bps / slippage_ratio_bps。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         commission=5.0, stamp_tax=0.0, slippage_cost=1.0),
            _make_record("exec_002", "2024-01-20", "000001.SZ", "sell", 100, 11.0,
                         commission=5.0, stamp_tax=1.0, slippage_cost=1.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        q = analyzer.get_execution_quality()

        # 总成交额：10*100 + 11*100 = 2100
        assert q["total_turnover"] == pytest.approx(2100.0)
        # 总成本：佣金 5+5 + 印花 0+1 + 滑点 1+1 = 13
        assert q["total_cost"] == pytest.approx(13.0)
        # 成本占比 bps = 13 / 2100 * 10000 ≈ 61.90
        assert q["cost_ratio_bps"] == pytest.approx(13 / 2100 * 10000, abs=0.1)
        # 滑点占比 bps = 2 / 2100 * 10000 ≈ 9.52
        assert q["slippage_ratio_bps"] == pytest.approx(2 / 2100 * 10000, abs=0.1)
        # 平均/最大单笔成交额
        assert q["avg_trade_size"] == pytest.approx(1050.0)
        assert q["max_trade_size"] == pytest.approx(1100.0)

    # ------------------------------------------------------------------
    # 8. get_nav_series()
    # ------------------------------------------------------------------

    def test_get_nav_series(self, tmp_path):
        """含 nav_after 字段的记录，验证返回 pandas Series 且日期正确。"""
        mod = _load_analyzer()
        records = [
            _make_record("exec_001", "2024-01-15", "000001.SZ", "buy", 100, 10.0,
                         nav_after=1_000_000.0),
            _make_record("exec_002", "2024-01-16", "000001.SZ", "sell", 100, 11.0,
                         stamp_tax=1.0, nav_after=1_000_100.0),
            _make_record("exec_003", "2024-01-17", "600000.SH", "buy", 100, 20.0,
                         nav_after=1_000_050.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        nav = analyzer.get_nav_series()

        assert isinstance(nav, pd.Series)
        assert len(nav) == 3
        # 索引为日期（DatetimeIndex）
        assert str(nav.index[0].date()) == "2024-01-15"
        assert str(nav.index[1].date()) == "2024-01-16"
        assert str(nav.index[2].date()) == "2024-01-17"
        # 值正确
        assert nav.iloc[0] == pytest.approx(1_000_000.0)
        assert nav.iloc[1] == pytest.approx(1_000_100.0)
        assert nav.iloc[2] == pytest.approx(1_000_050.0)

    # ------------------------------------------------------------------
    # 9. get_stress_period_performance()
    # ------------------------------------------------------------------

    def test_get_stress_period_performance(self, tmp_path):
        """记录覆盖 2020-01 至 2020-03，验证压力期被识别。

        压力期 "2020年疫情" 范围：2020-01-20 至 2020-03-23。
        构造该期间内多日的 nav 记录，期初 100 万 → 期末 95 万。
        """
        mod = _load_analyzer()
        records = [
            _make_record("d1", "2020-01-20", "000001.SZ", "buy", 100, 10.0,
                         nav_after=1_000_000.0),
            _make_record("d2", "2020-02-10", "000001.SZ", "buy", 100, 9.0,
                         nav_after=900_000.0),
            _make_record("d3", "2020-03-23", "000001.SZ", "buy", 100, 9.5,
                         nav_after=950_000.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        stress = analyzer.get_stress_period_performance()

        # 应包含 2020 年疫情压力期
        assert "2020年疫情" in stress
        period = stress["2020年疫情"]
        assert "return_pct" in period
        assert "max_drawdown_pct" in period
        # 期初 1_000_000，期末 950_000 → 收益率 -5%
        assert period["return_pct"] == pytest.approx(-5.0, abs=0.1)
        # 最大回撤：1_000_000 → 900_000 = -10%
        assert period["max_drawdown_pct"] == pytest.approx(-10.0, abs=0.1)

    # ------------------------------------------------------------------
    # 10. get_consecutive_stats()
    # ------------------------------------------------------------------

    def test_get_consecutive_stats(self, tmp_path):
        """胜负交替的 round-trip，验证 max_win_streak / max_loss_streak。

        构造 4 个交替胜负的 round-trip：胜-负-胜-负。
        胜：买入 10，卖出 11
        负：买入 10，卖出 9
        """
        mod = _load_analyzer()
        records = [
            # RT1 胜
            _make_record("b1", "2024-01-15", "000001.SZ", "buy", 100, 10.0),
            _make_record("s1", "2024-01-16", "000001.SZ", "sell", 100, 11.0, stamp_tax=1.0),
            # RT2 负
            _make_record("b2", "2024-01-17", "000001.SZ", "buy", 100, 10.0),
            _make_record("s2", "2024-01-18", "000001.SZ", "sell", 100, 9.0, stamp_tax=1.0),
            # RT3 胜
            _make_record("b3", "2024-01-19", "000001.SZ", "buy", 100, 10.0),
            _make_record("s3", "2024-01-20", "000001.SZ", "sell", 100, 11.0, stamp_tax=1.0),
            # RT4 负
            _make_record("b4", "2024-01-21", "000001.SZ", "buy", 100, 10.0),
            _make_record("s4", "2024-01-22", "000001.SZ", "sell", 100, 9.0, stamp_tax=1.0),
        ]
        ledger_path = _write_ledger(tmp_path, records)

        analyzer = mod.AttributionAnalyzer(ledger_path)
        analyzer.load()
        analyzer.build_round_trips()
        stats = analyzer.get_consecutive_stats()

        # 胜负交替 → 最大连胜 = 1，最大连败 = 1
        assert stats["max_win_streak"] == 1
        assert stats["max_loss_streak"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
