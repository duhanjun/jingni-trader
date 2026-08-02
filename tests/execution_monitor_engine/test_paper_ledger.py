"""P1-1 追加式 JSONL Paper Trading 账本测试

覆盖（PRD P1-1.9）：
1. 追加 append_paper_trade
2. 去重（execution_id 重复 raise）
3. 重放 replay_ledger 重建状态
4. T+1 强制（卖出当日买入标的 raise）
5. confirmed=False raise
6. 损坏行 skip + warning
7. position_after_shares 重建
8. 旧状态迁移成功
9. 旧状态迁移字段缺失 raise
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest


# ============================================================================
# 动态加载 paper_ledger 模块（避免 sys.path 污染）
# ============================================================================

@pytest.fixture(scope="module")
def paper_ledger_mod():
    """动态加载 execution-monitor-engine/scripts/paper_ledger.py"""
    import importlib.util as ilu
    import sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    mod_path = repo_root / "skills" / "execution-monitor-engine" / "scripts" / "paper_ledger.py"
    spec = ilu.spec_from_file_location("_paper_ledger_test", str(mod_path))
    mod = ilu.module_from_spec(spec)
    # 必须先注册到 sys.modules，否则 dataclass 装饰器无法解析 __module__
    sys.modules["_paper_ledger_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_record(paper_ledger_mod, **overrides):
    """构造一条 PaperTradeRecordV1，默认合法"""
    defaults = dict(
        execution_id="test_001",
        trade_date="2026-08-01",
        code="000001.SZ",
        side="buy",
        shares=200,
        price=10.0,
        commission=5.0,
        stamp_tax=0.0,
        slippage_cost=1.0,
        position_after_shares=200,
        cash_after=98995.0,
        nav_after=100000.0,
        confirmed=True,
        created_at=datetime(2026, 8, 1, 10, 0, 0),
    )
    defaults.update(overrides)
    return paper_ledger_mod.PaperTradeRecordV1(**defaults)


# ============================================================================
# 1. 追加
# ============================================================================

class TestAppendPaperTrade:
    def test_append_creates_file(self, tmp_path, paper_ledger_mod):
        """追加记录后文件存在且内容正确"""
        ledger = tmp_path / "ledger.jsonl"
        rec = _make_record(paper_ledger_mod)
        paper_ledger_mod.append_paper_trade(ledger, rec)
        assert ledger.exists()
        content = ledger.read_text(encoding="utf-8").strip()
        obj = json.loads(content)
        assert obj["execution_id"] == "test_001"
        assert obj["side"] == "buy"

    def test_append_multiple_records(self, tmp_path, paper_ledger_mod):
        """追加多条记录"""
        ledger = tmp_path / "ledger.jsonl"
        for i in range(3):
            rec = _make_record(paper_ledger_mod, execution_id=f"test_{i:03d}")
            paper_ledger_mod.append_paper_trade(ledger, rec)
        lines = ledger.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3


# ============================================================================
# 2. 去重
# ============================================================================

class TestDedup:
    def test_duplicate_execution_id_raises(self, tmp_path, paper_ledger_mod):
        """execution_id 重复时 raise"""
        ledger = tmp_path / "ledger.jsonl"
        rec = _make_record(paper_ledger_mod, execution_id="dup_001")
        paper_ledger_mod.append_paper_trade(ledger, rec)
        rec2 = _make_record(paper_ledger_mod, execution_id="dup_001", shares=300)
        with pytest.raises(ValueError, match="重复"):
            paper_ledger_mod.append_paper_trade(ledger, rec2)


# ============================================================================
# 3. 重放
# ============================================================================

class TestReplayLedger:
    def test_replay_rebuilds_state(self, tmp_path, paper_ledger_mod):
        """replay_ledger 正确重建账户状态"""
        ledger = tmp_path / "ledger.jsonl"
        rec = _make_record(
            paper_ledger_mod,
            execution_id="r1",
            side="buy",
            shares=200,
            price=10.0,
            position_after_shares=200,
            cash_after=98995.0,
            nav_after=100000.0,
        )
        paper_ledger_mod.append_paper_trade(ledger, rec)
        snapshot = paper_ledger_mod.replay_ledger(ledger, init_capital=100000.0)
        assert snapshot.last_trade_date == "2026-08-01"
        assert "000001.SZ" in snapshot.positions
        assert snapshot.positions["000001.SZ"].shares == 200
        assert snapshot.cash == 98995.0

    def test_replay_empty_ledger(self, tmp_path, paper_ledger_mod):
        """空 ledger 返回初始资金快照"""
        ledger = tmp_path / "nonexistent.jsonl"
        snapshot = paper_ledger_mod.replay_ledger(ledger, init_capital=50000.0)
        assert snapshot.cash == 50000.0
        assert snapshot.positions == {}
        assert snapshot.last_trade_date == ""


# ============================================================================
# 4. T+1 强制
# ============================================================================

class TestTPlusOne:
    def test_sell_same_day_buy_raises(self, tmp_path, paper_ledger_mod):
        """T+1: 卖出当日买入标的 raise"""
        ledger = tmp_path / "ledger.jsonl"
        # 买入
        buy_rec = _make_record(
            paper_ledger_mod,
            execution_id="t1_buy",
            side="buy",
            shares=200,
            position_after_shares=200,
            cash_after=98995.0,
        )
        paper_ledger_mod.append_paper_trade(ledger, buy_rec)
        # 同日卖出 → 应 raise
        sell_rec = _make_record(
            paper_ledger_mod,
            execution_id="t1_sell",
            side="sell",
            shares=200,
            position_after_shares=0,
            cash_after=100000.0,
        )
        paper_ledger_mod.append_paper_trade(ledger, sell_rec)
        with pytest.raises(ValueError, match=r"T\+1"):
            paper_ledger_mod.replay_ledger(ledger)

    def test_sell_next_day_ok(self, tmp_path, paper_ledger_mod):
        """T+1: 次日卖出合法"""
        ledger = tmp_path / "ledger.jsonl"
        # Day1 买入
        buy_rec = _make_record(
            paper_ledger_mod,
            execution_id="d1_buy",
            trade_date="2026-08-01",
            side="buy",
            shares=200,
            position_after_shares=200,
            cash_after=98995.0,
        )
        paper_ledger_mod.append_paper_trade(ledger, buy_rec)
        # Day2 卖出（不同日期）
        sell_rec = _make_record(
            paper_ledger_mod,
            execution_id="d2_sell",
            trade_date="2026-08-02",
            side="sell",
            shares=200,
            position_after_shares=0,
            cash_after=100000.0,
        )
        paper_ledger_mod.append_paper_trade(ledger, sell_rec)
        snapshot = paper_ledger_mod.replay_ledger(ledger)
        # 卖完后持仓清空
        assert "000001.SZ" not in snapshot.positions


# ============================================================================
# 5. confirmed=False raise
# ============================================================================

class TestConfirmedConstraint:
    def test_confirmed_false_raises_on_construct(self, paper_ledger_mod):
        """confirmed=False 在模型构造时 raise（P1-1.7）"""
        with pytest.raises(Exception):
            _make_record(paper_ledger_mod, confirmed=False)

    def test_confirmed_false_raises_on_append(self, tmp_path, paper_ledger_mod):
        """append_paper_trade 对 confirmed=False raise"""
        ledger = tmp_path / "ledger.jsonl"
        # 直接构造一个 confirmed=False 的 record 会先在 Pydantic 构造时 raise
        # 所以这里测试的是构造层面的拦截
        with pytest.raises(Exception):
            rec = _make_record(paper_ledger_mod, confirmed=False)
            paper_ledger_mod.append_paper_trade(ledger, rec)


# ============================================================================
# 6. 损坏行 skip
# ============================================================================

class TestCorruptLineSkip:
    def test_corrupt_lines_skipped(self, tmp_path, paper_ledger_mod):
        """损坏行被 skip，不阻断读取"""
        ledger = tmp_path / "ledger.jsonl"
        # 写入一条合法 + 一条损坏 + 一条合法
        rec1 = _make_record(paper_ledger_mod, execution_id="ok1")
        paper_ledger_mod.append_paper_trade(ledger, rec1)
        with ledger.open("a", encoding="utf-8") as f:
            f.write("{bad json line\n")
        rec2 = _make_record(paper_ledger_mod, execution_id="ok2", shares=300)
        paper_ledger_mod.append_paper_trade(ledger, rec2)

        records = paper_ledger_mod.read_paper_trades(ledger)
        assert len(records) == 2  # 损坏行被 skip
        assert records[0].execution_id == "ok1"
        assert records[1].execution_id == "ok2"


# ============================================================================
# 7. position_after_shares 重建
# ============================================================================

class TestPositionRebuild:
    def test_position_after_shares_tracked(self, tmp_path, paper_ledger_mod):
        """多笔交易后 position_after_shares 正确"""
        ledger = tmp_path / "ledger.jsonl"
        # 买入 200
        r1 = _make_record(
            paper_ledger_mod, execution_id="p1",
            side="buy", shares=200, position_after_shares=200,
        )
        paper_ledger_mod.append_paper_trade(ledger, r1)
        # 次日再买 100
        r2 = _make_record(
            paper_ledger_mod, execution_id="p2",
            trade_date="2026-08-02",
            side="buy", shares=100, position_after_shares=300,
        )
        paper_ledger_mod.append_paper_trade(ledger, r2)

        snapshot = paper_ledger_mod.replay_ledger(ledger)
        assert snapshot.positions["000001.SZ"].shares == 300


# ============================================================================
# 8. 旧状态迁移成功
# ============================================================================

class TestLegacyMigration:
    def test_migration_success(self, tmp_path, paper_ledger_mod):
        """account_state.json 存在但 ledger 不存在 → 自动迁移"""
        ledger = tmp_path / "ledger.jsonl"
        state_path = tmp_path / "account_state.json"
        # 写入旧状态文件
        legacy_state = {
            "nav": 100000.0,
            "available_cash": 50000.0,
            "positions": {
                "600000.SH": {"volume": 200, "avg_cost": 10.0, "available_volume": 200}
            },
        }
        state_path.write_text(json.dumps(legacy_state), encoding="utf-8")

        migrated = paper_ledger_mod.migrate_legacy_state(ledger, state_path, 100000.0)
        assert migrated is True
        assert ledger.exists()
        # 备份文件存在
        backups = list(tmp_path.glob("account_state.json.bak.*"))
        assert len(backups) == 1
        # ledger 可重放
        snapshot = paper_ledger_mod.replay_ledger(ledger)
        assert snapshot.cash == 50000.0

    def test_migration_noop_when_ledger_exists(self, tmp_path, paper_ledger_mod):
        """ledger 已存在 → 不迁移"""
        ledger = tmp_path / "ledger.jsonl"
        state_path = tmp_path / "account_state.json"
        # ledger 已存在
        rec = _make_record(paper_ledger_mod)
        paper_ledger_mod.append_paper_trade(ledger, rec)
        state_path.write_text("{}", encoding="utf-8")

        migrated = paper_ledger_mod.migrate_legacy_state(ledger, state_path)
        assert migrated is False

    def test_migration_field_missing_raises(self, tmp_path, paper_ledger_mod):
        """旧状态文件缺 nav/cash → raise"""
        ledger = tmp_path / "ledger.jsonl"
        state_path = tmp_path / "account_state.json"
        # 缺 nav 字段
        state_path.write_text(json.dumps({"available_cash": 50000.0}), encoding="utf-8")

        with pytest.raises(ValueError, match="字段缺失"):
            paper_ledger_mod.migrate_legacy_state(ledger, state_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
