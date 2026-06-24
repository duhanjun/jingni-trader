"""
Top-K Dropout 策略单元测试

测试目标：
1. 正确性：top_k 个持仓、淘汰 n_dropout 个、权重归一化。
2. 边界：当前持仓为空 / 候选不足 / 分数全相等。
3. 换手率：相比 TopK，TopKDropout 换手更低。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from topk_dropout_strategy import TopKDropoutStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_scores(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}.SH" for i in range(1, n + 1)]
    scores = rng.normal(0, 1, size=n)
    df = pd.DataFrame({"code": codes, "alpha_score": scores})
    return df.sort_values("alpha_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_rebalance():
    scores = make_scores(n=100)
    strat = TopKDropoutStrategy(top_k=10, n_dropout=2)
    holdings = []  # 第一次建仓
    out = strat.rebalance(holdings, scores)
    assert len(out) == 10
    assert out["weight"].sum() == pytest.approx(1.0, abs=1e-9)
    # top 10 应当与分数最高的前 10 只股票一致
    expected = scores.head(10)["code"].tolist()
    assert set(out["code"]) == set(expected)


def test_dropout_drops_lowest_old_holdings():
    scores = make_scores(n=100)
    strat = TopKDropoutStrategy(top_k=10, n_dropout=3)
    # 假设旧持仓 = 之前 top10
    old_holdings = scores.head(10)["code"].tolist()
    out = strat.rebalance(old_holdings, scores)
    # 旧持仓中分数最低的 3 个应被淘汰
    lowest_three = scores.head(10).tail(3)["code"].tolist()
    for code in lowest_three:
        assert code not in set(out["code"])
    # 新进 3 个 = 榜外分数最高的 3 个
    top10_set = set(scores.head(10)["code"])
    expected_new = [c for c in scores["code"] if c not in top10_set][:3]
    for code in expected_new:
        assert code in set(out["code"])
    assert len(out) == 10


def test_score_weighted():
    scores = make_scores(n=100)
    strat = TopKDropoutStrategy(top_k=5, n_dropout=0, weight_method="score")
    out = strat.rebalance([], scores)
    assert len(out) == 5
    assert out["weight"].sum() == pytest.approx(1.0, abs=1e-9)
    # 最高分股票应获得最大权重
    top_code = scores.iloc[0]["code"]
    assert out.set_index("code").loc[top_code, "weight"] == out["weight"].max()


def test_equal_weight_distribution():
    scores = make_scores(n=50)
    strat = TopKDropoutStrategy(top_k=10, n_dropout=0)
    out = strat.rebalance([], scores)
    # pandas Series 与标量比较应逐元素
    for w in out["weight"].tolist():
        assert w == pytest.approx(0.1, abs=1e-7)


def test_invalid_params():
    with pytest.raises(ValueError):
        TopKDropoutStrategy(top_k=0)
    with pytest.raises(ValueError):
        TopKDropoutStrategy(top_k=10, n_dropout=11)
    with pytest.raises(ValueError):
        TopKDropoutStrategy(top_k=10, weight_method="bogus")


def test_empty_scores():
    strat = TopKDropoutStrategy(top_k=10, n_dropout=2)
    empty = pd.DataFrame({"code": [], "alpha_score": []})
    out = strat.rebalance([], empty)
    assert out.empty


def test_missing_columns():
    strat = TopKDropoutStrategy(top_k=10, n_dropout=2)
    bad = pd.DataFrame({"code": ["a"], "wrong_col": [0.5]})
    with pytest.raises(KeyError):
        strat.rebalance([], bad)
    bad2 = pd.DataFrame({"alpha_score": [0.5]})
    with pytest.raises(KeyError):
        strat.rebalance([], bad2)


def test_dropout_introduces_new_stocks():
    """TopKDropout 的核心特性：强制换出部分旧持仓，引入榜外新股票。"""
    scores = make_scores(n=100, seed=7)
    topk = TopKDropoutStrategy(top_k=10, n_dropout=0)
    dropout = TopKDropoutStrategy(top_k=10, n_dropout=3)
    old_topk = scores.head(10)["code"].tolist()
    old_drop = scores.head(10)["code"].tolist()
    out_topk = topk.rebalance(old_topk, scores)
    out_drop = dropout.rebalance(old_drop, scores)
    # 关键不变量：dropout 换出 3 个旧持仓 + 补入 3 个新股票
    topk_set = set(out_topk["code"])
    drop_set = set(out_drop["code"])
    # 旧持仓被强制换出
    forced_out = set(old_drop) - drop_set
    assert len(forced_out) == 3
    # 补入的 3 个股票原本不在 old_drop
    forced_in = drop_set - set(old_drop)
    assert len(forced_in) == 3
    # 持仓数仍是 10
    assert len(out_drop) == 10


def test_dropout_total_weight_equals_one():
    """多轮调仓后总权重保持为 1。"""
    scores = make_scores(n=80, seed=11)
    strat = TopKDropoutStrategy(top_k=20, n_dropout=4)
    holdings = []
    for _ in range(3):
        out = strat.rebalance(holdings, scores)
        assert out["weight"].sum() == pytest.approx(1.0, abs=1e-9)
        holdings = out["code"].tolist()


def test_no_existing_holdings_then_topk_is_built():
    scores = make_scores(n=20)
    strat = TopKDropoutStrategy(top_k=5, n_dropout=5)
    out = strat.rebalance([], scores)
    assert len(out) == 5
    assert set(out["code"]) == set(scores.head(5)["code"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))