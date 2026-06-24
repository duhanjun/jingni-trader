"""
OPTIMIZATION 3 (part 2) 验证：胜率计算修复
============================================
测试内容：
- 构造合成 trades（买入 pnl 恒负 + 卖出有正有负）
- 断言原始 win_rate 被低估（分母含买入）
- 断言修正 win_rate 仅统计 sell，等于 (pnl>0 的 sell) / (sell 总数)

运行：python tests/test_metrics_fix.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from metrics_fix import calc_win_rate_original, calc_win_rate_corrected


def test_win_rate_fix():
    print("\n=== [3] 胜率修复测试 ===")
    # 3 笔买入(pnl 恒负) + 4 笔卖出(2 正 2 负)
    trades = pd.DataFrame([
        {"action": "buy",  "pnl": -10050.0},
        {"action": "buy",  "pnl": -20300.0},
        {"action": "buy",  "pnl": -5100.0},
        {"action": "sell", "pnl":  1200.0},
        {"action": "sell", "pnl":  -800.0},
        {"action": "sell", "pnl":  3500.0},
        {"action": "sell", "pnl":  -200.0},
    ])

    wr_orig = calc_win_rate_original(trades)
    wr_corr = calc_win_rate_corrected(trades)

    sells = trades[trades["action"] == "sell"]
    expected = (sells["pnl"] > 0).sum() / len(sells)  # 2/4 = 0.5

    print(f"  原始(有bug) win_rate: {wr_orig:.4f}  (期望 2/7={2/7:.4f})")
    print(f"  修正       win_rate: {wr_corr:.4f}  (期望 2/4={0.5:.4f})")
    print(f"  sell 笔数: {len(sells)}, 其中盈利: {(sells['pnl']>0).sum()}")

    # 原始被低估：分母含 3 笔买入
    assert abs(wr_orig - 2 / 7) < 1e-9, f"原始胜率应为 2/7, got {wr_orig}"
    # 修正值正确
    assert abs(wr_corr - expected) < 1e-9, f"修正胜率应为 {expected}, got {wr_corr}"
    # 修正值应高于原始（买入拉低了原始）
    assert wr_corr > wr_orig, "修正胜率应高于原始（原始被买入拉低）"
    print("  [PASS] 胜率修复正确：仅统计 sell 成交")


def test_empty_and_no_sell():
    print("\n=== [3] 边界：空 trades / 无 sell ===")
    # 空
    assert calc_win_rate_original(pd.DataFrame(columns=["action", "pnl"])) == 0.0
    assert calc_win_rate_corrected(pd.DataFrame(columns=["action", "pnl"])) == 0.0
    # 只有买入
    only_buy = pd.DataFrame([
        {"action": "buy", "pnl": -100.0},
        {"action": "buy", "pnl": -200.0},
    ])
    assert calc_win_rate_corrected(only_buy) == 0.0  # 无 sell
    assert calc_win_rate_original(only_buy) == 0.0   # 全负 -> 0
    print("  [PASS] 空 trades / 无 sell 返回 0")


def test_all_wins_and_all_losses():
    print("\n=== [3] 边界：全胜 / 全负 sell ===")
    all_win = pd.DataFrame([
        {"action": "sell", "pnl": 100.0},
        {"action": "sell", "pnl": 200.0},
    ])
    assert calc_win_rate_corrected(all_win) == 1.0
    all_lose = pd.DataFrame([
        {"action": "sell", "pnl": -100.0},
        {"action": "sell", "pnl": -200.0},
    ])
    assert calc_win_rate_corrected(all_lose) == 0.0
    print("  [PASS] 全胜=1.0, 全负=0.0")


def run_all():
    test_win_rate_fix()
    test_empty_and_no_sell()
    test_all_wins_and_all_losses()
    print("\n=== 全部胜率修复测试通过 ===")


if __name__ == "__main__":
    run_all()
