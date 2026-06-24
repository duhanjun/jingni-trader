"""
CPCV vs jingni-trader 原 purged_group_ts_split 对比

由于 jingni-trader 子模块需要 PYTHONPATH 设置，
这里使用直接复制原 split 逻辑的方式进行对比
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, "/workspace")
from skills.quant-optimizations.quant_opt_20260619.cpcv import CombinatorialPurgedCV, CPVCSplit


def jingni_purged_group_ts_split(dates: pd.Series, n_splits: int = 5, purge_gap_days: int = 5):
    """
    复刻自 /workspace/skills/strategy-model-engine/engine.py:109 purged_group_ts_split
    用于在脱离 jingni-trader 主项目的情况下做对比
    """
    unique_dates = sorted(dates.unique())
    n_dates = len(unique_dates)
    splits = []
    test_size = n_dates // (n_splits + 1)
    for i in range(n_splits):
        train_end_idx = n_dates - (n_splits - i) * test_size
        val_start_idx = train_end_idx + 1
        val_end_idx = min(val_start_idx + test_size, n_dates)
        if val_start_idx >= n_dates:
            break
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        if purge_gap_days > 0:
            purge_date = unique_dates[train_end_idx] - timedelta(days=purge_gap_days)
            train_dates = [d for d in train_dates if d <= purge_date]
        train_idx = dates[dates.isin(train_dates)].index.values
        val_idx = dates[dates.isin(val_dates)].index.values
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits


def main():
    print("=" * 70)
    print("CPCV vs jingni-trader 原 purged_group_ts_split 对比")
    print("=" * 70)

    dates = pd.date_range("2023-01-01", periods=480, freq="D")
    codes = [f"{600000 + i:06d}" for i in range(20)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"code": c, "date": d, "value": np.random.randn()})
    df = pd.DataFrame(rows)
    print(f"数据集: {len(df)} 行, {df['date'].nunique()} 个交易日, "
          f"{df['code'].nunique()} 只股票")

    print("\n--- jingni 原 purged_group_ts_split (n_splits=3, purge_gap=5d) ---")
    legacy = jingni_purged_group_ts_split(df["date"], n_splits=3, purge_gap_days=5)
    print(f"生成 {len(legacy)} 个 (train, val) 切分")
    total_train_legacy = sum(len(s[0]) for s in legacy)
    total_test_legacy = sum(len(s[1]) for s in legacy)
    print(f"总训练样本: {total_train_legacy}, 总验证样本: {total_test_legacy}")
    print(f"平均 train/test 比例: {total_train_legacy/total_test_legacy:.2f}")

    print("\n--- 新 CPCV (n_splits=5, n_test=2, embargo=1%, purge=1%) ---")
    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2,
                                  embargo_pct=0.01, purge_pct=0.01)
    new_splits = list(cpcv.split(df))
    print(f"生成 {len(new_splits)} 条 (train, test) 路径")
    total_train_new = sum(len(s.train_idx) for s in new_splits)
    total_test_new = sum(len(s.test_idx) for s in new_splits)
    print(f"总训练样本: {total_train_new}, 总测试样本: {total_test_new}")
    print(f"平均 train/test 比例: {total_train_new/total_test_new:.2f}")

    print(f"\n--- 关键差异 ---")
    print(f"路径数: {len(legacy)} → {len(new_splits)} "
          f"(提升 {len(new_splits)/len(legacy):.1f}x)")
    print(f"样本覆盖: 单一 train/val 切分 -> C(5,2)=10 条独立路径")
    print(f"  优势：可生成 10 条独立的 equity 曲线，更稳健地估计策略表现")
    print(f"  优势：每条路径覆盖不同 test fold 组合，避免单一切分的过拟合风险")

    print("\n--- Embargo 隔离期效果验证 ---")
    cv_no_emb = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo_pct=0.0, purge_pct=0.0)
    cv_emb = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo_pct=0.05, purge_pct=0.0)
    s_no = next(cv_no_emb.split(df))
    s_emb = next(cv_emb.split(df))
    print(f"无 embargo: train_size = {len(s_no.train_idx)}")
    print(f"有 embargo (5%): train_size = {len(s_emb.train_idx)}")
    print(f"被 embargo 切除: {len(s_no.train_idx) - len(s_emb.train_idx)} 样本")
    print("=" * 70)


if __name__ == "__main__":
    main()