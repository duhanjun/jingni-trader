"""
quant_opt 模块综合使用示例

展示三个模块如何联动：
1. 因子表达式引擎 → 计算多个因子
2. 向量化 IC 分析 → 评估因子
3. 向量化回测 → 验证策略

不修改 main 分支代码，仅作为参考演示。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_opt_20260617.core.vectorized_ic import VectorizedICAnalyzer
from quant_opt_20260617.core.vectorized_backtest import VectorizedBacktester
from quant_opt_20260617.core.factor_expression import (
    FactorExpressionEngine,
    ALPHA101_DEMO,
)


def _make_demo_data(n_stocks: int = 30, n_days: int = 200, seed: int = 42):
    """合成 A 股风格数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_days)
    codes = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]

    # 每只股票自己的漂移与波动（与隐含因子弱相关）
    code_quality = rng.normal(0, 1, n_stocks)
    rows = []
    for ci, c in enumerate(codes):
        quality = code_quality[ci]
        start = rng.uniform(10, 50)
        ret = rng.normal(0.0005 + 0.0003 * quality, 0.02, n_days)
        price = start * (1 + ret).cumprod()
        for i, d in enumerate(dates):
            rows.append({
                "date": d, "code": c,
                "open": price[i], "high": price[i] * 1.01, "low": price[i] * 0.99,
                "close": price[i],
                "volume": int(rng.lognormal(15 + 0.3 * quality, 0.3)),
                "amount": float(price[i] * rng.lognormal(15 + 0.3 * quality, 0.3)),
                "is_limit_up": False, "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("  quant_opt 综合示例：因子计算 + IC 分析 + 回测")
    print("=" * 70)

    # 1) 准备数据
    print("\n[1/5] 准备合成数据 ...")
    data = _make_demo_data(n_stocks=30, n_days=200)
    print(f"  数据规模: {len(data)} 行 ({data['code'].nunique()} 支 × {data['date'].nunique()} 天)")

    # 2) 因子计算
    print("\n[2/5] 用因子表达式引擎计算 5 个 Alpha101 风格因子 ...")
    eng = FactorExpressionEngine(data)
    factors_df = eng.compute_batch(ALPHA101_DEMO)
    print(f"  因子列: {[c for c in factors_df.columns if c not in ('date', 'code')]}")

    # 3) 准备前向收益
    print("\n[3/5] 计算前向 5 日收益 ...")
    data_sorted = data.sort_values(["code", "date"]).reset_index(drop=True)
    data_sorted["ret_forward_5d"] = data_sorted.groupby("code")["close"].pct_change(5).shift(-5)
    ret_df = data_sorted[["date", "code", "ret_forward_5d"]]

    # 4) IC 分析
    print("\n[4/5] 向量化 IC 分析 ...")
    factor_cols = [c for c in factors_df.columns if c not in ("date", "code")]
    ic_analyzer = VectorizedICAnalyzer()
    ic_results = ic_analyzer.analyze(factors_df, ret_df, factor_cols, periods=(5,))

    print(f"  {'因子':<35} {'IC':>8} {'ICIR':>8} {'RankIC':>8} {'t_HAC':>8} {'半衰期':>8}")
    print("  " + "-" * 80)
    for fname in factor_cols:
        m = ic_results[fname][5]
        hl = f"{m['ic_decay_halflife']:.1f}" if np.isfinite(m["ic_decay_halflife"]) else "n/a"
        print(f"  {fname:<35} {m['ic_mean']:>8.4f} {m['ic_ir']:>8.3f} "
              f"{m.get('rank_ic_mean', 0):>8.4f} {m['ic_t_stat_hac']:>8.3f} {hl:>8}")

    # 自动筛选
    selected = ic_analyzer.auto_select(ic_results, primary_period=5)
    print(f"\n  自动筛选通过的因子: {selected}")

    # 5) 用选出的因子生成 topk 信号并回测
    if selected:
        print(f"\n[5/5] 用 '{selected[0]}' 因子生成 top10 信号并回测 ...")
        chosen_factor = selected[0]
        # 每日按因子值取 top10
        signal_rows = []
        for d, g in factors_df.groupby("date"):
            topk = g.nlargest(10, chosen_factor)["code"].tolist()
            for c in g["code"]:
                signal_rows.append({
                    "date": d, "code": c,
                    "signal": 1 if c in topk else 0,
                })
        signals = pd.DataFrame(signal_rows)

        bt = VectorizedBacktester(init_capital=1_000_000)
        result = bt.run(data, signals)
        print(f"  净值终值: {result.equity[-1]:,.0f}")
        print(f"  总收益:   {result.metrics['total_return']*100:.2f}%")
        print(f"  年化收益: {result.metrics['annual_return']*100:.2f}%")
        print(f"  夏普比率: {result.metrics['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {result.metrics['max_drawdown']*100:.2f}%")
    else:
        print("\n  [5/5] 无有效因子通过筛选，跳过回测")

    print("\n" + "=" * 70)
    print("  完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()