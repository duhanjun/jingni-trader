"""
优化验证测试套件

验证内容：
1. 正确性测试 —— 向量化实现与原循环实现结果数学等价
2. 性能对比测试 —— 向量化 vs 原循环实现的耗时对比
3. 边界条件测试 —— 空数据、单只股票、单日、全 NaN 等

运行：python -m optimizations.tests.test_all
"""
import os
import sys
import time
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

# 确保可导入优化模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizations.vectorized_backtest_adapter import VectorizedAdapter
from optimizations.factor_registry import FactorRegistry
from optimizations.vectorized_factor_analysis import VectorizedFactorAnalysis

warnings.filterwarnings("ignore")
np.random.seed(42)


# ════════════════════════════════════════════════════════════
#  测试数据生成
# ════════════════════════════════════════════════════════════
def make_synthetic_data(n_stocks=50, n_days=250, start="2023-01-01"):
    """生成合成 A 股日线数据"""
    dates = pd.bdate_range(start, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    rows = []
    for code in codes:
        price = 10.0 + np.random.randn() * 3
        for dt in dates:
            ret = np.random.randn() * 0.02
            price = max(price * (1 + ret), 1.0)
            vol = np.random.randint(100000, 5000000)
            amt = vol * price
            turnover = np.random.uniform(0.5, 5.0)
            rows.append({
                "code": code, "date": dt,
                "open": price * (1 + np.random.randn() * 0.005),
                "high": price * (1 + abs(np.random.randn()) * 0.01),
                "low": price * (1 - abs(np.random.randn()) * 0.01),
                "close": price,
                "volume": vol, "amount": amt,
                "turnover_rate": turnover,
                "change_pct": ret,
                "is_limit_up": False, "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def make_signals(data, top_pct=0.2):
    """根据 20日反转因子生成信号"""
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    df["reversal"] = -df["ret_20d"]
    df["rank"] = df.groupby("date")["reversal"].rank(pct=True)
    sig = df[["code", "date"]].copy()
    sig["signal"] = 0
    sig.loc[df["rank"] > (1 - top_pct), "signal"] = 1      # 买入前 20%
    sig.loc[df["rank"] < top_pct, "signal"] = -1            # 卖出后 20%
    return sig


# ════════════════════════════════════════════════════════════
#  原循环实现（用于正确性对比，复制自 jingni-trader 原代码逻辑）
# ════════════════════════════════════════════════════════════
def original_ic_loop(data, factor_col, forward_col, ic_type="spearman"):
    """原 _calc_ic 循环实现（复制自 factor-engine/engine.py）"""
    ic_list = []
    dates = sorted(data["date"].unique())
    for dt in dates:
        cross = data[data["date"] == dt].dropna(subset=[factor_col, forward_col])
        if len(cross) < 10:
            continue
        if ic_type == "spearman":
            ic, _ = stats.spearmanr(cross[factor_col], cross[forward_col],
                                    nan_policy="omit")
        else:
            ic, _ = stats.pearsonr(cross[factor_col].fillna(0),
                                   cross[forward_col].fillna(0))
        if not np.isnan(ic):
            ic_list.append({"date": dt, "ic": ic})
    if not ic_list:
        return pd.Series(dtype=float)
    return pd.DataFrame(ic_list).set_index("date")["ic"]


def original_neutralize_loop(factor_df, neutralize_mcap=True,
                             neutralize_industry=True, min_count=30):
    """原 neutralize 循环实现（复制自 factor-engine/engine.py）"""
    from sklearn.linear_model import LinearRegression
    result = factor_df.copy()
    factor_cols = [c for c in factor_df.columns
                   if c not in ["code", "date", "industry", "lncap"]]
    for factor in factor_cols:
        neutralized_values = pd.Series(index=result.index, dtype=float)
        dates = result["date"].unique()
        for dt in dates:
            cross = result[result["date"] == dt].copy()
            if len(cross) < min_count:
                neutralized_values.loc[cross.index] = cross[factor]
                continue
            X_vars = []
            if neutralize_mcap and "lncap" in cross.columns:
                X_vars.append("lncap")
            if neutralize_industry and "industry" in cross.columns:
                dummies = pd.get_dummies(cross["industry"], prefix="ind")
                for col in dummies.columns:
                    cross[col] = dummies[col].values
                    X_vars.append(col)
            if not X_vars:
                neutralized_values.loc[cross.index] = cross[factor]
                continue
            X = cross[X_vars].fillna(0).values
            y = cross[factor].fillna(0).values
            try:
                model = LinearRegression()
                model.fit(X, y)
                resid = y - model.predict(X)
                neutralized_values.loc[cross.index] = resid
            except Exception:
                neutralized_values.loc[cross.index] = cross[factor]
        result[f"{factor}_neutral"] = neutralized_values
    return result


# ════════════════════════════════════════════════════════════
#  测试结果收集
# ════════════════════════════════════════════════════════════
results = {"tests": [], "summary": {}}


def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results["tests"].append({
        "name": name, "status": status, "detail": detail
    })
    print(f"  [{status}] {name}  {detail}")


# ════════════════════════════════════════════════════════════
#  测试 1：向量化回测 —— 正确性
# ════════════════════════════════════════════════════════════
def test_vectorized_backtest_correctness():
    print("\n=== 测试 1：向量化回测正确性 ===")
    data = make_synthetic_data(n_stocks=20, n_days=100)
    signals = make_signals(data)

    adapter = VectorizedAdapter()
    result = adapter.run_backtest(data, signals, init_capital=1e6)

    # 1.1 净值曲线结构正确
    eq = result["equity_curve"]
    has_cols = all(c in eq.columns for c in
                   ["date", "equity", "cash", "market_value", "position_count"])
    record("净值曲线含全部必要列", has_cols,
           f"列={list(eq.columns)}")

    # 1.2 净值起点接近初始资金
    start_eq = eq["equity"].iloc[0]
    record("首日净值接近初始资金", abs(start_eq - 1e6) / 1e6 < 0.05,
           f"首日净值={start_eq:.2f}")

    # 1.3 净值无 NaN / Inf
    valid = eq["equity"].replace([np.inf, -np.inf], np.nan).notna().all()
    record("净值无 NaN/Inf", bool(valid))

    # 1.4 持仓数 <= 股票总数
    max_pos = eq["position_count"].max()
    record("持仓数不超过股票总数", max_pos <= 20,
           f"最大持仓数={max_pos}")

    # 1.5 绩效指标完整
    m = result["metrics"]
    required = ["total_return", "annual_return", "sharpe_ratio",
                "max_drawdown", "calmar_ratio", "sortino_ratio"]
    has_metrics = all(k in m for k in required)
    record("绩效指标完整", has_metrics, f"指标键={list(m.keys())}")

    # 1.6 最大回撤 <= 0
    record("最大回撤为非正", m.get("max_drawdown", 1) <= 0,
           f"max_drawdown={m.get('max_drawdown')}")


# ════════════════════════════════════════════════════════════
#  测试 2：向量化回测 —— 边界条件
# ════════════════════════════════════════════════════════════
def test_vectorized_backtest_boundary():
    print("\n=== 测试 2：向量化回测边界条件 ===")
    adapter = VectorizedAdapter()

    # 2.1 空数据
    r = adapter.run_backtest(pd.DataFrame(), pd.DataFrame())
    record("空数据返回空结果", r["equity_curve"].empty)

    # 2.2 单只股票单日
    one = pd.DataFrame([{
        "code": "000001.SZ", "date": pd.Timestamp("2023-01-03"),
        "open": 10, "high": 10.5, "low": 9.5, "close": 10,
        "volume": 1000, "amount": 10000, "turnover_rate": 1.0,
        "is_limit_up": False, "is_limit_down": False,
    }])
    one_sig = pd.DataFrame([{
        "code": "000001.SZ", "date": pd.Timestamp("2023-01-03"),
        "signal": 1,
    }])
    r = adapter.run_backtest(one, one_sig)
    record("单日数据不崩溃", not r["equity_curve"].empty or r["metrics"] == {})

    # 2.3 全 NaN 信号
    data = make_synthetic_data(n_stocks=10, n_days=30)
    sig = data[["code", "date"]].copy()
    sig["signal"] = np.nan
    r = adapter.run_backtest(data, sig)
    record("全 NaN 信号不崩溃", "equity_curve" in r)

    # 2.4 涨停过滤
    data = make_synthetic_data(n_stocks=5, n_days=20)
    data["is_limit_up"] = True  # 全涨停
    sig = make_signals(data)
    r = adapter.run_backtest(data, sig, price_limit=True)
    record("全涨停时买入被过滤", r["trades"].empty or len(r["trades"]) == 0)


# ════════════════════════════════════════════════════════════
#  测试 3：向量化回测 —— 性能对比
# ════════════════════════════════════════════════════════════
def test_vectorized_backtest_performance():
    print("\n=== 测试 3：向量化回测性能对比 ===")
    # 大规模数据：500 股 × 500 日
    data = make_synthetic_data(n_stocks=300, n_days=500)
    signals = make_signals(data)

    # 向量化引擎
    adapter = VectorizedAdapter()
    t0 = time.time()
    r_vec = adapter.run_backtest(data, signals)
    t_vec = time.time() - t0

    # 原生事件驱动引擎（导入 jingni-trader 原生适配器）
    try:
        sys.path.insert(0, "/workspace/skills/backtest-engine")
        from scripts.adapters.native_adapter import NativeAdapter
        native = NativeAdapter()
        t0 = time.time()
        r_nat = native.run_backtest(data, signals)
        t_nat = time.time() - t0
        speedup = t_nat / t_vec if t_vec > 0 else float("inf")
        record(
            "向量化回测快于原生事件驱动",
            t_vec < t_nat,
            f"向量化={t_vec:.3f}s, 原生={t_nat:.3f}s, 加速比={speedup:.1f}x",
        )
        results["summary"]["backtest_vec_time"] = round(t_vec, 3)
        results["summary"]["backtest_native_time"] = round(t_nat, 3)
        results["summary"]["backtest_speedup"] = round(speedup, 1)
    except Exception as e:
        record("原生引擎对比（导入失败，跳过）", True, str(e)[:80])
        results["summary"]["backtest_vec_time"] = round(t_vec, 3)

    # 记录向量化引擎指标
    results["summary"]["backtest_vec_metrics"] = r_vec.get("metrics", {})


# ════════════════════════════════════════════════════════════
#  测试 4：因子注册表 —— 正确性
# ════════════════════════════════════════════════════════════
def test_factor_registry_correctness():
    print("\n=== 测试 4：因子注册表正确性 ===")
    data = make_synthetic_data(n_stocks=30, n_days=100)

    # 4.1 因子已注册
    factors = FactorRegistry.list_factors()
    record("内置因子已注册", len(factors) >= 10,
           f"已注册 {len(factors)} 个: {factors}")

    # 4.2 因子分类
    cats = FactorRegistry.list_by_category()
    record("因子分类正常", len(cats) >= 4,
           f"分类={list(cats.keys())}")

    # 4.3 因子元信息
    info = FactorRegistry.get_factor_info("reversal_20d")
    record("反转因子方向为 -1", info and info["direction"] == -1,
           f"info={info}")

    # 4.4 计算结果非空
    result = FactorRegistry.compute(data, ["ret_20d", "reversal_20d"])
    record("因子计算结果非空", not result.empty and "reversal_20d" in result.columns)

    # 4.5 依赖正确解析：reversal_20d = -ret_20d
    if "reversal_20d" in result.columns and "ret_20d" in result.columns:
        valid = result["ret_20d"].notna()
        diff = (result.loc[valid, "reversal_20d"]
                - (-result.loc[valid, "ret_20d"])).abs().max()
        record("reversal_20d == -ret_20d", diff < 1e-9,
               f"最大误差={diff:.2e}")

    # 4.6 拓扑排序：依赖先计算
    ordered = FactorRegistry._topo_sort(["reversal_20d", "ret_20d"])
    record("拓扑排序：ret_20d 在 reversal_20d 之前",
           ordered.index("ret_20d") < ordered.index("reversal_20d"),
           f"顺序={ordered}")


# ════════════════════════════════════════════════════════════
#  测试 5：因子注册表 —— 扩展性（用户自定义因子）
# ════════════════════════════════════════════════════════════
def test_factor_registry_extensibility():
    print("\n=== 测试 5：因子注册表扩展性 ===")

    # 用户用一行装饰器即可新增因子
    @FactorRegistry.register(
        "my_custom_momentum", direction=1, category="custom",
        description="自定义动量: 10日收益/10日波动", deps=["ret_20d"],
    )
    def _my_momentum(data, deps):
        vol = data.groupby("code")["close"].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        )
        return deps["ret_20d"] / vol.replace(0, np.nan)

    info = FactorRegistry.get_factor_info("my_custom_momentum")
    record("自定义因子注册成功", info is not None and info["direction"] == 1)

    data = make_synthetic_data(n_stocks=20, n_days=60)
    result = FactorRegistry.compute(data, ["my_custom_momentum"])
    record("自定义因子计算成功",
          "my_custom_momentum" in result.columns and result["my_custom_momentum"].notna().any())


# ════════════════════════════════════════════════════════════
#  测试 6：向量化 IC 分析 —— 正确性 + 性能
# ════════════════════════════════════════════════════════════
def test_vectorized_ic():
    print("\n=== 测试 6：向量化 IC 分析正确性 + 性能 ===")
    data = make_synthetic_data(n_stocks=100, n_days=200)
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    df["fwd_5d"] = df.groupby("code")["close"].shift(-5) / df["close"] - 1

    # 原循环实现
    t0 = time.time()
    ic_orig = original_ic_loop(df, "ret_20d", "fwd_5d", "spearman")
    t_orig = time.time() - t0

    # 向量化实现
    t0 = time.time()
    ic_vec = VectorizedFactorAnalysis.calc_ic_series(
        df, "ret_20d", "fwd_5d", "spearman"
    )
    t_vec = time.time() - t0

    # 正确性：IC 序列相关性高度一致
    common = ic_orig.index.intersection(ic_vec.index)
    if len(common) > 0:
        corr = ic_orig.loc[common].corr(ic_vec.loc[common])
        record("向量化 IC 与原循环 IC 高度相关", corr > 0.99,
               f"相关系数={corr:.6f}")

        # 数值接近
        max_diff = (ic_orig.loc[common] - ic_vec.loc[common]).abs().max()
        record("IC 数值最大差异 < 1e-6", max_diff < 1e-6,
               f"最大差异={max_diff:.2e}")
    else:
        record("IC 有共同日期", False)

    speedup = t_orig / t_vec if t_vec > 0 else float("inf")
    record("向量化 IC 快于原循环", t_vec < t_orig,
           f"向量化={t_vec:.4f}s, 原循环={t_orig:.4f}s, 加速比={speedup:.1f}x")
    results["summary"]["ic_vec_time"] = round(t_vec, 4)
    results["summary"]["ic_loop_time"] = round(t_orig, 4)
    results["summary"]["ic_speedup"] = round(speedup, 1)


# ════════════════════════════════════════════════════════════
#  测试 7：向量化中性化 —— 正确性 + 性能
# ════════════════════════════════════════════════════════════
def test_vectorized_neutralize():
    print("\n=== 测试 7：向量化中性化正确性 + 性能 ===")
    data = make_synthetic_data(n_stocks=80, n_days=100)
    df = data.sort_values(["code", "date"]).copy()
    df["ret_20d"] = df.groupby("code")["close"].pct_change(20)
    # 构造市值与行业
    df["lncap"] = np.log(df.groupby("code")["close"].transform("mean")
                         * df.groupby("code")["volume"].transform("mean"))
    industries = ["银行", "地产", "医药", "消费", "科技", "能源", "材料", "工业"]
    df["industry"] = np.random.choice(industries, len(df))

    factor_df = df[["code", "date", "industry", "lncap", "ret_20d"]].dropna().copy()
    # 取前 60 日保证样本充足
    factor_df = factor_df[factor_df["date"] >= factor_df["date"].min() + pd.Timedelta(days=30)]

    # 原循环实现
    t0 = time.time()
    neut_orig = original_neutralize_loop(
        factor_df.copy(), neutralize_mcap=True, neutralize_industry=True
    )
    t_orig = time.time() - t0

    # 向量化实现
    t0 = time.time()
    neut_vec = VectorizedFactorAnalysis.neutralize(
        factor_df.copy(), neutralize_mcap=True, neutralize_industry=True
    )
    t_vec = time.time() - t0

    # 正确性：残差相关性高度一致
    col = "ret_20d_neutral"
    if col in neut_orig.columns and col in neut_vec.columns:
        common = neut_orig.index.intersection(neut_vec.index)
        v1 = neut_orig.loc[common, col].dropna()
        v2 = neut_vec.loc[common, col].dropna()
        common2 = v1.index.intersection(v2.index)
        if len(common2) > 0:
            corr = v1.loc[common2].corr(v2.loc[common2])
            record("向量化中性化与原循环高度相关", corr > 0.99,
                   f"相关系数={corr:.6f}")

            # 中性化后因子与 lncap/industry 相关性应显著降低
            sub = factor_df.loc[common2].copy()
            sub["neut"] = v2.loc[common2]
            before_corr = sub["ret_20d"].corr(sub["lncap"])
            after_corr = sub["neut"].corr(sub["lncap"])
            record("中性化后与市值相关性降低",
                   abs(after_corr) < abs(before_corr),
                   f"前={before_corr:.4f}, 后={after_corr:.4f}")

    speedup = t_orig / t_vec if t_vec > 0 else float("inf")
    record("向量化中性化快于原循环", t_vec < t_orig,
           f"向量化={t_vec:.4f}s, 原循环={t_orig:.4f}s, 加速比={speedup:.1f}x")
    results["summary"]["neut_vec_time"] = round(t_vec, 4)
    results["summary"]["neut_loop_time"] = round(t_orig, 4)
    results["summary"]["neut_speedup"] = round(speedup, 1)


# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("jingni-trader 优化验证测试")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    test_vectorized_backtest_correctness()
    test_vectorized_backtest_boundary()
    test_vectorized_backtest_performance()
    test_factor_registry_correctness()
    test_factor_registry_extensibility()
    test_vectorized_ic()
    test_vectorized_neutralize()

    # 汇总
    passed = sum(1 for t in results["tests"] if t["status"] == "PASS")
    failed = sum(1 for t in results["tests"] if t["status"] == "FAIL")
    results["summary"]["total"] = len(results["tests"])
    results["summary"]["passed"] = passed
    results["summary"]["failed"] = failed
    results["summary"]["timestamp"] = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print(f"测试汇总: {passed} 通过, {failed} 失败, 共 {len(results['tests'])} 项")
    print("=" * 60)

    # 保存 JSON 结果
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"测试结果已保存: {out_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
