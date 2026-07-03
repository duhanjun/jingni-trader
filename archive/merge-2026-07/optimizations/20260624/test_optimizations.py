"""
验证测试：三大优化点的正确性、性能对比、边界条件

测试覆盖：
  1. Exchange 防前视回测引擎
     - 正确性：涨跌停/停牌/T+1/成交量容量/冲击成本/费用
     - 性能对比：与现有 native_adapter 对比
     - 边界：空数据/单只股票/资金不足
  2. 因子表达式引擎
     - 正确性：表达式解析/算子计算/与硬编码对比
     - 性能：批量计算耗时
     - 边界：未知字段/无效表达式/空数据
  3. 三重约束因子筛选器
     - 正确性：IC门槛/容量门槛/低冗余贪心去重
     - 对比：与现有 correlation_analysis 对比
     - 边界：全不达标/全冗余/单因子
"""
from __future__ import annotations

import os
import sys
import time
import json
import traceback
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# 确保能 import 优化模块
OPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OPT_DIR)

from exchange import (
    Exchange, Account, Lot, TradeRecord,
    PRICE_OPEN, PRICE_CLOSE, LIMIT_THRESHOLD_NORMAL, LIMIT_THRESHOLD_ST,
    run_backtest_with_exchange,
)
from factor_expression import (
    ExpressionParser, OperatorRegistry, FactorExpressionEngine,
    FactorFamily, Feature, Constant, Ref, Mean, Std, Add, Sub, Mul, Div,
    ElemOperator,
)
from factor_screener import (
    ThreeConstraintScreener, FactorMetrics, FactorLineage, compare_with_legacy,
)

# 测试结果收集
TEST_RESULTS: List[Dict[str, Any]] = []


def record(name: str, passed: bool, detail: str = "", **extra):
    TEST_RESULTS.append({
        "name": name, "passed": passed, "detail": detail, **extra
    })
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if detail and not passed:
        print(f"         {detail}")


# ════════════════════════════════════════════════════════════
# 合成数据生成
# ════════════════════════════════════════════════════════════
def gen_synthetic_data(
    n_codes: int = 20, n_days: int = 60, seed: int = 42
) -> pd.DataFrame:
    """生成合成日线数据（含 OHLCV、涨跌停标记、停牌标记）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    rows = []
    for code in codes:
        price = 10.0
        for dt in dates:
            ret = rng.normal(0, 0.02)
            pre_close = price
            open_p = price * (1 + rng.normal(0, 0.005))
            close = price * (1 + ret)
            high = max(open_p, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_p, close) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.uniform(1e6, 1e8))
            amount = volume * (open_p + close) / 2
            turnover_rate = rng.uniform(0.005, 0.05)

            # 涨跌停标记
            change = close / pre_close - 1 if pre_close > 0 else 0
            is_limit_up = change >= 0.099
            is_limit_down = change <= -0.099
            # 偶尔停牌（5% 概率）
            is_suspended = rng.random() < 0.02

            rows.append({
                "code": code, "date": dt,
                "open": round(open_p, 3), "high": round(high, 3),
                "low": round(low, 3), "close": round(close, 3),
                "pre_close": round(pre_close, 3),
                "volume": volume, "amount": round(amount, 2),
                "turnover_rate": round(turnover_rate, 6),
                "change_pct": round(ret, 6),
                "is_limit_up": is_limit_up, "is_limit_down": is_limit_down,
                "is_suspended": is_suspended,
                "is_st": False,
                "lncap": round(np.log(amount / turnover_rate * 100), 4) if turnover_rate > 0 else np.nan,
            })
            price = close

    df = pd.DataFrame(rows)
    # 停牌日 close 置 None（Qlib 约定）
    df.loc[df["is_suspended"], "close"] = np.nan
    df.loc[df["is_suspended"], "open"] = np.nan
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def gen_signals(data: pd.DataFrame, top_pct: float = 0.2) -> pd.DataFrame:
    """基于动量生成简单买卖信号（T 日收盘后生成）"""
    signals = []
    for code, grp in data.groupby("code"):
        grp = grp.sort_values("date").copy()
        grp["mom_5d"] = grp["close"].shift(5) / grp["close"] - 1
        # 反转策略：5 日跌幅大的买入
        grp["signal"] = 0
        grp.loc[grp["mom_5d"] < grp["mom_5d"].quantile(0.2), "signal"] = 1
        signals.append(grp[["code", "date", "signal"]])
    return pd.concat(signals).dropna(subset=["signal"]).reset_index(drop=True)


def gen_factor_data(n_codes: int = 50, n_days: int = 100, seed: int = 7) -> pd.DataFrame:
    """生成合成因子数据（含多个有/无预测力的因子）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    industries = ["银行", "地产", "医药", "电子", "消费"]

    rows = []
    for code in codes:
        industry = rng.choice(industries)
        for dt in dates:
            # 真实 Alpha 因子（与未来收益正相关）
            true_alpha = rng.normal(0, 1)
            forward_ret = 0.05 * true_alpha + rng.normal(0, 0.02)
            # 伪 Alpha（高 IC 但来自市值暴露）
            size_factor = np.log(1e9 + rng.uniform(0, 1e10))
            rows.append({
                "code": code, "date": dt, "industry": industry,
                "lncap": size_factor,
                "amount": rng.uniform(1e7, 1e9),
                # 有效因子
                "good_factor_1": true_alpha + rng.normal(0, 0.1),
                "good_factor_2": true_alpha * 0.8 + rng.normal(0, 0.2),
                # 冗余因子（与 good_factor_1 高相关）
                "redundant_factor": true_alpha + rng.normal(0, 0.05),
                # 无效因子（纯噪声）
                "noise_factor": rng.normal(0, 1),
                # 高换手因子（IC 达标但容量不足）
                "high_turnover_factor": rng.choice([-1, 1], p=[0.5, 0.5]) * true_alpha,
                # 前瞻收益
                "ret_forward_5d": forward_ret,
            })
    return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════
# 测试 1：Exchange 防前视回测引擎
# ════════════════════════════════════════════════════════════
def test_exchange_correctness():
    """正确性测试：涨跌停/停牌/T+1/成交量容量/冲击成本"""
    print("\n=== 测试组 1: Exchange 正确性 ===")

    # 1.1 涨跌停拒绝交易
    try:
        ex = Exchange(deal_price=PRICE_CLOSE)
        # 构造涨停行
        row = pd.Series({
            "open": 11.0, "close": 11.0, "pre_close": 10.0,
            "volume": 1e7, "is_limit_up": True, "is_limit_down": False,
            "is_suspended": False,
        })
        tradable, reason = ex.is_stock_tradable("600000.SH", row, "buy")
        record("涨停拒绝买入", not tradable and "涨停" in reason, f"reason={reason}")
    except Exception as e:
        record("涨停拒绝买入", False, str(e))

    # 1.2 跌停拒绝卖出
    try:
        row = pd.Series({
            "open": 9.0, "close": 9.0, "pre_close": 10.0,
            "volume": 1e7, "is_limit_up": False, "is_limit_down": True,
            "is_suspended": False,
        })
        tradable, reason = ex.is_stock_tradable("600000.SH", row, "sell")
        record("跌停拒绝卖出", not tradable and "跌停" in reason, f"reason={reason}")
    except Exception as e:
        record("跌停拒绝卖出", False, str(e))

    # 1.3 停牌拒绝交易（close 为 NaN）
    try:
        row = pd.Series({
            "open": np.nan, "close": np.nan, "pre_close": 10.0,
            "volume": 0, "is_limit_up": False, "is_limit_down": False,
            "is_suspended": True,
        })
        tradable, reason = ex.is_stock_tradable("600000.SH", row, "buy")
        record("停牌拒绝交易", not tradable and "停牌" in reason, f"reason={reason}")
    except Exception as e:
        record("停牌拒绝交易", False, str(e))

    # 1.4 T+1 约束：买入当日不可卖出
    try:
        ex = Exchange(t_plus_1=True, deal_price=PRICE_CLOSE)
        account = Account(1e6, ex)
        row = pd.Series({
            "open": 10.0, "close": 10.0, "pre_close": 10.0,
            "volume": 1e8, "is_limit_up": False, "is_limit_down": False,
            "is_suspended": False,
        })
        # T 日买入
        lot = account.buy("600000.SH", 50000, row, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
        bought = lot is not None
        # T 日尝试卖出（应被 T+1 拒绝）
        sold = account.sell("600000.SH", 1000, row, pd.Timestamp("2024-01-01"))
        record("T+1 当日买入不可卖", bought and sold == 0, f"bought={bought}, sold={sold}")
    except Exception as e:
        record("T+1 当日买入不可卖", False, str(e))

    # 1.5 T+1 次日可卖
    try:
        ex = Exchange(t_plus_1=True, deal_price=PRICE_CLOSE)
        account = Account(1e6, ex)
        row = pd.Series({
            "open": 10.0, "close": 10.0, "pre_close": 10.0,
            "volume": 1e8, "is_limit_up": False, "is_limit_down": False,
            "is_suspended": False,
        })
        lot = account.buy("600000.SH", 50000, row, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
        # T+1 日卖出（available_date=2024-01-02）
        sold = account.sell("600000.SH", 1000, row, pd.Timestamp("2024-01-02"))
        record("T+1 次日可卖", lot is not None and sold > 0, f"sold={sold}")
    except Exception as e:
        record("T+1 次日可卖", False, str(e))

    # 1.6 成交量容量裁剪
    try:
        ex = Exchange(volume_threshold_ratio=0.1, deal_price=PRICE_CLOSE)
        row = pd.Series({
            "open": 10.0, "close": 10.0, "pre_close": 10.0,
            "volume": 10000, "is_limit_up": False, "is_limit_down": False,
            "is_suspended": False,
        })
        # 目标买 10000 股，但当日成交量 10000，10% 上限 = 1000 股
        clipped = ex.clip_amount_by_volume(10000, row)
        record("成交量容量裁剪", clipped <= 1000, f"target=10000, clipped={clipped}, vol=10000")
    except Exception as e:
        record("成交量容量裁剪", False, str(e))

    # 1.7 冲击成本计算
    try:
        ex = Exchange(impact_cost_rate=0.001, deal_price=PRICE_CLOSE)
        commission, tax, transfer, impact = ex.calc_buy_cost(100000)
        record("冲击成本计算", impact == 100.0 and tax == 0.0, f"impact={impact}, tax={tax}")
    except Exception as e:
        record("冲击成本计算", False, str(e))

    # 1.8 卖出含印花税
    try:
        ex = Exchange(impact_cost_rate=0.0, deal_price=PRICE_CLOSE)
        commission, tax, transfer, impact = ex.calc_sell_cost(100000)
        record("卖出含印花税", tax == 100.0, f"tax={tax}")
    except Exception as e:
        record("卖出含印花税", False, str(e))

    # 1.9 最低手续费
    try:
        ex = Exchange(open_cost=0.00025, min_cost=5.0, deal_price=PRICE_CLOSE)
        commission, _, _, _ = ex.calc_buy_cost(1000)  # 金额 1000，佣金应为 5（最低）
        record("最低手续费 5 元", commission == 5.0, f"commission={commission}")
    except Exception as e:
        record("最低手续费 5 元", False, str(e))


def test_exchange_lookahead_bias():
    """防前视偏差对比测试：deal_price=open vs close"""
    print("\n=== 测试组 2: Exchange 防前视偏差 ===")

    data = gen_synthetic_data(n_codes=10, n_days=40, seed=123)
    signals = gen_signals(data)

    # 用开盘价成交（防前视）
    t0 = time.time()
    res_open = run_backtest_with_exchange(
        data, signals, deal_price=PRICE_OPEN, t_plus_1=True
    )
    t_open = time.time() - t0

    # 用收盘价成交（有前视，仅用于对比）
    t0 = time.time()
    res_close = run_backtest_with_exchange(
        data, signals, deal_price=PRICE_CLOSE, t_plus_1=True
    )
    t_close = time.time() - t0

    eq_open = res_open["equity_curve"]["equity"].iloc[-1]
    eq_close = res_close["equity_curve"]["equity"].iloc[-1]

    # 两者净值应不同（证明成交价语义生效）
    record(
        "开盘价 vs 收盘价成交净值不同",
        abs(eq_open - eq_close) > 1,
        f"open_eq={eq_open:.0f}, close_eq={eq_close:.0f}"
    )
    # 开盘价成交耗时合理
    record("开盘价回测性能 < 5s", t_open < 5.0, f"耗时={t_open:.3f}s")


def test_exchange_boundary():
    """边界条件测试"""
    print("\n=== 测试组 3: Exchange 边界条件 ===")

    # 3.1 空数据
    try:
        res = run_backtest_with_exchange(
            pd.DataFrame(columns=["code", "date", "open", "close", "volume"]),
            pd.DataFrame(columns=["code", "date", "signal"]),
        )
        record("空数据不崩溃", res["equity_curve"].empty, "")
    except Exception as e:
        record("空数据不崩溃", False, str(e))

    # 3.2 资金不足
    try:
        ex = Exchange(deal_price=PRICE_CLOSE)
        account = Account(1000, ex)  # 只有 1000 元
        row = pd.Series({
            "open": 100.0, "close": 100.0, "pre_close": 100.0,
            "volume": 1e8, "is_limit_up": False, "is_limit_down": False,
            "is_suspended": False,
        })
        lot = account.buy("600000.SH", 50000, row, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))
        record("资金不足拒绝买入", lot is None, f"cash=1000, target=50000")
    except Exception as e:
        record("资金不足拒绝买入", False, str(e))

    # 3.3 全停牌
    try:
        data = gen_synthetic_data(n_codes=5, n_days=10, seed=99)
        data["is_suspended"] = True
        data["close"] = np.nan
        data["open"] = np.nan
        # 全停牌时无法生成有效信号，直接构造一个信号
        signals = pd.DataFrame({
            "code": ["600000.SH"], "date": [data["date"].iloc[5]], "signal": [1]
        })
        res = run_backtest_with_exchange(data, signals, deal_price=PRICE_OPEN)
        # 全停牌应无有效成交（shares 为 0 或 trades 为空）
        no_trade = len(res["trades"]) == 0 or (res["trades"]["shares"].sum() == 0)
        record("全停牌无成交", no_trade, "")
    except Exception as e:
        record("全停牌无成交", False, str(e))


# ════════════════════════════════════════════════════════════
# 测试 2：因子表达式引擎
# ════════════════════════════════════════════════════════════
def test_expression_parser():
    """表达式解析正确性"""
    print("\n=== 测试组 4: 表达式解析 ===")

    parser = ExpressionParser()

    # 4.1 字段引用
    try:
        ast = parser.parse("$close")
        record("解析字段引用", isinstance(ast, Feature) and ast.field == "close", str(ast))
    except Exception as e:
        record("解析字段引用", False, str(e))

    # 4.2 常量
    try:
        ast = parser.parse("20")
        record("解析常量", isinstance(ast, Constant) and ast.value == 20.0, str(ast))
    except Exception as e:
        record("解析常量", False, str(e))

    # 4.3 函数调用
    try:
        ast = parser.parse("Ref($close, 20)")
        record("解析函数调用", isinstance(ast, Ref) and ast.window == 20, str(ast))
    except Exception as e:
        record("解析函数调用", False, str(e))

    # 4.4 二元运算
    try:
        ast = parser.parse("Ref($close, 20) / $close")
        record("解析二元运算", isinstance(ast, Div), str(ast))
    except Exception as e:
        record("解析二元运算", False, str(e))

    # 4.5 嵌套表达式
    try:
        ast = parser.parse("Mean(Ref($close, 5), 20)")
        record("解析嵌套表达式", isinstance(ast, Mean) and isinstance(ast.operand, Ref), str(ast))
    except Exception as e:
        record("解析嵌套表达式", False, str(e))

    # 4.6 序列化往返
    try:
        expr_str = "Corr($close, $volume, 20)"
        ast = parser.parse(expr_str)
        record("序列化往返", str(ast) == expr_str, f"原={expr_str}, 序列化={ast}")
    except Exception as e:
        record("序列化往返", False, str(e))


def test_expression_compute():
    """因子计算正确性（与硬编码对比）"""
    print("\n=== 测试组 5: 因子计算正确性 ===")

    data = gen_synthetic_data(n_codes=10, n_days=60, seed=42)
    engine = FactorExpressionEngine()

    # 5.1 Ref 计算正确性
    try:
        factor = engine.compute_factor(data, "Ref($close, 5)")
        # 手动计算
        expected = data.groupby("code")["close"].shift(5)
        aligned = pd.concat([factor.reset_index(drop=True), expected.reset_index(drop=True)], axis=1).dropna()
        correct = np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])
        record("Ref 计算正确", correct, "")
    except Exception as e:
        record("Ref 计算正确", False, str(e))

    # 5.2 Mean 计算正确性
    try:
        factor = engine.compute_factor(data, "Mean($close, 10)")
        expected = data.groupby("code")["close"].transform(lambda x: x.rolling(10, min_periods=5).mean())
        aligned = pd.concat([factor.reset_index(drop=True), expected.reset_index(drop=True)], axis=1).dropna()
        correct = np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])
        record("Mean 计算正确", correct, "")
    except Exception as e:
        record("Mean 计算正确", False, str(e))

    # 5.3 二元运算：20 日收益率
    try:
        factor = engine.compute_factor(data, "Ref($close, 20) / $close - 1")
        expected = data.groupby("code")["close"].shift(20) / data["close"] - 1
        aligned = pd.concat([factor.reset_index(drop=True), expected.reset_index(drop=True)], axis=1).dropna()
        correct = np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])
        record("二元运算(收益率)正确", correct, "")
    except Exception as e:
        record("二元运算(收益率)正确", False, str(e))

    # 5.4 Std 波动率
    try:
        factor = engine.compute_factor(data, "Std($close / Ref($close, 1) - 1, 20)")
        rets = data.groupby("code")["close"].pct_change()
        expected = rets.groupby(data["code"]).transform(lambda x: x.rolling(20, min_periods=10).std())
        aligned = pd.concat([factor.reset_index(drop=True), expected.reset_index(drop=True)], axis=1).dropna()
        correct = np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], equal_nan=True)
        record("Std 波动率正确", correct, "")
    except Exception as e:
        record("Std 波动率正确", False, str(e))


def test_expression_extended_window():
    """get_extended_window_size 边界声明正确性"""
    print("\n=== 测试组 6: 回看窗口声明 ===")

    parser = ExpressionParser()

    # Ref($close, 20) 需要 20 天回看
    try:
        ast = parser.parse("Ref($close, 20)")
        left, right = ast.get_extended_window_size()
        record("Ref 回看窗口", left == 20, f"left={left}")
    except Exception as e:
        record("Ref 回看窗口", False, str(e))

    # Mean(Ref($close, 5), 20) 需要 5+20=25 天回看
    try:
        ast = parser.parse("Mean(Ref($close, 5), 20)")
        left, right = ast.get_extended_window_size()
        record("嵌套回看窗口累加", left == 25, f"left={left}")
    except Exception as e:
        record("嵌套回看窗口累加", False, str(e))


def test_expression_family():
    """因子族批量计算"""
    print("\n=== 测试组 7: 因子族批量计算 ===")

    data = gen_synthetic_data(n_codes=10, n_days=60, seed=42)
    engine = FactorExpressionEngine()

    try:
        family = FactorFamily.alpha158_lite()
        factor_df, metadata = engine.compute_family(data, family)
        total_factors = sum(len(v) for v in family.values())
        record(
            "因子族批量计算",
            len(factor_df.columns) > 2 and len(metadata) == total_factors,
            f"列数={len(factor_df.columns)}, 因子数={len(metadata)}, 期望={total_factors}"
        )
        # 检查元信息含回看窗口
        has_extend = all("left_extend" in m for m in metadata if "error" not in m)
        record("元信息含回看窗口", has_extend, "")
    except Exception as e:
        record("因子族批量计算", False, str(e))
        traceback.print_exc()


def test_expression_performance():
    """性能测试"""
    print("\n=== 测试组 8: 表达式引擎性能 ===")

    data = gen_synthetic_data(n_codes=50, n_days=120, seed=42)
    engine = FactorExpressionEngine()

    try:
        family = FactorFamily.alpha158_lite()
        t0 = time.time()
        factor_df, _ = engine.compute_family(data, family)
        elapsed = time.time() - t0
        n_factors = sum(len(v) for v in family.values())
        record(
            "25 因子 × 50 股 × 120 日 < 10s",
            elapsed < 10.0,
            f"耗时={elapsed:.3f}s, 因子数={n_factors}"
        )
    except Exception as e:
        record("性能测试", False, str(e))


def test_expression_boundary():
    """边界条件"""
    print("\n=== 测试组 9: 表达式边界条件 ===")

    parser = ExpressionParser()

    # 未知字段
    try:
        data = gen_synthetic_data(n_codes=5, n_days=10, seed=1)
        engine = FactorExpressionEngine()
        factor = engine.compute_factor(data, "$unknown_field")
        record("未知字段报错", False, "应抛异常但未抛")
    except KeyError:
        record("未知字段报错", True, "")
    except Exception as e:
        record("未知字段报错", False, f"异常类型错误: {type(e).__name__}")

    # 无效表达式
    try:
        parser.parse("Ref($close, )")
        record("无效表达式报错", False, "应抛异常但未抛")
    except SyntaxError:
        record("无效表达式报错", True, "")
    except Exception as e:
        record("无效表达式报错", False, f"异常类型错误: {type(e).__name__}")

    # 自定义算子注册
    try:
        class DoubleUp(ElemOperator):
            def _apply(self, series):
                return series * 2
        OperatorRegistry.register("DoubleUp", DoubleUp)
        ast = parser.parse("DoubleUp($close)")
        data = gen_synthetic_data(n_codes=3, n_days=5, seed=1)
        val = ast.load(data)
        expected = data["close"] * 2
        correct = np.allclose(val.values, expected.values)
        record("自定义算子注册", correct, "")
    except Exception as e:
        record("自定义算子注册", False, str(e))


# ════════════════════════════════════════════════════════════
# 测试 3：三重约束因子筛选器
# ════════════════════════════════════════════════════════════
def test_screener_correctness():
    """三重约束筛选正确性"""
    print("\n=== 测试组 10: 三重约束筛选 ===")

    factor_df = gen_factor_data(n_codes=50, n_days=100, seed=7)
    factor_cols = [
        "good_factor_1", "good_factor_2", "redundant_factor",
        "noise_factor", "high_turnover_factor",
    ]

    screener = ThreeConstraintScreener(
        min_rank_ic=0.05, min_ic_ir=0.3, min_ic_t_stat=2.0,
        max_correlation=0.7, max_turnover=0.95, min_liquidity=1e6,
    )

    try:
        all_metrics, lineages = screener.screen(
            factor_df, factor_cols, "ret_forward_5d",
            industry_col="industry", mcap_col="lncap", amount_col="amount",
        )

        # 10.1 噪声因子应被 IC 门槛过滤
        noise_m = next((m for m in all_metrics if m.name == "noise_factor"), None)
        record("噪声因子被 IC 过滤", noise_m is not None and not noise_m.pass_ic, 
               f"IC={noise_m.rank_ic_mean:.4f}" if noise_m else "")

        # 10.2 有效因子应通过 IC
        good1_m = next((m for m in all_metrics if m.name == "good_factor_1"), None)
        record("有效因子通过 IC", good1_m is not None and good1_m.pass_ic,
               f"IC={good1_m.rank_ic_mean:.4f}" if good1_m else "")

        # 10.3 冗余因子对去重：good_factor_1 与 redundant_factor 高相关，至多一个入库
        good1_m = next((m for m in all_metrics if m.name == "good_factor_1"), None)
        redundant_m = next((m for m in all_metrics if m.name == "redundant_factor"), None)
        both_selected = (
            good1_m is not None and good1_m.selected
            and redundant_m is not None and redundant_m.selected
        )
        record("冗余因子对去重", not both_selected,
               f"good1_selected={good1_m.selected if good1_m else None}, "
               f"redundant_selected={redundant_m.selected if redundant_m else None}")

        # 10.4 入库因子数量合理（1-3 个）
        selected_names = [m.name for m in all_metrics if m.selected]
        record("入库因子数量合理", 1 <= len(selected_names) <= 4,
               f"selected={selected_names}")

        # 10.5 谱系记录存在
        record("谱系记录存在", len(lineages) == len(selected_names) and len(lineages) > 0,
               f"lineages={len(lineages)}")
    except Exception as e:
        record("三重约束筛选", False, str(e))
        traceback.print_exc()


def test_screener_neutral_ic():
    """中性化 IC（A 股增强）"""
    print("\n=== 测试组 11: 中性化 IC ===")

    factor_df = gen_factor_data(n_codes=50, n_days=80, seed=11)
    screener = ThreeConstraintScreener(use_neutral_ic=True)

    try:
        m = screener.evaluate_factor(
            factor_df, "good_factor_1", "ret_forward_5d",
            industry_col="industry", mcap_col="lncap", amount_col="amount",
        )
        record(
            "中性化 IC 计算",
            not np.isnan(m.neutral_ic_mean),
            f"raw_IC={m.rank_ic_mean:.4f}, neutral_IC={m.neutral_ic_mean:.4f}"
        )
    except Exception as e:
        record("中性化 IC 计算", False, str(e))


def test_screener_vs_legacy():
    """与现有 correlation_analysis 对比"""
    print("\n=== 测试组 12: 与现有方法对比 ===")

    factor_df = gen_factor_data(n_codes=50, n_days=100, seed=7)
    factor_cols = [
        "good_factor_1", "good_factor_2", "redundant_factor",
        "noise_factor", "high_turnover_factor",
    ]

    try:
        comparison = compare_with_legacy(factor_df, factor_cols, "ret_forward_5d", max_correlation=0.7)

        # 现有方法不过滤噪声因子（仅相关性去冗余）
        legacy_keeps_noise = "noise_factor" in comparison["legacy_selected"]
        # 三重约束过滤噪声
        new_drops_noise = "noise_factor" not in comparison["new_selected"]

        record(
            "三重约束过滤噪声(现有方法不过滤)",
            legacy_keeps_noise and new_drops_noise,
            f"legacy_keeps_noise={legacy_keeps_noise}, new_drops_noise={new_drops_noise}"
        )
        record(
            "对比报告生成",
            "legacy_count" in comparison and "new_count" in comparison,
            f"legacy={comparison['legacy_count']}, new={comparison['new_count']}"
        )
    except Exception as e:
        record("与现有方法对比", False, str(e))
        traceback.print_exc()


def test_screener_boundary():
    """边界条件"""
    print("\n=== 测试组 13: 筛选器边界条件 ===")

    # 13.1 单因子
    try:
        factor_df = gen_factor_data(n_codes=30, n_days=50, seed=3)
        screener = ThreeConstraintScreener(min_rank_ic=0.001, min_ic_ir=0.01, min_ic_t_stat=0.5)
        all_metrics, lineages = screener.screen(factor_df, ["good_factor_1"], "ret_forward_5d")
        record("单因子筛选不崩溃", len(all_metrics) == 1, "")
    except Exception as e:
        record("单因子筛选不崩溃", False, str(e))

    # 13.2 全不达标
    try:
        factor_df = gen_factor_data(n_codes=30, n_days=50, seed=3)
        screener = ThreeConstraintScreener(min_rank_ic=0.5, min_ic_ir=5.0)  # 极高门槛
        all_metrics, lineages = screener.screen(
            factor_df, ["good_factor_1", "noise_factor"], "ret_forward_5d"
        )
        record("全不达标返回空", len(lineages) == 0, f"selected={len(lineages)}")
    except Exception as e:
        record("全不达标返回空", False, str(e))


# ════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("jingni-trader 优化验证测试")
    print("分支: feat/quant-opt-20260624")
    print("日期: 2026-06-24")
    print("=" * 60)

    test_exchange_correctness()
    test_exchange_lookahead_bias()
    test_exchange_boundary()

    test_expression_parser()
    test_expression_compute()
    test_expression_extended_window()
    test_expression_family()
    test_expression_performance()
    test_expression_boundary()

    test_screener_correctness()
    test_screener_neutral_ic()
    test_screener_vs_legacy()
    test_screener_boundary()

    # 汇总
    print("\n" + "=" * 60)
    total = len(TEST_RESULTS)
    passed = sum(1 for r in TEST_RESULTS if r["passed"])
    failed = total - passed
    print(f"测试汇总: {passed}/{total} 通过, {failed} 失败")
    print("=" * 60)

    # 保存结果
    result_path = os.path.join(OPT_DIR, "test_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(TEST_RESULTS, f, ensure_ascii=False, indent=2, default=str)
    print(f"测试结果已保存: {result_path}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
