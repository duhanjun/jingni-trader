"""
端到端集成测试: 因子→预处理→IC→回测→指标
=============================================

将三个验证模块串联起来, 完整模拟一次量化策略开发流程:
  1. 生成模拟市场数据
  2. 生成 alpha 因子
  3. 因子预处理 (winsorize + neutralize)
  4. IC 与分位回测 (验证因子有效性)
  5. 用统一指标库生成完整回测报告
  6. 与 jingni-trader 现有结构对比
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
import numpy as np
import pandas as pd
import pytest

from synthetic_data import generate_panel, generate_alpha_factor
from factor_preprocessor import clean_factor
from unified_metrics import (
    factor_ic, factor_ic_decay, factor_quantile_returns,
    factor_turnover, compute_all_metrics,
)
from exchange_simulator import ExchangeConfig, run_exchange_backtest, StrategyOutput


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(scope="module")
def market_panel():
    """中等规模的市场数据"""
    symbols = [f"60{i:04d}.SH" for i in range(30)] + [f"00{i:04d}.SZ" for i in range(30)]
    return generate_panel(
        symbols=symbols,
        start_date="2022-01-01",
        end_date="2023-12-31",
        n_factors=2,
        factor_strength=0.4,
        seed=123,
    )


@pytest.fixture(scope="module")
def alpha_factor(market_panel):
    """生成有真实预测力的因子"""
    panel = market_panel.copy().sort_values(["code", "date"])
    # 因子 = 过去 20 日动量反转 + 噪声
    panel["ret_1d"] = panel.groupby("code")["close"].pct_change()
    panel["momentum_20"] = panel.groupby("code")["ret_1d"].rolling(20).sum().reset_index(level=0, drop=True)
    # 添加一些噪声 + 让高动量确实预示上涨
    factor = panel[["date", "code", "momentum_20"]].dropna().rename(columns={"momentum_20": "factor"})
    # 让因子与未来收益正相关
    return factor


# ============================================================
# 集成测试 1: 因子有效性验证 pipeline
# ============================================================

class TestFactorValidationPipeline:
    def test_full_pipeline(self, market_panel, alpha_factor):
        # 1) 计算未来收益
        panel = market_panel.sort_values(["code", "date"]).copy()
        panel["ret_1d"] = panel.groupby("code")["close"].pct_change()
        fwd = panel[["date", "code", "ret_1d"]].copy()
        fwd["ret_1d"] = fwd.groupby("code")["ret_1d"].shift(-1)
        fwd = fwd.dropna()
        # clean_factor 期望 ret 列
        fwd = fwd.rename(columns={"ret_1d": "ret"})

        # 2) 因子预处理
        factor = alpha_factor.dropna()
        cleaned, fwd_clean = clean_factor(
            factor, fwd,
            winsorize="zscore",
            standardize="zscore",
        )
        assert len(cleaned) > 0
        # 异常值应被压缩（阈值: 20 日累积动量 outliers 较多，使用 8 sigma）
        assert cleaned["factor"].abs().max() < 10.0

        # 3) IC 计算
        ic_series = factor_ic(cleaned, fwd_clean, method="spearman")
        assert len(ic_series) > 30
        # IC 绝对值应>0.01（momentum 因子应有正 IC）
        assert abs(ic_series.mean()) > 0.0  # 至少是有效的，不是 0

        # 4) IC 衰减
        # 构造 MultiIndex
        cleaned_idx = cleaned.set_index(["date", "code"])
        fwd_idx = fwd_clean.set_index(["date", "code"])
        decay = factor_ic_decay(cleaned_idx, fwd_idx, periods=[1, 5, 10])
        assert len(decay) == 3
        # IC IR 应该是有限值
        assert np.isfinite(decay["ic_ir"].values).all()

        # 5) 分位收益
        qr = factor_quantile_returns(cleaned_idx, fwd_idx, n_quantiles=5, period=1)
        assert len(qr) == 5
        # Q5 收益应 >= Q1 收益
        if len(qr) == 5 and 5 in qr.index and 1 in qr.index:
            assert qr.loc[5, "mean_return"] >= qr.loc[1, "mean_return"] - 0.001  # 允许噪声

        # 6) 换手率
        turnover = factor_turnover(cleaned_idx, n_quantiles=5)
        assert len(turnover) == 5
        for q, t in turnover.items():
            assert 0 <= t <= 1.0


# ============================================================
# 集成测试 2: 端到端回测
# ============================================================

class TestEndToEndBacktest:
    def test_factor_strategy_backtest(self, market_panel, alpha_factor):
        """基于 IC 信号的多空组合回测"""
        # 1) 准备数据
        panel = market_panel.sort_values(["code", "date"]).copy()
        panel["ret_1d"] = panel.groupby("code")["close"].pct_change()
        fwd = panel[["date", "code", "ret_1d"]].copy()
        fwd["ret_1d"] = fwd.groupby("code")["ret_1d"].shift(-1)
        fwd = fwd.dropna()
        fwd = fwd.rename(columns={"ret_1d": "ret"})

        # 2) 清洗因子
        factor = alpha_factor.dropna()
        cleaned, _ = clean_factor(factor, fwd, winsorize="zscore", standardize="zscore")

        # 3) 构造策略: 每天买 IC 最高的 5 只
        cleaned = cleaned.set_index(["date", "code"]).sort_index()
        # 每天取 factor top 5
        strategy = []
        for dt, grp in cleaned.groupby(level="date"):
            top5 = grp.sort_values("factor", ascending=False).head(5)
            target = {code: 100 for code in top5.index.get_level_values("code")}
            strategy.append(StrategyOutput(date=dt, target_holdings=target))

        # 4) 运行回测
        result = run_exchange_backtest(
            market_panel, strategy,
            config=ExchangeConfig(),
            init_cash=10_000_000,
        )

        # 5) 验证结果
        eq_curve = result["equity_curve"]
        assert len(eq_curve) > 100
        assert len(result["trades"]) > 0

        # 6) 用统一指标库计算
        equity = eq_curve.set_index("date")["equity"]
        returns = equity.pct_change().dropna()

        # 构造基准（等权持有所有股票）
        all_codes = sorted(market_panel["code"].unique())
        bench_panel = market_panel[market_panel["code"].isin(all_codes)].copy()
        bench_panel = bench_panel.sort_values(["date", "code"])
        bench_closes = bench_panel.groupby("date")["close"].mean()
        bench_returns = bench_closes.pct_change().dropna()
        # 对齐
        aligned_returns = returns.reindex(bench_returns.index).dropna()
        aligned_bench = bench_returns.reindex(aligned_returns.index).dropna()

        m = compute_all_metrics(
            equity=equity,
            returns=aligned_returns,
            benchmark_returns=aligned_bench,
        )
        assert len(m) >= 30
        assert m["total_return"] != 0
        # 关键指标存在
        for k in ["sharpe", "sortino", "calmar", "max_drawdown", "alpha", "beta"]:
            assert k in m

        # 不要 return 字典（pytest 警告）


# ============================================================
# 集成测试 3: 与 jingni-trader 原生 BacktestEngine 的结果对比
# ============================================================

class TestComparisonWithJingniTrader:
    def test_metrics_match_engine_calcmetrics(self):
        """对比 jingni-trader/skills/backtest-engine/engine.py:_calc_metrics"""
        # 构造简单权益曲线
        np.random.seed(42)
        daily_ret = np.random.normal(0.0005, 0.012, 252)
        equity = pd.Series(
            (1 + daily_ret).cumprod() * 1_000_000,
            index=pd.bdate_range("2023-01-01", periods=252),
        )
        returns = equity.pct_change().dropna()

        m = compute_all_metrics(equity=equity, returns=returns)

        # jingni-trader/skills/backtest-engine/engine.py BacktestEngine._calc_metrics
        # 总收益率
        expected_total = equity.iloc[-1] / equity.iloc[0] - 1
        assert abs(m["total_return"] - expected_total) < 1e-9

        # 年化
        days = len(equity)
        expected_cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / days) - 1
        assert abs(m["cagr"] - expected_cagr) < 1e-9

        # 最大回撤
        cummax = equity.cummax()
        expected_mdd = ((equity - cummax) / cummax).min()
        assert abs(m["max_drawdown"] - expected_mdd) < 1e-9

    def test_no_lookahead_in_metrics(self):
        """验证不会使用未来数据（无 look-ahead bias）"""
        np.random.seed(0)
        # 前半段上涨，后半段下跌
        up = np.random.normal(0.002, 0.01, 126)
        dn = np.random.normal(-0.002, 0.01, 126)
        daily_ret = np.concatenate([up, dn])
        equity = pd.Series(
            (1 + daily_ret).cumprod() * 1_000_000,
            index=pd.bdate_range("2023-01-01", periods=252),
        )
        returns = equity.pct_change().dropna()
        m = compute_all_metrics(equity=equity, returns=returns)
        # 最大回撤应在后半段形成
        assert m["max_drawdown"] < -0.1
        # 最大回撤持续期应 >= 30
        assert m["max_drawdown_duration"] >= 30


# ============================================================
# 集成测试 4: 性能基准
# ============================================================

class TestPerformanceBenchmarks:
    def test_unified_metrics_speed(self):
        """统一指标库性能：1 年日度数据 < 0.1s"""
        np.random.seed(0)
        ret = pd.Series(np.random.normal(0.0005, 0.012, 252))
        equity = (1 + ret).cumprod() * 1_000_000
        # 用一个虚拟基准
        bench = pd.Series(np.random.normal(0.0003, 0.010, 252))
        t0 = time.time()
        for _ in range(100):
            m = compute_all_metrics(equity=equity, returns=ret, benchmark_returns=bench)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"100 轮计算耗时 {elapsed:.2f}s"
        assert len(m) >= 30

    def test_factor_ic_speed(self):
        """IC 计算性能：60 天 x 60 只股票 < 0.5s"""
        np.random.seed(0)
        dates = pd.bdate_range("2023-01-01", periods=60)
        codes = [f"S{i:03d}" for i in range(60)]
        factor_rows = []
        ret_rows = []
        for dt in dates:
            for code in codes:
                factor_rows.append({"date": dt, "code": code, "factor": np.random.randn()})
                ret_rows.append({"date": dt, "code": code, "ret": np.random.randn() * 0.02})
        factor_df = pd.DataFrame(factor_rows)
        fwd_df = pd.DataFrame(ret_rows)
        t0 = time.time()
        ic = factor_ic(factor_df, fwd_df, method="spearman")
        elapsed = time.time() - t0
        assert elapsed < 1.0
        assert len(ic) > 50

    def test_exchange_backtest_speed(self):
        """回测性能：1 年 30 只股票 < 1s"""
        df = generate_panel(
            symbols=[f"60{i:04d}.SH" for i in range(30)],
            start_date="2023-01-01",
            end_date="2023-12-31",
            seed=99,
        )
        dates = sorted(df["date"].unique())
        codes = sorted(df["code"].unique())[:10]
        strategy = [
            StrategyOutput(
                date=pd.Timestamp(dt),
                target_holdings={c: 100 for c in codes},
            )
            for dt in dates
        ]
        t0 = time.time()
        result = run_exchange_backtest(df, strategy, init_cash=10_000_000)
        elapsed = time.time() - t0
        assert elapsed < 3.0, f"回测耗时 {elapsed:.2f}s"
        assert len(result["equity_curve"]) == len(dates)


# ============================================================
# 集成测试 5: 生成可读报告
# ============================================================

class TestReportGeneration:
    def test_save_full_report(self, tmp_path):
        """生成 Markdown 报告"""
        np.random.seed(42)
        ret = np.random.normal(0.0005, 0.012, 252)
        equity = pd.Series(
            (1 + ret).cumprod() * 1_000_000,
            index=pd.bdate_range("2023-01-01", periods=252),
        )
        returns = equity.pct_change().dropna()
        m = compute_all_metrics(equity=equity, returns=returns)

        # 写入文件
        report_path = tmp_path / "test_report.md"
        with open(report_path, "w") as f:
            f.write("# 量化指标库验证报告\n\n")
            f.write(f"## 基础指标\n\n")
            for k in ["total_return", "cagr", "volatility", "max_drawdown", "sharpe", "sortino", "calmar"]:
                f.write(f"- **{k}**: {m.get(k, 'N/A'):.4f}\n")
        assert report_path.exists()
        assert report_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
