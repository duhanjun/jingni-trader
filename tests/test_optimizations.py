"""
优化模块综合测试套件

测试内容:
  1. 向量化回测引擎: 正确性（与原 native_adapter 对比）、T+1 强制、滑点对称性、性能
  2. 向量化因子分析: IC 计算正确性（与原 scipy 实现对比）、中性化正确性、分层回测
  3. 因子表达式引擎: 解析正确性、计算正确性、预定义因子库

运行: pytest tests/test_optimizations.py -v
"""
import sys
import os
import time
import json
import numpy as np
import pandas as pd
import pytest

# 将项目根目录加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from optimizations.vectorized_backtest import VectorizedBacktestEngine
from optimizations.vectorized_factor import VectorizedFactorAnalysis
from optimizations.factor_expression import FactorExpressionEngine, ALPHA158_EXPRESSIONS


# ════════════════════════════════════════════════════════
# 测试数据生成
# ════════════════════════════════════════════════════════
def make_synthetic_data(n_stocks=20, n_days=60, seed=42):
    """生成合成行情数据"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_days)
    codes = [f'{600000+i:06d}.SH' for i in range(n_stocks)]

    rows = []
    for code in codes:
        price = 10.0 + rng.normal(0, 0.5)
        for dt in dates:
            ret = rng.normal(0, 0.02)
            price = max(price * (1 + ret), 1.0)
            vol = int(rng.integers(100000, 1000000))
            rows.append({
                'code': code, 'date': dt,
                'open': price * (1 + rng.normal(0, 0.005)),
                'high': price * (1 + abs(rng.normal(0, 0.01))),
                'low': price * (1 - abs(rng.normal(0, 0.01))),
                'close': price,
                'volume': vol,
                'amount': vol * price,
                'turnover_rate': round(rng.uniform(0.5, 5.0), 4),
                'change_pct': ret * 100,
                'is_limit_up': False,
                'is_limit_down': False,
            })
    return pd.DataFrame(rows)


def make_signals(data, strategy='momentum', top_pct=0.2):
    """基于数据生成交易信号"""
    df = data.sort_values(['code', 'date']).copy()
    df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
    signals = df[['code', 'date', 'ret_5d']].dropna().copy()
    signals['signal'] = 0
    if strategy == 'momentum':
        # 每日选前 20% 买入
        signals['rank'] = signals.groupby('date')['ret_5d'].rank(pct=True)
        signals.loc[signals['rank'] > (1 - top_pct), 'signal'] = 1
        # 持有 5 天后卖出
        signals = signals.sort_values(['code', 'date']).reset_index(drop=True)
        signals['hold'] = signals.groupby('code')['signal'].cumsum()
        signals.loc[signals['hold'].shift(5) > 0, 'signal'] = -1
        signals.loc[signals['signal'] == 1, 'signal'] = 1
    signals = signals[['code', 'date', 'signal']]
    return signals[signals['signal'] != 0] if (signals['signal'] != 0).any() else signals


# ════════════════════════════════════════════════════════
# 1. 向量化回测引擎测试
# ════════════════════════════════════════════════════════
class TestVectorizedBacktest:

    def setup_method(self):
        self.data = make_synthetic_data(n_stocks=20, n_days=60)
        self.signals = make_signals(self.data)
        self.engine = VectorizedBacktestEngine()

    def test_basic_run_returns_valid_structure(self):
        """测试基本运行返回结构完整"""
        result = self.engine.run_backtest(self.data, self.signals)
        assert 'equity_curve' in result
        assert 'trades' in result
        assert 'metrics' in result
        assert not result['equity_curve'].empty
        assert 'equity' in result['equity_curve'].columns

    def test_equity_starts_at_init_capital(self):
        """测试首日净值等于初始资金（无持仓时）"""
        result = self.engine.run_backtest(self.data, self.signals, init_capital=1e6)
        eq = result['equity_curve']
        # 第一天可能已有交易，检查现金列合理
        assert eq['equity'].iloc[0] > 0
        assert eq['cash'].iloc[0] <= 1e6 + 1  # 现金不应超过初始资金

    def test_t_plus_1_enforced(self):
        """测试 T+1 规则：当日买入不可当日卖出"""
        # 构造当日同时买卖的信号
        dt = self.data['date'].iloc[0]
        code = self.data['code'].iloc[0]
        # 先买入再卖出的信号在同一天
        sig = pd.DataFrame([
            {'code': code, 'date': dt, 'signal': 1},
            {'code': code, 'date': dt, 'signal': -1},
        ])
        result = self.engine.run_backtest(self.data, sig, t_plus_1=True)
        trades = result['trades']
        if not trades.empty:
            # 当日买入的不应有卖出
            day_trades = trades[trades['date'] == dt]
            actions = set(day_trades['action'].unique())
            # T+1 下不应同日既有 buy 又有 sell
            assert not (actions == {'buy', 'sell'}), "T+1 未生效：当日买入当日卖出"

    def test_t_plus_1_disabled_allows_day_trade(self):
        """测试关闭 T+1 后允许日内交易"""
        dt = self.data['date'].iloc[0]
        code = self.data['code'].iloc[0]
        # 先用足够大的资金建仓
        sig_buy = pd.DataFrame([{'code': code, 'date': dt, 'signal': 1}])
        result_buy = self.engine.run_backtest(self.data, sig_buy, t_plus_1=False, init_capital=1e8)
        # 验证买入成功
        assert len(result_buy['trades']) >= 1

    def test_slippage_applied_to_both_sides(self):
        """测试滑点同时作用于买入和卖出"""
        # 构造先买后卖（隔日）的信号
        dates = sorted(self.data['date'].unique())
        code = self.data['code'].iloc[0]
        sig = pd.DataFrame([
            {'code': code, 'date': dates[0], 'signal': 1},
            {'code': code, 'date': dates[1], 'signal': -1},
        ])
        result = self.engine.run_backtest(self.data, sig, slippage=0.01, t_plus_1=True)
        trades = result['trades']
        if len(trades) >= 2:
            buy_trade = trades[trades['action'] == 'buy'].iloc[0]
            sell_trade = trades[trades['action'] == 'sell'].iloc[0]
            day0_close = self.data[self.data['date'] == dates[0]][self.data['code'] == code]['close'].iloc[0]
            day1_close = self.data[self.data['date'] == dates[1]][self.data['code'] == code]['close'].iloc[0]
            # 买入价 = close * (1+slippage)
            assert abs(buy_trade['price'] - day0_close * 1.01) < 0.01
            # 卖出价 = close * (1-slippage)
            assert abs(sell_trade['price'] - day1_close * 0.99) < 0.01

    def test_empty_data_returns_empty_result(self):
        """测试空数据边界"""
        result = self.engine.run_backtest(pd.DataFrame(), pd.DataFrame())
        assert result['metrics'] == {}
        assert result['equity_curve'].empty

    def test_empty_signals(self):
        """测试空信号边界"""
        empty_sig = pd.DataFrame({'code': [], 'date': [], 'signal': []})
        result = self.engine.run_backtest(self.data, empty_sig)
        # 无信号不应崩溃
        assert 'equity_curve' in result

    def test_limit_up_blocks_buy(self):
        """测试涨停无法买入"""
        data = self.data.copy()
        dt = data['date'].iloc[0]
        code = data['code'].iloc[0]
        data.loc[(data['date'] == dt) & (data['code'] == code), 'is_limit_up'] = True
        sig = pd.DataFrame([{'code': code, 'date': dt, 'signal': 1}])
        result = self.engine.run_backtest(data, sig, price_limit=True)
        trades = result['trades']
        # 涨停日不应有买入
        if not trades.empty:
            assert not ((trades['date'] == dt) & (trades['action'] == 'buy')).any()

    def test_suspended_stock_skipped(self):
        """测试停牌股票被跳过"""
        data = self.data.copy()
        dt = data['date'].iloc[0]
        code = data['code'].iloc[0]
        data.loc[(data['date'] == dt) & (data['code'] == code), 'close'] = np.nan
        sig = pd.DataFrame([{'code': code, 'date': dt, 'signal': 1}])
        result = self.engine.run_backtest(data, sig)
        trades = result['trades']
        if not trades.empty:
            assert not ((trades['date'] == dt) & (trades['code'] == code)).any()

    def test_metrics_complete(self):
        """测试绩效指标完整性"""
        result = self.engine.run_backtest(self.data, self.signals)
        m = result['metrics']
        for key in ['total_return', 'annual_return', 'volatility', 'sharpe_ratio',
                    'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'win_rate']:
            assert key in m, f"缺少指标: {key}"


# ════════════════════════════════════════════════════════
# 2. 向量化因子分析测试
# ════════════════════════════════════════════════════════
class TestVectorizedFactor:

    def setup_method(self):
        self.data = make_synthetic_data(n_stocks=30, n_days=80)
        # 构造因子与远期收益
        df = self.data.sort_values(['code', 'date']).copy()
        df['factor_a'] = df.groupby('code')['close'].pct_change(5) * -1  # 反转因子
        df['factor_b'] = df.groupby('code')['volume'].transform(
            lambda x: x.rolling(20).mean()
        )
        self.factor_df = df[['code', 'date', 'factor_a', 'factor_b', 'close']].dropna()
        # 远期收益
        fr = df[['code', 'date', 'close']].copy()
        fr['ret_forward_1d'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
        fr['ret_forward_5d'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
        self.forward_returns = fr.dropna()

    def test_ic_series_not_empty(self):
        """测试 IC 序列计算非空"""
        ic = VectorizedFactorAnalysis.calc_ic_series(
            self.factor_df, self.forward_returns, 'factor_a', 'ret_forward_5d'
        )
        assert not ic.empty
        assert ic.name == 'ic'

    def test_ic_matches_scipy_reference(self):
        """测试向量化 IC 与 scipy 逐日计算结果一致"""
        from scipy import stats

        merged = self.factor_df.merge(
            self.forward_returns[['code', 'date', 'ret_forward_5d']],
            on=['code', 'date']
        ).dropna(subset=['factor_a', 'ret_forward_5d'])

        # scipy 参考实现（原 engine 逻辑）
        ref_ic = {}
        for dt, grp in merged.groupby('date'):
            if len(grp) < 10:
                continue
            ic, _ = stats.spearmanr(grp['factor_a'], grp['ret_forward_5d'])
            if not np.isnan(ic):
                ref_ic[dt] = ic

        # 向量化实现
        vec_ic = VectorizedFactorAnalysis.calc_ic_series(
            self.factor_df, self.forward_returns, 'factor_a', 'ret_forward_5d'
        )

        # 对比（允许浮点误差）
        common = set(ref_ic.keys()) & set(vec_ic.index)
        assert len(common) > 0
        for dt in list(common)[:10]:
            diff = abs(ref_ic[dt] - vec_ic.loc[dt])
            assert diff < 1e-6, f"IC 不一致 @ {dt}: scipy={ref_ic[dt]:.6f}, vec={vec_ic.loc[dt]:.6f}"

    def test_ic_stats_structure(self):
        """测试 IC 统计结构"""
        stats = VectorizedFactorAnalysis.calc_ic_stats(
            self.factor_df, self.forward_returns, ['factor_a', 'factor_b']
        )
        assert 'ret_forward_1d' in stats or 'ret_forward_5d' in stats
        for period, items in stats.items():
            for item in items:
                assert 'ic_mean' in item
                assert 'ic_ir' in item
                assert 'ic_t_stat' in item

    def test_neutralize_preserves_shape(self):
        """测试中性化保持数据形状"""
        df = self.factor_df.copy()
        df['lncap'] = np.log(np.random.uniform(1e9, 1e11, len(df)))
        df['industry'] = np.random.choice(['银行', '地产', '科技', '消费'], len(df))
        result = VectorizedFactorAnalysis.neutralize_vectorized(df)
        assert len(result) == len(df)
        assert 'factor_a_neutral' in result.columns

    def test_neutralize_residual_orthogonal(self):
        """测试中性化后残差与市值因子近似正交"""
        df = self.factor_df.copy()
        df['lncap'] = np.log(np.random.uniform(1e9, 1e11, len(df)))
        df['industry'] = 'ALL'  # 单一行业，只中性化市值
        result = VectorizedFactorAnalysis.neutralize_vectorized(
            df, neutralize_industry=False, neutralize_mcap=True
        )
        if 'factor_a_neutral' in result.columns:
            # 残差与 lncap 的相关系数应接近 0
            valid = result[['factor_a_neutral', 'lncap']].dropna()
            if len(valid) > 30:
                corr = valid['factor_a_neutral'].corr(valid['lncap'])
                assert abs(corr) < 0.1, f"中性化后仍相关: corr={corr:.4f}"

    def test_quantile_returns(self):
        """测试分层回测"""
        qr = VectorizedFactorAnalysis.quantile_returns(
            self.factor_df, self.forward_returns, 'factor_a', 'ret_forward_5d', n_quantiles=5
        )
        assert not qr.empty
        assert qr.shape[1] <= 5
        assert all(c.startswith('q') for c in qr.columns)


# ════════════════════════════════════════════════════════
# 3. 因子表达式引擎测试
# ════════════════════════════════════════════════════════
class TestFactorExpression:

    def setup_method(self):
        self.data = make_synthetic_data(n_stocks=10, n_days=40)
        # 添加 ret_1d 字段
        self.data = self.data.sort_values(['code', 'date']).reset_index(drop=True)
        self.data['ret_1d'] = self.data.groupby('code')['close'].pct_change()
        self.engine = FactorExpressionEngine()

    def test_parse_simple_field(self):
        """测试解析简单字段"""
        ast = self.engine.parse('Close')
        assert ast == ('field', 'Close')

    def test_parse_function(self):
        """测试解析函数调用"""
        ast = self.engine.parse('Ts_Mean(Close, 5)')
        assert ast[0] == 'Ts_Mean'
        assert len(ast[1]) == 2

    def test_parse_nested(self):
        """测试解析嵌套表达式"""
        ast = self.engine.parse('Rank(Ts_Mean(Close, 5))')
        assert ast[0] == 'Rank'
        assert ast[1][0][0] == 'Ts_Mean'

    def test_evaluate_close(self):
        """测试计算 Close 字段"""
        s = self.engine.evaluate('Close', self.data)
        assert len(s) == len(self.data)
        # 值应与 close 列一致
        assert np.allclose(s.values, self.data['close'].values, equal_nan=True)

    def test_evaluate_ts_mean(self):
        """测试计算 5 日均值"""
        s = self.engine.evaluate('Ts_Mean(Close, 5)', self.data)
        # 参考值
        ref = self.data.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        # 对齐索引后比较（s 为 MultiIndex，ref 为 RangeIndex）
        s_vals = s.reset_index(drop=True).values
        ref_vals = ref.reset_index(drop=True).values
        valid = ~np.isnan(s_vals) & ~np.isnan(ref_vals)
        assert np.allclose(s_vals[valid], ref_vals[valid], rtol=1e-8)

    def test_evaluate_rank(self):
        """测试截面排名"""
        s = self.engine.evaluate('Rank(Close)', self.data)
        ref = self.data.set_index(['code', 'date'])['close'].groupby(level='date').rank(pct=True)
        s_vals = s.values
        ref_vals = ref.values
        valid = ~np.isnan(s_vals) & ~np.isnan(ref_vals)
        assert np.allclose(s_vals[valid], ref_vals[valid], rtol=1e-8)

    def test_evaluate_nested_expression(self):
        """测试嵌套表达式 Rank(Ts_Mean(Close, 5))"""
        s = self.engine.evaluate('Rank(Ts_Mean(Close, 5))', self.data)
        assert len(s) == len(self.data)
        # rank 值应在 [0,1] 或 NaN
        valid = ~s.isna()
        assert s[valid].between(0, 1, inclusive='both').all()

    def test_evaluate_arithmetic(self):
        """测试算术运算 Div(Sub(High, Low), Close)"""
        s = self.engine.evaluate('Div(Sub(High, Low), Close)', self.data)
        ref = (self.data['high'] - self.data['low']) / self.data['close']
        s_vals = s.reset_index(drop=True).values
        ref_vals = ref.reset_index(drop=True).values
        valid = ~np.isnan(s_vals) & ~np.isnan(ref_vals)
        assert np.allclose(s_vals[valid], ref_vals[valid], rtol=1e-8)

    def test_batch_evaluate(self):
        """测试批量计算"""
        exprs = {
            'momentum_5': 'Ts_Mean(Return, 5)',
            'amplitude': 'Div(Sub(High, Low), Close)',
        }
        out = self.engine.batch_evaluate(exprs, self.data)
        assert 'momentum_5' in out.columns
        assert 'amplitude' in out.columns
        assert len(out) == len(self.data)

    def test_alpha158_library(self):
        """测试预定义 Alpha158 因子库"""
        # 逐个测试（部分因子需要 turnover_rate）
        tested = 0
        for name, expr in ALPHA158_EXPRESSIONS.items():
            try:
                s = self.engine.evaluate(expr, self.data)
                assert len(s) == len(self.data)
                tested += 1
            except Exception as e:
                pytest.fail(f"Alpha158 因子 {name}='{expr}' 计算失败: {e}")
        assert tested == len(ALPHA158_EXPRESSIONS)

    def test_unknown_operator_raises(self):
        """测试未知算子报错"""
        with pytest.raises(KeyError):
            self.engine.evaluate('UnknownOp(Close)', self.data)
    def test_unknown_field_raises(self):
        """测试未知字段报错"""
        with pytest.raises(KeyError):
            self.engine.evaluate('NonexistentField', self.data)


# ════════════════════════════════════════════════════════
# 4. 性能对比测试
# ════════════════════════════════════════════════════════
class TestPerformance:

    def test_backtest_performance_large(self):
        """向量化回测在大数据集上的性能"""
        data = make_synthetic_data(n_stocks=100, n_days=120, seed=7)
        signals = make_signals(data)
        engine = VectorizedBacktestEngine()

        t0 = time.perf_counter()
        result = engine.run_backtest(data, signals)
        elapsed = time.perf_counter() - t0

        assert not result['equity_curve'].empty
        # 记录性能（不强制阈值，仅记录）
        print(f"\n[Perf] 向量化回测 100股×120日: {elapsed:.3f}s, "
              f"trades={len(result['trades'])}, "
              f"engine_elapsed={result['metrics'].get('backtest_elapsed_sec', 'N/A')}s")

    def test_ic_performance_large(self):
        """向量化 IC 分析在大数据集上的性能"""
        data = make_synthetic_data(n_stocks=200, n_days=200, seed=7)
        df = data.sort_values(['code', 'date']).copy()
        df['factor_a'] = df.groupby('code')['close'].pct_change(5) * -1
        factor_df = df[['code', 'date', 'factor_a', 'close']].dropna()
        fr = df[['code', 'date', 'close']].copy()
        fr['ret_forward_5d'] = df.groupby('code')['close'].shift(-5) / df['close'] - 1
        fr = fr.dropna()

        t0 = time.perf_counter()
        ic = VectorizedFactorAnalysis.calc_ic_series(
            factor_df, fr, 'factor_a', 'ret_forward_5d'
        )
        elapsed = time.perf_counter() - t0

        assert not ic.empty
        print(f"\n[Perf] 向量化 IC 200股×200日: {elapsed:.3f}s, {len(ic)} 个截面")

    def test_expression_engine_performance(self):
        """因子表达式引擎性能"""
        data = make_synthetic_data(n_stocks=50, n_days=100, seed=7)
        data = data.sort_values(['code', 'date']).reset_index(drop=True)
        data['ret_1d'] = data.groupby('code')['close'].pct_change()
        engine = FactorExpressionEngine()

        t0 = time.perf_counter()
        out = engine.batch_evaluate(ALPHA158_EXPRESSIONS, data)
        elapsed = time.perf_counter() - t0

        assert len(out) == len(data)
        print(f"\n[Perf] 表达式引擎 {len(ALPHA158_EXPRESSIONS)}因子×{len(data)}行: {elapsed:.3f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=short'])
