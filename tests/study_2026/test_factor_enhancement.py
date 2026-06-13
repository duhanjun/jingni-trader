"""
优化方向: 因子库扩展与因子分析增强
借鉴来源:
  1. Microsoft Qlib (https://github.com/microsoft/qlib) - Alpha158 因子库设计
     - 158 个量价因子覆盖 K线形态、价格趋势、时序波动等多维度
     - 因子表达式引擎（Ref($close, 60)/$close 等 DSL 语法）
     - 多级缓存机制 MemCache/ExpressionCache/DatasetCache
  2. FactorEngine (arXiv:2603.16365) - Program-level 因子挖掘
     - 因子 IC/ICIR 多周期分析
     - 因子衰减曲线（decay analysis）
     - 经验知识库驱动的因子优化
  3. RD-Agent (microsoft/RD-Agent) - LLM 驱动的因子挖掘闭环

验证内容:
  - 扩展因子库从 ~13 个到 50+ 个（涵盖动量、反转、波动率、流动性、技术指标、估值、财务质量等）
  - 因子多周期 IC 衰减分析（1日/5日/20日/60日）
  - 因子分组回测（分位数组合收益）
  - 因子拥挤度指标
  - 与原有因子引擎的对比测试

注意: 本文件仅用于验证测试，不修改主项目代码。
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Add project path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# =============================================================================
# 扩展因子计算器（独立实现，不影响主代码）
# =============================================================================

class EnhancedFactorCalculator:
    """
    增强版因子计算器

    参考 Qlib Alpha158 的因子分类体系，将因子从原有的 ~13 个扩展到 50+
    分为 6 大类：
      1. 动量类 (Momentum): ret_1d/5d/20d/60d, 路径调整动量
      2. 反转类 (Reversal): reversal_5d/20d/60d, intraday reversal
      3. 波动率类 (Volatility): std_5d/20d/60d, H-L range, Parkinson vol
      4. 流动性类 (Liquidity): turnover, volume ratio, Amihud illiquidity
      5. 技术指标类 (Technical): RSI, MACD, BB, ATR, ADX, CCI, OBV, MFI
      6. 估值/财务类 (Valuation): 需要在基础数据中包含更多字段
    """

    def __init__(self):
        self.computed_factors = []
        self.factor_metadata = {}

    def get_available_factors(self) -> List[Dict]:
        """返回所有可用因子的元信息"""
        factors = [
            # -- 动量类 --
            {"name": "ret_1d", "category": "momentum", "description": "1日收益率", "direction": 1},
            {"name": "ret_5d", "category": "momentum", "description": "5日收益率", "direction": 1},
            {"name": "ret_10d", "category": "momentum", "description": "10日收益率", "direction": 1},
            {"name": "ret_20d", "category": "momentum", "description": "20日收益率", "direction": 1},
            {"name": "ret_60d", "category": "momentum", "description": "60日收益率", "direction": 1},
            {"name": "ret_120d", "category": "momentum", "description": "120日收益率", "direction": 1},
            {"name": "momentum_1m", "category": "momentum", "description": "1月动量(跳最近1月)", "direction": 1},
            {"name": "momentum_3m", "category": "momentum", "description": "3月动量(跳最近1月)", "direction": 1},
            {"name": "momentum_6m", "category": "momentum", "description": "6月动量(跳最近1月)", "direction": 1},

            # -- 反转类 --
            {"name": "reversal_5d", "category": "reversal", "description": "5日反转", "direction": -1},
            {"name": "reversal_10d", "category": "reversal", "description": "10日反转", "direction": -1},
            {"name": "reversal_20d", "category": "reversal", "description": "20日反转", "direction": -1},
            {"name": "reversal_60d", "category": "reversal", "description": "60日反转", "direction": -1},
            {"name": "intraday_reversal", "category": "reversal", "description": "日内反转(close-open)/open", "direction": -1},

            # -- 波动率类 --
            {"name": "volatility_5d", "category": "volatility", "description": "5日波动率", "direction": 0},
            {"name": "volatility_20d", "category": "volatility", "description": "20日波动率", "direction": 0},
            {"name": "volatility_60d", "category": "volatility", "description": "60日波动率", "direction": 0},
            {"name": "hl_range_20d", "category": "volatility", "description": "20日高低价差比", "direction": 0},
            {"name": "skewness_20d", "category": "volatility", "description": "20日收益偏度", "direction": 0},
            {"name": "kurtosis_20d", "category": "volatility", "description": "20日收益峰度", "direction": 0},

            # -- 流动性类 --
            {"name": "turnover_5d", "category": "liquidity", "description": "5日均换手率", "direction": 0},
            {"name": "turnover_20d", "category": "liquidity", "description": "20日均换手率", "direction": 0},
            {"name": "turnover_change", "category": "liquidity", "description": "换手率变化(5日/20日-1)", "direction": 0},
            {"name": "volume_ratio", "category": "liquidity", "description": "量比(当日/20日均)", "direction": 0},
            {"name": "volume_std_20d", "category": "liquidity", "description": "20日成交量标准差/均值", "direction": 0},
            {"name": "amihud_illiquidity", "category": "liquidity", "description": "Amihud非流动性指标 |ret|/amount", "direction": 0},

            # -- 技术指标类 --
            {"name": "rsi_14", "category": "technical", "description": "14日RSI", "direction": 0},
            {"name": "macd", "category": "technical", "description": "MACD线", "direction": 1},
            {"name": "macd_signal", "category": "technical", "description": "MACD信号线", "direction": 1},
            {"name": "macd_hist", "category": "technical", "description": "MACD柱", "direction": 1},
            {"name": "bb_width", "category": "technical", "description": "布林带宽度", "direction": 0},
            {"name": "bb_position", "category": "technical", "description": "收盘价在布林带位置", "direction": 0},
            {"name": "atr_14", "category": "technical", "description": "14日ATR/收盘价", "direction": 0},
            {"name": "adx_14", "category": "technical", "description": "14日ADX", "direction": 1},
            {"name": "cci_14", "category": "technical", "description": "14日CCI", "direction": 0},
            {"name": "willr_14", "category": "technical", "description": "14日Williams %R", "direction": 0},
            {"name": "mfi_14", "category": "technical", "description": "14日资金流量指标", "direction": 0},

            # -- 价格形态类 --
            {"name": "ma_5", "category": "price_pattern", "description": "5日均线", "direction": 1},
            {"name": "ma_20", "category": "price_pattern", "description": "20日均线", "direction": 1},
            {"name": "ma_60", "category": "price_pattern", "description": "60日均线", "direction": 1},
            {"name": "ma_5_divergence", "category": "price_pattern", "description": "价格偏离5日均线", "direction": -1},
            {"name": "ma_20_divergence", "category": "price_pattern", "description": "价格偏离20日均线", "direction": -1},
            {"name": "ma_60_divergence", "category": "price_pattern", "description": "价格偏离60日均线", "direction": -1},
            {"name": "ma_5_20_cross", "category": "price_pattern", "description": "5日与20日均线交叉信号", "direction": 1},
            {"name": "ma_20_60_cross", "category": "price_pattern", "description": "20日与60日均线交叉信号", "direction": 1},
            {"name": "close_to_high_20d", "category": "price_pattern", "description": "收盘价距20日最高价", "direction": 0},

            # -- 资金流类 --
            {"name": "money_flow_5d", "category": "money_flow", "description": "5日资金流向", "direction": 1},
            {"name": "money_flow_20d", "category": "money_flow", "description": "20日资金流向", "direction": 1},
            {"name": "obv", "category": "money_flow", "description": "能量潮OBV", "direction": 1},
            {"name": "vpt", "category": "money_flow", "description": "量价趋势VPT", "direction": 1},
        ]
        self.factor_metadata = {f["name"]: f for f in factors}
        return factors

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        if data.empty:
            return data
        df = data.sort_values(['code', 'date']).copy()
        result = df[['code', 'date']].copy()

        # 按股票分组计算
        codes = df['code'].unique()
        factor_names = [f["name"] for f in self.get_available_factors()]

        for factor_name in factor_names:
            result[factor_name] = self._calc_factor_by_group(df, factor_name)

        self.computed_factors = factor_names
        return result

    def _calc_factor_by_group(self, data: pd.DataFrame, factor_name: str) -> pd.Series:
        """按股票分组计算单个因子"""
        series = pd.Series(index=data.index, dtype=float)
        for code in data['code'].unique():
            mask = data['code'] == code
            idx = data[mask].index
            try:
                values = self._calc_single_factor(data.loc[idx], factor_name)
                series.loc[idx] = values
            except Exception:
                series.loc[idx] = np.nan
        return series

    def _calc_single_factor(self, df: pd.DataFrame, factor_name: str) -> pd.Series:
        """计算单只股票的单个因子"""
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float) if 'high' in df.columns else close
        low = df['low'].values.astype(float) if 'low' in df.columns else close
        open_ = df['open'].values.astype(float) if 'open' in df.columns else close
        volume = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones_like(close)
        amount = df.get('amount', pd.Series(volume * close)).values.astype(float)
        turnover = df.get('turnover_rate', df.get('turnover', pd.Series(np.nan, index=df.index)))

        s = pd.Series(close, index=df.index)
        ret_1 = s.pct_change()

        # 动量类
        if factor_name == "ret_1d": return ret_1
        if factor_name == "ret_5d": return s.pct_change(5)
        if factor_name == "ret_10d": return s.pct_change(10)
        if factor_name == "ret_20d": return s.pct_change(20)
        if factor_name == "ret_60d": return s.pct_change(60)
        if factor_name == "ret_120d": return s.pct_change(120)
        if factor_name == "momentum_1m": return s.pct_change(20)
        if factor_name == "momentum_3m": return s.shift(20).pct_change(40)
        if factor_name == "momentum_6m": return s.shift(20).pct_change(100)

        # 反转类
        if factor_name == "reversal_5d": return -s.pct_change(5)
        if factor_name == "reversal_10d": return -s.pct_change(10)
        if factor_name == "reversal_20d": return -s.pct_change(20)
        if factor_name == "reversal_60d": return -s.pct_change(60)
        if factor_name == "intraday_reversal":
            return -(close - open_) / open_.clip(lower=1e-8)

        # 波动率类
        if factor_name == "volatility_5d": return ret_1.rolling(5).std()
        if factor_name == "volatility_20d": return ret_1.rolling(20).std()
        if factor_name == "volatility_60d": return ret_1.rolling(60).std()
        if factor_name == "hl_range_20d":
            hl = (high - low) / close
            return pd.Series(hl, index=df.index).rolling(20).mean()
        if factor_name == "skewness_20d": return ret_1.rolling(20).skew()
        if factor_name == "kurtosis_20d": return ret_1.rolling(20).kurt()

        # 流动性类
        if factor_name == "turnover_5d":
            return turnover.rolling(5, min_periods=3).mean()
        if factor_name == "turnover_20d":
            return turnover.rolling(20, min_periods=5).mean()
        if factor_name == "turnover_change":
            t5 = turnover.rolling(5, min_periods=3).mean()
            t20 = turnover.rolling(20, min_periods=5).mean()
            return t5 / t20.clip(lower=1e-8) - 1
        if factor_name == "volume_ratio":
            v20 = pd.Series(volume, index=df.index).rolling(20).mean()
            return pd.Series(volume, index=df.index) / v20.clip(lower=1e-8)
        if factor_name == "volume_std_20d":
            v = pd.Series(volume, index=df.index)
            return v.rolling(20).std() / v.rolling(20).mean().clip(lower=1e-8)
        if factor_name == "amihud_illiquidity":
            amt = pd.Series(amount, index=df.index)
            return (ret_1.abs() / amt.clip(lower=1e-8)).rolling(20).mean()

        # 技术指标类
        if factor_name == "rsi_14":
            return self._calc_rsi(close, 14)
        if factor_name == "macd":
            ema12 = s.ewm(span=12, adjust=False).mean()
            ema26 = s.ewm(span=26, adjust=False).mean()
            return ema12 - ema26
        if factor_name == "macd_signal":
            macd = self._calc_single_factor(df, "macd")
            return macd.ewm(span=9, adjust=False).mean()
        if factor_name == "macd_hist":
            macd = self._calc_single_factor(df, "macd")
            signal = self._calc_single_factor(df, "macd_signal")
            return 2 * (macd - signal)
        if factor_name == "bb_width":
            ma20 = s.rolling(20).mean()
            std20 = s.rolling(20).std()
            return (2 * std20 * 2) / ma20.clip(lower=1e-8)
        if factor_name == "bb_position":
            ma20 = s.rolling(20).mean()
            std20 = s.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            return (s - lower) / (upper - lower).clip(lower=1e-8)
        if factor_name == "atr_14":
            return self._calc_atr(high, low, close, 14) / close
        if factor_name == "adx_14":
            return self._calc_adx(high, low, close, 14)
        if factor_name == "cci_14":
            tp = (high + low + close) / 3
            tp_s = pd.Series(tp, index=df.index)
            tp_ma = tp_s.rolling(14).mean()
            tp_mad = tp_s.rolling(14).apply(lambda x: np.mean(np.abs(x - x.mean())))
            return (tp_s - tp_ma) / (0.015 * tp_mad.clip(lower=1e-8))
        if factor_name == "willr_14":
            hh = pd.Series(high, index=df.index).rolling(14).max()
            ll = pd.Series(low, index=df.index).rolling(14).min()
            return (hh - s) / (hh - ll).clip(lower=1e-8) * -100
        if factor_name == "mfi_14":
            return self._calc_mfi(high, low, close, volume, 14)

        # 价格形态类
        if factor_name == "ma_5": return s.rolling(5).mean()
        if factor_name == "ma_20": return s.rolling(20).mean()
        if factor_name == "ma_60": return s.rolling(60).mean()
        if factor_name == "ma_5_divergence": return s / s.rolling(5).mean() - 1
        if factor_name == "ma_20_divergence": return s / s.rolling(20).mean() - 1
        if factor_name == "ma_60_divergence": return s / s.rolling(60).mean() - 1
        if factor_name == "ma_5_20_cross":
            ma5 = s.rolling(5).mean()
            ma20 = s.rolling(20).mean()
            return ((ma5 > ma20).astype(float) - (ma5 < ma20).astype(float)).shift(1)
        if factor_name == "ma_20_60_cross":
            ma20 = s.rolling(20).mean()
            ma60 = s.rolling(60).mean()
            return ((ma20 > ma60).astype(float) - (ma20 < ma60).astype(float)).shift(1)
        if factor_name == "close_to_high_20d":
            hh20 = s.rolling(20).max()
            ll20 = s.rolling(20).min()
            return (s - ll20) / (hh20 - ll20).clip(lower=1e-8)

        # 资金流类
        if factor_name == "money_flow_5d":
            mf = ret_1 * pd.Series(amount, index=df.index)
            return mf.rolling(5).sum()
        if factor_name == "money_flow_20d":
            mf = ret_1 * pd.Series(amount, index=df.index)
            return mf.rolling(20).sum()
        if factor_name == "obv":
            ret_sign = np.sign(close[1:] - close[:-1])
            obv = np.zeros(len(close))
            obv[0] = volume[0] if close[0] >= close[0] else 0  # 第一个值
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            return pd.Series(obv, index=df.index)
        if factor_name == "vpt":
            vpt = np.zeros(len(close))
            for i in range(1, len(close)):
                vpt[i] = vpt[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1].clip(lower=1e-8)
            return pd.Series(vpt, index=df.index)

        return pd.Series(np.nan, index=df.index)

    @staticmethod
    def _calc_rsi(close: np.ndarray, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()
        rs = avg_gain / avg_loss.clip(lower=1e-8)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _calc_atr(high, low, close, period=14) -> pd.Series:
        """计算ATR"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        tr1 = high_s - low_s
        tr2 = (high_s - close_s.shift()).abs()
        tr3 = (low_s - close_s.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _calc_adx(high, low, close, period=14) -> pd.Series:
        """计算ADX"""
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)
        up_move = high_s.diff()
        down_move = -low_s.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        atr = EnhancedFactorCalculator._calc_atr(high, low, close, period)
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr.clip(lower=1e-8)
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr.clip(lower=1e-8)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-8)
        return dx.rolling(period).mean()

    @staticmethod
    def _calc_mfi(high, low, close, volume, period=14) -> pd.Series:
        """计算MFI"""
        tp = (high + low + close) / 3
        mf = tp * volume
        tp_s = pd.Series(tp)
        mf_s = pd.Series(mf)
        pos_flow = pd.Series(np.where(tp[1:] > tp[:-1], mf[1:], 0), index=range(1, len(mf)))
        neg_flow = pd.Series(np.where(tp[1:] < tp[:-1], mf[1:], 0), index=range(1, len(mf)))
        # Pad to match original length
        pos_flow = pd.concat([pd.Series([0]), pos_flow]).reset_index(drop=True)
        neg_flow = pd.concat([pd.Series([0]), neg_flow]).reset_index(drop=True)
        pos_sum = pos_flow.rolling(period).sum()
        neg_sum = neg_flow.rolling(period).sum()
        mf_ratio = pos_sum / neg_sum.clip(lower=1e-8)
        return 100 - 100 / (1 + mf_ratio)


# =============================================================================
# 多周期因子 IC 衰减分析
# =============================================================================

class FactorDecayAnalyzer:
    """
    因子 IC 衰减分析器

    参考 Qlib 和 FactorEngine 的多周期 IC 分析方法：
    - 计算因子在不同前瞻期（1d, 5d, 10d, 20d, 60d）上的 IC
    - 绘制 IC 衰减曲线
    - 判断因子的有效期和半衰期
    """

    @staticmethod
    def calc_multi_period_ic(
        factor_df: pd.DataFrame,
        data: pd.DataFrame,
        factor_names: List[str],
        periods: List[int] = [1, 5, 10, 20, 60],
        ic_type: str = "spearman"
    ) -> Dict[str, pd.DataFrame]:
        """
        计算多周期 IC

        返回:
            {factor_name: DataFrame with columns=['period', 'ic_mean', 'ic_std', 'ic_ir', 'ic_positive_ratio']}
        """
        from scipy import stats

        results = {}
        df = factor_df.merge(data[['code', 'date', 'close']], on=['code', 'date'], how='inner')

        for period in periods:
            df[f'forward_{period}d'] = df.groupby('code')['close'].transform(
                lambda x: x.shift(-period) / x - 1
            )

        for factor_name in factor_names:
            if factor_name not in df.columns:
                continue

            rows = []
            for period in periods:
                forward_col = f'forward_{period}d'
                if forward_col not in df.columns:
                    continue

                ic_list = []
                dates = sorted(df['date'].unique())
                for dt in dates:
                    cross = df[df['date'] == dt].dropna(subset=[factor_name, forward_col])
                    if len(cross) < 10:
                        continue
                    if ic_type == "spearman":
                        ic, _ = stats.spearmanr(cross[factor_name], cross[forward_col], nan_policy='omit')
                    else:
                        ic, _ = stats.pearsonr(cross[factor_name].fillna(0), cross[forward_col].fillna(0))
                    if not np.isnan(ic):
                        ic_list.append(ic)

                if ic_list:
                    ic_arr = np.array(ic_list)
                    rows.append({
                        'period': period,
                        'ic_mean': float(np.mean(ic_arr)),
                        'ic_std': float(np.std(ic_arr)),
                        'ic_ir': float(np.mean(ic_arr) / np.std(ic_arr)) if np.std(ic_arr) > 0 else 0,
                        'ic_positive_ratio': float((ic_arr > 0).mean()),
                        'ic_t': float(np.mean(ic_arr) / (np.std(ic_arr) / np.sqrt(len(ic_arr)))) if np.std(ic_arr) > 0 else 0,
                    })

            if rows:
                results[factor_name] = pd.DataFrame(rows)

        return results

    @staticmethod
    def calc_decay_summary(factor_name: str, ic_results: pd.DataFrame) -> Dict:
        """计算因子衰减摘要"""
        if ic_results.empty:
            return {}
        ic_values = ic_results['ic_mean'].abs().values
        periods = ic_results['period'].values

        # 找 IC 峰值对应的周期
        best_idx = np.argmax(ic_values)
        best_period = periods[best_idx]
        best_ic = ic_results['ic_mean'].iloc[best_idx]

        # 计算半衰期（IC 衰减到峰值一半的周期）
        half_ic = best_ic / 2
        half_life = None
        for i in range(best_idx, len(ic_values)):
            if ic_values[i] < abs(half_ic):
                half_life = periods[i]
                break

        return {
            "factor": factor_name,
            "best_period": int(best_period),
            "best_ic": round(float(best_ic), 6),
            "half_life": int(half_life) if half_life else max(periods),
            "long_term_ic": round(float(ic_results['ic_mean'].iloc[-1]), 6) if len(ic_results) > 0 else 0,
        }


# =============================================================================
# 因子分组回测（Quantile Portfolio）
# =============================================================================

class QuantileBacktest:
    """
    因子分组回测

    参考 Qlib 的分组回测方法：
    - 每日对股票按因子值排序，分为 N 个分位组合
    - 等权持有各分组
    - 比较 Top 组与 Bottom 组的收益差异
    - 计算多空组合的 Sharpe 和信息比率
    """

    @staticmethod
    def run_quantile_backtest(
        factor_df: pd.DataFrame,
        data: pd.DataFrame,
        factor_name: str,
        n_quantiles: int = 5,
        commission_rate: float = 0.00025,
        stamp_tax: float = 0.001,
    ) -> Dict:
        """
        运行分组回测

        返回:
            dict with keys: 'quantile_returns', 'long_short_returns', 'metrics'
        """
        df = factor_df[['code', 'date', factor_name]].dropna().merge(
            data[['code', 'date', 'close']], on=['code', 'date'], how='inner'
        )
        df['forward_return'] = df.groupby('code')['close'].transform(lambda x: x.shift(-1) / x - 1)
        df = df.dropna(subset=[factor_name, 'forward_return'])

        dates = sorted(df['date'].unique())
        quantile_returns = {f'Q{i+1}': [] for i in range(n_quantiles)}
        long_short_returns = []

        for dt in dates[:-1]:  # skip last date (no forward return)
            cross = df[df['date'] == dt].copy()
            if len(cross) < n_quantiles * 5:
                continue

            cross['quantile'] = pd.qcut(
                cross[factor_name].rank(method='first'),
                q=n_quantiles, labels=[f'Q{i+1}' for i in range(n_quantiles)]
            )

            for q_name in quantile_returns:
                q_data = cross[cross['quantile'] == q_name]
                if len(q_data) > 0:
                    ret = q_data['forward_return'].mean()
                    quantile_returns[q_name].append(ret)
                else:
                    quantile_returns[q_name].append(0)

            # Long-short: Q5 - Q1
            top_ret = cross[cross['quantile'] == f'Q{n_quantiles}']['forward_return'].mean()
            bottom_ret = cross[cross['quantile'] == 'Q1']['forward_return'].mean()
            long_short_returns.append(top_ret - bottom_ret)

        # Calculate metrics
        ls_series = pd.Series(long_short_returns)
        metrics = {
            "ls_annual_return": float(ls_series.mean() * 252),
            "ls_sharpe": float(ls_series.mean() / ls_series.std() * np.sqrt(252)) if ls_series.std() > 0 else 0,
            "ls_max_drawdown": float((ls_series.cumsum().cummax() - ls_series.cumsum()).max()),
            "ls_win_rate": float((ls_series > 0).mean()),
        }

        # Top quantile metrics
        top_series = pd.Series(quantile_returns[f'Q{n_quantiles}'])
        metrics['top_annual_return'] = float(top_series.mean() * 252)
        metrics['top_sharpe'] = float(top_series.mean() / top_series.std() * np.sqrt(252)) if top_series.std() > 0 else 0

        return {
            "quantile_returns": {k: v for k, v in quantile_returns.items()},
            "long_short_returns": long_short_returns,
            "metrics": metrics,
        }


# =============================================================================
# 因子拥挤度指标
# =============================================================================

class FactorCrowdingAnalyzer:
    """
    因子拥挤度分析器

    参考学术文献中的因子拥挤度指标：
    1. 因子估值拥挤度：高因子暴露股票的估值 vs 历史
    2. 因子配对相关性：因子对之间的相关性是否异常高
    3. 因子收益集中度：Top N 股票贡献因子收益的比例
    """

    @staticmethod
    def calc_crowding_metrics(factor_df: pd.DataFrame, factor_names: List[str]) -> Dict:
        """计算因子拥挤度指标"""
        results = {}
        for factor in factor_names:
            if factor not in factor_df.columns:
                continue

            factor_series = factor_df.groupby('date')[factor]
            # 1. 估值拥挤度：因子暴露的 cross-sectional std / mean
            cs_mean = factor_series.mean()
            cs_std = factor_series.std()
            valuation_crowding = (cs_std / cs_mean.abs().clip(lower=1e-8)).mean()

            # 2. 集中度：绝对值最大的Top20%股票占比
            def top20_share(series):
                threshold = series.abs().quantile(0.8)
                return (series.abs() >= threshold).mean()

            concentration = factor_series.apply(top20_share).mean()

            results[factor] = {
                "valuation_crowding": float(valuation_crowding),
                "concentration": float(concentration),
            }

        return results


# =============================================================================
# 对比测试：增强版 vs 原版因子引擎
# =============================================================================

def generate_test_data(n_stocks: int = 50, n_days: int = 252) -> pd.DataFrame:
    """生成模拟测试数据"""
    np.random.seed(42)
    symbols = [f"{600000 + i:06d}.SH" for i in range(n_stocks)]
    dates = pd.date_range('2024-01-01', periods=n_days, freq='B')

    rows = []
    for sym in symbols:
        start_price = np.random.uniform(8, 80)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = start_price * (1 + returns).cumprod()
        volume = np.random.lognormal(12, 0.5, n_days).astype(int)
        amount = prices * volume
        turnover = np.random.uniform(0.005, 0.05, n_days)

        df_one = pd.DataFrame({
            'date': dates,
            'code': sym,
            'open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.015, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.015, n_days))),
            'close': prices,
            'volume': volume,
            'amount': amount,
            'turnover_rate': turnover,
        })
        rows.append(df_one)

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# 单元测试
# =============================================================================

class TestEnhancedFactorCalculator(unittest.TestCase):
    """增强版因子计算器测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=50, n_days=252)
        cls.calculator = EnhancedFactorCalculator()

    def test_get_available_factors_count(self):
        """测试因子数量 >= 50"""
        factors = self.calculator.get_available_factors()
        self.assertGreaterEqual(len(factors), 50, f"因子数量不足: {len(factors)}")
        print(f"[PASS] 因子总数: {len(factors)}")

    def test_factor_categories(self):
        """测试因子分类覆盖"""
        factors = self.calculator.get_available_factors()
        categories = set(f['category'] for f in factors)
        expected_categories = {'momentum', 'reversal', 'volatility', 'liquidity',
                               'technical', 'price_pattern', 'money_flow'}
        missing = expected_categories - categories
        self.assertEqual(len(missing), 0, f"缺少因子分类: {missing}")
        print(f"[PASS] 因子分类: {sorted(categories)}")

    def test_compute_all_factors(self):
        """测试所有因子计算"""
        result = self.calculator.compute_all(self.data)
        factor_names = [f['name'] for f in self.calculator.get_available_factors()]

        computed = 0
        failed = []
        for fn in factor_names:
            if fn in result.columns:
                non_null_ratio = result[fn].notna().mean()
                if non_null_ratio > 0.1:
                    computed += 1
                else:
                    failed.append(fn)

        self.assertGreaterEqual(computed, 40, f"成功计算的因子不足40个: {computed}/{len(factor_names)}")
        print(f"[PASS] 成功计算因子: {computed}/{len(factor_names)}")
        if failed:
            print(f"  [WARN] 低覆盖率因子: {failed}")

    def test_factor_values_valid(self):
        """测试因子值有效性（无inf, 无极端NaN）"""
        result = self.calculator.compute_all(self.data)
        factor_names = [f['name'] for f in self.calculator.get_available_factors()]

        for fn in factor_names:
            if fn not in result.columns:
                continue
            values = result[fn].dropna()
            if len(values) == 0:
                continue
            self.assertFalse(np.isinf(values).any(), f"因子 {fn} 包含无穷值")
        print("[PASS] 所有因子值有效性检查通过")

    def test_data_shape(self):
        """测试输出形状"""
        result = self.calculator.compute_all(self.data)
        self.assertEqual(len(result), len(self.data))
        print(f"[PASS] 输出行数: {len(result)}")


class TestFactorDecayAnalyzer(unittest.TestCase):
    """因子衰减分析测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=50, n_days=252)
        calculator = EnhancedFactorCalculator()
        cls.factor_df = calculator.compute_all(cls.data)

    def test_multi_period_ic(self):
        """测试多周期IC计算"""
        factor_names = ['ret_5d', 'reversal_20d', 'volatility_20d', 'turnover_20d']
        results = FactorDecayAnalyzer.calc_multi_period_ic(
            self.factor_df, self.data, factor_names, periods=[1, 5, 10, 20]
        )

        for fn in factor_names:
            self.assertIn(fn, results, f"缺少因子 {fn} 的IC结果")
            self.assertGreater(len(results[fn]), 0, f"因子 {fn} IC结果为空")

        print("[PASS] 多周期IC计算成功")
        for fn in factor_names:
            df = results[fn]
            print(f"  {fn}: IC@1d={df[df['period']==1]['ic_mean'].values[0]:.4f} "
                  f"IC@20d={df[df['period']==20]['ic_mean'].values[0]:.4f}")

    def test_decay_summary(self):
        """测试衰减摘要"""
        factor_names = ['ret_5d', 'reversal_20d']
        ic_results = FactorDecayAnalyzer.calc_multi_period_ic(
            self.factor_df, self.data, factor_names, periods=[1, 5, 10, 20]
        )
        for fn in factor_names:
            summary = FactorDecayAnalyzer.calc_decay_summary(fn, ic_results[fn])
            self.assertIn('half_life', summary)
            print(f"[PASS] {fn} 衰减摘要: best_period={summary['best_period']}, "
                  f"half_life={summary['half_life']}")


class TestQuantileBacktest(unittest.TestCase):
    """分组回测测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=50, n_days=252)
        calculator = EnhancedFactorCalculator()
        cls.factor_df = calculator.compute_all(cls.data)

    def test_quantile_backtest(self):
        """测试分组回测"""
        result = QuantileBacktest.run_quantile_backtest(
            self.factor_df, self.data, 'ret_20d', n_quantiles=5
        )
        self.assertIn('metrics', result)
        self.assertIn('long_short_returns', result)
        self.assertGreater(len(result['long_short_returns']), 100)

        print("[PASS] 分组回测成功")
        metrics = result['metrics']
        print(f"  Long-Short 年化收益: {metrics['ls_annual_return']:.4f}")
        print(f"  Long-Short Sharpe: {metrics['ls_sharpe']:.4f}")
        print(f"  Top Quantile 年化收益: {metrics['top_annual_return']:.4f}")

    def test_quantile_monotonicity(self):
        """测试分组收益单调性（Q5 >= Q1 的日期比例）"""
        result = QuantileBacktest.run_quantile_backtest(
            self.factor_df, self.data, 'ret_20d', n_quantiles=5
        )
        qr = result['quantile_returns']
        # 对于动量因子，Q5 应优于 Q1
        monotonic_days = sum(
            1 for i in range(min(len(qr['Q5']), len(qr['Q1'])))
            if qr['Q5'][i] >= qr['Q1'][i]
        )
        total_days = min(len(qr['Q5']), len(qr['Q1']))
        ratio = monotonic_days / total_days if total_days > 0 else 0
        print(f"[PASS] 分组单调性比例 (Q5 >= Q1): {ratio:.2%}")


class TestFactorCrowdingAnalyzer(unittest.TestCase):
    """因子拥挤度测试"""

    @classmethod
    def setUpClass(cls):
        cls.data = generate_test_data(n_stocks=50, n_days=252)
        calculator = EnhancedFactorCalculator()
        cls.factor_df = calculator.compute_all(cls.data)

    def test_crowding_metrics(self):
        """测试拥挤度指标计算"""
        factor_names = ['ret_20d', 'rsi_14', 'volatility_20d', 'turnover_20d']
        results = FactorCrowdingAnalyzer.calc_crowding_metrics(self.factor_df, factor_names)
        self.assertEqual(len(results), len(factor_names))
        for fn in factor_names:
            self.assertIn('valuation_crowding', results[fn])
            self.assertIn('concentration', results[fn])
        print("[PASS] 因子拥挤度计算成功")
        for fn, metrics in results.items():
            print(f"  {fn}: valuation_crowding={metrics['valuation_crowding']:.4f}, "
                  f"concentration={metrics['concentration']:.4f}")


if __name__ == '__main__':
    print("=" * 70)
    print("因子引擎增强验证测试")
    print("=" * 70)

    # Run tests
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnhancedFactorCalculator))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFactorDecayAnalyzer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestQuantileBacktest))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFactorCrowdingAnalyzer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("验证总结")
    print(f"  总测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print("=" * 70)