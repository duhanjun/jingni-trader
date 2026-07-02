"""
Qlib Alpha158 风格因子库
借鉴来源: Microsoft Qlib https://github.com/microsoft/qlib

因子分类:
  - 动量: momentum_1d/5d/10d/20d/60d/120d, macd/macd_signal/macd_hist,
          bias_5d/10d/20d/60d, roc_10d/roc_20d
  - 反转: reversal_1d/5d/10d/20d, rsi_6/14/24
  - 波动率: volatility_5d/10d/20d/60d, atr_14, hl_ratio_20d
  - 成交量: volume_ratio_5d/20d, volume_trend_5d/20d, turnover_5d/turnover_20d
  - 技术指标: bb_width, bb_position, kdj_k, kdj_d, kdj_j
  - 资金流向: money_flow_5d/20d, cmf_20d, obv_change_20d
  - 日内: intraday_range, intraday_return, volume_price_corr_20d
  - 估值: lncap
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


class Alpha158FactorEngine:
    """Alpha158 风格扩展因子引擎

    用法:
        engine = Alpha158FactorEngine()
        factors = engine.compute_all(data)  # 计算所有因子
        factors = engine.compute(data, ["momentum_5d", "rsi_14"])  # 指定因子
    """

    CATEGORIES = {
        "momentum": "动量/趋势因子",
        "reversal": "反转因子",
        "volatility": "波动率因子",
        "volume": "成交量因子",
        "technical": "技术指标因子",
        "money_flow": "资金流向因子",
    }

    def __init__(self):
        self._factors: Dict[str, dict] = {}
        self._register_factors()

    def _compute_rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def _compute_bb(self, close: pd.Series, window: int = 20, n_std: float = 2) -> tuple:
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = ma + n_std * std
        lower = ma - n_std * std
        return (upper - lower) / ma, (close - lower) / (upper - lower + 1e-10)

    def _compute_kdj(self, df: pd.DataFrame, n: int = 9) -> tuple:
        low_n = df["low"].rolling(n).min()
        high_n = df["high"].rolling(n).max()
        rsv = (df["close"] - low_n) / (high_n - low_n + 1e-10) * 100
        k = rsv.ewm(alpha=1 / 3, min_periods=3).mean()
        d = k.ewm(alpha=1 / 3, min_periods=3).mean()
        j = 3 * k - 2 * d
        return k, d, j

    def _compute_cmf(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        mf_volume = mf_multiplier * volume
        return mf_volume.rolling(window).sum() / volume.rolling(window).sum()

    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        close_diff = df["close"].diff()
        obv = pd.Series(0, index=df.index, dtype=float)
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df["volume"].iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        return obv

    def _register_factors(self):
        # ── 动量因子 ──
        def momentum(p, n):
            name = f"momentum_{n}d"
            desc = f"{n}日动量: close_t / close_{{t-{n}}} - 1"
            self._factors[name] = {
                "category": "momentum", "description": desc,
                "fn": lambda g: g["close"].pct_change(n)
            }

        for n in [1, 5, 10, 20, 60, 120]:
            momentum(None, n)

        # MACD 系列
        self._factors["macd"] = {
            "category": "momentum", "description": "MACD: EMA12 - EMA26",
            "fn": lambda g: g["close"].ewm(span=12).mean() - g["close"].ewm(span=26).mean()
        }
        self._factors["macd_signal"] = {
            "category": "momentum", "description": "MACD Signal: EMA9 of MACD",
            "fn": lambda g: (g["close"].ewm(span=12).mean() - g["close"].ewm(span=26).mean()).ewm(span=9).mean()
        }
        self._factors["macd_hist"] = {
            "category": "momentum", "description": "MACD Histogram: MACD - Signal",
            "fn": lambda g: (g["close"].ewm(span=12).mean() - g["close"].ewm(span=26).mean()) - (g["close"].ewm(span=12).mean() - g["close"].ewm(span=26).mean()).ewm(span=9).mean()
        }

        # 乖离率
        def bias(n):
            name = f"bias_{n}d"
            desc = f"{n}日乖离率: close / MA{n} - 1"
            self._factors[name] = {
                "category": "momentum", "description": desc,
                "fn": lambda g: g["close"] / g["close"].rolling(n).mean() - 1
            }

        for n in [5, 10, 20, 60]:
            bias(n)

        # ROC (变化率)
        def roc(n):
            name = f"roc_{n}d"
            desc = f"{n}日变化率: (close - close_{n}) / close_{n}"
            self._factors[name] = {
                "category": "momentum", "description": desc,
                "fn": lambda g: (g["close"] - g["close"].shift(n)) / g["close"].shift(n).replace(0, np.nan)
            }

        for n in [10, 20]:
            roc(n)

        # ── 反转因子 ──
        def reversal(n):
            name = f"reversal_{n}d"
            desc = f"{n}日反转: -momentum"
            self._factors[name] = {
                "category": "reversal", "description": desc,
                "fn": lambda g: -g["close"].pct_change(n)
            }

        for n in [1, 5, 10, 20]:
            reversal(n)

        # RSI
        for n in [6, 14, 24]:
            self._factors[f"rsi_{n}"] = {
                "category": "reversal", "description": f"{n}日RSI",
                "fn": lambda g: self._compute_rsi(g["close"], n)
            }

        # ── 波动率因子 ──
        def vol(n):
            name = f"volatility_{n}d"
            desc = f"{n}日波动率: ret_1d rolling std"
            self._factors[name] = {
                "category": "volatility", "description": desc,
                "fn": lambda g: g["close"].pct_change().rolling(n).std()
            }

        for n in [5, 10, 20, 60]:
            vol(n)

        self._factors["atr_14"] = {
            "category": "volatility", "description": "14日ATR",
            "fn": lambda g: self._compute_atr(g, 14)
        }

        self._factors["hl_ratio_20d"] = {
            "category": "volatility", "description": "20日高低价差比: mean((high-low)/close)",
            "fn": lambda g: ((g["high"] - g["low"]) / g["close"]).rolling(20).mean()
        }

        # ── 成交量因子 ──
        def vol_ratio(n):
            name = f"volume_ratio_{n}d"
            desc = f"{n}日量比: volume / volume_ma{n}"
            self._factors[name] = {
                "category": "volume", "description": desc,
                "fn": lambda g: g["volume"] / g["volume"].rolling(n).mean()
            }

        for n in [5, 20]:
            vol_ratio(n)

        def vol_trend(n):
            name = f"volume_trend_{n}d"
            desc = f"{n}日成交量趋势: pct_change(volume_ma{n})"
            self._factors[name] = {
                "category": "volume", "description": desc,
                "fn": lambda g: g["volume"].rolling(n).mean().pct_change(n)
            }

        for n in [5, 20]:
            vol_trend(n)

        def turnover(n):
            name = f"turnover_{n}d"
            desc = f"{n}日平均换手率"
            self._factors[name] = {
                "category": "volume", "description": desc,
                "fn": lambda g: g.get("turnover_rate", g["volume"]).rolling(n).mean()
            }

        for n in [5, 20]:
            turnover(n)

        # ── 技术指标 ──
        def bb(name, func):
            self._factors[name] = {
                "category": "technical",
                "description": "布林带指标",
                "fn": lambda g: func(*self._compute_bb(g["close"], 20, 2))
            }

        bb("bb_width", lambda w, p: w)
        bb("bb_position", lambda w, p: p)

        def kdj(which):
            self._factors[f"kdj_{which}"] = {
                "category": "technical", "description": f"KDJ {which.upper()}值",
                "fn": lambda g: {
                    "k": lambda g: self._compute_kdj(g)[0],
                    "d": lambda g: self._compute_kdj(g)[1],
                    "j": lambda g: self._compute_kdj(g)[2],
                }[which](g)
            }

        for which in ["k", "d", "j"]:
            kdj(which)

        # ── 资金流向因子 ──
        def mf(n):
            name = f"money_flow_{n}d"
            desc = f"{n}日资金流向: sum(ret * volume)"
            self._factors[name] = {
                "category": "money_flow", "description": desc,
                "fn": lambda g: (g["close"].pct_change() * g["volume"]).rolling(n).sum()
            }

        for n in [5, 20]:
            mf(n)

        self._factors["cmf_20d"] = {
            "category": "money_flow", "description": "20日Chaikin资金流向",
            "fn": lambda g: self._compute_cmf(g, 20)
        }

        self._factors["obv_change_20d"] = {
            "category": "money_flow", "description": "20日OBV变化率",
            "fn": lambda g: self._compute_obv(g).pct_change(20)
        }

        # ── 通用 ──
        self._factors["lncap"] = {
            "category": "momentum", "description": "对数市值",
            "fn": lambda g: np.log(g.get("amount", g["volume"]) / g.get("turnover_rate", pd.Series(1, index=g.index)).replace(0, np.nan) * 100)
        }

        # ── 日内 ──
        self._factors["intraday_range"] = {
            "category": "volatility", "description": "日内振幅: (high - low) / open",
            "fn": lambda g: (g["high"] - g["low"]) / g["open"]
        }

        self._factors["intraday_return"] = {
            "category": "momentum", "description": "日内收益: (close - open) / open",
            "fn": lambda g: (g["close"] - g["open"]) / g["open"]
        }

        # ── 复合 ──
        self._factors["volume_price_corr_20d"] = {
            "category": "volume", "description": "20日量价相关性",
            "fn": lambda g: g["volume"].rolling(20).corr(g["close"])
        }

    def list_factors(self) -> List[dict]:
        """列出所有因子信息"""
        return [
            {
                "name": name,
                "category": info["category"],
                "description": info["description"],
            }
            for name, info in self._factors.items()
        ]

    def get_by_category(self, category: str) -> List[str]:
        return [name for name, info in self._factors.items() if info["category"] == category]

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        return self.compute(data, list(self._factors.keys()))

    def compute(
        self,
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """计算指定因子集合"""
        if factor_names is None:
            factor_names = list(self._factors.keys())

        data = data.sort_values(["code", "date"]).copy()
        result = data[["code", "date"]].copy()

        for name in factor_names:
            if name not in self._factors:
                continue

            fn = self._factors[name]["fn"]
            vals = pd.Series(index=data.index, dtype=float)

            for code, group in data.groupby("code"):
                group = group.sort_values("date")
                try:
                    factor_vals = fn(group)
                    vals.loc[group.index] = factor_vals.values
                except Exception:
                    pass

            result[name] = vals

        return result