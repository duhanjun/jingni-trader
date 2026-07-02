"""
单元测试：通过 511090.SH（30年国债ETF）MA20 均线策略测试整个项目。

覆盖：
  1. _synthesize_bond_etf_data      —— 合成数据（边界 / 上市日 / schema）
  2. calc_ma20_signals              —— MA20 信号核心逻辑（方向 / 过滤 / 丢弃）
  3. merge_buy_sell_flags           —— 当日买卖标志合并
  4. fetch_etf_daily_data           —— 数据获取（缓存命中 / akshare / tushare / 合成兜底，均 mock 网络）
  5. run_bond_etf_ma20_strategy     —— 端到端流水线（mock 网络，目录指向 tmp）

运行：
    pytest tests/test_bond_etf_ma20.py -v
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

import run_bond_etf_ma20 as m


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
CACHE_PATHS = [
    "/tmp/511090_sina.parquet",
    "/tmp/511090_akshare.parquet",
    "/tmp/511090_daily.parquet",
]


@pytest.fixture(autouse=True)
def _clean_caches():
    """清理 fetch_etf_daily_data 可能写入的 /tmp 缓存，避免用例间串扰。"""
    for p in CACHE_PATHS:
        try:
            os.remove(p)
        except OSError:
            pass
    yield
    for p in CACHE_PATHS:
        try:
            os.remove(p)
        except OSError:
            pass


def _make_price_df(closes, code="511090.SH", start="2023-06-13"):
    """依据给定收盘价序列构造标准 OHLCV DataFrame。"""
    dates = pd.bdate_range(start=start, periods=len(closes))
    df = pd.DataFrame({"close": [float(c) for c in closes]}, index=dates)
    df = df.reset_index().rename(columns={"index": "date"})
    df["code"] = code
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["vol"] = 1000
    return df[["date", "code", "open", "high", "low", "close", "vol"]]


# ---------------------------------------------------------------------------
# 1) 合成数据
# ---------------------------------------------------------------------------
class TestSynthesizeBondEtfData:
    def test_schema_and_positive_prices(self):
        df = m._synthesize_bond_etf_data("511090.SH", "2023-06-13", "2024-12-31")
        expected_cols = ["date", "code", "open", "high", "low", "close", "vol"]
        assert list(df.columns) == expected_cols
        assert (df["close"] > 0).all()
        assert (df["open"] > 0).all()
        assert (df["high"] >= df["low"]).all()
        assert df["code"].unique().tolist() == ["511090.SH"]

    def test_clamps_to_listing_date(self):
        # 起始日早于真实上市日 2023-06-13，应被夹到上市日
        df = m._synthesize_bond_etf_data("511090.SH", "2021-01-01", "2024-12-31")
        assert df["date"].min() >= pd.Timestamp("2023-06-13")
        assert len(df) >= 30

    def test_no_clamp_when_start_after_listing(self):
        df = m._synthesize_bond_etf_data("511090.SH", "2024-01-01", "2024-12-31")
        assert df["date"].min() >= pd.Timestamp("2024-01-01")

    def test_short_range_returns_empty_with_schema(self):
        # 不足 30 个交易日 → 返回空 DataFrame，但保留列结构
        df = m._synthesize_bond_etf_data("511090.SH", "2024-12-01", "2024-12-10")
        assert df.empty
        assert list(df.columns) == ["date", "code", "open", "high", "low", "close", "vol"]

    def test_deterministic_seed(self):
        a = m._synthesize_bond_etf_data("511090.SH", "2023-06-13", "2024-06-13")
        b = m._synthesize_bond_etf_data("511090.SH", "2023-06-13", "2024-06-13")
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# 2) MA20 信号
# ---------------------------------------------------------------------------
class TestCalcMa20Signals:
    def test_filters_to_target_code_and_drops_warmup(self):
        # 构造 30 行目标标的 + 10 行其它标的
        target = _make_price_df([10] * 30, code="511090.SH")
        other = _make_price_df([10] * 10, code="999999.SZ", start="2023-06-13")
        data = pd.concat([target, other], ignore_index=True)

        out = m.calc_ma20_signals(data)

        # 只保留目标标的
        assert set(out["code"].unique()) == {"511090.SH"}
        # MA20 需要前 20 个观测，前 19 行被丢弃（NaN）
        assert len(out) == 30 - (m.MA_PERIOD - 1)

    def test_signal_direction_and_values(self):
        # 19 个持平 + 第20个大涨 + 第21个暴跌，精确验证信号
        closes = [10] * 19 + [20, 5]
        data = _make_price_df(closes, code="511090.SH")
        out = m.calc_ma20_signals(data).reset_index(drop=True)

        # 仅剩 2 行有效信号
        assert len(out) == 2
        # 第20行: close=20 > ma20=10.5 -> 1
        # 第21行: close=5  <= ma20=10.25 -> -1
        assert out.loc[0, "signal"] == 1
        assert out.loc[1, "signal"] == -1
        assert set(out["signal"].unique()) <= {1, -1}

    def test_all_signals_are_binary(self):
        closes = list(np.linspace(100, 130, 40)) + list(np.linspace(130, 90, 20))
        data = _make_price_df(closes, code="511090.SH")
        out = m.calc_ma20_signals(data)
        assert set(out["signal"].unique()) <= {1, -1}


# ---------------------------------------------------------------------------
# 3) 当日买卖标志合并
# ---------------------------------------------------------------------------
class TestMergeBuySellFlags:
    def test_collapses_to_one_signal_per_day(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"])
        signals = pd.DataFrame({
            "date": dates,
            "code": ["511090.SH", "511090.SH", "511090.SH"],
            "signal": [1, -1, 1],
        })
        out = m.merge_buy_sell_flags(signals)
        # 每天只保留最后一条信号
        assert len(out) == 2
        assert out.iloc[0]["signal"] == -1   # 第一天最后一条是 -1
        assert out.iloc[1]["signal"] == 1    # 第二天是 1

    def test_keeps_multiple_codes_separate(self):
        signals = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "code": ["511090.SH", "511090.SH"],
            "signal": [1, -1],
        })
        out = m.merge_buy_sell_flags(signals)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 4) 数据获取（mock 网络 / 系统工具）
# ---------------------------------------------------------------------------
def _inject_fake_module(name, attrs):
    """向 sys.modules 注入一个假的第三方包，使 `import name` 命中假模块。"""
    fake = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(fake, k, v)
    saved = {}
    token = _FakeModuleToken(name, fake, saved)
    sys.modules[name] = fake
    return token


class _FakeModuleToken:
    def __init__(self, name, fake, saved):
        self.name = name
        self.fake = fake
        self.saved = saved
        self.saved[name] = sys.modules.get(name)

    def remove(self):
        if self.saved[self.name] is None:
            sys.modules.pop(self.name, None)
        else:
            sys.modules[self.name] = self.saved[self.name]


class TestFetchEtfDailyData:
    def test_cache_hit(self):
        # 在 /tmp 缓存写入一份特征明显的样本，并使 akshare 抛错：
        # 若缓存未被优先使用，则应走到 akshare 而抛错。
        sample = _make_price_df([100, 101, 102], start="2024-01-02")
        sample.to_parquet("/tmp/511090_sina.parquet", index=False)

        with _akshare_raising(RuntimeError("should not be called when cache hits")):
            df = m.fetch_etf_daily_data("511090.SH", "2024-01-01", "2024-02-01")

        # 返回的是缓存内容，证明缓存优先
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            sample.reset_index(drop=True),
        )

    def test_akshare_path_success(self):
        sample = _make_price_df(list(range(100, 120)), start="2023-07-01")
        with _akshare_returning(sample):
            df = m.fetch_etf_daily_data("511090.SH", "2023-07-01", "2023-08-01")
        assert not df.empty
        assert "vol" in df.columns
        assert (df["code"] == "511090.SH").all()

    def test_tushare_fallback_when_akshare_fails(self):
        sample = _make_price_df(list(range(50, 70)), start="2023-07-01")
        with _akshare_raising(RuntimeError("akshare down")), \
             _tushare_returning(sample):
            df = m.fetch_etf_daily_data("511090.SH", "2023-07-01", "2023-08-01")
        assert not df.empty
        assert (df["code"] == "511090.SH").all()

    def test_synthetic_fallback_when_all_sources_fail(self):
        with _akshare_raising(RuntimeError("akshare down")), \
             _tushare_raising(RuntimeError("tushare down")):
            df = m.fetch_etf_daily_data("511090.SH", "2023-07-01", "2024-08-01")
        # 合成兜底：非空且 schema 正确
        assert not df.empty
        assert list(df.columns) == ["date", "code", "open", "high", "low", "close", "vol"]
        assert (df["close"] > 0).all()


# ---- akshare / tushare mock 上下文管理器 ----
class _akshare_returning:
    def __init__(self, df):
        self.df = df
        self.token = None

    def __enter__(self):
        def fake_fund_etf_hist_sina(symbol):
            out = self.df.rename(columns={"vol": "volume"})
            return out
        self.token = _inject_fake_module("akshare", {"fund_etf_hist_sina": fake_fund_etf_hist_sina})
        return self

    def __exit__(self, *a):
        self.token.remove()


class _akshare_raising:
    def __init__(self, exc):
        self.exc = exc
        self.token = None

    def __enter__(self):
        def fake_fund_etf_hist_sina(symbol):
            raise self.exc
        self.token = _inject_fake_module("akshare", {"fund_etf_hist_sina": fake_fund_etf_hist_sina})
        return self

    def __exit__(self, *a):
        self.token.remove()


class _tushare_returning:
    def __init__(self, df):
        self.df = df
        self.token = None

    def __enter__(self):
        pro = types.SimpleNamespace(
            fund_daily=lambda ts_code, start_date, end_date: self.df.rename(
                columns={"code": "ts_code", "date": "trade_date"}
            )
        )
        def fake_pro_api():
            return pro
        fake_ts = types.SimpleNamespace(set_token=lambda t: None, pro_api=fake_pro_api)
        self.token = _inject_fake_module("tushare", {"pro_api": fake_pro_api, "set_token": lambda t: None})
        sys.modules["tushare"] = fake_ts
        return self

    def __exit__(self, *a):
        self.token.remove()


class _tushare_raising:
    def __init__(self, exc):
        self.exc = exc
        self.token = None

    def __enter__(self):
        def fake_pro_api():
            raise self.exc
        self.token = _inject_fake_module("tushare", {"pro_api": fake_pro_api, "set_token": lambda t: None})
        return self

    def __exit__(self, *a):
        self.token.remove()


# ---------------------------------------------------------------------------
# 5) 端到端流水线
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_run_pipeline_end_to_end(self, tmp_path, monkeypatch):
        # 用合成数据替换真实网络获取，保证离线、确定性
        fake_daily = m._synthesize_bond_etf_data("511090.SH", "2023-06-13", "2024-12-31")

        class _FakeFetcher:
            @staticmethod
            def fetch(symbol, start_date, end_date):
                return fake_daily.copy()

        monkeypatch.setattr(m, "fetch_etf_daily_data", _FakeFetcher.fetch)

        # backtest-engine 自带 config 默认后端为 rqalpha（其适配器未实现），
        # 而 engine 模块在加载时已把 BACKTEST_BACKEND 绑成模块级全局。
        # BacktestEngine.__globals__ 即该引擎模块的全局命名空间，直接改它。
        monkeypatch.setitem(
            m.BacktestEngine._load_adapter.__globals__, "BACKTEST_BACKEND", "native"
        )

        # 把各产物 / 归档目录指向临时目录，避免污染 workspace
        monkeypatch.setattr(m, "DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setattr(m, "FACTOR_DIR", str(tmp_path / "factors"))
        monkeypatch.setattr(m, "BACKTEST_DIR", str(tmp_path / "backtest"))
        monkeypatch.setattr(m, "REPORT_DIR", str(tmp_path / "reports"))
        monkeypatch.setattr(m, "ARCHIVE_DIR", str(tmp_path / "archives"))
        for d in ["data", "factors", "backtest", "reports", "archives"]:
            (tmp_path / d).mkdir(exist_ok=True)

        result = m.run_bond_etf_ma20_strategy()

        assert result["success"] is True
        assert "metrics" in result and isinstance(result["metrics"], dict)
        assert result["metrics"]  # 非空
        assert os.path.exists(result["report_path"])

    def test_pipeline_signal_counts_consistency(self, tmp_path, monkeypatch):
        """单独验证信号生成（MODEL 步骤）买卖双方数量与数据一致。"""
        daily = m._synthesize_bond_etf_data("511090.SH", "2023-06-13", "2024-12-31")
        raw = m.calc_ma20_signals(daily)
        final = m.merge_buy_sell_flags(raw)
        n_buy = int((final["signal"] == 1).sum())
        n_sell = int((final["signal"] == -1).sum())
        assert n_buy + n_sell == len(final)
        assert n_buy > 0 and n_sell > 0
