"""
K线图生成模块
生成带均线叠加和成交量的交互式HTML K线图
支持 Plotly 和 TradingView lightweight-charts 两种渲染引擎
"""
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

try:
    from scripts.config import CHART_THEME
except ImportError:
    CHART_THEME = "plotly_white"

# A股颜色惯例：红涨绿跌
UP_COLOR = '#d62728'    # 红色 - 上涨
DOWN_COLOR = '#2ca02c'  # 绿色 - 下跌
MA_COLORS = ['#ff7f0e', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']


class KlineChartGenerator:
    """K线图生成器"""

    def generate(self, df: pd.DataFrame,
                 stock_code: str = "",
                 stock_name: str = "",
                 ma_periods: List[int] = None,
                 support_resistance: Optional[Dict] = None,
                 height: int = 600) -> go.Figure:
        """
        生成K线图

        参数:
            df: OHLCV数据，含 date, open, high, low, close, volume 列
            stock_code: 股票代码
            stock_name: 股票名称
            ma_periods: 均线周期列表，默认 [5, 10, 20, 60]
            support_resistance: 支撑阻力位数据，用于在图上标注水平线
            height: 图表高度

        返回:
            plotly Figure 对象
        """
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        if ma_periods is None:
            ma_periods = [5, 10, 20, 60]

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # 创建子图：价格（80%）+ 成交量（20%）
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.8, 0.2],
            subplot_titles=("价格走势", "成交量"),
        )

        # K线图 - A股惯例：红涨绿跌
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing=dict(line=dict(color=UP_COLOR), fillcolor=UP_COLOR),
                decreasing=dict(line=dict(color=DOWN_COLOR), fillcolor=DOWN_COLOR),
                name='K线',
                showlegend=False,
            ),
            row=1, col=1
        )

        # 均线叠加
        for i, period in enumerate(ma_periods):
            if period <= 0:
                continue
            ma = df['close'].rolling(window=period).mean()
            color = MA_COLORS[i % len(MA_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=ma,
                    mode='lines',
                    name=f'MA{period}',
                    line=dict(color=color, width=1.5),
                ),
                row=1, col=1
            )

        # 成交量柱状图 - 按涨跌着色
        vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
                      for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(
                x=df['date'],
                y=df['volume'],
                marker_color=vol_colors,
                name='成交量',
                showlegend=False,
            ),
            row=2, col=1
        )

        # 支撑阻力位水平线
        if support_resistance:
            self._add_support_resistance_lines(fig, support_resistance)

        # 标题
        title_parts = []
        if stock_name:
            title_parts.append(stock_name)
        if stock_code:
            title_parts.append(stock_code)
        title = f"{' '.join(title_parts)} K线图" if title_parts else "K线图"

        fig.update_layout(
            title=title,
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
        )

        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        return fig

    def generate_full_chart(self, df: pd.DataFrame,
                            stock_code: str = "",
                            stock_name: str = "",
                            ma_periods: List[int] = None,
                            support_resistance: Optional[Dict] = None,
                            height: int = 1200) -> go.Figure:
        """
        生成K线+成交量+MACD+RSI+KDJ五合一联动图
        所有子图共享x轴，悬停时十字光标同步联动

        参数:
            df: OHLCV数据，含 date, open, high, low, close, volume 列
            stock_code: 股票代码
            stock_name: 股票名称
            ma_periods: 均线周期列表，默认 [5, 10, 20, 60]
            support_resistance: 支撑阻力位数据
            height: 图表高度

        返回:
            plotly Figure 对象
        """
        if df is None or df.empty:
            fig = go.Figure()
            fig.update_layout(title="无数据", template=CHART_THEME)
            return fig

        if ma_periods is None:
            ma_periods = [5, 10, 20, 60]

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        close = df['close']

        # ---- 预计算所有指标 ----
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)
        hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in macd_hist]

        # RSI
        period = 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50)

        # KDJ
        n_kdj = 9
        low_min = df['low'].rolling(window=n_kdj, min_periods=1).min()
        high_max = df['high'].rolling(window=n_kdj, min_periods=1).max()
        rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        rsv = rsv.fillna(50)
        k_val = rsv.ewm(com=2, adjust=False).mean()
        d_val = k_val.ewm(com=2, adjust=False).mean()
        j_val = 3 * k_val - 2 * d_val

        # ---- 创建5行联动子图 ----
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.40, 0.12, 0.16, 0.16, 0.16],
            subplot_titles=(
                f"{' '.join([stock_name, stock_code])} K线走势".strip(),
                "成交量",
                "MACD",
                "RSI(14)",
                "KDJ(9,3,3)",
            ),
        )

        # Row 1: K线
        fig.add_trace(
            go.Candlestick(
                x=df['date'], open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                increasing=dict(line=dict(color=UP_COLOR), fillcolor=UP_COLOR),
                decreasing=dict(line=dict(color=DOWN_COLOR), fillcolor=DOWN_COLOR),
                name='K线', showlegend=False,
            ), row=1, col=1
        )

        # Row 1: 均线叠加
        for i, period_ma in enumerate(ma_periods):
            if period_ma <= 0:
                continue
            ma = close.rolling(window=period_ma).mean()
            color = MA_COLORS[i % len(MA_COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=df['date'], y=ma,
                    mode='lines', name=f'MA{period_ma}',
                    line=dict(color=color, width=1.5),
                ), row=1, col=1
            )

        # Row 1: 支撑阻力位
        if support_resistance:
            self._add_support_resistance_lines(fig, support_resistance)

        # Row 2: 成交量
        vol_colors = [UP_COLOR if c >= o else DOWN_COLOR
                      for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(
                x=df['date'], y=df['volume'],
                marker_color=vol_colors, name='成交量', showlegend=False,
            ), row=2, col=1
        )

        # Row 3: MACD
        fig.add_trace(
            go.Scatter(x=df['date'], y=dif, mode='lines', name='DIF',
                       line=dict(color='#1f77b4', width=1.5), showlegend=False),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=dea, mode='lines', name='DEA',
                       line=dict(color='#ff7f0e', width=1.5), showlegend=False),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=df['date'], y=macd_hist, marker_color=hist_colors,
                   name='MACD', showlegend=False),
            row=3, col=1
        )
        fig.add_hline(y=0, line=dict(color='gray', width=0.5), row=3, col=1)

        # Row 4: RSI
        fig.add_trace(
            go.Scatter(x=df['date'], y=rsi, mode='lines', name='RSI',
                       line=dict(color='#1f77b4', width=1.5), showlegend=False),
            row=4, col=1
        )
        fig.add_hline(y=70, line=dict(color=UP_COLOR, width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=30, line=dict(color=DOWN_COLOR, width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=50, line=dict(color='gray', width=0.5, dash='dot'), row=4, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=UP_COLOR, opacity=0.05, line_width=0, row=4, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=DOWN_COLOR, opacity=0.05, line_width=0, row=4, col=1)

        # Row 5: KDJ
        fig.add_trace(
            go.Scatter(x=df['date'], y=k_val, mode='lines', name='K',
                       line=dict(color='#1f77b4', width=1.5), showlegend=False),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=d_val, mode='lines', name='D',
                       line=dict(color='#ff7f0e', width=1.5), showlegend=False),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=j_val, mode='lines', name='J',
                       line=dict(color='#9467bd', width=1.5), showlegend=False),
            row=5, col=1
        )
        fig.add_hline(y=80, line=dict(color=UP_COLOR, width=0.8, dash='dash'), row=5, col=1)
        fig.add_hline(y=20, line=dict(color=DOWN_COLOR, width=0.8, dash='dash'), row=5, col=1)
        fig.add_hline(y=50, line=dict(color='gray', width=0.5, dash='dot'), row=5, col=1)

        # ---- 全局布局 ----
        title_parts = []
        if stock_name:
            title_parts.append(stock_name)
        if stock_code:
            title_parts.append(stock_code)
        title = f"{' '.join(title_parts)} 综合技术分析图" if title_parts else "综合技术分析图"

        fig.update_layout(
            title=title,
            height=height,
            template=CHART_THEME,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False,
            xaxis3_rangeslider_visible=False,
            xaxis4_rangeslider_visible=False,
            xaxis5_rangeslider_visible=False,
        )

        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)
        fig.update_yaxes(title_text="KDJ", range=[0, 100], row=5, col=1)
        fig.update_xaxes(title_text="日期", row=5, col=1)

        return fig

    # ================================================================
    # TradingView lightweight-charts 渲染引擎
    # ================================================================

    def generate_tradingview_chart(self, df: pd.DataFrame,
                                   stock_code: str = "",
                                   stock_name: str = "",
                                   ma_periods: List[int] = None,
                                   support_resistance: Optional[Dict] = None,
                                   show_support_resistance: bool = False,
                                   height: int = 800) -> str:
        """
        使用 TradingView lightweight-charts 生成专业级 K线图
        K线主图占大部分空间，下方有指标切换标签栏
        点击标签切换技术指标面板（成交量/MACD/RSI/KDJ/BOLL）
        所有面板与主图共享时间轴，十字光标同步联动
        """
        if df is None or df.empty:
            return '<div style="text-align:center;padding:40px;color:#888;">无数据</div>'

        if ma_periods is None:
            ma_periods = [5, 10, 20, 60]

        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        close = df['close']

        # ---- 预计算所有指标 ----
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = 2 * (dif - dea)

        # RSI
        period_rsi = 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period_rsi, min_periods=period_rsi, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period_rsi, min_periods=period_rsi, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - 100 / (1 + rs)).fillna(50)

        # KDJ
        n_kdj = 9
        low_min = df['low'].rolling(window=n_kdj, min_periods=1).min()
        high_max = df['high'].rolling(window=n_kdj, min_periods=1).max()
        rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
        rsv = rsv.fillna(50)
        k_val = rsv.ewm(com=2, adjust=False).mean()
        d_val = k_val.ewm(com=2, adjust=False).mean()
        j_val = 3 * k_val - 2 * d_val

        # BOLL
        boll_period = 20
        boll_mid = close.rolling(window=boll_period).mean()
        boll_std = close.rolling(window=boll_period).std()
        boll_upper = boll_mid + 2 * boll_std
        boll_lower = boll_mid - 2 * boll_std

        # WR (Williams %R)
        wr_period = 14
        wr_high = df['high'].rolling(window=wr_period).max()
        wr_low = df['low'].rolling(window=wr_period).min()
        wr = (wr_high - close) / (wr_high - wr_low).replace(0, np.nan) * -100

        # CCI (Commodity Channel Index)
        cci_period = 20
        tp = (df['high'] + df['low'] + close) / 3
        tp_sma = tp.rolling(window=cci_period).mean()
        tp_mad = tp.rolling(window=cci_period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

        # OBV (On Balance Volume)
        obv = pd.Series(0.0, index=df.index)
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df['volume'].iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        # ---- 构建JSON数据 ----
        chart_data = []
        for i, row in df.iterrows():
            d = row['date']
            if hasattr(d, 'strftime'):
                ts = d.strftime('%Y-%m-%d')
            else:
                ts = str(d)
            entry = {
                "time": ts,
                "open": round(float(row['open']), 2),
                "high": round(float(row['high']), 2),
                "low": round(float(row['low']), 2),
                "close": round(float(row['close']), 2),
                "volume": int(row['volume']),
            }
            idx = df.index.get_loc(i)
            # 均线
            for p in ma_periods:
                if p > 0 and idx >= p - 1:
                    ma_val = close.iloc[max(0, idx - p + 1):idx + 1].mean()
                    entry[f"ma{p}"] = round(float(ma_val), 2)
            # MACD
            entry["dif"] = round(float(dif.iloc[idx]), 4) if not np.isnan(dif.iloc[idx]) else None
            entry["dea"] = round(float(dea.iloc[idx]), 4) if not np.isnan(dea.iloc[idx]) else None
            entry["macd"] = round(float(macd_hist.iloc[idx]), 4) if not np.isnan(macd_hist.iloc[idx]) else None
            # RSI
            entry["rsi"] = round(float(rsi.iloc[idx]), 2) if not np.isnan(rsi.iloc[idx]) else None
            # KDJ
            entry["k"] = round(float(k_val.iloc[idx]), 2) if not np.isnan(k_val.iloc[idx]) else None
            entry["d"] = round(float(d_val.iloc[idx]), 2) if not np.isnan(d_val.iloc[idx]) else None
            entry["j"] = round(float(j_val.iloc[idx]), 2) if not np.isnan(j_val.iloc[idx]) else None
            # BOLL
            entry["boll_upper"] = round(float(boll_upper.iloc[idx]), 2) if not np.isnan(boll_upper.iloc[idx]) else None
            entry["boll_mid"] = round(float(boll_mid.iloc[idx]), 2) if not np.isnan(boll_mid.iloc[idx]) else None
            entry["boll_lower"] = round(float(boll_lower.iloc[idx]), 2) if not np.isnan(boll_lower.iloc[idx]) else None
            # WR
            entry["wr"] = round(float(wr.iloc[idx]), 2) if not np.isnan(wr.iloc[idx]) else None
            # CCI
            entry["cci"] = round(float(cci.iloc[idx]), 2) if not np.isnan(cci.iloc[idx]) else None
            # OBV
            entry["obv"] = round(float(obv.iloc[idx]), 0) if not np.isnan(obv.iloc[idx]) else None
            chart_data.append(entry)

        # 支撑阻力位（仅在显式要求时显示）
        sr_lines = []
        if show_support_resistance and support_resistance:
            for level in (support_resistance.get('resistance', []))[:4]:
                price = self._extract_level_price(level)
                if price and not np.isnan(price):
                    sr_lines.append({"price": round(price, 2), "color": UP_COLOR, "title": f"阻力 {price:.2f}"})
            for level in (support_resistance.get('support', []))[:4]:
                price = self._extract_level_price(level)
                if price and not np.isnan(price):
                    sr_lines.append({"price": round(price, 2), "color": DOWN_COLOR, "title": f"支撑 {price:.2f}"})

        import uuid
        container_id = "tv_chart_" + uuid.uuid4().hex[:8]

        title_parts = []
        if stock_name:
            title_parts.append(stock_name)
        if stock_code:
            title_parts.append(stock_code)
        chart_title = f"{' '.join(title_parts)}" if title_parts else ""

        data_json = json.dumps(chart_data, ensure_ascii=False)
        sr_json = json.dumps(sr_lines, ensure_ascii=False)
        ma_periods_json = json.dumps(ma_periods)

        # K线主图高度占65%，指标面板占35%
        kline_height = int(height * 0.62)
        indicator_height = int(height * 0.35)

        html = f'''<div id="{container_id}" style="width:100%;font-family:'PingFang SC','Microsoft YaHei',sans-serif;"></div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
    var container = document.getElementById('{container_id}');
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<p style="text-align:center;color:#888;padding:20px;">图表库加载失败，请检查网络连接</p>';
        return;
    }}

    var data = {data_json};
    var srLines = {sr_json};
    var maPeriods = {ma_periods_json};
    var maColors = {{5:'#ff7f0e',10:'#1f77b4',20:'#9467bd',60:'#8c564b',120:'#e377c2',250:'#17becf'}};
    var UP = '#ef5350', DOWN = '#26a69a';

    // ---- 工具栏（股票信息 + 当前价格） ----
    var toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex;align-items:center;gap:12px;padding:8px 12px;background:#fff;border:1px solid #e0e3eb;border-radius:4px 4px 0 0;border-bottom:none;font-size:13px;';
    var nameSpan = document.createElement('span');
    nameSpan.style.cssText = 'font-weight:600;color:#262833;font-size:15px;';
    nameSpan.textContent = '{chart_title}';
    toolbar.appendChild(nameSpan);
    var lastBar = data[data.length - 1];
    var priceSpan = document.createElement('span');
    var isUp = lastBar.close >= lastBar.open;
    priceSpan.style.cssText = 'font-weight:600;font-size:15px;color:' + (isUp ? UP : DOWN) + ';';
    priceSpan.textContent = lastBar.close.toFixed(2);
    toolbar.appendChild(priceSpan);
    var chgSpan = document.createElement('span');
    var chgPct = ((lastBar.close - lastBar.open) / lastBar.open * 100);
    chgSpan.style.cssText = 'font-size:12px;color:' + (isUp ? UP : DOWN) + ';';
    chgSpan.textContent = (isUp ? '+' : '') + (lastBar.close - lastBar.open).toFixed(2) + ' (' + (isUp ? '+' : '') + chgPct.toFixed(2) + '%)';
    toolbar.appendChild(chgSpan);
    container.appendChild(toolbar);

    // ---- K线主图容器 ----
    var klineWrapper = document.createElement('div');
    klineWrapper.style.cssText = 'position:relative;width:100%;border:1px solid #e0e3eb;border-bottom:none;';
    var klineDiv = document.createElement('div');
    klineDiv.style.cssText = 'width:100%;height:{kline_height}px;';
    klineWrapper.appendChild(klineDiv);
    container.appendChild(klineWrapper);

    // ---- 指标标签栏 ----
    var tabBar = document.createElement('div');
    tabBar.style.cssText = 'display:flex;gap:0;border:1px solid #e0e3eb;border-top:none;border-bottom:none;background:#fafbfc;';
    var indicators = [
        {{key:'vol', label:'成交量'}},
        {{key:'macd', label:'MACD'}},
        {{key:'rsi', label:'RSI'}},
        {{key:'kdj', label:'KDJ'}},
        {{key:'boll', label:'BOLL'}},
        {{key:'wr', label:'WR'}},
        {{key:'cci', label:'CCI'}},
        {{key:'obv', label:'OBV'}},
    ];
    var tabs = {{}};
    indicators.forEach(function(ind, i) {{
        var tab = document.createElement('div');
        tab.style.cssText = 'padding:6px 16px;font-size:12px;cursor:pointer;border-right:1px solid #e0e3eb;color:#787b86;transition:all 0.15s;user-select:none;';
        if (i === 0) {{
            tab.style.background = '#fff';
            tab.style.color = '#262833';
            tab.style.fontWeight = '600';
            tab.style.borderBottom = '2px solid #2962ff';
        }}
        tab.textContent = ind.label;
        tab.dataset.key = ind.key;
        tab.addEventListener('mouseenter', function() {{
            if (this.style.borderBottom.indexOf('2px') === -1) this.style.background = '#f0f3fa';
        }});
        tab.addEventListener('mouseleave', function() {{
            if (this.style.borderBottom.indexOf('2px') === -1) this.style.background = '#fafbfc';
        }});
        tabBar.appendChild(tab);
        tabs[ind.key] = tab;
    }});
    container.appendChild(tabBar);

    // ---- 指标面板容器 ----
    var indicatorWrapper = document.createElement('div');
    indicatorWrapper.style.cssText = 'position:relative;width:100%;border:1px solid #e0e3eb;border-top:none;border-radius:0 0 4px 4px;';
    var indicatorDiv = document.createElement('div');
    indicatorDiv.style.cssText = 'width:100%;height:{indicator_height}px;';
    indicatorWrapper.appendChild(indicatorDiv);
    container.appendChild(indicatorWrapper);

    // ---- 创建K线主图 ----
    var mainChart = LightweightCharts.createChart(klineDiv, {{
        width: klineDiv.clientWidth,
        height: {kline_height},
        layout: {{
            background: {{ type: 'solid', color: '#ffffff' }},
            textColor: '#787b86',
            fontSize: 11,
            fontFamily: "'PingFang SC','Microsoft YaHei',sans-serif",
        }},
        grid: {{
            vertLines: {{ color: '#f0f3fa' }},
            horzLines: {{ color: '#f0f3fa' }},
        }},
        crosshair: {{
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {{ color: '#787b86', width: 1, style: LightweightCharts.LineStyle.Dashed, labelBackgroundColor: '#2962ff' }},
            horzLine: {{ color: '#787b86', width: 1, style: LightweightCharts.LineStyle.Dashed, labelBackgroundColor: '#2962ff' }},
        }},
        rightPriceScale: {{ borderColor: '#e0e3eb', scaleMargins: {{ top: 0.08, bottom: 0.08 }} }},
        timeScale: {{ borderColor: '#e0e3eb', timeVisible: false, secondsVisible: false }},
        localization: {{ locale: 'zh-CN', priceFormatter: function(p) {{ return p.toFixed(2); }} }},
    }});

    var candleSeries = mainChart.addCandlestickSeries({{
        upColor: UP, downColor: DOWN,
        borderUpColor: UP, borderDownColor: DOWN,
        wickUpColor: UP, wickDownColor: DOWN,
    }});
    candleSeries.setData(data.map(function(d) {{
        return {{ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }};
    }}));

    // 均线
    var maSeriesMap = {{}};
    maPeriods.forEach(function(p) {{
        if (p <= 0) return;
        var maSeries = mainChart.addLineSeries({{
            color: maColors[p] || '#999', lineWidth: 1,
            priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
        }});
        maSeries.setData(data.filter(function(d) {{ return d['ma' + p] != null; }})
            .map(function(d) {{ return {{ time: d.time, value: d['ma' + p] }}; }}));
        maSeriesMap[p] = maSeries;
    }});

    // 支撑阻力位
    srLines.forEach(function(sr) {{
        candleSeries.createPriceLine({{
            price: sr.price, color: sr.color, lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true, title: sr.title,
        }});
    }});

    // ---- 创建指标子图 ----
    var subChart = LightweightCharts.createChart(indicatorDiv, {{
        width: indicatorDiv.clientWidth,
        height: {indicator_height},
        layout: {{
            background: {{ type: 'solid', color: '#ffffff' }},
            textColor: '#787b86', fontSize: 11,
            fontFamily: "'PingFang SC','Microsoft YaHei',sans-serif",
        }},
        grid: {{
            vertLines: {{ color: '#f0f3fa' }},
            horzLines: {{ color: '#f0f3fa' }},
        }},
        crosshair: {{
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {{ color: '#787b86', width: 1, style: LightweightCharts.LineStyle.Dashed, labelBackgroundColor: '#2962ff' }},
            horzLine: {{ color: '#787b86', width: 1, style: LightweightCharts.LineStyle.Dashed, labelBackgroundColor: '#2962ff' }},
        }},
        rightPriceScale: {{ borderColor: '#e0e3eb' }},
        timeScale: {{ borderColor: '#e0e3eb', timeVisible: false, secondsVisible: false }},
        localization: {{ locale: 'zh-CN' }},
    }});

    // 当前活跃的指标系列
    var activeSeries = [];
    var activePriceLines = [];

    function clearSubChart() {{
        while (activeSeries.length > 0) {{
            var s = activeSeries.pop();
            try {{ subChart.removeSeries(s); }} catch(e) {{}}
        }}
        // 清除价格线
        activePriceLines.forEach(function(pl) {{
            try {{ pl; }} catch(e) {{}}
        }});
        activePriceLines = [];
    }}

    function renderIndicator(key) {{
        clearSubChart();

        if (key === 'vol') {{
            var volSeries = subChart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }}, priceLineVisible: false, lastValueVisible: false,
            }});
            volSeries.setData(data.map(function(d) {{
                return {{ time: d.time, value: d.volume, color: d.close >= d.open ? UP : DOWN }};
            }}));
            activeSeries.push(volSeries);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.15, bottom: 0.05 }} }});

        }} else if (key === 'macd') {{
            var difS = subChart.addLineSeries({{ color: '#1f77b4', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false }});
            var deaS = subChart.addLineSeries({{ color: '#ff7f0e', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false }});
            var histS = subChart.addHistogramSeries({{ priceLineVisible: false, lastValueVisible: false }});
            difS.setData(data.filter(function(d) {{ return d.dif != null; }}).map(function(d) {{ return {{ time: d.time, value: d.dif }}; }}));
            deaS.setData(data.filter(function(d) {{ return d.dea != null; }}).map(function(d) {{ return {{ time: d.time, value: d.dea }}; }}));
            histS.setData(data.map(function(d) {{
                if (d.macd == null) return {{ time: d.time, value: 0, color: 'rgba(0,0,0,0)' }};
                return {{ time: d.time, value: d.macd, color: d.macd >= 0 ? UP : DOWN }};
            }}));
            activeSeries.push(difS, deaS, histS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.15, bottom: 0.15 }} }});

        }} else if (key === 'rsi') {{
            var rsiS = subChart.addLineSeries({{ color: '#1f77b4', lineWidth: 1.5, priceLineVisible: true, lastValueVisible: true }});
            rsiS.setData(data.filter(function(d) {{ return d.rsi != null; }}).map(function(d) {{ return {{ time: d.time, value: d.rsi }}; }}));
            rsiS.createPriceLine({{ price: 70, color: UP, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '70' }});
            rsiS.createPriceLine({{ price: 50, color: '#999', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false }});
            rsiS.createPriceLine({{ price: 30, color: DOWN, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '30' }});
            activeSeries.push(rsiS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});

        }} else if (key === 'kdj') {{
            var kS = subChart.addLineSeries({{ color: '#1f77b4', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false }});
            var dS = subChart.addLineSeries({{ color: '#ff7f0e', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false }});
            var jS = subChart.addLineSeries({{ color: '#9467bd', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false }});
            kS.setData(data.filter(function(d) {{ return d.k != null; }}).map(function(d) {{ return {{ time: d.time, value: d.k }}; }}));
            dS.setData(data.filter(function(d) {{ return d.d != null; }}).map(function(d) {{ return {{ time: d.time, value: d.d }}; }}));
            jS.setData(data.filter(function(d) {{ return d.j != null; }}).map(function(d) {{ return {{ time: d.time, value: d.j }}; }}));
            kS.createPriceLine({{ price: 80, color: UP, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '80' }});
            kS.createPriceLine({{ price: 50, color: '#999', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false }});
            kS.createPriceLine({{ price: 20, color: DOWN, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '20' }});
            activeSeries.push(kS, dS, jS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});

        }} else if (key === 'boll') {{
            var upperS = subChart.addLineSeries({{ color: UP, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
            var midS = subChart.addLineSeries({{ color: '#999', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, priceLineVisible: false, lastValueVisible: false }});
            var lowerS = subChart.addLineSeries({{ color: DOWN, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }});
            upperS.setData(data.filter(function(d) {{ return d.boll_upper != null; }}).map(function(d) {{ return {{ time: d.time, value: d.boll_upper }}; }}));
            midS.setData(data.filter(function(d) {{ return d.boll_mid != null; }}).map(function(d) {{ return {{ time: d.time, value: d.boll_mid }}; }}));
            lowerS.setData(data.filter(function(d) {{ return d.boll_lower != null; }}).map(function(d) {{ return {{ time: d.time, value: d.boll_lower }}; }}));
            activeSeries.push(upperS, midS, lowerS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});

        }} else if (key === 'wr') {{
            var wrS = subChart.addLineSeries({{ color: '#1f77b4', lineWidth: 1.5, priceLineVisible: true, lastValueVisible: true }});
            wrS.setData(data.filter(function(d) {{ return d.wr != null; }}).map(function(d) {{ return {{ time: d.time, value: d.wr }}; }}));
            wrS.createPriceLine({{ price: -20, color: UP, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '-20' }});
            wrS.createPriceLine({{ price: -50, color: '#999', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false }});
            wrS.createPriceLine({{ price: -80, color: DOWN, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '-80' }});
            activeSeries.push(wrS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});

        }} else if (key === 'cci') {{
            var cciS = subChart.addLineSeries({{ color: '#1f77b4', lineWidth: 1.5, priceLineVisible: true, lastValueVisible: true }});
            cciS.setData(data.filter(function(d) {{ return d.cci != null; }}).map(function(d) {{ return {{ time: d.time, value: d.cci }}; }}));
            cciS.createPriceLine({{ price: 100, color: UP, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '100' }});
            cciS.createPriceLine({{ price: 0, color: '#999', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: false }});
            cciS.createPriceLine({{ price: -100, color: DOWN, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title: '-100' }});
            activeSeries.push(cciS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.1, bottom: 0.1 }} }});

        }} else if (key === 'obv') {{
            var obvS = subChart.addLineSeries({{ color: '#1f77b4', lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, priceFormat: {{ type: 'volume' }} }});
            obvS.setData(data.filter(function(d) {{ return d.obv != null; }}).map(function(d) {{ return {{ time: d.time, value: d.obv }}; }}));
            activeSeries.push(obvS);
            subChart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.15, bottom: 0.05 }} }});
        }}

        // 同步时间轴范围
        var range = mainChart.timeScale().getVisibleLogicalRange();
        if (range) subChart.timeScale().setVisibleLogicalRange(range);
    }}

    // 默认显示成交量
    renderIndicator('vol');

    // ---- 标签切换 ----
    Object.keys(tabs).forEach(function(key) {{
        tabs[key].addEventListener('click', function() {{
            // 更新标签样式
            Object.keys(tabs).forEach(function(k) {{
                tabs[k].style.background = '#fafbfc';
                tabs[k].style.color = '#787b86';
                tabs[k].style.fontWeight = '400';
                tabs[k].style.borderBottom = 'none';
            }});
            this.style.background = '#fff';
            this.style.color = '#262833';
            this.style.fontWeight = '600';
            this.style.borderBottom = '2px solid #2962ff';
            renderIndicator(key);
        }});
    }});

    // ---- 主图与子图时间轴联动 ----
    function syncRange(source, target) {{
        var range = source.timeScale().getVisibleLogicalRange();
        if (range) target.timeScale().setVisibleLogicalRange(range);
    }}

    mainChart.timeScale().subscribeVisibleLogicalRangeChange(function() {{
        syncRange(mainChart, subChart);
    }});
    subChart.timeScale().subscribeVisibleLogicalRangeChange(function() {{
        syncRange(subChart, mainChart);
    }});

    // 十字光标联动
    mainChart.subscribeCrosshairMove(function(param) {{
        if (!param.time || !param.point) return;
        subChart.setCrosshairPosition(param.point.y, param.time, candleSeries);
    }});
    subChart.subscribeCrosshairMove(function(param) {{
        if (!param.time || !param.point) return;
        mainChart.setCrosshairPosition(param.point.y, param.time, candleSeries);
    }});

    // ---- 自适应宽度 ----
    var resizeObserver = new ResizeObserver(function() {{
        var w = container.clientWidth - 2;
        mainChart.applyOptions({{ width: w }});
        subChart.applyOptions({{ width: w }});
    }});
    resizeObserver.observe(container);

    // 初始适配
    setTimeout(function() {{
        var w = container.clientWidth - 2;
        mainChart.applyOptions({{ width: w }});
        subChart.applyOptions({{ width: w }});
        // 默认显示最近1年（约250个交易日）的K线
        var totalBars = data.length;
        var visibleBars = Math.min(250, totalBars);
        mainChart.timeScale().setVisibleLogicalRange({{
            from: Math.max(0, totalBars - visibleBars),
            to: totalBars - 1 + 0.5,
        }});
    }}, 100);
}})();
</script>'''

        return html

    def _add_support_resistance_lines(self, fig: go.Figure, sr: Dict):
        """添加支撑阻力位水平线"""
        # 支撑位
        support_levels = sr.get('support', sr.get('support_levels', []))
        if isinstance(support_levels, (int, float)):
            support_levels = [support_levels]

        for level in support_levels:
            price = self._extract_level_price(level)
            if price is None or (isinstance(price, float) and np.isnan(price)):
                continue
            label = f"支撑: {price:.2f}"
            if isinstance(level, dict) and level.get('type'):
                label = f"支撑({level['type']}): {price:.2f}"
            fig.add_hline(
                y=price,
                line=dict(color=DOWN_COLOR, width=1, dash='dash'),
                annotation_text=label,
                annotation_position="bottom left",
                row=1, col=1,
            )

        # 阻力位
        resistance_levels = sr.get('resistance', sr.get('resistance_levels', []))
        if isinstance(resistance_levels, (int, float)):
            resistance_levels = [resistance_levels]

        for level in resistance_levels:
            price = self._extract_level_price(level)
            if price is None or (isinstance(price, float) and np.isnan(price)):
                continue
            label = f"阻力: {price:.2f}"
            if isinstance(level, dict) and level.get('type'):
                label = f"阻力({level['type']}): {price:.2f}"
            fig.add_hline(
                y=price,
                line=dict(color=UP_COLOR, width=1, dash='dash'),
                annotation_text=label,
                annotation_position="top left",
                row=1, col=1,
            )

    @staticmethod
    def _extract_level_price(level) -> Optional[float]:
        """从支撑阻力位数据中提取价格值"""
        if level is None:
            return None
        if isinstance(level, (int, float)):
            return float(level)
        if isinstance(level, dict):
            return level.get('price')
        return None
