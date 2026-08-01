"""指标面板渲染器（带信号标签）"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.indicator_panel")


def render_indicator_panel(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染指标面板（含信号标签）"""
    try:
        from scripts.factors.registry import get_factor_meta
    except Exception:
        get_factor_meta = None

    cards = []
    for name, val in factor_values.items():
        label = name
        if get_factor_meta:
            meta = get_factor_meta(name)
            if meta:
                label = meta.description

        formatted = _format_value(val)
        signal = _get_signal(name, val)
        signal_html = f'<span class="signal-tag {signal["cls"]}">{signal["text"]}</span>' if signal else ''

        cards.append(f'''
        <div class="metric-card">
            <div class="metric-value">{formatted} {signal_html}</div>
            <div class="metric-label">{label}</div>
        </div>''')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    return f'{hint_html}<div class="metrics-grid">{"".join(cards)}</div>'


def _format_value(val: Any) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def _get_signal(name: str, val: Any) -> Dict[str, str]:
    """根据因子名和值返回信号标签"""
    if val is None or (isinstance(val, float) and val != val):
        return {}
    try:
        v = float(val)
    except (ValueError, TypeError):
        return {}

    if name == "rsi_14":
        if v > 70:
            return {"cls": "signal-bearish", "text": "超买"}
        if v < 30:
            return {"cls": "signal-bullish", "text": "超卖"}
        return {"cls": "signal-neutral", "text": "中性"}
    if name == "kdj_j":
        if v > 100:
            return {"cls": "signal-bearish", "text": "超买"}
        if v < 0:
            return {"cls": "signal-bullish", "text": "超卖"}
        return {"cls": "signal-neutral", "text": "中性"}
    if name == "macd_hist":
        if v > 0:
            return {"cls": "signal-bullish", "text": "多头"}
        return {"cls": "signal-bearish", "text": "空头"}
    if name == "wr":
        if v > -20:
            return {"cls": "signal-bearish", "text": "超买"}
        if v < -80:
            return {"cls": "signal-bullish", "text": "超卖"}
        return {"cls": "signal-neutral", "text": "中性"}
    return {}
