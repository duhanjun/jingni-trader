"""通道图渲染器（布林带）"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.band_chart")


def render_band_chart(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染布林带通道"""
    ub = factor_values.get("boll_ub")
    mid = factor_values.get("boll_mid")
    lb = factor_values.get("boll_lb")

    rows = []
    rows.append(_make_row("上轨 (UB)", ub, "upper"))
    rows.append(_make_row("中轨 (MID)", mid, "mid"))
    rows.append(_make_row("下轨 (LB)", lb, "lower"))

    # 通道宽度
    width = ""
    if ub is not None and lb is not None and mid is not None and mid != 0:
        try:
            w = (float(ub) - float(lb)) / float(mid) * 100
            width = f'<div class="band-width">通道宽度: {w:.2f}%</div>'
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    table_html = f'<table class="band-table">{"".join(rows)}</table>'
    return f'{hint_html}{table_html}{width}'


def _make_row(label: str, val: Any, cls: str) -> str:
    formatted = "—"
    if val is not None:
        try:
            formatted = f"{float(val):.2f}"
        except (ValueError, TypeError):
            formatted = str(val)
    return f'<tr class="{cls}"><td>{label}</td><td>{formatted}</td></tr>'
