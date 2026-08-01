"""K线形态信号列表渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.pattern_list")


def render_pattern_list(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染K线形态信号列表"""
    try:
        from scripts.factors.registry import get_factor_meta
    except Exception:
        get_factor_meta = None

    items = []
    for name, val in factor_values.items():
        label = name
        if get_factor_meta:
            meta = get_factor_meta(name)
            if meta:
                label = meta.description
        formatted = str(val) if val is not None else "—"
        items.append(f'<div class="pattern-item"><span class="pattern-label">{label}</span><span class="pattern-value">{formatted}</span></div>')

    if not items:
        items.append('<div class="pattern-item"><span class="pattern-label">近期无明显K线形态信号</span></div>')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    return f'{hint_html}<div class="pattern-list">{"".join(items)}</div>'
