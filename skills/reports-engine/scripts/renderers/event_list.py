"""事件列表渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.event_list")


def render_event_list(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染事件列表（龙虎榜等）"""
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

        formatted = "—"
        if val is not None:
            try:
                v = float(val)
                if 'count' in name:
                    formatted = f"{int(v)} 次"
                elif abs(v) >= 1e8:
                    formatted = f"{v/1e8:.2f} 亿"
                elif abs(v) >= 1e4:
                    formatted = f"{v/1e4:.2f} 万"
                else:
                    formatted = f"{v:.2f}"
            except (ValueError, TypeError):
                formatted = str(val)

        items.append(f'''
        <div class="event-item">
            <span class="event-label">{label}</span>
            <span class="event-value">{formatted}</span>
        </div>''')

    if not items:
        items.append('<div class="event-item"><span class="event-label">近期无上榜记录</span></div>')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    return f'{hint_html}<div class="event-list">{"".join(items)}</div>'
