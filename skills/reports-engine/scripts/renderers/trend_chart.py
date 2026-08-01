"""趋势图渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.trend_chart")


def render_trend_chart(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染趋势图（当前值+方向标签）"""
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
        direction = ""
        direction_cls = ""
        if val is not None:
            try:
                v = float(val)
                formatted = f"{v:.2f}%"
                if v > 0:
                    direction = "↑"
                    direction_cls = "trend-up"
                elif v < 0:
                    direction = "↓"
                    direction_cls = "trend-down"
            except (ValueError, TypeError):
                formatted = str(val)

        items.append(f'''
        <div class="trend-item">
            <span class="trend-label">{label}</span>
            <span class="trend-value {direction_cls}">{formatted} {direction}</span>
        </div>''')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    return f'{hint_html}<div class="trend-list">{"".join(items)}</div>'
