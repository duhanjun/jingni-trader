"""分位图渲染器"""
import logging
from typing import Dict, Any

logger = logging.getLogger("renderers.percentile_chart")


def render_percentile_chart(factor_values: Dict[str, Any], hint: str = "") -> str:
    """渲染分位图（PE/PB 历史分位）"""
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

        if 'percentile' in name and val is not None:
            try:
                pct = float(val) * 100
                level = "高估" if pct > 80 else ("低估" if pct < 20 else "合理")
                level_cls = "overvalued" if pct > 80 else ("undervalued" if pct < 20 else "fair")
                items.append(f'''
                <div class="percentile-item">
                    <div class="percentile-label">{label}</div>
                    <div class="percentile-bar-container">
                        <div class="percentile-bar {level_cls}" style="width: {pct:.1f}%"></div>
                    </div>
                    <div class="percentile-value">{pct:.1f}% <span class="level-tag {level_cls}">{level}</span></div>
                </div>''')
            except (ValueError, TypeError):
                items.append(f'<div class="percentile-item"><div class="percentile-label">{label}</div><div class="percentile-value">—</div></div>')
        else:
            formatted = _format_value(val)
            items.append(f'<div class="percentile-item"><div class="percentile-label">{label}</div><div class="percentile-value">{formatted}</div></div>')

    hint_html = f'<p class="analysis-hint">{hint}</p>' if hint else ''
    return f'{hint_html}<div class="percentile-list">{"".join(items)}</div>'


def _format_value(val: Any) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "—"
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)
