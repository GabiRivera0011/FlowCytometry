from __future__ import annotations

import plotly.graph_objects as go

TEXT_COLOR = "#0b0b0b"

# matplotlib's Set3 (ColorBrewer qualitative, 12 colors) — categorical use only.
# Never for magnitude/signed data: Set3 has no inherent low->high or -/+ order,
# unlike SEQUENTIAL_BLUE/DIVERGING_RED_GRAY_BLUE in viz/heatmap.py.
SET3 = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462",
    "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f",
]


def set3_color(index: int) -> str:
    return SET3[index % len(SET3)]


def build_before_after_bar(
    labels: list[str], before, after, before_label: str, after_label: str, title: str, xaxis_title: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name=before_label, y=labels, x=before, orientation="h", marker_color=set3_color(0))
    fig.add_bar(name=after_label, y=labels, x=after, orientation="h", marker_color=set3_color(3))
    fig.update_layout(
        barmode="group", template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=xaxis_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def build_ranked_bar(pair_labels: list[str], values, title: str, xaxis_title: str, color_index: int = 0) -> go.Figure:
    fig = go.Figure(go.Bar(y=pair_labels, x=values, orientation="h", marker_color=set3_color(color_index)))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),
    )
    return fig
