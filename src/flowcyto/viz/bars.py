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


# The four darkest Set3 hues by relative luminance (purple, coral, blue,
# lavender) — used on the gating pages, where markers, outlines, and card
# borders all need to read clearly against a white background.
DARK_SET3 = ["#bc80bd", "#fb8072", "#80b1d3", "#bebada"]


def dark_set3_color(index: int) -> str:
    return DARK_SET3[index % len(DARK_SET3)]


def build_before_after_bar(
    labels: list[str], before, after, before_label: str, after_label: str, title: str, xaxis_title: str,
    before_color: str = SET3[0], after_color: str = SET3[3],
    xaxis_range: tuple[float, float] | None = None, reference_line_x: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(name=before_label, y=labels, x=before, orientation="h", marker_color=before_color)
    fig.add_bar(name=after_label, y=labels, x=after, orientation="h", marker_color=after_color)
    fig.update_layout(
        barmode="group", template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=xaxis_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if xaxis_range is not None:
        fig.update_xaxes(range=list(xaxis_range))
    if reference_line_x is not None:
        fig.add_vline(x=reference_line_x, line_dash="dash", line_color=TEXT_COLOR, line_width=1)
    return fig


def build_ranked_bar(
    pair_labels: list[str], values, title: str, xaxis_title: str, color_index: int = 0,
    xaxis_range: tuple[float, float] | None = None, reference_line_x: float | None = None,
) -> go.Figure:
    fig = go.Figure(go.Bar(y=pair_labels, x=values, orientation="h", marker_color=set3_color(color_index)))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=xaxis_title,
        yaxis=dict(autorange="reversed"),
    )
    if xaxis_range is not None:
        fig.update_xaxes(range=list(xaxis_range))
    if reference_line_x is not None:
        fig.add_vline(x=reference_line_x, line_dash="dash", line_color=TEXT_COLOR, line_width=1)
    return fig
