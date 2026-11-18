from __future__ import annotations

import plotly.graph_objects as go

# Sequential single-hue ramp for magnitude (0-1+) — matches the dataviz skill's
# validated default. Never used for a signed quantity (see DIVERGING below).
SEQUENTIAL_BLUE = [
    (0.00, "#cde2fb"), (0.15, "#9ec5f4"), (0.30, "#6da7ec"), (0.45, "#3987e5"),
    (0.60, "#2a78d6"), (0.75, "#1c5cab"), (0.90, "#104281"), (1.00, "#0d366b"),
]

# Sequential single-hue ramp (forest green), same light->dark structure as
# SEQUENTIAL_BLUE above but anchored on this app's existing dark-green accent
# (MARKER_COLOR, "#014421" in viz/raw.py) at the top of the range.
SEQUENTIAL_GREEN = [
    (0.00, "#d9f0d3"), (0.15, "#b7e0ac"), (0.30, "#91cb86"), (0.45, "#69b668"),
    (0.60, "#479a4c"), (0.75, "#2d7a35"), (0.90, "#1a5c24"), (1.00, "#0d3d16"),
]

# Diverging red<->blue through neutral gray at 0, for signed metrics (the slope
# diagnostic) — sign is the signal there, which a sequential scale would hide.
DIVERGING_RED_GRAY_BLUE = [(0, "#e34948"), (0.5, "#f0efec"), (1, "#2a78d6")]

TEXT_COLOR = "#0b0b0b"


def build_spillover_heatmap(matrix, labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix.values if hasattr(matrix, "values") else matrix,
        x=labels, y=labels,
        colorscale=SEQUENTIAL_BLUE, zmin=0, zmax=1,
        hovertemplate="spillover from %{y}<br>into %{x}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="fraction"),
    ))
    fig.update_layout(
        title=title, xaxis_title="into channel", yaxis_title="from channel",
        template="simple_white", font_color=TEXT_COLOR,
        yaxis_autorange="reversed",
    )
    return fig


def build_slope_heatmap(slopes, labels: list[str], title: str, bound: float) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=slopes.values if hasattr(slopes, "values") else slopes,
        x=labels, y=labels,
        colorscale=DIVERGING_RED_GRAY_BLUE, zmin=-bound, zmax=bound,
        hovertemplate="slope of %{x}<br>on %{y}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="slope"),
    ))
    fig.update_layout(
        title=title, xaxis_title="channel (y-axis)", yaxis_title="regressed against",
        template="simple_white", font_color=TEXT_COLOR,
        yaxis_autorange="reversed",
    )
    return fig
