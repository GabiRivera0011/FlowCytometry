from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from flowcyto.viz.heatmap import SEQUENTIAL_GREEN

TEXT_COLOR = "#e5e5e5"
NEUTRAL_COLOR = "#89cff0"  # baby blue


def _percentile_range(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    return float(lo), float(hi)


def build_marker_scatter3d_figure(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, x_label: str, y_label: str, z_label: str,
    color_values: np.ndarray | None = None, color_label: str | None = None,
    highlights: list[tuple[np.ndarray, str, str]] | None = None,
    pct_range: tuple[float, float] = (2.5, 97.5),
) -> go.Figure:
    """A literal 3D scatter of every subsampled event: X/Y/Z are three
    actual marker channels chosen directly (not an abstract PCA/UMAP
    embedding) — exactly like this app's 2D gating plots, just with a
    third axis, so each axis keeps its own real meaning and units.

    Coloring is one of two modes, not both at once (mixing a continuous
    gradient with discrete "in/out of gate" categories on the same points
    isn't readable):
    - `color_values`/`color_label`: a 4th marker's own (already
      display-transformed) expression, as a continuous gradient.
    - `highlights`: a list of `(mask, label, color)` tuples, each one
      previously-defined gate's population in its own solid color: every
      other point falls back to a neutral gray so the highlighted
      population(s) still stand out.
    """
    if color_values is not None:
        lo, hi = _percentile_range(color_values, *pct_range)
        marker = dict(
            size=3, color=color_values, colorscale=SEQUENTIAL_GREEN, cmin=lo, cmax=hi, reversescale=True,
            colorbar=dict(title=color_label, thickness=12, len=0.9), opacity=0.8,
        )
    else:
        marker = dict(size=3, color=NEUTRAL_COLOR, opacity=0.5)

    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker, name="All events", showlegend=False))

    for mask, label, color in highlights or []:
        if mask is None or not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=x[mask], y=y[mask], z=z[mask], mode="markers",
            marker=dict(size=4, color=color, opacity=0.9),
            name=label, showlegend=True,
        ))

    # A black scene AND paper background (rather than white or gray) gives
    # the pale end of the green colorscale something to actually pop
    # against, with no mismatched white margin around the cube.
    scene_pane = dict(
        backgroundcolor="#000000", gridcolor="#4d4d4d", zerolinecolor="#4d4d4d",
        linecolor="#4d4d4d", showbackground=True, tickfont=dict(color="#d6d6d6"),
    )

    def _axis(title: str) -> dict:
        return dict(title=dict(text=title, font=dict(color="#d6d6d6")), **scene_pane)

    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR, height=650,
        paper_bgcolor="#000000",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        scene=dict(xaxis=_axis(x_label), yaxis=_axis(y_label), zaxis=_axis(z_label)),
    )
    return fig
