from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

TEXT_COLOR = "#0b0b0b"
MARKER_COLOR = "#014421"


def _percentile_range(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    # explanation: autoscale to a central percentile band, not the full
    # min-max — a few extreme outliers otherwise stretch an axis far past
    # where the bulk of the data actually sits.
    lo, hi = np.percentile(values, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    return float(lo), float(hi)


def build_scatter_figure(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str) -> go.Figure:
    # Each axis autoscales independently to its own central 95% (2.5-97.5
    # percentile) of the selected channel's data.
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=3, color=MARKER_COLOR, opacity=0.4)))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=x_label, yaxis_title=y_label,
    )
    fig.update_xaxes(range=_percentile_range(x, 2.5, 97.5))
    fig.update_yaxes(range=_percentile_range(y, 2.5, 97.5))
    return fig


def build_histogram_figure(values: np.ndarray, channel_name: str, nbins: int = 100) -> go.Figure:
    lo, hi = _percentile_range(values, 0.5, 99.5)
    fig = go.Figure(go.Histogram(
        x=values, marker_color=MARKER_COLOR,
        xbins=dict(start=lo, end=hi, size=(hi - lo) / nbins),
    ))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=channel_name, xaxis_title=channel_name,
    )
    fig.update_xaxes(range=[lo, hi])
    return fig
