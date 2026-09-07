from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

TEXT_COLOR = "#0b0b0b"
MARKER_COLOR = "#2a78d6"


def build_scatter_figure(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str) -> go.Figure:
    fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=3, color=MARKER_COLOR, opacity=0.4)))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=x_label, yaxis_title=y_label,
    )
    return fig


def build_histogram_figure(values: np.ndarray, channel_name: str, nbins: int = 100) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, marker_color=MARKER_COLOR, nbinsx=nbins))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=channel_name, xaxis_title=channel_name,
    )
    return fig
