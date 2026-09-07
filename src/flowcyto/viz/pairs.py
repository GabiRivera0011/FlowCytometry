from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flowcyto.viz.bars import set3_color

TEXT_COLOR = "#0b0b0b"


def rank_notable_pairs(matrix: pd.DataFrame, top_k: int = 16, seed: int = 0) -> list[tuple[float, str, str]]:
    """Rank off-diagonal (from, into) pairs by |matrix value| — an O(1) lookup
    against the live matrix, so this stays current without an extra data pass.

    Shuffled before sorting so ties break randomly rather than by insertion
    order — on an identity matrix (every pair tied at 0, the common case for
    this dataset) a stable sort would otherwise always return one channel's
    entire row instead of a representative sample."""
    values = matrix.values
    n = values.shape[0]
    pairs = [
        (abs(values[i, j]), matrix.index[i], matrix.columns[j])
        for i in range(n) for j in range(n) if i != j
    ]
    np.random.default_rng(seed).shuffle(pairs)
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs[:top_k]


def build_notable_pairs_grid(
    compensated_subsample: np.ndarray, channel_names: list[str], pairs: list[tuple[float, str, str]], n_cols: int = 4,
) -> go.Figure:
    """One Scattergl trace per pair — not a shared Splom trace — so a click's
    curve_number reliably indexes back into `pairs`. A Splom's curve_number is
    constant across all its cells and can't identify which one was clicked."""
    n_rows = -(-len(pairs) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[f"{f} vs {t}" for _, f, t in pairs])
    index_of = {name: i for i, name in enumerate(channel_names)}
    for k, (_, from_name, into_name) in enumerate(pairs):
        row, col = k // n_cols + 1, k % n_cols + 1
        fig.add_trace(
            go.Scattergl(
                x=compensated_subsample[:, index_of[from_name]],
                y=compensated_subsample[:, index_of[into_name]],
                mode="markers", marker=dict(size=3, color=set3_color(k), opacity=0.5),
                showlegend=False,
            ),
            row=row, col=col,
        )
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR, height=220 * n_rows,
        title="Most notable channel pairs (click one to zoom in below)",
    )
    fig.update_annotations(font_color=TEXT_COLOR)
    return fig


def build_pair_detail_figure(
    compensated_subsample: np.ndarray, channel_names: list[str], from_name: str, into_name: str,
) -> go.Figure:
    index_of = {name: i for i, name in enumerate(channel_names)}
    fig = go.Figure(go.Scattergl(
        x=compensated_subsample[:, index_of[from_name]],
        y=compensated_subsample[:, index_of[into_name]],
        mode="markers", marker=dict(size=4, color=set3_color(0), opacity=0.5),
    ))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=f"{from_name} vs {into_name}", xaxis_title=from_name, yaxis_title=into_name,
        height=500,
    )
    return fig
