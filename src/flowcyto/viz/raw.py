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


def _point_density(x: np.ndarray, y: np.ndarray, bins: int = 60) -> np.ndarray:
    # Per-point local density via 2D-histogram binning: fast (O(n)), and the
    # same approach FlowJo-style pseudocolor plots use under the hood.
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xi = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    return h[xi, yi]


def _density_grid(x: np.ndarray, y: np.ndarray, bins: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # log1p-compressed 2D density on a grid, for contour/zebra fills. Flow
    # data is heavily right-skewed (a near-empty background, one sharp
    # peak) — contouring the raw counts leaves every band but the peak's
    # own tip looking identical, so we log-compress the dynamic range first.
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    xcenters, ycenters = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    return np.log1p(h.T), xcenters, ycenters


def _zebra_colorscale(n_bands: int = 8) -> list:
    # A hard-edged (not gradient) alternating colorscale: each band is listed
    # as two identical stops at its start/end fraction, which is the standard
    # Plotly trick for a stepped rather than interpolated colorscale.
    scale = []
    for i in range(n_bands):
        color = "#111111" if i % 2 == 0 else "#f5f5f5"
        scale.append([i / n_bands, color])
        scale.append([(i + 1) / n_bands, color])
    return scale


def build_scatter_figure(
    x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str,
    pct_range: tuple[float, float] = (2.5, 97.5), color: str = MARKER_COLOR, mode: str = "dot",
) -> go.Figure:
    # Each axis autoscales independently to its own central band of the
    # selected channel's data (2.5-97.5 percentile, i.e. 95%, by default).
    x_range, y_range = _percentile_range(x, *pct_range), _percentile_range(y, *pct_range)

    if mode in ("contour", "zebra"):
        # Density-only reference views (FlowJo-style contour/zebra): drawn
        # from a 2D density surface, not one mark per event, so box/lasso
        # gating can't map back to individual events here — the caller keeps
        # dragmode off "select" for these two modes.
        z, xcenters, ycenters = _density_grid(x, y)
        if mode == "zebra":
            contour_kwargs = dict(colorscale=_zebra_colorscale(), showscale=False, ncontours=10)
        else:
            contour_kwargs = dict(colorscale="Jet", showscale=True, ncontours=15)
        fig = go.Figure(go.Contour(
            x=xcenters, y=ycenters, z=z, contours=dict(coloring="fill", showlines=True),
            line=dict(width=0.5, color="rgba(0,0,0,0.4)"), **contour_kwargs,
        ))
    else:
        if mode == "pseudocolor":
            # FlowJo-style density plot: still one point per event (so
            # box/lasso gating keeps working identically), just colored by
            # local density on the classic blue-to-red "Jet" rainbow instead
            # of a flat color.
            marker = dict(
                size=3, color=_point_density(x, y), colorscale="Jet", opacity=0.85,
                colorbar=dict(title="density", thickness=12, len=0.9),
            )
        else:
            marker = dict(size=3, color=color, opacity=0.6, line=dict(width=0.5, color="#4d4d4d"))
        fig = go.Figure(go.Scattergl(x=x, y=y, mode="markers", marker=marker))

    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=title, xaxis_title=x_label, yaxis_title=y_label,
    )
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    return fig


def build_histogram_figure(values: np.ndarray, channel_name: str, nbins: int = 100, color: str = MARKER_COLOR) -> go.Figure:
    lo, hi = _percentile_range(values, 0.5, 99.5)
    fig = go.Figure(go.Histogram(
        x=values, marker_color=color,
        xbins=dict(start=lo, end=hi, size=(hi - lo) / nbins),
    ))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=channel_name, xaxis_title=channel_name,
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def build_log_histogram_figure(values: np.ndarray, channel_name: str, nbins: int = 60, color: str = MARKER_COLOR) -> go.Figure:
    """Log-scale histogram with bins evenly spaced in log space (not the
    linear bins `go.Histogram` would otherwise draw unevenly on a log axis).
    Non-positive values have no log and are dropped."""
    positive = values[values > 0]
    if len(positive) < 2:
        fig = go.Figure()
        fig.update_layout(template="simple_white", font_color=TEXT_COLOR, title=f"{channel_name} (log)")
        return fig

    lo, hi = _percentile_range(positive, 0.5, 99.5)
    lo = max(lo, float(positive.min()))
    if hi <= lo:
        lo, hi = float(positive.min()), float(positive.max())
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    counts, _ = np.histogram(positive, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric center: correct for log-spaced bins
    widths = edges[1:] - edges[:-1]

    fig = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color=color))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR,
        title=f"{channel_name} (log)", xaxis_title=channel_name,
    )
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    return fig
