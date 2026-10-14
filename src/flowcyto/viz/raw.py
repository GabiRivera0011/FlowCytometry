from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, map_coordinates

TEXT_COLOR = "#0b0b0b"
MARKER_COLOR = "#014421"
BACKDROP_COLOR = "#c9c9c9"


def _percentile_range(values: np.ndarray, low_pct: float, high_pct: float) -> tuple[float, float]:
    # explanation: autoscale to a central percentile band, not the full
    # min-max — a few extreme outliers otherwise stretch an axis far past
    # where the bulk of the data actually sits.
    lo, hi = np.percentile(values, [low_pct, high_pct])
    if hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    return float(lo), float(hi)


def _smoothed_density(x: np.ndarray, y: np.ndarray, bins: int = 100, sigma: float = 1.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A 2D histogram plus a Gaussian blur: a fast stand-in for a true kernel
    # density estimate. FlowJo's pseudocolor/contour plots are themselves
    # built on a smoothed density surface, not a raw per-bin count — using
    # the raw count directly (our first attempt) gave visible flat "patches"
    # wherever many points shared one bin, instead of a continuous gradient.
    h, xedges, yedges = np.histogram2d(x, y, bins=bins)
    h = gaussian_filter(h, sigma=sigma)
    return h, xedges, yedges


def _point_density(x: np.ndarray, y: np.ndarray, bins: int = 100, sigma: float = 1.6) -> np.ndarray:
    # Per-point density, bilinearly interpolated from the smoothed grid.
    # Nearest-bin lookup (np.digitize) still shows a visible block edge
    # wherever two neighboring points fall in different bins, even once the
    # grid itself is smoothed — interpolating between bin centers removes
    # that last bit of blockiness and gives a genuinely continuous gradient.
    h, xedges, yedges = _smoothed_density(x, y, bins=bins, sigma=sigma)
    xwidth = (xedges[-1] - xedges[0]) / bins
    ywidth = (yedges[-1] - yedges[0]) / bins
    xi = np.clip((x - xedges[0]) / xwidth - 0.5, 0, bins - 1)
    yi = np.clip((y - yedges[0]) / ywidth - 0.5, 0, bins - 1)
    return map_coordinates(h, [xi, yi], order=1, mode="nearest")


def _contour_grid(x: np.ndarray, y: np.ndarray, bins: int = 100, sigma: float = 1.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # log1p-compressed, smoothed 2D density on a grid, for contour/zebra. A
    # lighter blur than the pseudocolor grid — enough to draw a clean line,
    # not so much that the outline balloons past where the dots actually sit.
    h, xedges, yedges = _smoothed_density(x, y, bins=bins, sigma=sigma)
    xcenters, ycenters = (xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2
    return np.log1p(h.T), xcenters, ycenters


def _contour_levels(z: np.ndarray, n_levels: int, low_pct: float = 55.0) -> tuple[float, float, float]:
    # Start contouring above the bottom `low_pct` of nonzero density, not at
    # the fringe of a single stray point — log-compression otherwise pulls
    # the outermost line out to hug every sparse outlier, making the contour
    # look far bigger than the actual dot cloud it's supposed to trace.
    nonzero = z[z > 0]
    if len(nonzero) == 0:
        return 0.0, 1.0, 1.0
    z_min = float(np.percentile(nonzero, low_pct))
    z_max = float(z.max())
    if z_max <= z_min:
        z_min = 0.0
    size = (z_max - z_min) / n_levels if z_max > z_min else 1.0
    return z_min, z_max, size


def _zebra_colorscale(n_bands: int = 8) -> list:
    # A hard-edged (not gradient) alternating colorscale: each band is listed
    # as two identical stops at its start/end fraction, which is the standard
    # Plotly trick for a stepped rather than interpolated colorscale.
    scale = []
    for i in range(n_bands):
        color = "#f5f5f5" if i % 2 == 0 else "#111111"
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
        # Density-only reference views (FlowJo-style contour/zebra), drawn
        # from a 2D density surface rather than one mark per event.
        z, xcenters, ycenters = _contour_grid(x, y)
        if mode == "zebra":
            z_start, z_end, z_size = _contour_levels(z, n_levels=8)
            fig = go.Figure(go.Contour(
                x=xcenters, y=ycenters, z=z, showscale=False,
                contours=dict(start=z_start, end=z_end, size=z_size, coloring="fill", showlines=True),
                colorscale=_zebra_colorscale(), line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
            ))
        else:
            # FlowJo's classic contour plot: solid black outline levels on a
            # plain white background. `coloring="lines"` would still color
            # each level off the colorscale — "none" is what makes every
            # line just take the constant `line.color` below.
            z_start, z_end, z_size = _contour_levels(z, n_levels=12)
            fig = go.Figure(go.Contour(
                x=xcenters, y=ycenters, z=z, showscale=False,
                contours=dict(start=z_start, end=z_end, size=z_size, coloring="none", showlabels=False),
                line=dict(width=1.1, color="#1a1a1a"),
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
        # Keeps the user's own zoom/pan across a rerun that redraws this same
        # plot (e.g. adjusting a gate) — without it, Plotly treats each
        # redraw as a brand new figure and snaps back to `x_range`/`y_range`
        # below, which is exactly the "zoom resets while gating" bug this
        # fixes. Stable as long as the channel pair (encoded in `title`)
        # doesn't change; a genuinely different plot still resets normally.
        uirevision=title,
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
        uirevision=channel_name,
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
        fig.update_layout(template="simple_white", font_color=TEXT_COLOR, title=f"{channel_name} (log)", uirevision=f"{channel_name}_log")
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
        uirevision=f"{channel_name}_log",
    )
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    return fig


def build_gated_histogram_figure(
    all_values: np.ndarray, gated_values: np.ndarray, channel_name: str, color: str, nbins: int = 100,
) -> go.Figure:
    """Full incoming population as a fixed gray backdrop, with the currently
    gated subset overlaid in color on the same bins. The axis range and bin
    edges come only from `all_values`, so the histogram itself never moves —
    only the highlighted overlay changes shape as the gate is adjusted."""
    lo, hi = _percentile_range(all_values, 0.5, 99.5)
    xbins = dict(start=lo, end=hi, size=(hi - lo) / nbins)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=all_values, name="All events", marker_color=BACKDROP_COLOR, xbins=xbins, opacity=0.7))
    fig.add_trace(go.Histogram(x=gated_values, name="Gated", marker_color=color, xbins=xbins, opacity=0.9))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR, barmode="overlay",
        title=channel_name, xaxis_title=channel_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uirevision=channel_name,
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def build_gated_log_histogram_figure(
    all_values: np.ndarray, gated_values: np.ndarray, channel_name: str, color: str, nbins: int = 60,
) -> go.Figure:
    """Log-scale counterpart of `build_gated_histogram_figure`: bins are
    fixed from `all_values` (evenly spaced in log space), and the gated
    subset is rebinned onto those same edges so the two overlay cleanly."""
    positive_all = all_values[all_values > 0]
    if len(positive_all) < 2:
        fig = go.Figure()
        fig.update_layout(template="simple_white", font_color=TEXT_COLOR, title=f"{channel_name} (log)", uirevision=f"{channel_name}_log")
        return fig

    lo, hi = _percentile_range(positive_all, 0.5, 99.5)
    lo = max(lo, float(positive_all.min()))
    if hi <= lo:
        lo, hi = float(positive_all.min()), float(positive_all.max())
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = edges[1:] - edges[:-1]

    all_counts, _ = np.histogram(positive_all, bins=edges)
    gated_counts, _ = np.histogram(gated_values[gated_values > 0], bins=edges)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=all_counts, width=widths, name="All events", marker_color=BACKDROP_COLOR, opacity=0.7))
    fig.add_trace(go.Bar(x=centers, y=gated_counts, width=widths, name="Gated", marker_color=color, opacity=0.9))
    fig.update_layout(
        template="simple_white", font_color=TEXT_COLOR, barmode="overlay",
        title=f"{channel_name} (log)", xaxis_title=channel_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uirevision=f"{channel_name}_log",
    )
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)])
    return fig
