"""Shared interactive gating UI, used by every gating page (Live Cells,
Broad Cellular Lineage, ...). Streamlit-coupled by design (session_state,
plotly_chart events) — per docs/ARCHITECTURE.md this belongs in `app/`, not
the framework-agnostic `src/flowcyto/`."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_scatter_figure

GATE_PCT = (5, 95)  # autoscale each axis to its central 90% of the data

# Plotly resets a plot's zoom/pan to its explicit axis range any time the
# figure's shapes change — true even with a matching `uirevision`, confirmed
# by direct probing (isolated from this app's own code). Since every gate
# edit redraws that gate's shape, mouse-driven zoom can never survive a gate
# edit here. This is the actual, persistent way to "zoom out": a normal
# session_state-backed widget, so it just works like any other Streamlit
# control — no reliance on the browser remembering anything across a rerun.
AXIS_ZOOM_OPTIONS = {
    "Central 90% (default)": GATE_PCT,
    "Central 98%": (1, 99),
    "Central 99.8%": (0.1, 99.9),
    "All events": (0, 100),
}


def _persisted_default(key: str, fallback):
    """The last value `_persist_value` saved for `key`, or `fallback` if
    none exists yet. Streamlit drops a widget's own stored value the
    moment that widget stops being rendered for even one run — which
    happens any time the user visits a different page and comes back —
    silently resetting it to its default. A plain (non-widget) mirror key
    survives that, since it isn't part of Streamlit's widget registry."""
    return st.session_state.get(f"{key}__persist", fallback)


def _persist_value(key: str, current) -> None:
    """Call right after the widget with the same `key`, passing its
    return value, to save it into the mirror `_persisted_default` reads."""
    st.session_state[f"{key}__persist"] = current


def collect_section_gates(section: str) -> list[dict]:
    """Every gate currently drawn in one section, read straight from
    session_state rather than by re-rendering it — for read-only views
    (a tabulation page, a cross-page population picker) that need every
    gate across every gating page, not just whichever one is on screen."""
    n_gates = st.session_state.get(f"n_gates_{section}", 1)
    entries = []
    for i in range(n_gates):
        mask = st.session_state.get(f"gate_{section}_{i}_mask")
        if mask is None:
            continue
        entries.append({
            "name": st.session_state.get(f"gate_{section}_{i}_name", f"Gate {i + 1}"),
            "mask": mask,
            "color": dark_set3_color(i),
            "ref": ("gate", section, i),
        })
    return entries


# --- Exact full-dataset resolution -----------------------------------
#
# Every gate the UI draws only ever touches the fixed ~20k-event plotting
# subsample, for speed. To answer "how many cells are *actually* in this
# gate" (not a ratio-scaled estimate), each gate's own boundary — not its
# subsample mask — gets replayed against the full, un-subsampled dataset:
# recursively resolving whatever incoming population it was drawn against,
# all the way back to "the full file," then re-applying each gate's exact
# shape at that resolution. A "ref" is how one gate or picker choice points
# at another, so this chain can be walked after the fact, independent of
# whichever page happens to be on screen:
#   ("full",)                - the entire file, unconditionally
#   ("gate", section, index) - one specific drawn gate
#   ("picker", key)          - one population-picker's chosen combination


def _mask_from_shape(shape: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The same geometry test as `_mask_from_selection`, but from a saved
    shape dict (what every gate persists) instead of a live Plotly
    selection event — so a gate's exact boundary can be replayed against
    any x/y arrays, including the full dataset, long after it was drawn."""
    kind = shape.get("type")
    if kind == "box":
        x0, x1 = shape["x"]
        y0, y1 = shape["y"]
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if kind == "points":
        mask = np.zeros(len(x), dtype=bool)
        for px, py in zip(shape["x"], shape["y"]):
            mask |= (x == px) & (y == py)
        return mask
    if kind == "cross":
        cx, cy = shape["x"], shape["y"]
        quadrant = shape.get("quadrant", "UR")
        x_ge = x >= cx if "R" in quadrant else x < cx
        y_ge = y >= cy if quadrant.startswith("U") else y < cy
        return x_ge & y_ge
    # lasso
    return _points_in_polygon(x, y, np.asarray(shape["x"]), np.asarray(shape["y"]))


def resolve_full_mask(ref, channel_fn, n_full: int, cache: dict) -> np.ndarray:
    """Recursively resolve a ref to its exact mask over the full,
    un-subsampled dataset. `channel_fn(channel_name)` must return that
    channel's (compensated) values for every event in the full file.
    `cache` memoizes within one resolution pass — callers should pass the
    same dict across all the refs they resolve, so shared ancestors (most
    gates eventually trace back to the same Live Cells gate) are only
    computed once."""
    if ref is None:
        ref = ("full",)
    if ref in cache:
        return cache[ref]

    kind = ref[0]
    if kind == "gate":
        _, section, index = ref
        shape = st.session_state.get(f"gate_{section}_{index}_shape")
        incoming_ref = st.session_state.get(f"incoming_ref_{section}", ("full",))
        incoming_full = resolve_full_mask(incoming_ref, channel_fn, n_full, cache)
        x_label = st.session_state.get(f"section_{section}_x_label")
        y_label = st.session_state.get(f"section_{section}_y_label")
        if shape is None or x_label is None or y_label is None:
            # Never actually drawn (still the default "Gate 1") — passes
            # its incoming population through unmodified, same as the
            # subsample side does.
            result = incoming_full
        else:
            x_full, y_full = channel_fn(x_label), channel_fn(y_label)
            result = _mask_from_shape(shape, x_full, y_full) & incoming_full
    elif kind == "picker":
        _, key = ref
        definition = st.session_state.get(f"{key}__def")
        refs = (definition or {}).get("refs") or []
        if not refs:
            result = np.ones(n_full, dtype=bool)
        else:
            sub_masks = [resolve_full_mask(r, channel_fn, n_full, cache) for r in refs]
            if len(sub_masks) == 1:
                result = sub_masks[0]
            elif definition.get("op") == "union":
                result = sub_masks[0].copy()
                for m in sub_masks[1:]:
                    result |= m
            else:
                result = sub_masks[0].copy()
                for m in sub_masks[1:]:
                    result &= m
    else:  # "full", or anything unrecognized
        result = np.ones(n_full, dtype=bool)

    cache[ref] = result
    return result

GATING_WIDGET_CSS = """<style>
.gate-flow {
    display: flex; align-items: center; justify-content: center;
    gap: 0.6rem; margin: 0.25rem 0;
}
.gate-flow-arrow { font-size: 2.5rem; font-weight: 700; line-height: 1; color: #4d4d4d; }
.gate-flow-text { color: #4d4d4d; font-size: 0.875rem; }
button[data-variant="segmented_control"][aria-checked="true"] {
    color: #1a1a1a !important;
    border-color: #1a1a1a !important;
    background-color: rgba(0,0,0,0.06) !important;
}
</style>"""


def _points_in_polygon(x: np.ndarray, y: np.ndarray, poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray-casting point-in-polygon test."""
    inside = np.zeros(len(x), dtype=bool)
    j = len(poly_x) - 1
    for i in range(len(poly_x)):
        xi, yi, xj, yj = poly_x[i], poly_y[i], poly_x[j], poly_y[j]
        crosses = (yi > y) != (yj > y)
        x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def _mask_from_selection(selection, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Compute a keep-mask straight from the selection's drawn coordinates
    against the real per-event x/y values. This works identically no matter
    what trace type is on screen (Scattergl, Contour, ...) — Plotly's own
    `point_indices` come back empty for anything but a plain marker trace,
    which is why box/lasso select looked broken on Contour and Zebra."""
    if selection.box:
        b = selection.box[0]
        x0, x1 = sorted(b["x"])
        y0, y1 = sorted(b["y"])
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if selection.lasso:
        lasso = selection.lasso[0]
        return _points_in_polygon(x, y, np.asarray(lasso["x"]), np.asarray(lasso["y"]))
    if selection.points:
        # Point-click selection: Plotly hands back each clicked point's own
        # data coordinates (not an index into `x`/`y`), so match on the
        # coordinate pair itself — safe here since these values are read
        # straight out of the same array, never computed from it.
        mask = np.zeros(len(x), dtype=bool)
        for p in selection.points:
            mask |= (x == p["x"]) & (y == p["y"])
        return mask
    return None


def _geometry_from_selection(selection) -> dict | None:
    """The selection's raw drawn coordinates, kept so the gate's outline can
    be redrawn on any figure — Plotly itself only shows its native selection
    overlay on the exact trace it was drawn on, so switching view mode
    (a new trace underneath) would otherwise make the gate look like it
    vanished even though the underlying mask is untouched."""
    if selection.box:
        b = selection.box[0]
        return {"type": "box", "x": sorted(b["x"]), "y": sorted(b["y"])}
    if selection.lasso:
        lasso = selection.lasso[0]
        return {"type": "lasso", "x": list(lasso["x"]), "y": list(lasso["y"])}
    if selection.points:
        return {"type": "points", "x": [p["x"] for p in selection.points], "y": [p["y"] for p in selection.points]}
    return None


def _to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _draw_gate_shape(fig: go.Figure, shape: dict, color: str, label: str) -> None:
    """Draw one gate's boundary directly on the shared plot, in its own
    color, with its name floating right at the shape — a white halo behind
    the line keeps it legible no matter what's underneath (a busy
    pseudocolor/contour surface would otherwise swallow a plain thin line)."""
    fill = _to_rgba(color, 0.15)
    if shape["type"] == "box":
        x0, x1 = shape["x"]
        y0, y1 = shape["y"]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer="above", line=dict(color="white", width=6))
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer="above", line=dict(color=color, width=3, dash="dash"), fillcolor=fill)
        label_x, label_y = x1, y1
    elif shape["type"] == "cross":
        cx, cy = shape["x"], shape["y"]
        quadrant = shape.get("quadrant", "UR")
        if quadrant == "UR":
            # All 4 quadrants of one crosshair share the same pair of lines —
            # only the first one draws them, so 4 differently-colored gates
            # don't paint 4 overlapping lines on top of each other.
            fig.add_vline(x=cx, line=dict(color="white", width=5))
            fig.add_vline(x=cx, line=dict(color="#333333", width=2, dash="dash"))
            fig.add_hline(y=cy, line=dict(color="white", width=5))
            fig.add_hline(y=cy, line=dict(color="#333333", width=2, dash="dash"))
        xr, yr = fig.layout.xaxis.range, fig.layout.yaxis.range
        dx = (xr[1] - xr[0]) * 0.06 if xr else 0.0
        dy = (yr[1] - yr[0]) * 0.06 if yr else 0.0
        label_x = cx + (dx if "R" in quadrant else -dx)
        label_y = cy + (dy if quadrant.startswith("U") else -dy)
    elif shape["type"] == "points":
        xs, ys = shape["x"], shape["y"]
        ring = dict(size=15, color="rgba(0,0,0,0)", symbol="circle")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker={**ring, "line": dict(color="white", width=4)}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker={**ring, "line": dict(color=color, width=2.5)}, showlegend=False, hoverinfo="skip"))
        top_i = int(np.argmax(ys))
        label_x, label_y = xs[top_i], ys[top_i]
    else:
        xs, ys = shape["x"], shape["y"]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="white", width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            line=dict(color=color, width=3, dash="dash"), fillcolor=fill, showlegend=False, hoverinfo="skip",
        ))
        top_i = int(np.argmax(ys))
        label_x, label_y = xs[top_i], ys[top_i]
    fig.add_annotation(
        x=label_x, y=label_y, text=f"<b>{label}</b>", showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor=color, borderwidth=1, borderpad=3,
    )


def clear_section(section: str, gate_index: int | None = None) -> None:
    """Clear one gate (or, if `gate_index` is None, every gate) in a
    section, bumping each cleared gate's nonce so its shape disappears from
    the shared plot immediately rather than lingering from stale state.
    Does not rerun — call `st.rerun()` once after clearing everything you
    need to, so clearing multiple sections doesn't cut itself short."""
    n_gates = st.session_state.get(f"n_gates_{section}", 1)
    indices = range(n_gates) if gate_index is None else [gate_index]
    for i in indices:
        prefix = f"gate_{section}_{i}"
        st.session_state[f"{prefix}_nonce"] = st.session_state.get(f"{prefix}_nonce", 0) + 1
        for suffix in ("_mask", "_shape", "_name"):
            st.session_state.pop(f"{prefix}{suffix}", None)


def reset_section(section: str, gate_index: int | None = None) -> None:
    clear_section(section, gate_index)
    st.rerun()


def _combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    if len(masks) == 1:
        return masks[0]
    combined = masks[0].copy()
    for m in masks[1:]:
        combined &= m
    return combined


def _union_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Events captured by ANY of a section's own gates — used for that
    section's own progress summary and gated-histogram preview. A section's
    gates are alternatives read off the same plot (most starkly a Quadrant
    set, which is 4 gates by construction disjoint), not a chain to
    intersect — ANDing them (as `_combine_masks` does for the cross-section
    picker, where that's the right operation) would silently read as 0
    events any time a section holds more than one non-overlapping gate."""
    if len(masks) == 1:
        return masks[0]
    combined = masks[0].copy()
    for m in masks[1:]:
        combined |= m
    return combined


def render_section_gates(
    section: str, incoming_mask: np.ndarray, x_label: str, y_label: str,
    x_values: np.ndarray, y_values: np.ndarray, color_offset: int,
) -> tuple[list[str], list[str], list[np.ndarray], list[tuple]]:
    """Every gate in a section lives on ONE shared plot: a picker chooses
    which gate a new box/lasso selection edits, and every gate already
    drawn stays visible as its own colored, named outline on that same
    figure — adding a gate never opens a second plot. Returns the section's
    (names, colors, masks, refs) — refs are `("gate", section, index)`
    pointers `resolve_full_mask` uses to replay a gate's exact boundary
    against the full dataset later, on the Tabulation page.
    """
    count_key = f"n_gates_{section}"
    n_gates = st.session_state.get(count_key, 1)
    # Which two channels this section's plot is drawn on — needed later to
    # recompute a gate's exact mask against the full dataset, since the
    # shape alone (x/y coordinates) is meaningless without knowing which
    # channels those coordinates are in.
    st.session_state[f"section_{section}_x_label"] = x_label
    st.session_state[f"section_{section}_y_label"] = y_label

    visible_idx = np.flatnonzero(incoming_mask)
    no_events = len(visible_idx) == 0
    if no_events:
        # Still fall through to render the plot (empty) rather than bailing
        # out entirely — a bare warning with no chart at all left no way to
        # see why, switch the picker's combine mode, or just wait out a
        # temporary 0-event state while a gate elsewhere gets adjusted.
        st.warning("No events remain in the incoming population for this gate. The plot below will remain empty until an earlier gate is loosened or the combination method above is changed.")

    names = [st.session_state.get(f"gate_{section}_{i}_name", f"Gate {i + 1}") for i in range(n_gates)]
    colors = [dark_set3_color(color_offset + i) for i in range(n_gates)]

    # A widget's own session_state key can't be reassigned after it's been
    # instantiated this run — so "start editing the gate I just finished
    # drawing" is staged here, before the selectbox exists, and consumed
    # from a pending flag set (further down) on the previous run. Falling
    # back to the persisted mirror (below) covers the other way this key
    # goes missing: Streamlit drops it outright after a visit to another
    # page, with no pending flag involved.
    active_key = f"active_gate_{section}"
    pending_key = f"pending_active_{section}"
    if pending_key in st.session_state:
        st.session_state[active_key] = st.session_state.pop(pending_key)
    elif active_key not in st.session_state:
        restored = _persisted_default(active_key, None)
        if restored == "new" or (isinstance(restored, int) and restored < n_gates):
            st.session_state[active_key] = restored

    ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 0.9, 1.2])
    with ctrl_col0:
        # `segmented_control` deselects (returns None) if you click the
        # already-active option again — fall back to Freeform rather than
        # crashing on the dict lookups below, same defensive shape as the
        # View control's `.get(view, "dot")` just underneath.
        gate_shape = st.segmented_control(
            "Gate shape", ["Freeform", "Point", "Quadrant"],
            default=_persisted_default(f"gate_shape_{section}", "Freeform"), key=f"gate_shape_{section}",
        ) or "Freeform"
        _persist_value(f"gate_shape_{section}", gate_shape)
    with ctrl_col1:
        view = st.segmented_control(
            "View", ["Dot plot", "Pseudocolor", "Contour", "Zebra"],
            default=_persisted_default(f"view_mode_{section}", "Dot plot"), key=f"view_mode_{section}",
        )
        if view is not None:
            _persist_value(f"view_mode_{section}", view)
    mode = {"Pseudocolor": "pseudocolor", "Contour": "contour", "Zebra": "zebra"}.get(view, "dot")

    with ctrl_col2:
        options = list(range(n_gates)) + ["new"]
        active = st.selectbox(
            "Editing", options, index=0,
            format_func=lambda v: "+ New gate" if v == "new" else names[v],
            key=active_key,
        )
        _persist_value(active_key, active)
    active_idx = n_gates if active == "new" else active

    with ctrl_col3:
        if active == "new" and gate_shape == "Quadrant":
            default_name = "Quadrant"
        else:
            default_name = names[active_idx] if active_idx < n_gates else f"Gate {n_gates + 1}"
        typed_name = st.text_input("Name this gate", value=default_name, key=f"gate_{section}_{active_idx}_nameinput")

    shape_hint = {
        "Freeform": "Drag to draw a rectangular gate, or select the lasso tool from the plot toolbar to draw a freeform boundary.",
        "Point": "Click individual points to select them. Hold Shift while clicking to add further points; a plain click replaces the current selection.",
        "Quadrant": "Click once to place a crosshair, creating four gates simultaneously (upper-right, upper-left, lower-left, lower-right), named using the field above.",
    }[gate_shape]
    zoom_col, hint_col = st.columns([1, 2.5])
    with zoom_col:
        zoom_options = list(AXIS_ZOOM_OPTIONS.keys())
        zoom_choice = st.selectbox(
            "Axis zoom", zoom_options,
            index=zoom_options.index(_persisted_default(f"axis_zoom_{section}", zoom_options[0])),
            key=f"axis_zoom_{section}",
            help="Zooming or scrolling directly on the plot does not persist, as Plotly resets the view on every gate edit. Use this control instead to view outlier events beyond the default range; the selection is preserved.",
        )
        _persist_value(f"axis_zoom_{section}", zoom_choice)
    with hint_col:
        st.caption(shape_hint)
    pct_range = AXIS_ZOOM_OPTIONS[zoom_choice]

    x_visible, y_visible = x_values[visible_idx], y_values[visible_idx]
    base_color = dark_set3_color(color_offset)
    if no_events:
        # Percentile/density math (and Contour/Zebra's grid) can't run on 0
        # points — build the axes from the full, always-non-empty per-event
        # arrays instead, in plain dot mode, then drop the actual markers so
        # the plot still reads as "correctly empty," not broken.
        fig = build_scatter_figure(x_values, y_values, x_label, y_label, f"{x_label} vs {y_label}", pct_range=pct_range, color=base_color, mode="dot")
        fig.data[0].x, fig.data[0].y = [], []
    else:
        fig = build_scatter_figure(x_visible, y_visible, x_label, y_label, f"{x_label} vs {y_label}", pct_range=pct_range, color=base_color, mode=mode)
    active_color = dark_set3_color(color_offset + active_idx)
    # Box/lasso need Plotly's drag-rect layer ("select"); point and quadrant
    # clicks need it OUT OF THE WAY — a plain click under dragmode="select"
    # is consumed as a zero-area box gesture and never reaches `points` at
    # all, confirmed by direct probing of the selection event payload.
    plot_dragmode = "select" if gate_shape == "Freeform" else False
    fig.update_layout(
        dragmode=plot_dragmode,
        clickmode="event+select",
        newselection=dict(line=dict(color=active_color, width=2, dash="dash")),
        activeselection=dict(fillcolor=active_color, opacity=0.15),
    )

    if not no_events and mode in ("contour", "zebra") and gate_shape in ("Point", "Quadrant"):
        # Contour/Zebra are density surfaces with no individual markers to
        # click — an invisible-ish points layer gives click gating something
        # to hit, so it isn't silently unavailable on these two views.
        fig.add_trace(go.Scattergl(
            x=x_visible, y=y_visible, mode="markers", marker=dict(size=5, color="rgba(20,20,20,0.25)"), showlegend=False,
        ))

    for i in range(n_gates):
        if i == active_idx:
            continue  # the gate being actively edited draws from its live selection below, not a stale shape
        shape = st.session_state.get(f"gate_{section}_{i}_shape")
        if shape is not None:
            _draw_gate_shape(fig, shape, colors[i], names[i])

    saved_shape = st.session_state.get(f"gate_{section}_{active_idx}_shape")
    if saved_shape is not None:
        _draw_gate_shape(fig, saved_shape, active_color, typed_name)

    nonce = st.session_state.get(f"gate_{section}_{active_idx}_nonce", 0)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", key=f"plot_{section}_g{active_idx}_n{nonce}", config={"displayModeBar": True})

    if gate_shape == "Quadrant":
        if event and event.selection and event.selection.points:
            cx, cy = float(event.selection.points[0]["x"]), float(event.selection.points[0]["y"])
            # Re-clicking one sibling of an already-placed crosshair should
            # move that same set of 4, not spawn a new set — the anchor
            # (the set's lowest gate index) is carried in every sibling's
            # own shape so this resolves correctly no matter which of the
            # 4 gates happens to be selected in "Editing" right now.
            current_shape = st.session_state.get(f"gate_{section}_{active_idx}_shape")
            if active != "new" and isinstance(current_shape, dict) and current_shape.get("type") == "cross":
                anchor = current_shape.get("anchor", active_idx)
            else:
                anchor = n_gates
            quadrant_specs = [
                ("UR", (x_visible >= cx) & (y_visible >= cy)),
                ("UL", (x_visible < cx) & (y_visible >= cy)),
                ("LL", (x_visible < cx) & (y_visible < cy)),
                ("LR", (x_visible >= cx) & (y_visible < cy)),
            ]
            for j, (qlabel, qmask_visible) in enumerate(quadrant_specs):
                gate_i = anchor + j
                gated_mask = np.zeros(len(incoming_mask), dtype=bool)
                gated_mask[visible_idx[qmask_visible]] = True
                st.session_state[f"gate_{section}_{gate_i}_mask"] = gated_mask
                st.session_state[f"gate_{section}_{gate_i}_shape"] = {"type": "cross", "x": cx, "y": cy, "quadrant": qlabel, "anchor": anchor}
                st.session_state[f"gate_{section}_{gate_i}_name"] = f"{typed_name} {qlabel}"
            st.session_state[count_key] = max(n_gates, anchor + 4)
            st.session_state[pending_key] = anchor
            st.rerun()
    else:
        selection_mask = _mask_from_selection(event.selection, x_visible, y_visible) if event and event.selection else None
        if selection_mask is not None:
            gated_mask = np.zeros(len(incoming_mask), dtype=bool)
            gated_mask[visible_idx[selection_mask]] = True
            st.session_state[f"gate_{section}_{active_idx}_mask"] = gated_mask
            st.session_state[f"gate_{section}_{active_idx}_shape"] = _geometry_from_selection(event.selection)
            st.session_state[f"gate_{section}_{active_idx}_name"] = typed_name
            if active_idx == n_gates:
                st.session_state[count_key] = n_gates + 1
                st.session_state[pending_key] = active_idx  # keep editing the gate just drawn, not jump to another new one
                st.rerun()
        elif active_idx < n_gates:
            st.session_state[f"gate_{section}_{active_idx}_name"] = typed_name

    n_gates = st.session_state.get(count_key, n_gates)
    names = [st.session_state.get(f"gate_{section}_{i}_name", f"Gate {i + 1}") for i in range(n_gates)]
    colors = [dark_set3_color(color_offset + i) for i in range(n_gates)]
    masks = [st.session_state.get(f"gate_{section}_{i}_mask", incoming_mask.copy()) for i in range(n_gates)]

    reset_col1, reset_col2 = st.columns(2)
    with reset_col1:
        if st.button(f"Reset {names[active_idx] if active_idx < len(names) else 'gate'}", key=f"reset_one_{section}"):
            reset_section(section, active_idx if active_idx < n_gates else None)
    with reset_col2:
        if n_gates > 1 and st.button("Reset all gates in this section", key=f"reset_all_{section}"):
            reset_section(section)

    refs = [("gate", section, i) for i in range(n_gates)]
    return names, colors, masks, refs


def render_flow(kept: int, incoming: int) -> None:
    pct = (kept / incoming * 100) if incoming else 0.0
    st.html(
        '<div class="gate-flow">'
        '<span class="gate-flow-arrow">↓</span>'
        f'<span class="gate-flow-text">{kept:,} of {incoming:,} events ({pct:.1f}%) proceed to the next gate.</span>'
        "</div>"
    )


def extend_registry(
    registry: list[tuple[str, str, str, np.ndarray, tuple]], section_title: str,
    gates: tuple[list[str], list[str], list[np.ndarray]], refs: list[tuple],
) -> None:
    """Add one section's gates to the running registry every later section
    draws its population picker from. `refs` (one per gate, same order —
    typically `render_section_gates`'s own 4th return value) is what lets
    `resolve_full_mask` replay this gate's exact boundary against the full
    dataset later, regardless of which page is on screen at the time."""
    labels, colors, masks = gates
    registry.extend(zip([section_title] * len(labels), labels, colors, masks, refs))


def render_population_picker(registry: list[tuple[str, str, str, np.ndarray, tuple]], default_section_title: str, key: str, n_events: int) -> np.ndarray:
    """Every gate drawn anywhere so far — not just the immediately prior
    section's — is available here, and any combination of them can be
    ANDed or ORed together to build the next step's incoming population.
    This is what makes a gate reusable at any later point, not just a
    one-shot hand-off to the very next plot. Defaults to the immediately
    prior section's FIRST gate, matching the natural sequential flow, but
    nothing stops you from reaching further back, adding more, or skipping
    gating entirely (clear the picker to fall back to the full population).
    `n_events` is the length of the page's plot subsample, for the
    empty-registry/empty-selection fallback (all events).

    Also persists this picker's own choice — as refs, not just subsample
    masks — under `{key}__def`, so `resolve_full_mask(("picker", key), ...)`
    can replay the exact same combination against the full dataset later."""
    if not registry:
        st.session_state[f"{key}__def"] = {"refs": [], "op": "intersect"}
        return np.ones(n_events, dtype=bool)

    display = [f"{sec} — {label}" for sec, label, _color, _mask, _ref in registry]
    # Only the first gate, not every gate from that section — a section can
    # hold several mutually-exclusive gates (a Quadrant split, most starkly),
    # and defaulting to all of them would AND disjoint gates together into a
    # silent, confusing 0.
    default = [d for d, (sec, *_rest) in zip(display, registry) if sec == default_section_title][:1]
    # A prior, explicit choice (including a deliberately-cleared picker)
    # beats that computed default — but only once it's actually been made;
    # restoring it is also what undoes Streamlit dropping this widget's own
    # state after a visit to another page and back.
    persisted = _persisted_default(key, None)
    if persisted is not None:
        default = [d for d in persisted if d in display]

    # Each gate's own event count is shown up front — combining two gates
    # that individually look fine can still legitimately AND down to zero
    # if they don't overlap, and this is what makes that visible instead
    # of a confusing "0 events" with no way to tell which gate caused it.
    swatches = "".join(
        f'<span style="display:inline-block;width:0.7em;height:0.7em;border-radius:50%;'
        f'background:{color};margin-right:0.3em;"></span>{d} <span style="color:#888;">({int(mask.sum()):,})</span>&nbsp;&nbsp;'
        for d, (_sec, _label, color, mask, _ref) in zip(display, registry)
    )
    st.markdown(swatches, unsafe_allow_html=True)
    chosen = st.multiselect(
        "Select the incoming population for this step:", display, default=default, key=key,
        help="All gates defined so far are listed here, each with its corresponding event count. Any combination may be selected, including gates from earlier steps. Clear the selection to use the full population.",
    )
    _persist_value(key, chosen)

    if not chosen:
        st.session_state[f"{key}__def"] = {"refs": [], "op": "intersect"}
        return np.ones(n_events, dtype=bool)
    chosen_masks = [mask for d, (_sec, _label, _color, mask, _ref) in zip(display, registry) if d in chosen]
    chosen_refs = [ref for d, (_sec, _label, _color, _mask, ref) in zip(display, registry) if d in chosen]
    if len(chosen_masks) == 1:
        st.session_state[f"{key}__def"] = {"refs": chosen_refs, "op": "intersect"}
        return chosen_masks[0]

    combine_key = f"{key}_combine_op"
    combine_options = ["Match ALL selected (intersection)", "Match ANY selected (union)"]
    combine_op = st.radio(
        "Combine the selected gates as", combine_options, horizontal=True, key=combine_key,
        index=combine_options.index(_persisted_default(combine_key, combine_options[0])),
        help="ALL (intersection): an event must satisfy every selected gate simultaneously, producing a narrower subset. ANY (union): an event need only satisfy one of the selected gates — appropriate when combining alternative populations, such as the four gates produced by a Quadrant split, which would otherwise always intersect to zero.",
    )
    _persist_value(combine_key, combine_op)
    op = "union" if combine_op == "Match ANY selected (union)" else "intersect"
    st.session_state[f"{key}__def"] = {"refs": chosen_refs, "op": op}
    if combine_op == "Match ANY selected (union)":
        return _union_masks(chosen_masks)
    return _combine_masks(chosen_masks)
