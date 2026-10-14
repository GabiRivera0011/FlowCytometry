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
) -> tuple[list[str], list[str], list[np.ndarray]]:
    """Every gate in a section lives on ONE shared plot: a picker chooses
    which gate a new box/lasso selection edits, and every gate already
    drawn stays visible as its own colored, named outline on that same
    figure — adding a gate never opens a second plot. Returns the section's
    (names, colors, masks), which the next section's picker builds on."""
    count_key = f"n_gates_{section}"
    n_gates = st.session_state.get(count_key, 1)

    visible_idx = np.flatnonzero(incoming_mask)
    no_events = len(visible_idx) == 0
    if no_events:
        # Still fall through to render the plot (empty) rather than bailing
        # out entirely — a bare warning with no chart at all left no way to
        # see why, switch the picker's combine mode, or just wait out a
        # temporary 0-event state while a gate elsewhere gets adjusted.
        st.warning("No events remain going into this gate — the plot below is empty until you loosen an earlier gate or change how gates are combined above.")

    names = [st.session_state.get(f"gate_{section}_{i}_name", f"Gate {i + 1}") for i in range(n_gates)]
    colors = [dark_set3_color(color_offset + i) for i in range(n_gates)]

    # A widget's own session_state key can't be reassigned after it's been
    # instantiated this run — so "start editing the gate I just finished
    # drawing" is staged here, before the selectbox exists, and consumed
    # from a pending flag set (further down) on the previous run.
    pending_key = f"pending_active_{section}"
    if pending_key in st.session_state:
        st.session_state[f"active_gate_{section}"] = st.session_state.pop(pending_key)

    ctrl_col0, ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 0.9, 1.2])
    with ctrl_col0:
        # `segmented_control` deselects (returns None) if you click the
        # already-active option again — fall back to Freeform rather than
        # crashing on the dict lookups below, same defensive shape as the
        # View control's `.get(view, "dot")` just underneath.
        gate_shape = st.segmented_control(
            "Gate shape", ["Freeform", "Point", "Quadrant"], default="Freeform", key=f"gate_shape_{section}",
        ) or "Freeform"
    with ctrl_col1:
        view = st.segmented_control(
            "View", ["Dot plot", "Pseudocolor", "Contour", "Zebra"], default="Dot plot", key=f"view_mode_{section}",
        )
    mode = {"Pseudocolor": "pseudocolor", "Contour": "contour", "Zebra": "zebra"}.get(view, "dot")

    with ctrl_col2:
        options = list(range(n_gates)) + ["new"]
        active = st.selectbox(
            "Editing", options, index=0,
            format_func=lambda v: "+ New gate" if v == "new" else names[v],
            key=f"active_gate_{section}",
        )
    active_idx = n_gates if active == "new" else active

    with ctrl_col3:
        if active == "new" and gate_shape == "Quadrant":
            default_name = "Quadrant"
        else:
            default_name = names[active_idx] if active_idx < n_gates else f"Gate {n_gates + 1}"
        typed_name = st.text_input("Name this gate", value=default_name, key=f"gate_{section}_{active_idx}_nameinput")

    shape_hint = {
        "Freeform": "Drag a box, or switch to the lasso tool in the plot's toolbar, to draw this gate.",
        "Point": "Click individual points to select them — shift-click to add more, a plain click replaces the selection.",
        "Quadrant": "Click once to drop a crosshair — creates 4 gates at once (upper-right, upper-left, lower-left, lower-right), named from the field above.",
    }[gate_shape]
    zoom_col, hint_col = st.columns([1, 2.5])
    with zoom_col:
        zoom_choice = st.selectbox(
            "Axis zoom", list(AXIS_ZOOM_OPTIONS.keys()), key=f"axis_zoom_{section}",
            help="Dragging/scrolling to zoom on the plot itself doesn't stick — Plotly resets it on every gate edit. Use this instead to see outlier cells beyond the default view; it persists.",
        )
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

    return names, colors, masks


def render_flow(kept: int, incoming: int) -> None:
    pct = (kept / incoming * 100) if incoming else 0.0
    st.html(
        '<div class="gate-flow">'
        '<span class="gate-flow-arrow">↓</span>'
        f'<span class="gate-flow-text">{kept:,} / {incoming:,} events ({pct:.1f}%) continue to the next gate.</span>'
        "</div>"
    )


def extend_registry(registry: list[tuple[str, str, str, np.ndarray]], section_title: str, gates: tuple[list[str], list[str], list[np.ndarray]]) -> None:
    """Add one section's gates to the running registry every later section
    draws its population picker from."""
    labels, colors, masks = gates
    registry.extend(zip([section_title] * len(labels), labels, colors, masks))


def render_population_picker(registry: list[tuple[str, str, str, np.ndarray]], default_section_title: str, key: str, n_events: int) -> np.ndarray:
    """Every gate drawn anywhere so far — not just the immediately prior
    section's — is available here, and any combination of them can be
    ANDed or ORed together to build the next step's incoming population.
    This is what makes a gate reusable at any later point, not just a
    one-shot hand-off to the very next plot. Defaults to the immediately
    prior section's FIRST gate, matching the natural sequential flow, but
    nothing stops you from reaching further back, adding more, or skipping
    gating entirely (clear the picker to fall back to the full population).
    `n_events` is the length of the page's plot subsample, for the
    empty-registry/empty-selection fallback (all events)."""
    if not registry:
        return np.ones(n_events, dtype=bool)

    display = [f"{sec} — {label}" for sec, label, _color, _mask in registry]
    # Only the first gate, not every gate from that section — a section can
    # hold several mutually-exclusive gates (a Quadrant split, most starkly),
    # and defaulting to all of them would AND disjoint gates together into a
    # silent, confusing 0.
    default = [d for d, (sec, *_rest) in zip(display, registry) if sec == default_section_title][:1]

    # Each gate's own event count is shown up front — combining two gates
    # that individually look fine can still legitimately AND down to zero
    # if they don't overlap, and this is what makes that visible instead
    # of a confusing "0 events" with no way to tell which gate caused it.
    swatches = "".join(
        f'<span style="display:inline-block;width:0.7em;height:0.7em;border-radius:50%;'
        f'background:{color};margin-right:0.3em;"></span>{d} <span style="color:#888;">({int(mask.sum()):,})</span>&nbsp;&nbsp;'
        for d, (_sec, _label, color, mask) in zip(display, registry)
    )
    st.markdown(swatches, unsafe_allow_html=True)
    chosen = st.multiselect(
        "Build this step's incoming population from:", display, default=default, key=key,
        help="Every gate drawn so far is available here, with its own event count — combine any of them, from any earlier step, or clear this to skip gating.",
    )

    if not chosen:
        return np.ones(n_events, dtype=bool)
    chosen_masks = [mask for d, (_sec, _label, _color, mask) in zip(display, registry) if d in chosen]
    if len(chosen_masks) == 1:
        return chosen_masks[0]

    combine_op = st.radio(
        "Combine the selected gates as", ["Match ALL selected (intersection)", "Match ANY selected (union)"], horizontal=True, key=f"{key}_combine_op",
        help="ALL: an event must satisfy every selected gate at once — narrowing to a subset. ANY: an event only needs to satisfy one of them — two alternative populations (like a Quadrant's 4 gates) would otherwise always AND down to zero.",
    )
    if combine_op == "Match ANY selected (union)":
        return _union_masks(chosen_masks)
    return _combine_masks(chosen_masks)
