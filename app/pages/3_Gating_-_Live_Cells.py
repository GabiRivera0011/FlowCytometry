import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_gated_histogram_figure, build_gated_log_histogram_figure, build_scatter_figure

st.set_page_config(page_title="Gating - Live Cells — FlowCytometry", layout="wide")
st.title("Gating — Live Cells")

SECTION_ORDER = ["acquisition", "debris", "singlets", "live"]


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


def build_gate_order() -> list[str]:
    """The full sequence of gate keys currently on the page, in section then
    gate-index order — computed fresh each run since gate counts can grow."""
    order = []
    for section in SECTION_ORDER:
        for i in range(st.session_state.get(f"n_gates_{section}", 1)):
            order.append(f"gate_{section}_{i}")
    return order


st.html(
    """<style>
    .st-key-gate_card_acquisition, .st-key-gate_card_debris,
    .st-key-gate_card_singlets, .st-key-gate_card_live {
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        padding: 1.5rem 1.75rem;
    }
    .st-key-gate_card_acquisition { border: 2px solid """ + dark_set3_color(0) + """ !important; }
    .st-key-gate_card_debris      { border: 2px solid """ + dark_set3_color(1) + """ !important; }
    .st-key-gate_card_singlets    { border: 2px solid """ + dark_set3_color(2) + """ !important; }
    .st-key-gate_card_live        { border: 2px solid """ + dark_set3_color(3) + """ !important; }
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
)

workspace = require_sample()
sample = workspace.sample

st.caption(f"Current sample: **{sample.filename}** — {sample.event_count:,} events, {len(sample.pnn_labels)} channels.")

REQUIRED_CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "Time"]
channel_index = {name: i for i, name in enumerate(sample.pnn_labels)}
missing = [c for c in REQUIRED_CHANNELS if c not in channel_index]
if missing:
    st.warning(f"This file is missing standard channel(s) {', '.join(missing)} — some gating plots below may be skipped.")

if workspace.plot_subsample_indices is None:
    rng = np.random.default_rng(0)
    workspace.plot_subsample_indices = rng.choice(
        len(sample.raw_values), size=min(20_000, len(sample.raw_values)), replace=False
    )
plot_idx = workspace.plot_subsample_indices
plot_values = sample.raw_values[plot_idx]

# Fluorescence channels get compensated (if a matrix has been set on the
# Compensation page) before use here — scatter/time channels never carry
# spillover, so they always stay raw.
fluor_channels = sample.fluorescence_channels
fluor_indices = [channel_index[c] for c in fluor_channels]
if workspace.spillover_matrix is not None:
    compensated_fluor = plot_values[:, fluor_indices] @ np.linalg.inv(workspace.spillover_matrix.values)
else:
    compensated_fluor = None


def channel_values(channel: str) -> np.ndarray:
    if compensated_fluor is not None and channel in fluor_channels:
        return compensated_fluor[:, fluor_channels.index(channel)]
    return plot_values[:, channel_index[channel]]


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
    if len(visible_idx) == 0:
        st.warning("No events remain going into this gate — loosen an earlier gate above.")
        empty = np.zeros(len(incoming_mask), dtype=bool)
        names = [st.session_state.get(f"gate_{section}_{i}_name", f"Gate {i + 1}") for i in range(n_gates)]
        colors = [dark_set3_color(color_offset + i) for i in range(n_gates)]
        return names, colors, [empty] * n_gates

    names = [st.session_state.get(f"gate_{section}_{i}_name", f"Gate {i + 1}") for i in range(n_gates)]
    colors = [dark_set3_color(color_offset + i) for i in range(n_gates)]

    # A widget's own session_state key can't be reassigned after it's been
    # instantiated this run — so "start editing the gate I just finished
    # drawing" is staged here, before the selectbox exists, and consumed
    # from a pending flag set (further down) on the previous run.
    pending_key = f"pending_active_{section}"
    if pending_key in st.session_state:
        st.session_state[f"active_gate_{section}"] = st.session_state.pop(pending_key)

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.3, 1, 1.3])
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
        default_name = names[active_idx] if active_idx < n_gates else f"Gate {n_gates + 1}"
        typed_name = st.text_input("Name this gate", value=default_name, key=f"gate_{section}_{active_idx}_nameinput")

    x_visible, y_visible = x_values[visible_idx], y_values[visible_idx]
    base_color = dark_set3_color(color_offset)
    fig = build_scatter_figure(x_visible, y_visible, x_label, y_label, f"{x_label} vs {y_label}", pct_range=GATE_PCT, color=base_color, mode=mode)
    active_color = dark_set3_color(color_offset + active_idx)
    fig.update_layout(
        dragmode="select",
        newselection=dict(line=dict(color=active_color, width=2, dash="dash")),
        activeselection=dict(fillcolor=active_color, opacity=0.15),
    )

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


def render_gate_picker(section_title: str, labels: list[str], colors: list[str], masks: list[np.ndarray], key: str) -> np.ndarray:
    """Shown at the top of the NEXT section, not the bottom of the one the
    gates came from — this is where the choice actually takes effect. A
    section with only one gate needs no picker at all."""
    if len(labels) <= 1:
        return masks[0]

    swatches = "".join(
        f'<span style="display:inline-block;width:0.7em;height:0.7em;border-radius:50%;'
        f'background:{color};margin-right:0.3em;"></span>{label}&nbsp;&nbsp;'
        for label, color in zip(labels, colors)
    )
    st.markdown(f"Gates from **{section_title}**: {swatches}", unsafe_allow_html=True)
    options = ["All gates (combined)"] + labels
    choice = st.selectbox(f"Use which gate from {section_title} for this step?", options, key=key)

    if choice == "All gates (combined)":
        return _combine_masks(masks)
    return masks[labels.index(choice)]


metadata = build_channel_metadata(sample)
viability_matches = metadata.loc[metadata["target"].fillna("").str.lower() == "live dead", "channel"].tolist()
default_viability = viability_matches[0] if viability_matches else sample.pnn_labels[0]

GATE_PCT = (5, 95)  # autoscale each axis to its central 90% of the data
keep_mask = np.ones(len(plot_idx), dtype=bool)
acq_gates: tuple[list[str], list[str], list[np.ndarray]] = (["Gate 1"], [dark_set3_color(0)], [keep_mask])

# --- Acquisition ---
with st.container(key="gate_card_acquisition", border=True):
    st.header("Acquisition")
    st.caption("Signal vs. acquisition time. Use this to spot flow-rate drift or instrument instability during the run.")
    if "Time" in channel_index:
        scatter_channels = [c for c in ["FSC-A", "FSC-H", "SSC-A"] if c in channel_index]
        acq_channel = st.selectbox("Channel vs. Time", scatter_channels, index=scatter_channels.index("FSC-A") if "FSC-A" in scatter_channels else 0)
        incoming = int(keep_mask.sum())
        acq_gates = render_section_gates(
            "acquisition", keep_mask, "Time", acq_channel, channel_values("Time"), channel_values(acq_channel), color_offset=0,
        )
    else:
        st.caption("No Time channel in this file — skipping.")
        incoming = int(keep_mask.sum())
        acq_gates = (["Gate 1"], [dark_set3_color(0)], [keep_mask])
render_flow(int(_combine_masks(acq_gates[2]).sum()), incoming)

# --- Cellular debris ---
with st.container(key="gate_card_debris", border=True):
    st.header("Cellular debris")
    st.caption("Forward vs. side scatter. Debris and dead-cell fragments cluster at low FSC/SSC; the main cell population sits apart.")
    keep_mask = render_gate_picker("Acquisition", *acq_gates, key="picker_debris")
    if "FSC-A" in channel_index and "SSC-A" in channel_index:
        incoming = int(keep_mask.sum())
        pre_gate_idx = np.flatnonzero(keep_mask)
        debris_gates = render_section_gates(
            "debris", keep_mask, "FSC-A", "SSC-A", channel_values("FSC-A"), channel_values("SSC-A"), color_offset=1,
        )
        if len(pre_gate_idx) > 0:
            gated_idx = np.flatnonzero(_combine_masks(debris_gates[2]))
            st.caption("FSC-A distribution: the full incoming population, with the gate(s) drawn above highlighted.")
            st.plotly_chart(
                build_gated_histogram_figure(
                    channel_values("FSC-A")[pre_gate_idx], channel_values("FSC-A")[gated_idx],
                    "FSC-A", dark_set3_color(1),
                ),
                width="stretch",
            )
    else:
        st.caption("No FSC-A/SSC-A channels in this file — skipping.")
        incoming = int(keep_mask.sum())
        debris_gates = (["Gate 1"], [dark_set3_color(1)], [keep_mask])
render_flow(int(_combine_masks(debris_gates[2]).sum()), incoming)

# --- Single cells ---
with st.container(key="gate_card_singlets", border=True):
    st.header("Single cells")
    st.caption("FSC-A vs. FSC-H. Singlets fall along the diagonal; doublets and clumps read higher area relative to height.")
    keep_mask = render_gate_picker("Cellular debris", *debris_gates, key="picker_singlets")
    if "FSC-A" in channel_index and "FSC-H" in channel_index:
        incoming = int(keep_mask.sum())
        singlets_gates = render_section_gates(
            "singlets", keep_mask, "FSC-A", "FSC-H", channel_values("FSC-A"), channel_values("FSC-H"), color_offset=2,
        )
    else:
        st.caption("No FSC-A/FSC-H channels in this file — skipping.")
        incoming = int(keep_mask.sum())
        singlets_gates = (["Gate 1"], [dark_set3_color(2)], [keep_mask])
render_flow(int(_combine_masks(singlets_gates[2]).sum()), incoming)

# --- Live cells ---
with st.container(key="gate_card_live", border=True):
    st.header("Live cells")
    st.caption("Viability stain vs. side scatter. Live cells exclude the dye and read low; dead cells take it up and read high.")
    keep_mask = render_gate_picker("Single cells", *singlets_gates, key="picker_live")
    if "SSC-A" in channel_index:
        viability_channel = default_viability
        if workspace.spillover_matrix is None and viability_channel in fluor_channels:
            st.caption("Showing the raw viability channel — visit Compensation first for a cleaner live/dead separation.")
        incoming = int(keep_mask.sum())
        pre_gate_idx = np.flatnonzero(keep_mask)
        live_gates = render_section_gates(
            "live", keep_mask, "SSC-A", viability_channel, channel_values("SSC-A"), channel_values(viability_channel), color_offset=3,
        )
        if len(pre_gate_idx) > 0:
            gated_idx = np.flatnonzero(_combine_masks(live_gates[2]))
            all_viability = channel_values(viability_channel)[pre_gate_idx]
            gated_viability = channel_values(viability_channel)[gated_idx]
            st.caption(f"{viability_channel} distribution, linear and log: the full incoming population, with the gate(s) drawn above highlighted.")
            hist_col1, hist_col2 = st.columns(2)
            with hist_col1:
                st.plotly_chart(
                    build_gated_histogram_figure(all_viability, gated_viability, viability_channel, dark_set3_color(3)),
                    width="stretch",
                )
            with hist_col2:
                st.plotly_chart(
                    build_gated_log_histogram_figure(all_viability, gated_viability, viability_channel, dark_set3_color(3)),
                    width="stretch",
                )
    else:
        st.caption("No SSC-A channel in this file — skipping.")
        incoming = int(keep_mask.sum())
        live_gates = (["Gate 1"], [dark_set3_color(3)], [keep_mask])

st.divider()
final_mask = render_gate_picker("Live cells", *live_gates, key="picker_final")
st.success(f"**{int(final_mask.sum()):,}** of {len(plot_idx):,} subsampled events survive all four gates ({final_mask.sum() / len(plot_idx) * 100:.1f}%).")
if st.button("Reset all gates"):
    for section in SECTION_ORDER:
        clear_section(section)
    st.rerun()

st.divider()
st.html(
    """<style>
    .st-key-next_button button[kind="primary"] {
        background-color: #8fd9b6; color: #014421; border-color: #6cc79b;
    }
    .st-key-next_button button[kind="primary"]:hover,
    .st-key-next_button button[kind="primary"]:active,
    .st-key-next_button button[kind="primary"]:focus:not(:active) {
        background-color: #7bcea3; color: #014421; border-color: #5cb98a;
    }
    </style>"""
)
back_col, next_col = st.columns(2)
with back_col:
    if st.button("← Back: Compensation", width="stretch"):
        st.switch_page("pages/2_Compensation.py")
with next_col:
    if st.button("Next: Broad Cellular Lineage →", width="stretch", type="primary", key="next_button"):
        st.switch_page("pages/4_Gating_-_Broad_Cellular_Lineage.py")
