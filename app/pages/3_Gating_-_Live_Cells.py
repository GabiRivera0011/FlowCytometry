import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_histogram_figure, build_log_histogram_figure, build_scatter_figure

st.set_page_config(page_title="Gating - Live Cells — FlowCytometry", layout="wide")
st.title("Gating — Live Cells")

GATE_ORDER = ["gate_acquisition", "gate_debris", "gate_singlets", "gate_live"]
SELECT_LINE_COLOR = "#e34948"  # red dashes: the drawn gate should pop against any marker color

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


def reset_from(prefix: str) -> None:
    """Clear this gate's selection and every gate after it — their incoming
    population is about to change, so any selection they hold is stale. Also
    bumps each gate's render nonce so its chart remounts fresh client-side;
    otherwise Plotly keeps showing the old selection rectangle even though
    the underlying mask has already reset."""
    clear_prefixes = GATE_ORDER[GATE_ORDER.index(prefix):]
    for p in clear_prefixes:
        st.session_state[f"nonce_{p}"] = st.session_state.get(f"nonce_{p}", 0) + 1
    for k in list(st.session_state):
        if any(k.startswith(p) for p in clear_prefixes) and not k.startswith("nonce_"):
            del st.session_state[k]
    st.rerun()


def render_gate(prefix: str, key: str, x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str, color: str, keep_mask: np.ndarray) -> np.ndarray:
    """Render one interactive gate. A box/lasso selection on the plot
    replaces the incoming population with just the enclosed events; with no
    selection yet, everything entering this gate passes through untouched.
    Returns the outgoing mask that feeds the next gate down the page."""
    visible_idx = np.flatnonzero(keep_mask)
    if len(visible_idx) == 0:
        st.warning("No events remain going into this gate — loosen an earlier gate above.")
        return keep_mask

    view = st.segmented_control(
        "View", ["Dot plot", "Pseudocolor", "Contour", "Zebra"], default="Dot plot", key=f"view_mode_{prefix}",
    )
    mode = {"Pseudocolor": "pseudocolor", "Contour": "contour", "Zebra": "zebra"}.get(view, "dot")
    interactive = mode in ("dot", "pseudocolor")
    if not interactive:
        st.caption("Density-only view — switch to Dot plot or Pseudocolor to draw or adjust a gate.")

    fig = build_scatter_figure(x[visible_idx], y[visible_idx], x_label, y_label, title, pct_range=GATE_PCT, color=color, mode=mode)
    fig.update_layout(
        dragmode="select" if interactive else "pan",
        newselection=dict(line=dict(color=SELECT_LINE_COLOR, width=2, dash="dash")),
        activeselection=dict(fillcolor=SELECT_LINE_COLOR, opacity=0.1),
    )
    nonce = st.session_state.get(f"nonce_{prefix}", 0)
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", key=f"{key}_n{nonce}", config={"displayModeBar": True})

    # Plotly drops its selection state on a trace-type change (Scattergl ->
    # Contour and back), so switching to a density-only view and back would
    # otherwise silently clear the gate. Persist the gate ourselves, keyed by
    # this gate's prefix, and only touch it when a live selection actually
    # arrives — reusing it whenever the widget itself reports nothing new.
    saved_key = f"{prefix}_saved_mask"
    if event and event.selection and event.selection.point_indices:
        gated_mask = np.zeros(len(keep_mask), dtype=bool)
        gated_mask[visible_idx[event.selection.point_indices]] = True
        st.session_state[saved_key] = gated_mask

    outgoing_mask = keep_mask.copy()
    saved_mask = st.session_state.get(saved_key)
    if saved_mask is not None:
        outgoing_mask &= saved_mask

    if st.button("Reset", key=f"reset_btn_{prefix}"):
        reset_from(prefix)

    return outgoing_mask


def render_flow(kept: int, incoming: int) -> None:
    st.html(
        '<div class="gate-flow">'
        '<span class="gate-flow-arrow">↓</span>'
        f'<span class="gate-flow-text">{kept:,} / {incoming:,} events ({kept / incoming * 100:.1f}%) continue to the next gate.</span>'
        "</div>"
    )


metadata = build_channel_metadata(sample)
viability_matches = metadata.loc[metadata["target"].fillna("").str.lower() == "live dead", "channel"].tolist()
default_viability = viability_matches[0] if viability_matches else sample.pnn_labels[0]

GATE_PCT = (5, 95)  # autoscale each axis to its central 90% of the data
keep_mask = np.ones(len(plot_idx), dtype=bool)

# --- Acquisition ---
with st.container(key="gate_card_acquisition", border=True):
    st.header("Acquisition")
    st.caption("Signal vs. acquisition time. Use this to spot flow-rate drift or instrument instability during the run.")
    if "Time" in channel_index:
        scatter_channels = [c for c in ["FSC-A", "FSC-H", "SSC-A"] if c in channel_index]
        acq_channel = st.selectbox("Channel vs. Time", scatter_channels, index=scatter_channels.index("FSC-A") if "FSC-A" in scatter_channels else 0)
        incoming = int(keep_mask.sum())
        keep_mask = render_gate(
            "gate_acquisition", f"gate_acquisition_{acq_channel}", channel_values("Time"), channel_values(acq_channel),
            "Time", acq_channel, f"Time vs {acq_channel}", dark_set3_color(0), keep_mask,
        )
    else:
        st.caption("No Time channel in this file — skipping.")
        incoming = int(keep_mask.sum())
render_flow(int(keep_mask.sum()), incoming)

# --- Cellular debris ---
with st.container(key="gate_card_debris", border=True):
    st.header("Cellular debris")
    st.caption("Forward vs. side scatter. Debris and dead-cell fragments cluster at low FSC/SSC; the main cell population sits apart.")
    if "FSC-A" in channel_index and "SSC-A" in channel_index:
        incoming = int(keep_mask.sum())
        keep_mask = render_gate(
            "gate_debris", "gate_debris", channel_values("FSC-A"), channel_values("SSC-A"),
            "FSC-A", "SSC-A", "FSC-A vs SSC-A", dark_set3_color(1), keep_mask,
        )
        gated_idx = np.flatnonzero(keep_mask)
        st.caption("FSC-A distribution of the gated population above — moves with the gate you draw.")
        st.plotly_chart(
            build_histogram_figure(channel_values("FSC-A")[gated_idx], "FSC-A", color=dark_set3_color(1)),
            width="stretch",
        )
    else:
        st.caption("No FSC-A/SSC-A channels in this file — skipping.")
        incoming = int(keep_mask.sum())
render_flow(int(keep_mask.sum()), incoming)

# --- Single cells ---
with st.container(key="gate_card_singlets", border=True):
    st.header("Single cells")
    st.caption("FSC-A vs. FSC-H. Singlets fall along the diagonal; doublets and clumps read higher area relative to height.")
    if "FSC-A" in channel_index and "FSC-H" in channel_index:
        incoming = int(keep_mask.sum())
        keep_mask = render_gate(
            "gate_singlets", "gate_singlets", channel_values("FSC-A"), channel_values("FSC-H"),
            "FSC-A", "FSC-H", "FSC-A vs FSC-H", dark_set3_color(2), keep_mask,
        )
    else:
        st.caption("No FSC-A/FSC-H channels in this file — skipping.")
        incoming = int(keep_mask.sum())
render_flow(int(keep_mask.sum()), incoming)

# --- Live cells ---
with st.container(key="gate_card_live", border=True):
    st.header("Live cells")
    st.caption("Viability stain vs. side scatter. Live cells exclude the dye and read low; dead cells take it up and read high.")
    if "SSC-A" in channel_index:
        viability_channel = st.selectbox(
            "Viability channel", sample.pnn_labels, index=channel_index.get(default_viability, 0)
        )
        if workspace.spillover_matrix is None and viability_channel in fluor_channels:
            st.caption("Showing the raw viability channel — visit Compensation first for a cleaner live/dead separation.")
        incoming = int(keep_mask.sum())
        keep_mask = render_gate(
            "gate_live", f"gate_live_{viability_channel}", channel_values("SSC-A"), channel_values(viability_channel),
            "SSC-A", viability_channel, f"SSC-A vs {viability_channel}", dark_set3_color(3), keep_mask,
        )
        gated_idx = np.flatnonzero(keep_mask)
        gated_viability = channel_values(viability_channel)[gated_idx]
        st.caption(f"{viability_channel} distribution of the gated population above, linear and log — moves with the gate you draw.")
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            st.plotly_chart(
                build_histogram_figure(gated_viability, viability_channel, color=dark_set3_color(3)),
                width="stretch",
            )
        with hist_col2:
            st.plotly_chart(
                build_log_histogram_figure(gated_viability, viability_channel, color=dark_set3_color(3)),
                width="stretch",
            )
    else:
        st.caption("No SSC-A channel in this file — skipping.")
        incoming = int(keep_mask.sum())

st.success(f"**{int(keep_mask.sum()):,}** of {len(plot_idx):,} subsampled events survive all four gates ({keep_mask.sum() / len(plot_idx) * 100:.1f}%).")
if st.button("Reset all gates"):
    reset_from(GATE_ORDER[0])

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
