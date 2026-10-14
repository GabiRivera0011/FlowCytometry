import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from _gating_widgets import (
    GATING_WIDGET_CSS,
    _union_masks,
    clear_section,
    extend_registry,
    render_flow,
    render_population_picker,
    render_section_gates,
)
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_gated_histogram_figure, build_gated_log_histogram_figure

st.set_page_config(page_title="Gating - Live Cells — FlowCytometry", layout="wide")
st.title("Gating — Live Cells")

SECTION_ORDER = ["acquisition", "debris", "singlets", "live"]

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
    </style>"""
    + GATING_WIDGET_CSS
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


metadata = build_channel_metadata(sample)
viability_matches = metadata.loc[metadata["target"].fillna("").str.lower() == "live dead", "channel"].tolist()
default_viability = viability_matches[0] if viability_matches else sample.pnn_labels[0]

keep_mask = np.ones(len(plot_idx), dtype=bool)
acq_gates: tuple[list[str], list[str], list[np.ndarray]] = (["Gate 1"], [dark_set3_color(0)], [keep_mask])
gate_registry: list[tuple[str, str, str, np.ndarray]] = []

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
render_flow(int(_union_masks(acq_gates[2]).sum()), incoming)
extend_registry(gate_registry, "Acquisition", acq_gates)

# --- Cellular debris ---
with st.container(key="gate_card_debris", border=True):
    st.header("Cellular debris")
    st.caption("Forward vs. side scatter. Debris and dead-cell fragments cluster at low FSC/SSC; the main cell population sits apart.")
    keep_mask = render_population_picker(gate_registry, "Acquisition", key="picker_debris", n_events=len(plot_idx))
    if "FSC-A" in channel_index and "SSC-A" in channel_index:
        incoming = int(keep_mask.sum())
        pre_gate_idx = np.flatnonzero(keep_mask)
        debris_gates = render_section_gates(
            "debris", keep_mask, "FSC-A", "SSC-A", channel_values("FSC-A"), channel_values("SSC-A"), color_offset=1,
        )
        if len(pre_gate_idx) > 0:
            gated_idx = np.flatnonzero(_union_masks(debris_gates[2]))
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
render_flow(int(_union_masks(debris_gates[2]).sum()), incoming)
extend_registry(gate_registry, "Cellular debris", debris_gates)

# --- Single cells ---
with st.container(key="gate_card_singlets", border=True):
    st.header("Single cells")
    st.caption("FSC-A vs. FSC-H. Singlets fall along the diagonal; doublets and clumps read higher area relative to height.")
    keep_mask = render_population_picker(gate_registry, "Cellular debris", key="picker_singlets", n_events=len(plot_idx))
    if "FSC-A" in channel_index and "FSC-H" in channel_index:
        incoming = int(keep_mask.sum())
        singlets_gates = render_section_gates(
            "singlets", keep_mask, "FSC-A", "FSC-H", channel_values("FSC-A"), channel_values("FSC-H"), color_offset=2,
        )
    else:
        st.caption("No FSC-A/FSC-H channels in this file — skipping.")
        incoming = int(keep_mask.sum())
        singlets_gates = (["Gate 1"], [dark_set3_color(2)], [keep_mask])
render_flow(int(_union_masks(singlets_gates[2]).sum()), incoming)
extend_registry(gate_registry, "Single cells", singlets_gates)

# --- Live cells ---
with st.container(key="gate_card_live", border=True):
    st.header("Live cells")
    st.caption("Viability stain vs. side scatter. Live cells exclude the dye and read low; dead cells take it up and read high.")
    keep_mask = render_population_picker(gate_registry, "Single cells", key="picker_live", n_events=len(plot_idx))
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
            gated_idx = np.flatnonzero(_union_masks(live_gates[2]))
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

extend_registry(gate_registry, "Live cells", live_gates)

st.divider()
st.subheader("Final population")
final_mask = render_population_picker(gate_registry, "Live cells", key="picker_final", n_events=len(plot_idx))
workspace.live_cells_mask = final_mask
st.success(f"**{int(final_mask.sum()):,}** of {len(plot_idx):,} subsampled events survive the selected gate(s) ({final_mask.sum() / len(plot_idx) * 100:.1f}%).")
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
