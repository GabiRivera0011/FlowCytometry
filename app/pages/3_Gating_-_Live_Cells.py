import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from _gating_widgets import (
    GATING_WIDGET_CSS,
    _persist_value,
    _persisted_default,
    _union_masks,
    clear_section,
    extend_registry,
    render_flow,
    render_population_picker,
    render_section_gates,
)
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.transforms import arcsinh_transform
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_gated_histogram_figure

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
st.caption("Fluorescence channels are displayed on an arcsinh transform, which spreads out the dim and negative events a linear scale would compress near zero. Forward/side scatter and Time remain linear.")

REQUIRED_CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "Time"]
channel_index = {name: i for i, name in enumerate(sample.pnn_labels)}
missing = [c for c in REQUIRED_CHANNELS if c not in channel_index]
if missing:
    st.warning(f"This file is missing the following standard channel(s): {', '.join(missing)}. Certain gating plots below will be omitted as a result.")

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
    # Fluorescence channels are shown on an arcsinh display transform, not
    # the raw linear scale — a linear axis compresses the dim/negative
    # population into a sliver near zero. FSC/SSC/Time (not in
    # `fluor_channels`) stay linear, which is standard for scatter channels.
    if channel in fluor_channels:
        raw = compensated_fluor[:, fluor_channels.index(channel)] if compensated_fluor is not None else plot_values[:, channel_index[channel]]
        return arcsinh_transform(raw)
    return plot_values[:, channel_index[channel]]


metadata = build_channel_metadata(sample)
viability_matches = metadata.loc[metadata["target"].fillna("").str.lower() == "live dead", "channel"].tolist()
default_viability = viability_matches[0] if viability_matches else sample.pnn_labels[0]

keep_mask = np.ones(len(plot_idx), dtype=bool)
acq_gates: tuple[list[str], list[str], list[np.ndarray]] = (["Gate 1"], [dark_set3_color(0)], [keep_mask])
acq_refs: list[tuple] = [("full",)]
gate_registry: list[tuple[str, str, str, np.ndarray, tuple]] = []

# --- Acquisition ---
with st.container(key="gate_card_acquisition", border=True):
    st.header("Acquisition")
    st.caption("Signal intensity plotted against acquisition time, used to detect flow-rate drift or instrument instability during data collection.")
    if "Time" in channel_index:
        scatter_channels = [c for c in ["FSC-A", "FSC-H", "SSC-A"] if c in channel_index]
        default_acq_channel = "FSC-A" if "FSC-A" in scatter_channels else scatter_channels[0]
        restored_acq_channel = _persisted_default("acq_channel", default_acq_channel)
        restored_acq_channel = restored_acq_channel if restored_acq_channel in scatter_channels else default_acq_channel
        acq_channel = st.selectbox("Channel vs. Time", scatter_channels, index=scatter_channels.index(restored_acq_channel), key="acq_channel")
        _persist_value("acq_channel", acq_channel)
        incoming = int(keep_mask.sum())
        acq_names, acq_colors, acq_masks, acq_refs = render_section_gates(
            "acquisition", keep_mask, "Time", acq_channel, channel_values("Time"), channel_values(acq_channel),
            color_offset=0, subsample_full_indices=plot_idx,
        )
        acq_gates = (acq_names, acq_colors, acq_masks)
    else:
        st.caption("No Time channel is present in this file; this step has been omitted.")
        incoming = int(keep_mask.sum())
        acq_gates = (["Gate 1"], [dark_set3_color(0)], [keep_mask])
        acq_refs = [("full",)]
render_flow(int(_union_masks(acq_gates[2]).sum()), incoming)
extend_registry(gate_registry, "Acquisition", acq_gates, acq_refs)

# --- Cellular debris ---
with st.container(key="gate_card_debris", border=True):
    st.header("Cellular debris")
    st.caption("Forward scatter plotted against side scatter. Debris and dead-cell fragments cluster at low FSC/SSC values, while the main cell population is distinctly separated.")
    keep_mask = render_population_picker(gate_registry, "Acquisition", key="picker_debris", n_events=len(plot_idx))
    st.session_state["incoming_ref_debris"] = ("picker", "picker_debris")
    if "FSC-A" in channel_index and "SSC-A" in channel_index:
        incoming = int(keep_mask.sum())
        pre_gate_idx = np.flatnonzero(keep_mask)
        debris_names, debris_colors, debris_masks, debris_refs = render_section_gates(
            "debris", keep_mask, "FSC-A", "SSC-A", channel_values("FSC-A"), channel_values("SSC-A"),
            color_offset=1, subsample_full_indices=plot_idx,
        )
        debris_gates = (debris_names, debris_colors, debris_masks)
        if len(pre_gate_idx) > 0:
            gated_idx = np.flatnonzero(_union_masks(debris_gates[2]))
            st.caption("FSC-A distribution for the full incoming population, with the gate(s) defined above highlighted.")
            st.plotly_chart(
                build_gated_histogram_figure(
                    channel_values("FSC-A")[pre_gate_idx], channel_values("FSC-A")[gated_idx],
                    "FSC-A", dark_set3_color(1),
                ),
                width="stretch",
            )
    else:
        st.caption("No FSC-A/SSC-A channels are present in this file; this step has been omitted.")
        incoming = int(keep_mask.sum())
        debris_gates = (["Gate 1"], [dark_set3_color(1)], [keep_mask])
        debris_refs = [("picker", "picker_debris")]
render_flow(int(_union_masks(debris_gates[2]).sum()), incoming)
extend_registry(gate_registry, "Cellular debris", debris_gates, debris_refs)

# --- Single cells ---
with st.container(key="gate_card_singlets", border=True):
    st.header("Single cells")
    st.caption("FSC-A plotted against FSC-H. Singlets fall along the diagonal; doublets and clumps exhibit higher area relative to height.")
    keep_mask = render_population_picker(gate_registry, "Cellular debris", key="picker_singlets", n_events=len(plot_idx))
    st.session_state["incoming_ref_singlets"] = ("picker", "picker_singlets")
    if "FSC-A" in channel_index and "FSC-H" in channel_index:
        incoming = int(keep_mask.sum())
        singlets_names, singlets_colors, singlets_masks, singlets_refs = render_section_gates(
            "singlets", keep_mask, "FSC-A", "FSC-H", channel_values("FSC-A"), channel_values("FSC-H"),
            color_offset=2, subsample_full_indices=plot_idx,
        )
        singlets_gates = (singlets_names, singlets_colors, singlets_masks)
    else:
        st.caption("No FSC-A/FSC-H channels are present in this file; this step has been omitted.")
        incoming = int(keep_mask.sum())
        singlets_gates = (["Gate 1"], [dark_set3_color(2)], [keep_mask])
        singlets_refs = [("picker", "picker_singlets")]
render_flow(int(_union_masks(singlets_gates[2]).sum()), incoming)
extend_registry(gate_registry, "Single cells", singlets_gates, singlets_refs)

# --- Live cells ---
with st.container(key="gate_card_live", border=True):
    st.header("Live cells")
    st.caption("Viability stain plotted against side scatter. Live cells exclude the dye and read low; dead cells take up the dye and read high.")
    keep_mask = render_population_picker(gate_registry, "Single cells", key="picker_live", n_events=len(plot_idx))
    st.session_state["incoming_ref_live"] = ("picker", "picker_live")
    if "SSC-A" in channel_index:
        viability_channel = default_viability
        if workspace.spillover_matrix is None and viability_channel in fluor_channels:
            st.caption("The raw, uncompensated viability channel is shown. Visit the Compensation page first for a cleaner live/dead separation.")
        incoming = int(keep_mask.sum())
        pre_gate_idx = np.flatnonzero(keep_mask)
        live_names, live_colors, live_masks, live_refs = render_section_gates(
            "live", keep_mask, "SSC-A", viability_channel, channel_values("SSC-A"), channel_values(viability_channel),
            color_offset=3, subsample_full_indices=plot_idx,
        )
        live_gates = (live_names, live_colors, live_masks)
        if len(pre_gate_idx) > 0:
            gated_idx = np.flatnonzero(_union_masks(live_gates[2]))
            all_viability = channel_values(viability_channel)[pre_gate_idx]
            gated_viability = channel_values(viability_channel)[gated_idx]
            st.caption(f"{viability_channel} distribution (arcsinh scale) for the full incoming population, with the gate(s) defined above highlighted.")
            st.plotly_chart(
                build_gated_histogram_figure(all_viability, gated_viability, viability_channel, dark_set3_color(3)),
                width="stretch",
            )
    else:
        st.caption("No SSC-A channel is present in this file; this step has been omitted.")
        incoming = int(keep_mask.sum())
        live_gates = (["Gate 1"], [dark_set3_color(3)], [keep_mask])
        live_refs = [("picker", "picker_live")]

extend_registry(gate_registry, "Live cells", live_gates, live_refs)

st.divider()
st.subheader("Final population")
final_mask = render_population_picker(gate_registry, "Live cells", key="picker_final", n_events=len(plot_idx))
workspace.live_cells_mask = final_mask
workspace.live_cells_mask_ref = ("picker", "picker_final")
st.success(f"**{int(final_mask.sum()):,}** of {len(plot_idx):,} subsampled events satisfy the selected gate(s) ({final_mask.sum() / len(plot_idx) * 100:.1f}%).")
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
