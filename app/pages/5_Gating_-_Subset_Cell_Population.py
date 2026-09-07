import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_scatter_figure

st.set_page_config(page_title="Gating - Subset Cell Population — FlowCytometry", layout="wide")
st.title("Gating — Subset Cell Population")

workspace = require_sample()
sample = workspace.sample

st.caption(f"Current sample: **{sample.filename}** — {sample.event_count:,} events, {len(sample.pnn_labels)} channels.")

channel_index = {name: i for i, name in enumerate(sample.pnn_labels)}

if workspace.plot_subsample_indices is None:
    rng = np.random.default_rng(0)
    workspace.plot_subsample_indices = rng.choice(
        len(sample.raw_values), size=min(20_000, len(sample.raw_values)), replace=False
    )
plot_idx = workspace.plot_subsample_indices
plot_values = sample.raw_values[plot_idx]

# Fluorescence channels get compensated (if a matrix has been set on the
# Compensation page) before use here — scatter channels never carry
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
target_to_channel = dict(zip(metadata["target"], metadata["channel"]))

if workspace.spillover_matrix is None:
    st.caption("Showing raw marker channels — visit Compensation first for a cleaner separation.")

GATE_PCT = (5, 95)  # autoscale each axis to its central 90% of the data

# --- Monocytes vs dendritic cells ---
st.header("Monocytes vs. dendritic cells")
st.caption("Within the myeloid/APC gate: monocytes read CD14+; dendritic cells read CD14- and HLA-DR+.")
mono_channel, apc_channel = target_to_channel.get("CD14"), target_to_channel.get("HLA-DR")
if mono_channel and apc_channel:
    fig = build_scatter_figure(
        channel_values(mono_channel), channel_values(apc_channel),
        mono_channel, apc_channel, f"{mono_channel} vs {apc_channel}", pct_range=GATE_PCT, color=dark_set3_color(0),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("No CD14/HLA-DR channels found in this file — skipping.")

st.divider()

# --- Dendritic cell subsets ---
st.header("Dendritic cell subsets")
st.caption(
    "cDC1 read CD141+/XCR1+/CADM1+; cDC2 read CD1c+/CD172a+; plasmacytoid DCs "
    "read CD303+/FcER1a+. Pick any two markers below to explore the split."
)
dc_subset_targets = ["CD141", "XCR1", "CADM1", "CD1c", "CD172a", "CD303", "FcER1a"]
dc_candidates = [target_to_channel[t] for t in dc_subset_targets if t in target_to_channel]
if len(dc_candidates) >= 2:
    default_x = target_to_channel.get("CD1c", dc_candidates[0])
    default_y = target_to_channel.get("CD141", dc_candidates[1])
    dc_col1, dc_col2 = st.columns(2)
    with dc_col1:
        x_channel = st.selectbox("Marker X", dc_candidates, index=dc_candidates.index(default_x))
    with dc_col2:
        y_channel = st.selectbox("Marker Y", dc_candidates, index=dc_candidates.index(default_y))
    fig = build_scatter_figure(
        channel_values(x_channel), channel_values(y_channel),
        x_channel, y_channel, f"{x_channel} vs {y_channel}", pct_range=GATE_PCT, color=dark_set3_color(1),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("Not enough DC-subset markers (CD141/XCR1/CADM1/CD1c/CD172a/CD303/FcER1a) found in this file — skipping.")

st.divider()
if st.button("← Back: Broad Cellular Lineage", width="stretch"):
    st.switch_page("pages/4_Gating_-_Broad_Cellular_Lineage.py")
