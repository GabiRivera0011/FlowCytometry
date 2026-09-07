import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.raw import build_scatter_figure

st.set_page_config(page_title="Gating - Broad Cellular Lineage — FlowCytometry", layout="wide")
st.title("Gating — Broad Cellular Lineage")

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
lineage_candidates = metadata.loc[metadata["target"].isin(["CD3", "CD19", "CD20", "CD56"]), "channel"].tolist()
apc_candidates = metadata.loc[metadata["target"] == "HLA-DR", "channel"].tolist()

GATE_PCT = (5, 95)  # autoscale each axis to its central 90% of the data

# --- Lymphocytes vs myeloid ---
st.header("Lymphocytes vs. myeloid cells")
st.caption("Forward vs. side scatter. Lymphocytes read small and agranular (lower FSC/SSC); myeloid cells read larger and more granular (higher FSC/SSC).")
if "FSC-A" in channel_index and "SSC-A" in channel_index:
    fig = build_scatter_figure(
        channel_values("FSC-A"), channel_values("SSC-A"),
        "FSC-A", "SSC-A", "FSC-A vs SSC-A", pct_range=GATE_PCT, color=dark_set3_color(0),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("No FSC-A/SSC-A channels in this file — skipping.")

st.divider()

# --- Lineage vs HLA-DR ---
st.header("Lineage vs. HLA-DR")
st.caption(
    "Antigen-presenting/myeloid cells read HLA-DR+ and lineage-negative. "
    "T, B, and NK lymphocytes read lineage-positive and HLA-DR-negative."
)
if lineage_candidates and apc_candidates:
    lineage_channel = st.selectbox("Lineage marker (CD3 / CD19 / CD20 / CD56)", lineage_candidates)
    apc_channel = apc_candidates[0]
    if workspace.spillover_matrix is None:
        st.caption("Showing raw marker channels — visit Compensation first for a cleaner separation.")
    fig = build_scatter_figure(
        channel_values(lineage_channel), channel_values(apc_channel),
        lineage_channel, apc_channel, f"{lineage_channel} vs {apc_channel}", pct_range=GATE_PCT, color=dark_set3_color(1),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.caption("No lineage marker (CD3/CD19/CD20/CD56) or HLA-DR channel found in this file — skipping.")

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
    if st.button("← Back: Gating — Live Cells", width="stretch"):
        st.switch_page("pages/3_Gating_-_Live_Cells.py")
with next_col:
    if st.button("Next: Subset Cell Population →", width="stretch", type="primary", key="next_button"):
        st.switch_page("pages/5_Gating_-_Subset_Cell_Population.py")
