import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from _gating_widgets import GATING_WIDGET_CSS, _persist_value, _persisted_default, render_section_gates, transform_channel
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.lineage import build_live_split_diagram

st.set_page_config(page_title="Gating - Broad Cellular Lineage — FlowCytometry", layout="wide")
st.title("Gating — Broad Cellular Lineage")

st.html(
    """<style>
    .st-key-gate_card_lymphocytes, .st-key-gate_card_myeloid {
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        padding: 1.5rem 1.75rem;
    }
    .st-key-gate_card_lymphocytes { border: 2px solid """ + dark_set3_color(0) + """ !important; }
    .st-key-gate_card_myeloid     { border: 2px solid """ + dark_set3_color(2) + """ !important; }
    </style>"""
    + GATING_WIDGET_CSS
)

workspace = require_sample()
sample = workspace.sample

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
    # Fluorescence channels are shown on a Logicle display transform, not
    # the raw linear scale — see Gating — Live Cells for the full rationale.
    if channel in fluor_channels:
        raw = compensated_fluor[:, fluor_channels.index(channel)] if compensated_fluor is not None else plot_values[:, channel_index[channel]]
        return transform_channel(workspace, channel, raw, fluor_channels)
    return plot_values[:, channel_index[channel]]


# The starting population is whatever Live Cells (the previous page) landed
# on — not re-derived here, so this page always reflects that page's gates,
# including any later adjustment there.
if workspace.live_cells_mask is not None and len(workspace.live_cells_mask) == len(plot_idx):
    live_mask = workspace.live_cells_mask
    live_mask_ref = workspace.live_cells_mask_ref or ("full",)
    incoming = int(live_mask.sum())
    st.caption(f"Starting population: **{incoming:,}** Live cells, carried over from the preceding page.")
else:
    live_mask = np.ones(len(plot_idx), dtype=bool)
    live_mask_ref = ("full",)
    incoming = int(live_mask.sum())
    st.info("No Live cells gate has been defined yet; the full subsampled population is shown instead. Visit **Gating — Live Cells** first for a cleaner starting point.")
st.caption("Fluorescence channels are displayed on a Logicle transform; forward/side scatter remain linear.")
st.session_state["incoming_ref_lymphocytes"] = live_mask_ref
st.session_state["incoming_ref_myeloid"] = live_mask_ref

metadata = build_channel_metadata(sample)
target_by_channel = dict(zip(metadata["channel"], metadata["target"]))


def channel_label(channel: str) -> str:
    target = target_by_channel.get(channel)
    return f"{channel} — {target}" if isinstance(target, str) and target else channel


# Pre-seed each plot's single gate with the name this page exists to make,
# so both open ready to draw rather than with a generic "Gate 1".
if "n_gates_lymphocytes" not in st.session_state:
    st.session_state["n_gates_lymphocytes"] = 1
    st.session_state["gate_lymphocytes_0_name"] = "Lymphocytes"
if "n_gates_myeloid" not in st.session_state:
    st.session_state["n_gates_myeloid"] = 1
    st.session_state["gate_myeloid_0_name"] = "Myeloid cells"

st.header("Lymphocytes vs. myeloid cells")

channel_options = list(sample.pnn_labels)
default_x = "FSC-A" if "FSC-A" in channel_options else channel_options[0]
default_y = "SSC-A" if "SSC-A" in channel_options else channel_options[min(1, len(channel_options) - 1)]


def render_axis_pickers(key_prefix: str):
    # Streamlit drops a selectbox's own stored value once it stops being
    # rendered for even one run — i.e. the moment the user visits another
    # page — silently resetting it to `index=`'s default. The persisted
    # mirror is what makes a chosen marker pair stick across that.
    x_key, y_key = f"{key_prefix}_x", f"{key_prefix}_y"
    restored_x = _persisted_default(x_key, default_x)
    restored_y = _persisted_default(y_key, default_y)
    restored_x = restored_x if restored_x in channel_options else default_x
    restored_y = restored_y if restored_y in channel_options else default_y
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X axis marker", channel_options, index=channel_options.index(restored_x), format_func=channel_label, key=x_key)
        _persist_value(x_key, x)
    with col2:
        y = st.selectbox("Y axis marker", channel_options, index=channel_options.index(restored_y), format_func=channel_label, key=y_key)
        _persist_value(y_key, y)
    if workspace.spillover_matrix is None and (x in fluor_channels or y in fluor_channels):
        st.caption("Raw, uncompensated marker channels are shown. Visit the Compensation page first for a cleaner separation.")
    return x, y


with st.container(key="gate_card_lymphocytes", border=True):
    st.subheader("Lymphocytes")
    lymph_x, lymph_y = render_axis_pickers("lymph")
    lymph_names, lymph_colors, lymph_masks, lymph_refs = render_section_gates(
        "lymphocytes", live_mask, lymph_x, lymph_y, channel_values(lymph_x), channel_values(lymph_y),
        color_offset=0, subsample_full_indices=plot_idx,
    )

with st.container(key="gate_card_myeloid", border=True):
    st.subheader("Myeloid cells")
    myeloid_x, myeloid_y = render_axis_pickers("myeloid")
    myeloid_names, myeloid_colors, myeloid_masks, myeloid_refs = render_section_gates(
        "myeloid", live_mask, myeloid_x, myeloid_y, channel_values(myeloid_x), channel_values(myeloid_y),
        color_offset=2, subsample_full_indices=plot_idx,
    )

lymph_label = lymph_names[0] if lymph_names else "Lymphocytes"
myeloid_label = myeloid_names[0] if myeloid_names else "Myeloid cells"
lymph_mask = lymph_masks[0] if lymph_masks else np.zeros(len(plot_idx), dtype=bool)
myeloid_mask = myeloid_masks[0] if myeloid_masks else np.zeros(len(plot_idx), dtype=bool)
lymph_ref = lymph_refs[0] if lymph_refs else ("full",)
myeloid_ref = myeloid_refs[0] if myeloid_refs else ("full",)

st.plotly_chart(
    build_live_split_diagram(incoming, int(lymph_mask.sum()), int(myeloid_mask.sum()), lymph_label, myeloid_label),
    width="stretch", config={"displayModeBar": False},
)
extra_lymph, extra_myeloid = len(lymph_names) - 1, len(myeloid_names) - 1
if extra_lymph > 0 or extra_myeloid > 0:
    bits = [f"{n} extra {label} gate(s)" for n, label in [(extra_lymph, "Lymphocytes"), (extra_myeloid, "Myeloid")] if n > 0]
    st.caption(f"{' and '.join(bits)} defined above are not shown in the split diagram, but remain available to later pages.")

workspace.lymphocytes_mask = lymph_mask
workspace.myeloid_mask = myeloid_mask
workspace.lymphocytes_mask_ref = lymph_ref
workspace.myeloid_mask_ref = myeloid_ref

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
