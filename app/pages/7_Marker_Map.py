import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from _gating_widgets import _persist_value, _persisted_default, collect_section_gates
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.transforms import arcsinh_transform
from flowcyto.viz.bars import dark_set3_color
from flowcyto.viz.scatter3d import build_marker_scatter3d_figure

st.set_page_config(page_title="Marker Map — FlowCytometry", layout="wide")

workspace = require_sample()
sample = workspace.sample

title_help = (
    "Raw, uncompensated marker channels are shown. Visit the Compensation page first for a cleaner map."
    if workspace.spillover_matrix is None else None
)
st.title("Marker Map", help=title_help)

st.caption(
    "A three-dimensional scatter plot of every subsampled event, with X, Y, and Z assigned directly to markers "
    "of choice, consistent with the two-dimensional gating plots. Points may be colored by the expression of a "
    "fourth marker or by membership in one or more previously defined gates."
)

if workspace.plot_subsample_indices is None:
    rng = np.random.default_rng(0)
    workspace.plot_subsample_indices = rng.choice(
        len(sample.raw_values), size=min(20_000, len(sample.raw_values)), replace=False
    )
plot_idx = workspace.plot_subsample_indices
plot_values = sample.raw_values[plot_idx]
n_events = len(plot_idx)

channel_index = {name: i for i, name in enumerate(sample.pnn_labels)}
fluor_channels = sample.fluorescence_channels
fluor_indices = [channel_index[c] for c in fluor_channels]
if workspace.spillover_matrix is not None:
    compensated_fluor = plot_values[:, fluor_indices] @ np.linalg.inv(workspace.spillover_matrix.values)
else:
    compensated_fluor = None

_empty_channels = set(build_channel_metadata(sample).query("category == 'empty'")["channel"])
marker_channels = [c for c in fluor_channels if c not in _empty_channels]
scatter_channels = [c for c in ["FSC-A", "FSC-H", "SSC-A"] if c in channel_index]
axis_channels = scatter_channels + marker_channels

metadata = build_channel_metadata(sample)
target_by_channel = dict(zip(metadata["channel"], metadata["target"]))


def channel_label(channel: str) -> str:
    target = target_by_channel.get(channel)
    return f"{channel} — {target}" if isinstance(target, str) and target else channel


def channel_values(channel: str) -> np.ndarray:
    # Fluorescence channels get the same arcsinh display transform used
    # throughout the gating pages — a linear scale would compress the
    # dim/negative population near zero. FSC/SSC stay linear, consistent
    # with how the gating pages treat scatter channels.
    if channel in fluor_channels:
        raw = compensated_fluor[:, fluor_channels.index(channel)] if compensated_fluor is not None else plot_values[:, channel_index[channel]]
        return arcsinh_transform(raw)
    return plot_values[:, channel_index[channel]]


if len(axis_channels) < 3:
    st.info("At least 3 channels (scatter or named markers) are needed for a 3D scatter.")
    st.stop()

axis_defaults = [
    _persisted_default("map_x_axis", axis_channels[0]),
    _persisted_default("map_y_axis", axis_channels[min(1, len(axis_channels) - 1)]),
    _persisted_default("map_z_axis", axis_channels[min(2, len(axis_channels) - 1)]),
]
axis_defaults = [c if c in axis_channels else axis_channels[0] for c in axis_defaults]

x_col, y_col, z_col = st.columns(3)
with x_col:
    x_axis = st.selectbox("X axis", axis_channels, index=axis_channels.index(axis_defaults[0]), format_func=channel_label)
    _persist_value("map_x_axis", x_axis)
with y_col:
    y_axis = st.selectbox("Y axis", axis_channels, index=axis_channels.index(axis_defaults[1]), format_func=channel_label)
    _persist_value("map_y_axis", y_axis)
with z_col:
    z_axis = st.selectbox("Z axis", axis_channels, index=axis_channels.index(axis_defaults[2]), format_func=channel_label)
    _persist_value("map_z_axis", z_axis)

if len({x_axis, y_axis, z_axis}) < 3:
    st.warning("Pick three different channels for X, Y, and Z — a repeated axis collapses the scatter flat.")

color_mode = st.segmented_control(
    "Color by", ["Marker", "Gated population(s)"],
    default=_persisted_default("map_color_mode", "Marker"), key="map_color_mode_widget",
    help="Marker: a continuous gradient of a 4th marker's expression. Gated population(s): one or more "
    "already-defined gates, each in its own color, with everything else shown in neutral gray.",
) or "Marker"
_persist_value("map_color_mode", color_mode)

color_values = color_label = None
highlights: list[tuple[np.ndarray, str, str]] = []

if color_mode == "Marker":
    restored_marker = _persisted_default("map_color_channel", axis_channels[0])
    restored_marker = restored_marker if restored_marker in axis_channels else axis_channels[0]
    color_channel = st.selectbox(
        "Color by marker", axis_channels, index=axis_channels.index(restored_marker), format_func=channel_label,
    )
    _persist_value("map_color_channel", color_channel)
    color_values, color_label = channel_values(color_channel), channel_label(color_channel)
else:
    gate_options: dict[str, np.ndarray] = {}
    for section, location_label in [
        ("acquisition", "Acquisition"), ("debris", "Cellular debris"),
        ("singlets", "Single cells"), ("live", "Live cells"),
        ("lymphocytes", "Lymphocytes"), ("myeloid", "Myeloid cells"),
    ]:
        for gate in collect_section_gates(section):
            if len(gate["mask"]) == n_events:
                gate_options[f"{location_label} — {gate['name']}"] = gate["mask"]
    for page_id in st.session_state.get("subset_pages", []):
        page_title = st.session_state.get(f"subset_page_{page_id}_title", "Page")
        for plot_id in st.session_state.get(f"subset_page_{page_id}_plot_ids", []):
            section = f"subset_{page_id}_{plot_id}"
            plot_title = st.session_state.get(f"subset_{section}_title", "Plot")
            for gate in collect_section_gates(section):
                if len(gate["mask"]) == n_events:
                    gate_options[f"{page_title} — {plot_title} — {gate['name']}"] = gate["mask"]

    highlight_choices = st.multiselect("Gated population(s)", list(gate_options.keys()))
    highlights = [(gate_options[label], label, dark_set3_color(i)) for i, label in enumerate(highlight_choices)]

st.plotly_chart(
    build_marker_scatter3d_figure(
        channel_values(x_axis), channel_values(y_axis), channel_values(z_axis),
        channel_label(x_axis), channel_label(y_axis), channel_label(z_axis),
        color_values=color_values, color_label=color_label, highlights=highlights,
    ),
    width="stretch",
)
st.caption(
    f"Scatter computed from **{n_events:,}** subsampled events. Named marker axes are arcsinh-transformed, "
    "same as the gating pages; FSC/SSC axes remain linear."
)

st.divider()
if st.button("← Back: Gating Summary", width="stretch"):
    st.switch_page("pages/6_Gating_Summary.py")
