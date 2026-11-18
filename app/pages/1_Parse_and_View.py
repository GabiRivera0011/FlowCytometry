from pathlib import Path

import streamlit as st

from _common import ensure_src_on_path, get_workspace

ensure_src_on_path()

import numpy as np
import pandas as pd

from flowcyto.io.csv_source import load_csv
from flowcyto.io.fcs import load_fcs
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.io.types import LoadedSample
from flowcyto.viz.raw import build_histogram_figure, build_scatter_figure

st.set_page_config(page_title="Parse and View — FlowCytometry", layout="wide")
st.title("Parse and View Dataset")

workspace = get_workspace()

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PLOT_SUBSAMPLE_SIZE = 20_000


@st.cache_resource(scope="session", max_entries=1, show_spinner="Parsing sample file...")
def _cached_load_sample(content_hash: str, filename: str, _file_or_path) -> LoadedSample:
    if filename.lower().endswith(".csv"):
        return load_csv(_file_or_path, filename)
    return load_fcs(_file_or_path, filename)


@st.cache_resource(scope="session", show_spinner="Parsing control file...")
def _cached_load_control(content_hash: str, filename: str, _file_or_path) -> LoadedSample:
    return load_fcs(_file_or_path, filename)


@st.cache_data(show_spinner=False)
def _cached_metadata(content_hash: str, _sample: LoadedSample) -> pd.DataFrame:
    return build_channel_metadata(_sample)


def _content_hash(file_or_path) -> str:
    import hashlib

    data = file_or_path.read_bytes() if isinstance(file_or_path, Path) else file_or_path.getvalue()
    return hashlib.md5(data).hexdigest()


# --- Upload ---

st.header("Load data")
sample_col, controls_col = st.columns(2)

with sample_col:
    st.subheader("Sample file")
    sample_upload = st.file_uploader("FCS (a full acquisition, e.g. PBMC_*.fcs) or CSV", type=["fcs", "csv"])
    bundled_samples = sorted(DATA_DIR.glob("PBMC_*.fcs")) + sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []
    bundled_sample_choice = None
    if bundled_samples and sample_upload is None:
        bundled_sample_choice = st.selectbox(
            "Or select a bundled sample (development use only)", ["(none)"] + [p.name for p in bundled_samples]
        )
        bundled_sample_choice = None if bundled_sample_choice == "(none)" else bundled_sample_choice

with controls_col:
    st.subheader("Compensation controls (optional)")
    control_uploads = st.file_uploader(
        "Single-stain Comp_*.fcs files", type=["fcs"], accept_multiple_files=True
    )
    bundled_controls = sorted(DATA_DIR.glob("Comp_*.fcs")) if DATA_DIR.is_dir() else []
    use_bundled_controls = False
    if bundled_controls and not control_uploads:
        use_bundled_controls = st.checkbox(
            f"Or use the {len(bundled_controls)} bundled control files (development use only)"
        )

sample_source = sample_upload or (
    next(p for p in bundled_samples if p.name == bundled_sample_choice) if bundled_sample_choice else None
)

if sample_source is not None:
    filename = sample_source.name
    content_hash = _content_hash(sample_source)
    if workspace.sample is None or workspace.sample.content_hash != content_hash:
        workspace.sample = _cached_load_sample(content_hash, filename, sample_source)
        workspace.metadata = None  # force a metadata rebuild for the new sample
        workspace.spillover_matrix = None
        workspace.default_scenario = None
        workspace.last_diagnostic = None

control_sources = list(control_uploads) if control_uploads else (bundled_controls if use_bundled_controls else [])
control_hashes = tuple(sorted(_content_hash(c) for c in control_sources))
if st.session_state.get("_last_control_hashes") != control_hashes:
    st.session_state["_last_control_hashes"] = control_hashes
    workspace.spillover_matrix = None  # controls changed — force a scenario re-comparison
    workspace.default_scenario = None
    workspace.last_diagnostic = None

if control_sources:
    loaded_controls = []
    for control_source in control_sources:
        control_filename = control_source.name
        control_hash = _content_hash(control_source)
        loaded_controls.append(_cached_load_control(control_hash, control_filename, control_source))
    workspace.control_samples = loaded_controls
    workspace.controls_available = True
else:
    workspace.control_samples = []
    workspace.controls_available = False

if workspace.sample is None:
    st.info("Upload a sample file, or select a bundled sample, to continue.")
    st.stop()

sample = workspace.sample

st.divider()

# --- Metadata (automatic, minimal) ---

st.header("Metadata")

if workspace.metadata is None:
    workspace.metadata = _cached_metadata(sample.content_hash, sample)
metadata = workspace.metadata

category_order = {"scatter": 0, "time": 1, "target-marker": 2, "empty": 3}
display_metadata = metadata.copy()
display_metadata["_order"] = display_metadata["category"].map(category_order)
display_metadata = display_metadata.sort_values(["_order", "channel"]).drop(columns="_order")
display_metadata["target"] = display_metadata["target"].fillna("—")
display_metadata["marker"] = display_metadata["marker"].fillna("—")
display_metadata["category"] = display_metadata["category"].str.replace("-", " ").str.title()

st.dataframe(
    display_metadata.rename(columns={"channel": "Channel", "category": "Category", "target": "Target", "marker": "Marker"}),
    width="stretch",
    hide_index=True,
    column_config={
        "Category": st.column_config.TextColumn(width="small"),
    },
)

category_counts = metadata["category"].value_counts()
n_scatter_time = category_counts.get("scatter", 0) + category_counts.get("time", 0)
n_markers = category_counts.get("target-marker", 0)
st.caption(
    f"{len(metadata)} channels — {n_scatter_time} scatter/time, {n_markers} markers · "
    f"{sample.event_count:,} events"
)

array_size_mb = sample.raw_values.nbytes / (1024 * 1024)
if array_size_mb > 300:
    st.warning(
        f"This file's event data occupies approximately {array_size_mb:.0f} MB in memory, which is large "
        "relative to Streamlit Community Cloud's free-tier budget of approximately 1 GB. Certain views "
        "therefore render a subsample of the data to remain responsive."
    )

st.divider()

# --- Basic plots ---

st.header("Raw view")

rng = np.random.default_rng(0)
plot_idx = rng.choice(len(sample.raw_values), size=min(PLOT_SUBSAMPLE_SIZE, len(sample.raw_values)), replace=False)
workspace.plot_subsample_indices = plot_idx
plot_values = sample.raw_values[plot_idx]
channel_index = {name: i for i, name in enumerate(sample.pnn_labels)}

default_x_index = channel_index.get("FSC-A", 0)
default_y_index = channel_index.get("SSC-A", min(1, len(sample.pnn_labels) - 1))

col_a, col_b = st.columns(2)
with col_a:
    x_col, y_col = st.columns(2)
    with x_col:
        x_channel = st.selectbox("Scatter X axis", sample.pnn_labels, index=default_x_index)
    with y_col:
        y_channel = st.selectbox("Scatter Y axis", sample.pnn_labels, index=default_y_index)
    fig = build_scatter_figure(
        plot_values[:, channel_index[x_channel]], plot_values[:, channel_index[y_channel]],
        x_channel, y_channel, f"{x_channel} vs {y_channel}",
    )
    st.plotly_chart(fig, width="stretch")

with col_b:
    histogram_channel = st.selectbox("Histogram channel", sample.pnn_labels, index=0)
    fig = build_histogram_figure(plot_values[:, channel_index[histogram_channel]], histogram_channel)
    st.plotly_chart(fig, width="stretch")

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
    if st.button("← Back: Home", width="stretch"):
        st.switch_page("Home.py")
with next_col:
    if st.button("Next: Compensation →", width="stretch", type="primary", key="next_button"):
        st.switch_page("pages/2_Compensation.py")
