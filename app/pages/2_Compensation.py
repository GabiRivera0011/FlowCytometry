import numpy as np
import pandas as pd
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from flowcyto.compensation.diagnostics import (
    matrix_fingerprint,
    pick_default_scenario,
    run_slope_diagnostic,
    run_validated_search,
    score_scenario,
)
from flowcyto.compensation.matrix import apply_matrix_pct_negative, build_control_matrix, read_spill_metadata
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import build_before_after_bar, build_ranked_bar
from flowcyto.viz.heatmap import build_slope_heatmap, build_spillover_heatmap
from flowcyto.viz.pairs import build_notable_pairs_grid, build_pair_detail_figure, rank_notable_pairs

st.set_page_config(page_title="Compensation — FlowCytometry", layout="wide")
st.title("Compensation Assessment")

with st.expander("How the scoring works"):
    st.markdown(
        "Each matrix is scored with a Theil-Sen slope fit per channel pair, "
        "rather than a simple correlation. Theil-Sen (the median of all "
        "pairwise slopes) is robust to the outlier-heavy distributions "
        "typical of flow cytometry data, unlike ordinary least squares.\n\n"
        "The fit uses only events where the source channel is in its "
        "dimmest 20th percentile. At that low level, real co-expression "
        "biology should not produce a relationship, so any slope that "
        "remains points to spillover or autofluorescence instead.\n\n"
        "A negative slope is the tell for over-compensation: real spillover "
        "only adds signal, it never subtracts it. The score shown below "
        '("% flagged pairs") is the average, across channels, of how many '
        "incoming pairs come out negative. Lower is better. It is what the "
        "app uses to pick the default matrix and, once you run the full "
        "diagnostic, to rank the notable pairs further down this page."
    )

workspace = require_sample()
sample = workspace.sample
fluor_channels = sample.fluorescence_channels
raw_values = sample.raw_values[:, [sample.pnn_labels.index(c) for c in fluor_channels]]

# Unlabeled detector channels (no marker assigned) still take part in the
# compensation matrix — spillover is a detector property, not a naming one —
# but they carry no marker identity, so the EDA views below (which are about
# marker relationships) are restricted to named channels only.
_empty_channels = set(build_channel_metadata(sample).query("category == 'empty'")["channel"])
marker_channels = [c for c in fluor_channels if c not in _empty_channels]


@st.cache_data(show_spinner="Comparing compensation scenarios...")
def _cached_scenarios(sample_hash: str, control_hashes: tuple, _raw_values, _fluor_channels, _sample, _control_samples):
    identity_matrix = pd.DataFrame(np.eye(len(_fluor_channels)), index=_fluor_channels, columns=_fluor_channels)
    scenarios = [score_scenario(_raw_values, _fluor_channels, "Already compensated (raw)", identity_matrix, "No correction applied.")]
    uncontrolled, control_files = [], {}
    if _control_samples:
        control_matrix, uncontrolled, control_files = build_control_matrix(_sample, _control_samples)
        control_matrix = control_matrix.loc[_fluor_channels, _fluor_channels]
        detail = f"Built from {len(control_files)} uploaded single-stain control(s)."
        scenarios.append(score_scenario(_raw_values, _fluor_channels, "Control-based matrix", control_matrix, detail))
    return scenarios, uncontrolled, control_files


@st.cache_data(show_spinner="Running full diagnostic (this takes ~20-30s)...")
def _cached_diagnostic(sample_hash: str, matrix_fp: str, _raw_values, _fluor_channels, _matrix):
    return run_slope_diagnostic(_raw_values, _fluor_channels, _matrix, fit_size=1200)


@st.cache_data(show_spinner="Searching for a better matrix (no controls — this takes ~20-30s)...")
def _cached_validated_search(sample_hash: str, _raw_values, _fluor_channels):
    return run_validated_search(_raw_values, _fluor_channels)


control_hashes = tuple(sorted(c.content_hash for c in workspace.control_samples))
scenarios, uncontrolled, control_files = _cached_scenarios(
    sample.content_hash, control_hashes, raw_values, fluor_channels, sample, workspace.control_samples
)
workspace.uncontrolled_channels = uncontrolled
workspace.control_filenames = control_files

# --- No controls: offer the validated automatic search as a real (if likely small) improvement ---
st.header("Compensation scenario")

if not workspace.controls_available:
    st.caption("No compensation controls uploaded — comparing raw data against an automatic, control-free search.")
    if st.button("Run automatic search"):
        search_matrix, accepted = _cached_validated_search(sample.content_hash, raw_values, fluor_channels)
        detail = (
            f"Searched {len(fluor_channels) * (len(fluor_channels) - 1)} channel pairs; "
            f"{len(accepted)} held-out-validated correction(s) found."
        )
        scenarios.append(score_scenario(raw_values, fluor_channels, "Automatic search", search_matrix, detail))
        st.session_state["_search_accepted"] = accepted

if workspace.spillover_matrix is None:
    workspace.default_scenario = pick_default_scenario(scenarios)
    workspace.spillover_matrix = workspace.default_scenario.matrix.copy()

# Headline
best = pick_default_scenario(scenarios)
score_lines = "; ".join(f"{s.name}: {s.score:.1f}% flagged pairs" for s in scenarios)
if best.score < 5:
    st.success(f"Showing: **{best.name}** — {best.detail} ({score_lines})")
else:
    st.warning(f"Showing: **{best.name}** — {best.detail} ({score_lines})")

if workspace.uncontrolled_channels:
    st.caption(f"{len(workspace.uncontrolled_channels)}/{len(fluor_channels)} channels have no matching control (left as identity).")

st.divider()

# --- Editable matrix ---
st.header("Compensation matrix")
st.caption("Edit any cell, then **Apply changes**. **Reset matrix** restores the scenario chosen above.")

edited = st.data_editor(
    workspace.spillover_matrix,
    key="matrix_editor",
    width="stretch",
    column_config={
        channel: st.column_config.NumberColumn(channel, format="%.3f", step=0.001)
        for channel in fluor_channels
    },
)

apply_col, _spacer_col, reset_col = st.columns([1, 3, 1])
with apply_col:
    if st.button("Apply changes"):
        try:
            np.linalg.inv(edited.values)
        except np.linalg.LinAlgError:
            st.error("That edit makes the matrix singular (not invertible) — not applied.")
        else:
            workspace.spillover_matrix = edited.copy()
            st.rerun()
with reset_col:
    if st.button("Reset matrix"):
        workspace.spillover_matrix = best.matrix.copy()
        if "matrix_editor" in st.session_state:
            del st.session_state["matrix_editor"]
        st.rerun()

current_matrix = workspace.spillover_matrix
pct_negative_before = (raw_values < 0).mean(axis=0) * 100
pct_negative_after = apply_matrix_pct_negative(raw_values, current_matrix)

st.plotly_chart(
    build_before_after_bar(
        fluor_channels, pct_negative_before, pct_negative_after,
        "raw (already-compensated)", "current matrix",
        "Percent of events with a negative value, per channel", "% negative events",
        before_color="#4d4d4d", after_color="#8dd3c7",
        xaxis_range=(0, 50), reference_line_x=10,
    ),
    width="stretch",
)

st.divider()

# --- Ranked pairs + notable-pairs grid ---
st.header("Exploratory Compensation Matrix (ECM)")

matrix_marker = current_matrix.loc[marker_channels, marker_channels]
diag = workspace.last_diagnostic
diag_is_current = diag is not None and diag.for_matrix_fingerprint == matrix_fingerprint(current_matrix)

if diag_is_current:
    ranking_source = diag.slopes.loc[marker_channels, marker_channels]
    ranking_note = "Ranked by the full diagnostic's observed channel-to-channel slope — data-driven, not just the matrix's declared coefficients."
    ranked_bar_title, ranked_bar_xaxis = "Most notable channel pairs (by observed slope)", "slope"
else:
    ranking_source = matrix_marker
    ranking_note = "Ranked by the current matrix's coefficients. Run the full diagnostic below for data-driven ranking — matrix coefficients are 0 for every pair on an unmodified identity matrix, so this ranking is uninformative until then."
    ranked_bar_title, ranked_bar_xaxis = "Largest spillover coefficients in the current matrix", "coefficient"

top_pairs = rank_notable_pairs(ranking_source, top_k=10)
st.plotly_chart(
    build_ranked_bar(
        [f"{f} → {t}" for _, f, t in top_pairs], [v for v, _, _ in top_pairs],
        ranked_bar_title, ranked_bar_xaxis,
    ),
    width="stretch",
)
st.caption(ranking_note)

if st.button("Run full diagnostic"):
    diagnostic = _cached_diagnostic(sample.content_hash, str(hash(tuple(current_matrix.values.flatten()))), raw_values, fluor_channels, current_matrix)
    workspace.last_diagnostic = diagnostic
    st.rerun()

if workspace.last_diagnostic is not None:
    diag = workspace.last_diagnostic
    st.plotly_chart(
        build_ranked_bar(
            fluor_channels, diag.pct_negative_per_channel, "% of incoming pairs with a negative slope", "% negative-slope pairs",
            xaxis_range=(0, 50), reference_line_x=10,
        ),
        width="stretch",
    )

st.subheader("Notable channel pairs")
grid_idx = workspace.plot_subsample_indices if workspace.plot_subsample_indices is not None else np.arange(min(5000, len(raw_values)))
compensated_subsample = raw_values[grid_idx] @ np.linalg.inv(current_matrix.values)
grid_pairs = rank_notable_pairs(ranking_source, top_k=12)
grid_fig = build_notable_pairs_grid(compensated_subsample, fluor_channels, grid_pairs)
grid_event = st.plotly_chart(grid_fig, width="stretch", on_select="rerun", key="pairs_grid")

selected_pair = None
if grid_event and grid_event.selection and grid_event.selection.points:
    curve_number = grid_event.selection.points[0]["curve_number"]
    if curve_number < len(grid_pairs):
        selected_pair = grid_pairs[curve_number][1:]

st.caption("Or pick any pair directly:")
pick_col1, pick_col2 = st.columns(2)
with pick_col1:
    x_channel = st.selectbox("Channel X", marker_channels, index=marker_channels.index(selected_pair[0]) if selected_pair else 0)
with pick_col2:
    y_channel = st.selectbox("Channel Y", marker_channels, index=marker_channels.index(selected_pair[1]) if selected_pair else 1)

st.plotly_chart(build_pair_detail_figure(compensated_subsample, fluor_channels, x_channel, y_channel), width="stretch")

st.divider()

# --- Full matrix ---
with st.expander("Full matrix (heatmap)"):
    st.caption(
        "This heatmap shows the compensation matrix currently in effect: "
        "identity, control-derived, or your edits above."
    )
    st.plotly_chart(build_spillover_heatmap(current_matrix, fluor_channels, "Current spillover matrix"), width="stretch")
    if workspace.last_diagnostic is not None:
        st.caption(
            "This heatmap shows the Theil-Sen slope diagnostic computed from the "
            "actual event data (see 'How the scoring works' above)."
        )
        bound = float(np.percentile(np.abs(workspace.last_diagnostic.slopes.values), 90))
        st.plotly_chart(
            build_slope_heatmap(workspace.last_diagnostic.slopes, fluor_channels, "Slope diagnostic", bound),
            width="stretch",
        )

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
    if st.button("← Back: Parse and View", width="stretch"):
        st.switch_page("pages/1_Parse_and_View.py")
with next_col:
    if st.button("Next: Gating — Live Cells →", width="stretch", type="primary", key="next_button"):
        st.switch_page("pages/3_Gating_-_Live_Cells.py")
