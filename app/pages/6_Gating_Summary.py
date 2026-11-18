import numpy as np
import pandas as pd
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from _gating_widgets import collect_section_gates, resolve_full_mask
from flowcyto.transforms import arcsinh_transform

st.set_page_config(page_title="Gating Summary — FlowCytometry", layout="wide")
st.title("Gating Summary")

workspace = require_sample()
sample = workspace.sample

if workspace.plot_subsample_indices is None:
    st.info("No gating has been performed yet. Visit the gating pages first, then return to this page.")
    st.stop()
n_subsample = len(workspace.plot_subsample_indices)
n_full = len(sample.raw_values)

subsample_pct = (n_subsample / n_full * 100) if n_full else 0.0
st.caption(
    f"This sample contains **{n_full:,}** events in total. For performance, the gating pages render a fixed "
    f"**{n_subsample:,}**-event random subsample (**{subsample_pct:.1f}%** of the file); however, each count "
    f"below is computed by applying the exact boundary of each gate to the full, un-subsampled file, rather "
    f"than extrapolating from the subsample."
)

channel_index = {name: i for i, name in enumerate(sample.pnn_labels)}
fluor_channels = sample.fluorescence_channels
fluor_indices = tuple(channel_index[c] for c in fluor_channels)


@st.cache_data(show_spinner=False)
def _full_compensated_fluor(raw_values: np.ndarray, fluor_idx: tuple, spillover_values: np.ndarray) -> np.ndarray:
    return raw_values[:, list(fluor_idx)] @ np.linalg.inv(spillover_values)


if workspace.spillover_matrix is not None:
    full_fluor = _full_compensated_fluor(sample.raw_values, fluor_indices, workspace.spillover_matrix.values)
else:
    full_fluor = None


def full_channel_values(channel: str) -> np.ndarray:
    # Must match the gating pages' own `channel_values` transform exactly —
    # a gate's saved boundary was drawn in arcsinh-transformed coordinates
    # for fluorescence channels, so resolving it against the full file has
    # to test in that same transformed space, not raw linear units.
    if channel in fluor_channels:
        raw = full_fluor[:, fluor_channels.index(channel)] if full_fluor is not None else sample.raw_values[:, channel_index[channel]]
        return arcsinh_transform(raw)
    return sample.raw_values[:, channel_index[channel]]


full_mask_cache: dict = {}
live_n_full = None
if workspace.live_cells_mask_ref is not None:
    live_n_full = int(resolve_full_mask(workspace.live_cells_mask_ref, full_channel_values, n_full, full_mask_cache).sum())

rows: list[dict] = []


def add_rows(page_label: str, section: str, location_label: str) -> None:
    for gate in collect_section_gates(section):
        n_sub = int(gate["mask"].sum())
        n_full_count = int(resolve_full_mask(gate["ref"], full_channel_values, n_full, full_mask_cache).sum())
        pct_sub = (n_sub / n_subsample * 100) if n_subsample else 0.0
        pct_full = (n_full_count / n_full * 100) if n_full else 0.0
        row = {
            "Page": page_label,
            "Gate location": location_label,
            "Gate": gate["name"],
            "Events (subsample)": n_sub,
            "% of subsample": pct_sub,
            "Events in full file": n_full_count,
            "% of full file": pct_full,
            "Subsample vs. full-file gap (pp)": pct_full - pct_sub,
        }
        if live_n_full is not None:
            row["% of Live cells"] = (n_full_count / live_n_full * 100) if live_n_full else 0.0
        rows.append(row)


# --- Gating — Live Cells ---
for section, location_label in [
    ("acquisition", "Acquisition"), ("debris", "Cellular debris"),
    ("singlets", "Single cells"), ("live", "Live cells"),
]:
    add_rows("Gating — Live Cells", section, location_label)

# --- Gating — Broad Cellular Lineage ---
for section, location_label in [("lymphocytes", "Lymphocytes"), ("myeloid", "Myeloid cells")]:
    add_rows("Gating — Broad Cellular Lineage", section, location_label)

# --- Gating — Subset Cell Population (dynamic pages/plots) ---
for page_id in st.session_state.get("subset_pages", []):
    page_title = st.session_state.get(f"subset_page_{page_id}_title", "Page")
    for plot_id in st.session_state.get(f"subset_page_{page_id}_plot_ids", []):
        section = f"subset_{page_id}_{plot_id}"
        plot_title = st.session_state.get(f"subset_{section}_title", "Plot")
        add_rows("Gating — Subset Cell Population", section, f"{page_title} — {plot_title}")

if not rows:
    st.info("No gates have been defined yet. Visit the gating pages, define one or more gates, then return to this page.")
else:
    df = pd.DataFrame(rows)

    divergence_threshold = st.slider(
        "Flag gates where the subsample and full-file percentages diverge by more than this many percentage points",
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
        help="A gate boundary drawn against the sparse subsample can end up placed slightly off from where it "
        "would fall against the full, denser file. When a gate's two percentages differ by more than this "
        "amount, its boundary is likely worth revisiting: try switching that gate's View to Pseudocolor or "
        "Contour, then redraw it.",
    )
    flagged = df[df["Subsample vs. full-file gap (pp)"].abs() > divergence_threshold]
    if not flagged.empty:
        gate_list = "; ".join(
            f"**{row['Gate location']} — {row['Gate']}** ({row['Subsample vs. full-file gap (pp)']:+.1f}pp)"
            for _, row in flagged.iterrows()
        )
        st.warning(
            f"{len(flagged)} gate(s) show a subsample-vs-full-file gap larger than {divergence_threshold:.1f} "
            f"percentage points: {gate_list}. This usually indicates the boundary was drawn against a sparse "
            "or misleading view of the population; consider switching that gate's View to Pseudocolor or "
            "Contour and redrawing it."
        )

    column_config = {
        "Events (subsample)": st.column_config.NumberColumn(format="%d"),
        "% of subsample": st.column_config.NumberColumn(format="%.1f%%"),
        "Events in full file": st.column_config.NumberColumn(format="%d"),
        "% of full file": st.column_config.NumberColumn(format="%.1f%%"),
        "Subsample vs. full-file gap (pp)": st.column_config.NumberColumn(format="%+.1f"),
    }
    if "% of Live cells" in df.columns:
        column_config["% of Live cells"] = st.column_config.NumberColumn(format="%.1f%%")

    st.dataframe(df, width="stretch", hide_index=True, column_config=column_config)

    st.download_button(
        "Download as CSV", df.to_csv(index=False), file_name="gated_cell_counts.csv", mime="text/csv",
    )

st.divider()
if st.button("← Back: Subset Cell Population", width="stretch"):
    st.switch_page("pages/5_Gating_-_Subset_Cell_Population.py")
