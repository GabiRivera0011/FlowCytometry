import numpy as np
import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

from _gating_widgets import (
    GATING_WIDGET_CSS,
    _persist_value,
    _persisted_default,
    collect_section_gates,
    extend_registry,
    render_population_picker,
    render_section_gates,
)
from flowcyto.io.metadata import build_channel_metadata
from flowcyto.viz.bars import dark_set3_color

st.set_page_config(page_title="Gating - Subset Cell Population — FlowCytometry", layout="wide")
st.title("Gating — Subset Cell Population")

st.html(
    """<style>
    div[class*="st-key-subset_plot_card_"] {
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        padding: 1.5rem 1.75rem;
        border: 2px solid #999 !important;
    }
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
n_events = len(plot_idx)

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
target_by_channel = dict(zip(metadata["channel"], metadata["target"]))


def channel_label(channel: str) -> str:
    target = target_by_channel.get(channel)
    return f"{channel} — {target}" if isinstance(target, str) and target else channel


if workspace.spillover_matrix is None:
    st.caption("Raw, uncompensated marker channels are shown. Visit the Compensation page first for a cleaner separation.")

st.caption(
    "This page supports custom gating workspaces for additional rounds of subsetting. Each page holds its own "
    "set of plots, and a later page may branch from a gate defined on any earlier one. Within a page, any number "
    "of plots may be added; each plot allows selection of any two channels and may be gated using boxes, lassos, "
    "points, or quadrants, consistent with the earlier gating pages."
)

channel_options = list(sample.pnn_labels)
default_x = "FSC-A" if "FSC-A" in channel_options else channel_options[0]
default_y = "SSC-A" if "SSC-A" in channel_options else channel_options[min(1, len(channel_options) - 1)]


# --- Page bookkeeping: each "page" is its own named, switchable set of
# plots — letting a user organize deep/branching subsetting into separate
# groups instead of one long scroll of unrelated plots. ---

def new_page_defaults(page_id: int) -> None:
    st.session_state[f"subset_page_{page_id}_title"] = f"Page {page_id + 1}"
    st.session_state[f"subset_page_{page_id}_plot_ids"] = [0]
    st.session_state[f"subset_page_{page_id}_plot_next_id"] = 1


if "subset_pages" not in st.session_state:
    st.session_state["subset_pages"] = [0]
    st.session_state["subset_pages_next_id"] = 1
    new_page_defaults(0)
    st.session_state["subset_active_page"] = 0


def remove_plot(page_id: int, plot_id: int) -> None:
    section = f"subset_{page_id}_{plot_id}"
    exact_keys = [
        f"n_gates_{section}", f"gate_shape_{section}", f"view_mode_{section}", f"axis_zoom_{section}",
        f"active_gate_{section}", f"pending_active_{section}",
        f"subset_{section}_title", f"subset_{section}_title_input", f"subset_{section}_x", f"subset_{section}_y",
        f"subset_{section}_picker", f"subset_{section}_picker_combine_op",
    ]
    for k in exact_keys:
        st.session_state.pop(k, None)
    prefixes = (f"gate_{section}_", f"plot_{section}_", f"reset_one_{section}", f"reset_all_{section}")
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in prefixes):
            del st.session_state[k]
    st.session_state[f"subset_page_{page_id}_plot_ids"].remove(plot_id)


def remove_page(page_id: int) -> None:
    for plot_id in list(st.session_state.get(f"subset_page_{page_id}_plot_ids", [])):
        remove_plot(page_id, plot_id)
    for k in [f"subset_page_{page_id}_title", f"subset_page_{page_id}_plot_ids", f"subset_page_{page_id}_plot_next_id"]:
        st.session_state.pop(k, None)
    st.session_state["subset_pages"].remove(page_id)


def page_gates(page_id: int) -> list[tuple[str, str, str, np.ndarray, tuple]]:
    """A page's gates, read straight from session_state rather than by
    re-rendering it — so a later page can start from an earlier page's
    gates even while that earlier page isn't the one currently shown."""
    entries: list[tuple[str, str, str, np.ndarray, tuple]] = []
    page_title = st.session_state.get(f"subset_page_{page_id}_title", "Page")
    for plot_id in st.session_state.get(f"subset_page_{page_id}_plot_ids", []):
        section = f"subset_{page_id}_{plot_id}"
        plot_title = st.session_state.get(f"subset_{section}_title", "Plot")
        for g in collect_section_gates(section):
            entries.append((f"{page_title} — {plot_title}", g["name"], g["color"], g["mask"], g["ref"]))
    return entries


# A widget's own session_state key can't be reassigned after it's been
# instantiated this run — so switching pages (new/removed) is staged here,
# before the radio exists, and consumed from a pending flag set below.
# Falling back to the persisted mirror covers the other way this key goes
# missing: Streamlit drops it outright after a visit to another page, with
# no pending flag involved — otherwise you'd always land back on Page 1.
if "pending_active_subset_page" in st.session_state:
    st.session_state["subset_active_page"] = st.session_state.pop("pending_active_subset_page")
elif "subset_active_page" not in st.session_state:
    restored = _persisted_default("subset_active_page", None)
    if restored in st.session_state["subset_pages"]:
        st.session_state["subset_active_page"] = restored

page_ids: list[int] = st.session_state["subset_pages"]
page_titles = {pid: st.session_state.get(f"subset_page_{pid}_title", f"Page") for pid in page_ids}

nav_col, add_col, remove_col = st.columns([5, 1, 1])
with nav_col:
    active_page = st.radio(
        "Subsetting page", page_ids, format_func=lambda pid: page_titles[pid], horizontal=True, key="subset_active_page",
    )
    _persist_value("subset_active_page", active_page)
with add_col:
    st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    if st.button("+ New page"):
        new_id = st.session_state["subset_pages_next_id"]
        st.session_state["subset_pages_next_id"] += 1
        new_page_defaults(new_id)
        st.session_state["subset_pages"].append(new_id)
        st.session_state["pending_active_subset_page"] = new_id
        st.rerun()
with remove_col:
    st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    if len(page_ids) > 1 and st.button("Remove page"):
        remove_page(active_page)
        st.session_state["pending_active_subset_page"] = st.session_state["subset_pages"][0]
        st.rerun()

st.session_state[f"subset_page_{active_page}_title"] = st.text_input(
    "Page title", value=page_titles[active_page], key=f"subset_page_{active_page}_title_input",
)

# Seed the starting-population registry with whatever upstream gates exist
# — Myeloid/Lymphocytes/Live cells from the earlier pages, plus every gate
# drawn on any OTHER subsetting page — so a plot here can branch off any of
# them, not just the plot immediately above it.
registry: list[tuple[str, str, str, np.ndarray, tuple]] = []
if workspace.myeloid_mask is not None and len(workspace.myeloid_mask) == n_events:
    extend_registry(registry, "Broad Cellular Lineage", (["Myeloid cells"], [dark_set3_color(2)], [workspace.myeloid_mask]), [workspace.myeloid_mask_ref or ("full",)])
if workspace.lymphocytes_mask is not None and len(workspace.lymphocytes_mask) == n_events:
    extend_registry(registry, "Broad Cellular Lineage", (["Lymphocytes"], [dark_set3_color(1)], [workspace.lymphocytes_mask]), [workspace.lymphocytes_mask_ref or ("full",)])
if workspace.live_cells_mask is not None and len(workspace.live_cells_mask) == n_events:
    extend_registry(registry, "Live Cells", (["Live cells"], [dark_set3_color(0)], [workspace.live_cells_mask]), [workspace.live_cells_mask_ref or ("full",)])
for pid in page_ids:
    if pid != active_page:
        registry.extend(page_gates(pid))
if not registry:
    st.info("No upstream gates have been defined yet; plots below will use the full subsampled population. Visit the earlier gating pages first for a cleaner starting point.")

plot_ids: list[int] = st.session_state[f"subset_page_{active_page}_plot_ids"]
if not plot_ids:
    st.info("No plots have been added to this page yet. Add one below to begin gating.")

previous_title = "Broad Cellular Lineage"
for idx, plot_id in enumerate(plot_ids):
    section = f"subset_{active_page}_{plot_id}"
    with st.container(key=f"subset_plot_card_{section}", border=True):
        title_col, remove_plot_col = st.columns([5, 1])
        with title_col:
            # A plain mirror key, not the widget's own — Streamlit forgets a
            # widget's value once it stops being rendered (e.g. this plot's
            # page isn't the active one), but `page_gates()` on another page
            # still needs to read this plot's title to label its gates.
            title = st.text_input("Plot title", value=st.session_state.get(f"subset_{section}_title", f"Plot {idx + 1}"), key=f"subset_{section}_title_input")
            st.session_state[f"subset_{section}_title"] = title
        with remove_plot_col:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if len(plot_ids) > 1 and st.button("Remove", key=f"subset_{section}_remove"):
                remove_plot(active_page, plot_id)
                st.rerun()

        # Defaults to the immediately prior plot on this page (matching the
        # sequential-flow default used on the earlier gating pages) for
        # every plot after the first; the first plot on a page defaults to
        # Myeloid cells, the natural starting point for this page.
        picker_key = f"subset_{section}_picker"
        if registry:
            incoming_mask = render_population_picker(registry, previous_title, key=picker_key, n_events=n_events)
        else:
            incoming_mask = np.ones(n_events, dtype=bool)
        st.session_state[f"incoming_ref_{section}"] = ("picker", picker_key)

        # Streamlit drops a selectbox's own stored value once it stops being
        # rendered for even one run — i.e. the moment the user visits
        # another page — silently resetting it to `index=`'s default. The
        # persisted mirror is what makes a chosen marker pair stick.
        x_key, y_key = f"subset_{section}_x", f"subset_{section}_y"
        restored_x = _persisted_default(x_key, default_x)
        restored_y = _persisted_default(y_key, default_y)
        restored_x = restored_x if restored_x in channel_options else default_x
        restored_y = restored_y if restored_y in channel_options else default_y
        axis_col1, axis_col2 = st.columns(2)
        with axis_col1:
            x_channel = st.selectbox("X axis marker", channel_options, index=channel_options.index(restored_x), format_func=channel_label, key=x_key)
            _persist_value(x_key, x_channel)
        with axis_col2:
            y_channel = st.selectbox("Y axis marker", channel_options, index=channel_options.index(restored_y), format_func=channel_label, key=y_key)
            _persist_value(y_key, y_channel)

        names, colors, masks, refs = render_section_gates(
            section, incoming_mask, x_channel, y_channel, channel_values(x_channel), channel_values(y_channel), color_offset=(idx * 2) % 6,
        )
        extend_registry(registry, title, (names, colors, masks), refs)
        previous_title = title

if st.button("+ Add another plot"):
    new_id = st.session_state[f"subset_page_{active_page}_plot_next_id"]
    st.session_state[f"subset_page_{active_page}_plot_next_id"] += 1
    st.session_state[f"subset_page_{active_page}_plot_ids"].append(new_id)
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
    if st.button("← Back: Broad Cellular Lineage", width="stretch"):
        st.switch_page("pages/4_Gating_-_Broad_Cellular_Lineage.py")
with next_col:
    if st.button("Next: Tabulation →", width="stretch", type="primary", key="next_button"):
        st.switch_page("pages/6_Gating_-_Tabulation.py")
