import streamlit as st

from _common import ensure_src_on_path, require_sample

ensure_src_on_path()

st.set_page_config(page_title="Gating - Live Cells — FlowCytometry", layout="wide")
st.title("Gating — Live Cells")

workspace = require_sample()
sample = workspace.sample

st.info("Coming soon — this page will gate down to live, single cells before further analysis.")
st.caption(f"Current sample: **{sample.filename}** — {sample.event_count:,} events, {len(sample.pnn_labels)} channels.")
