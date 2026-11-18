import streamlit as st

from _common import ensure_src_on_path, get_workspace

ensure_src_on_path()

from flowcyto.viz.lineage import (
    build_b_cell_diagram,
    build_cd4_differentiation_diagram,
    build_cd8_differentiation_diagram,
    build_lineage_diagram,
    build_myeloid_nk_diagram,
    build_th_diagram,
    build_treg_diagram,
)

st.set_page_config(page_title="FlowCytometry", layout="wide")
st.title("Immune Cell Identification using Flow Cytometry")
st.caption("A flow cytometry analysis workflow.")

workspace = get_workspace()

st.markdown(
    "Load an `.fcs` or `.csv` sample, assess and adjust compensation, then gate "
    "the populations of interest. Each step builds upon the one before it and "
    "is accessible from its own page in the sidebar."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Parse and View")
    st.write("Load a sample, with optional compensation controls, to view channel metadata and preview raw scatter and histogram plots.")
    st.page_link("pages/1_Parse_and_View.py", label="Open →")

with col2:
    st.subheader("2. Compensation")
    st.write("Compare compensation scenarios, edit the spillover matrix directly, and inspect the most notable channel pairs.")
    st.page_link("pages/2_Compensation.py", label="Open →")

with col3:
    st.subheader("3. Gating — Live Cells")
    st.write("Exclude debris, doublets, and dead cells through acquisition quality control, forward/side scatter, singlet discrimination, and viability gating.")
    st.page_link("pages/3_Gating_-_Live_Cells.py", label="Open →")

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("4. Gating — Broad Cellular Lineage")
    st.write("Split live cells into lymphocytes vs. myeloid cells.")
    st.page_link("pages/4_Gating_-_Broad_Cellular_Lineage.py", label="Open →")

with col5:
    st.subheader("5. Gating — Subset Cell Population")
    st.write("Define custom gating workspaces for further rounds of subsetting, branching from any earlier gate as needed.")
    st.page_link("pages/5_Gating_-_Subset_Cell_Population.py", label="Open →")

with col6:
    st.subheader("6. Gating Summary")
    st.write("View every gate defined across the gating pages in a single table, including event counts, percentage of subsample, percentage of Live cells, and exact counts in the full file.")
    st.page_link("pages/6_Gating_Summary.py", label="Open →")

col7, _col8, _col9 = st.columns(3)

with col7:
    st.subheader("7. Marker Map")
    st.write("A literal 3D scatter of every subsampled event across three markers you pick directly, colored by a 4th marker or by one or more already-defined gates.")
    st.page_link("pages/7_Marker_Map.py", label="Open →")

st.divider()

if workspace.sample is None:
    st.info("No sample has been loaded yet. Begin with **Parse and View**.")
else:
    sample = workspace.sample
    status = f"**{sample.filename}** loaded: {sample.event_count:,} events, {len(sample.pnn_labels)} channels"
    if workspace.controls_available:
        status += f", {len(workspace.control_samples)} compensation control(s)"
    st.success(status)

st.divider()

st.subheader("Identifying immune cell subsets by gating")
st.write(
    "Each gate applies one marker decision that narrows live cells toward a "
    "specific lineage. Population identity depends on the combination of "
    "markers co-expressed by a cell and their relative intensity."
)
st.plotly_chart(build_lineage_diagram(), width="stretch", config={"displayModeBar": False})
st.caption(
    "Illustrative gating tree (original), following the general strategy "
    "described in Maecker, H. T., McCoy, J. P., & Nussenblatt, R. (2012). "
    "Standardizing immunophenotyping for the Human Immunology Project. "
    "*Nature Reviews Immunology*, 12(3), 191–200. "
    "[doi:10.1038/nri3158](https://doi.org/10.1038/nri3158)."
)

with st.expander("T cell subsets in detail: naive/memory/effector, T-reg, TH1/TH2/TH17"):
    st.markdown("**CD4+ differentiation**")
    st.plotly_chart(build_cd4_differentiation_diagram(), width="stretch", config={"displayModeBar": False})
    st.markdown("**CD8+ differentiation**")
    st.plotly_chart(build_cd8_differentiation_diagram(), width="stretch", config={"displayModeBar": False})
    st.markdown("**T-reg subsets**")
    st.plotly_chart(build_treg_diagram(), width="stretch", config={"displayModeBar": False})
    st.markdown("**TH1/TH2/TH17 subsets**")
    st.plotly_chart(build_th_diagram(), width="stretch", config={"displayModeBar": False})

with st.expander("B cell subsets in detail: naive, memory, transitional, plasmablast"):
    st.plotly_chart(build_b_cell_diagram(), width="stretch", config={"displayModeBar": False})

with st.expander("Myeloid and NK subsets in detail: monocytes, dendritic cells, NK cells"):
    st.plotly_chart(build_myeloid_nk_diagram(), width="stretch", config={"displayModeBar": False})

st.divider()

st.subheader("About the sample dataset")
st.markdown(
    """
The sample dataset comes from a published dendritic cell and monocyte
phenotyping panel:

**Mair, F., & Liechti, T. (2021).** Comprehensive Phenotyping of Human
Dendritic Cells and Monocytes. *Cytometry Part A*, 99(3), 231–242.
[doi:10.1002/cyto.a.24269](https://doi.org/10.1002/cyto.a.24269)

The paper addresses the increasingly blurred distinction between human
dendritic cells (DCs) and monocytes, proposing two marker-pair additions to
existing panels: **CD88/CD89** to positively identify monocytes, and
**CD5/CD163** to subset cDC2 from the newer DC3 population.
"""
)

st.markdown("**Marker-to-panel role mapping**")
st.markdown(
    """
| Columns | Role |
|---|---|
| `FSC-A`, `FSC-H`, `SSC-A`, `Time` | Acquisition channels: cell size/granularity and time |
| `Live Dead UV Blue` | Viability stain: excludes dead cells prior to phenotyping |
| `CD3`, `CD19`, `CD20`, `CD56` | Lineage-exclusion channel: T cells, B cells, and NK cells |
| `HLA-DR` | Pan-antigen-presenting-cell marker: gates in DCs/monocytes broadly |
| `CD14`, `CD16`, `CD88`, `CD89` | Monocyte identification (`CD88`/`CD89` are the paper's proposed addition) |
| `CD1c`, `CD172a` | cDC2 markers |
| `CD5`, `CD163` | cDC2-vs-DC3 subsetting (the paper's second proposed addition) |
| `CD141`, `XCR1`, `CADM1` | cDC1 markers |
| `CD303`, `FcER1a` | Plasmacytoid DC (pDC) markers |
| `CD11c`, `CD45RA`, `CD301` | Additional DC/monocyte phenotyping markers |
"""
)
