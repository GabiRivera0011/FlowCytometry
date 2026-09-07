from __future__ import annotations

import plotly.graph_objects as go

TEXT_COLOR = "#0b0b0b"
LYMPHOID_COLOR = "#80b1d3"
MYELOID_COLOR = "#fdb462"
ROOT_COLOR = "#8dd3c7"
TREG_COLOR = "#fccde5"
TH_COLOR = "#bebada"
BCELL_COLOR = "#8dd3c7"

BOX_HALF_WIDTH = 0.46
BOX_HALF_HEIGHT = 0.30


def _build_tree_figure(
    nodes: list[tuple], x_range: list[float], y_range: list[float],
    font_size: int = 12, height: int = 430, box_half_width: float = BOX_HALF_WIDTH,
    fig_width: int | None = None,
) -> go.Figure:
    """Shared renderer for an original gating-tree illustration: boxes for
    populations, lines for parent -> child gates. Not a reproduction of any
    published figure — general immunology gating logic, drawn fresh to match
    this app's own visual language."""
    positions = {node_id: (x, y) for node_id, _, x, y, _, _ in nodes}
    fig = go.Figure()

    for node_id, _, x, y, parent_id, _ in nodes:
        if parent_id is None:
            continue
        px, py = positions[parent_id]
        fig.add_shape(
            type="line", x0=px, y0=py - BOX_HALF_HEIGHT, x1=x, y1=y + BOX_HALF_HEIGHT,
            line=dict(color="#c3c2b7", width=1.5),
        )

    for _, label, x, y, _, color in nodes:
        fig.add_shape(
            type="rect",
            x0=x - box_half_width, x1=x + box_half_width,
            y0=y - BOX_HALF_HEIGHT, y1=y + BOX_HALF_HEIGHT,
            line=dict(color=TEXT_COLOR, width=1),
            fillcolor=color,
        )
        title, _, subtitle = label.partition("\n")
        text = f"<b>{title}</b>" + (f"<br>{subtitle}" if subtitle else "")
        fig.add_annotation(
            x=x, y=y, text=text, showarrow=False,
            font=dict(color=TEXT_COLOR, size=font_size), align="center",
        )

    layout_kwargs = dict(
        template="simple_white",
        showlegend=False,
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
    )
    if fig_width is not None:
        layout_kwargs["width"] = fig_width
    fig.update_layout(**layout_kwargs)
    return fig


# --- Overview: the top-level lineage split ---

OVERVIEW_NODES = [
    ("live", "PBMC (Live)", 3.0, 3, None, ROOT_COLOR),
    ("lymph", "Lymphocytes\nFSC/SSC: small, agranular", 1.3, 2, "live", LYMPHOID_COLOR),
    ("myeloid", "Myeloid cells\nFSC/SSC: larger, granular", 4.7, 2, "live", MYELOID_COLOR),
    ("t", "T cells\nCD3+", 0.3, 1, "lymph", LYMPHOID_COLOR),
    ("b", "B cells\nCD19+/CD20+", 1.3, 1, "lymph", LYMPHOID_COLOR),
    ("nk", "NK cells\nCD3-, CD56+", 2.3, 1, "lymph", LYMPHOID_COLOR),
    ("mono", "Monocytes\nCD14+", 4.1, 1, "myeloid", MYELOID_COLOR),
    ("dc", "Dendritic cells\nHLA-DR+", 5.4, 1, "myeloid", MYELOID_COLOR),
    ("th", "Helper T\nCD4+", -0.2, 0, "t", LYMPHOID_COLOR),
    ("tc", "Cytotoxic T\nCD8+", 0.8, 0, "t", LYMPHOID_COLOR),
]


def build_lineage_diagram() -> go.Figure:
    return _build_tree_figure(OVERVIEW_NODES, x_range=[-0.9, 6.1], y_range=[-0.5, 3.5])


# --- Detail: T cell subsets, split into four small trees so every box has
# room for its full marker label — one wide 20-node tree does not fit any
# reasonably-sized page column without shrinking text past legibility. ---

CD4_DIFFERENTIATION_NODES = [
    ("cd4", "CD4+ T cells\nHelper", 2.8, 1, None, LYMPHOID_COLOR),
    ("naive", "Naive\nCCR7+ CD45RA+", 0.0, 0, "cd4", LYMPHOID_COLOR),
    ("cm", "Central memory\nCCR7+ CD45RA-", 1.4, 0, "cd4", LYMPHOID_COLOR),
    ("em", "Effector memory\nCCR7- CD45RA-", 2.8, 0, "cd4", LYMPHOID_COLOR),
    ("eff", "Effector\nCCR7- CD45RA+", 4.2, 0, "cd4", LYMPHOID_COLOR),
    ("act", "Activated\nCD38+ HLA-DR+", 5.6, 0, "cd4", LYMPHOID_COLOR),
]


def build_cd4_differentiation_diagram() -> go.Figure:
    return _build_tree_figure(CD4_DIFFERENTIATION_NODES, x_range=[-0.7, 6.3], y_range=[-0.6, 1.6], height=290)


CD8_DIFFERENTIATION_NODES = [
    ("cd8", "CD8+ T cells\nCytotoxic", 2.8, 1, None, LYMPHOID_COLOR),
    ("naive", "Naive\nCCR7+ CD45RA+", 0.0, 0, "cd8", LYMPHOID_COLOR),
    ("cm", "Central memory\nCCR7+ CD45RA-", 1.4, 0, "cd8", LYMPHOID_COLOR),
    ("em", "Effector memory\nCCR7- CD45RA-", 2.8, 0, "cd8", LYMPHOID_COLOR),
    ("eff", "Effector\nCCR7- CD45RA+", 4.2, 0, "cd8", LYMPHOID_COLOR),
    ("act", "Activated\nCD38+ HLA-DR+", 5.6, 0, "cd8", LYMPHOID_COLOR),
]


def build_cd8_differentiation_diagram() -> go.Figure:
    return _build_tree_figure(CD8_DIFFERENTIATION_NODES, x_range=[-0.7, 6.3], y_range=[-0.6, 1.6], height=290)


TREG_NODES = [
    ("treg", "T-reg\nCCR4+ CD25+ CD127low", 1.0, 2, None, TREG_COLOR),
    ("naive", "Naive T-reg\nCD45RO-", 0.0, 1, "treg", TREG_COLOR),
    ("mem", "Memory T-reg\nCD45RO+", 2.0, 1, "treg", TREG_COLOR),
    ("act", "Activated T-reg\nHLA-DR+", 2.0, 0, "mem", TREG_COLOR),
]


def build_treg_diagram() -> go.Figure:
    return _build_tree_figure(TREG_NODES, x_range=[-0.6, 2.6], y_range=[-0.6, 2.6], height=330)


TH_NODES = [
    ("cd4", "CD4+ T cells\nHelper subsets", 1.2, 1, None, LYMPHOID_COLOR),
    ("th1", "TH1\nCXCR3+ CCR6-", 0.0, 0, "cd4", TH_COLOR),
    ("th2", "TH2\nCXCR3- CCR6-", 1.2, 0, "cd4", TH_COLOR),
    ("th17", "TH17\nCXCR3- CCR6+", 2.4, 0, "cd4", TH_COLOR),
]


def build_th_diagram() -> go.Figure:
    return _build_tree_figure(TH_NODES, x_range=[-0.6, 3.0], y_range=[-0.6, 1.6], height=290)


# --- Detail: B cell subsets (naive, memory, transitional, plasmablast) ---

B_CELL_NODES = [
    ("b", "B cells\nCD19+/CD20+", 2.0, 2, None, BCELL_COLOR),
    ("transitional", "Transitional B cell\nCD24hi CD38hi", 0.0, 1, "b", BCELL_COLOR),
    ("naive_b", "Naive B cell\nCD27-", 1.3, 1, "b", BCELL_COLOR),
    ("memory_b", "Memory B cell\nCD27+, IgD+/-", 2.7, 1, "b", BCELL_COLOR),
    ("plasmablast", "Plasmablast\nCD27+CD20-CD38+", 4.0, 1, "b", BCELL_COLOR),
]


def build_b_cell_diagram() -> go.Figure:
    return _build_tree_figure(B_CELL_NODES, x_range=[-0.6, 4.6], y_range=[0.5, 2.5], font_size=11, height=260)


# --- Detail: myeloid + NK subsets (monocytes, DCs, NK) ---

MYELOID_NK_NODES = [
    ("myeloid", "Myeloid cells & NK\nCD3-, CD19-", 3.0, 3, None, MYELOID_COLOR),
    ("mono", "Monocytes\nCD14+", 1.0, 2, "myeloid", MYELOID_COLOR),
    ("mono_classical", "Classical monocyte\nCD16-", 0.5, 1, "mono", MYELOID_COLOR),
    ("mono_nonclassical", "Non-classical monocyte\nCD16+", 1.5, 1, "mono", MYELOID_COLOR),
    ("dc_nk", "DCs & NK cells\nCD14-", 5.0, 2, "myeloid", MYELOID_COLOR),
    ("dc", "Dendritic cells\nHLA-DR+", 4.0, 1, "dc_nk", MYELOID_COLOR),
    ("myeloid_dc", "Myeloid DC\nCD11c+", 3.5, 0, "dc", MYELOID_COLOR),
    ("pdc", "Plasmacytoid DC\nCD123+", 4.5, 0, "dc", MYELOID_COLOR),
    ("nk", "NK cells\nCD16+", 6.0, 1, "dc_nk", LYMPHOID_COLOR),
    ("nk_low", "NK cell\nCD56low", 5.5, 0, "nk", LYMPHOID_COLOR),
    ("nk_hi", "NK cell\nCD56hi", 6.5, 0, "nk", LYMPHOID_COLOR),
]


def build_myeloid_nk_diagram() -> go.Figure:
    return _build_tree_figure(MYELOID_NK_NODES, x_range=[-0.1, 7.1], y_range=[-0.5, 3.5], font_size=10, height=430)
