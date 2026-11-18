import numpy as np

from flowcyto.viz.scatter3d import build_marker_scatter3d_figure


def _sample_xyz(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)


def test_build_marker_scatter3d_figure_continuous_color_mode():
    x, y, z = _sample_xyz()
    color_values = np.linspace(0, 1, len(x))

    fig = build_marker_scatter3d_figure(x, y, z, "CD3", "CD19", "CD56", color_values=color_values, color_label="CD14")

    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.marker.colorscale is not None
    assert trace.marker.colorbar.title.text == "CD14"
    assert fig.layout.scene.xaxis.title.text == "CD3"
    assert fig.layout.scene.yaxis.title.text == "CD19"
    assert fig.layout.scene.zaxis.title.text == "CD56"


def test_build_marker_scatter3d_figure_highlight_mode_adds_one_trace_per_gate():
    x, y, z = _sample_xyz(n=20)
    mask_a = np.array([True] * 10 + [False] * 10)
    mask_b = np.array([False] * 5 + [True] * 5 + [False] * 10)

    fig = build_marker_scatter3d_figure(
        x, y, z, "CD3", "CD19", "CD56",
        highlights=[(mask_a, "Lymphocytes", "#bc80bd"), (mask_b, "Myeloid", "#fb8072")],
    )

    assert len(fig.data) == 3  # base trace + 2 highlight traces
    assert fig.data[1].name == "Lymphocytes"
    assert fig.data[2].name == "Myeloid"
    assert len(fig.data[1].x) == mask_a.sum()
    assert len(fig.data[2].x) == mask_b.sum()


def test_build_marker_scatter3d_figure_skips_empty_highlight_masks():
    x, y, z = _sample_xyz(n=10)
    empty_mask = np.zeros(10, dtype=bool)

    fig = build_marker_scatter3d_figure(x, y, z, "CD3", "CD19", "CD56", highlights=[(empty_mask, "Nothing", "#bc80bd")])

    assert len(fig.data) == 1  # the empty-mask highlight is skipped entirely
