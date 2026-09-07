from pathlib import Path

import numpy as np

from flowcyto.compensation.diagnostics import pick_default_scenario, score_scenario, slope_matrix
from flowcyto.compensation.matrix import apply_matrix_pct_negative, build_control_matrix, fluorescence_indices
from flowcyto.io.fcs import load_fcs

FIXTURES = Path(__file__).parent / "fixtures"


def _load_sample_and_controls():
    sample = load_fcs(FIXTURES / "pbmc_sample.fcs", "pbmc_sample.fcs")
    controls = [
        load_fcs(FIXTURES / "comp_cd11c_apc.fcs", "comp_cd11c_apc.fcs"),
        load_fcs(FIXTURES / "comp_cd123_bb660.fcs", "comp_cd123_bb660.fcs"),
    ]
    return sample, controls


def test_build_control_matrix_controls_two_channels_leaves_rest_identity():
    sample, controls = _load_sample_and_controls()
    matrix, uncontrolled, control_files = build_control_matrix(sample, controls)

    fluor_channels = [sample.pnn_labels[i] for i in fluorescence_indices(sample)]
    assert set(matrix.index) == set(fluor_channels)
    assert len(control_files) == 2
    assert len(uncontrolled) == len(fluor_channels) - 2

    # diagonal is always 1
    assert np.allclose(np.diag(matrix.values), 1.0)
    # an uncontrolled channel's row is untouched identity
    uncontrolled_channel = uncontrolled[0]
    row = matrix.loc[uncontrolled_channel]
    assert row[uncontrolled_channel] == 1.0
    assert (row.drop(uncontrolled_channel) == 0.0).all()


def test_apply_matrix_pct_negative_is_a_noop_for_identity():
    sample, _ = _load_sample_and_controls()
    fluor_idx = fluorescence_indices(sample)
    fluor_channels = [sample.pnn_labels[i] for i in fluor_idx]
    raw_values = sample.raw_values[:, fluor_idx]

    import pandas as pd

    identity = pd.DataFrame(np.eye(len(fluor_channels)), index=fluor_channels, columns=fluor_channels)
    pct_negative = apply_matrix_pct_negative(raw_values, identity)
    expected = (raw_values < 0).mean(axis=0) * 100
    assert np.allclose(pct_negative, expected)


def test_slope_matrix_diagonal_is_untouched_zero():
    sample, _ = _load_sample_and_controls()
    fluor_idx = fluorescence_indices(sample)
    fluor_channels = [sample.pnn_labels[i] for i in fluor_idx]
    raw_values = sample.raw_values[:, fluor_idx]

    slopes = slope_matrix(raw_values, fluor_channels, fit_size=100, sample_size=2000)
    assert np.allclose(np.diag(slopes.values), 0.0)


def test_pick_default_scenario_returns_lowest_score():
    sample, controls = _load_sample_and_controls()
    fluor_idx = fluorescence_indices(sample)
    fluor_channels = [sample.pnn_labels[i] for i in fluor_idx]
    raw_values = sample.raw_values[:, fluor_idx]

    import pandas as pd

    identity = pd.DataFrame(np.eye(len(fluor_channels)), index=fluor_channels, columns=fluor_channels)
    worse = identity.copy()
    worse.iloc[0, 1] = 5.0  # an absurd coefficient guaranteed to score worse

    good_scenario = score_scenario(raw_values, fluor_channels, "good", identity, "", fit_size=100)
    bad_scenario = score_scenario(raw_values, fluor_channels, "bad", worse, "", fit_size=100)

    assert pick_default_scenario([good_scenario, bad_scenario]).name == "good"
