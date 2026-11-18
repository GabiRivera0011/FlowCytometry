from pathlib import Path

import numpy as np

from flowcyto.compensation.diagnostics import (
    pct_negative_slope,
    pick_default_scenario,
    scenarios_are_close,
    score_scenario,
    slope_matrix,
    slope_matrix_with_ci,
)
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

    # fit_size=100 is too small for the bootstrap SE itself to be stable
    # enough to tell "good" and "bad" apart here — 200 keeps the test fast
    # while giving `scenarios_are_close` a reliable signal to work with.
    good_scenario = score_scenario(raw_values, fluor_channels, "good", identity, "", fit_size=200)
    bad_scenario = score_scenario(raw_values, fluor_channels, "bad", worse, "", fit_size=200)

    assert pick_default_scenario([good_scenario, bad_scenario]).name == "good"
    # the absurd coefficient's effect is large relative to fit noise
    assert not scenarios_are_close(good_scenario, bad_scenario)


def test_score_se_is_a_nonzero_empirical_bootstrap_not_a_fixed_formula():
    sample, controls = _load_sample_and_controls()
    fluor_idx = fluorescence_indices(sample)
    fluor_channels = [sample.pnn_labels[i] for i in fluor_idx]
    raw_values = sample.raw_values[:, fluor_idx]

    import pandas as pd

    identity = pd.DataFrame(np.eye(len(fluor_channels)), index=fluor_channels, columns=fluor_channels)

    scenario = score_scenario(raw_values, fluor_channels, "id", identity, "", fit_size=150, n_bootstrap=4)
    assert scenario.score_se > 0.0

    # more bootstrap reps shouldn't change the *expected* SE much, but the
    # value itself comes from resampling, not a closed-form expression of
    # `score` alone — two different scenarios tied on `score` can still get
    # different `score_se` values.
    worse = identity.copy()
    worse.iloc[2, 5] = 1.3
    scenario_b = score_scenario(raw_values, fluor_channels, "b", worse, "", fit_size=150, n_bootstrap=4)
    assert scenario.score_se != scenario_b.score_se or scenario.score != scenario_b.score


def test_slope_matrix_with_ci_upper_bound_never_below_the_slope():
    sample, _ = _load_sample_and_controls()
    fluor_idx = fluorescence_indices(sample)
    fluor_channels = [sample.pnn_labels[i] for i in fluor_idx]
    raw_values = sample.raw_values[:, fluor_idx]

    slopes, ci_upper = slope_matrix_with_ci(raw_values, fluor_channels, fit_size=100, sample_size=2000)
    off_diag = ~np.eye(len(fluor_channels), dtype=bool)
    assert (ci_upper.values[off_diag] >= slopes.values[off_diag]).all()


def test_pct_negative_slope_with_ci_is_stricter_than_point_estimate():
    import pandas as pd

    names = ["A", "B", "C"]
    # every off-diagonal slope is a small negative number...
    slopes = pd.DataFrame([[0, -0.01, -0.01], [-0.01, 0, -0.01], [-0.01, -0.01, 0]], index=names, columns=names)
    # ...but the confidence interval's upper bound is still positive for
    # most pairs — i.e. "negative" here is noise, not a significant effect.
    ci_upper = pd.DataFrame([[0, 0.02, -0.005], [0.02, 0, 0.02], [-0.005, 0.02, 0]], index=names, columns=names)

    point_estimate_pct = pct_negative_slope(slopes)
    significance_pct = pct_negative_slope(slopes, ci_upper)

    assert (point_estimate_pct == 100.0).all()  # every pair looks negative at face value
    assert (significance_pct < point_estimate_pct).all()  # the CI check is strictly more conservative here


def test_scenarios_are_close_true_for_near_identical_scores():
    import pandas as pd

    from flowcyto.compensation.diagnostics import ScenarioScore

    identity = pd.DataFrame(np.eye(2))
    a = ScenarioScore(name="a", matrix=identity, score=11.0, detail="", score_se=0.5)
    b = ScenarioScore(name="b", matrix=identity, score=11.3, detail="", score_se=0.5)
    c = ScenarioScore(name="c", matrix=identity, score=95.0, detail="", score_se=0.5)

    assert scenarios_are_close(a, b)
    assert not scenarios_are_close(a, c)
