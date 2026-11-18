import numpy as np

from flowcyto.transforms import estimate_logicle_params, logicle_transform


def test_logicle_transform_handles_zero_and_negative_values():
    values = np.array([-500.0, -10.0, 0.0, 10.0, 500.0, 50_000.0])
    t, w, m, a = estimate_logicle_params(values)
    result = logicle_transform(values, t, w, m, a)
    assert np.all(np.isfinite(result))


def test_logicle_transform_is_monotonically_increasing():
    values = np.linspace(-10_000, 10_000, 500)
    t, w, m, a = estimate_logicle_params(values)
    result = logicle_transform(values, t, w, m, a)
    assert np.all(np.diff(result) > 0)


def test_logicle_transform_maps_top_of_scale_near_one():
    values = np.concatenate([np.linspace(-500, 500, 400), np.full(10, 100_000.0)])
    t, w, m, a = estimate_logicle_params(values)
    result = logicle_transform(np.array([t]), t, w, m, a)
    assert np.isclose(result[0], 1.0, atol=1e-6)


def test_estimate_logicle_params_widens_w_for_larger_negative_spread():
    rng = np.random.default_rng(0)
    narrow = np.concatenate([rng.normal(0, 20, 2000), rng.normal(5000, 1000, 2000)])
    wide = np.concatenate([rng.normal(0, 800, 2000), rng.normal(5000, 1000, 2000)])

    _, w_narrow, _, _ = estimate_logicle_params(narrow)
    _, w_wide, _, _ = estimate_logicle_params(wide)
    assert w_wide > w_narrow


def test_estimate_logicle_params_handles_degenerate_input():
    t, w, m, a = estimate_logicle_params(np.zeros(5))
    assert np.isfinite(t) and np.isfinite(w)
    assert 0 < w < m / 2

    t2, w2, m2, a2 = estimate_logicle_params(np.array([]))
    assert np.isfinite(t2) and np.isfinite(w2)
