import numpy as np

from flowcyto.transforms import arcsinh_transform


def test_arcsinh_transform_handles_zero_and_negative_values():
    values = np.array([-500.0, -10.0, 0.0, 10.0, 500.0, 50_000.0])
    result = arcsinh_transform(values, cofactor=150.0)
    assert np.all(np.isfinite(result))
    # odd function: arcsinh(-x) == -arcsinh(x)
    assert np.isclose(result[0], -arcsinh_transform(np.array([500.0]), cofactor=150.0)[0])


def test_arcsinh_transform_is_monotonically_increasing():
    values = np.linspace(-10_000, 10_000, 500)
    result = arcsinh_transform(values)
    assert np.all(np.diff(result) > 0)


def test_arcsinh_transform_compresses_large_values_relative_to_linear_spacing():
    # two widely-separated bright values should end up much closer together
    # than they started, post-transform — the whole point of the transform.
    lo, hi = arcsinh_transform(np.array([10_000.0, 1_000_000.0]))
    assert (hi - lo) < (1_000_000.0 - 10_000.0) / 1000
