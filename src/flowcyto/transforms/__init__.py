from __future__ import annotations

import numpy as np

# A reasonable default for compensated flow cytometry fluorescence data
# (comparable to common Cytobank/CyTOF defaults). Scatter/time channels
# should never be passed through this transform.
DEFAULT_COFACTOR = 150.0


def arcsinh_transform(values: np.ndarray, cofactor: float = DEFAULT_COFACTOR) -> np.ndarray:
    """Standard flow cytometry display transform for fluorescence channels:
    arcsinh(x / cofactor). Unlike a plain log, this handles zero and negative
    values natively — compensated fluorescence routinely has both — while
    behaving like a log transform well above the cofactor and like a linear
    one well below it. This spreads out the dim/negative population (which a
    linear axis compresses into a sliver near zero) without the axis
    breaking down at or below zero the way a plain log does."""
    return np.arcsinh(np.asarray(values, dtype=np.float64) / cofactor)
