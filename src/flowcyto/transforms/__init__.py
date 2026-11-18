from __future__ import annotations

import flowkit
import numpy as np

# flowkit exposes `transforms` as an attribute of the top-level package
# (`flowkit/__init__.py` does `from ._models import transforms`), not as a
# separately importable `flowkit.transforms` submodule — hence the
# attribute access below rather than `from flowkit.transforms import ...`.
LogicleTransform = flowkit.transforms.LogicleTransform

# Standard defaults from the Logicle literature (Parks, Moore & Herzenberg
# 2006) and widely used as-is in flow software: ~4.5 decades of display
# range, no extra negative decades beyond what `param_w` already covers.
DEFAULT_M = 4.5
DEFAULT_A = 0.0


def estimate_logicle_params(
    values: np.ndarray, m: float = DEFAULT_M, a: float = DEFAULT_A,
) -> tuple[float, float, float, float]:
    """Data-driven per-channel Logicle parameters (T, W, M, A), instead of
    one fixed cofactor shared by every channel.

    T (top of scale) is a high percentile of the channel's own values
    (99.9th, not the literal max — consistent with this app's existing
    percentile-based axis autoscaling, and more robust to a single extreme
    outlier than `max()`).

    W (the approximate width of the linear region, in decades) follows the
    relationship in the Logicle paper: W = (M - log10(T / |r|)) / 2, where r
    is a robust estimate of the negative population's spread (the 5th
    percentile of the channel's negative values, when there are enough of
    them to estimate from). A channel with little/no negative signal gets a
    narrow linear region (closer to a pure log); a channel with wide
    compensation-driven negative spread gets a wider one. W is clamped to
    (0, M/2) — outside that range the Logicle transform is undefined/degenerate.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0, 0.5, m, a

    t = max(float(np.percentile(finite, 99.9)), 1.0)

    negative = finite[finite < 0]
    if negative.size >= 10:
        r = float(np.percentile(negative, 5))
    else:
        r = -t / 100.0  # no reliable negative spread to estimate from — a small nominal one
    if r == 0:
        r = -t / 100.0

    w = (m - np.log10(t / abs(r))) / 2.0
    w = float(np.clip(w, 0.1, m / 2.0 - 0.05))
    return t, w, m, a


def logicle_transform(values: np.ndarray, t: float, w: float, m: float = DEFAULT_M, a: float = DEFAULT_A) -> np.ndarray:
    """Apply the Logicle biexponential display transform (Parks, Moore &
    Herzenberg 2006; GatingML 2.0 spec) via FlowKit's implementation, with
    the given (T, W, M, A) parameters. Unlike a plain log, Logicle handles
    zero and negative values natively — compensated fluorescence routinely
    has both — by blending a linear region of width `w` (in decades) around
    zero into a logarithmic one further out, rather than breaking down at or
    below zero. `t`/`w` should come from `estimate_logicle_params`, computed
    once per channel and reused everywhere that channel is transformed (see
    `app/_gating_widgets.py:transform_channel` for why that matters)."""
    transform = LogicleTransform(param_t=t, param_w=w, param_m=m, param_a=a)
    return np.asarray(transform.apply(np.asarray(values, dtype=np.float64)))
