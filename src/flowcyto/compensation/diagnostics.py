from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import theilslopes


def matrix_fingerprint(matrix: pd.DataFrame) -> str:
    return hashlib.md5(matrix.values.tobytes()).hexdigest()


def _fit_slope_matrix(
    events: np.ndarray, channel_names: list[str], fit_size: int, dim_pct: float,
    sample_size: int, seed: int, want_ci: bool, ci: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Shared fitting loop behind `slope_matrix` and `slope_matrix_with_ci`.
    Robust (Theil-Sen) slope of channel j on channel i, fit only on i's dim
    population — isolates spillover-like structure from real co-expression.
    Sign matters more than magnitude: real autofluorescence keeps slopes away
    from 0 regardless of compensation quality, but over-compensation reliably
    flips them negative. When `want_ci`, also keeps each slope's upper
    confidence bound (`theilslopes` computes this in the same call, at no
    extra cost) — used to test whether a negative slope is statistically
    significant rather than merely numerically negative."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(events), size=min(sample_size, len(events)), replace=False)
    sample = events[idx]
    n = len(channel_names)
    out = np.zeros((n, n))
    upper = np.zeros((n, n)) if want_ci else None
    for i in range(n):
        dim_cutoff = np.percentile(sample[:, i], dim_pct)
        dim_idx = np.flatnonzero(sample[:, i] <= dim_cutoff)
        pick = rng.choice(dim_idx, size=min(fit_size, len(dim_idx)), replace=False)
        x = sample[pick, i]
        for j in range(n):
            if i == j:
                continue
            result = theilslopes(sample[pick, j], x, alpha=ci)
            out[i, j] = result.slope
            if want_ci:
                upper[i, j] = result.high_slope
    slopes_df = pd.DataFrame(out, index=channel_names, columns=channel_names)
    upper_df = pd.DataFrame(upper, index=channel_names, columns=channel_names) if want_ci else None
    return slopes_df, upper_df


def slope_matrix(
    events: np.ndarray, channel_names: list[str], fit_size: int = 1200, dim_pct: float = 20,
    sample_size: int = 50_000, seed: int = 1,
) -> pd.DataFrame:
    """Robust (Theil-Sen) slope of channel j on channel i, fit only on i's dim
    population — isolates spillover-like structure from real co-expression.
    Sign matters more than magnitude: real autofluorescence keeps slopes away
    from 0 regardless of compensation quality, but over-compensation reliably
    flips them negative."""
    slopes, _ = _fit_slope_matrix(events, channel_names, fit_size, dim_pct, sample_size, seed, want_ci=False, ci=0.95)
    return slopes


def slope_matrix_with_ci(
    events: np.ndarray, channel_names: list[str], fit_size: int = 1200, dim_pct: float = 20,
    sample_size: int = 50_000, seed: int = 1, ci: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same as `slope_matrix`, but also returns each pairwise slope's upper
    confidence bound at `ci` confidence (95% by default). Pass the upper-bound
    matrix to `pct_negative_slope` to flag a pair only when it's
    *significantly* negative (upper bound still below 0), not merely
    negative at the point estimate — the point estimate alone is noisy on a
    `fit_size`-event sample and can flip sign from sampling noise alone."""
    return _fit_slope_matrix(events, channel_names, fit_size, dim_pct, sample_size, seed, want_ci=True, ci=ci)


def pct_negative_slope(slopes: pd.DataFrame, ci_upper: pd.DataFrame | None = None) -> np.ndarray:
    """Per target channel, % of incoming slopes (from every other channel)
    that are negative — the over-compensation tell. When `ci_upper` is given
    (from `slope_matrix_with_ci`), a pair counts as negative only when its
    confidence interval's upper bound is also below 0 — i.e. significantly
    negative, not just numerically negative at the point estimate.

    The diagonal is excluded via `np.where(..., np.nan)` rather than
    `np.fill_diagonal` + a `< 0` comparison: `NaN < 0` evaluates to `False`,
    not `NaN`, so a comparison-based approach would silently count the
    diagonal as "not negative" instead of excluding it, diluting every
    channel's score by a factor of (n-1)/n."""
    off_diagonal = ~np.eye(slopes.shape[0], dtype=bool)
    source = ci_upper.values if ci_upper is not None else slopes.values
    is_negative = np.where(off_diagonal, source < 0, np.nan)
    return np.nanmean(is_negative, axis=0) * 100


@dataclass
class SlopeDiagnosticResult:
    slopes: pd.DataFrame
    ci_upper: pd.DataFrame
    pct_negative_per_channel: np.ndarray
    fit_size: int
    for_matrix_fingerprint: str

    @property
    def mean_pct_negative(self) -> float:
        return float(np.mean(self.pct_negative_per_channel))


def run_slope_diagnostic(
    raw_values: np.ndarray, channel_names: list[str], matrix: pd.DataFrame, fit_size: int = 1200,
) -> SlopeDiagnosticResult:
    compensated = raw_values @ np.linalg.inv(matrix.values)
    slopes, ci_upper = slope_matrix_with_ci(compensated, channel_names, fit_size=fit_size)
    return SlopeDiagnosticResult(
        slopes=slopes,
        ci_upper=ci_upper,
        pct_negative_per_channel=pct_negative_slope(slopes, ci_upper),
        fit_size=fit_size,
        for_matrix_fingerprint=matrix_fingerprint(matrix),
    )


@dataclass
class ScenarioScore:
    name: str
    matrix: pd.DataFrame
    score: float  # mean % of pairs with a significantly negative slope after applying this matrix — lower is better
    detail: str
    pct_negative_per_channel: np.ndarray

    @property
    def score_se(self) -> float:
        """Approximate standard error of `score`, in percentage points.
        Each channel's flagged-pair fraction is treated as a binomial
        proportion over its (n_channels - 1) incoming pairs; averaging their
        variances across channels gives a heuristic (not exact — channels
        aren't fully independent) standard error on the overall mean score.
        Used to judge whether two scenarios' scores are meaningfully
        different, or just noise from the random fit subsample."""
        p = self.pct_negative_per_channel / 100
        n_channels = len(p)
        if n_channels == 0:
            return 0.0
        n_pairs = max(n_channels - 1, 1)
        mean_channel_variance = float(np.mean(p * (1 - p) / n_pairs))
        return float(np.sqrt(mean_channel_variance / n_channels) * 100)


def score_scenario(raw_values: np.ndarray, channel_names: list[str], name: str, matrix: pd.DataFrame, detail: str, fit_size: int = 400) -> ScenarioScore:
    """Quick-tier score (fit_size=400, ~2-3s) for comparing candidate matrices
    on page load — the full fit_size=1200 diagnostic is reserved for the
    explicit 'Run full diagnostic' action, not scenario selection."""
    result = run_slope_diagnostic(raw_values, channel_names, matrix, fit_size=fit_size)
    return ScenarioScore(
        name=name, matrix=matrix, score=result.mean_pct_negative, detail=detail,
        pct_negative_per_channel=result.pct_negative_per_channel,
    )


def scenarios_are_close(a: ScenarioScore, b: ScenarioScore, n_se: float = 2.0) -> bool:
    """True if `a` and `b`'s scores are within `n_se` combined standard
    errors of each other — i.e. not meaningfully different given the noise
    of the random fit subsample, even though one is numerically lower."""
    combined_se = float(np.hypot(a.score_se, b.score_se))
    return abs(a.score - b.score) < n_se * combined_se


def pick_default_scenario(scenarios: list[ScenarioScore]) -> ScenarioScore:
    return min(scenarios, key=lambda s: s.score)


def run_validated_search(
    raw_values: np.ndarray, channel_names: list[str], train_size: int = 50_000, val_size: int = 50_000, seed: int = 0,
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Control-free automatic search for the no-controls fallback. Proposes a
    per-pair correction by minimizing % negative events on a training sample,
    but only keeps it if it *also* helps on a separate held-out sample —
    without this gate, the naive version overfits to sampling noise and makes
    the full dataset worse regardless of regularization (verified during
    development; do not remove the train/validation split). Returns the
    resulting matrix and the list of accepted (from, into, value) updates —
    on already-compensated data, expect most proposals to be rejected."""
    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(raw_values))
    train_sample = raw_values[shuffled_idx[:train_size]]
    validation_sample = raw_values[shuffled_idx[train_size : train_size + val_size]]

    n = len(channel_names)
    matrix = np.eye(n)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    rng.shuffle(pairs)

    def pct_negative(events: np.ndarray, m: np.ndarray) -> float:
        compensated = events @ np.linalg.inv(m)
        return (compensated < 0).mean() * 100

    best_validation_score = pct_negative(validation_sample, matrix)
    accepted_updates: list[tuple[str, str, float]] = []

    for i, j in pairs:
        def train_objective(value: float, i: int = i, j: int = j) -> float:
            candidate = matrix.copy()
            candidate[i, j] = value
            return pct_negative(train_sample, candidate)

        proposed = minimize_scalar(train_objective, bounds=(0, 2), method="bounded", options={"xatol": 1e-2})
        candidate_matrix = matrix.copy()
        candidate_matrix[i, j] = proposed.x
        candidate_score = pct_negative(validation_sample, candidate_matrix)

        if candidate_score <= best_validation_score:
            matrix[i, j] = proposed.x
            best_validation_score = candidate_score
            accepted_updates.append((channel_names[i], channel_names[j], float(proposed.x)))

    return pd.DataFrame(matrix, index=channel_names, columns=channel_names), accepted_updates
