from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import theilslopes


def matrix_fingerprint(matrix: pd.DataFrame) -> str:
    return hashlib.md5(matrix.values.tobytes()).hexdigest()


def slope_matrix(
    events: np.ndarray, channel_names: list[str], fit_size: int = 1200, dim_pct: float = 20,
    sample_size: int = 50_000, seed: int = 1,
) -> pd.DataFrame:
    """Robust (Theil-Sen) slope of channel j on channel i, fit only on i's dim
    population — isolates spillover-like structure from real co-expression.
    Sign matters more than magnitude: real autofluorescence keeps slopes away
    from 0 regardless of compensation quality, but over-compensation reliably
    flips them negative."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(events), size=min(sample_size, len(events)), replace=False)
    sample = events[idx]
    n = len(channel_names)
    out = np.zeros((n, n))
    for i in range(n):
        dim_cutoff = np.percentile(sample[:, i], dim_pct)
        dim_idx = np.flatnonzero(sample[:, i] <= dim_cutoff)
        pick = rng.choice(dim_idx, size=min(fit_size, len(dim_idx)), replace=False)
        x = sample[pick, i]
        for j in range(n):
            if i == j:
                continue
            out[i, j] = theilslopes(sample[pick, j], x)[0]
    return pd.DataFrame(out, index=channel_names, columns=channel_names)


def pct_negative_slope(slopes: pd.DataFrame) -> np.ndarray:
    """Per target channel, % of incoming slopes (from every other channel)
    that are negative — the over-compensation tell."""
    values = slopes.values.copy()
    np.fill_diagonal(values, np.nan)
    return np.nanmean(values < 0, axis=0) * 100


@dataclass
class SlopeDiagnosticResult:
    slopes: pd.DataFrame
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
    slopes = slope_matrix(compensated, channel_names, fit_size=fit_size)
    return SlopeDiagnosticResult(
        slopes=slopes,
        pct_negative_per_channel=pct_negative_slope(slopes),
        fit_size=fit_size,
        for_matrix_fingerprint=matrix_fingerprint(matrix),
    )


@dataclass
class ScenarioScore:
    name: str
    matrix: pd.DataFrame
    score: float  # mean % of pairs with a negative slope after applying this matrix — lower is better
    detail: str


def score_scenario(raw_values: np.ndarray, channel_names: list[str], name: str, matrix: pd.DataFrame, detail: str, fit_size: int = 400) -> ScenarioScore:
    """Quick-tier score (fit_size=400, ~2-3s) for comparing candidate matrices
    on page load — the full fit_size=1200 diagnostic is reserved for the
    explicit 'Run full diagnostic' action, not scenario selection."""
    result = run_slope_diagnostic(raw_values, channel_names, matrix, fit_size=fit_size)
    return ScenarioScore(name=name, matrix=matrix, score=result.mean_pct_negative, detail=detail)


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
