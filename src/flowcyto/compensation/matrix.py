from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flowcyto.io.types import LoadedSample

SCATTER_TIME_CHANNELS = {"FSC-A", "FSC-H", "SSC-A", "Time"}


def channel_identity(sample: LoadedSample, index: int) -> str:
    """Cross-format channel identity: marker name (FCS $PnS) when present,
    else the raw column header — CSV headers already read as marker names,
    so this lets a CSV sample match FCS-control-derived marker identities."""
    if sample.pns_labels and sample.pns_labels[index]:
        return sample.pns_labels[index]
    return sample.pnn_labels[index]


def fluorescence_indices(sample: LoadedSample) -> list[int]:
    return [i for i, name in enumerate(sample.pnn_labels) if name not in SCATTER_TIME_CHANNELS]


@dataclass
class ControlMatch:
    marker_identity: str
    control_sample: LoadedSample
    control_primary_index: int


def discover_controls(control_samples: list[LoadedSample]) -> dict[str, ControlMatch]:
    """Map marker identity -> the control that stains it. A single-stain control
    has exactly one channel with a non-blank marker identity; skip any that don't."""
    matches: dict[str, ControlMatch] = {}
    for control in control_samples:
        primary_index = next(
            (i for i in range(len(control.pnn_labels)) if control.pns_labels and control.pns_labels[i]),
            None,
        )
        if primary_index is None:
            continue
        marker = channel_identity(control, primary_index)
        matches[marker] = ControlMatch(marker, control, primary_index)
    return matches


def compute_spillover_row(
    control: LoadedSample, primary_index: int, target_indices: list[int], pos_pct: float = 90, neg_pct: float = 10
) -> np.ndarray:
    """One spillover-matrix row: fraction of the primary channel's signal
    appearing in each target detector, from a single-stain control."""
    values = control.raw_values
    primary_values = values[:, primary_index]
    positive = primary_values > np.percentile(primary_values, pos_pct)
    negative = primary_values < np.percentile(primary_values, neg_pct)

    primary_baseline = np.median(primary_values[negative])
    primary_bright = np.median(primary_values[positive]) - primary_baseline
    if primary_bright <= 0:
        raise ValueError(f"{control.filename}: no separable positive population")

    row = np.zeros(len(target_indices))
    for row_i, target_index in enumerate(target_indices):
        target_values = values[:, target_index]
        baseline = np.median(target_values[negative])
        bright = np.median(target_values[positive]) - baseline
        row[row_i] = bright / primary_bright
    return row


def build_control_matrix(
    sample: LoadedSample, control_samples: list[LoadedSample]
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Spillover matrix for `sample`'s fluorescence channels from uploaded
    single-stain controls, matched by marker identity (not detector name) so a
    CSV sample can still use FCS controls. Uncontrolled channels default to
    identity. Returns (matrix, uncontrolled_channel_names, control_filenames_by_channel)."""
    fluor_idx = fluorescence_indices(sample)
    channel_names = [sample.pnn_labels[i] for i in fluor_idx]
    identities = [channel_identity(sample, i) for i in fluor_idx]
    name_by_identity = dict(zip(identities, channel_names))

    control_matches = discover_controls(control_samples)

    matrix = pd.DataFrame(np.eye(len(channel_names)), index=channel_names, columns=channel_names)
    uncontrolled: list[str] = []
    control_filenames: dict[str, str] = {}

    for name, identity in zip(channel_names, identities):
        match = control_matches.get(identity)
        if match is None:
            uncontrolled.append(name)
            continue
        control_fluor_idx = fluorescence_indices(match.control_sample)
        try:
            row = compute_spillover_row(match.control_sample, match.control_primary_index, control_fluor_idx)
        except ValueError:
            uncontrolled.append(name)
            continue
        for value, control_index in zip(row, control_fluor_idx):
            target_name = name_by_identity.get(channel_identity(match.control_sample, control_index))
            if target_name is not None:
                matrix.loc[name, target_name] = value
        control_filenames[name] = match.control_sample.filename

    return matrix, uncontrolled, control_filenames


def apply_matrix_pct_negative(raw_values: np.ndarray, matrix: pd.DataFrame) -> np.ndarray:
    """Compensate raw_values with `matrix` (true = observed @ inverse(matrix)) and
    return the per-channel % of resulting events that are negative."""
    compensated = raw_values @ np.linalg.inv(matrix.values)
    return (compensated < 0).mean(axis=0) * 100


@dataclass
class SpillCheckResult:
    spill_present: bool
    is_identity: bool
    apply_compensation_flag: str | None


def read_spill_metadata(fcs_keywords: dict | None) -> SpillCheckResult | None:
    """Cheap Step-0 check: does the file's own $SPILL matrix already claim
    'no correction needed' (an identity matrix)? None for CSV — no FCS keywords."""
    if not fcs_keywords or "spill" not in fcs_keywords:
        return None
    parts = fcs_keywords["spill"].split(",")
    n = int(parts[0])
    values = [float(x) for x in parts[1 + n : 1 + n + n * n]]
    matrix = np.array(values).reshape(n, n)
    return SpillCheckResult(
        spill_present=True,
        is_identity=bool(np.allclose(matrix, np.eye(n))),
        apply_compensation_flag=fcs_keywords.get("apply compensation"),
    )
