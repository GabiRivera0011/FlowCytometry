from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class LoadedSample:
    """One loaded sample's raw event data plus its channel metadata.

    `raw_values` columns are aligned to `pnn_labels` (detector names). For a
    CSV source, `pnn_labels` holds the column headers directly and
    `pns_labels`/`fcs_keywords` are None — there's no FCS-specific metadata.
    """

    source_format: Literal["fcs", "csv"]
    filename: str
    content_hash: str
    pnn_labels: list[str]
    pns_labels: list[str] | None
    raw_values: np.ndarray  # float32, shape (n_events, n_channels)
    fluorescence_channels: list[str]
    event_count: int
    fcs_keywords: dict | None
