from __future__ import annotations

from pathlib import Path

import flowkit as fk
import numpy as np

from flowcyto.io._util import content_hash as _content_hash
from flowcyto.io.types import LoadedSample

SCATTER_TIME_CHANNELS = {"FSC-A", "FSC-H", "SSC-A", "Time"}


def load_fcs(file_or_path, filename: str) -> LoadedSample:
    """Load an FCS file (path or an uploaded file-like object) into a
    LoadedSample. Downcasts raw event values to float32 and lets the FlowKit
    Sample object go out of scope immediately after extraction — holding a
    second full-size copy isn't affordable on the free tier's ~1GB budget.
    A path is handed to FlowKit directly (it does its own file reading)
    rather than pre-buffering the whole file into a bytes object first."""
    content_hash = _content_hash(file_or_path)

    fk_sample = fk.Sample(str(file_or_path) if isinstance(file_or_path, (str, Path)) else file_or_path)
    pnn_labels = list(fk_sample.pnn_labels)
    pns_labels = list(fk_sample.pns_labels)
    fcs_keywords = dict(fk_sample.get_metadata())
    raw_values = fk_sample.as_dataframe(source="raw").to_numpy(dtype=np.float32)
    event_count = fk_sample.event_count
    del fk_sample

    fluorescence_channels = [name for name in pnn_labels if name not in SCATTER_TIME_CHANNELS]

    return LoadedSample(
        source_format="fcs",
        filename=filename,
        content_hash=content_hash,
        pnn_labels=pnn_labels,
        pns_labels=pns_labels,
        raw_values=raw_values,
        fluorescence_channels=fluorescence_channels,
        event_count=event_count,
        fcs_keywords=fcs_keywords,
    )
