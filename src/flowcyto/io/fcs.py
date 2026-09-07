from __future__ import annotations

import hashlib
import io
from pathlib import Path

import flowkit as fk
import numpy as np

from flowcyto.io.types import LoadedSample

SCATTER_TIME_CHANNELS = {"FSC-A", "FSC-H", "SSC-A", "Time"}


def _read_bytes(file_or_path) -> bytes:
    if isinstance(file_or_path, (str, Path)):
        return Path(file_or_path).read_bytes()
    return file_or_path.getvalue()  # Streamlit's UploadedFile / a BytesIO


def load_fcs(file_or_path, filename: str) -> LoadedSample:
    """Load an FCS file (path or an uploaded file-like object) into a
    LoadedSample. Downcasts raw event values to float32 and lets the FlowKit
    Sample object go out of scope immediately after extraction — holding a
    second full-size copy isn't affordable on the free tier's ~1GB budget."""
    data = _read_bytes(file_or_path)
    content_hash = hashlib.md5(data).hexdigest()

    fk_sample = fk.Sample(io.BytesIO(data))
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
