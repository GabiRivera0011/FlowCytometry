from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from flowcyto.io.types import LoadedSample

SCATTER_TIME_CHANNELS = {"FSC-A", "FSC-H", "SSC-A", "Time"}


def _read_bytes(file_or_path) -> bytes:
    if isinstance(file_or_path, (str, Path)):
        return Path(file_or_path).read_bytes()
    return file_or_path.getvalue()


def load_csv(file_or_path, filename: str) -> LoadedSample:
    """Load a CSV event export into a LoadedSample. Columns are used directly
    as channel identities (a CSV's headers already read as marker names, e.g.
    'CD11c APC') — there's no FCS-specific $PnN/$PnS/keyword metadata, so
    pns_labels and fcs_keywords are None. The data may already be compensated
    (this is never assumed either way here)."""
    data = _read_bytes(file_or_path)
    content_hash = hashlib.md5(data).hexdigest()

    df = pd.read_csv(io.BytesIO(data))
    pnn_labels = list(df.columns)
    raw_values = df.to_numpy(dtype=np.float32)
    fluorescence_channels = [name for name in pnn_labels if name not in SCATTER_TIME_CHANNELS]

    return LoadedSample(
        source_format="csv",
        filename=filename,
        content_hash=content_hash,
        pnn_labels=pnn_labels,
        pns_labels=None,
        raw_values=raw_values,
        fluorescence_channels=fluorescence_channels,
        event_count=len(df),
        fcs_keywords=None,
    )
