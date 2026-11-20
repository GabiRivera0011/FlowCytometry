from __future__ import annotations

import io
from pathlib import Path

import flowio
import flowkit as fk
import numpy as np

from flowcyto.io._util import content_hash as _content_hash
from flowcyto.io.types import LoadedSample

SCATTER_TIME_CHANNELS = {"FSC-A", "FSC-H", "SSC-A", "Time"}

# A float32 (event_count * channel_count) array above this many cells would
# need >~500MB for that array alone, before FlowKit's own same-sized float64
# original (held briefly alongside it while downcasting) and everything else
# the session already holds (controls, compensation temporaries, other
# subsamples) — well past what fits in Streamlit Community Cloud's ~1GB
# free-tier container. Rejecting up front, from a cheap header/TEXT-segment
# peek, turns what would otherwise be a mid-parse OOM crash of the whole app
# into one clean, recoverable error message.
MAX_FCS_CELLS = 125_000_000


class FCSTooLargeError(Exception):
    """Raised when an FCS file's event_count * channel_count would exceed
    what can be safely parsed into memory on this deployment."""


def _peek_fcs_dimensions(file_or_path) -> tuple[int, int]:
    """Cheaply read an FCS file's event/channel counts from its HEADER and
    TEXT segments only ($TOT/$PAR) — `only_text=True` never touches the
    (potentially huge) DATA segment, so this is fast and uses negligible
    memory regardless of file size.

    A path gets its own file handle, opened and closed entirely inside
    flowio, with no effect on the caller. A file-like upload is instead
    wrapped in a throwaway `BytesIO` copy of its bytes: flowio
    unconditionally closes whatever handle it's given, and an uploaded
    file's own handle must stay open and re-readable for the real parse
    that follows this check."""
    if isinstance(file_or_path, (str, Path)):
        probe = flowio.FlowData(str(file_or_path), only_text=True)
    else:
        probe = flowio.FlowData(io.BytesIO(file_or_path.getvalue()), only_text=True)
    return probe.event_count, probe.channel_count


def load_fcs(file_or_path, filename: str) -> LoadedSample:
    """Load an FCS file (path or an uploaded file-like object) into a
    LoadedSample. Downcasts raw event values to float32 and lets the FlowKit
    Sample object go out of scope immediately after extraction — holding a
    second full-size copy isn't affordable on the free tier's ~1GB budget.
    A path is handed to FlowKit directly (it does its own file reading)
    rather than pre-buffering the whole file into a bytes object first.

    Raises `FCSTooLargeError` before attempting the full parse if the file's
    declared dimensions would be unsafe to hold in memory."""
    event_count_hint, channel_count_hint = _peek_fcs_dimensions(file_or_path)
    if event_count_hint * channel_count_hint > MAX_FCS_CELLS:
        raise FCSTooLargeError(
            f"{filename} declares {event_count_hint:,} events across {channel_count_hint} channels "
            f"({event_count_hint * channel_count_hint:,} cells), which is too large to load safely on this "
            "deployment's memory budget. Try a smaller acquisition, or pre-subsample the file before uploading."
        )

    content_hash = _content_hash(file_or_path)

    fk_sample = fk.Sample(str(file_or_path) if isinstance(file_or_path, (str, Path)) else file_or_path)
    pnn_labels = list(fk_sample.pnn_labels)
    pns_labels = list(fk_sample.pns_labels)
    fcs_keywords = dict(fk_sample.get_metadata())
    # `get_events` returns FlowKit's own internal array directly (by
    # reference, not a copy) in `pnn_labels` column order — going straight
    # here instead of through `as_dataframe().to_numpy()` skips an entire
    # extra full-size pandas DataFrame copy that briefly doubled peak memory
    # during parsing for no benefit.
    raw_values = fk_sample.get_events(source="raw").astype(np.float32)
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
