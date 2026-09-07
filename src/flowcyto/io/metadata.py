from __future__ import annotations

import pandas as pd

from flowcyto.io.types import LoadedSample

SCATTER_TIME_CATEGORY = {"FSC-A": "scatter", "FSC-H": "scatter", "SSC-A": "scatter", "Time": "time"}
TARGET_MARKER_OVERRIDES = {
    "Live Dead UV Blue": ("Live Dead", "UV Blue"),  # viability dye, not a CD antigen
}


def classify_channel(detector_name: str, display_label: str | None) -> dict:
    """Classify one channel: scatter/time (by detector name), empty (no
    display label), or target+marker (split "<target> <marker>"). One shared
    classifier for both formats — FCS feeds it $PnN/$PnS, CSV feeds it its
    column header as both (CSV headers already read as marker names)."""
    if detector_name in SCATTER_TIME_CATEGORY:
        return {"category": SCATTER_TIME_CATEGORY[detector_name], "target": None, "marker": None}
    if not display_label or display_label.startswith("Unnamed"):
        return {"category": "empty", "target": None, "marker": None}
    if display_label in TARGET_MARKER_OVERRIDES:
        target, marker = TARGET_MARKER_OVERRIDES[display_label]
    elif " " in display_label:
        target, marker = display_label.split(" ", 1)
    else:
        target, marker = display_label, None
    return {"category": "target-marker", "target": target, "marker": marker}


def build_channel_metadata(sample: LoadedSample) -> pd.DataFrame:
    rows = []
    for i, detector_name in enumerate(sample.pnn_labels):
        display_label = sample.pns_labels[i] if sample.pns_labels else detector_name
        rows.append({"channel": detector_name, **classify_channel(detector_name, display_label)})
    return pd.DataFrame(rows)
