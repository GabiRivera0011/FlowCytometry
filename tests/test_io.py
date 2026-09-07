from pathlib import Path

from flowcyto.io.fcs import load_fcs
from flowcyto.io.metadata import build_channel_metadata

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_fcs_reads_events_and_labels():
    sample = load_fcs(FIXTURES / "pbmc_sample.fcs", "pbmc_sample.fcs")
    assert sample.source_format == "fcs"
    assert sample.event_count == 2000
    assert sample.raw_values.shape == (2000, len(sample.pnn_labels))
    assert sample.raw_values.dtype.name == "float32"
    assert "FSC-A" in sample.pnn_labels
    assert "CD11c APC" in sample.pns_labels


def test_channel_metadata_classifies_scatter_time_and_markers():
    sample = load_fcs(FIXTURES / "pbmc_sample.fcs", "pbmc_sample.fcs")
    metadata = build_channel_metadata(sample)

    fsc_row = metadata[metadata["channel"] == "FSC-A"].iloc[0]
    assert fsc_row["category"] == "scatter"

    time_row = metadata[metadata["channel"] == "Time"].iloc[0]
    assert time_row["category"] == "time"

    cd11c_index = sample.pns_labels.index("CD11c APC")
    marker_row = metadata[metadata["channel"] == sample.pnn_labels[cd11c_index]].iloc[0]
    assert marker_row["category"] == "target-marker"
    assert marker_row["target"] == "CD11c"
    assert marker_row["marker"] == "APC"
