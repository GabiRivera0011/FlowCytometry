from pathlib import Path

from flowcyto.io.csv_source import load_csv
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


def test_load_csv_reads_events_and_uses_headers_as_labels():
    sample = load_csv(FIXTURES / "pbmc_sample.csv", "pbmc_sample.csv")
    assert sample.source_format == "csv"
    assert sample.event_count == 2000
    assert sample.raw_values.shape == (2000, len(sample.pnn_labels))
    assert sample.raw_values.dtype.name == "float32"
    assert sample.pns_labels is None
    assert sample.fcs_keywords is None
    assert "CD11c APC" in sample.pnn_labels


def test_csv_metadata_matches_fcs_metadata_for_the_same_data():
    csv_sample = load_csv(FIXTURES / "pbmc_sample.csv", "pbmc_sample.csv")
    csv_metadata = build_channel_metadata(csv_sample)

    marker_row = csv_metadata[csv_metadata["channel"] == "CD11c APC"].iloc[0]
    assert marker_row["category"] == "target-marker"
    assert marker_row["target"] == "CD11c"
    assert marker_row["marker"] == "APC"

    fsc_row = csv_metadata[csv_metadata["channel"] == "FSC-A"].iloc[0]
    assert fsc_row["category"] == "scatter"
