from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from flowcyto.compensation.diagnostics import ScenarioScore, SlopeDiagnosticResult
from flowcyto.io.types import LoadedSample


@dataclass
class Workspace:
    """Single source of app state, held in st.session_state. No Streamlit
    import here — this stays framework-agnostic per docs/ARCHITECTURE.md so
    it's usable/testable outside the app."""

    sample: LoadedSample | None = None
    control_samples: list[LoadedSample] = field(default_factory=list)
    metadata: pd.DataFrame | None = None
    spillover_matrix: pd.DataFrame | None = None
    default_scenario: ScenarioScore | None = None
    uncontrolled_channels: list[str] = field(default_factory=list)
    controls_available: bool = False
    control_filenames: dict[str, str] = field(default_factory=dict)
    last_diagnostic: SlopeDiagnosticResult | None = None
    plot_subsample_indices: np.ndarray | None = None

    # Final gated population handed from one gating page to the next —
    # boolean mask over `plot_subsample_indices`, not the full event array.
    live_cells_mask: np.ndarray | None = None
    lymphocytes_mask: np.ndarray | None = None
    myeloid_mask: np.ndarray | None = None

    # stubs for later phases, per docs/ARCHITECTURE.md's full stated Workspace scope
    gating_strategy: Any | None = None
    transforms: dict | None = None
    cluster_result: Any | None = None
