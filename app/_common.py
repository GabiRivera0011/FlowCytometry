from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


def ensure_src_on_path() -> None:
    """Streamlit fixes sys.path to app/Home.py's directory for every page (not
    the currently-running page's own directory) — src/flowcyto needs this
    explicit bootstrap to import from any app/*.py or app/pages/*.py file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        src_dir = parent / "src"
        if src_dir.is_dir():
            src_str = str(src_dir)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)
            return
    raise RuntimeError("Could not locate src/ from app/_common.py")


def get_workspace():
    # Imported lazily: this module must be importable before src/ is on sys.path.
    from flowcyto.state import Workspace

    if "workspace" not in st.session_state:
        st.session_state["workspace"] = Workspace()
    return st.session_state["workspace"]


def require_sample():
    workspace = get_workspace()
    if workspace.sample is None:
        st.info("Upload a sample file on the Home page first.")
        st.stop()
    return workspace
