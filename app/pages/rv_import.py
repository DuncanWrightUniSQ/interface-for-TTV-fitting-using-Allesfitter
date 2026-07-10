"""RV import and preparation page."""

from __future__ import annotations

import streamlit as st

from app.plots import PLOT_CONFIG, rv_preview
from app.ui import page_header, pending_panel


def render() -> None:
    page_header(
        "Import and Prepare RV Data",
        "Load radial velocity tables, map instruments, inspect offsets, and prepare allesfitter RV inputs.",
        actions=["Validate Columns", "Save RV Set"],
    )

    data_col, control_col = st.columns([0.58, 0.42], vertical_alignment="top")
    with data_col:
        files = st.file_uploader(
            "RV files",
            type=["csv", "txt", "dat", "tsv"],
            accept_multiple_files=True,
            key="rv_file_upload",
        )
        st.session_state["rv_files"] = files or []
        st.plotly_chart(rv_preview(), use_container_width=True, config=PLOT_CONFIG, key="rv_import_preview")

    with control_col:
        st.subheader("Column mapping")
        st.selectbox("Time column", ["Auto detect", "bjd_tdb", "bjd", "time"])
        st.selectbox("RV column", ["Auto detect", "rv", "radial_velocity"])
        st.selectbox("RV uncertainty column", ["Auto detect", "rv_err", "sigma_rv"])
        st.selectbox("Instrument column", ["None", "instrument", "source", "telescope"])
        st.checkbox("Fit independent RV offsets", value=True, key="rv_import_fit_offsets")
        st.checkbox("Fit RV jitter per instrument", value=True, key="rv_import_fit_jitter")
        st.button("Preview RV preparation", use_container_width=True, disabled=True)

    pending_panel(
        "Next wiring",
        [
            "Parse instrument-specific RV tables",
            "Convert units and time standards",
            "Flag outliers and known bad nights",
            "Export allesfitter-compatible RV files",
        ],
    )
