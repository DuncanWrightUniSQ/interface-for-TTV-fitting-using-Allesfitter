"""Transit timing variation workflow page."""

from __future__ import annotations

import streamlit as st

from app.allesfitter_config import render_config_editors, render_context_controls, render_docs_note
from app.plots import PLOT_CONFIG, folded_transit_preview, ttv_oc_preview
from app.ui import metric_strip, page_header, pending_panel


def render() -> None:
    page_header(
        "Transit Timing Variations",
        "Measure individual transit times, inspect O-C structure, and prepare TTV fits.",
        actions=["Find Transits", "Run TTV Fit"],
    )

    workflow_tab, parameter_tab, output_tab = st.tabs(["Workflow", "allesfitter Parameters", "Outputs"])

    with workflow_tab:
        metric_strip(
            [
                ("Loaded sectors", "0", ""),
                ("Transit windows", "0", ""),
                ("Timing model", "TTV", ""),
                ("Fit status", "not run", ""),
            ]
        )

        control_col, plot_col = st.columns([0.34, 0.66], vertical_alignment="top")
        with control_col:
            st.subheader("TTV setup")
            st.number_input("Reference epoch T0", value=0.0, format="%.8f")
            st.number_input("Period", value=1.0, min_value=0.0, format="%.8f")
            st.number_input("Transit duration [hours]", value=3.0, min_value=0.0, step=0.25)
            st.selectbox("Timing fit mode", ["Fit each transit independently", "Shared shape with per-transit T0"])
            st.checkbox("Freeze linear epoch/period before TTV run", value=True, key="ttv_freeze_linear_ephemeris")
            st.button("Prepare TTV fit", use_container_width=True, disabled=True)

        with plot_col:
            top, bottom = st.tabs(["O-C Diagram", "Transit Window"])
            with top:
                st.plotly_chart(ttv_oc_preview(), use_container_width=True, config=PLOT_CONFIG, key="ttv_oc_preview")
            with bottom:
                st.plotly_chart(
                    folded_transit_preview(),
                    use_container_width=True,
                    config=PLOT_CONFIG,
                    key="ttv_transit_window_preview",
                )

    with parameter_tab:
        render_docs_note(ttv=True)
        st.info(
            "allesfitter's TTV tutorial recommends first fitting a linear ephemeris, then setting "
            "`fit_ttvs` to `True`, freezing `[companion]_epoch` and `[companion]_period`, and running "
            "`allesfitter.prepare_ttv_fit(...)` to generate per-transit midpoint rows."
        )
        context = render_context_controls("ttv", include_photometry=True, include_rv=False, fit_ttvs=True)
        render_config_editors("ttv", context)

    with output_tab:
        pending_panel(
            "TTV build-out",
            [
                "Call allesfitter.prepare_ttv_fit on a selected fit directory",
                "Append generated per-transit midpoint rows to params.csv",
                "Store timing table with epoch, T0, uncertainty, source sector",
                "Plot O-C residuals and export fit-ready timing data",
            ],
        )
