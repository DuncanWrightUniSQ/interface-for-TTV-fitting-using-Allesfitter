"""RV fitting page."""

from __future__ import annotations

import streamlit as st

from app.allesfitter_config import render_config_editors, render_context_controls, render_docs_note
from app.plots import PLOT_CONFIG, corner_placeholder, rv_preview
from app.ui import page_header, pending_panel


def render() -> None:
    page_header(
        "RV Fit",
        "Configure Keplerian RV models, offsets, jitters, and residual diagnostics.",
        actions=["Write RV Config", "Run RV Fit"],
    )

    fit_tab, parameter_tab, output_tab = st.tabs(["Fit Setup", "allesfitter Parameters", "Outputs"])

    with fit_tab:
        setup_col, plot_col = st.columns([0.34, 0.66], vertical_alignment="top")
        with setup_col:
            st.subheader("Keplerian setup")
            st.number_input("Number of planets", value=1, min_value=1, max_value=8)
            st.checkbox("Share period with photometry ephemeris", value=True, key="rv_fit_share_period")
            st.checkbox("Fit eccentricity using f_c/f_s", value=False, key="rv_fit_eccentricity")
            st.checkbox("Fit RV jitter per instrument", value=True, key="rv_fit_jitter")
            st.selectbox("RV prior style", ["Broad physical", "Literature centered", "Custom"])
            st.button("Build RV model", use_container_width=True, disabled=True)

        with plot_col:
            model_tab, posterior_tab = st.tabs(["RV Model", "Posterior Preview"])
            with model_tab:
                st.plotly_chart(rv_preview(), use_container_width=True, config=PLOT_CONFIG, key="rv_fit_model_preview")
            with posterior_tab:
                st.plotly_chart(
                    corner_placeholder(), use_container_width=True, config=PLOT_CONFIG, key="rv_fit_posterior_preview"
                )

    with parameter_tab:
        render_docs_note()
        st.caption(
            "RV fits add `[companion]_K`, `ln_jitter_rv_[inst]`, and baseline rows whose names "
            "must match `baseline_rv_[inst]` in settings.csv. Eccentricity is represented by "
            "`[companion]_f_c` and `[companion]_f_s`."
        )
        context = render_context_controls("rv_fit", include_photometry=False, include_rv=True)
        render_config_editors("rv_fit", context)

    with output_tab:
        pending_panel(
            "Next wiring",
            [
                "Create instrument offset and jitter parameters",
                "Run RV-only allesfitter models",
                "Plot phase-folded RVs and residuals",
                "Compare circular and eccentric solutions",
            ],
        )
