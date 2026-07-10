"""Global model fitting page."""

from __future__ import annotations

import streamlit as st

from app.allesfitter_config import render_config_editors, render_context_controls, render_docs_note
from app.plots import PLOT_CONFIG, isochrone_preview, rv_preview, folded_transit_preview
from app.ui import page_header, pending_panel


def render() -> None:
    page_header(
        "Global Photometry, RV, and Isochrone Model",
        "Combine transit photometry, RVs, and stellar context into a joint model when available.",
        actions=["Assemble Model", "Run Global Fit"],
    )

    setup_tab, parameter_tab, output_tab = st.tabs(["Global Setup", "allesfitter Parameters", "Outputs"])

    with setup_tab:
        setup_col, plot_col = st.columns([0.32, 0.68], vertical_alignment="top")
        with setup_col:
            st.subheader("Model ingredients")
            st.checkbox("Include prepared photometry", value=True, key="global_include_phot")
            st.checkbox("Include prepared RVs", value=True, key="global_include_rv")
            st.checkbox("Include stellar priors", value=False, key="global_include_stellar_priors")
            st.checkbox("Include isochrone constraints", value=False, key="global_include_isochrones")
            st.selectbox("Global sampler", ["MCMC", "Nested sampling"])
            st.text_area("Notes", placeholder="Record literature priors, assumptions, and model choices.")

        with plot_col:
            phot_tab, rv_tab, iso_tab = st.tabs(["Transit", "RV", "Isochrone"])
            with phot_tab:
                st.plotly_chart(
                    folded_transit_preview(),
                    use_container_width=True,
                    config=PLOT_CONFIG,
                    key="global_model_transit_preview",
                )
            with rv_tab:
                st.plotly_chart(rv_preview(), use_container_width=True, config=PLOT_CONFIG, key="global_model_rv_preview")
            with iso_tab:
                st.plotly_chart(
                    isochrone_preview(),
                    use_container_width=True,
                    config=PLOT_CONFIG,
                    key="global_model_isochrone_preview",
                )

    with parameter_tab:
        render_docs_note()
        st.caption(
            "The global model combines the photometry and RV parameter sets: shared orbital rows "
            "for the same companion, photometry instrument rows, RV instrument rows, and shared "
            "settings for companions/instruments."
        )
        context = render_context_controls("global_model", include_photometry=True, include_rv=True)
        render_config_editors("global_model", context)

    with output_tab:
        pending_panel(
            "Next wiring",
            [
                "Map prepared datasets into a single allesfitter directory",
                "Represent shared parameters across photometry and RVs",
                "Decide how to connect isochrone constraints",
                "Compare global posteriors against single-dataset fits",
            ],
        )
