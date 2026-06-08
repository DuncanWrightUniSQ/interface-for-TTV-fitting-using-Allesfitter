"""Streamlit entry point for the TTV fitter."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import pandas as pd
import streamlit as st

from ttv_fitter.fitting import (
    build_cutouts,
    fit_cutout_t0,
    fit_rv_curve,
    fit_single_transit_shape,
    run_emcee_for_transit,
)
from ttv_fitter.io import (
    normalize_photometry,
    normalize_rv,
    normalize_timings,
    output_path,
    parse_allesfitter_params,
    read_table,
)
from ttv_fitter.models import (
    PlanetParams,
    coerce_planet_table,
    derive_a_over_rstar,
    derive_inclination_deg,
    planet_from_row,
)
from ttv_fitter.plots import PLOT_CONFIG, oc_figure, photometry_figure, rv_figure, system_3d_figure
from ttv_fitter.ttv import allesfitter_ttv_rows, fit_linear_ephemeris, fit_sinusoidal_ttv


st.set_page_config(page_title="TTV Fitter", page_icon="TTV", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("photometry", pd.DataFrame())
    st.session_state.setdefault("rv", pd.DataFrame())
    st.session_state.setdefault("timings", pd.DataFrame())
    st.session_state.setdefault("planets", coerce_planet_table(None))
    st.session_state.setdefault("linear_fit", {})
    st.session_state.setdefault("ttv_model_table", pd.DataFrame())


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def data_import_tab() -> None:
    st.subheader("Data Import")
    st.caption("Load photometry, RVs, timing tables, and allesfitter-style parameter files.")
    metric_row(
        [
            ("Photometry points", str(len(st.session_state.photometry))),
            ("RV points", str(len(st.session_state.rv))),
            ("Timing rows", str(len(st.session_state.timings))),
            ("Planets", str(len(st.session_state.planets))),
        ]
    )
    col_a, col_b = st.columns(2)
    with col_a:
        phot_file = st.file_uploader("Photometry CSV/TSV/TXT", type=["csv", "tsv", "txt", "dat"], key="phot_upload")
        if phot_file is not None:
            st.session_state.photometry = normalize_photometry(read_table(phot_file))
        rv_file = st.file_uploader("RV CSV/TSV/TXT", type=["csv", "tsv", "txt", "dat"], key="rv_upload")
        if rv_file is not None:
            st.session_state.rv = normalize_rv(read_table(rv_file))
    with col_b:
        timing_file = st.file_uploader("Transit timings CSV/TSV/TXT", type=["csv", "tsv", "txt", "dat"], key="timing_upload")
        if timing_file is not None:
            st.session_state.timings = normalize_timings(read_table(timing_file))
        params_file = st.file_uploader("allesfitter params.csv", type=["csv"], key="params_upload")
        if params_file is not None:
            imported = parse_allesfitter_params(read_table(params_file))
            if not imported.empty:
                st.session_state.planets = coerce_planet_table(imported)
                st.success("Imported planet parameters from params.csv.")

    preview_a, preview_b = st.columns(2)
    with preview_a:
        st.plotly_chart(
            photometry_figure(st.session_state.photometry, st.session_state.planets),
            use_container_width=True,
            config=PLOT_CONFIG,
            key="import_photometry_plot",
        )
    with preview_b:
        st.plotly_chart(
            rv_figure(st.session_state.rv, st.session_state.planets),
            use_container_width=True,
            config=PLOT_CONFIG,
            key="import_rv_plot",
        )


def planet_editor(prefix: str = "planet") -> pd.DataFrame:
    edited = st.data_editor(
        coerce_planet_table(st.session_state.planets),
        use_container_width=True,
        num_rows="dynamic",
        key=f"{prefix}_editor",
        column_config={
            "color": st.column_config.TextColumn("color", help="Hex color for plotting"),
            "duration_hours": st.column_config.NumberColumn("duration_hours", min_value=0.01),
            "radius_ratio": st.column_config.NumberColumn("radius_ratio", min_value=0.0001, max_value=1.0),
        },
    )
    st.session_state.planets = coerce_planet_table(edited)
    return st.session_state.planets


def linear_fit_tab() -> None:
    st.subheader("Normal Linear Transit/RV Fits")
    st.caption("Use least-squares for fast non-TTV fits, then optionally sample the transit solution with MCMC.")
    planets = planet_editor("linear")
    selected_name = st.selectbox("Planet to fit", planets["name"].tolist(), key="linear_planet_select")
    idx = planets.index[planets["name"] == selected_name][0]
    guess = planet_from_row(planets.loc[idx])
    controls, plots = st.columns([0.28, 0.72])
    with controls:
        if st.button("Run transit LS fit", use_container_width=True, disabled=st.session_state.photometry.empty):
            result = fit_single_transit_shape(st.session_state.photometry, guess)
            st.session_state.linear_fit = result.params
            if result.success:
                for key, value in result.params.items():
                    if key in planets.columns:
                        planets.loc[idx, key] = value
                st.session_state.planets = coerce_planet_table(planets)
                st.success("Transit least-squares fit complete.")
            else:
                st.error(result.message)
        if st.button("Run RV LS fit", use_container_width=True, disabled=st.session_state.rv.empty):
            result = fit_rv_curve(st.session_state.rv, guess)
            if result.success:
                for key, value in result.params.items():
                    if key in planets.columns:
                        planets.loc[idx, key] = value
                st.session_state.planets = coerce_planet_table(planets)
                st.success("RV least-squares fit complete.")
            else:
                st.error(result.message)
        nsteps = st.number_input("MCMC steps", min_value=100, max_value=10000, value=800, step=100)
        if st.button("Sample transit fit with MCMC", use_container_width=True, disabled=not bool(st.session_state.linear_fit)):
            result = run_emcee_for_transit(st.session_state.photometry, st.session_state.linear_fit, int(nsteps))
            if result.success:
                st.session_state.mcmc_samples = result.samples
                st.dataframe(result.samples.describe().T, use_container_width=True)
            else:
                st.error(result.message)
    with plots:
        top, bottom = st.tabs(["Photometry", "RV"])
        with top:
            st.plotly_chart(
                photometry_figure(st.session_state.photometry, st.session_state.planets),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="linear_photometry_plot",
            )
        with bottom:
            st.plotly_chart(
                rv_figure(st.session_state.rv, st.session_state.planets),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="linear_rv_plot",
            )


def per_transit_tab() -> None:
    st.subheader("Per-Transit T0 Fit")
    st.caption("Build transit cutouts from a linear ephemeris and refit only each midpoint.")
    planets = coerce_planet_table(st.session_state.planets)
    selected_name = st.selectbox("Planet", planets["name"].tolist(), key="cutout_planet_select")
    row = planets.loc[planets["name"] == selected_name].iloc[0]
    c1, c2, c3 = st.columns(3)
    t0 = c1.number_input("Linear T0", value=float(row["t0"]), format="%.10f")
    period = c2.number_input("Linear period [days]", value=float(row["period"]), min_value=1e-8, format="%.10f")
    half_width = c3.number_input(
        "Cutout half-width [days]",
        value=max(float(row["duration_hours"]) / 24.0, 0.1),
        min_value=0.001,
        format="%.5f",
    )
    cutouts = build_cutouts(st.session_state.photometry, t0, period, half_width)
    if cutouts.empty:
        st.info("Load photometry to build cutouts.")
        return
    edited = st.data_editor(cutouts, use_container_width=True, key="cutout_editor", column_config={"fit": st.column_config.CheckboxColumn("fit")})
    search_half_width = st.number_input(
        "T0 search half-width [days]",
        value=min(half_width, max(float(row["duration_hours"]) / 24.0, 0.05)),
        min_value=1e-5,
        format="%.5f",
    )
    if st.button("Fit selected cutout midpoints", use_container_width=True):
        results = []
        phot = st.session_state.photometry
        for item in edited.loc[edited["fit"]].to_dict("records"):
            mask = (phot["time"] >= item["start"]) & (phot["time"] <= item["end"])
            result = fit_cutout_t0(
                phot.loc[mask],
                period,
                float(item["expected_tmid"]),
                float(row["radius_ratio"]),
                float(row["duration_hours"]),
                search_half_width,
            )
            if result.success:
                results.append(
                    {
                        "epoch": int(item["epoch"]),
                        "tmid": result.params["tmid"],
                        "tmid_err": result.params["tmid_err"],
                        "expected_tmid": item["expected_tmid"],
                        "points": int(item["points"]),
                    }
                )
        st.session_state.timings = pd.DataFrame(results)
        st.success(f"Fit {len(results)} transit midpoint(s).")
    st.plotly_chart(
        oc_figure(st.session_state.timings, t0, period),
        use_container_width=True,
        config=PLOT_CONFIG,
        key="cutout_oc_plot",
    )
    if not st.session_state.timings.empty:
        st.dataframe(st.session_state.timings, use_container_width=True)
        st.download_button("Download timing table", st.session_state.timings.to_csv(index=False), "ttv_timings.csv", "text/csv")


def ttv_model_tab() -> None:
    st.subheader("TTV Interface")
    st.caption("Edit star and planet parameters, fit a simple O-C sinusoid, and export allesfitter-style TTV rows.")
    cols = st.columns(3)
    mass = cols[0].number_input("Host mass [Msun]", value=1.0, min_value=0.01, format="%.4f")
    radius = cols[1].number_input("Host radius [Rsun]", value=1.0, min_value=0.01, format="%.4f")
    companion = cols[2].text_input("allesfitter companion label", value="b")
    planets = planet_editor("ttv")
    if st.button("Derive a/Rstar and inclination from star/impact", use_container_width=True):
        updated = planets.copy()
        for i, row in updated.iterrows():
            a_rs = derive_a_over_rstar(float(row["period"]), mass, radius)
            updated.loc[i, "a_over_rstar"] = a_rs
            updated.loc[i, "inclination_deg"] = derive_inclination_deg(
                a_rs,
                float(row["impact"]),
                float(row["ecc"]),
                float(row["omega_deg"]),
            )
        st.session_state.planets = coerce_planet_table(updated)
        st.success("Updated geometry columns.")
    if st.session_state.timings.empty:
        st.info("Load a timing table or fit per-transit midpoints first.")
        return
    ephem = fit_linear_ephemeris(st.session_state.timings)
    c1, c2 = st.columns(2)
    t0 = c1.number_input("Reference T0", value=float(ephem.get("t0", planets.iloc[0]["t0"])), format="%.10f")
    period = c2.number_input(
        "Reference period",
        value=float(ephem.get("period", planets.iloc[0]["period"])),
        min_value=1e-8,
        format="%.10f",
    )
    if st.button("Fit sinusoidal TTV model", use_container_width=True):
        params, table = fit_sinusoidal_ttv(st.session_state.timings, t0, period)
        st.session_state.ttv_params = params
        st.session_state.ttv_model_table = table
        st.success("TTV model fit complete.")
    model_table = st.session_state.get("ttv_model_table", pd.DataFrame())
    st.plotly_chart(
        oc_figure(st.session_state.timings, t0, period, model_table),
        use_container_width=True,
        config=PLOT_CONFIG,
        key="ttv_oc_plot",
    )
    if st.session_state.get("ttv_params"):
        st.json(st.session_state.ttv_params)
    rows = allesfitter_ttv_rows(st.session_state.timings, companion=companion)
    st.dataframe(rows, use_container_width=True)
    st.download_button("Download allesfitter TTV parameter rows", rows.to_csv(index=False), "ttv_params_rows.csv", "text/csv")


def model_3d_tab() -> None:
    st.subheader("3D Multi-Planet System Model")
    st.caption("Render the current planet table as a shared transit/RV/orbital parameter set.")
    planets = planet_editor("model3d")
    phase = st.slider("Display orbital phase", 0.0, 1.0, 0.0, 0.005)
    st.plotly_chart(
        system_3d_figure(planets, phase=phase),
        use_container_width=True,
        config=PLOT_CONFIG,
        key="system_3d_plot",
    )
    path = output_path("current_planets.csv")
    if st.button("Save current planet parameters", use_container_width=True):
        planets.to_csv(path, index=False)
        st.success(f"Saved {path}.")


def main() -> None:
    init_state()
    st.title("TTV Fitter")
    st.caption("Transit timing variation fitting, linear transit/RV fits, and 3D multiplanet rendering.")
    tabs = st.tabs(
        [
            "Data Import",
            "Linear Transit/RV Fit",
            "Per-Transit T0 Fit",
            "TTV Model",
            "3D System Model",
        ]
    )
    with tabs[0]:
        data_import_tab()
    with tabs[1]:
        linear_fit_tab()
    with tabs[2]:
        per_transit_tab()
    with tabs[3]:
        ttv_model_tab()
    with tabs[4]:
        model_3d_tab()


if __name__ == "__main__":
    main()
