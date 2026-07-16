"""Streamlit entry point for the TTV fitter."""

from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import numpy as np
import pandas as pd
import streamlit as st

from ttv_fitter.dynamics import compare_model_to_timings, rebound_available, run_physical_ttv_model
from ttv_fitter.alles_workflows import (
    _clean_fit_sector_table,
    _default_sector_directory,
    _sector_tables_from_directory,
    render_import_workflows,
    render_linear_transit_workflow,
    import_allesfitter_pages,
)
from ttv_fitter.batch_workflow import parse_target_list, run_target_batch
from ttv_fitter.fitting import (
    build_cutouts,
    fit_cutout_t0,
    fit_rv_curve,
    fit_single_transit_shape,
    run_cutout_t0_mcmc,
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
from ttv_fitter.plots import (
    PLOT_CONFIG,
    cutout_fit_figure,
    oc_figure,
    photometry_figure,
    rv_figure,
    system_3d_figure,
    t0_histogram_figure,
)
from ttv_fitter.plots import physical_oc_figure, physical_rv_figure, physical_transit_figure, rebound_3d_figure
from ttv_fitter.ttv import allesfitter_ttv_rows, fit_linear_ephemeris, fit_sinusoidal_ttv, oc_table


st.set_page_config(page_title="TTV Fitter", page_icon="TTV", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("photometry", pd.DataFrame())
    st.session_state.setdefault("rv", pd.DataFrame())
    st.session_state.setdefault("timings", pd.DataFrame())
    st.session_state.setdefault("planets", coerce_planet_table(None))
    st.session_state.setdefault("ttv_host_mass", 1.0)
    st.session_state.setdefault("ttv_host_radius", 1.0)
    st.session_state.setdefault("nontransiting_planets", pd.DataFrame())
    st.session_state.setdefault("linear_fit", {})
    st.session_state.setdefault("ttv_model_table", pd.DataFrame())
    st.session_state.setdefault("physical_ttv_result", None)
    st.session_state.setdefault("physical_ttv_comparison", pd.DataFrame())


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def data_import_tab() -> None:
    render_import_workflows()


def planet_editor(prefix: str = "planet") -> pd.DataFrame:
    edited = st.data_editor(
        coerce_planet_table(st.session_state.planets),
        use_container_width=True,
        num_rows="dynamic",
        key=f"{prefix}_editor",
        column_config={
            "name": st.column_config.TextColumn("name", help="Planet label used throughout the app, usually b, c, d..."),
            "color": st.column_config.TextColumn("color", help="Hex color for plotting this planet in 2D and 3D views."),
            "mass_jupiter": st.column_config.NumberColumn(
                "mass_jupiter",
                min_value=0.0,
                help="Planet mass in Jupiter masses. REBOUND uses this to compute gravitational interactions, TTVs, and the host-star RV reflex motion.",
            ),
            "period": st.column_config.NumberColumn(
                "period",
                min_value=1e-8,
                help="Linear orbital period in days. Used as the initial Keplerian orbit and as the comparison ephemeris for O-C/TTV plots.",
            ),
            "t0": st.column_config.NumberColumn(
                "t0",
                help="Reference mid-transit time in the same time system as the data. With 'Initialize phases from T0', REBOUND phases are chosen so this is near inferior conjunction.",
            ),
            "radius_ratio": st.column_config.NumberColumn(
                "radius_ratio",
                min_value=0.0001,
                max_value=1.0,
                help="Rp/Rstar. Sets the approximate transit depth in the lightweight transit overlays and broadens the transit-crossing acceptance window.",
            ),
            "impact": st.column_config.NumberColumn(
                "impact",
                help="Projected star-planet separation at transit in stellar radii. Used with a/Rstar to derive inclination for simple geometry.",
            ),
            "duration_hours": st.column_config.NumberColumn(
                "duration_hours",
                min_value=0.01,
                help="Approximate total transit duration. Used for cutout widths and the simple trapezoid transit overlays.",
            ),
            "a_over_rstar": st.column_config.NumberColumn(
                "a_over_rstar",
                min_value=1e-8,
                help="Semi-major axis divided by stellar radius. Can be derived from period, stellar mass, and stellar radius.",
            ),
            "inclination_deg": st.column_config.NumberColumn(
                "inclination_deg",
                min_value=0.0,
                max_value=180.0,
                help="Orbital inclination in degrees. Near 90 degrees means edge-on and likely transiting.",
            ),
            "ecc": st.column_config.NumberColumn(
                "ecc",
                min_value=0.0,
                max_value=0.95,
                help="Orbital eccentricity. REBOUND integrates eccentric orbits directly; very high values may require smaller sample steps.",
            ),
            "omega_deg": st.column_config.NumberColumn(
                "omega_deg",
                min_value=0.0,
                max_value=360.0,
                help="Argument of periastron in degrees. Changes where periastron lies relative to the transit line of sight.",
            ),
            "mean_anomaly_deg": st.column_config.NumberColumn(
                "mean_anomaly_deg",
                min_value=0.0,
                max_value=360.0,
                help="Initial mean anomaly at the REBOUND reference time. Ignored/recomputed when 'Initialize phases from T0' is enabled.",
            ),
            "rv_k": st.column_config.NumberColumn(
                "rv_k",
                help="Keplerian RV semi-amplitude in m/s for the lightweight non-integrated RV preview. Physical REBOUND RV uses mass and orbit instead.",
            ),
        },
    )
    st.session_state.planets = coerce_planet_table(edited)
    return st.session_state.planets


def linear_fit_tab() -> None:
    render_linear_transit_workflow()
    _render_streamlined_ttv_timing_section()


def batch_workflow_tab() -> None:
    """Visible entry point for the automatic multi-target simplified workflow."""
    st.subheader("Automatic multi-target TTV workflow")
    st.caption(
        "Upload one target per line. Each target is queried at all available TESS cadences, "
        "prepared, fitted, and saved before the next target starts."
    )
    photometry_import, _rv_import, photometry_fit = import_allesfitter_pages()
    uploaded = st.file_uploader(
        "Upload target list",
        # Do not rely on the browser's extension/MIME filter: macOS and synced
        # folders can report ordinary .txt files with a non-text MIME type.
        type=None,
        accept_multiple_files=False,
        key="batch_target_list_upload",
        help="One target name or TIC ID per line. Blank lines and lines beginning with # are ignored.",
    )
    if uploaded is not None:
        raw = uploaded.getvalue()
        text = None
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            st.error("Could not decode the target list as plain text (UTF-8, UTF-16, or Windows-1252).")
            return
        targets = parse_target_list(text)
        st.session_state["batch_target_list_text"] = text
        st.session_state["batch_targets"] = targets
        if targets:
            st.success(f"Loaded {len(targets)} target(s) from {uploaded.name}.")
        else:
            st.warning("The uploaded target list contains no usable target lines.")

    targets = st.session_state.get("batch_targets", [])
    if not targets:
        st.info("Upload a UTF-8 text file containing one star name or TIC ID per line to begin.")
        return
    st.caption("Diamante joined products are excluded automatically; all other MAST products are downloaded and used when readable.")
    if st.button("Run automatic workflow for all targets", type="primary", use_container_width=True, key="run_batch_workflow"):
        summaries: list[dict[str, object]] = []
        st.session_state["batch_results"] = []
        overall = st.progress(0.0)
        for index, target in enumerate(targets, start=1):
            status = st.status(f"{index}/{len(targets)} — {target}: starting", expanded=True)

            def report_step(message: str, *, status=status, index=index, target=target) -> None:
                status.write(message)
                status.update(label=f"{index}/{len(targets)} — {target}: {message}", state="running")

            try:
                result = run_target_batch(
                    target,
                    photometry_import,
                    photometry_fit,
                    progress=report_step,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch processing moving per target
                result = {"target": target, "status": "error", "error": str(exc)}
                status.update(label=f"{index}/{len(targets)} — {target}: failed", state="error")
            else:
                result["status"] = "complete"
                status.update(label=f"{index}/{len(targets)} — {target}: complete", state="complete")
            summaries.append(result)
            st.session_state["batch_results"] = summaries.copy()
            overall.progress(index / len(targets))
        st.success(f"Automatic workflow finished for {len(targets)} target(s).")

    results = st.session_state.get("batch_results", [])
    if results:
        st.subheader("Workflow summary")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


def _fit_param_labels(params: pd.DataFrame | None) -> list[str]:
    if params is None or params.empty or "fit" not in params or "name" not in params:
        return []
    fit_mask = params["fit"].astype(str).str.lower().isin(["1", "true", "yes"])
    return params.loc[fit_mask, "name"].astype(str).tolist()


def _float_param_values(params: pd.DataFrame | None) -> dict[str, float]:
    values: dict[str, float] = {}
    if params is None or params.empty or "name" not in params or "value" not in params:
        return values
    for _, row in params.iterrows():
        try:
            values[str(row["name"])] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return values


def _instrument_suffix(values: dict[str, float]) -> str:
    for name in values:
        if name.startswith("baseline_offset_flux_"):
            return name.removeprefix("baseline_offset_flux_")
    for name in values:
        if name.startswith("host_ldc_u1_"):
            return name.removeprefix("host_ldc_u1_")
    return "TESS"


def _linear_fit_parameter_set(planet_name: str, source_kind: str) -> tuple[dict[str, float], str]:
    if source_kind == "single_ls":
        all_fits = st.session_state.get("phot_fit_single_transit_ls_results_by_planet", {})
        fit = all_fits.get(str(planet_name)) if isinstance(all_fits, dict) else None
        if fit is None:
            fit = st.session_state.get("phot_fit_single_transit_ls_result")
        if not isinstance(fit, dict) or not fit:
            return {}, ""
        if str(fit.get("planet", planet_name)) != str(planet_name):
            return {}, ""
        selected = {
            "t0": fit.get("t0", np.nan),
            "period": fit.get("period", np.nan),
            "radius_ratio": fit.get("radius_ratio", np.nan),
            "impact": fit.get("impact", np.nan),
            "a_over_rstar": fit.get("a_over_rstar", np.nan),
            "duration_hours": fit.get("duration_hours", np.nan),
            "limb_darkening_u1": fit.get("limb_darkening_u1", np.nan),
            "limb_darkening_u2": fit.get("limb_darkening_u2", np.nan),
            "baseline_offset": fit.get("baseline_offset", 0.0),
        }
        return selected, "latest single-transit least-squares fit from the TTV fitting tab"

    params = st.session_state.get("phot_fit_generated_params")
    fit_dir = st.session_state.get("phot_fit_directory", "")
    param_table = params if isinstance(params, pd.DataFrame) else None
    labels = _fit_param_labels(param_table)
    values = _float_param_values(param_table)
    source = "current generated TTV fitting parameter table"

    if source_kind == "mcmc":
        backend_path = Path(fit_dir) / "results" / "mcmc_save.h5" if fit_dir else None
        if backend_path is None or not backend_path.exists() or not labels:
            return {}, ""
        try:
            import emcee

            reader = emcee.backends.HDFBackend(str(backend_path), read_only=True)
            burn = min(int(st.session_state.get("phot_fit_mcmc_burn_steps", 0)), max(int(reader.iteration) - 1, 0))
            samples = reader.get_chain(discard=burn, flat=True)
            if samples.size and samples.shape[1] == len(labels):
                medians = np.nanmedian(samples, axis=0)
                values.update({label: float(value) for label, value in zip(labels, medians)})
                source = "posterior median from the latest TTV fitting MCMC chain"
            else:
                return {}, ""
        except Exception:
            return {}, ""
    elif not values:
        return {}, ""

    instrument = _instrument_suffix(values)
    rr = values.get(f"{planet_name}_rr", values.get(f"{planet_name}_radius_ratio"))
    rsuma = values.get(f"{planet_name}_rsuma", np.nan)
    cosi = values.get(f"{planet_name}_cosi", np.nan)
    a_over_rstar = values.get(f"{planet_name}_a", values.get(f"{planet_name}_a_over_rstar", np.nan))
    if rr is not None and np.isfinite(rsuma) and rsuma > 0:
        a_over_rstar = (1.0 + float(rr)) / float(rsuma)
    impact = values.get(f"{planet_name}_impact", np.nan)
    if not np.isfinite(impact) and np.isfinite(a_over_rstar) and np.isfinite(cosi):
        impact = float(a_over_rstar) * float(cosi)

    selected = {
        "t0": values.get(f"{planet_name}_epoch", np.nan),
        "period": values.get(f"{planet_name}_period", np.nan),
        "radius_ratio": rr if rr is not None else np.nan,
        "impact": impact,
        "a_over_rstar": a_over_rstar,
        "limb_darkening_u1": values.get(f"host_ldc_u1_{instrument}", np.nan),
        "limb_darkening_u2": values.get(f"host_ldc_u2_{instrument}", np.nan),
        "baseline_offset": values.get(f"baseline_offset_flux_{instrument}", 0.0),
    }
    return selected, source


def _latest_timing_table_for_ttv() -> tuple[pd.DataFrame, str]:
    mcmc = st.session_state.get("cutout_mcmc_timings", pd.DataFrame())
    if isinstance(mcmc, pd.DataFrame) and not mcmc.empty and {"epoch", "tmid"}.issubset(mcmc.columns):
        return mcmc.copy(), "latest per-transit MCMC T0 fits"
    timings = st.session_state.get("timings", pd.DataFrame())
    if isinstance(timings, pd.DataFrame) and not timings.empty:
        return timings.copy(), "loaded/manual timing table"
    return pd.DataFrame(), ""


def _stellar_values_from_linear_fit(default_mass: float = 1.0, default_radius: float = 1.0) -> tuple[float, float]:
    tables = [
        st.session_state.get("phot_fit_latest_edited_priors"),
        st.session_state.get("phot_fit_planet_priors_data"),
    ]
    mass = float(default_mass)
    radius = float(default_radius)
    for table in tables:
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        required = {"planet", "parameter", "value"}
        if not required.issubset(table.columns):
            continue
        host = table.loc[table["planet"].astype(str) == "host"]
        for parameter in ["stellar_mass", "stellar_radius"]:
            match = host.loc[host["parameter"].astype(str) == parameter] if not host.empty else pd.DataFrame()
            if match.empty:
                continue
            value = pd.to_numeric(pd.Series([match.iloc[0].get("value")]), errors="coerce").iloc[0]
            if np.isfinite(value) and value > 0:
                if parameter == "stellar_mass":
                    mass = float(value)
                else:
                    radius = float(value)
        if np.isfinite(mass) and np.isfinite(radius):
            break
    return max(float(mass), 0.01), max(float(radius), 0.01)


def _planet_labels_from_linear_fit(values: dict[str, float]) -> list[str]:
    labels: list[str] = []
    for name in values:
        if name.endswith("_period"):
            label = name[: -len("_period")]
            if label and label not in labels and not label.startswith("baseline_offset_flux"):
                labels.append(label)
    return labels


def _current_planet_row_by_name(planets: pd.DataFrame, name: str) -> dict[str, object]:
    table = coerce_planet_table(planets)
    match = table.loc[table["name"].astype(str) == str(name)]
    if not match.empty:
        return match.iloc[0].to_dict()
    return PlanetParams(name=str(name)).to_row()


def _linear_fit_planet_table_from_ls() -> tuple[pd.DataFrame, float, float]:
    params = st.session_state.get("phot_fit_generated_params")
    values = _float_param_values(params if isinstance(params, pd.DataFrame) else None)
    if not values:
        return pd.DataFrame(), np.nan, np.nan

    mass, radius = _stellar_values_from_linear_fit(
        float(st.session_state.get("ttv_host_mass", 1.0)),
        float(st.session_state.get("ttv_host_radius", 1.0)),
    )
    current = coerce_planet_table(st.session_state.get("planets"))
    rows = []
    labels = _planet_labels_from_linear_fit(values)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2"]
    for index, label in enumerate(labels):
        existing = _current_planet_row_by_name(current, label)
        rr = float(pd.to_numeric(pd.Series([values.get(f"{label}_rr", values.get(f"{label}_radius_ratio", existing.get("radius_ratio", 0.08)))]), errors="coerce").iloc[0])
        if not np.isfinite(rr) or rr <= 0:
            rr = float(existing.get("radius_ratio", 0.08))
        epoch = float(pd.to_numeric(pd.Series([values.get(f"{label}_epoch", existing.get("t0", 0.0))]), errors="coerce").iloc[0])
        if not np.isfinite(epoch):
            epoch = float(existing.get("t0", 0.0))
        period = float(pd.to_numeric(pd.Series([values.get(f"{label}_period", existing.get("period", 3.0))]), errors="coerce").iloc[0])
        if not np.isfinite(period) or period <= 0:
            period = float(existing.get("period", 3.0))
        rsuma = float(pd.to_numeric(pd.Series([values.get(f"{label}_rsuma", np.nan)]), errors="coerce").iloc[0])
        cosi = float(pd.to_numeric(pd.Series([values.get(f"{label}_cosi", np.nan)]), errors="coerce").iloc[0])
        a_over_rstar = float(
            pd.to_numeric(
                pd.Series([values.get(f"{label}_a", values.get(f"{label}_a_over_rstar", np.nan))]),
                errors="coerce",
            ).iloc[0]
        )
        if np.isfinite(rsuma) and float(rsuma) > 0:
            a_over_rstar = (1.0 + rr) / float(rsuma)
        if not np.isfinite(a_over_rstar) or float(a_over_rstar) <= 0:
            a_over_rstar = derive_a_over_rstar(period, mass, radius)

        impact = float(pd.to_numeric(pd.Series([values.get(f"{label}_impact", np.nan)]), errors="coerce").iloc[0])
        if not np.isfinite(impact) and np.isfinite(a_over_rstar) and np.isfinite(cosi):
            impact = float(a_over_rstar) * float(cosi)
        if not np.isfinite(impact):
            impact = existing.get("impact", 0.4)

        inclination = np.nan
        if np.isfinite(a_over_rstar) and float(a_over_rstar) > 0 and np.isfinite(impact):
            inclination = derive_inclination_deg(
                float(a_over_rstar),
                float(impact),
                float(existing.get("ecc", 0.0)),
                float(existing.get("omega_deg", 90.0)),
            )
        if not np.isfinite(inclination):
            inclination = existing.get("inclination_deg", 87.0)

        row = dict(existing)
        row.update(
            {
                "name": str(label),
                "period": period,
                "t0": _align_t0_to_photometry_time(epoch, st.session_state.get("photometry", pd.DataFrame())),
                "radius_ratio": rr,
                "impact": float(impact),
                "a_over_rstar": float(a_over_rstar),
                "inclination_deg": float(inclination),
                "color": existing.get("color") or colors[index % len(colors)],
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(), mass, radius
    return coerce_planet_table(pd.DataFrame(rows)), mass, radius


def _nontransiting_planet_editor(base_planets: pd.DataFrame, star_mass: float, star_radius: float) -> pd.DataFrame:
    seed = st.session_state.get("nontransiting_planets")
    if not isinstance(seed, pd.DataFrame) or seed.empty:
        base_period = float(base_planets["period"].iloc[0]) if not base_planets.empty else 3.0
        base_t0 = float(base_planets["t0"].iloc[0]) if not base_planets.empty else 0.0
        a_rs = derive_a_over_rstar(base_period * 1.5, star_mass, star_radius)
        seed = pd.DataFrame(
            [
                {
                    **PlanetParams(name="x", mass_jupiter=0.01, period=base_period * 1.5, t0=base_t0).to_row(),
                    "radius_ratio": 0.001,
                    "impact": 5.0,
                    "a_over_rstar": a_rs,
                    "inclination_deg": 80.0,
                    "rv_k": 0.0,
                    "color": "#9333ea",
                }
            ]
        )
    edited = st.data_editor(
        coerce_planet_table(seed),
        use_container_width=True,
        num_rows="dynamic",
        key="nontransiting_planet_editor",
        column_config={
            "name": st.column_config.TextColumn("name"),
            "mass_jupiter": st.column_config.NumberColumn("mass_jupiter", min_value=0.0),
            "period": st.column_config.NumberColumn("period", min_value=1e-8),
            "t0": st.column_config.NumberColumn("t0"),
            "radius_ratio": st.column_config.NumberColumn("radius_ratio", min_value=0.0, max_value=1.0),
            "impact": st.column_config.NumberColumn("impact"),
            "a_over_rstar": st.column_config.NumberColumn("a_over_rstar", min_value=1e-8),
            "inclination_deg": st.column_config.NumberColumn("inclination_deg", min_value=0.0, max_value=180.0),
            "ecc": st.column_config.NumberColumn("ecc", min_value=0.0, max_value=0.95),
            "omega_deg": st.column_config.NumberColumn("omega_deg", min_value=0.0, max_value=360.0),
            "mean_anomaly_deg": st.column_config.NumberColumn("mean_anomaly_deg", min_value=0.0, max_value=360.0),
            "color": st.column_config.TextColumn("color"),
        },
    )
    table = coerce_planet_table(edited)
    st.session_state.nontransiting_planets = table
    return table


def _seed_nontransiting_planet_from_sinusoid(base_planets: pd.DataFrame, star_mass: float, star_radius: float) -> pd.DataFrame:
    params = st.session_state.get("ttv_params", {}) or {}
    base = coerce_planet_table(base_planets).iloc[0].to_dict()
    base_period = float(base["period"])
    super_epochs = float(params.get("super_period_epochs", 2.0))
    period_guess = max(abs(super_epochs * base_period), base_period * 1.05)
    signed_amplitude = float(params.get("amplitude_minutes", 1.0))
    amplitude = abs(signed_amplitude)
    mass_guess = max(min(amplitude / 50.0, 13.0), 0.001)
    phase = float(params.get("phase_rad", 0.0))
    target_arg = np.pi / 2 if signed_amplitude >= 0 else -np.pi / 2
    epoch_offset = ((target_arg - phase) / (2 * np.pi) * max(abs(super_epochs), 1e-6)) % max(abs(super_epochs), 1e-6)
    t0_guess = float(base["t0"]) + epoch_offset * base_period
    a_rs = derive_a_over_rstar(period_guess, star_mass, star_radius)
    row = {
        **PlanetParams(name="x", mass_jupiter=mass_guess, period=period_guess, t0=t0_guess).to_row(),
        "radius_ratio": 0.001,
        "impact": 5.0,
        "a_over_rstar": a_rs,
        "inclination_deg": 80.0,
        "rv_k": 0.0,
        "color": "#9333ea",
    }
    table = coerce_planet_table(pd.DataFrame([row]))
    st.session_state.nontransiting_planets = table
    return table


def _populate_cutout_parameter_controls(planet_name: str, values: dict[str, float]) -> None:
    key_map = {
        "t0": f"cutout_linear_t0_{planet_name}",
        "period": f"cutout_linear_period_{planet_name}",
        "radius_ratio": f"cutout_radius_ratio_{planet_name}",
        "impact": f"cutout_impact_{planet_name}",
        "a_over_rstar": f"cutout_a_over_rstar_{planet_name}",
        "limb_darkening_u1": f"cutout_ld_u1_{planet_name}",
        "limb_darkening_u2": f"cutout_ld_u2_{planet_name}",
        "baseline_offset": f"cutout_baseline_offset_{planet_name}",
    }
    for name, key in key_map.items():
        value = values.get(name)
        if value is None or not np.isfinite(float(value)):
            continue
        if name == "t0":
            st.session_state[key] = _align_t0_to_photometry_time(float(value), st.session_state.photometry)
        else:
            st.session_state[key] = float(value)
    duration = values.get("duration_hours")
    if duration is not None and np.isfinite(float(duration)) and float(duration) > 0:
        planets = coerce_planet_table(st.session_state.get("planets"))
        mask = planets["name"].astype(str) == str(planet_name)
        if mask.any():
            planets.loc[mask, "duration_hours"] = float(duration)
            st.session_state["planets"] = coerce_planet_table(planets)


def _planet_row_cutout_values(row: pd.Series | dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for source, target in [
        ("t0", "t0"),
        ("period", "period"),
        ("radius_ratio", "radius_ratio"),
        ("impact", "impact"),
        ("a_over_rstar", "a_over_rstar"),
        ("duration_hours", "duration_hours"),
    ]:
        try:
            value = float(row.get(source, np.nan))
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            values[target] = value
    values.setdefault("limb_darkening_u1", 0.5)
    values.setdefault("limb_darkening_u2", 0.1)
    values.setdefault("baseline_offset", 0.0)
    return values


def _parse_single_transit_parameter_text(text: str) -> dict[str, dict[str, float]]:
    planets: dict[str, dict[str, float]] = {}
    current: str | None = None
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.fullmatch(r"\[planet\s+(.+?)\]", line, flags=re.IGNORECASE)
        if match:
            current = match.group(1).strip()
            planets.setdefault(current, {})
            continue
        if current is None or ":" not in line:
            continue
        key, value_text = line.split(":", 1)
        key = key.strip()
        try:
            value = float(value_text.strip())
        except ValueError:
            continue
        if np.isfinite(value):
            planets.setdefault(current, {})[key] = value
    return planets


def _update_planets_from_single_transit_file(parsed: dict[str, dict[str, float]]) -> None:
    if not parsed:
        return
    planets = coerce_planet_table(st.session_state.get("planets"))
    rows = planets.to_dict("records")
    by_name = {str(row.get("name", "")): index for index, row in enumerate(rows)}
    for planet_name, values in parsed.items():
        if not values:
            continue
        if planet_name in by_name:
            row = rows[by_name[planet_name]]
        else:
            row = PlanetParams(name=planet_name).to_row()
            rows.append(row)
            by_name[planet_name] = len(rows) - 1
        for key in ["t0", "period", "radius_ratio", "impact", "duration_hours", "a_over_rstar"]:
            value = values.get(key)
            if value is not None and np.isfinite(float(value)):
                row[key] = float(value)
    st.session_state["planets"] = coerce_planet_table(pd.DataFrame(rows))
    existing = st.session_state.get("phot_fit_single_transit_ls_results_by_planet", {})
    if not isinstance(existing, dict):
        existing = {}
    for planet_name, values in parsed.items():
        existing[str(planet_name)] = {"planet": str(planet_name), **values}
    st.session_state["phot_fit_single_transit_ls_results_by_planet"] = existing


def _align_t0_to_photometry_time(t0: float, photometry: pd.DataFrame) -> float:
    if photometry.empty or "time" not in photometry:
        return float(t0)
    median_time = float(pd.to_numeric(photometry["time"], errors="coerce").median())
    if not np.isfinite(median_time):
        return float(t0)
    if median_time < 100000 and t0 > 2400000:
        return float(t0 - 2457000.0)
    if median_time > 2400000 and t0 < 100000:
        return float(t0 + 2457000.0)
    return float(t0)


def _load_cutout_sector_tables(tables: dict[str, pd.DataFrame], source: str) -> bool:
    cleaned = []
    for key, table in tables.items():
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        data = _clean_fit_sector_table(table, str(key))
        if "is_outlier" in data:
            data = data.loc[~data["is_outlier"].astype(bool)].copy()
        if not data.empty:
            cleaned.append(data)
    if not cleaned:
        return False
    photometry = pd.concat(cleaned, ignore_index=True).sort_values("time").reset_index(drop=True)
    st.session_state.photometry = normalize_photometry(photometry)
    st.session_state["cutout_photometry_source"] = source
    return True


def _load_cutout_photometry_controls() -> None:
    fit_data = st.session_state.get("photometry_fit_data")
    fit_sector_data = st.session_state.get("photometry_fit_sector_data")
    prepared = st.session_state.get("prepared_photometry")
    prepared_sectors = st.session_state.get("prepared_sector_photometry")
    cols = st.columns(3)
    with cols[0]:
        if isinstance(fit_data, pd.DataFrame) and not fit_data.empty:
            if st.button("Load photometry from Linear Transit Fit page", use_container_width=True):
                st.session_state.photometry = normalize_photometry(fit_data)
                st.session_state["cutout_photometry_source"] = "Linear Transit Fit page"
                st.success(f"Loaded {len(st.session_state.photometry):,} photometry points from the Linear Transit Fit page.")
                st.rerun()
        elif isinstance(prepared, pd.DataFrame) and not prepared.empty:
            if st.button("Load prepared photometry", use_container_width=True):
                export = prepared.loc[~prepared.get("is_outlier", False)].copy() if "is_outlier" in prepared else prepared.copy()
                st.session_state.photometry = normalize_photometry(export)
                st.session_state["cutout_photometry_source"] = "prepared stitched photometry"
                st.success(f"Loaded {len(st.session_state.photometry):,} prepared photometry points.")
                st.rerun()
    with cols[1]:
        if isinstance(fit_sector_data, dict) and fit_sector_data:
            if st.button("Load sector data from Linear Transit Fit page", use_container_width=True):
                if _load_cutout_sector_tables(fit_sector_data, "Linear Transit Fit sector data"):
                    st.success(f"Loaded {len(st.session_state.photometry):,} photometry points from Linear Transit Fit sectors.")
                    st.rerun()
                st.warning("No usable sector data was found on the Linear Transit Fit page.")
        elif isinstance(prepared_sectors, dict) and prepared_sectors:
            if st.button("Load sector data from data preparation", use_container_width=True):
                if _load_cutout_sector_tables(prepared_sectors, "TTV data preparation sector data"):
                    st.success(f"Loaded {len(st.session_state.photometry):,} photometry points from prepared sectors.")
                    st.rerun()
                st.warning("No usable prepared sector data was found.")
        else:
            st.button("Load sector data", use_container_width=True, disabled=True)
            st.info("No sector data has been loaded or prepared yet.")
    with cols[2]:
        directory = st.text_input("Sector directory", value=_default_sector_directory(), key="cutout_sector_directory")
        if st.button("Load sector directory for cutouts", use_container_width=True):
            tables = _sector_tables_from_directory(directory)
            if _load_cutout_sector_tables(tables, f"Sector directory: {directory}"):
                st.success(f"Loaded {len(st.session_state.photometry):,} photometry points from {len(tables)} sector file(s).")
                st.rerun()
            st.warning("No usable sector CSV files were found in that directory.")
    upload = st.file_uploader("Upload photometry for cutouts", type=["csv", "tsv", "txt", "dat"], key="cutout_phot_upload")
    if upload is not None:
        st.session_state.photometry = normalize_photometry(read_table(upload))
        st.session_state["cutout_photometry_source"] = upload.name
        st.success(f"Loaded {len(st.session_state.photometry):,} uploaded photometry points.")
        st.rerun()


def _render_selected_cutout_fit(planet_name: str) -> None:
    details = st.session_state.get("cutout_fit_details", pd.DataFrame())
    if not isinstance(details, pd.DataFrame) or details.empty:
        return
    if "planet" in details.columns:
        details = details.loc[details["planet"].astype(str) == str(planet_name)].copy()
    if details.empty:
        return

    details = details.sort_values("epoch")
    options = details["epoch"].astype(int).tolist()
    label_by_epoch = {}
    for row in details.to_dict("records"):
        oc_minutes = (float(row["tmid"]) - float(row["expected_tmid"])) * 24.0 * 60.0
        label_by_epoch[int(row["epoch"])] = (
            f"Epoch {int(row['epoch'])}: T0 {float(row['tmid']):.10f} "
            f"(O-C {oc_minutes:+.2f} min, {int(row['points'])} points)"
        )

    selected_epoch = st.selectbox(
        "Inspect fitted transit",
        options,
        format_func=lambda epoch: label_by_epoch.get(int(epoch), f"Epoch {int(epoch)}"),
        key=f"cutout_fit_inspect_{planet_name}",
        help="Choose one fitted transit for an interactive view of its photometry and fitted midpoint model.",
    )
    selected = details.loc[details["epoch"].astype(int) == int(selected_epoch)].iloc[0].to_dict()
    st.plotly_chart(
        cutout_fit_figure(st.session_state.photometry, selected),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=f"cutout_fit_plot_{planet_name}_{int(selected_epoch)}",
    )
    filename = f"{_target_label()}_{_safe_filename_part(planet_name)}_Tr{int(selected_epoch)}_LS.png"
    st.download_button(
        "Download selected LS fit PNG",
        _fit_png_bytes(st.session_state.photometry, selected),
        filename,
        "image/png",
        use_container_width=True,
    )


def _render_cutout_mcmc_results(planet_name: str, t0: float, period: float) -> None:
    details = st.session_state.get("cutout_mcmc_fit_details", pd.DataFrame())
    timings = st.session_state.get("cutout_mcmc_timings", pd.DataFrame())
    samples_by_epoch = st.session_state.get("cutout_mcmc_samples", {})
    if not isinstance(details, pd.DataFrame) or details.empty:
        return
    details = details.loc[details["planet"].astype(str) == str(planet_name)].copy()
    if details.empty:
        return

    st.subheader("Per-transit T0 MCMC results")
    fit_photometry = st.session_state.get("cutout_mcmc_photometry", st.session_state.photometry)
    details = details.sort_values("epoch")
    options = details["epoch"].astype(int).tolist()
    selected_epoch = st.selectbox(
        "Inspect MCMC transit fit",
        options,
        format_func=lambda epoch: f"Transit {int(epoch)}",
        key=f"cutout_mcmc_fit_select_{planet_name}",
    )
    selected = details.loc[details["epoch"].astype(int) == int(selected_epoch)].iloc[0].to_dict()
    st.plotly_chart(
        cutout_fit_figure(fit_photometry, selected),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=f"cutout_mcmc_fit_plot_{planet_name}_{int(selected_epoch)}",
    )
    fit_filename = f"{_target_label()}_{_safe_filename_part(planet_name)}_Tr{int(selected_epoch)}.png"
    st.download_button(
        "Download selected MCMC fit PNG",
        _fit_png_bytes(fit_photometry, selected),
        fit_filename,
        "image/png",
        use_container_width=True,
    )

    selected_samples = samples_by_epoch.get(int(selected_epoch), pd.DataFrame())
    if not isinstance(selected_samples, pd.DataFrame):
        selected_samples = pd.DataFrame(selected_samples)
    st.plotly_chart(
        t0_histogram_figure(
            selected_samples,
            float(selected["tmid"]),
            float(selected["tmid_err_minus"]),
            float(selected["tmid_err_plus"]),
        ),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=f"cutout_mcmc_hist_{planet_name}_{int(selected_epoch)}",
    )

    if isinstance(timings, pd.DataFrame) and not timings.empty:
        current = timings.loc[timings["planet"].astype(str) == str(planet_name)].copy()
        st.plotly_chart(
            oc_figure(current, t0, period),
            use_container_width=True,
            config=PLOT_CONFIG,
            key=f"cutout_mcmc_oc_{planet_name}",
        )
        oc_filename = f"{_target_label()}_{_safe_filename_part(planet_name)}_OC_MCMC.png"
        st.download_button(
            "Download MCMC O-C PNG",
            _oc_png_bytes(current, t0, period),
            oc_filename,
            "image/png",
            use_container_width=True,
        )
        st.dataframe(current, use_container_width=True)
        st.download_button(
            "Download MCMC timing table",
            current.to_csv(index=False),
            "ttv_timings_mcmc.csv",
            "text/csv",
            use_container_width=True,
        )


def _current_planet_timings(planet_name: str) -> pd.DataFrame:
    timings = st.session_state.timings
    if not isinstance(timings, pd.DataFrame) or timings.empty:
        return pd.DataFrame()
    if "planet" not in timings.columns:
        return timings
    return timings.loc[timings["planet"].astype(str) == str(planet_name)].copy()


def _safe_filename_part(value: object) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "target"))
    return cleaned.strip("_") or "target"


def _target_label() -> str:
    return _safe_filename_part(
        st.session_state.get("target_name")
        or st.session_state.get("photometry_target_input")
        or st.session_state.get("mast_resolved_target")
        or "target"
    )


def _cutout_window(phot: pd.DataFrame, fit: dict[str, float]) -> pd.DataFrame:
    return phot.loc[(phot["time"] >= float(fit["start"])) & (phot["time"] <= float(fit["end"]))].copy()


def _fit_png_bytes(phot: pd.DataFrame, fit: dict[str, float]) -> bytes:
    import matplotlib.pyplot as plt
    from ttv_fitter.models import limb_darkened_transit_model

    window = _cutout_window(phot, fit)
    time = np.linspace(float(fit["start"]), float(fit["end"]), 800)
    model = limb_darkened_transit_model(
        time,
        float(fit["period"]),
        float(fit["tmid"]),
        float(fit["radius_ratio"]),
        float(fit["impact"]),
        float(fit["duration_hours"]),
        float(fit["limb_darkening_u1"]),
        float(fit["limb_darkening_u2"]),
        baseline_offset=float(fit.get("baseline_offset", 0.0)),
        a_over_rstar=float(fit.get("a_over_rstar", np.nan)),
    )
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    yerr = window["flux_err"] if "flux_err" in window else None
    ax.errorbar(window["time"], window["flux"], yerr=yerr, fmt=".", color="#334155", alpha=0.45, ms=3)
    ax.plot(time, model, color="#dc2626", lw=2.4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.set_title(f"Transit {int(fit['epoch'])} T0 fit")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _oc_png_bytes(timings: pd.DataFrame, t0: float, period: float) -> bytes:
    import matplotlib.pyplot as plt

    table = oc_table(timings, t0, period)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    err = table["tmid_err"] * 1440.0 if "tmid_err" in table.columns else None
    ax.errorbar(table["epoch"], table["oc_minutes"], yerr=err, fmt="o", color="#1d4ed8")
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_xlabel("Transit epoch")
    ax.set_ylabel("O-C [minutes]")
    ax.set_title("MCMC T0 O-C")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _t0_histogram_png_bytes(samples: pd.DataFrame, tmid: float, err_minus: float, err_plus: float) -> bytes:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    if isinstance(samples, pd.DataFrame) and "tmid" in samples.columns and not samples.empty:
        ax.hist(samples["tmid"].to_numpy(dtype=float), bins=40, color="#2563eb", alpha=0.72)
    ax.axvline(tmid, color="#dc2626", lw=2.4, label="median")
    ax.axvspan(tmid - err_minus, tmid + err_plus, color="#dc2626", alpha=0.14)
    ax.set_xlabel("T0")
    ax.set_ylabel("Samples")
    ax.set_title("T0 posterior")
    ax.legend(loc="best")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _finite_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(parsed) if np.isfinite(parsed) else float(default)


def _ensure_ttv_timing_photometry() -> bool:
    # The timing section belongs to the fit workflow above it.  Always prefer
    # that workflow's explicit data source over the generic bridge photometry,
    # which may still contain a different target from an earlier app run.
    for key, source in [
        ("photometry_fit_sector_data", "TTV fitting sector data"),
    ]:
        tables = st.session_state.get(key)
        if isinstance(tables, dict) and tables and _load_cutout_sector_tables(tables, source):
            return True

    for key, source in [
        ("photometry_fit_data", "TTV fitting photometry"),
    ]:
        data = st.session_state.get(key)
        if isinstance(data, pd.DataFrame) and not data.empty:
            if "is_outlier" in data:
                data = data.loc[~data["is_outlier"].astype(bool)].copy()
            st.session_state.photometry = normalize_photometry(data)
            st.session_state["cutout_photometry_source"] = source
            return True

    for key, source in [
        ("prepared_sector_photometry", "TTV data preparation sector data"),
    ]:
        tables = st.session_state.get(key)
        if isinstance(tables, dict) and tables and _load_cutout_sector_tables(tables, source):
            return True

    for key, source in [
        ("prepared_photometry", "prepared stitched photometry"),
    ]:
        data = st.session_state.get(key)
        if isinstance(data, pd.DataFrame) and not data.empty:
            if "is_outlier" in data:
                data = data.loc[~data["is_outlier"].astype(bool)].copy()
            st.session_state.photometry = normalize_photometry(data)
            st.session_state["cutout_photometry_source"] = source
            return True

    photometry = st.session_state.get("photometry", pd.DataFrame())
    if isinstance(photometry, pd.DataFrame) and not photometry.empty and {"time", "flux"}.issubset(photometry.columns):
        st.session_state.photometry = normalize_photometry(photometry)
        st.session_state.setdefault("cutout_photometry_source", "previously loaded photometry")
        return True

    return False


def _streamlined_timing_parameters() -> tuple[str, dict[str, float], str]:
    planets = coerce_planet_table(st.session_state.get("planets"))
    names = planets["name"].astype(str).tolist() if not planets.empty else ["b"]
    planet_name = "b" if "b" in names else names[0]
    values, source = _linear_fit_parameter_set(planet_name, "single_ls")
    if not values and not planets.empty:
        row = planets.loc[planets["name"].astype(str) == str(planet_name)].iloc[0]
        values = _planet_row_cutout_values(row)
        source = "current planet table"

    defaults = {
        "t0": 0.0,
        "period": 3.0,
        "radius_ratio": 0.08,
        "impact": 0.4,
        "a_over_rstar": 8.0,
        "duration_hours": 3.0,
        "limb_darkening_u1": 0.5,
        "limb_darkening_u2": 0.1,
        "baseline_offset": 0.0,
    }
    clean = {key: _finite_float(values.get(key), default) for key, default in defaults.items()}
    clean["period"] = max(clean["period"], 1e-8)
    clean["duration_hours"] = max(clean["duration_hours"], 1e-4)
    clean["radius_ratio"] = float(np.clip(clean["radius_ratio"], 1e-6, 1.0))
    clean["impact"] = float(np.clip(clean["impact"], 0.0, 2.0))
    clean["a_over_rstar"] = max(clean["a_over_rstar"], 1e-6)
    clean["t0"] = _align_t0_to_photometry_time(clean["t0"], st.session_state.get("photometry", pd.DataFrame()))
    # A single-transit fit stores the transit number shown in the selector.
    # Convert its fitted midpoint back to the same epoch-zero reference so the
    # timing stage does not silently rename that transit as epoch zero.
    epoch_offset = 0
    single_fit = st.session_state.get("phot_fit_single_transit_ls_result")
    if isinstance(single_fit, dict) and str(single_fit.get("planet", planet_name)) == str(planet_name):
        try:
            epoch_offset = int(single_fit.get("epoch", 0))
        except (TypeError, ValueError):
            epoch_offset = 0
    clean["reference_epoch"] = float(epoch_offset)
    clean["t0"] = float(clean["t0"] - epoch_offset * clean["period"])
    return planet_name, clean, source


def _timing_photometry_signature(photometry: pd.DataFrame, source: str) -> tuple[object, ...]:
    if not isinstance(photometry, pd.DataFrame) or photometry.empty or "time" not in photometry:
        return (str(source), 0, np.nan, np.nan)
    time = pd.to_numeric(photometry["time"], errors="coerce")
    sectors = tuple(sorted(photometry["sector"].dropna().astype(str).unique())) if "sector" in photometry else ()
    files = tuple(sorted(photometry["source_file"].dropna().astype(str).unique())) if "source_file" in photometry else ()
    return (str(source), int(len(photometry)), float(time.min()), float(time.max()), sectors, files)


def _clear_stale_timing_results() -> None:
    for key in [
        "timings",
        "cutout_fit_details",
        "cutout_mcmc_timings",
        "cutout_mcmc_fit_details",
        "cutout_mcmc_samples",
        "cutout_mcmc_photometry",
    ]:
        st.session_state.pop(key, None)


def _ttv_timing_output_dirs() -> tuple[Path, Path]:
    target = _target_label()
    base = Path("data") / "prepared" / target
    plots_dir = Path(st.session_state.get("prepared_sector_plot_directory") or (base / "plots"))
    results_dir = base / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, results_dir


def _write_plotly_html(fig, path: Path) -> None:
    fig.update_layout(margin=dict(l=90, r=150, t=55, b=75))
    fig.write_html(path, include_plotlyjs="cdn")


def _save_streamlined_ttv_outputs(planet_name: str, t0: float, period: float) -> None:
    details = st.session_state.get("cutout_mcmc_fit_details", pd.DataFrame())
    timings = st.session_state.get("cutout_mcmc_timings", pd.DataFrame())
    samples_by_epoch = st.session_state.get("cutout_mcmc_samples", {})
    if not isinstance(details, pd.DataFrame) or details.empty or not isinstance(timings, pd.DataFrame) or timings.empty:
        return

    current_details = details.loc[details["planet"].astype(str) == str(planet_name)].copy()
    current_timings = timings.loc[timings["planet"].astype(str) == str(planet_name)].copy()
    if current_details.empty or current_timings.empty:
        return

    fit_photometry = st.session_state.get("cutout_mcmc_photometry", st.session_state.photometry)
    plots_dir, results_dir = _ttv_timing_output_dirs()
    target = _target_label()
    planet_part = _safe_filename_part(planet_name)
    saved = 0
    for detail in current_details.to_dict("records"):
        epoch = int(detail["epoch"])
        stem = f"{target}_{planet_part}_Tr{epoch}"
        fit_fig = cutout_fit_figure(fit_photometry, detail)
        _write_plotly_html(fit_fig, plots_dir / f"{stem}_mcmc_fit.html")
        (plots_dir / f"{stem}_mcmc_fit.png").write_bytes(_fit_png_bytes(fit_photometry, detail))

        samples = samples_by_epoch.get(epoch, pd.DataFrame())
        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame(samples)
        hist_fig = t0_histogram_figure(
            samples,
            float(detail["tmid"]),
            float(detail["tmid_err_minus"]),
            float(detail["tmid_err_plus"]),
        )
        _write_plotly_html(hist_fig, plots_dir / f"{stem}_t0_distribution.html")
        (plots_dir / f"{stem}_t0_distribution.png").write_bytes(
            _t0_histogram_png_bytes(
                samples,
                float(detail["tmid"]),
                float(detail["tmid_err_minus"]),
                float(detail["tmid_err_plus"]),
            )
        )
        saved += 3

    oc_fig = oc_figure(current_timings, t0, period)
    for directory in [plots_dir, results_dir]:
        _write_plotly_html(oc_fig, directory / f"{target}_{planet_part}_OC_MCMC.html")
        (directory / f"{target}_{planet_part}_OC_MCMC.png").write_bytes(_oc_png_bytes(current_timings, t0, period))

    oc_data = oc_table(current_timings, t0, period)
    oc_data.to_csv(results_dir / f"{target}_{planet_part}_oc_table.csv", index=False)
    current_timings.to_csv(results_dir / f"{target}_{planet_part}_mcmc_timings.csv", index=False)
    st.session_state["streamlined_ttv_plots_dir"] = str(plots_dir)
    st.session_state["streamlined_ttv_results_dir"] = str(results_dir)
    st.session_state["streamlined_ttv_saved_plot_count"] = saved + 2


def _render_streamlined_ttv_timing_section() -> None:
    st.divider()
    st.subheader("Transit timing fits")
    st.caption("Fit every suitable transit cutout, then estimate T0 uncertainties with a one-parameter MCMC for each transit.")

    if not _ensure_ttv_timing_photometry():
        st.info("Complete the single-transit fit and import sector data above before running timing fits.")
        return

    timing_source = str(st.session_state.get("cutout_photometry_source", "TTV fitting photometry"))
    timing_signature = _timing_photometry_signature(st.session_state.photometry, timing_source)
    previous_signature = st.session_state.get("streamlined_ttv_data_signature")
    has_unbound_results = (
        previous_signature is None
        and isinstance(st.session_state.get("cutout_mcmc_fit_details"), pd.DataFrame)
        and not st.session_state.get("cutout_mcmc_fit_details", pd.DataFrame()).empty
        and not isinstance(st.session_state.get("cutout_mcmc_photometry"), pd.DataFrame)
    )
    if has_unbound_results or (previous_signature is not None and previous_signature != timing_signature):
        _clear_stale_timing_results()
    st.session_state["streamlined_ttv_data_signature"] = timing_signature

    planet_name, params, source = _streamlined_timing_parameters()
    if params["t0"] == 0.0 or params["period"] <= 0:
        st.info("Complete the single-transit fit above so the timing fitter has a T0, period, and transit shape.")
        return

    duration_days = max(params["duration_hours"] / 24.0, 1e-4)
    default_cutout = max(4.0 * duration_days, 0.125)
    default_search = min(default_cutout * 0.8, max(duration_days, 0.05))
    st.session_state.setdefault("streamlined_cutout_half_width", float(default_cutout))
    st.session_state.setdefault("streamlined_t0_search_half_width", float(default_search))

    c1, c2 = st.columns(2)
    half_width = c1.number_input(
        "Cutout half-width [days]",
        min_value=0.001,
        format="%.5f",
        key="streamlined_cutout_half_width",
        help="Half-width of each transit cutout around the predicted linear ephemeris midpoint.",
    )
    search_half_width = c2.number_input(
        "T0 search half-width [days]",
        min_value=1e-5,
        format="%.5f",
        key="streamlined_t0_search_half_width",
        help="Allowed midpoint shift for LS and MCMC timing fits. MCMC ranges are centred on the LS midpoint for each transit.",
    )

    cutouts = build_cutouts(st.session_state.photometry, params["t0"], params["period"], half_width)
    suitable = cutouts.loc[cutouts["points"].astype(int) >= 20].copy() if not cutouts.empty else pd.DataFrame()
    st.caption(
        f"Using planet {planet_name} parameters from {source or 'the current fit state'}. "
        f"Photometry source: {timing_source} ({len(st.session_state.photometry):,} points; "
        f"BTJD {st.session_state.photometry['time'].min():.3f} to {st.session_state.photometry['time'].max():.3f}). "
        f"{len(suitable)} of {len(cutouts)} predicted cutout(s) have enough points to fit."
    )
    if suitable.empty:
        st.info("No suitable transit cutouts were found for the current ephemeris and cutout width.")
        return

    if st.button("Fit cutout midpoints and run T0 MCMC", use_container_width=True, type="primary"):
        phot = st.session_state.photometry
        results: list[dict[str, object]] = []
        fit_details: list[dict[str, object]] = []
        ls_progress = st.progress(0.0)
        ls_status = st.empty()
        for index, item in enumerate(suitable.to_dict("records"), start=1):
            ls_status.info(f"Least-squares fitting transit {index}/{len(suitable)} (epoch {int(item['epoch'])})...")
            mask = (phot["time"] >= item["start"]) & (phot["time"] <= item["end"])
            result = fit_cutout_t0(
                phot.loc[mask],
                params["period"],
                float(item["expected_tmid"]),
                params["radius_ratio"],
                params["impact"],
                params["a_over_rstar"],
                params["duration_hours"],
                params["limb_darkening_u1"],
                params["limb_darkening_u2"],
                params["baseline_offset"],
                search_half_width,
                use_t0_multistart=True,
            )
            if result.success:
                row = {
                    "planet": planet_name,
                    "epoch": int(item["epoch"]),
                    "tmid": result.params["tmid"],
                    "expected_tmid": item["expected_tmid"],
                    "points": int(item["points"]),
                }
                results.append(row)
                fit_details.append(
                    {
                        **row,
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "period": params["period"],
                        "radius_ratio": params["radius_ratio"],
                        "impact": params["impact"],
                        "a_over_rstar": params["a_over_rstar"],
                        "duration_hours": params["duration_hours"],
                        "limb_darkening_u1": params["limb_darkening_u1"],
                        "limb_darkening_u2": params["limb_darkening_u2"],
                        "baseline_offset": params["baseline_offset"],
                    }
                )
            ls_progress.progress(index / len(suitable))

        if not fit_details:
            ls_status.warning("No cutout midpoint fits converged.")
            return

        st.session_state.timings = pd.DataFrame(results)
        st.session_state.cutout_fit_details = pd.DataFrame(fit_details)
        ls_status.success(f"Least-squares midpoint fits completed for {len(fit_details)} transit(s).")

        mcmc_rows: list[dict[str, object]] = []
        mcmc_details: list[dict[str, object]] = []
        samples_by_epoch: dict[int, pd.DataFrame] = {}
        mcmc_progress = st.progress(0.0)
        mcmc_status = st.empty()
        for index, fit in enumerate(fit_details, start=1):
            epoch = int(fit["epoch"])
            mcmc_status.info(f"MCMC fitting transit {index}/{len(fit_details)} (epoch {epoch})...")
            mask = (phot["time"] >= fit["start"]) & (phot["time"] <= fit["end"])
            result = run_cutout_t0_mcmc(
                phot.loc[mask],
                float(fit["period"]),
                float(fit["tmid"]),
                float(fit["radius_ratio"]),
                float(fit["impact"]),
                float(fit["a_over_rstar"]),
                float(fit["duration_hours"]),
                float(fit["limb_darkening_u1"]),
                float(fit["limb_darkening_u2"]),
                float(fit["baseline_offset"]),
                search_half_width_days=float(search_half_width),
                nwalkers=6,
                nsteps=1000,
                burn=500,
                trim_nonconverged_walkers=True,
            )
            if result.success:
                row = {
                    "planet": planet_name,
                    "epoch": epoch,
                    "tmid": result.params["tmid"],
                    "tmid_err": result.params["tmid_err"],
                    "tmid_err_minus": result.params["tmid_err_minus"],
                    "tmid_err_plus": result.params["tmid_err_plus"],
                    "expected_tmid": fit["expected_tmid"],
                    "points": int(fit["points"]),
                    "nwalkers_used": result.params.get("nwalkers_used", np.nan),
                    "nwalkers_trimmed": result.params.get("nwalkers_trimmed", np.nan),
                }
                detail = dict(fit)
                detail.update(result.params)
                mcmc_rows.append(row)
                mcmc_details.append(detail)
                samples_by_epoch[epoch] = result.samples
            mcmc_progress.progress(index / len(fit_details))

        if not mcmc_rows:
            mcmc_status.warning("No MCMC timing fits converged.")
            return

        st.session_state.cutout_mcmc_timings = pd.DataFrame(mcmc_rows)
        st.session_state.cutout_mcmc_fit_details = pd.DataFrame(mcmc_details)
        st.session_state.cutout_mcmc_samples = samples_by_epoch
        st.session_state.cutout_mcmc_photometry = phot.copy()
        _save_streamlined_ttv_outputs(planet_name, params["t0"], params["period"])
        plots_dir = st.session_state.get("streamlined_ttv_plots_dir")
        results_dir = st.session_state.get("streamlined_ttv_results_dir")
        mcmc_status.success(f"MCMC timing fits completed. Saved plots to {plots_dir} and results to {results_dir}.")

    _render_cutout_mcmc_results(planet_name, params["t0"], params["period"])


def per_transit_tab() -> None:
    st.subheader("Per-Transit T0 Fit")
    st.caption("Build transit cutouts from a linear ephemeris and refit only each midpoint.")
    with st.expander("Load photometry for per-transit cutouts", expanded=st.session_state.photometry.empty):
        _load_cutout_photometry_controls()
        source = st.session_state.get("cutout_photometry_source")
        if not st.session_state.photometry.empty:
            suffix = f" from {source}" if source else ""
            st.caption(f"Current cutout photometry: {len(st.session_state.photometry):,} point(s){suffix}.")

    planets = coerce_planet_table(st.session_state.planets)
    selected_name = st.selectbox(
        "Planet",
        planets["name"].tolist(),
        key="cutout_planet_select",
        help="Planet whose linear ephemeris will be used to make transit cutouts and fit individual midpoints.",
    )
    row = planets.loc[planets["name"] == selected_name].iloc[0]
    t0_key = f"cutout_linear_t0_{selected_name}"
    period_key = f"cutout_linear_period_{selected_name}"
    radius_ratio_key = f"cutout_radius_ratio_{selected_name}"
    impact_key = f"cutout_impact_{selected_name}"
    a_over_rstar_key = f"cutout_a_over_rstar_{selected_name}"
    ld_u1_key = f"cutout_ld_u1_{selected_name}"
    ld_u2_key = f"cutout_ld_u2_{selected_name}"
    baseline_offset_key = f"cutout_baseline_offset_{selected_name}"
    if t0_key not in st.session_state:
        st.session_state[t0_key] = _align_t0_to_photometry_time(float(row["t0"]), st.session_state.photometry)
    if period_key not in st.session_state:
        st.session_state[period_key] = float(row["period"])
    st.session_state.setdefault(radius_ratio_key, float(row["radius_ratio"]))
    st.session_state.setdefault(impact_key, float(row["impact"]))
    st.session_state.setdefault(a_over_rstar_key, float(row["a_over_rstar"]))
    st.session_state.setdefault(ld_u1_key, 0.5)
    st.session_state.setdefault(ld_u2_key, 0.1)
    st.session_state.setdefault(baseline_offset_key, 0.0)
    aligned_t0 = _align_t0_to_photometry_time(float(st.session_state[t0_key]), st.session_state.photometry)
    if aligned_t0 != float(st.session_state[t0_key]):
        st.session_state[t0_key] = aligned_t0
    pull_cols = st.columns(4)
    with pull_cols[0]:
        if st.button("Load current planet-table parameters", use_container_width=True):
            values = _planet_row_cutout_values(row)
            if not values:
                st.warning("No usable parameters were found for the selected planet.")
            else:
                _populate_cutout_parameter_controls(selected_name, values)
                st.success("Loaded T0, period, and transit-shape start values from the current planet table.")
                st.rerun()
    with pull_cols[1]:
        if st.button(
            "Pull latest MCMC median Linear T0 parameters from the Linear Transit Fit page",
            use_container_width=True,
        ):
            values, source = _linear_fit_parameter_set(selected_name, "mcmc")
            if not values:
                st.warning("No Linear Transit Fit MCMC chain was found yet. Run MCMC on the Linear Transit Fit page first.")
            else:
                _populate_cutout_parameter_controls(selected_name, values)
                st.success(f"Pulled fitted transit-shape parameters from {source}.")
                st.rerun()
    with pull_cols[2]:
        if st.button(
            "Pull latest LS fit Linear T0 parameters from the Linear Transit Fit page",
            use_container_width=True,
        ):
            values, source = _linear_fit_parameter_set(selected_name, "ls")
            if not values:
                st.warning("No Linear Transit Fit parameter table was found yet. Run or generate parameters on the Linear Transit Fit page first.")
            else:
                _populate_cutout_parameter_controls(selected_name, values)
                st.success(f"Pulled fitted transit-shape parameters from {source}.")
                st.rerun()
    with pull_cols[3]:
        if st.button(
            "Pull latest Single transit LS fit parameters from the Linear Transit Fit page",
            use_container_width=True,
        ):
            values, source = _linear_fit_parameter_set(selected_name, "single_ls")
            if not values:
                st.warning("No matching single-transit LS fit was found yet. Run Fit one transit on the Linear Transit Fit page first.")
            else:
                _populate_cutout_parameter_controls(selected_name, values)
                st.success(f"Pulled fitted transit-shape parameters from {source}.")
                st.rerun()
    uploaded_single_transit_params = st.file_uploader(
        "Or load single-transit planet parameters file",
        type=["txt"],
        key="cutout_single_transit_parameter_upload",
        help="Reads the text file downloaded from Linear Transit Fit > Fit one transit, with [planet b] sections.",
    )
    if uploaded_single_transit_params is not None:
        try:
            parsed_params = _parse_single_transit_parameter_text(uploaded_single_transit_params.getvalue().decode("utf-8"))
        except UnicodeDecodeError:
            parsed_params = {}
        if not parsed_params:
            st.warning("No planet parameter blocks were found in that file.")
        elif st.button("Load uploaded single-transit parameters", use_container_width=True):
            _update_planets_from_single_transit_file(parsed_params)
            selected_values = parsed_params.get(str(selected_name), {})
            if selected_values:
                _populate_cutout_parameter_controls(selected_name, selected_values)
                st.success(f"Loaded parameters for planet {selected_name} and updated {len(parsed_params)} planet block(s) from the file.")
            else:
                st.success(f"Updated {len(parsed_params)} planet block(s) from the file. Select one of those planets to load its cutout controls.")
            st.rerun()
    c1, c2, c3 = st.columns(3)
    t0 = c1.number_input(
        "Linear T0",
        key=t0_key,
        format="%.10f",
        help="Reference midpoint for the linear ephemeris used to predict each transit window.",
    )
    period = c2.number_input(
        "Linear period [days]",
        key=period_key,
        min_value=1e-8,
        format="%.10f",
        help="Linear spacing between expected transits in days.",
    )
    half_width = c3.number_input(
        "Cutout half-width [days]",
        value=max(float(row["duration_hours"]) / 24.0, 0.1),
        min_value=0.001,
        format="%.5f",
        help="Half-width of each data cutout around the predicted transit midpoint. Larger windows include more baseline but can include extra structure.",
    )
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    radius_ratio = s1.number_input(
        "Radius ratio",
        key=radius_ratio_key,
        min_value=1e-6,
        max_value=1.0,
        format="%.8f",
        help="Planet-to-star radius ratio used by the limb-darkened transit shape.",
    )
    impact = s2.number_input(
        "Impact parameter",
        key=impact_key,
        min_value=0.0,
        max_value=2.0,
        format="%.8f",
        help="Transit chord impact parameter used by the limb-darkened transit shape.",
    )
    a_over_rstar = s3.number_input(
        "a/Rstar",
        key=a_over_rstar_key,
        min_value=1e-6,
        format="%.8f",
        help="Semi-major axis divided by stellar radius. Pulled from allesfitter rsuma so the per-transit model uses the LS transit width.",
    )
    limb_darkening_u1 = s4.number_input(
        "Limb darkening u1",
        key=ld_u1_key,
        min_value=-1.0,
        max_value=1.0,
        format="%.8f",
        help="Quadratic limb-darkening coefficient u1.",
    )
    limb_darkening_u2 = s5.number_input(
        "Limb darkening u2",
        key=ld_u2_key,
        min_value=-1.0,
        max_value=1.0,
        format="%.8f",
        help="Quadratic limb-darkening coefficient u2.",
    )
    baseline_offset = s6.number_input(
        "Baseline offset flux",
        key=baseline_offset_key,
        min_value=-0.5,
        max_value=0.5,
        format="%.8f",
        help="Additive flux offset used as the starting baseline for each cutout fit.",
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
        help="Allowed midpoint shift during each individual T0 fit. Keep this smaller than the cutout half-width.",
    )
    use_ls_multistart = st.checkbox(
        "Run multiple LS fits with a range of T0 offset startpoints - will take longer to run LS fits",
        value=bool(st.session_state.get("cutout_ls_multistart", False)),
        key="cutout_ls_multistart",
        help="Tiles the T0 search range by roughly one transit duration, up to 10 starts, and keeps the lowest-residual LS solution.",
    )
    if st.button("Fit selected cutout midpoints", use_container_width=True):
        results = []
        fit_details = []
        phot = st.session_state.photometry
        for item in edited.loc[edited["fit"]].to_dict("records"):
            mask = (phot["time"] >= item["start"]) & (phot["time"] <= item["end"])
            result = fit_cutout_t0(
                phot.loc[mask],
                period,
                float(item["expected_tmid"]),
                radius_ratio,
                impact,
                a_over_rstar,
                float(row["duration_hours"]),
                limb_darkening_u1,
                limb_darkening_u2,
                baseline_offset,
                search_half_width,
                use_t0_multistart=use_ls_multistart,
            )
            if result.success:
                results.append(
                    {
                        "planet": selected_name,
                        "epoch": int(item["epoch"]),
                        "tmid": result.params["tmid"],
                        "expected_tmid": item["expected_tmid"],
                        "points": int(item["points"]),
                    }
                )
                fit_details.append(
                    {
                        "planet": selected_name,
                        "epoch": int(item["epoch"]),
                        "tmid": result.params["tmid"],
                        "expected_tmid": item["expected_tmid"],
                        "start": item["start"],
                        "end": item["end"],
                        "points": int(item["points"]),
                        "baseline": result.params["baseline"],
                        "baseline_offset": result.params["baseline_offset"],
                        "period": float(period),
                        "radius_ratio": float(radius_ratio),
                        "impact": float(impact),
                        "a_over_rstar": float(a_over_rstar),
                        "limb_darkening_u1": float(limb_darkening_u1),
                        "limb_darkening_u2": float(limb_darkening_u2),
                        "duration_hours": float(row["duration_hours"]),
                        "residual_sum_squares": result.params.get("residual_sum_squares", np.nan),
                        "n_t0_startpoints": result.params.get("n_t0_startpoints", 1.0),
                    }
                )
        st.session_state.timings = pd.DataFrame(results)
        st.session_state.cutout_fit_details = pd.DataFrame(fit_details)
        st.success(f"Fit {len(results)} transit midpoint(s).")
    _render_selected_cutout_fit(selected_name)
    current_timings = _current_planet_timings(selected_name)
    st.plotly_chart(
        oc_figure(current_timings, t0, period),
        use_container_width=True,
        config=PLOT_CONFIG,
        key="cutout_oc_plot",
    )
    if not current_timings.empty:
        st.dataframe(current_timings, use_container_width=True)
        st.download_button("Download timing table", current_timings.to_csv(index=False), "ttv_timings.csv", "text/csv")
    st.markdown("**T0 MCMC tuning**")
    mcmc_cols = st.columns(4)
    mcmc_search_half_width = mcmc_cols[0].number_input(
        "MCMC T0 search half-width [days]",
        min_value=1e-5,
        value=float(st.session_state.get("cutout_mcmc_search_half_width_days", 1.5 / 24.0)),
        step=0.01,
        format="%.5f",
        key="cutout_mcmc_search_half_width_days",
        help="Each one-parameter MCMC samples T0 uniformly within the LS midpoint plus or minus this value.",
    )
    mcmc_walkers = int(
        mcmc_cols[1].number_input(
            "MCMC walkers",
            min_value=2,
            value=int(st.session_state.get("cutout_mcmc_walkers", 6)),
            step=1,
            key="cutout_mcmc_walkers",
        )
    )
    mcmc_steps = int(
        mcmc_cols[2].number_input(
            "MCMC steps",
            min_value=2,
            value=int(st.session_state.get("cutout_mcmc_steps", 1000)),
            step=100,
            key="cutout_mcmc_steps",
        )
    )
    mcmc_burn = int(
        mcmc_cols[3].number_input(
            "MCMC burn-in steps",
            min_value=0,
            value=min(int(st.session_state.get("cutout_mcmc_burn", 500)), max(mcmc_steps - 1, 0)),
            step=50,
            key="cutout_mcmc_burn",
        )
    )
    mcmc_burn = min(mcmc_burn, max(mcmc_steps - 1, 0))
    trim_mcmc_walkers = st.checkbox(
        "Trim non-converged MCMC walkers before calculating T0 uncertainties",
        value=bool(st.session_state.get("cutout_mcmc_trim_walkers", True)),
        key="cutout_mcmc_trim_walkers",
        help=(
            "After burn-in, exclude whole walkers whose T0 median is clearly separated from the dominant posterior cluster. "
            "If trimming would leave too few walkers, the full chain is kept."
        ),
    )
    st.caption(
        "Walkers are initialized across the full T0 search range. For missed transits, broaden the search half-width first, "
        "then increase walkers or steps if the histogram remains broad or multi-peaked."
    )
    if st.button("Prepare MCMC fits for each transit", use_container_width=True):
        details = st.session_state.get("cutout_fit_details", pd.DataFrame())
        if not isinstance(details, pd.DataFrame) or details.empty:
            st.warning("Run the LS cutout midpoint fits before preparing MCMC fits.")
        else:
            mcmc_rows = []
            mcmc_details = []
            samples_by_epoch = {}
            phot = st.session_state.photometry
            current_details = details.loc[details["planet"].astype(str) == str(selected_name)].copy()
            progress = st.progress(0.0)
            with st.spinner("Running one-parameter T0 MCMC for each selected transit..."):
                for index, fit in enumerate(current_details.to_dict("records"), start=1):
                    mask = (phot["time"] >= fit["start"]) & (phot["time"] <= fit["end"])
                    result = run_cutout_t0_mcmc(
                        phot.loc[mask],
                        float(fit["period"]),
                        float(fit["tmid"]),
                        float(fit["radius_ratio"]),
                        float(fit["impact"]),
                        float(fit["a_over_rstar"]),
                        float(fit["duration_hours"]),
                        float(fit["limb_darkening_u1"]),
                        float(fit["limb_darkening_u2"]),
                        float(fit["baseline_offset"]),
                        search_half_width_days=mcmc_search_half_width,
                        nwalkers=mcmc_walkers,
                        nsteps=mcmc_steps,
                        burn=mcmc_burn,
                        trim_nonconverged_walkers=trim_mcmc_walkers,
                    )
                    if result.success:
                        epoch = int(fit["epoch"])
                        samples_by_epoch[epoch] = result.samples
                        row = {
                            "planet": selected_name,
                            "epoch": epoch,
                            "tmid": result.params["tmid"],
                            "tmid_err": result.params["tmid_err"],
                            "tmid_err_minus": result.params["tmid_err_minus"],
                            "tmid_err_plus": result.params["tmid_err_plus"],
                            "expected_tmid": fit["expected_tmid"],
                            "points": int(fit["points"]),
                            "nwalkers_used": result.params.get("nwalkers_used", np.nan),
                            "nwalkers_trimmed": result.params.get("nwalkers_trimmed", np.nan),
                        }
                        mcmc_rows.append(row)
                        detail = dict(fit)
                        detail.update(result.params)
                        mcmc_details.append(detail)
                    progress.progress(index / max(len(current_details), 1))
            st.session_state.cutout_mcmc_timings = pd.DataFrame(mcmc_rows)
            st.session_state.cutout_mcmc_fit_details = pd.DataFrame(mcmc_details)
            st.session_state.cutout_mcmc_samples = samples_by_epoch
            st.success(f"Prepared MCMC timing fits for {len(mcmc_rows)} transit(s).")
    _render_cutout_mcmc_results(selected_name, t0, period)


def ttv_model_tab() -> None:
    st.subheader("TTV Interface")
    st.caption("Measure per-transit timings, inspect phenomenological O-C curves, or run a physical REBOUND model.")
    if st.button(
        "Retrieve star-planet parameters from the LS fit in the TTV fitting tab",
        use_container_width=True,
        help="Copies the latest least-squares star values and derived planet geometry from the TTV fitting tab into this TTV model setup.",
    ):
        retrieved_planets, retrieved_mass, retrieved_radius = _linear_fit_planet_table_from_ls()
        if retrieved_planets.empty:
            st.warning("No TTV fitting LS parameter table is available yet.")
        else:
            st.session_state.ttv_host_mass = retrieved_mass
            st.session_state.ttv_host_radius = retrieved_radius
            st.session_state.planets = retrieved_planets
            st.session_state.physical_ttv_result = None
            st.session_state.physical_ttv_comparison = pd.DataFrame()
            st.success(f"Retrieved {len(retrieved_planets)} planet row(s) and host parameters from the LS fit.")
            st.rerun()
    cols = st.columns(3)
    mass = cols[0].number_input(
        "Host mass [Msun]",
        value=float(st.session_state.get("ttv_host_mass", 1.0)),
        min_value=0.01,
        format="%.4f",
        help="Stellar mass in solar masses. Used to derive a/Rstar and as the central mass in the REBOUND N-body model.",
    )
    radius = cols[1].number_input(
        "Host radius [Rsun]",
        value=float(st.session_state.get("ttv_host_radius", 1.0)),
        min_value=0.01,
        format="%.4f",
        help="Stellar radius in solar radii. Used to convert a/Rstar to AU and to decide whether a model conjunction is a transit.",
    )
    st.session_state.ttv_host_mass = float(mass)
    st.session_state.ttv_host_radius = float(radius)
    companion = cols[2].text_input(
        "allesfitter companion label",
        value="b",
        help="Label used when exporting allesfitter-style per-transit timing rows.",
    )
    planets = planet_editor("ttv")
    timings_for_ttv, timing_source = _latest_timing_table_for_ttv()

    quick_tab, physical_tab = st.tabs(["Measured / Quick O-C", "Physical REBOUND Model"])

    with quick_tab:
        if timings_for_ttv.empty:
            st.info("Load a timing table or fit per-transit midpoints first.")
        else:
            st.caption(f"Using {timing_source}.")
            ephem = fit_linear_ephemeris(timings_for_ttv)
            c1, c2 = st.columns(2)
            t0 = c1.number_input(
                "Reference T0",
                value=float(ephem.get("t0", planets.iloc[0]["t0"])),
                format="%.10f",
                key="quick_t0",
                help="Linear reference midpoint used to calculate observed minus calculated timing residuals.",
            )
            period = c2.number_input(
                "Reference period",
                value=float(ephem.get("period", planets.iloc[0]["period"])),
                min_value=1e-8,
                format="%.10f",
                key="quick_period",
                help="Linear period used to calculate O-C residuals. This is not a dynamical period fit.",
            )
            if st.button(
                "Fit sinusoidal TTV model",
                use_container_width=True,
                help="Fits a simple sinusoid to the O-C points for quick pattern inspection. This is phenomenological, not an N-body model.",
            ):
                params, table = fit_sinusoidal_ttv(timings_for_ttv, t0, period)
                st.session_state.ttv_params = params
                st.session_state.ttv_model_table = table
                st.success("TTV model fit complete.")
            model_table = st.session_state.get("ttv_model_table", pd.DataFrame())
            st.plotly_chart(
                oc_figure(timings_for_ttv, t0, period, model_table),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="ttv_oc_plot",
            )
            if st.session_state.get("ttv_params"):
                st.json(st.session_state.ttv_params)
            rows = allesfitter_ttv_rows(timings_for_ttv, companion=companion)
            st.dataframe(rows, use_container_width=True)
            st.download_button("Download allesfitter TTV parameter rows", rows.to_csv(index=False), "ttv_params_rows.csv", "text/csv")

    with physical_tab:
        ok, message = rebound_available()
        if not ok:
            st.error(f"REBOUND is not available in this environment: {message}")
            return
        st.caption(
            "Uses the current transiting planet table plus any extra non-transiting perturbers as the initial condition."
        )
        if st.button(
            "Seed non-transiting planet from sinusoidal TTV fit",
            use_container_width=True,
            help="Uses the quick sinusoid super-period and amplitude as an indicative starting point for an extra non-transiting perturber.",
        ):
            _seed_nontransiting_planet_from_sinusoid(planets, mass, radius)
            st.success("Seeded one indicative non-transiting perturber from the sinusoidal TTV fit.")
            st.rerun()
        nontransiting = _nontransiting_planet_editor(planets, mass, radius)
        dynamics_planets = coerce_planet_table(pd.concat([planets, nontransiting], ignore_index=True))
        time_defaults = []
        for frame in [st.session_state.photometry, st.session_state.rv, timings_for_ttv]:
            if not frame.empty:
                col = "tmid" if "tmid" in frame.columns else "time" if "time" in frame.columns else None
                if col:
                    time_defaults.extend(pd.to_numeric(frame[col], errors="coerce").dropna().tolist())
        if time_defaults:
            default_start = float(min(time_defaults))
            default_end = float(max(time_defaults))
        else:
            default_start = float(planets["t0"].min() - planets["period"].max())
            default_end = float(planets["t0"].max() + 10.0 * planets["period"].max())
        if default_end <= default_start:
            default_end = default_start + float(planets["period"].max() * 10.0)

        setup_cols = st.columns(4)
        reference_time = setup_cols[0].number_input(
            "Reference time",
            value=float(default_start),
            format="%.8f",
            help="Absolute time assigned to the initial REBOUND orbital elements. If phases are initialized from T0, mean anomalies are computed at this time.",
        )
        start_time = setup_cols[1].number_input(
            "Start time",
            value=float(default_start),
            format="%.8f",
            help="First time to integrate and sample. Use the same time system as photometry, RVs, and transit timings.",
        )
        end_time = setup_cols[2].number_input(
            "End time",
            value=float(default_end),
            format="%.8f",
            help="Last time to integrate and sample. Longer ranges reveal more TTV structure but take longer.",
        )
        sample_step = setup_cols[3].number_input(
            "Sample step [days]",
            value=max(float(planets["period"].min()) / 80.0, 0.005),
            min_value=0.0005,
            format="%.5f",
            help="Time spacing used to sample the integration for plots and to bracket transit crossings. Smaller values improve transit finding and eccentric-orbit fidelity but run slower.",
        )

        option_cols = st.columns(4)
        initialize_from_t0 = option_cols[0].checkbox(
            "Initialize phases from T0",
            value=True,
            help="When enabled, the app computes each planet's mean anomaly so its listed T0 is near inferior conjunction at the reference ephemeris. Disable to use mean_anomaly_deg directly.",
        )
        integrator = option_cols[1].selectbox(
            "Integrator",
            ["ias15", "whfast"],
            help="IAS15 is adaptive, high-accuracy, and robust for eccentric or close encounters. WHFast is faster for long, stable, near-Keplerian systems but requires care with timestep choice.",
        )
        grazing_scale = option_cols[2].number_input(
            "Transit search width [stellar radii]",
            value=1.25,
            min_value=0.5,
            max_value=5.0,
            step=0.05,
            help="Maximum sky-projected separation, in stellar radii, accepted as a transit crossing. 1.0 is the stellar limb; values above 1 allow grazing transits and numerical tolerance.",
        )
        compare_planet = option_cols[3].selectbox(
            "Compare planet",
            planets["name"].tolist(),
            key="physical_compare_planet",
            help="Planet whose modeled transit times are compared against the measured timing table in the TTV/O-C plot.",
        )

        if st.button(
            "Run physical REBOUND model",
            use_container_width=True,
            help="Builds a star-plus-planets REBOUND simulation, integrates it over the selected time range, detects model transit crossings, and computes RV/transit/3D outputs.",
        ):
            try:
                result = run_physical_ttv_model(
                    dynamics_planets,
                    star_mass_solar=mass,
                    star_radius_solar=radius,
                    reference_time=reference_time,
                    start_time=start_time,
                    end_time=end_time,
                    sample_step_days=sample_step,
                    initialize_from_t0=initialize_from_t0,
                    integrator=integrator,
                    grazing_scale=grazing_scale,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"REBOUND model failed: {exc}")
            else:
                st.session_state.physical_ttv_result = result
                compare_row = planets.loc[planets["name"] == compare_planet].iloc[0]
                st.session_state.physical_ttv_comparison = compare_model_to_timings(
                    timings_for_ttv,
                    result.model_timings,
                    compare_planet,
                    float(compare_row["t0"]),
                    float(compare_row["period"]),
                )
                st.success(
                    f"Integrated {len(dynamics_planets)} planet(s), found {len(result.model_timings)} model transit crossing(s)."
                )

        result = st.session_state.get("physical_ttv_result")
        comparison = st.session_state.get("physical_ttv_comparison", pd.DataFrame())
        if result is None:
            st.info("Run the physical model to generate TTV, RV, transit, and 3D outputs.")
            return

        plot_tabs = st.tabs(["TTV / O-C", "Host RV", "Transit Model", "3D Orbits", "Tables"])
        with plot_tabs[0]:
            st.plotly_chart(physical_oc_figure(comparison), use_container_width=True, config=PLOT_CONFIG, key="physical_oc_plot")
        with plot_tabs[1]:
            st.plotly_chart(physical_rv_figure(st.session_state.rv, result.rv_curve), use_container_width=True, config=PLOT_CONFIG, key="physical_rv_plot")
        with plot_tabs[2]:
            st.plotly_chart(
                physical_transit_figure(st.session_state.photometry, result.transit_flux),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="physical_transit_plot",
            )
        with plot_tabs[3]:
            phase_index = st.slider(
                "3D trace marker",
                0,
                max(len(result.rv_curve) - 1, 0),
                max(len(result.rv_curve) - 1, 0),
                help="Selects which sampled integration time is marked by planet dots on the integrated 3D orbit traces.",
            )
            st.plotly_chart(
                rebound_3d_figure(result.orbit_trace, phase_index=phase_index),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="physical_3d_plot",
            )
        with plot_tabs[4]:
            table_a, table_b = st.tabs(["Model timings", "Comparison"])
            with table_a:
                st.dataframe(result.model_timings, use_container_width=True)
                st.download_button("Download REBOUND timings", result.model_timings.to_csv(index=False), "rebound_model_timings.csv", "text/csv")
            with table_b:
                st.dataframe(comparison, use_container_width=True)
                st.download_button("Download REBOUND comparison", comparison.to_csv(index=False), "rebound_ttv_comparison.csv", "text/csv")


def model_3d_tab() -> None:
    st.subheader("3D Multi-Planet System Model")
    st.caption("Render the current planet table as a shared transit/RV/orbital parameter set.")
    planets = planet_editor("model3d")
    phase = st.slider(
        "Display orbital phase",
        0.0,
        1.0,
        0.0,
        0.005,
        help="Common display phase for the lightweight geometric preview. This does not change fitted parameters.",
    )
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
    st.title("Simplified TTV Fitter")
    batch_workflow_tab()


if __name__ == "__main__":
    main()
