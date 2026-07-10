"""Standard photometry transit fitting page."""

from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import batman

from app.allesfitter_config import PARAM_COLUMNS, csv_bytes
from app.allesfitter_config import render_config_editors, render_context_controls, render_docs_note
from app.exofop import cached_toi_info, extract_tic_id, find_toi_parameters, load_toi_table, preferred_toi_columns
from app.plots import (
    PLOT_CONFIG,
    mcmc_chain_plot,
    mcmc_corner_plot,
    photometry_model_overlay,
    prepared_photometry_preview,
)
from app.ui import page_header, pending_panel


REQUIRED_PHOTOMETRY_COLUMNS = ["time", "flux", "flux_err"]
R_EARTH_OVER_R_SUN = 0.0091577
R_SUN_AU = 0.00465047
FIT_ROOT = Path("data") / "fits"
EXOFOP_UNIFORM_SIGMA_MULTIPLIER = 50
TESS_TIME_OFFSET = 2457000.0


def _finite_float(value, fallback: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _prior_bounds(prior_type: str, value: float, sigma: float, lower: float, upper: float) -> str:
    value = _finite_float(value, 0.0)
    sigma = _finite_float(sigma, float("nan"))
    lower = _finite_float(lower, float("nan"))
    upper = _finite_float(upper, float("nan"))
    if prior_type == "Gaussian":
        sigma = abs(sigma) if math.isfinite(sigma) and sigma != 0 else max(abs(value) * 0.05, 1e-6)
        return f"normal {value:.10g} {sigma:.10g}"
    lower = lower if math.isfinite(lower) else value - max(abs(value) * 0.2, 1e-6)
    upper = upper if math.isfinite(upper) else value + max(abs(value) * 0.2, 1e-6)
    if lower > upper:
        lower, upper = upper, lower
    return f"uniform {lower:.10g} {upper:.10g}"


def _init_err_from_bounds(value: float, bounds: str) -> float:
    value = _finite_float(value, 0.0)
    parts = str(bounds).split()
    try:
        if parts[0] == "uniform" and len(parts) >= 3:
            width = abs(float(parts[2]) - float(parts[1]))
            return max(width * 1e-4, 1e-10)
        if parts[0] in {"normal", "trunc_normal"}:
            sigma = abs(float(parts[-1]))
            return max(sigma * 1e-3, 1e-10)
    except (TypeError, ValueError, IndexError):
        pass
    return max(abs(value) * 1e-8, 1e-10)


def _param_row(name: str, value: object, fit: int, bounds: str, label: str, unit: str = "") -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "fit": fit,
        "bounds": bounds,
        "label": label,
        "unit": unit,
        "coupled_with": "",
        "truth": "",
        "init_err": _init_err_from_bounds(_finite_float(value, 0.0), bounds) if int(fit) else "",
    }


def _kepler_a_over_rstar(period_days: float, stellar_mass: float, stellar_radius: float) -> float:
    if period_days <= 0 or stellar_mass <= 0 or stellar_radius <= 0:
        return float("nan")
    period_years = period_days / 365.25
    semi_major_axis_au = (stellar_mass * period_years**2) ** (1.0 / 3.0)
    return semi_major_axis_au / (stellar_radius * R_SUN_AU)


def _default_planet_rows(matches: pd.DataFrame, planet_count: int, prior_type: str) -> pd.DataFrame:
    rows = []
    for idx in range(planet_count):
        source = matches.iloc[idx] if matches is not None and idx < len(matches) else pd.Series(dtype=object)
        companion = chr(ord("b") + idx)
        period = _finite_float(source.get("Period (days)", 1.0), 1.0)
        epoch = _finite_float(source.get("Epoch (BJD)", 0.0), 0.0)
        period_err = _finite_float(source.get("Period (days) err", np.nan), max(period * 0.001, 1e-5))
        epoch_err = _finite_float(source.get("Epoch (BJD) err", np.nan), 0.01)
        stellar_radius = _finite_float(source.get("Stellar Radius (R_Sun)", np.nan), 1.0)
        planet_radius = _finite_float(source.get("Planet Radius (R_Earth)", np.nan), np.nan)
        radius_ratio = planet_radius * R_EARTH_OVER_R_SUN / stellar_radius if np.isfinite(planet_radius) and stellar_radius > 0 else 0.1
        depth_ppm = _finite_float(source.get("Depth (ppm)", np.nan), np.nan)
        if not np.isfinite(radius_ratio) and np.isfinite(depth_ppm):
            radius_ratio = math.sqrt(max(depth_ppm, 0) / 1_000_000.0)
        if not np.isfinite(radius_ratio):
            radius_ratio = 0.1

        rows.extend(
            [
                {
                    "planet": companion,
                    "parameter": "T0",
                    "value": epoch,
                    "prior_type": prior_type,
                    "sigma": epoch_err,
                    "lower": epoch - EXOFOP_UNIFORM_SIGMA_MULTIPLIER * epoch_err,
                    "upper": epoch + EXOFOP_UNIFORM_SIGMA_MULTIPLIER * epoch_err,
                    "fit": True,
                },
                {
                    "planet": companion,
                    "parameter": "period",
                    "value": period,
                    "prior_type": prior_type,
                    "sigma": period_err,
                    "lower": max(period - EXOFOP_UNIFORM_SIGMA_MULTIPLIER * period_err, 1e-6),
                    "upper": period + EXOFOP_UNIFORM_SIGMA_MULTIPLIER * period_err,
                    "fit": True,
                },
                {
                    "planet": companion,
                    "parameter": "radius_ratio",
                    "value": radius_ratio,
                    "prior_type": "Uniform",
                    "sigma": max(radius_ratio * 0.05, 1e-4),
                    "lower": max(radius_ratio * 0.5, 1e-5),
                    "upper": min(radius_ratio * 1.5, 1.0),
                    "fit": True,
                },
                {
                    "planet": companion,
                    "parameter": "impact",
                    "value": 0.5,
                    "prior_type": "Uniform",
                    "sigma": 0.1,
                    "lower": 0.0,
                    "upper": 1.2,
                    "fit": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def _flux_error_scale_from_data(data: pd.DataFrame | None) -> float:
    if data is None or data.empty:
        return 1e-3
    if "flux_err" in data.columns:
        flux_err = pd.to_numeric(data["flux_err"], errors="coerce").to_numpy(dtype=float)
        finite_err = flux_err[np.isfinite(flux_err) & (flux_err > 0)]
        if finite_err.size:
            return float(np.nanmedian(finite_err))
    if "flux" in data.columns:
        flux = pd.to_numeric(data["flux"], errors="coerce").to_numpy(dtype=float)
        finite_flux = flux[np.isfinite(flux)]
        if finite_flux.size:
            scatter = 1.4826 * np.nanmedian(np.abs(finite_flux - np.nanmedian(finite_flux)))
            if np.isfinite(scatter) and scatter > 0:
                return float(scatter)
    return 1e-3


def _default_ln_err_row(data: pd.DataFrame | None) -> dict[str, object]:
    sigma_flux = max(_flux_error_scale_from_data(data), 1e-12)
    ln_sigma = float(np.log(sigma_flux))
    return {
        "parameter": "ln_err_flux",
        "value": ln_sigma,
        "prior_type": "Uniform",
        "sigma": 0.5,
        "lower": float(np.log(max(sigma_flux / 10.0, 1e-12))),
        "upper": float(np.log(sigma_flux * 10.0)),
        "fit": False,
    }


def _default_nuisance_rows(phot_inst: str, data: pd.DataFrame | None = None, ld_space: str = "u") -> pd.DataFrame:
    ld_space = "u" if str(ld_space).lower().startswith("u") else "q"
    if ld_space == "u":
        ld_rows = [
            {
                "parameter": "host_ldc_u1",
                "value": 0.5,
                "prior_type": "Uniform",
                "sigma": 0.05,
                "lower": 0.0,
                "upper": 1.0,
                "fit": False,
            },
            {
                "parameter": "host_ldc_u2",
                "value": 0.1,
                "prior_type": "Uniform",
                "sigma": 0.05,
                "lower": 0.0,
                "upper": 1.0,
                "fit": False,
            },
        ]
    else:
        ld_rows = [
            {
                "parameter": "host_ldc_q1",
                "value": 0.36,
                "prior_type": "Uniform",
                "sigma": 0.1,
                "lower": 0.0,
                "upper": 1.0,
                "fit": False,
            },
            {
                "parameter": "host_ldc_q2",
                "value": 0.4166666667,
                "prior_type": "Uniform",
                "sigma": 0.1,
                "lower": 0.0,
                "upper": 1.0,
                "fit": False,
            },
        ]
    return pd.DataFrame(
        ld_rows
        + [
            _default_ln_err_row(data),
            {
                "parameter": "baseline_offset_flux",
                "value": 0.0,
                "prior_type": "Gaussian",
                "sigma": 0.01,
                "lower": -0.1,
                "upper": 0.1,
                "fit": True,
            },
            {
                "parameter": "dil",
                "value": 0.0,
                "prior_type": "Uniform",
                "sigma": 0.1,
                "lower": 0.0,
                "upper": 1.0,
                "fit": False,
            },
        ]
    )


def _planet_prior_column_config() -> dict[str, object]:
    return {
        "planet": st.column_config.TextColumn("planet", help="allesfitter companion label, usually b, c, d..."),
        "parameter": st.column_config.SelectboxColumn(
            "parameter",
            options=["T0", "period", "radius_ratio", "impact"],
            help="User-facing transit parameter. impact is converted to cosi for allesfitter.",
        ),
        "prior_type": st.column_config.SelectboxColumn("prior_type", options=["Gaussian", "Uniform"]),
        "value": st.column_config.NumberColumn("value", format="%.10g"),
        "sigma": st.column_config.NumberColumn("sigma", format="%.10g"),
        "lower": st.column_config.NumberColumn("lower", format="%.10g"),
        "upper": st.column_config.NumberColumn("upper", format="%.10g"),
        "fit": st.column_config.CheckboxColumn("fit"),
    }


def _planet_table_value(table: pd.DataFrame, planet: str, parameter: str, default: float) -> float:
    match = table[(table["planet"] == planet) & (table["parameter"] == parameter)]
    if match.empty:
        return default
    return _finite_float(match.iloc[0]["value"], default)


def _planet_table_row(table: pd.DataFrame, planet: str, parameter: str) -> pd.Series | None:
    match = table[(table["planet"] == planet) & (table["parameter"] == parameter)]
    if match.empty:
        return None
    return match.iloc[0]


def _build_allesfitter_params_from_setup(
    planet_table: pd.DataFrame,
    nuisance_table: pd.DataFrame,
    stellar_mass: float,
    stellar_radius: float,
    phot_inst: str,
    ld_space: str = "u",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    derived_rows = []
    for planet in list(dict.fromkeys(planet_table["planet"].astype(str))):
        subset = planet_table[planet_table["planet"].astype(str) == planet]
        period = _planet_table_value(subset, planet, "period", 1.0)
        rr = _planet_table_value(subset, planet, "radius_ratio", 0.1)
        impact = _planet_table_value(subset, planet, "impact", 0.5)
        period_row = _planet_table_row(subset, planet, "period")
        rr_row = _planet_table_row(subset, planet, "radius_ratio")
        impact_row = _planet_table_row(subset, planet, "impact")
        a_over_rstar = _kepler_a_over_rstar(period, stellar_mass, stellar_radius)
        rsuma = (1.0 + rr) / a_over_rstar if np.isfinite(a_over_rstar) and a_over_rstar > 0 else 0.1
        cosi = impact / a_over_rstar if np.isfinite(a_over_rstar) and a_over_rstar > 0 else 0.02
        if impact_row is not None and np.isfinite(a_over_rstar) and a_over_rstar > 0:
            cosi_bounds = _prior_bounds(
                impact_row["prior_type"],
                cosi,
                _finite_float(impact_row["sigma"], 0.1) / a_over_rstar,
                _finite_float(impact_row["lower"], 0.0) / a_over_rstar,
                _finite_float(impact_row["upper"], 1.2) / a_over_rstar,
            )
            cosi_fit = int(bool(impact_row["fit"]))
        else:
            cosi_bounds = "uniform 0 1"
            cosi_fit = 0
        geometry_is_sampled = bool(
            (period_row is not None and bool(period_row.get("fit")))
            or (rr_row is not None and bool(rr_row.get("fit")))
        )
        rsuma_sigma = max(abs(rsuma) * 0.1, 1e-4)
        rsuma_bounds = _prior_bounds(
            "Uniform",
            rsuma,
            rsuma_sigma,
            max(rsuma - 5.0 * rsuma_sigma, 1e-6),
            min(rsuma + 5.0 * rsuma_sigma, 1.0),
        )

        derived_rows.append(
            {
                "planet": planet,
                "a_over_Rstar": a_over_rstar,
                "allesfitter_rsuma": rsuma,
                "allesfitter_cosi": cosi,
            }
        )

        param_map = {
            "T0": (f"{planet}_epoch", "d", f"{planet} epoch"),
            "period": (f"{planet}_period", "d", f"{planet} period"),
            "radius_ratio": (f"{planet}_rr", "", f"{planet} radius ratio"),
        }
        for _, item in subset.iterrows():
            parameter = item["parameter"]
            if parameter not in param_map:
                continue
            name, unit, label = param_map[parameter]
            bounds = _prior_bounds(item["prior_type"], item["value"], item["sigma"], item["lower"], item["upper"])
            rows.append(
                _param_row(
                    name,
                    item["value"],
                    int(bool(item["fit"])),
                    bounds,
                    label,
                    unit,
                )
            )
        rows.extend(
            [
                _param_row(f"{planet}_rsuma", rsuma, int(geometry_is_sampled), rsuma_bounds, f"{planet} (Rstar+Rp)/a from Kepler setup"),
                _param_row(f"{planet}_cosi", cosi, cosi_fit, cosi_bounds, f"{planet} cos inclination from impact setup"),
                _param_row(f"{planet}_f_c", 0.0, 0, "uniform -1 1", f"{planet} sqrt(e) cos omega"),
                _param_row(f"{planet}_f_s", 0.0, 0, "uniform -1 1", f"{planet} sqrt(e) sin omega"),
            ]
        )

    ld_space = "u" if str(ld_space).lower().startswith("u") else "q"
    if ld_space == "u":
        limb_names = {
            "host_ldc_u1": (f"host_ldc_u1_{phot_inst}", f"host u1 {phot_inst}"),
            "host_ldc_u2": (f"host_ldc_u2_{phot_inst}", f"host u2 {phot_inst}"),
        }
    else:
        limb_names = {
            "host_ldc_q1": (f"host_ldc_q1_{phot_inst}", f"host q1 {phot_inst}"),
            "host_ldc_q2": (f"host_ldc_q2_{phot_inst}", f"host q2 {phot_inst}"),
        }
    nuisance_names = {
        **limb_names,
        "ln_err_flux": (f"ln_err_flux_{phot_inst}", f"flux error scale {phot_inst}"),
        "baseline_offset_flux": (f"baseline_offset_flux_{phot_inst}", f"baseline flux {phot_inst} offset"),
        "dil": (f"dil_{phot_inst}", f"dilution {phot_inst}"),
    }
    for _, item in nuisance_table.iterrows():
        parameter = item["parameter"]
        if parameter not in nuisance_names:
            continue
        name, label = nuisance_names[parameter]
        bounds = _prior_bounds(item["prior_type"], item["value"], item["sigma"], item["lower"], item["upper"])
        rows.append(
            _param_row(
                name,
                item["value"],
                int(bool(item["fit"])),
                bounds,
                label,
            )
        )
    return pd.DataFrame(rows, columns=PARAM_COLUMNS), pd.DataFrame(derived_rows)


def _with_data_seeded_ln_err(nuisance_table: pd.DataFrame, data: pd.DataFrame | None) -> pd.DataFrame:
    table = nuisance_table.copy()
    seed = _default_ln_err_row(data)
    match = table["parameter"].astype(str) == "ln_err_flux" if "parameter" in table.columns else pd.Series(False, index=table.index)
    if match.any():
        idx = table.index[match][0]
        current = _finite_float(table.at[idx, "value"], float("nan"))
        current_lower = _finite_float(table.at[idx, "lower"], float("nan"))
        current_upper = _finite_float(table.at[idx, "upper"], float("nan"))
        still_placeholder = (
            not np.isfinite(current)
            or abs(current) < 1e-12
            or (abs(current_lower + 10.0) < 1e-12 and abs(current_upper - 10.0) < 1e-12)
        )
        if still_placeholder:
            for key, value in seed.items():
                if key in table.columns:
                    table.at[idx, key] = value
    else:
        table = pd.concat([table, pd.DataFrame([seed])], ignore_index=True)
    return table


def _target_filename_part() -> str:
    target = st.session_state.get("target_name") or st.session_state.get("photometry_target_input") or "target"
    cleaned = "".join(str(target).split())
    return cleaned or "target"


def _settings_from_fit_setup(
    companions: list[str],
    phot_inst: str,
    sampler: str,
    shift_epoch: bool,
    fast_fit: bool,
    sampler_controls: dict[str, object],
    ld_space: str = "u",
) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [
        ("companions_phot", " ".join(companions)),
        ("companions_rv", ""),
        ("inst_phot", phot_inst),
        ("inst_rv", ""),
        ("multiprocess", str(bool(sampler_controls.get("multiprocess", False)))),
        ("multiprocess_cores", int(sampler_controls.get("multiprocess_cores", 1))),
        ("print_progress", "True"),
        ("time_format", "BJD_TDB"),
        ("shift_epoch", str(bool(shift_epoch))),
        ("fast_fit", str(bool(fast_fit))),
        ("fast_fit_width", 0.333333),
        ("secondary_eclipse", "False"),
        ("phase_curve", "False"),
        ("fit_ttvs", "False"),
        (f"host_ld_law_{phot_inst}", "quad"),
        (f"host_ld_space_{phot_inst}", "u" if str(ld_space).lower().startswith("u") else "q"),
        (f"baseline_flux_{phot_inst}", "sample_offset"),
        (f"error_flux_{phot_inst}", "sample"),
    ]
    for companion in companions:
        rows.extend(
            [
                (f"{companion}_ld_law_{phot_inst}", "None"),
                (f"inst_for_{companion}_epoch", phot_inst),
            ]
        )
    if sampler == "MCMC":
        rows.extend(
            [
                ("mcmc_nwalkers", int(sampler_controls.get("mcmc_nwalkers", 64))),
                ("mcmc_total_steps", int(sampler_controls.get("mcmc_total_steps", 2000))),
                ("mcmc_burn_steps", int(sampler_controls.get("mcmc_burn_steps", 1000))),
                ("mcmc_thin_by", int(sampler_controls.get("mcmc_thin_by", 1))),
                ("mcmc_pre_run_loops", int(sampler_controls.get("mcmc_pre_run_loops", 0))),
                ("mcmc_pre_run_steps", int(sampler_controls.get("mcmc_pre_run_steps", 0))),
                ("mcmc_initialization", sampler_controls.get("mcmc_initialization", "uniform")),
            ]
        )
    else:
        rows.extend(
            [
                ("ns_modus", sampler_controls.get("ns_modus", "dynamic")),
                ("ns_nlive", int(sampler_controls.get("ns_nlive", 300))),
                ("ns_tol", float(sampler_controls.get("ns_tol", 0.1))),
                ("ns_bound", sampler_controls.get("ns_bound", "single")),
                ("ns_sample", sampler_controls.get("ns_sample", "rwalk")),
            ]
        )
    return pd.DataFrame(rows, columns=["name", "value"])


def _validate_fit_inputs(data: pd.DataFrame | None, params: pd.DataFrame | None, phot_inst: str, ld_space: str = "u") -> list[str]:
    issues = []
    if data is None or data.empty:
        issues.append("Load photometry data first.")
    else:
        missing_data = [column for column in REQUIRED_PHOTOMETRY_COLUMNS if column not in data.columns]
        if missing_data:
            issues.append(f"Photometry data is missing: {', '.join(missing_data)}.")
        if (data.get("flux_err", pd.Series(dtype=float)) <= 0).any():
            issues.append("Photometry uncertainties must all be positive.")
    if params is None or params.empty:
        issues.append("Generate the allesfitter params table first.")
    else:
        missing_param_cols = [column for column in PARAM_COLUMNS if column not in params.columns]
        if missing_param_cols:
            issues.append(f"params.csv is missing columns: {', '.join(missing_param_cols)}.")
        if str(ld_space).lower().startswith("u"):
            ld_patterns = [f"host_ldc_u1_{phot_inst}", f"host_ldc_u2_{phot_inst}"]
        else:
            ld_patterns = [f"host_ldc_q1_{phot_inst}", f"host_ldc_q2_{phot_inst}"]
        required_patterns = ld_patterns + [f"dil_{phot_inst}", f"ln_err_flux_{phot_inst}", f"baseline_offset_flux_{phot_inst}"]
        param_names = set(params["name"].astype(str)) if "name" in params else set()
        for name in required_patterns:
            if name not in param_names:
                issues.append(f"Missing required/generated photometry parameter `{name}`.")
        companions = sorted({name.split("_", 1)[0] for name in param_names if "_" in name and name.split("_", 1)[0] not in {"host", "dil", "ln", "baseline"}})
        for companion in companions:
            for suffix in ["rr", "rsuma", "cosi", "epoch", "period", "f_c", "f_s"]:
                name = f"{companion}_{suffix}"
                if name not in param_names:
                    issues.append(f"Missing companion parameter `{name}`.")
    return issues


def _bounds_width(bounds: object) -> float:
    parts = str(bounds).split()
    if len(parts) >= 3 and parts[0] == "uniform":
        try:
            return abs(float(parts[2]) - float(parts[1]))
        except ValueError:
            return float("nan")
    if len(parts) >= 3 and parts[0] == "normal":
        try:
            return 2.0 * abs(float(parts[2]))
        except ValueError:
            return float("nan")
    if len(parts) >= 5 and parts[0] == "trunc_normal":
        try:
            return abs(float(parts[2]) - float(parts[1]))
        except ValueError:
            return float("nan")
    return float("nan")


def _prior_sanity_warnings(params: pd.DataFrame | None, *, shift_epoch: bool) -> list[str]:
    if params is None or params.empty or "name" not in params or "bounds" not in params:
        return []
    warnings: list[str] = []
    for companion in sorted({str(name).split("_", 1)[0] for name in params["name"] if str(name).endswith("_period")}):
        period_row = params[params["name"].astype(str) == f"{companion}_period"]
        epoch_row = params[params["name"].astype(str) == f"{companion}_epoch"]
        if period_row.empty or epoch_row.empty:
            continue
        period_width = _bounds_width(period_row.iloc[0]["bounds"])
        epoch_width = _bounds_width(epoch_row.iloc[0]["bounds"])
        if np.isfinite(period_width) and period_width > 0.002:
            warnings.append(
                f"`{companion}_period` prior width is {period_width:.5g} d. This is valid, but it makes the timing posterior "
                "broader/more multimodal; use pre-runs, more walkers, and longer chains."
            )
        if np.isfinite(epoch_width) and epoch_width > 0.05:
            warnings.append(
                f"`{companion}_epoch` prior width is {epoch_width:.5g} d. This is valid, but the posterior median may not be "
                "a representative transit model until the chain is well mixed."
            )
        if shift_epoch and np.isfinite(period_width) and period_width > 0.001:
            warnings.append(
                "With `shift_epoch` enabled, allesfitter shifts epoch bounds using the period bounds; broad period bounds "
                "can make the midpoint epoch prior broader than the original epoch row alone suggests."
            )
    return warnings


def _write_fit_directory(
    data: pd.DataFrame,
    params: pd.DataFrame,
    settings: pd.DataFrame,
    phot_inst: str,
    *,
    time_offset: float,
) -> Path:
    target = _target_filename_part()
    fit_dir = FIT_ROOT / target / "photometry_transit_fit"
    fit_dir.mkdir(parents=True, exist_ok=True)

    clean = data.loc[:, ["time", "flux", "flux_err"]].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna().sort_values("time")
    clean["time"] = clean["time"].astype(float) + time_offset
    clean.to_csv(fit_dir / f"{phot_inst}.csv", index=False, header=False)
    safe_params = params.copy()
    for column in ["label", "unit", "coupled_with", "truth", "bounds"]:
        if column in safe_params.columns:
            safe_params[column] = safe_params[column].astype(str).str.replace(",", " ", regex=False)
            safe_params[column] = safe_params[column].replace("nan", "")
    safe_params.to_csv(fit_dir / "params.csv", index=False)
    settings.to_csv(fit_dir / "settings.csv", index=False, header=False)

    notes = [
        "allesfitter workbench generated fit directory",
        f"photometry instrument: {phot_inst}",
        f"data time offset applied before writing {phot_inst}.csv: {time_offset:.0f}",
        "TESS offset note: TESS light-curve times are BTJD, so +2457000 is applied for BJD_TDB when instrument is TESS.",
    ]
    (fit_dir / "WORKBENCH_NOTES.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return fit_dir


def _archive_previous_sampler_outputs(fit_dir: Path, sampler: str) -> Path | None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = fit_dir / "results" / "archived_runs" / f"{sampler.lower().replace(' ', '_')}_{timestamp}"
    candidates: list[Path] = []
    if sampler == "MCMC":
        candidates.extend([fit_dir / "mcmc_run.log", fit_dir / "results" / "mcmc_save.h5"])
        candidates.extend((fit_dir / "results").glob("mcmc_*"))
    else:
        candidates.extend([fit_dir / "nested_sampling_run.log"])
        candidates.extend((fit_dir / "results").glob("ns_*"))
        candidates.extend((fit_dir / "results").glob("save_ns*"))

    moved = False
    for path in candidates:
        if not path.exists() or archive_dir in path.parents:
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / path.name
        if destination.exists():
            destination = archive_dir / f"{path.stem}_{timestamp}{path.suffix}"
        shutil.move(str(path), str(destination))
        moved = True
    return archive_dir if moved else None


def _launch_sampler(fit_dir: Path, sampler: str, *, initial_guess_first: bool, fresh_start: bool) -> tuple[subprocess.Popen, Path | None]:
    archive_dir = _archive_previous_sampler_outputs(fit_dir, sampler) if fresh_start else None
    if sampler == "MCMC":
        call = "allesfitter.mcmc_fit(datadir)"
    else:
        call = "allesfitter.ns_fit(datadir)"
    initial_guess_call = "allesfitter.show_initial_guess(datadir, quiet=True, do_plot=True);" if initial_guess_first else ""
    code = f"import numpy as np; np.float=float; np.int=int; import allesfitter; datadir={str(fit_dir)!r}; {initial_guess_call} {call}"
    log_path = fit_dir / ("mcmc_run.log" if sampler == "MCMC" else "nested_sampling_run.log")
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(Path.cwd() / "conda-allesfitter" / "lib")
    env["MPLCONFIGDIR"] = str(Path.cwd() / ".matplotlib")
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(Path.cwd() / "conda-allesfitter" / "bin" / "python"), "-c", code],
        cwd=str(Path.cwd()),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return process, archive_dir


def _process_is_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _stop_sampler(pid: int | None) -> tuple[bool, str]:
    if not pid:
        return False, "No sampler process is recorded."
    try:
        os.kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        return False, "That sampler process has already stopped."
    except OSError as exc:
        return False, f"Could not stop sampler process: {exc}"
    return True, f"Sent stop signal to sampler PID {pid}."


def _read_log_tail(log_path: str | None, lines: int = 25) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _mcmc_backend_path(fit_dir: str | Path | None) -> Path | None:
    if not fit_dir:
        return None
    return Path(fit_dir) / "results" / "mcmc_save.h5"


def _mcmc_labels(params: pd.DataFrame | None, expected: int | None = None) -> list[str]:
    if params is None or params.empty or "fit" not in params or "name" not in params:
        labels: list[str] = []
    else:
        fit_mask = params["fit"].astype(str).str.lower().isin(["1", "true", "yes"])
        labels = params.loc[fit_mask, "name"].astype(str).tolist()
    if expected is not None and len(labels) != expected:
        labels = [f"parameter {idx + 1}" for idx in range(expected)]
    return labels


def _fit_directory_labels(fit_dir: str | Path | None, expected: int | None = None) -> list[str]:
    if not fit_dir:
        return []
    params_path = Path(fit_dir) / "params.csv"
    if not params_path.exists():
        return []
    try:
        params = pd.read_csv(params_path)
    except Exception:
        return []
    labels = _mcmc_labels(params, None)
    if expected is not None and len(labels) != expected:
        return []
    return labels


def _mcmc_iteration(fit_dir: str | Path | None) -> int | None:
    backend_path = _mcmc_backend_path(fit_dir)
    if backend_path is None or not backend_path.exists():
        return None
    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        import emcee

        reader = emcee.backends.HDFBackend(str(backend_path), read_only=True)
        return int(reader.iteration)
    except (BlockingIOError, OSError):
        return None
    except Exception:
        return None


def _sampler_is_active(
    pid: int | None,
    sampler: str,
    fit_dir: str | Path | None,
    total_steps: int,
    launched_total_steps: int | None = None,
) -> bool:
    if not _process_is_running(pid):
        return False
    if sampler == "MCMC":
        iteration = _mcmc_iteration(fit_dir)
        target_steps = int(launched_total_steps or total_steps)
        if iteration is not None and iteration >= target_steps:
            return False
    return True


def _load_mcmc_chain(
    fit_dir: str | Path | None,
    burn_steps: int,
    thin_by: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str], str | None]:
    backend_path = _mcmc_backend_path(fit_dir)
    if backend_path is None or not backend_path.exists():
        return None, None, None, [], "MCMC backend file is not available yet."
    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        import emcee

        reader = emcee.backends.HDFBackend(str(backend_path), read_only=True)
        iteration = int(reader.iteration)
        discard = max(0, min(int(burn_steps), max(iteration - 1, 0)))
        thin = max(1, int(thin_by))
        chain = reader.get_chain(discard=discard, thin=thin, flat=False)
        samples = reader.get_chain(discard=discard, thin=thin, flat=True)
        log_prob = reader.get_log_prob(discard=discard, thin=thin, flat=True)
        labels = _mcmc_labels(st.session_state.get("phot_fit_generated_params"), chain.shape[2])
        if not labels or all(label.startswith("parameter ") for label in labels):
            labels = _fit_directory_labels(fit_dir, chain.shape[2]) or labels
        return chain, samples, log_prob, labels, None
    except Exception as exc:
        return None, None, None, [], f"Could not read MCMC samples: {exc}"


def _load_full_mcmc_chain(
    fit_dir: str | Path | None,
) -> tuple[np.ndarray | None, list[str], str | None]:
    backend_path = _mcmc_backend_path(fit_dir)
    if backend_path is None or not backend_path.exists():
        return None, [], "MCMC backend file is not available yet."
    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        import emcee

        reader = emcee.backends.HDFBackend(str(backend_path), read_only=True)
        chain = reader.get_chain(discard=0, thin=1, flat=False)
        labels = _mcmc_labels(st.session_state.get("phot_fit_generated_params"), chain.shape[2])
        if not labels or all(label.startswith("parameter ") for label in labels):
            labels = _fit_directory_labels(fit_dir, chain.shape[2]) or labels
        return chain, labels, None
    except Exception as exc:
        return None, [], f"Could not read full MCMC samples: {exc}"


def _display_labels_and_samples(samples: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    display = np.array(samples, dtype=float, copy=True)
    display_labels = list(labels)
    for idx, label in enumerate(labels):
        if "epoch" in label.lower() and np.nanmedian(display[..., idx]) > TESS_TIME_OFFSET:
            display[..., idx] -= TESS_TIME_OFFSET
            display_labels[idx] = f"{label} [BTJD]"
    return display, display_labels


def _corner_timing_offset_display(samples: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    display = np.array(samples, dtype=float, copy=True)
    display_labels = list(labels)
    for idx, label in enumerate(labels):
        lowered = label.lower()
        if "epoch" in lowered or "period" in lowered:
            center = float(np.nanmedian(display[..., idx]))
            display[..., idx] = (display[..., idx] - center) * 24.0 * 60.0
            display_labels[idx] = f"{label} - median [min]"
    return display, display_labels


def _posterior_summary(samples: np.ndarray, labels: list[str]) -> pd.DataFrame:
    display_samples, display_labels = _display_labels_and_samples(samples, labels)
    rows = []
    for idx, label in enumerate(display_labels):
        lower, median, upper = np.nanpercentile(display_samples[:, idx], [15.87, 50.0, 84.13])
        rows.append(
            {
                "parameter": label,
                "median": median,
                "lower_1sigma": lower,
                "upper_1sigma": upper,
                "minus": median - lower,
                "plus": upper - median,
            }
        )
    return pd.DataFrame(rows)


def _fit_directory_instrument(fit_dir: str | Path | None) -> str:
    if not fit_dir:
        return "TESS"
    settings_path = Path(fit_dir) / "settings.csv"
    if not settings_path.exists():
        return "TESS"
    try:
        settings = pd.read_csv(settings_path, header=None, names=["name", "value"])
    except Exception:
        return "TESS"
    match = settings[settings["name"].astype(str) == "inst_phot"]
    if match.empty:
        return "TESS"
    return str(match.iloc[0]["value"]).split()[0]


def _fit_directory_setting(fit_dir: str | Path | None, key: str, default: str = "") -> str:
    if not fit_dir:
        return default
    settings_path = Path(fit_dir) / "settings.csv"
    if not settings_path.exists():
        return default
    try:
        settings = pd.read_csv(settings_path, header=None, names=["name", "value"])
    except Exception:
        return default
    match = settings[settings["name"].astype(str) == key]
    if match.empty:
        return default
    return str(match.iloc[0]["value"])


def _load_fit_photometry_for_plot(fit_dir: str | Path | None) -> tuple[pd.DataFrame | None, np.ndarray | None, str]:
    if not fit_dir:
        return None, None, "TESS"
    fit_path = Path(fit_dir)
    phot_inst = _fit_directory_instrument(fit_path)
    data_path = fit_path / f"{phot_inst}.csv"
    if not data_path.exists():
        return None, None, phot_inst
    try:
        raw = pd.read_csv(data_path, header=None, names=["time", "flux", "flux_err"])
    except Exception:
        return None, None, phot_inst
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna().sort_values("time")
    model_time = raw["time"].to_numpy(dtype=float)
    plot_data = raw.copy()
    if phot_inst.upper() == "TESS" and np.nanmedian(plot_data["time"]) > TESS_TIME_OFFSET:
        plot_data["time"] = plot_data["time"].astype(float) - TESS_TIME_OFFSET
    return plot_data, model_time, phot_inst


def _params_value_map(fit_dir: str | Path | None) -> dict[str, float]:
    if not fit_dir:
        return {}
    params_path = Path(fit_dir) / "params.csv"
    if not params_path.exists():
        return {}
    try:
        params = pd.read_csv(params_path)
    except Exception:
        return {}
    values: dict[str, float] = {}
    for _, row in params.iterrows():
        values[str(row.get("name", ""))] = _finite_float(row.get("value"), float("nan"))
    return values


def _point_summary(values: np.ndarray, labels: list[str], value_label: str = "value") -> pd.DataFrame:
    display_values, display_labels = _display_labels_and_samples(np.asarray(values, dtype=float).reshape(1, -1), labels)
    return pd.DataFrame({"parameter": display_labels, value_label: display_values[0]})


def _model_from_parameter_values(
    fit_dir: str | Path | None,
    parameter_values: np.ndarray,
    labels: list[str],
) -> tuple[pd.DataFrame | None, np.ndarray | None, str | None]:
    plot_data, model_time, phot_inst = _load_fit_photometry_for_plot(fit_dir)
    if plot_data is None or model_time is None or plot_data.empty:
        return None, None, "No fit-directory photometry file is available for the best-fit plot."

    values = _params_value_map(fit_dir)
    for idx, label in enumerate(labels):
        values[label] = float(parameter_values[idx])
    phot_inst = _fit_directory_instrument(fit_dir)
    ld_space = _fit_directory_setting(fit_dir, f"host_ld_space_{phot_inst}", "q")
    if str(ld_space).lower().startswith("u"):
        limb_darkening = [
            values.get(f"host_ldc_u1_{phot_inst}", 0.5),
            values.get(f"host_ldc_u2_{phot_inst}", 0.1),
        ]
    else:
        q_table = pd.DataFrame(
            [
                {"parameter": "host_ldc_q1", "value": values.get(f"host_ldc_q1_{phot_inst}", 0.36)},
                {"parameter": "host_ldc_q2", "value": values.get(f"host_ldc_q2_{phot_inst}", 0.4166666667)},
            ]
        )
        limb_darkening = _limb_darkening_from_table(q_table)

    companions = sorted({name.split("_", 1)[0] for name in values if name.endswith("_period")})
    if not companions:
        return plot_data, None, "No companion period parameters were found in params.csv."

    model = np.ones_like(model_time, dtype=float)
    for companion in companions:
        period = values.get(f"{companion}_period")
        epoch = values.get(f"{companion}_epoch")
        rr = values.get(f"{companion}_rr")
        rsuma = values.get(f"{companion}_rsuma")
        cosi = values.get(f"{companion}_cosi")
        if not all(np.isfinite([period, epoch, rr, rsuma, cosi])):
            continue
        a_over_rstar = (1.0 + rr) / rsuma if rsuma and rsuma > 0 else float("nan")
        if not np.isfinite(a_over_rstar) or a_over_rstar <= 0:
            continue
        transit = batman.TransitParams()
        transit.t0 = epoch
        transit.per = period
        transit.rp = rr
        transit.a = a_over_rstar
        transit.inc = math.degrees(math.acos(float(np.clip(cosi, -1.0, 1.0))))
        transit.ecc = 0.0
        transit.w = 90.0
        transit.u = limb_darkening
        transit.limb_dark = "quadratic"
        model *= batman.TransitModel(transit, model_time).light_curve(transit)
    return plot_data, model, None


def _render_mcmc_diagnostics(fit_dir: str | Path | None, burn_steps: int, thin_by: int, key_prefix: str) -> None:
    chain, samples, log_prob, labels, error = _load_mcmc_chain(fit_dir, int(burn_steps), int(thin_by))
    if error:
        st.info(error)
        return
    if chain is None or samples is None or log_prob is None:
        return

    display_chain, display_labels = _display_labels_and_samples(chain, labels)
    display_samples, _ = _display_labels_and_samples(samples, labels)
    full_chain, full_labels, full_error = _load_full_mcmc_chain(fit_dir)
    if full_chain is not None:
        display_full_chain, display_full_labels = _display_labels_and_samples(full_chain, full_labels)
    else:
        display_full_chain, display_full_labels = None, []
    corner_samples, corner_labels = _corner_timing_offset_display(display_samples, display_labels)
    summary = _posterior_summary(samples, labels)
    median_sample = np.nanmedian(samples, axis=0)
    median_fit_data, median_fit_model, median_fit_error = _model_from_parameter_values(fit_dir, median_sample, labels)

    st.caption(f"Loaded {samples.shape[0]:,} flattened posterior samples from {chain.shape[1]:,} walkers.")
    full_chains_tab, chains_tab, corner_tab, median_tab = st.tabs(
        ["Full chains", "Burn-in trimmed chains", "Corner", "Posterior median"]
    )
    with full_chains_tab:
        st.caption("This view shows the complete saved backend chain, including pre-run/optimisation and burn-in steps when present.")
        if full_error:
            st.info(full_error)
        elif display_full_chain is not None:
            st.plotly_chart(
                mcmc_chain_plot(display_full_chain, display_full_labels, title="Full MCMC chains including pre-trim search"),
                use_container_width=True,
                config=PLOT_CONFIG,
                key=f"{key_prefix}_mcmc_full_chain_plot",
            )
    with chains_tab:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.plotly_chart(
            mcmc_chain_plot(display_chain, display_labels, title="MCMC chains after burn-in trim"),
            use_container_width=True,
            config=PLOT_CONFIG,
            key=f"{key_prefix}_mcmc_chain_plot",
        )
    with corner_tab:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption(
            "For the corner plot, epoch and period axes are shown as minute offsets from their posterior medians. "
            "This avoids absolute-date tick rounding and makes sub-minute structure visible."
        )
        st.plotly_chart(
            mcmc_corner_plot(corner_samples, corner_labels),
            use_container_width=True,
            config=PLOT_CONFIG,
            key=f"{key_prefix}_mcmc_corner_plot",
        )
    with median_tab:
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption("The posterior median can be a poor model if the chain is broad or multimodal.")
        if median_fit_error:
            st.info(median_fit_error)
        elif median_fit_data is not None:
            st.plotly_chart(
                photometry_model_overlay(
                    median_fit_data,
                    median_fit_model,
                    title="Posterior-median transit model on data",
                    model_name="Posterior median model",
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
                key=f"{key_prefix}_median_fit_model",
            )


def _latest_nested_progress(log_tail: str) -> str:
    for line in reversed(log_tail.splitlines()):
        if any(token in line.lower() for token in ["iter", "bound", "eff", "logl", "dlogz", "nc:"]):
            return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
    return ""


@st.fragment(run_every="5s")
def _render_sampler_monitor() -> None:
    fit_dir = st.session_state.get("phot_fit_directory", "")
    sampler = st.session_state.get("phot_fit_last_sampler", st.session_state.get("phot_fit_sampler_local", "MCMC"))
    pid = st.session_state.get("phot_fit_sampler_pid")
    log_path = st.session_state.get("phot_fit_sampler_log", "")
    total_steps = int(st.session_state.get("phot_fit_mcmc_total_steps", 2000))
    burn_steps = int(st.session_state.get("phot_fit_mcmc_burn_steps", 1000))
    thin_by = int(st.session_state.get("phot_fit_mcmc_thin_by", 1))
    log_tail = _read_log_tail(log_path)
    running = _process_is_running(pid)

    if sampler == "MCMC":
        iteration = _mcmc_iteration(fit_dir)
        if iteration is None:
            st.progress(0, text="MCMC waiting for backend file...")
        else:
            fraction = min(max(iteration / max(total_steps, 1), 0.0), 1.0)
            state = "running" if running and iteration < total_steps else "complete or stopped"
            st.progress(fraction, text=f"MCMC {state}: {iteration:,} / {total_steps:,} steps")
            if iteration > burn_steps:
                kept = max(iteration - burn_steps, 0) // max(thin_by, 1)
                st.caption(f"Post burn-in samples available per walker: {kept:,}")
            if running and iteration < total_steps:
                if st.button("Stop running sampler", use_container_width=True, key="phot_fit_monitor_stop_sampler"):
                    stopped, message = _stop_sampler(pid)
                    st.session_state["phot_fit_sampler_stopped"] = True
                    if stopped:
                        st.warning(message)
                    else:
                        st.info(message)
            elif iteration >= total_steps:
                st.success("MCMC finished. Loading posterior diagnostics below.")
                _render_mcmc_diagnostics(fit_dir, burn_steps, thin_by, "phot_fit_monitor")
    else:
        progress = _latest_nested_progress(log_tail)
        state = "running" if running else "complete or stopped"
        if progress:
            st.info(f"Nested sampling {state}: `{progress}`")
        else:
            st.info(f"Nested sampling {state}. Waiting for sampler progress output...")

    if "BlockingIOError" in log_tail and "Resource temporarily unavailable" in log_tail:
        st.warning(
            "The sampler hit an HDF5 file-lock collision while saving. This has been patched for new runs by disabling "
            "HDF5 file locking in both the app and sampler process."
        )
    elif "Traceback" in log_tail or "Error" in log_tail or "ImportError" in log_tail:
        st.error("The sampler log contains an error. Recent log lines are shown below.")
    if log_tail:
        with st.expander("Recent sampler log", expanded=False):
            st.code(log_tail)


def _batman_model_on_data(
    data: pd.DataFrame,
    planet_table: pd.DataFrame,
    derived: pd.DataFrame,
    nuisance_table: pd.DataFrame | None = None,
    *,
    time_offset: float = 0.0,
) -> np.ndarray | None:
    if data is None or data.empty or planet_table is None or planet_table.empty or derived is None or derived.empty:
        return None

    time = data["time"].to_numpy(dtype=float) + time_offset
    model = np.ones_like(time, dtype=float)
    limb_darkening = _limb_darkening_from_table(nuisance_table)
    for planet in list(dict.fromkeys(planet_table["planet"].astype(str))):
        geometry = derived[derived["planet"].astype(str) == planet]
        if geometry.empty:
            continue
        period = _planet_table_value(planet_table, planet, "period", 1.0)
        t0 = _planet_table_value(planet_table, planet, "T0", 0.0)
        rr = _planet_table_value(planet_table, planet, "radius_ratio", 0.1)
        a_over_rstar = _finite_float(geometry.iloc[0]["a_over_Rstar"], float("nan"))
        cosi = _finite_float(geometry.iloc[0]["allesfitter_cosi"], 0.02)
        if not all(np.isfinite(value) for value in [period, t0, rr, a_over_rstar, cosi]) or period <= 0 or a_over_rstar <= 0:
            continue

        params = batman.TransitParams()
        params.t0 = t0
        params.per = period
        params.rp = rr
        params.a = a_over_rstar
        params.inc = math.degrees(math.acos(float(np.clip(cosi, 0.0, 1.0))))
        params.ecc = 0.0
        params.w = 90.0
        params.u = limb_darkening
        params.limb_dark = "quadratic"
        model *= batman.TransitModel(params, time).light_curve(params)
    return model


def _limb_darkening_from_table(nuisance_table: pd.DataFrame | None) -> list[float]:
    if nuisance_table is None or nuisance_table.empty:
        return [0.5, 0.1]
    u1_row = nuisance_table[nuisance_table["parameter"].astype(str) == "host_ldc_u1"]
    u2_row = nuisance_table[nuisance_table["parameter"].astype(str) == "host_ldc_u2"]
    if not u1_row.empty or not u2_row.empty:
        u1 = _finite_float(u1_row.iloc[0]["value"], 0.5) if not u1_row.empty else 0.5
        u2 = _finite_float(u2_row.iloc[0]["value"], 0.1) if not u2_row.empty else 0.1
        return [float(np.clip(u1, 0.0, 1.0)), float(np.clip(u2, 0.0, 1.0))]
    q1_row = nuisance_table[nuisance_table["parameter"].astype(str) == "host_ldc_q1"]
    q2_row = nuisance_table[nuisance_table["parameter"].astype(str) == "host_ldc_q2"]
    q1 = _finite_float(q1_row.iloc[0]["value"], 0.36) if not q1_row.empty else 0.36
    q2 = _finite_float(q2_row.iloc[0]["value"], 0.4166666667) if not q2_row.empty else 0.4166666667
    sqrt_q1 = math.sqrt(float(np.clip(q1, 0.0, 1.0)))
    q2 = float(np.clip(q2, 0.0, 1.0))
    u1 = 2.0 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1.0 - 2.0 * q2)
    return [u1, u2]


def _batman_model_from_primary_rows(
    model_time: np.ndarray,
    planet_table: pd.DataFrame,
    stellar_mass: float,
    stellar_radius: float,
    limb_darkening: list[float],
) -> np.ndarray:
    model = np.ones_like(model_time, dtype=float)
    for planet in list(dict.fromkeys(planet_table["planet"].astype(str))):
        period = _planet_table_value(planet_table, planet, "period", float("nan"))
        t0 = _planet_table_value(planet_table, planet, "T0", float("nan"))
        rr = _planet_table_value(planet_table, planet, "radius_ratio", float("nan"))
        impact = _planet_table_value(planet_table, planet, "impact", float("nan"))
        a_over_rstar = _kepler_a_over_rstar(period, stellar_mass, stellar_radius)
        if not all(np.isfinite([period, t0, rr, impact, a_over_rstar])) or period <= 0 or rr <= 0 or a_over_rstar <= 0:
            continue
        cosi = float(np.clip(impact / a_over_rstar, -1.0, 1.0))
        transit = batman.TransitParams()
        transit.t0 = t0
        transit.per = period
        transit.rp = rr
        transit.a = a_over_rstar
        transit.inc = math.degrees(math.acos(cosi))
        transit.ecc = 0.0
        transit.w = 90.0
        transit.u = limb_darkening
        transit.limb_dark = "quadratic"
        model *= batman.TransitModel(transit, model_time).light_curve(transit)
    return model


def _run_least_squares_refinement(
    data: pd.DataFrame,
    planet_table: pd.DataFrame,
    nuisance_table: pd.DataFrame,
    stellar_mass: float,
    stellar_radius: float,
    *,
    time_offset: float,
    max_seconds: float = 180.0,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, object], str | None]:
    try:
        from scipy.optimize import least_squares
    except Exception as exc:  # noqa: BLE001 - optional dependency guard
        return None, None, {}, f"Could not import scipy.optimize.least_squares: {exc}"

    if data is None or data.empty:
        return None, None, {}, "Load photometry data before running least-squares refinement."
    fit_table = planet_table.copy()
    primary = {"T0", "period", "radius_ratio", "impact"}
    variable_rows = [
        idx
        for idx, row in fit_table.iterrows()
        if str(row.get("parameter")) in primary and bool(row.get("fit", True))
    ]
    if not variable_rows:
        return None, None, {}, "No checked primary parameters are available for least-squares refinement."

    time_values = data["time"].to_numpy(dtype=float) + time_offset
    flux = data["flux"].to_numpy(dtype=float)
    flux_err = data.get("flux_err", pd.Series(np.ones(len(data)), index=data.index)).to_numpy(dtype=float)
    flux_err = np.where(np.isfinite(flux_err) & (flux_err > 0), flux_err, np.nanmedian(flux_err[flux_err > 0]) if np.any(flux_err > 0) else 1.0)
    finite = np.isfinite(time_values) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    time_values = time_values[finite]
    flux = flux[finite]
    flux_err = flux_err[finite]
    limb_darkening = _limb_darkening_from_table(nuisance_table)

    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for idx in variable_rows:
        row = fit_table.loc[idx]
        value = _finite_float(row.get("value"), 0.0)
        lo = _finite_float(row.get("lower"), float("nan"))
        hi = _finite_float(row.get("upper"), float("nan"))
        parameter = str(row.get("parameter"))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            width = max(abs(value) * 0.05, 1e-5)
            lo, hi = value - width, value + width
        if parameter == "period":
            lo = max(lo, 1e-8)
        elif parameter == "radius_ratio":
            lo = max(lo, 1e-5)
            hi = min(max(hi, lo * 1.01), 1.5)
        elif parameter == "impact":
            lo = max(lo, 0.0)
        x0.append(float(np.clip(value, lo, hi)))
        lower.append(float(lo))
        upper.append(float(hi))

    started = time.monotonic()
    calls = {"n": 0}

    def residuals(values: np.ndarray) -> np.ndarray:
        if time.monotonic() - started > max_seconds:
            raise TimeoutError(f"Least-squares refinement exceeded {max_seconds:.0f} seconds.")
        calls["n"] += 1
        trial = fit_table.copy()
        for idx, value in zip(variable_rows, values):
            trial.loc[idx, "value"] = value
        model = _batman_model_from_primary_rows(time_values, trial, stellar_mass, stellar_radius, limb_darkening)
        return (flux - model) / flux_err

    try:
        result = least_squares(
            residuals,
            np.asarray(x0, dtype=float),
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            method="trf",
            x_scale="jac",
            max_nfev=500,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
    except TimeoutError as exc:
        return None, None, {"elapsed_seconds": time.monotonic() - started, "calls": calls["n"]}, str(exc)
    except Exception as exc:  # noqa: BLE001 - user-facing optimizer issue
        return None, None, {"elapsed_seconds": time.monotonic() - started, "calls": calls["n"]}, f"Least-squares refinement failed: {exc}"

    refined = fit_table.copy()
    for idx, value in zip(variable_rows, result.x):
        refined.loc[idx, "value"] = value
    plot_model = _batman_model_from_primary_rows(data["time"].to_numpy(dtype=float) + time_offset, refined, stellar_mass, stellar_radius, limb_darkening)
    rows = []
    for idx, value, lo, hi in zip(variable_rows, result.x, lower, upper):
        row = refined.loc[idx]
        span = max(abs(hi - lo), 1e-12)
        near_lower = abs(value - lo) <= max(1e-8, 1e-4 * span)
        near_upper = abs(value - hi) <= max(1e-8, 1e-4 * span)
        if near_lower:
            bound_status = "at lower bound"
        elif near_upper:
            bound_status = "at upper bound"
        else:
            bound_status = ""
        rows.append(
            {
                "planet": row["planet"],
                "parameter": row["parameter"],
                "value": value,
                "lower": lo,
                "upper": hi,
                "bound_status": bound_status,
            }
        )
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "calls": calls["n"],
        "elapsed_seconds": time.monotonic() - started,
        "plot_model": plot_model,
        "summary": pd.DataFrame(rows),
    }
    return refined, info["summary"], info, None


def _ls_summary_style(summary: pd.DataFrame):
    def row_style(row: pd.Series) -> list[str]:
        if str(row.get("bound_status", "")):
            return ["background-color: #fed7aa; color: #7c2d12"] * len(row)
        return [""] * len(row)

    return summary.style.apply(row_style, axis=1)


def _seed_post_ls_bounds(table: pd.DataFrame) -> pd.DataFrame:
    seeded = table.copy()
    for idx, row in seeded.iterrows():
        if not bool(row.get("fit", True)):
            continue
        value = _finite_float(row.get("value"), float("nan"))
        if not np.isfinite(value):
            continue
        parameter = str(row.get("parameter"))
        if parameter == "T0":
            half_width = 0.006
        elif parameter == "period":
            half_width = 0.00035
        elif parameter == "radius_ratio":
            half_width = max(abs(value) * 0.08, 0.003)
        elif parameter == "impact":
            half_width = 0.12
        else:
            continue
        lower = value - half_width
        upper = value + half_width
        if parameter == "period":
            lower = max(lower, 1e-8)
        elif parameter == "radius_ratio":
            lower = max(lower, 1e-5)
            upper = min(upper, 1.5)
        elif parameter == "impact":
            lower = max(lower, 0.0)
            upper = min(upper, 1.5)
        seeded.loc[idx, "lower"] = lower
        seeded.loc[idx, "upper"] = upper
        seeded.loc[idx, "sigma"] = max(half_width / 5.0, 1e-8)
        seeded.loc[idx, "prior_type"] = "Uniform"
    return seeded



def _load_fit_data(frame: pd.DataFrame, source: str) -> None:
    data = frame.copy()
    missing = [column for column in REQUIRED_PHOTOMETRY_COLUMNS if column not in data.columns]
    if missing:
        st.error(f"Photometry data is missing required column(s): {', '.join(missing)}")
        return

    data = data.dropna(subset=["time", "flux"]).copy()
    if "is_outlier" not in data.columns:
        data["is_outlier"] = False
    if "source_file" not in data.columns:
        data["source_file"] = source
    if "sector" not in data.columns:
        data["sector"] = ""

    st.session_state["photometry_fit_data"] = data
    st.session_state["photometry_fit_data_source"] = source
    st.success(f"Loaded {len(data):,} photometry points for fitting.")


def _fit_photometry_candidates() -> list[Path]:
    candidates: list[Path] = []
    if FIT_ROOT.exists():
        for path in FIT_ROOT.glob("*/photometry_transit_fit/*.csv"):
            if path.name in {"params.csv", "settings.csv"}:
                continue
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)


def _read_fit_photometry_file(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception:
        frame = pd.DataFrame()

    if not set(REQUIRED_PHOTOMETRY_COLUMNS).issubset(frame.columns):
        frame = pd.read_csv(path, header=None, names=REQUIRED_PHOTOMETRY_COLUMNS, usecols=[0, 1, 2])

    data = frame.copy()
    instrument = path.stem.upper()
    if instrument == "TESS" and "time" in data.columns:
        time_values = pd.to_numeric(data["time"], errors="coerce")
        if time_values.median(skipna=True) > TESS_TIME_OFFSET:
            data["time"] = time_values - TESS_TIME_OFFSET
            data["source_time_note"] = "BJD_TDB converted back to BTJD for GUI preview/fitting setup"
    return data


def _render_data_loader() -> None:
    st.subheader("Load photometry data")
    prepared = st.session_state.get("prepared_photometry")
    previous_fit_files = _fit_photometry_candidates()

    if (
        st.session_state.get("photometry_fit_data") is None
        and previous_fit_files
        and not st.session_state.get("photometry_fit_auto_loaded_previous")
    ):
        try:
            recovered = _read_fit_photometry_file(previous_fit_files[0])
        except Exception:
            recovered = pd.DataFrame()
        if not recovered.empty:
            st.session_state["photometry_fit_auto_loaded_previous"] = True
            _load_fit_data(recovered, str(previous_fit_files[0]))

    load_col, previous_col, upload_col = st.columns([0.3, 0.34, 0.36], vertical_alignment="top")
    with load_col:
        st.caption("Use the prepared light curve from the Photometry Import tab.")
        if prepared is None or prepared.empty:
            st.button("Load prepared photometry", use_container_width=True, disabled=True)
            st.info("No prepared photometry is currently available in this session.")
        elif st.button("Load prepared photometry", use_container_width=True, type="primary"):
            _load_fit_data(prepared.loc[~prepared["is_outlier"]].copy(), "Photometry Import tab")

    with previous_col:
        st.caption("Recover data from a written allesfitter directory.")
        if previous_fit_files:
            labels = [str(path) for path in previous_fit_files]
            selected = st.selectbox("Previous instrument CSV", labels, key="photometry_fit_previous_csv")
            if st.button("Load previous fit data", use_container_width=True):
                try:
                    frame = _read_fit_photometry_file(Path(selected))
                except Exception as exc:  # noqa: BLE001 - user-facing file read issue
                    st.error(f"Could not load previous fit data: {exc}")
                else:
                    _load_fit_data(frame, selected)
        else:
            st.button("Load previous fit data", use_container_width=True, disabled=True)
            st.info("No previous instrument CSV found.")

    with upload_col:
        uploaded = st.file_uploader(
            "Or load prepared photometry CSV",
            type=["csv"],
            key="photometry_fit_csv_upload",
            help="Expected columns: time, flux, flux_err. Optional columns such as sector and source_file are kept.",
        )
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
            except Exception as exc:  # noqa: BLE001 - user-facing file read issue
                st.error(f"Could not read CSV: {exc}")
            else:
                if st.button("Use uploaded CSV", use_container_width=True):
                    _load_fit_data(frame, uploaded.name)

    data = st.session_state.get("photometry_fit_data")
    if data is None or data.empty:
        return

    st.caption(f"Current fit data source: {st.session_state.get('photometry_fit_data_source', 'unknown')}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Points", f"{len(data):,}")
    metric_cols[1].metric("Sectors", f"{data['sector'].nunique() if 'sector' in data else 0}")
    metric_cols[2].metric("Time min", f"{data['time'].min():.4f}")
    metric_cols[3].metric("Time max", f"{data['time'].max():.4f}")
    st.plotly_chart(
        prepared_photometry_preview(data),
        use_container_width=True,
        config=PLOT_CONFIG,
        key="photometry_fit_loaded_data_preview",
    )


def _current_tic_id() -> str:
    for key in ["mast_resolved_target", "target_name", "photometry_target_input"]:
        tic = extract_tic_id(st.session_state.get(key))
        if tic:
            return tic
    return ""


def _render_exofop_retrieval() -> None:
    st.subheader("Retrieve parameters from ExoFOP")
    tic_id = _current_tic_id()
    target_name = st.session_state.get("target_name") or st.session_state.get("photometry_target_input") or ""
    cache_info = cached_toi_info()

    input_col, button_col = st.columns([0.7, 0.3], vertical_alignment="bottom")
    with input_col:
        tic_input = st.text_input(
            "TIC ID for ExoFOP search",
            value=tic_id,
            placeholder="243921117",
            key="phot_fit_exofop_tic_input",
            help="Uses the resolved TIC from the MAST/SIMBAD step when available.",
        )
    with button_col:
        retrieve = st.button("Search ExoFOP", use_container_width=True, type="primary")

    use_cached = st.checkbox(
        "Use pre-downloaded ExoFOP TOI list",
        value=True,
        key="phot_fit_use_cached_exofop",
        help="When ticked, search the local saved TOI table. Untick to download a fresh ExoFOP list and replace the saved copy.",
    )
    if cache_info["exists"]:
        obtained = cache_info.get("obtained_at") or "date unknown"
        rows = cache_info.get("rows")
        row_text = f"{int(rows):,} rows" if pd.notna(rows) else "row count unknown"
        st.caption(f"Saved TOI list: {row_text}, obtained {obtained}.")
    else:
        st.caption("No saved ExoFOP TOI list yet. The first search will download and save one.")

    if retrieve:
        search_tic = extract_tic_id(tic_input)
        if not search_tic:
            st.error("Enter a TIC ID before searching ExoFOP.")
        else:
            action = "Searching saved ExoFOP TOI list" if use_cached and cache_info["exists"] else "Downloading ExoFOP TOI list"
            with st.spinner(f"{action} for TIC {search_tic}..."):
                try:
                    toi_table, source_info = load_toi_table(use_cached=use_cached)
                    matches = find_toi_parameters(toi_table, search_tic)
                except Exception as exc:  # noqa: BLE001 - user-facing network/table issue
                    st.session_state["phot_fit_exofop_error"] = str(exc)
                    st.session_state.pop("phot_fit_exofop_matches", None)
                else:
                    st.session_state["phot_fit_exofop_matches"] = matches
                    st.session_state["phot_fit_exofop_tic"] = search_tic
                    st.session_state["phot_fit_exofop_source_info"] = source_info
                    st.session_state.pop("phot_fit_exofop_error", None)

    if st.session_state.get("phot_fit_exofop_error"):
        st.error(st.session_state["phot_fit_exofop_error"])

    matches = st.session_state.get("phot_fit_exofop_matches")
    if matches is None:
        st.caption(f"Target context: {target_name or 'not set'}. ExoFOP lookup will list every TOI row for the TIC.")
        return

    searched_tic = st.session_state.get("phot_fit_exofop_tic", tic_input)
    source_info = st.session_state.get("phot_fit_exofop_source_info", {})
    if source_info:
        st.caption(
            f"TOI list source: {source_info.get('source', 'unknown')} "
            f"({source_info.get('rows', 'unknown')} rows, obtained {source_info.get('obtained_at', 'date unknown')})."
        )
    if matches.empty:
        st.warning(f"No ExoFOP TOI rows found for TIC {searched_tic}.")
        return

    st.success(f"Found {len(matches)} ExoFOP TOI row(s) for TIC {searched_tic}.")
    display_columns = preferred_toi_columns(matches)
    st.dataframe(matches[display_columns], use_container_width=True, hide_index=True)

    with st.expander("Show all ExoFOP columns"):
        st.dataframe(matches, use_container_width=True, hide_index=True)


def _stellar_default(matches: pd.DataFrame, column: str, fallback: float) -> float:
    if matches is None or matches.empty or column not in matches.columns:
        return fallback
    value = _finite_float(matches.iloc[0].get(column), fallback)
    return value if value > 0 else fallback


def _render_transit_parameter_setup() -> None:
    st.subheader("Transit fit parameters")
    st.caption(
        "This editor uses impact parameter for readability, then converts to allesfitter's native "
        "`cosi` and `rsuma` rows using Kepler's law. The impact prior is mapped onto `cosi`; `rsuma` is derived "
        "from the current period, stellar mass, stellar radius, and radius ratio."
    )

    matches = st.session_state.get("phot_fit_exofop_matches")
    exofop_count = len(matches) if matches is not None and not matches.empty else 1
    cols = st.columns(4)
    planet_count = cols[0].number_input("Number of planets", min_value=1, max_value=8, value=int(exofop_count), step=1)
    default_prior = cols[1].radio("Default prior shape", ["Gaussian", "Uniform"], horizontal=True, key="phot_fit_default_prior")
    cols[1].caption(
        "When seeded from ExoFOP, uniform T0/period bounds default to ±50× the listed 1-sigma uncertainty. "
        "Radius ratio defaults to 0.5-1.5× the seed value; impact defaults to 0-1.2."
    )
    stellar_mass = cols[2].number_input(
        "Stellar mass [Msun]",
        min_value=0.01,
        value=float(_stellar_default(matches, "Stellar Mass (M_Sun)", 1.0)),
        format="%.6f",
        key="phot_fit_stellar_mass",
    )
    stellar_radius = cols[3].number_input(
        "Stellar radius [Rsun]",
        min_value=0.01,
        value=float(_stellar_default(matches, "Stellar Radius (R_Sun)", 1.0)),
        format="%.6f",
        key="phot_fit_stellar_radius",
    )

    seed_label = "Seed from ExoFOP" if matches is not None and not matches.empty else "Build blank parameter table"
    should_seed = st.button(seed_label, use_container_width=True, key="seed_phot_fit_params")
    if should_seed or "phot_fit_planet_priors_data" not in st.session_state:
        st.session_state["phot_fit_planet_priors_data"] = _default_planet_rows(matches, int(planet_count), default_prior)
        st.session_state["phot_fit_planet_prior_editor_version"] = st.session_state.get("phot_fit_planet_prior_editor_version", 0) + 1

    priors = st.session_state["phot_fit_planet_priors_data"]
    edited = st.data_editor(
        priors,
        use_container_width=True,
        num_rows="dynamic",
        key=f"phot_fit_planet_prior_editor_{st.session_state.get('phot_fit_planet_prior_editor_version', 0)}",
        column_config=_planet_prior_column_config(),
    )
    st.session_state["phot_fit_latest_edited_priors"] = edited

    phot_inst = st.text_input("Photometry instrument name for params/settings", value="TESS", key="phot_fit_param_inst")
    ld_space = st.radio(
        "Limb-darkening coefficient input",
        ["u coefficients", "q coefficients"],
        horizontal=True,
        key="phot_fit_ld_space",
        help="Use u coefficients for ExoCTK quadratic limb darkening values. q coefficients are Kipping's transformed sampling variables.",
    )
    ld_space_key = "u" if ld_space.startswith("u") else "q"
    if ld_space_key == "u":
        st.caption("ExoCTK quadratic values such as u1=0.5, u2=0.1 should be entered here as `host_ldc_u1`, `host_ldc_u2`.")
    else:
        st.caption("q-space uses Kipping coefficients. u1=0.5, u2=0.1 corresponds to q1=0.36, q2=0.4167.")
    time_offset = TESS_TIME_OFFSET if phot_inst.strip().upper() == "TESS" else 0.0
    if time_offset:
        st.info("TESS light-curve times are treated as BTJD, so 2457000 is added internally for the BATMAN initial model.")

    st.subheader("Instrument, baseline, and limb-darkening priors")
    st.caption(
        "`fit` checked means sample this parameter. `fit` unchecked means write `fit=0` to params.csv and keep exactly the listed value."
    )
    fit_data_for_noise = st.session_state.get("photometry_fit_data")
    noise_seed = _flux_error_scale_from_data(fit_data_for_noise)
    st.caption(
        f"`ln_err_flux` is the natural log of the relative-flux uncertainty. "
        f"From the loaded data, the current seed is ln({noise_seed:.4g}) = {np.log(max(noise_seed, 1e-12)):.4g}."
    )
    if "phot_fit_nuisance_priors_data" not in st.session_state:
        st.session_state["phot_fit_nuisance_priors_data"] = _default_nuisance_rows(phot_inst.strip() or "TESS", fit_data_for_noise, ld_space_key)
        st.session_state["phot_fit_nuisance_editor_version"] = st.session_state.get("phot_fit_nuisance_editor_version", 0) + 1
        st.session_state["phot_fit_nuisance_ld_space"] = ld_space_key
    if st.session_state.get("phot_fit_nuisance_ld_space") != ld_space_key:
        st.session_state["phot_fit_nuisance_priors_data"] = _default_nuisance_rows(phot_inst.strip() or "TESS", fit_data_for_noise, ld_space_key)
        st.session_state["phot_fit_nuisance_editor_version"] = st.session_state.get("phot_fit_nuisance_editor_version", 0) + 1
        st.session_state["phot_fit_nuisance_ld_space"] = ld_space_key
    if st.button("Reset instrument priors", use_container_width=True, key="reset_phot_fit_nuisance"):
        st.session_state["phot_fit_nuisance_priors_data"] = _default_nuisance_rows(phot_inst.strip() or "TESS", fit_data_for_noise, ld_space_key)
        st.session_state["phot_fit_nuisance_editor_version"] = st.session_state.get("phot_fit_nuisance_editor_version", 0) + 1

    nuisance = _with_data_seeded_ln_err(st.session_state["phot_fit_nuisance_priors_data"], fit_data_for_noise)
    st.session_state["phot_fit_nuisance_priors_data"] = nuisance
    edited_nuisance = st.data_editor(
        nuisance,
        use_container_width=True,
        num_rows="fixed",
        key=f"phot_fit_nuisance_prior_editor_{st.session_state.get('phot_fit_nuisance_editor_version', 0)}",
        column_config={
            "parameter": st.column_config.SelectboxColumn(
                "parameter",
                options=["host_ldc_u1", "host_ldc_u2", "host_ldc_q1", "host_ldc_q2", "ln_err_flux", "baseline_offset_flux", "dil"],
            ),
            "prior_type": st.column_config.SelectboxColumn("prior_type", options=["Gaussian", "Uniform"]),
            "value": st.column_config.NumberColumn("value", format="%.10g"),
            "sigma": st.column_config.NumberColumn("sigma", format="%.10g"),
            "lower": st.column_config.NumberColumn("lower", format="%.10g"),
            "upper": st.column_config.NumberColumn("upper", format="%.10g"),
            "fit": st.column_config.CheckboxColumn("fit"),
        },
    )
    st.session_state["phot_fit_latest_edited_nuisance_priors"] = edited_nuisance

    if st.button("Generate table and initial model from current priors", type="primary", use_container_width=True):
        edited_nuisance = _with_data_seeded_ln_err(edited_nuisance, st.session_state.get("photometry_fit_data"))
        st.session_state["phot_fit_planet_priors_data"] = edited.copy()
        st.session_state["phot_fit_nuisance_priors_data"] = edited_nuisance.copy()
        params, derived = _build_allesfitter_params_from_setup(
            edited,
            edited_nuisance,
            stellar_mass,
            stellar_radius,
            phot_inst.strip() or "TESS",
            ld_space_key,
        )
        st.session_state["phot_fit_generated_params"] = params
        st.session_state["phot_fit_derived_geometry"] = derived
        model_data = st.session_state.get("photometry_fit_data")
        st.session_state["phot_fit_model_time_offset"] = time_offset
        st.session_state["phot_fit_initial_model"] = _batman_model_on_data(model_data, edited, derived, edited_nuisance, time_offset=time_offset)
        st.success("Generated allesfitter parameters and initial BATMAN model from the current prior table.")

    params = st.session_state.get("phot_fit_generated_params")
    derived = st.session_state.get("phot_fit_derived_geometry")

    st.subheader("Derived geometry")
    if derived is None or derived.empty:
        st.info("Generate the table to calculate derived geometry.")
    else:
        st.dataframe(derived, use_container_width=True, hide_index=True)
        fit_data = st.session_state.get("photometry_fit_data")
        if fit_data is None or fit_data.empty:
            st.info("Load photometry data above to preview the initial model over the data.")
        else:
            model_flux = st.session_state.get("phot_fit_initial_model")
            model_time_offset = st.session_state.get("phot_fit_model_time_offset", 0.0)
            if model_time_offset:
                st.caption(f"Initial model evaluated at plotted time + {model_time_offset:.0f} to convert TESS BTJD to BJD.")
            st.plotly_chart(
                photometry_model_overlay(fit_data, model_flux),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="phot_fit_initial_model_overlay",
            )

    st.subheader("Least-squares starting-parameter refinement")
    st.caption(
        "Fits checked primary parameters (`T0`, `period`, `radius_ratio`, `impact`) with stellar mass/radius, limb darkening, "
        "and instrument nuisance parameters fixed. This is only for refining starting values; uncertainties still come from MCMC."
    )
    st.info("The least-squares refinement currently assumes circular, non-elliptical orbits (`ecc=0`, `omega=90 deg`).")
    fit_data = st.session_state.get("photometry_fit_data")
    can_refine = fit_data is not None and not fit_data.empty and edited is not None and not edited.empty
    if st.button("Run least-squares refinement", use_container_width=True, disabled=not can_refine):
        edited_nuisance_for_fit = _with_data_seeded_ln_err(edited_nuisance, fit_data)
        with st.spinner("Running least-squares refinement for up to 3 minutes..."):
            refined, summary, info, error = _run_least_squares_refinement(
                fit_data,
                edited,
                edited_nuisance_for_fit,
                stellar_mass,
                stellar_radius,
                time_offset=time_offset,
                max_seconds=180.0,
            )
        if error:
            st.error(error)
        elif refined is not None:
            refined_for_priors = _seed_post_ls_bounds(refined)
            st.session_state["phot_fit_planet_priors_data"] = refined_for_priors.copy()
            st.session_state["phot_fit_latest_edited_priors"] = refined_for_priors.copy()
            st.session_state["phot_fit_planet_prior_editor_version"] = st.session_state.get("phot_fit_planet_prior_editor_version", 0) + 1
            refined_params, refined_derived = _build_allesfitter_params_from_setup(
                refined_for_priors,
                edited_nuisance_for_fit,
                stellar_mass,
                stellar_radius,
                phot_inst.strip() or "TESS",
                ld_space_key,
            )
            st.session_state["phot_fit_generated_params"] = refined_params
            st.session_state["phot_fit_derived_geometry"] = refined_derived
            st.session_state["phot_fit_initial_model"] = info.get("plot_model")
            st.session_state["phot_fit_model_time_offset"] = time_offset
            st.session_state["phot_fit_least_squares_result"] = info
            st.success(
                f"Least-squares refinement finished in {info.get('elapsed_seconds', 0):.1f}s "
                f"using {info.get('nfev', 0)} optimizer evaluations."
            )
            if not info.get("success", False):
                st.warning(f"Least-squares stopped before formal convergence: {info.get('message', 'status unknown')}")
            if isinstance(summary, pd.DataFrame) and not summary.empty:
                if summary["bound_status"].astype(str).str.len().gt(0).any():
                    st.warning("One or more least-squares parameters ended at a prior bound; those rows are highlighted below.")
                st.dataframe(_ls_summary_style(summary), use_container_width=True, hide_index=True)
            st.plotly_chart(
                photometry_model_overlay(
                    fit_data,
                    info.get("plot_model"),
                    title="Least-squares refined transit model on data",
                    model_name="Least-squares model",
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="phot_fit_least_squares_model",
            )
            st.caption("The parameter editor will update with these refined values on the next rerun.")
            st.caption("Post-LS uniform bounds are reseeded around the LS solution; adjust them in the table below before running MCMC.")
    elif st.session_state.get("phot_fit_least_squares_result"):
        info = st.session_state["phot_fit_least_squares_result"]
        st.caption(
            f"Last least-squares run: {info.get('elapsed_seconds', 0):.1f}s, "
            f"{info.get('nfev', 0)} optimizer evaluations, cost={info.get('cost', float('nan')):.4g}."
        )
        summary = info.get("summary")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            if "bound_status" in summary and summary["bound_status"].astype(str).str.len().gt(0).any():
                st.warning("One or more least-squares parameters ended at a prior bound; those rows are highlighted below.")
            st.dataframe(_ls_summary_style(summary), use_container_width=True, hide_index=True)
        if fit_data is not None and not fit_data.empty and info.get("plot_model") is not None:
            st.plotly_chart(
                photometry_model_overlay(
                    fit_data,
                    info.get("plot_model"),
                    title="Least-squares refined transit model on data",
                    model_name="Least-squares model",
                ),
                use_container_width=True,
                config=PLOT_CONFIG,
                key="phot_fit_least_squares_model_saved",
            )

    if st.session_state.get("phot_fit_least_squares_result"):
        st.subheader("Adjust least-squares priors for allesfitter")
        st.caption(
            "The values below are seeded from the least-squares solution. Adjust the lower/upper bounds or prior type here "
            "before generating the final allesfitter params table."
        )
        ls_priors = st.session_state.get("phot_fit_planet_priors_data", edited).copy()
        adjusted_ls_priors = st.data_editor(
            ls_priors,
            use_container_width=True,
            num_rows="dynamic",
            key=f"phot_fit_post_ls_prior_editor_{st.session_state.get('phot_fit_planet_prior_editor_version', 0)}",
            column_config=_planet_prior_column_config(),
        )
        if st.button("Use adjusted least-squares priors for allesfitter", type="primary", use_container_width=True):
            adjusted_nuisance = _with_data_seeded_ln_err(edited_nuisance, fit_data)
            st.session_state["phot_fit_planet_priors_data"] = adjusted_ls_priors.copy()
            st.session_state["phot_fit_latest_edited_priors"] = adjusted_ls_priors.copy()
            st.session_state["phot_fit_nuisance_priors_data"] = adjusted_nuisance.copy()
            params, derived = _build_allesfitter_params_from_setup(
                adjusted_ls_priors,
                adjusted_nuisance,
                stellar_mass,
                stellar_radius,
                phot_inst.strip() or "TESS",
                ld_space_key,
            )
            st.session_state["phot_fit_generated_params"] = params
            st.session_state["phot_fit_derived_geometry"] = derived
            st.session_state["phot_fit_model_time_offset"] = time_offset
            st.session_state["phot_fit_initial_model"] = _batman_model_on_data(
                fit_data,
                adjusted_ls_priors,
                derived,
                adjusted_nuisance,
                time_offset=time_offset,
            )
            st.success("Generated allesfitter parameters from the adjusted least-squares prior table.")

    params = st.session_state.get("phot_fit_generated_params")
    st.subheader("Generated allesfitter params preview")
    if params is None or params.empty:
        st.info("Generate the table to preview allesfitter params.")
    else:
        st.dataframe(params, use_container_width=True, hide_index=True)


def render() -> None:
    page_header(
        "Photometry Transit Fit",
        "Configure standard transit fits for prepared light curves and review posterior diagnostics.",
    )

    fit_tab, parameter_tab, output_tab = st.tabs(["Fit Setup", "allesfitter Parameters", "Outputs"])

    with fit_tab:
        _render_data_loader()
        st.divider()
        _render_exofop_retrieval()
        st.divider()
        _render_transit_parameter_setup()
        st.divider()

        params_col = st.container()
        with params_col:
            st.subheader("Model setup")
            sampler = st.selectbox("Sampler", ["MCMC", "Nested sampling"], key="phot_fit_sampler_local")
            params_for_defaults = st.session_state.get("phot_fit_generated_params")
            free_param_count = len(_mcmc_labels(params_for_defaults))
            default_walkers = max(8, 8 * max(free_param_count, 1))
            previous_auto_walkers = st.session_state.get("phot_fit_auto_walkers")
            if "phot_fit_mcmc_nwalkers" not in st.session_state or st.session_state.get("phot_fit_mcmc_nwalkers") == previous_auto_walkers:
                st.session_state["phot_fit_mcmc_nwalkers"] = default_walkers
            st.session_state["phot_fit_auto_walkers"] = default_walkers
            if "phot_fit_mcmc_total_steps" not in st.session_state or st.session_state.get("phot_fit_mcmc_total_steps") == 2000:
                st.session_state["phot_fit_mcmc_total_steps"] = 3000
            if "phot_fit_mcmc_burn_steps" not in st.session_state:
                st.session_state["phot_fit_mcmc_burn_steps"] = 1000
            if "phot_fit_mcmc_thin_by" not in st.session_state:
                st.session_state["phot_fit_mcmc_thin_by"] = 1
            if "phot_fit_mcmc_pre_run_loops" not in st.session_state:
                st.session_state["phot_fit_mcmc_pre_run_loops"] = 0
            if "phot_fit_mcmc_pre_run_steps" not in st.session_state:
                st.session_state["phot_fit_mcmc_pre_run_steps"] = 500

            if sampler == "MCMC":
                cpu_count = os.cpu_count() or 1
                st.caption(
                    f"Current default walkers: 8 x {max(free_param_count, 1)} free parameter(s) = {default_walkers}."
                )
                use_multiprocess = st.checkbox("Use multiple CPU cores", value=True, key="phot_fit_multiprocess")
                multiprocess_cores = st.number_input(
                    "CPU cores",
                    min_value=1,
                    max_value=max(cpu_count, 1),
                    value=min(max(cpu_count - 1, 1), max(cpu_count, 1)),
                    step=1,
                    disabled=not use_multiprocess,
                    key="phot_fit_multiprocess_cores",
                )
                sampler_controls = {
                    "multiprocess": use_multiprocess,
                    "multiprocess_cores": multiprocess_cores if use_multiprocess else 1,
                    "mcmc_initialization": st.selectbox(
                        "Initial walker distribution",
                        ["uniform", "tight"],
                        index=0,
                        key="phot_fit_mcmc_initialization",
                        help="uniform draws walkers across each prior range for a fresh run; tight starts a small cloud around the listed values.",
                    ),
                    "mcmc_nwalkers": st.number_input(
                        "Walkers",
                        min_value=8,
                        max_value=512,
                        value=int(st.session_state.get("phot_fit_mcmc_nwalkers", default_walkers)),
                        step=8,
                        key="phot_fit_mcmc_nwalkers",
                    ),
                    "mcmc_total_steps": st.number_input(
                        "MCMC total steps",
                        min_value=10,
                        max_value=1_000_000,
                        value=int(st.session_state.get("phot_fit_mcmc_total_steps", 3000)),
                        step=100,
                        key="phot_fit_mcmc_total_steps",
                    ),
                    "mcmc_burn_steps": st.number_input(
                        "Burn-in steps",
                        min_value=0,
                        max_value=999_999,
                        value=int(st.session_state.get("phot_fit_mcmc_burn_steps", 1000)),
                        step=100,
                        key="phot_fit_mcmc_burn_steps",
                    ),
                    "mcmc_thin_by": st.number_input(
                        "Thin chains by",
                        min_value=1,
                        max_value=1000,
                        value=int(st.session_state.get("phot_fit_mcmc_thin_by", 1)),
                        step=1,
                        key="phot_fit_mcmc_thin_by",
                    ),
                    "mcmc_pre_run_loops": st.number_input(
                        "Pre-run loops",
                        min_value=0,
                        max_value=20,
                        value=int(st.session_state.get("phot_fit_mcmc_pre_run_loops", 0)),
                        step=1,
                        key="phot_fit_mcmc_pre_run_loops",
                        help="allesfitter exploratory MCMC loops before the saved production chain. Useful for broad priors.",
                    ),
                    "mcmc_pre_run_steps": st.number_input(
                        "Pre-run steps",
                        min_value=0,
                        max_value=100_000,
                        value=int(st.session_state.get("phot_fit_mcmc_pre_run_steps", 500)),
                        step=100,
                        key="phot_fit_mcmc_pre_run_steps",
                    ),
                }
                if sampler_controls["mcmc_pre_run_loops"] > 0:
                    st.caption(
                        "Pre-runs explore first, then restart walkers around the highest-likelihood point found. "
                        "Set pre-run loops to 0 if you want the saved full chain to show the whole broad-prior search."
                    )
                if sampler_controls["mcmc_burn_steps"] >= sampler_controls["mcmc_total_steps"]:
                    st.warning("Burn-in should be smaller than the total number of MCMC steps.")
            else:
                sampler_controls = {
                    "multiprocess": False,
                    "multiprocess_cores": 1,
                    "ns_modus": st.selectbox("Nested mode", ["dynamic", "static"], key="phot_fit_ns_modus"),
                    "ns_nlive": st.number_input("Live points", min_value=25, max_value=10000, value=300, step=25, key="phot_fit_ns_nlive"),
                    "ns_tol": st.number_input("Evidence tolerance", min_value=0.001, max_value=10.0, value=0.1, step=0.05, format="%.3f", key="phot_fit_ns_tol"),
                    "ns_bound": st.selectbox("Nested bound", ["single", "multi", "balls", "cubes"], key="phot_fit_ns_bound"),
                    "ns_sample": st.selectbox("Nested sample method", ["rwalk", "unif", "slice", "rslice", "hslice"], key="phot_fit_ns_sample"),
                }
            initial_guess_first = st.checkbox("Run initial guess plot first", value=True, key="phot_fit_initial_guess")
            fresh_sampler_start = st.checkbox(
                "Start sampler from scratch",
                value=True,
                help="Moves any existing sampler backend/logs into results/archived_runs before launching.",
                key="phot_fit_fresh_sampler_start",
            )
            fast_fit = st.checkbox("Use fast_fit transit windows", value=False, key="phot_fit_fast_fit")
            shift_epoch = st.checkbox("Shift epoch into data midpoint", value=True, key="phot_fit_shift_epoch")
            st.caption("Shift epoch moves the fitted reference transit time near the data midpoint to improve fitting performance.")

            params = st.session_state.get("phot_fit_generated_params")
            fit_data = st.session_state.get("photometry_fit_data")
            phot_inst = st.session_state.get("phot_fit_param_inst", "TESS").strip() or "TESS"
            time_offset = TESS_TIME_OFFSET if phot_inst.upper() == "TESS" else 0.0
            param_names = set(params["name"].astype(str)) if params is not None and not params.empty else set()
            companions = sorted(
                {
                    name.split("_", 1)[0]
                    for name in param_names
                    if "_" in name and name.split("_", 1)[0] not in {"host", "dil", "ln", "baseline"}
                }
            )
            ld_space_key = "u" if str(st.session_state.get("phot_fit_ld_space", "u coefficients")).startswith("u") else "q"
            settings = _settings_from_fit_setup(companions, phot_inst, sampler, shift_epoch, fast_fit, sampler_controls, ld_space_key) if companions else pd.DataFrame()
            issues = _validate_fit_inputs(fit_data, params, phot_inst, ld_space_key)
            prior_warnings = _prior_sanity_warnings(params, shift_epoch=shift_epoch)
            if issues:
                st.warning("Before writing the allesfitter directory:\n\n" + "\n".join(f"- {issue}" for issue in issues))
            if prior_warnings:
                st.info("Broad-prior sampling notes:\n\n" + "\n".join(f"- {warning}" for warning in prior_warnings))

            if time_offset:
                st.caption("When writing `TESS.csv`, BTJD times are converted to BJD_TDB by adding 2457000.")

            if st.button("Write allesfitter directory", use_container_width=True, disabled=bool(issues)):
                fit_dir = _write_fit_directory(fit_data, params, settings, phot_inst, time_offset=time_offset)
                st.session_state["phot_fit_directory"] = str(fit_dir)
                st.session_state["phot_fit_settings"] = settings
                st.success(f"Wrote allesfitter directory: {fit_dir}")

            fit_dir_value = st.session_state.get("phot_fit_directory", "")
            st.text_input("allesfitter directory", value=fit_dir_value, disabled=True)
            if fit_dir_value:
                st.caption("Directory contains params.csv, settings.csv, instrument CSV data, and WORKBENCH_NOTES.txt.")

            active_sampler = _sampler_is_active(
                st.session_state.get("phot_fit_sampler_pid"),
                st.session_state.get("phot_fit_last_sampler", sampler),
                fit_dir_value,
                int(st.session_state.get("phot_fit_mcmc_total_steps", sampler_controls.get("mcmc_total_steps", 3000))),
                int(st.session_state.get("phot_fit_sampler_total_steps", 0)) or None,
            )
            if not active_sampler and st.session_state.get("phot_fit_sampler_pid"):
                st.session_state.pop("phot_fit_sampler_pid", None)
            if active_sampler:
                st.warning("A sampler process is still running. Stop it before starting another run.")
            run_disabled = not bool(fit_dir_value) or active_sampler
            if st.button("Run selected sampler", use_container_width=True, disabled=run_disabled):
                launch_dir = Path(fit_dir_value)
                if fit_data is not None and params is not None and not issues:
                    launch_dir = _write_fit_directory(fit_data, params, settings, phot_inst, time_offset=time_offset)
                    st.session_state["phot_fit_directory"] = str(launch_dir)
                    st.session_state["phot_fit_settings"] = settings
                process, archive_dir = _launch_sampler(
                    launch_dir,
                    sampler,
                    initial_guess_first=initial_guess_first,
                    fresh_start=fresh_sampler_start,
                )
                log_file = "mcmc_run.log" if sampler == "MCMC" else "nested_sampling_run.log"
                st.session_state["phot_fit_sampler_pid"] = process.pid
                st.session_state["phot_fit_sampler_log"] = str(launch_dir / log_file)
                st.session_state["phot_fit_last_sampler"] = sampler
                if sampler == "MCMC":
                    st.session_state["phot_fit_sampler_total_steps"] = int(sampler_controls.get("mcmc_total_steps", 3000))
                else:
                    st.session_state.pop("phot_fit_sampler_total_steps", None)
                st.success(f"Started {sampler} with PID {process.pid}.")
                if archive_dir is not None:
                    st.caption(f"Previous sampler backend/logs moved to {archive_dir}.")

            if st.session_state.get("phot_fit_sampler_pid"):
                st.caption(f"Last sampler PID: {st.session_state['phot_fit_sampler_pid']}")
                st.text_input("Sampler log", value=st.session_state.get("phot_fit_sampler_log", ""), disabled=True)
                _render_sampler_monitor()

    with parameter_tab:
        render_docs_note()
        generated_params = st.session_state.get("phot_fit_generated_params")
        if generated_params is not None and not generated_params.empty:
            st.subheader("Generated from Fit Setup")
            edited_generated = st.data_editor(
                generated_params,
                use_container_width=True,
                num_rows="dynamic",
                key="phot_fit_generated_params_editor",
                column_config={"fit": st.column_config.CheckboxColumn("fit")},
            )
            st.download_button(
                "Download generated params.csv",
                data=csv_bytes(edited_generated),
                file_name="params.csv",
                mime="text/csv",
                key="phot_fit_download_generated_params",
            )
            st.divider()
        st.caption(
            "Core transit rows use `[companion]_rr`, `[companion]_rsuma`, `[companion]_cosi`, "
            "`[companion]_epoch`, and `[companion]_period`; photometry also usually needs "
            "`host_ldc_u1_[inst]`/`host_ldc_u2_[inst]` or `host_ldc_q1_[inst]`/`host_ldc_q2_[inst]`, "
            "`dil_[inst]`, `ln_err_flux_[inst]`, "
            "and baseline rows that match `baseline_flux_[inst]` in settings.csv."
        )
        context = render_context_controls("phot_fit", include_photometry=True, include_rv=False)
        render_config_editors("phot_fit", context)

    with output_tab:
        st.subheader("Sampler diagnostics")
        fit_dir_value = st.session_state.get("phot_fit_directory", "")
        if st.session_state.get("phot_fit_sampler_pid"):
            _render_sampler_monitor()
        elif fit_dir_value:
            st.caption("Run a sampler from Fit Setup, then this tab will populate with live status and posterior diagnostics.")
        else:
            st.caption("Write an allesfitter directory and run a sampler to populate diagnostics.")

        if fit_dir_value:
            burn_steps = st.number_input(
                "Diagnostic burn-in trim",
                min_value=0,
                max_value=1_000_000,
                value=int(st.session_state.get("phot_fit_mcmc_burn_steps", 1000)),
                step=100,
                key="phot_fit_diag_burn_steps",
            )
            thin_by = st.number_input(
                "Diagnostic thinning",
                min_value=1,
                max_value=1000,
                value=int(st.session_state.get("phot_fit_mcmc_thin_by", 1)),
                step=1,
                key="phot_fit_diag_thin_by",
            )
            chain, samples, log_prob, labels, error = _load_mcmc_chain(fit_dir_value, int(burn_steps), int(thin_by))
            if error:
                st.info(error)
            elif chain is not None and samples is not None and log_prob is not None:
                _render_mcmc_diagnostics(fit_dir_value, int(burn_steps), int(thin_by), "phot_fit_outputs")
