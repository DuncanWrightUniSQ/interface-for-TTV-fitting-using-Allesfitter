"""Helpers for building allesfitter settings and parameter skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import streamlit as st


PARAMETER_DOC_URL = "https://www.allesfitter.com/setup/parameters"
SETTINGS_DOC_URL = "https://www.allesfitter.com/setup/settings"
TTV_DOC_URL = "https://www.allesfitter.com/tutorials/11-ttvs"


PARAM_COLUMNS = ["name", "value", "fit", "bounds", "label", "unit", "coupled_with", "truth", "init_err"]
SETTING_COLUMNS = ["name", "value"]


@dataclass(frozen=True)
class FitContext:
    companion: str = "b"
    phot_inst: str = "TESS"
    rv_inst: str = "HARPS"
    include_photometry: bool = True
    include_rv: bool = False
    fit_ttvs: bool = False
    sampler: str = "Nested sampling"
    limb_darkening: str = "quad"
    phot_baseline: str = "sample_offset"
    rv_baseline: str = "sample_offset"
    error_model: str = "sample"


def _row(name: str, value: object, fit: int, bounds: str, label: str, unit: str = "") -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "fit": fit,
        "bounds": bounds,
        "label": label,
        "unit": unit,
        "coupled_with": "",
        "truth": "",
    }


def _baseline_rows(key: str, inst: str, model: str) -> list[dict[str, object]]:
    prefix = f"baseline_{key}_{inst}"
    if model == "sample_offset":
        return [_row(f"baseline_offset_{key}_{inst}", 0.0, 1, "normal 0 0.1", f"{prefix} offset")]
    if model == "sample_linear":
        return [
            _row(f"baseline_offset_{key}_{inst}", 0.0, 1, "normal 0 0.1", f"{prefix} offset"),
            _row(f"baseline_slope_{key}_{inst}", 0.0, 1, "normal 0 1", f"{prefix} slope"),
        ]
    if model == "sample_GP_Matern32":
        return [
            _row(f"baseline_gp_matern32_lnsigma_{key}_{inst}", -7.0, 1, "uniform -20 1", f"{prefix} GP ln sigma"),
            _row(f"baseline_gp_matern32_lnrho_{key}_{inst}", 0.0, 1, "uniform -10 10", f"{prefix} GP ln rho"),
        ]
    if model == "sample_GP_SHO":
        return [
            _row(f"baseline_gp_sho_lnS0_{key}_{inst}", -10.0, 1, "uniform -25 5", f"{prefix} SHO ln S0"),
            _row(f"baseline_gp_sho_lnQ_{key}_{inst}", 1.0, 1, "uniform -5 10", f"{prefix} SHO ln Q"),
            _row(f"baseline_gp_sho_lnomega0_{key}_{inst}", 0.0, 1, "uniform -10 10", f"{prefix} SHO ln omega0"),
        ]
    return []


def build_params(context: FitContext) -> pd.DataFrame:
    companion = context.companion.strip() or "b"
    rows = [
        _row(f"{companion}_rr", 0.1, 1, "uniform 0 1", f"{companion} radius ratio", ""),
        _row(f"{companion}_rsuma", 0.1, 1, "uniform 0 1", f"{companion} (Rstar+Rp)/a", ""),
        _row(f"{companion}_cosi", 0.02, 1, "uniform 0 1", f"{companion} cos inclination", ""),
        _row(f"{companion}_epoch", 0.0, 0 if context.fit_ttvs else 1, "normal 0 0.1", f"{companion} epoch", "d"),
        _row(f"{companion}_period", 1.0, 0 if context.fit_ttvs else 1, "normal 1 0.01", f"{companion} period", "d"),
        _row(f"{companion}_f_c", 0.0, 0, "uniform -1 1", f"{companion} sqrt(e) cos omega", ""),
        _row(f"{companion}_f_s", 0.0, 0, "uniform -1 1", f"{companion} sqrt(e) sin omega", ""),
    ]

    if context.include_rv:
        rows.append(_row(f"{companion}_K", 5.0, 1, "uniform 0 1000", f"{companion} RV semi-amplitude", "m/s"))

    if context.include_photometry:
        inst = context.phot_inst.strip() or "TESS"
        if context.limb_darkening in {"lin", "quad", "sing"}:
            rows.append(_row(f"host_ldc_q1_{inst}", 0.4, 1, "uniform 0 1", f"host q1 {inst}", ""))
        if context.limb_darkening in {"quad", "sing"}:
            rows.append(_row(f"host_ldc_q2_{inst}", 0.3, 1, "uniform 0 1", f"host q2 {inst}", ""))
        if context.limb_darkening == "sing":
            rows.append(_row(f"host_ldc_q3_{inst}", 0.2, 1, "uniform 0 1", f"host q3 {inst}", ""))
        rows.extend(
            [
                _row(f"dil_{inst}", 0.0, 0, "uniform 0 1", f"dilution {inst}", ""),
                _row(f"ln_err_flux_{inst}", -6.9, 0, "uniform -9.2 -4.6", f"flux error scale {inst}", ""),
            ]
        )
        rows.extend(_baseline_rows("flux", inst, context.phot_baseline))

    if context.include_rv:
        inst = context.rv_inst.strip() or "HARPS"
        rows.append(_row(f"ln_jitter_rv_{inst}", 0.0, 1, "uniform -10 10", f"RV jitter {inst}", "ln(m/s)"))
        rows.extend(_baseline_rows("rv", inst, context.rv_baseline))

    if context.fit_ttvs:
        rows.extend(
            [
                _row(f"{companion}_tmid_0000", 0.0, 1, "normal 0 0.02", f"{companion} transit 0 midpoint", "d"),
                _row(f"{companion}_tmid_0001", 1.0, 1, "normal 1 0.02", f"{companion} transit 1 midpoint", "d"),
            ]
        )

    return pd.DataFrame(rows, columns=PARAM_COLUMNS)


def build_settings(context: FitContext) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [
        ("companions_phot", context.companion if context.include_photometry else ""),
        ("companions_rv", context.companion if context.include_rv else ""),
        ("inst_phot", context.phot_inst if context.include_photometry else ""),
        ("inst_rv", context.rv_inst if context.include_rv else ""),
        ("multiprocess", "True"),
        ("multiprocess_cores", "all"),
        ("shift_epoch", "True"),
        (f"host_ld_law_{context.phot_inst}", context.limb_darkening if context.include_photometry else ""),
        (f"{context.companion}_ld_law_{context.phot_inst}", "None" if context.include_photometry else ""),
        (f"baseline_flux_{context.phot_inst}", context.phot_baseline if context.include_photometry else ""),
        (f"baseline_rv_{context.rv_inst}", context.rv_baseline if context.include_rv else ""),
        ("fit_ttvs", str(context.fit_ttvs)),
    ]
    if context.sampler == "MCMC":
        rows.extend([("mcmc_nwalkers", 100), ("mcmc_total_steps", 2000), ("mcmc_burn_steps", 1000)])
    else:
        rows.extend([("ns_modus", "dynamic"), ("ns_nlive", 500), ("ns_tol", 0.01)])
    return pd.DataFrame(rows, columns=SETTING_COLUMNS)


def csv_bytes(df: pd.DataFrame, include_header: bool = True) -> bytes:
    df = df.copy()
    if "fit" in df.columns:
        df["fit"] = df["fit"].map(lambda value: int(bool(value)) if value != "" else "")
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=include_header)
    return buffer.getvalue().encode("utf-8")


def render_docs_note(ttv: bool = False) -> None:
    links = f"[params.csv manual]({PARAMETER_DOC_URL}) | [settings.csv manual]({SETTINGS_DOC_URL})"
    if ttv:
        links += f" | [TTV tutorial]({TTV_DOC_URL})"
    st.caption(links)


def render_context_controls(prefix: str, *, include_photometry: bool, include_rv: bool, fit_ttvs: bool = False) -> FitContext:
    cols = st.columns(3)
    companion = cols[0].text_input("Companion name", value="b", key=f"{prefix}_companion")
    phot_inst = cols[1].text_input("Photometry instrument", value="TESS", key=f"{prefix}_phot_inst", disabled=not include_photometry)
    rv_inst = cols[2].text_input("RV instrument", value="HARPS", key=f"{prefix}_rv_inst", disabled=not include_rv)

    cols = st.columns(4)
    sampler = cols[0].selectbox("Sampler", ["Nested sampling", "MCMC"], key=f"{prefix}_sampler")
    limb_darkening = cols[1].selectbox("Host limb darkening", ["quad", "lin", "sing", "None"], key=f"{prefix}_ld")
    phot_baseline = cols[2].selectbox(
        "Flux baseline",
        ["sample_offset", "sample_linear", "sample_GP_Matern32", "sample_GP_SHO", "None"],
        key=f"{prefix}_phot_baseline",
        disabled=not include_photometry,
    )
    rv_baseline = cols[3].selectbox(
        "RV baseline",
        ["sample_offset", "sample_linear", "sample_GP_Matern32", "sample_GP_SHO", "None"],
        key=f"{prefix}_rv_baseline",
        disabled=not include_rv,
    )

    return FitContext(
        companion=companion,
        phot_inst=phot_inst,
        rv_inst=rv_inst,
        include_photometry=include_photometry,
        include_rv=include_rv,
        fit_ttvs=fit_ttvs,
        sampler=sampler,
        limb_darkening=limb_darkening,
        phot_baseline=phot_baseline,
        rv_baseline=rv_baseline,
    )


def render_config_editors(prefix: str, context: FitContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = build_params(context)
    settings = build_settings(context)

    st.subheader("params.csv")
    st.caption("Editable skeleton using allesfitter's name/value/fit/bounds/label/unit/coupling columns.")
    edited_params = st.data_editor(
        params,
        use_container_width=True,
        num_rows="dynamic",
        key=f"{prefix}_params_editor",
        column_config={"fit": st.column_config.CheckboxColumn("fit", help="Checked means sample this parameter.")},
    )
    st.download_button(
        "Download params.csv",
        data=csv_bytes(edited_params),
        file_name="params.csv",
        mime="text/csv",
        key=f"{prefix}_download_params",
    )

    st.subheader("settings.csv")
    st.caption("Editable two-column skeleton for instruments, companions, sampler, baselines, and TTV mode.")
    edited_settings = st.data_editor(settings, use_container_width=True, num_rows="dynamic", key=f"{prefix}_settings_editor")
    st.download_button(
        "Download settings.csv",
        data=csv_bytes(edited_settings, include_header=False),
        file_name="settings.csv",
        mime="text/csv",
        key=f"{prefix}_download_settings",
    )
    return edited_params, edited_settings
