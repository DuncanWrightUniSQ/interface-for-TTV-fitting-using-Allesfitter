"""Adapters for reusing the richer Allesfitter_work Streamlit workflows."""

from __future__ import annotations

from functools import lru_cache
from io import StringIO
from io import BytesIO
import gc
import math
import multiprocessing as mp
from pathlib import Path
import os
import re
import signal
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .io import normalize_photometry, read_table
from .models import coerce_planet_table, derive_a_over_rstar, derive_inclination_deg, limb_darkened_transit_model


ALLESFITTER_WORK = Path("/Users/u8009283/Documents/Allesfitter_work")
ALLESFITTER_CONDA = ALLESFITTER_WORK / "conda-allesfitter"
PATCHED_MAST_CACHE_VERSION = 9


SIMBAD_TAP_SYNC_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
LOCAL_EXOFOP_TOI_LIST = Path("data/exofop/toi_list.csv")
MAST_PRODUCT_TIMEOUT_SECONDS = 20


def ensure_allesfitter_workflows_available() -> None:
    """Make the existing Allesfitter_work package importable from this app."""
    os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
    if str(ALLESFITTER_WORK) not in sys.path:
        sys.path.insert(0, str(ALLESFITTER_WORK))
    Path(".matplotlib").mkdir(exist_ok=True)
    conda_link = Path("conda-allesfitter")
    if not conda_link.exists() and ALLESFITTER_CONDA.exists():
        try:
            conda_link.symlink_to(ALLESFITTER_CONDA, target_is_directory=True)
        except OSError:
            pass


def import_allesfitter_pages():
    ensure_allesfitter_workflows_available()
    from app import mast, plots  # type: ignore
    from app.pages import photometry_fit, photometry_import, rv_import  # type: ignore

    patch_mast_light_curve_filter(mast, photometry_import)
    patch_photometry_import_sigma_control(photometry_import)
    patch_photometry_fit_exofop_resolution(mast, photometry_fit)
    patch_photometry_loading_workflow(photometry_import, photometry_fit)
    patch_mast_product_download_button(photometry_import)
    patch_photometry_status_tables(photometry_import)
    patch_photometry_fit_prior_bounds(photometry_fit)
    patch_photometry_fit_impact_factor_language(photometry_fit)
    patch_photometry_fit_sampler_controls(photometry_fit)
    return photometry_import, rv_import, photometry_fit


def _extract_sector(value: object) -> int | None:
    text = str(value)
    match = re.search(r"(?:^|[-_])s(\d{3,5})(?:[-_]|$)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _is_light_curve_product(row: pd.Series) -> bool:
    filename = str(row.get("productFilename", "")).lower()
    description = str(row.get("description", "")).lower()
    subgroup = str(row.get("productSubGroupDescription", "")).lower()
    if any(token in filename for token in ["-lc.", "_lc.", "_llc.", "-llc.", "fast-lc"]):
        return True
    if subgroup in {"lc", "fast-lc", "llc"}:
        return True
    if "light curve" in description or "lightcurve" in description:
        return True
    return False


def _product_priority(row: pd.Series) -> tuple[int, int, int, str]:
    filename = str(row.get("productFilename", "")).lower()
    subgroup = str(row.get("productSubGroupDescription", "")).lower()
    provenance = str(row.get("provenance_name", "")).lower()
    # Lower is better. Prefer products that are explicitly light-curve files,
    # then SPOC/mission products over broad FFI HLSPs when exposure is tied.
    explicit_lc = 0 if any(token in filename for token in ["-lc.", "_lc.", "_llc.", "-llc.", "fast-lc"]) else 1
    spoc_rank = 0 if provenance == "spoc" or "spoc" in filename else 1
    subgroup_rank = {"fast-lc": 0, "lc": 1, "llc": 2}.get(subgroup, 3)
    return explicit_lc, spoc_rank, subgroup_rank, filename


def _best_light_curve_products(result) -> pd.DataFrame:
    products = result.products.copy()
    if products.empty:
        return products

    products = products.loc[products.apply(_is_light_curve_product, axis=1)].copy()
    if products.empty:
        return products

    observations = result.observations.copy()
    obs_cols = [col for col in ["obs_id", "sequence_number", "t_exptime", "provenance_name", "obs_collection"] if col in observations.columns]
    if "obs_id" in obs_cols:
        products = products.merge(observations[obs_cols], on="obs_id", how="left", suffixes=("", "_obs"))

    if "sequence_number" not in products.columns:
        products["sequence_number"] = products["productFilename"].map(_extract_sector)
    products["sequence_number"] = pd.to_numeric(products["sequence_number"], errors="coerce")
    products["t_exptime"] = pd.to_numeric(products.get("t_exptime", pd.Series(index=products.index)), errors="coerce")

    # If exposure metadata is missing for a product, infer the common cadence
    # from product naming so it can still compete sensibly.
    filename_lower = products["productFilename"].fillna("").astype(str).str.lower()
    products.loc[products["t_exptime"].isna() & filename_lower.str.contains("fast-lc"), "t_exptime"] = 20
    products.loc[products["t_exptime"].isna() & filename_lower.str.contains(r"[-_]lc\.", regex=True), "t_exptime"] = 120
    products.loc[products["t_exptime"].isna() & filename_lower.str.contains("_llc"), "t_exptime"] = 600

    priority = products.apply(_product_priority, axis=1, result_type="expand")
    priority.columns = ["explicit_lc_rank", "spoc_rank", "subgroup_rank", "filename_rank"]
    products = pd.concat([products, priority], axis=1)

    group_key = "sequence_number"
    if products["sequence_number"].isna().all():
        group_key = "obs_id"

    products = products.sort_values(
        [group_key, "t_exptime", "explicit_lc_rank", "spoc_rank", "subgroup_rank", "filename_rank"],
        na_position="last",
    )
    best = products.groupby(group_key, dropna=False, as_index=False).head(1)
    helper_cols = ["explicit_lc_rank", "spoc_rank", "subgroup_rank", "filename_rank"]
    return best.drop(columns=helper_cols, errors="ignore").reset_index(drop=True)


def _product_observation_indices(observations: pd.DataFrame, max_rows: int = 12) -> list[int]:
    if observations.empty:
        return []
    table = observations.copy()
    if "_mast_row" not in table.columns:
        table["_mast_row"] = table.index
    table["_sequence_sort"] = pd.to_numeric(table.get("sequence_number"), errors="coerce")
    table["_exptime_sort"] = pd.to_numeric(table.get("t_exptime"), errors="coerce")
    provenance = table.get("provenance_name", pd.Series("", index=table.index)).fillna("").astype(str).str.lower()
    obs_id = table.get("obs_id", pd.Series("", index=table.index)).fillna("").astype(str).str.lower()
    table["_mission_rank"] = np.where((provenance == "spoc") | obs_id.str.contains("spoc|tess\\d", regex=True), 0, 1)
    table["_lc_name_rank"] = np.where(obs_id.str.contains("fast|_lc|_llc|-lc", regex=True), 0, 1)
    table = table.sort_values(
        ["_sequence_sort", "_exptime_sort", "_mission_rank", "_lc_name_rank", "_mast_row"],
        na_position="last",
    )
    group_key = "_sequence_sort" if table["_sequence_sort"].notna().any() else "_mast_row"
    selected = table.groupby(group_key, dropna=False, as_index=False).head(1).head(max_rows)
    return [int(value) for value in selected["_mast_row"].tolist()]


def _product_filename_from_mast_uri(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.rstrip("/").rsplit("/", 1)[-1]


def _infer_product_subgroup(filename: str) -> str:
    lower = filename.lower()
    if "fast-lc" in lower:
        return "FAST-LC"
    if "_llc" in lower or "-llc" in lower:
        return "LLC"
    return "LC"


def _products_from_observation_data_urls(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty or "dataURL" not in observations.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in observations.iterrows():
        data_uri = str(row.get("dataURL", "") or "").strip()
        filename = _product_filename_from_mast_uri(data_uri)
        lower_filename = filename.lower()
        obs_id = str(row.get("obs_id", "") or "")
        lower_text = f"{lower_filename} {obs_id.lower()}"
        if not data_uri or not lower_filename.endswith(".fits"):
            continue
        if "dvt" in lower_filename:
            continue
        if not any(token in lower_text for token in ["-lc.", "_lc.", "_llc.", "-llc.", "fast-lc", "_fast-lc"]):
            continue
        rows.append(
            {
                "obs_id": obs_id,
                "productFilename": filename,
                "description": "Light curve FITS",
                "productType": "SCIENCE",
                "productSubGroupDescription": _infer_product_subgroup(filename),
                "size": row.get("size", np.nan),
                "dataURI": data_uri,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["obs_id", "productFilename", "dataURI"]).reset_index(drop=True)


def _mast_product_summary_worker(observation_subset, queue) -> None:
    try:
        from astroquery.mast import Observations

        products = Observations.get_product_list(observation_subset)
        if products is None or len(products) == 0:
            queue.put(("ok", pd.DataFrame()))
            return
        products_df = products.to_pandas()
        keep = [
            col
            for col in [
                "obs_id",
                "productFilename",
                "description",
                "productType",
                "productSubGroupDescription",
                "size",
                "dataURI",
            ]
            if col in products_df.columns
        ]
        products_df = products_df[keep]
        if "description" in products_df:
            light_curve_mask = products_df["description"].fillna("").str.contains("Light curves|Lightcurve|FITS", case=False)
            products_df = products_df[light_curve_mask]
        queue.put(("ok", products_df.reset_index(drop=True)))
    except Exception as exc:  # noqa: BLE001 - returned to parent process
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _product_summary_with_timeout(observation_subset, timeout_seconds: int = MAST_PRODUCT_TIMEOUT_SECONDS) -> pd.DataFrame:
    context = mp.get_context("fork") if hasattr(mp, "get_context") else mp
    queue = context.Queue()
    process = context.Process(target=_mast_product_summary_worker, args=(observation_subset, queue), daemon=True)
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(2)
        raise TimeoutError(f"MAST product-list request exceeded {timeout_seconds} seconds")
    if queue.empty():
        raise RuntimeError("MAST product-list worker exited without returning a result")
    status, payload = queue.get()
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))


def _single_observation_subset(observation_subset, index: int):
    if isinstance(observation_subset, pd.DataFrame):
        return observation_subset.iloc[[index]]
    return observation_subset[[index]]


def _product_summary_resilient(observation_subset, timeout_seconds: int = MAST_PRODUCT_TIMEOUT_SECONDS) -> tuple[pd.DataFrame, str]:
    try:
        count = len(observation_subset)
    except Exception:
        count = 0
    if count <= 1:
        return _product_summary_with_timeout(observation_subset, timeout_seconds), ""

    product_tables: list[pd.DataFrame] = []
    errors: list[str] = []
    per_observation_timeout = max(5, min(8, timeout_seconds // 2))
    for index in range(count):
        try:
            table = _product_summary_with_timeout(_single_observation_subset(observation_subset, index), per_observation_timeout)
        except Exception as exc:  # noqa: BLE001 - keep trying other observations
            errors.append(str(exc))
            continue
        if not table.empty:
            product_tables.append(table)
    if product_tables:
        products = pd.concat(product_tables, ignore_index=True).drop_duplicates().reset_index(drop=True)
        warning = ""
        if errors:
            warning = (
                f"MAST product lists were queried one observation at a time. "
                f"Recovered {len(products)} product row(s); {len(errors)} observation request(s) did not return promptly."
            )
        return products, warning
    return pd.DataFrame(), f"MAST product-list requests did not return promptly for {count} candidate observation(s)."


def patch_mast_light_curve_filter(mast_module, photometry_import_module) -> None:
    if getattr(mast_module, "_ttv_fitter_light_curve_patch", False):
        return

    original_query = mast_module.query_tess_photometry
    original_download = mast_module.download_tess_products

    def resolve_mast_target(target: str) -> str:
        text = " ".join(str(target or "").strip().split())
        normalized = mast_module.normalize_target_name(text)
        if mast_module.TIC_RE.match(text):
            return normalized
        tic = _find_tic_from_local_toi_list(normalized) or _find_tic_from_simbad_identifiers(normalized)
        return tic or normalized

    def query_best_light_curves(target: str, cadence: str = "2 min"):
        query_target = resolve_mast_target(target)
        observation_table = mast_module._query_by_target_name(query_target)
        observations = mast_module._filter_observations(mast_module._to_dataframe(observation_table), cadence)
        if observations.empty:
            products = pd.DataFrame()
        else:
            products = _products_from_observation_data_urls(observations)
            if not products.empty:
                st.session_state.pop("mast_product_warning", None)
            else:
                row_indices = _product_observation_indices(observations)
                if not row_indices:
                    row_indices = observations["_mast_row"].tolist() if "_mast_row" in observations else observations.index.tolist()
                if isinstance(observation_table, pd.DataFrame):
                    observation_subset = observation_table.iloc[row_indices] if len(observation_table) >= len(row_indices) else observation_table
                else:
                    observation_subset = observation_table[row_indices] if len(observation_table) >= len(row_indices) else observation_table
                portal = getattr(mast_module.Observations, "_portal_api_connection", None)
                old_timeout = getattr(portal, "TIMEOUT", None)
                if portal is not None:
                    portal.TIMEOUT = MAST_PRODUCT_TIMEOUT_SECONDS
                try:
                    products, product_warning = _product_summary_resilient(observation_subset)
                    if product_warning:
                        st.session_state["mast_product_warning"] = product_warning
                    else:
                        st.session_state.pop("mast_product_warning", None)
                except Exception as exc:  # noqa: BLE001 - MAST product service can hang or timeout
                    products = pd.DataFrame()
                    st.session_state["mast_product_warning"] = (
                        f"Found {len(observations)} MAST observation row(s) for TIC {query_target}, "
                        f"but the product-list request did not respond within {MAST_PRODUCT_TIMEOUT_SECONDS} seconds: {exc}"
                    )
                finally:
                    if portal is not None and old_timeout is not None:
                        portal.TIMEOUT = old_timeout
        result = mast_module.MastQueryResult(
            target=target,
            resolved_target=query_target,
            observations=observations.drop(columns=["_mast_row"], errors="ignore"),
            products=products,
        )
        products = _best_light_curve_products(result)
        observations = result.observations
        if not products.empty and "obs_id" in products.columns and "obs_id" in observations.columns:
            observations = observations.loc[observations["obs_id"].isin(products["obs_id"])].copy()
        return mast_module.MastQueryResult(
            target=target,
            resolved_target=result.resolved_target,
            observations=observations.reset_index(drop=True),
            products=products,
        )

    def download_best_light_curves(target: str, cadence: str, product_filenames: list[str], download_root: str | Path):
        if not product_filenames:
            return []
        result = query_best_light_curves(target, cadence)
        if result.products.empty or "productFilename" not in result.products.columns:
            return []
        selected = result.products.loc[result.products["productFilename"].astype(str).isin(product_filenames)].copy()
        if selected.empty:
            return []
        selected_status = _selected_product_local_paths(selected, product_filenames, download_root)
        existing = [Path(path) for path in selected_status.get("local_path", pd.Series(dtype=str)).astype(str) if Path(path).exists()]
        missing_names = (
            selected_status.loc[~selected_status["already_downloaded"], "productFilename"].astype(str).tolist()
            if not selected_status.empty and "already_downloaded" in selected_status
            else selected["productFilename"].astype(str).tolist()
        )
        if not missing_names:
            return existing
        selected = selected.loc[selected["productFilename"].astype(str).isin(missing_names)].copy()
        if "dataURI" in selected.columns and selected["dataURI"].fillna("").astype(str).str.len().gt(0).any():
            return existing + _download_products_by_uri(selected, download_root)
        names = selected["productFilename"].astype(str).tolist()
        return existing + _download_products_from_original(mast_module, original_query, result.resolved_target, cadence, names, download_root)

    mast_module.query_tess_photometry = query_best_light_curves
    mast_module.download_tess_products = download_best_light_curves
    photometry_import_module.query_tess_photometry = query_best_light_curves
    photometry_import_module.download_tess_products = download_best_light_curves
    photometry_import_module.CACHE_VERSION = PATCHED_MAST_CACHE_VERSION
    mast_module._ttv_fitter_original_query_tess_photometry = original_query
    mast_module._ttv_fitter_original_download_tess_products = original_download
    mast_module._ttv_fitter_resolve_mast_target = resolve_mast_target
    mast_module._ttv_fitter_light_curve_patch = True


def patch_photometry_import_sigma_control(photometry_import_module) -> None:
    """Hide the global prep-stage outlier sigma control while keeping later review controls."""
    if getattr(photometry_import_module, "_ttv_fitter_sigma_control_patch", False):
        return

    original_render = photometry_import_module.render

    def render_without_global_sigma_slider() -> None:
        original_slider = st.slider
        original_button = st.button

        def slider_without_prep_sigma(label, *args, **kwargs):
            if str(label) == "Outlier sigma clipping":
                return float(kwargs.get("value", 5.0))
            return original_slider(label, *args, **kwargs)

        def button_with_mast_note(label, *args, **kwargs):
            if str(label) == "Query MAST":
                st.caption("When multiple files are available the highest cadence file will be selected.")
            return original_button(label, *args, **kwargs)

        st.slider = slider_without_prep_sigma
        st.button = button_with_mast_note
        try:
            return original_render()
        finally:
            st.slider = original_slider
            st.button = original_button

    photometry_import_module.render = render_without_global_sigma_slider
    photometry_import_module._ttv_fitter_original_render = original_render
    photometry_import_module._ttv_fitter_sigma_control_patch = True


def _staged_workflow_status() -> None:
    stages = pd.DataFrame(
        [
            {
                "stage": "1. Uncertainty review",
                "status": "waiting for sector files",
                "what happens next": "Accept existing uncertainties or estimate them from a quiet region.",
            },
            {
                "stage": "2. Flattening and high-outlier rejection",
                "status": "waiting for uncertainties",
                "what happens next": "Detrend each sector with Wotan while preserving transit dips.",
            },
            {
                "stage": "3. Stitching",
                "status": "waiting for flattened sectors",
                "what happens next": "Combine accepted sectors into one prepared photometry table.",
            },
        ]
    )
    st.dataframe(stages, use_container_width=True, hide_index=True)


def _clear_photometry_preparation_state() -> None:
    for key in [
        "sector_frames",
        "sector_frames_key",
        "sector_uncertainties",
        "sector_flattening",
        "ttv_first_pass_flattened",
        "ttv_first_pass_window",
        "ttv_first_pass_method",
        "ttv_first_pass_cval",
        "ttv_exofop_planets",
        "ttv_exofop_planet_message",
        "ttv_exofop_planet_error",
        "ttv_found_transits",
        "ttv_transit_search_half_width",
        "ttv_second_pass_window",
        "ttv_second_pass_method",
        "ttv_second_pass_cval",
        "prepared_sector_photometry",
        "prepared_sector_summary",
        "prepared_sector_directory",
        "prepared_photometry",
        "prepared_photometry_summary",
    ]:
        st.session_state.pop(key, None)


def _reset_photometry_preparation_workflow() -> None:
    _clear_photometry_preparation_state()
    for key in [
        "uncertainty_sector_select",
        "flattening_sector_select",
        "ttv_first_pass_sector_select",
        "ttv_transit_mask_sector_select",
        "photometry_quality_zero_only",
    ]:
        st.session_state.pop(key, None)


def _workflow_sets_complete(stage_values: dict, sector_frames: dict[str, pd.DataFrame]) -> bool:
    return bool(sector_frames) and set(stage_values) == set(sector_frames)


def _workflow_sets_started(stage_values: dict, sector_frames: dict[str, pd.DataFrame]) -> bool:
    return bool(sector_frames) and bool(set(stage_values) & set(sector_frames))


def _first_pass_cache_dir(photometry_import_module) -> Path:
    target_part = "target"
    target_part_getter = getattr(photometry_import_module, "_target_filename_part", None)
    if callable(target_part_getter):
        try:
            target_part = str(target_part_getter() or target_part)
        except Exception:  # noqa: BLE001 - cache naming should not block detrending
            target_part = "target"
    return Path("data") / "prepared" / target_part / "first_pass"


def _first_pass_cache_path(photometry_import_module, sector_key: str) -> Path:
    clean_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sector_key)).strip("_") or "sector"
    return _first_pass_cache_dir(photometry_import_module) / f"{clean_key}_first_pass.pkl"


def _write_first_pass_cache(
    photometry_import_module,
    sector_key: str,
    first_pass: pd.DataFrame,
    *,
    window_length: float,
    cval: float,
    method: str = "biweight",
) -> dict[str, object]:
    path = _first_pass_cache_path(photometry_import_module, sector_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    first_pass.to_pickle(path)
    return {
        "path": str(path),
        "points": int(len(first_pass)),
        "high_outliers": int(first_pass["is_outlier"].sum()) if "is_outlier" in first_pass else 0,
        "window_length": float(window_length),
        "method": str(method),
        "cval": float(cval),
        "sector": str(first_pass["sector"].iloc[0]) if not first_pass.empty and "sector" in first_pass else str(sector_key),
        "source_file": str(first_pass["source_file"].iloc[0]) if not first_pass.empty and "source_file" in first_pass else "",
    }


def _read_first_pass_cache(first_pass_entry: object) -> pd.DataFrame:
    if isinstance(first_pass_entry, pd.DataFrame):
        return first_pass_entry
    if isinstance(first_pass_entry, dict):
        path = first_pass_entry.get("path")
    else:
        path = first_pass_entry
    if not path:
        return pd.DataFrame()
    cache_path = Path(str(path))
    if not cache_path.exists():
        return pd.DataFrame()
    return pd.read_pickle(cache_path)


def _first_pass_cache_entries_complete(first_pass_entries: dict, sector_frames: dict[str, pd.DataFrame]) -> bool:
    if not _workflow_sets_complete(first_pass_entries, sector_frames):
        return False
    for key in sector_frames:
        entry = first_pass_entries.get(key)
        if isinstance(entry, pd.DataFrame):
            continue
        path = entry.get("path") if isinstance(entry, dict) else entry
        if not path or not Path(str(path)).exists():
            return False
    return True


def _sector_frames_match_paths(paths: list[str]) -> bool:
    quality_zero_only = bool(st.session_state.get("photometry_quality_zero_only", True))
    return st.session_state.get("sector_frames_key") == (tuple(paths), quality_zero_only)


def _render_completed_step(title: str, detail: str, summary: pd.DataFrame | None = None) -> None:
    cols = st.columns([0.76, 0.24], vertical_alignment="center")
    cols[0].subheader(title)
    cols[1].success("Completed")
    with st.expander("Show completed-step summary", expanded=False):
        st.caption(detail)
        if summary is not None and not summary.empty:
            st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_waiting_step(title: str, detail: str) -> None:
    cols = st.columns([0.76, 0.24], vertical_alignment="center")
    cols[0].subheader(title)
    cols[1].info("Waiting")
    st.caption(detail)


def _data_uncertainty_medians_by_sector(sector_frames: dict[str, pd.DataFrame], normalized_sector_func) -> dict[str, float]:
    medians: dict[str, float] = {}
    for sector_key, frame in sector_frames.items():
        normalized = normalized_sector_func(frame)
        if "flux_err" not in normalized.columns:
            continue
        finite_err = pd.to_numeric(normalized["flux_err"], errors="coerce").replace([float("inf"), -float("inf")], np.nan).dropna()
        finite_err = finite_err[finite_err > 0]
        if not finite_err.empty:
            medians[sector_key] = float(finite_err.median())
    return medians


def _render_uncertainty_workflow_with_bulk_accept(photometry_import_module, paths: list[str], sigma_clip: float, stitch: bool) -> None:
    if not paths:
        return

    st.subheader("1. Examine or determine sector uncertainties")
    st.caption(
        "Each sector needs an adopted uncertainty before stitching. If the FITS file includes usable "
        "uncertainties, you can inspect and accept them. Otherwise select a quiet, structure-free "
        "region of the detrended light curve; points more than 5 sigma from that region are rejected "
        "before estimating the sector uncertainty."
    )
    quality_zero_only = st.checkbox("Use QUALITY == 0 only", value=True, key="photometry_quality_zero_only")
    sector_frames = photometry_import_module._load_sector_frames(paths, quality_zero_only)
    sector_uncertainties = st.session_state.setdefault("sector_uncertainties", {})
    status = photometry_import_module.sector_uncertainty_table(sector_frames, sector_uncertainties)
    if not status.empty:
        st.dataframe(photometry_import_module._style_status_table(status), use_container_width=True, hide_index=True)

    if not sector_frames:
        return

    data_medians = _data_uncertainty_medians_by_sector(sector_frames, photometry_import_module.normalized_sector)
    complete = set(sector_uncertainties)
    sector_options = list(sector_frames.keys())

    def sector_label(key: str) -> str:
        if key in complete:
            return f"🟢 {key}"
        if key in data_medians:
            return f"{key}"
        return f"{key} (needs estimate)"

    sector_key = st.selectbox(
        "Sector to review",
        sector_options,
        key="uncertainty_sector_select",
        format_func=sector_label,
    )
    frame = photometry_import_module.normalized_sector(sector_frames[sector_key])
    finite_err = pd.Series(frame["flux_err"]).replace([float("inf"), -float("inf")], pd.NA).dropna()
    finite_err = finite_err[finite_err > 0]
    has_uncertainties = not finite_err.empty

    mode_options = ["Determine uncertainty from a quiet region"]
    if has_uncertainties:
        mode_options.insert(0, "Examine uncertainties from data")
    mode = st.radio("Uncertainty action", mode_options, horizontal=True, key=f"uncertainty_mode_{sector_key}")

    if mode == "Examine uncertainties from data":
        st.plotly_chart(
            photometry_import_module.sector_uncertainty_preview(frame),
            use_container_width=True,
            config=photometry_import_module.PLOT_CONFIG,
            key=f"sector_uncertainty_data_{sector_key}",
        )
        median_uncertainty = float(finite_err.median())
        st.metric("Median uncertainty in data", f"{median_uncertainty:.6g}")
        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button("Accept data uncertainties for this sector", key=f"accept_data_unc_{sector_key}", use_container_width=True):
                sector_uncertainties[sector_key] = median_uncertainty
                st.session_state["sector_uncertainties"] = sector_uncertainties
                st.success(f"Sector uncertainty set to {median_uncertainty:.6g}.")
                st.rerun()
        with action_cols[1]:
            if st.button(
                "Accept data uncertainties for all sectors where uncertainties are provided",
                key="accept_data_unc_all_sectors",
                use_container_width=True,
                disabled=not bool(data_medians),
            ):
                sector_uncertainties.update(data_medians)
                st.session_state["sector_uncertainties"] = sector_uncertainties
                st.success(f"Accepted data uncertainties for {len(data_medians)} sector(s).")
                st.rerun()
        return

    window_length = st.slider(
        "Wotan detrending window [days]",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.05,
        key=f"wotan_window_{sector_key}",
    )
    trend = photometry_import_module.wotan_trend(sector_frames[sector_key], window_length)
    st.caption("Press the selection control in the Plotly toolbar, then drag over a quiet region with typical noise and no obvious structure.")
    selection_state = st.plotly_chart(
        photometry_import_module.sector_uncertainty_preview(frame, trend=trend),
        use_container_width=True,
        config=photometry_import_module.PLOT_CONFIG,
        key=f"sector_uncertainty_select_{sector_key}",
        on_select="rerun",
        selection_mode=("box", "lasso"),
    )
    selected_range = photometry_import_module._selection_x_range(selection_state)
    if selected_range is None:
        st.info("Select a quiet region of this sector to estimate the uncertainty.")
        return

    region, uncertainty = photometry_import_module.estimate_uncertainty_for_region(
        sector_frames[sector_key],
        selected_range[0],
        selected_range[1],
        window_length=window_length,
        sigma_clip=5.0,
    )
    st.plotly_chart(
        photometry_import_module.uncertainty_residual_preview(region),
        use_container_width=True,
        config=photometry_import_module.PLOT_CONFIG,
        key=f"uncertainty_residuals_{sector_key}",
    )
    kept = int((~region.get("uncertainty_outlier", pd.Series(dtype=bool))).sum())
    rejected = int(region.get("uncertainty_outlier", pd.Series(dtype=bool)).sum())
    cols = st.columns(3)
    cols[0].metric("Selected points", f"{len(region):,}")
    cols[1].metric("Used points", f"{kept:,}")
    cols[2].metric("Rejected >5 sigma", f"{rejected:,}")
    if pd.notna(uncertainty):
        st.metric("Estimated sector uncertainty", f"{uncertainty:.6g}")
        if st.button("Apply uncertainty to this sector", key=f"apply_uncertainty_{sector_key}"):
            sector_uncertainties[sector_key] = uncertainty
            st.session_state["sector_uncertainties"] = sector_uncertainties
            st.success(f"Sector uncertainty set to {uncertainty:.6g}.")
            st.rerun()


@lru_cache(maxsize=1)
def _statsmodels_available() -> bool:
    try:
        import statsmodels  # noqa: F401
    except Exception:
        return False
    return True


def _available_wotan_method(method: str) -> tuple[str, str]:
    requested = "huber" if str(method).lower().startswith("huber") else "biweight"
    if requested == "huber" and not _statsmodels_available():
        return (
            "biweight",
            "Huber smoothing requires the optional statsmodels package, so this sector is using biweight smoothing instead.",
        )
    return requested, ""


def _phase_distance(time: np.ndarray, t0: float, period: float) -> np.ndarray:
    return np.abs(((np.asarray(time, dtype=float) - float(t0) + 0.5 * float(period)) % float(period)) - 0.5 * float(period))


def _align_time_to_frame(t0: float, time: np.ndarray) -> float:
    if not np.isfinite(t0) or time.size == 0:
        return float(t0)
    median_time = float(np.nanmedian(time))
    if median_time < 100000 and t0 > 2400000:
        return float(t0 - 2457000.0)
    if median_time > 2400000 and t0 < 100000:
        return float(t0 + 2457000.0)
    return float(t0)


def _generated_ephemerides_from_params(params: pd.DataFrame | None) -> list[dict[str, float]]:
    if not isinstance(params, pd.DataFrame) or params.empty or not {"name", "value"}.issubset(params.columns):
        return []
    values: dict[str, float] = {}
    for _, row in params.iterrows():
        try:
            values[str(row["name"])] = float(row["value"])
        except (TypeError, ValueError):
            continue
    ephemerides = []
    for label in sorted({name[: -len("_period")] for name in values if name.endswith("_period")}):
        ephemerides.append(
            {
                "t0": values.get(f"{label}_epoch", np.nan),
                "period": values.get(f"{label}_period", np.nan),
                "duration_hours": values.get(f"{label}_duration", np.nan),
            }
        )
    return ephemerides


def _transit_mask_for_ephemerides(
    frame: pd.DataFrame,
    ephemerides: list[dict[str, float]],
    *,
    width_durations: float = 2.0,
) -> np.ndarray:
    if frame.empty or "time" not in frame:
        return np.zeros(len(frame), dtype=bool)
    time = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(time.size, dtype=bool)
    width_durations = max(float(width_durations), 0.1)
    for ephemeris in ephemerides:
        period = _coerce_float(ephemeris.get("period"), np.nan)
        t0 = _coerce_float(ephemeris.get("t0"), np.nan)
        duration_hours = _coerce_float(ephemeris.get("duration_hours"), np.nan)
        if not np.isfinite(period) or period <= 0 or not np.isfinite(t0) or not np.isfinite(duration_hours) or duration_hours <= 0:
            continue
        aligned_t0 = _align_time_to_frame(float(t0), time)
        half_width_days = 0.5 * (float(duration_hours) / 24.0) * width_durations
        if not np.isfinite(half_width_days) or half_width_days <= 0:
            continue
        mask |= _phase_distance(time, aligned_t0, float(period)) <= half_width_days
    return mask


def _duration_hours_from_exofop_row(row: pd.Series) -> float:
    for column in ["Duration (hours)", "Duration (hrs)", "Duration (days)", "Transit Duration (hours)", "Transit Duration (days)"]:
        duration = _coerce_float(row.get(column, np.nan), np.nan)
        if np.isfinite(duration):
            return float(duration * 24.0) if "days" in column.lower() else float(duration)
    return np.nan


def _exofop_ephemerides_from_matches(matches: pd.DataFrame | None) -> list[dict[str, float]]:
    if not isinstance(matches, pd.DataFrame) or matches.empty:
        return []
    ephemerides: list[dict[str, float]] = []
    for index, row in matches.iterrows():
        t0 = _coerce_float(row.get("Epoch (BJD)", np.nan), np.nan)
        period = _coerce_float(row.get("Period (days)", np.nan), np.nan)
        duration_hours = _duration_hours_from_exofop_row(row)
        if not np.isfinite(t0) or not np.isfinite(period) or period <= 0 or not np.isfinite(duration_hours) or duration_hours <= 0:
            continue
        planet_label = _clean_exofop_planet_label(row, len(ephemerides))
        ephemerides.append(
            {
                "planet": planet_label,
                "t0": float(t0),
                "period": float(period),
                "duration_hours": float(duration_hours),
            }
        )
    return ephemerides


def _target_name_for_exofop_lookup() -> str:
    for key in [
        "mast_resolved_target",
        "target_name",
        "photometry_target_input",
        "mast_target_input",
        "phot_fit_exofop_tic_input",
    ]:
        value = str(st.session_state.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _planet_color(index: int) -> str:
    colors = ["#ef4444", "#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#0891b2", "#db2777", "#65a30d"]
    return colors[int(index) % len(colors)]


def _planet_letter(index: int) -> str:
    return chr(ord("b") + int(index))


def _clean_exofop_planet_label(row: pd.Series | dict, index: int) -> str:
    for column in ["Planet Name", "Planet", "Planet Letter", "pl_letter", "letter"]:
        value = row.get(column, None)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "null"}:
                return text[-1].lower() if len(text) == 1 else text
    for column in ["TOI", "TOI Number"]:
        value = row.get(column, None)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        match = re.search(r"\.(\d+)$", text)
        if match:
            return _planet_letter(max(int(match.group(1)) - 1, 0))
    return _planet_letter(index)


def _exofop_planet_rows_from_matches(matches: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(matches, pd.DataFrame) or matches.empty:
        return pd.DataFrame()
    rows = []
    for index, row in matches.iterrows():
        t0 = _coerce_float(row.get("Epoch (BJD)", np.nan), np.nan)
        period = _coerce_float(row.get("Period (days)", np.nan), np.nan)
        duration_hours = _duration_hours_from_exofop_row(row)
        if not np.isfinite(t0) or not np.isfinite(period) or period <= 0 or not np.isfinite(duration_hours) or duration_hours <= 0:
            continue
        stellar_radius = _coerce_float(row.get("Stellar Radius (R_Sun)", np.nan), np.nan)
        planet_radius = _coerce_float(row.get("Planet Radius (R_Earth)", np.nan), np.nan)
        radius_ratio = np.nan
        if np.isfinite(planet_radius) and np.isfinite(stellar_radius) and stellar_radius > 0:
            radius_ratio = planet_radius * 0.0091577 / stellar_radius
        depth_ppm = _coerce_float(row.get("Depth (ppm)", np.nan), np.nan)
        if not np.isfinite(radius_ratio) and np.isfinite(depth_ppm):
            radius_ratio = math.sqrt(max(depth_ppm, 0.0) / 1_000_000.0)
        if not np.isfinite(radius_ratio) or radius_ratio <= 0:
            radius_ratio = 0.05
        planet_label = _clean_exofop_planet_label(row, len(rows))
        rows.append(
            {
                "planet": planet_label,
                "t0": float(t0),
                "period": float(period),
                "duration_hours": float(duration_hours),
                "radius_ratio": float(radius_ratio),
                "impact": 0.5,
                "limb_darkening_u1": 0.5,
                "limb_darkening_u2": 0.1,
                "mask_duration_multiplier": 2.0,
                "color": _planet_color(len(rows)),
            }
        )
    return pd.DataFrame(rows)


def _exofop_planet_rows_from_ephemerides(ephemerides: list[dict[str, float]]) -> pd.DataFrame:
    rows = []
    for index, eph in enumerate(ephemerides):
        planet = eph.get("planet", "")
        if pd.isna(planet) or not str(planet).strip() or str(planet).strip().lower() in {"nan", "none", "null"}:
            planet = _planet_letter(index)
        rows.append(
            {
                "planet": planet,
                "t0": _coerce_float(eph.get("t0"), np.nan),
                "period": _coerce_float(eph.get("period"), np.nan),
                "duration_hours": _coerce_float(eph.get("duration_hours"), np.nan),
                "radius_ratio": _coerce_float(eph.get("radius_ratio"), 0.05),
                "impact": _coerce_float(eph.get("impact"), 0.5),
                "limb_darkening_u1": _coerce_float(eph.get("limb_darkening_u1"), 0.5),
                "limb_darkening_u2": _coerce_float(eph.get("limb_darkening_u2"), 0.1),
                "mask_duration_multiplier": _coerce_float(eph.get("mask_duration_multiplier"), 2.0),
                "color": eph.get("color", _planet_color(index)),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.loc[np.isfinite(pd.to_numeric(table["period"], errors="coerce")) & (pd.to_numeric(table["period"], errors="coerce") > 0)].reset_index(drop=True)


def _ephemerides_from_planet_rows(planets: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    if not isinstance(planets, pd.DataFrame) or planets.empty:
        return rows
    for _, row in planets.iterrows():
        record = row.to_dict()
        if np.isfinite(_coerce_float(record.get("t0"), np.nan)) and np.isfinite(_coerce_float(record.get("period"), np.nan)):
            rows.append(record)
    return rows


def _flatten_sector_with_explicit_mask(
    photometry_import_module,
    frame: pd.DataFrame,
    uncertainty: float,
    *,
    window_length: float,
    method: str = "biweight",
    transit_mask: np.ndarray | None = None,
    cval: float = 5.0,
    sigma_clip: float = 5.0,
) -> pd.DataFrame:
    prepared = photometry_import_module.normalized_sector(frame)
    method, _ = _available_wotan_method(method)
    trend = _wotan_trend_with_method(photometry_import_module, frame, window_length, method, mask=transit_mask, cval=cval)
    prepared["trend"] = trend
    prepared["wotan_method"] = method
    prepared["wotan_cval"] = float(cval)
    prepared["wotan_transit_mask"] = transit_mask if transit_mask is not None else False
    prepared["flux_before_flatten"] = prepared["flux"]
    safe_trend = np.where(np.isfinite(trend) & (trend != 0), trend, np.nan)
    prepared["flux"] = prepared["flux_before_flatten"] / safe_trend
    prepared["flux_err"] = uncertainty / safe_trend
    residual = prepared["flux_before_flatten"].to_numpy(dtype=float) - safe_trend
    scatter = float(uncertainty) if np.isfinite(uncertainty) and uncertainty > 0 else float(np.nanmedian(np.abs(residual - np.nanmedian(residual))) * 1.4826)
    prepared["is_outlier"] = residual > sigma_clip * scatter if np.isfinite(scatter) and scatter > 0 else False
    return prepared


def _combined_transit_model(time: np.ndarray, events: list[dict[str, object]], centers: np.ndarray) -> np.ndarray:
    model = np.ones_like(np.asarray(time, dtype=float), dtype=float)
    for idx, event in enumerate(events):
        planet = event["planet"]
        shape = limb_darkened_transit_model(
            time,
            max(float(planet["period"]), 1e-8),
            float(centers[idx]),
            max(float(planet.get("radius_ratio", 0.05)), 1e-6),
            float(planet.get("impact", 0.5)),
            max(float(planet.get("duration_hours", 2.0)), 1e-4),
            float(planet.get("limb_darkening_u1", 0.5)),
            float(planet.get("limb_darkening_u2", 0.1)),
            baseline_offset=0.0,
        )
        model += shape - 1.0
    return model


def _predicted_events_for_sector(frame: pd.DataFrame, planets: pd.DataFrame, search_half_width_days: float) -> list[dict[str, object]]:
    if frame.empty or planets.empty or "time" not in frame:
        return []
    time = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    finite = time[np.isfinite(time)]
    if finite.size == 0:
        return []
    tmin, tmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    events: list[dict[str, object]] = []
    for planet_index, row in planets.reset_index(drop=True).iterrows():
        period = _coerce_float(row.get("period"), np.nan)
        t0 = _coerce_float(row.get("t0"), np.nan)
        duration_hours = _coerce_float(row.get("duration_hours"), np.nan)
        if not np.isfinite(period) or period <= 0 or not np.isfinite(t0) or not np.isfinite(duration_hours):
            continue
        aligned_t0 = _align_time_to_frame(float(t0), finite)
        first_epoch = int(np.floor((tmin - search_half_width_days - aligned_t0) / period)) - 1
        last_epoch = int(np.ceil((tmax + search_half_width_days - aligned_t0) / period)) + 1
        planet = row.to_dict()
        planet["t0"] = aligned_t0
        planet["planet_index"] = planet_index
        planet["color"] = planet.get("color", _planet_color(planet_index))
        for epoch in range(first_epoch, last_epoch + 1):
            expected = aligned_t0 + epoch * period
            duration_days = max(float(duration_hours) / 24.0, 1e-5)
            if expected + search_half_width_days + duration_days < tmin or expected - search_half_width_days - duration_days > tmax:
                continue
            events.append({"planet": planet, "epoch": epoch, "expected_tmid": expected, "tmid": expected})
    return sorted(events, key=lambda event: float(event["expected_tmid"]))


def _refine_transits_in_sector(first_pass: pd.DataFrame, planets: pd.DataFrame, search_half_width_days: float, grid_step_days: float = 0.01) -> pd.DataFrame:
    events = _predicted_events_for_sector(first_pass, planets, search_half_width_days)
    if not events:
        return pd.DataFrame()
    time = pd.to_numeric(first_pass["time"], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(first_pass["flux"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]
    if time.size == 0:
        return pd.DataFrame()

    centers = np.array([float(event["expected_tmid"]) for event in events], dtype=float)
    offsets = np.arange(-float(search_half_width_days), float(search_half_width_days) + grid_step_days / 2.0, grid_step_days)
    for _iteration in range(2):
        for idx, event in enumerate(events):
            planet = event["planet"]
            duration_days = max(float(planet.get("duration_hours", 2.0)) / 24.0, 1e-4)
            local_half = max(float(search_half_width_days) + 1.5 * duration_days, 2.0 * duration_days, 0.05)
            local = np.abs(time - centers[idx]) <= local_half
            if np.count_nonzero(local) < 5:
                continue
            best_center = centers[idx]
            best_sse = np.inf
            for offset in offsets:
                trial_centers = centers.copy()
                trial_centers[idx] = float(event["expected_tmid"]) + float(offset)
                model = _combined_transit_model(time[local], events, trial_centers)
                sse = float(np.nansum((flux[local] - model) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best_center = trial_centers[idx]
            centers[idx] = best_center

    rows = []
    for idx, event in enumerate(events):
        planet = event["planet"]
        duration_days = max(float(planet.get("duration_hours", 2.0)) / 24.0, 1e-5)
        mask_multiplier = max(float(planet.get("mask_duration_multiplier", 2.0)), 0.1)
        half_width = 0.5 * mask_multiplier * duration_days
        points = int(np.count_nonzero(np.abs(time - centers[idx]) <= half_width))
        rows.append(
            {
                "planet": planet.get("planet", ""),
                "planet_index": int(planet.get("planet_index", 0)),
                "epoch": int(event["epoch"]),
                "expected_tmid": float(event["expected_tmid"]),
                "tmid": float(centers[idx]),
                "offset_days": float(centers[idx] - float(event["expected_tmid"])),
                "duration_hours": float(planet.get("duration_hours", np.nan)),
                "mask_duration_multiplier": mask_multiplier,
                "mask_half_width_days": half_width,
                "points": points,
                "color": planet.get("color", _planet_color(int(planet.get("planet_index", 0)))),
            }
        )
    return pd.DataFrame(rows)


def _mask_from_found_transits(frame: pd.DataFrame, found: pd.DataFrame) -> np.ndarray:
    if frame.empty or found is None or found.empty or "time" not in frame:
        return np.zeros(len(frame), dtype=bool)
    time = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
    mask = np.zeros(time.size, dtype=bool)
    for _, row in found.iterrows():
        tmid = _coerce_float(row.get("tmid"), np.nan)
        half_width = _coerce_float(row.get("mask_half_width_days"), np.nan)
        if np.isfinite(tmid) and np.isfinite(half_width) and half_width > 0:
            mask |= np.abs(time - tmid) <= half_width
    return mask


def _transit_search_preview_figure(frame: pd.DataFrame, found: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    display = _sample_photometry_for_plot(frame, max_points=18_000) if len(frame) > 18_000 else frame
    fig.add_trace(
        go.Scattergl(
            x=display["time"],
            y=display["flux"],
            mode="markers",
            marker=dict(size=3, color="#2563eb", opacity=0.32),
            name="first-pass detrended flux",
        )
    )
    if found is not None and not found.empty:
        full_time = pd.to_numeric(frame["time"], errors="coerce").to_numpy(dtype=float)
        for planet, group in found.groupby("planet", dropna=False):
            planet_mask = np.zeros(len(frame), dtype=bool)
            color = str(group["color"].iloc[0]) if "color" in group else "#f59e0b"
            for _, row in group.iterrows():
                tmid = _coerce_float(row.get("tmid"), np.nan)
                half_width = _coerce_float(row.get("mask_half_width_days"), np.nan)
                if np.isfinite(tmid) and np.isfinite(half_width) and half_width > 0:
                    planet_mask |= np.abs(full_time - tmid) <= half_width
            masked = frame.loc[planet_mask]
            if not masked.empty:
                masked_display = _sample_photometry_for_plot(masked, max_points=4_000) if len(masked) > 4_000 else masked
                fig.add_trace(
                    go.Scattergl(
                        x=masked_display["time"],
                        y=masked_display["flux"],
                        mode="markers",
                        marker=dict(size=4, color=color, opacity=0.6),
                        name=f"masked {planet}",
                    )
                )
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=40, b=45),
        xaxis_title="Time - BTJD",
        yaxis_title="First-pass flattened flux",
        uirevision="ttv_transit_search_preview",
    )
    return fig


def _simple_display_stride(point_count: int, max_display_points: int = 20_000) -> int:
    if point_count <= max_display_points:
        return 1
    return max(_simple_plot_stride(point_count), int(math.ceil(point_count / max_display_points)))


def _stride_note(point_count: int, displayed_points: int, stride: int) -> str:
    return f"Only every {_ordinal_word(stride)} data point is plotted to save memory ({displayed_points:,} of {point_count:,} points shown)."


def _first_pass_preview_figure(first_pass: pd.DataFrame) -> tuple[go.Figure, str]:
    fig = go.Figure()
    if first_pass is None or first_pass.empty:
        return fig, ""
    stride = _simple_display_stride(len(first_pass), max_display_points=20_000)
    display = first_pass.iloc[::stride].copy() if stride > 1 else first_pass
    flux_column = "flux_before_flatten" if "flux_before_flatten" in display.columns else "flux"
    fig.add_trace(
        go.Scatter(
            x=display["time"],
            y=display[flux_column],
            mode="markers",
            marker={"size": 3, "color": "#2563eb", "opacity": 0.32},
            name="Flux",
        )
    )
    if "trend" in display.columns:
        fig.add_trace(
            go.Scatter(
                x=display["time"],
                y=display["trend"],
                mode="lines",
                line={"color": "#ef4444", "width": 3},
                name="Wotan trend",
            )
        )
    note = _stride_note(len(first_pass), len(display), stride) if stride > 1 else ""
    if note:
        fig.add_annotation(
            text=note,
            x=0.01,
            y=1.06,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font={"size": 12, "color": "#64748b"},
        )
        fig.update_layout(meta={"ttv_sampled_plot_note": note})
    fig.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=35, b=45),
        xaxis_title="Time - BTJD",
        yaxis_title="Relative flux",
        uirevision="ttv_first_pass_preview",
    )
    return fig, note


def _wotan_trend_with_method(
    photometry_import_module,
    frame: pd.DataFrame,
    window_length: float,
    method: str = "biweight",
    *,
    mask: np.ndarray | None = None,
    break_tolerance: float | None = None,
    cval: float = 5.0,
) -> np.ndarray:
    from wotan import flatten

    method, _ = _available_wotan_method(method)
    normalized = photometry_import_module.normalized_sector(frame)
    time = normalized["time"].to_numpy(dtype=float)
    flux = normalized["flux"].to_numpy(dtype=float)
    break_tolerance = float(window_length) / 2.0 if break_tolerance is None else float(break_tolerance)
    try:
        _, trend = flatten(
            time,
            flux,
            window_length=window_length,
            break_tolerance=break_tolerance,
            method=method,
            mask=mask,
            return_trend=True,
            cval=float(cval),
        )
    except ImportError as exc:
        if method == "huber" and "statsmodels" in str(exc).lower():
            _, trend = flatten(
                time,
                flux,
                window_length=window_length,
                break_tolerance=break_tolerance,
                method="biweight",
                mask=mask,
                return_trend=True,
                cval=float(cval),
            )
        else:
            raise
    return np.asarray(trend, dtype=float)


def _flatten_sector_with_method(
    photometry_import_module,
    frame: pd.DataFrame,
    uncertainty: float,
    *,
    window_length: float,
    method: str = "biweight",
    mask_transits: bool = False,
    mask_ephemerides: list[dict[str, float]] | None = None,
    mask_width_durations: float = 1.5,
    cval: float = 5.0,
    sigma_clip: float = 5.0,
) -> pd.DataFrame:
    prepared = photometry_import_module.normalized_sector(frame)
    method, _ = _available_wotan_method(method)
    transit_mask = _transit_mask_for_ephemerides(frame, mask_ephemerides or [], width_durations=mask_width_durations) if mask_transits else None
    trend = _wotan_trend_with_method(photometry_import_module, frame, window_length, method, mask=transit_mask, cval=cval)
    prepared["trend"] = trend
    prepared["wotan_method"] = method
    prepared["wotan_cval"] = float(cval)
    prepared["wotan_transit_mask"] = transit_mask if transit_mask is not None else False
    prepared["flux_before_flatten"] = prepared["flux"]

    safe_trend = np.where(np.isfinite(trend) & (trend != 0), trend, np.nan)
    prepared["flux"] = prepared["flux_before_flatten"] / safe_trend
    prepared["flux_err"] = uncertainty / safe_trend

    residual = prepared["flux_before_flatten"].to_numpy(dtype=float) - safe_trend
    scatter = float(uncertainty) if np.isfinite(uncertainty) and uncertainty > 0 else float(np.nanmedian(np.abs(residual - np.nanmedian(residual))) * 1.4826)
    if np.isfinite(scatter) and scatter > 0:
        prepared["is_outlier"] = residual > sigma_clip * scatter
    else:
        prepared["is_outlier"] = False
    return prepared


def _flattening_status_table_with_method(sector_frames: dict[str, pd.DataFrame], sector_flattening: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, frame in sector_frames.items():
        settings = sector_flattening.get(key, {})
        rows.append(
            {
                "status": "complete" if key in sector_flattening else "incomplete",
                "sector": frame["sector"].iloc[0] if not frame.empty and "sector" in frame else "",
                "source_file": frame["source_file"].iloc[0] if not frame.empty and "source_file" in frame else "",
                "points": len(frame),
                "wotan_window_days": settings.get("window_length", np.nan),
                "wotan_method": settings.get("method", "biweight") if key in sector_flattening else "",
                "wotan_cval": settings.get("cval", np.nan),
                "mask_transits": settings.get("mask_transits", np.nan),
                "mask_planets": settings.get("mask_planets", np.nan),
                "mask_width_durations": settings.get("mask_width_durations", np.nan),
                "masked_points": settings.get("masked_points", np.nan),
                "high_outliers": settings.get("high_outliers", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _stitch_flattened_sectors_with_method(
    photometry_import_module,
    sector_frames: dict[str, pd.DataFrame],
    sector_uncertainties: dict[str, float],
    sector_flattening: dict[str, dict],
    *,
    sigma_clip: float = 5.0,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summary_rows = []
    sector_items = list(sector_frames.items())
    total_sectors = len(sector_items)
    processed = 0
    for key, frame in sector_items:
        if key not in sector_uncertainties or key not in sector_flattening:
            continue
        processed += 1
        if callable(progress_callback):
            progress_callback(processed, total_sectors, key, frame)
        settings = sector_flattening[key]
        window_length = float(settings["window_length"])
        method = str(settings.get("method", "biweight"))
        cval = float(settings.get("cval", 5.0))
        found_records = settings.get("mask_found_transits", [])
        cached_first_pass = settings.get("first_pass_cache_path") if settings.get("use_cached_first_pass") else None
        if cached_first_pass:
            prepared = _read_first_pass_cache(cached_first_pass)
            if prepared.empty:
                continue
        elif found_records:
            found = pd.DataFrame(found_records)
            transit_mask = _mask_from_found_transits(frame, found)
            prepared = _flatten_sector_with_explicit_mask(
                photometry_import_module,
                frame,
                sector_uncertainties[key],
                window_length=window_length,
                method=method,
                transit_mask=transit_mask,
                cval=cval,
                sigma_clip=sigma_clip,
            )
        else:
            prepared = _flatten_sector_with_method(
                photometry_import_module,
                frame,
                sector_uncertainties[key],
                window_length=window_length,
                method=method,
                mask_transits=bool(settings.get("mask_transits", False)),
                mask_ephemerides=settings.get("mask_ephemerides", []),
                mask_width_durations=float(settings.get("mask_width_durations", 1.5)),
                cval=cval,
                sigma_clip=sigma_clip,
            )
        frames.append(prepared)
        summary_rows.append(
            {
                "source_file": prepared["source_file"].iloc[0] if "source_file" in prepared else "",
                "sector": prepared["sector"].iloc[0] if "sector" in prepared else "",
                "points_after_quality": len(prepared),
                "high_outliers": int(prepared["is_outlier"].sum()),
                "adopted_uncertainty": sector_uncertainties[key],
                "wotan_window_days": window_length,
                "wotan_method": method,
                "wotan_cval": cval,
                "mask_transits": bool(settings.get("mask_transits", False)),
                "mask_planets": int(settings.get("mask_planets", len(settings.get("mask_ephemerides", [])))),
                "mask_width_durations": float(settings.get("mask_width_durations", 1.5)),
                "masked_points": int(pd.Series(prepared.get("wotan_transit_mask", False)).sum()),
                "found_transits": int(settings.get("found_transits", len(found_records))),
                "flux_column": prepared["flux_column"].iloc[0] if "flux_column" in prepared else "",
                "err_column": prepared["err_column"].iloc[0] if "err_column" in prepared else "",
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    stitched = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    return stitched, summary


def _prepared_sector_tables_with_method(
    photometry_import_module,
    sector_frames: dict[str, pd.DataFrame],
    sector_uncertainties: dict[str, float],
    sector_flattening: dict[str, dict],
    *,
    sigma_clip: float = 5.0,
    progress_callback=None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sector_tables: dict[str, pd.DataFrame] = {}
    summary_rows = []
    sector_items = list(sector_frames.items())
    total_sectors = len(sector_items)
    processed = 0
    for key, frame in sector_items:
        if key not in sector_uncertainties or key not in sector_flattening:
            continue
        processed += 1
        if callable(progress_callback):
            progress_callback(processed, total_sectors, key, frame)
        settings = sector_flattening[key]
        window_length = float(settings["window_length"])
        method = str(settings.get("method", "biweight"))
        cval = float(settings.get("cval", 5.0))
        found_records = settings.get("mask_found_transits", [])
        cached_first_pass = settings.get("first_pass_cache_path") if settings.get("use_cached_first_pass") else None
        if cached_first_pass:
            prepared = _read_first_pass_cache(cached_first_pass)
            if prepared.empty:
                continue
        elif found_records:
            transit_mask = _mask_from_found_transits(frame, pd.DataFrame(found_records))
            prepared = _flatten_sector_with_explicit_mask(
                photometry_import_module,
                frame,
                sector_uncertainties[key],
                window_length=window_length,
                method=method,
                transit_mask=transit_mask,
                cval=cval,
                sigma_clip=sigma_clip,
            )
        else:
            prepared = _flatten_sector_with_method(
                photometry_import_module,
                frame,
                sector_uncertainties[key],
                window_length=window_length,
                method=method,
                mask_transits=bool(settings.get("mask_transits", False)),
                mask_ephemerides=settings.get("mask_ephemerides", []),
                mask_width_durations=float(settings.get("mask_width_durations", 1.5)),
                cval=cval,
                sigma_clip=sigma_clip,
            )
        sector_label = str(prepared["sector"].iloc[0]) if "sector" in prepared and not prepared.empty else str(key)
        sector_tables[key] = prepared.sort_values("time").reset_index(drop=True)
        summary_rows.append(
            {
                "sector_key": key,
                "sector": sector_label,
                "source_file": prepared["source_file"].iloc[0] if "source_file" in prepared else "",
                "points": len(prepared),
                "kept_points": int((~prepared["is_outlier"]).sum()) if "is_outlier" in prepared else len(prepared),
                "high_outliers": int(prepared["is_outlier"].sum()) if "is_outlier" in prepared else 0,
                "wotan_window_days": window_length,
                "wotan_method": method,
                "wotan_cval": cval,
            }
        )
    return sector_tables, pd.DataFrame(summary_rows)


def _retrieve_exofop_ephemerides_for_flattening(photometry_fit_module) -> tuple[list[dict[str, float]], str]:
    if photometry_fit_module is None:
        return [], "The ExoFOP retrieval helpers are not available in this app session."
    target = _target_name_for_exofop_lookup()
    if not target:
        return [], "Enter or query a target first so ExoFOP can be searched."
    resolve = getattr(photometry_fit_module, "_ttv_fitter_resolve_tic_from_text", None)
    if callable(resolve):
        tic, error = resolve(target)
    else:
        tic = _find_tic_from_local_toi_list(target) or _find_tic_from_simbad_identifiers(target)
        error = "" if tic else f"Could not resolve `{target}` to a TIC ID."
    if not tic:
        return [], error
    cache_info = photometry_fit_module.cached_toi_info()
    toi_table, source_info = photometry_fit_module.load_toi_table(use_cached=bool(cache_info.get("exists")))
    matches = photometry_fit_module.find_toi_parameters(toi_table, tic)
    ephemerides = _exofop_ephemerides_from_matches(matches)
    st.session_state["phot_fit_exofop_matches"] = matches
    st.session_state["phot_fit_exofop_tic"] = tic
    st.session_state["phot_fit_exofop_source_info"] = source_info
    if not ephemerides:
        return [], f"Found no ExoFOP rows with usable T0, period, and duration for TIC {tic}."
    return ephemerides, f"Retrieved {len(ephemerides)} ExoFOP planet ephemeris row(s) for TIC {tic}."


def _render_flattening_workflow_with_method(photometry_import_module, photometry_fit_module, sigma_clip: float) -> None:
    sector_frames = st.session_state.get("sector_frames", {})
    sector_uncertainties = st.session_state.get("sector_uncertainties", {})
    if not sector_frames:
        return

    st.subheader("2. Detrend, flatten, and reject high outliers")
    st.caption(
        "Flatten each sector with Wotan before stitching. Outlier rejection here is one-sided: "
        "only points above the fitted trend are flagged, so downward transit points remain in the data."
    )

    uncertainties_complete = set(sector_uncertainties) == set(sector_frames)
    if not uncertainties_complete:
        missing = [key for key in sector_frames if key not in sector_uncertainties]
        st.warning(f"Complete uncertainties for all sectors before flattening. Missing: {len(missing)}")
        return

    sector_flattening = st.session_state.setdefault("sector_flattening", {})
    first_pass_by_sector = st.session_state.setdefault("ttv_first_pass_flattened", {})
    found_by_sector = st.session_state.setdefault("ttv_found_transits", {})
    status = photometry_import_module.flattening_status_table(sector_frames, sector_flattening)
    if not status.empty:
        st.dataframe(photometry_import_module._style_status_table(status), use_container_width=True, hide_index=True)

    first_pass_window_col, first_pass_cval_col = st.columns([2, 1])
    with first_pass_window_col:
        first_pass_window = st.slider(
            "1st pass heavy detrend window [days]",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get("ttv_first_pass_window", 1.2)),
            step=0.1,
            key="ttv_first_pass_window_input",
            help="Use roughly 10x the longest expected transit duration. This first pass is only used to find and mask transits.",
        )
    with first_pass_cval_col:
        first_pass_cval = st.slider(
            "Wotan robust cval",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get("ttv_first_pass_cval", 5.0)),
            step=0.5,
            key="ttv_first_pass_cval_input",
            help="Robustness tuning for Wotan biweight/huber detrending. Lower values reject transit-like outliers more aggressively; Wotan's biweight default is 5.",
        )
    if st.button("1st pass heavy detrend (should be 10x longest transit duration)", use_container_width=True, key="ttv_first_pass_heavy_detrend"):
        first_pass_by_sector = {}
        with st.spinner("Running first-pass heavy detrend for every sector..."):
            progress = st.progress(0.0)
            status_text = st.empty()
            sector_items = list(sector_frames.items())
            total_sectors = len(sector_items)
            for index, (key, frame) in enumerate(sector_items, start=1):
                sector_label = frame["sector"].iloc[0] if not frame.empty and "sector" in frame else key
                status_text.caption(f"Processing sector {index}/{total_sectors}: {sector_label}")
                first_pass = _flatten_sector_with_explicit_mask(
                    photometry_import_module,
                    frame,
                    sector_uncertainties[key],
                    window_length=float(first_pass_window),
                    method="biweight",
                    transit_mask=None,
                    cval=float(first_pass_cval),
                    sigma_clip=sigma_clip,
                )
                first_pass_by_sector[key] = _write_first_pass_cache(
                    photometry_import_module,
                    key,
                    first_pass,
                    window_length=float(first_pass_window),
                    method="biweight",
                    cval=float(first_pass_cval),
                )
                del first_pass
                gc.collect()
                progress.progress(index / total_sectors)
            status_text.caption(f"First-pass heavy detrend complete for {total_sectors} sector(s).")
        st.session_state["ttv_first_pass_flattened"] = first_pass_by_sector
        st.session_state["ttv_first_pass_window"] = float(first_pass_window)
        st.session_state["ttv_first_pass_method"] = "biweight"
        st.session_state["ttv_first_pass_cval"] = float(first_pass_cval)
        st.session_state.pop("ttv_found_transits", None)
        st.session_state.pop("sector_flattening", None)
        st.session_state.pop("prepared_photometry", None)
        st.session_state.pop("prepared_photometry_summary", None)
        st.success(f"First-pass heavy detrend complete for {len(first_pass_by_sector)} sector(s).")
        st.rerun()

    first_pass_complete = _first_pass_cache_entries_complete(first_pass_by_sector, sector_frames)
    if not first_pass_complete:
        st.info("Run the first-pass heavy detrend before retrieving/searching transit masks.")
        return

    preview_key = st.selectbox("First-pass sector preview", list(sector_frames.keys()), key="ttv_first_pass_sector_select")
    preview_first_pass = _read_first_pass_cache(first_pass_by_sector.get(preview_key))
    if preview_first_pass.empty:
        st.warning("The cached first-pass data for this sector was not found. Re-run the first-pass heavy detrend.")
        return
    preview_figure, preview_note = _first_pass_preview_figure(preview_first_pass)
    if preview_note:
        st.info(preview_note)
    st.plotly_chart(
        preview_figure,
        use_container_width=True,
        config=photometry_import_module.PLOT_CONFIG,
        key=f"ttv_first_pass_trend_{preview_key}",
    )

    accept_col, exofop_col = st.columns(2)
    with accept_col:
        accept_first_pass = st.button(
            "accept the detrending as is and move on",
            use_container_width=True,
            key="ttv_accept_first_pass_as_final",
        )
    with exofop_col:
        retrieve_exofop = st.button("get all exofop planet parameters", use_container_width=True, key="ttv_get_exofop_planets")

    if accept_first_pass:
        first_pass_window_value = float(st.session_state.get("ttv_first_pass_window", first_pass_window))
        first_pass_cval_value = float(st.session_state.get("ttv_first_pass_cval", first_pass_cval))
        new_flattening: dict[str, dict] = {}
        for key, frame in sector_frames.items():
            entry = first_pass_by_sector.get(key)
            if isinstance(entry, dict):
                high_outliers = int(entry.get("high_outliers", 0))
                first_pass_cache_path = str(entry.get("path", ""))
            else:
                first_pass = _read_first_pass_cache(entry)
                high_outliers = int(first_pass["is_outlier"].sum()) if "is_outlier" in first_pass else np.nan
                first_pass_cache_path = str(entry) if entry else ""
            new_flattening[key] = {
                "window_length": first_pass_window_value,
                "method": "biweight",
                "cval": first_pass_cval_value,
                "use_cached_first_pass": bool(first_pass_cache_path),
                "first_pass_cache_path": first_pass_cache_path,
                "mask_transits": False,
                "mask_ephemerides": [],
                "mask_planets": 0,
                "mask_width_durations": np.nan,
                "masked_points": 0,
                "found_transits": 0,
                "high_outliers": high_outliers,
            }
        st.session_state["sector_flattening"] = new_flattening
        st.session_state.pop("prepared_sector_photometry", None)
        st.session_state.pop("prepared_sector_summary", None)
        st.session_state.pop("prepared_sector_directory", None)
        st.session_state.pop("prepared_photometry", None)
        st.session_state.pop("prepared_photometry_summary", None)
        st.success(f"Accepted first-pass detrending for {len(new_flattening)} sector(s).")
        st.rerun()

    if retrieve_exofop:
        with st.spinner("Retrieving ExoFOP planet parameters..."):
            try:
                ephemerides, message = _retrieve_exofop_ephemerides_for_flattening(photometry_fit_module)
            except Exception as exc:  # noqa: BLE001 - resolver/network/table issue should be shown in the UI
                st.session_state["ttv_exofop_planet_error"] = str(exc)
                st.session_state.pop("ttv_exofop_planets", None)
            else:
                planets = _exofop_planet_rows_from_ephemerides(ephemerides)
                if not planets.empty:
                    st.session_state["ttv_exofop_planets"] = planets
                    st.session_state["ttv_exofop_planet_message"] = message
                    st.session_state.pop("ttv_exofop_planet_error", None)
                else:
                    st.session_state["ttv_exofop_planet_error"] = message
                    st.session_state.pop("ttv_exofop_planets", None)
        st.rerun()

    if st.session_state.get("ttv_exofop_planet_error"):
        st.warning(st.session_state["ttv_exofop_planet_error"])
    if st.session_state.get("ttv_exofop_planet_message"):
        st.caption(st.session_state["ttv_exofop_planet_message"])

    planets = st.session_state.get("ttv_exofop_planets")
    if not isinstance(planets, pd.DataFrame) or planets.empty:
        st.info("Retrieve ExoFOP planet parameters before searching the first-pass data for transits.")
        return

    planets = planets.copy()
    if "mask_duration_multiplier" not in planets:
        planets["mask_duration_multiplier"] = 2.0
    if "planet" not in planets:
        planets["planet"] = [_planet_letter(index) for index in range(len(planets))]
    else:
        planets["planet"] = [
            _planet_letter(index) if pd.isna(value) or not str(value).strip() or str(value).strip().lower() in {"nan", "none", "null"} else str(value).strip()
            for index, value in enumerate(planets["planet"])
        ]
    st.caption("Set each planet's transit mask width as a multiple of its fitted transit duration before searching.")
    mask_cols = st.columns(min(max(len(planets), 1), 4))
    for display_index, (planet_index, row) in enumerate(planets.iterrows()):
        planet_name = str(row.get("planet", _planet_letter(display_index)))
        current_multiplier = _coerce_float(row.get("mask_duration_multiplier"), 2.0)
        with mask_cols[display_index % len(mask_cols)]:
            planets.at[planet_index, "mask_duration_multiplier"] = st.number_input(
                f"{planet_name} mask width [x duration]",
                min_value=0.5,
                max_value=10.0,
                value=float(current_multiplier if np.isfinite(current_multiplier) and current_multiplier > 0 else 2.0),
                step=0.1,
                format="%.2f",
                key=f"ttv_mask_duration_multiplier_{planet_index}",
            )
    st.session_state["ttv_exofop_planets"] = planets

    display_cols = ["planet", "t0", "period", "duration_hours", "mask_duration_multiplier", "radius_ratio", "impact"]
    st.dataframe(planets[[col for col in display_cols if col in planets.columns]], use_container_width=True, hide_index=True)

    search_half_width = st.number_input(
        "Transit T0 search half-width [days]",
        min_value=0.001,
        max_value=5.0,
        value=float(st.session_state.get("ttv_transit_search_half_width", 0.2)),
        step=0.01,
        format="%.4f",
        key="ttv_transit_search_half_width_input",
        help="Each ExoFOP-predicted transit midpoint is searched over this +/- range on the first-pass detrended data.",
    )
    if st.button("use planet parameters to search for transits", use_container_width=True, key="ttv_search_transits_from_planets"):
        found_by_sector = {}
        with st.spinner("Searching for transit centers in each sector..."):
            progress = st.progress(0.0)
            status_text = st.empty()
            sector_items = list(first_pass_by_sector.items())
            total_sectors = len(sector_items)
            for index, (key, first_pass_entry) in enumerate(sector_items, start=1):
                first_pass = _read_first_pass_cache(first_pass_entry)
                if first_pass.empty:
                    found_by_sector[key] = pd.DataFrame()
                    progress.progress(index / total_sectors)
                    continue
                sector_label = first_pass["sector"].iloc[0] if "sector" in first_pass else key
                status_text.caption(f"Searching sector {index}/{total_sectors}: {sector_label}")
                found_by_sector[key] = _refine_transits_in_sector(
                    first_pass,
                    planets,
                    float(search_half_width),
                    grid_step_days=0.01,
                )
                del first_pass
                gc.collect()
                progress.progress(index / total_sectors)
            status_text.caption(f"Transit search complete for {total_sectors} sector(s).")
        st.session_state["ttv_found_transits"] = found_by_sector
        st.session_state["ttv_transit_search_half_width"] = float(search_half_width)
        st.session_state.pop("sector_flattening", None)
        st.session_state.pop("prepared_photometry", None)
        st.session_state.pop("prepared_photometry_summary", None)
        total = sum(len(found) for found in found_by_sector.values())
        st.success(f"Found/refined {total} transit window(s) across {len(found_by_sector)} sector(s).")
        st.rerun()

    found_by_sector = st.session_state.get("ttv_found_transits", {})
    found_complete = isinstance(found_by_sector, dict) and set(found_by_sector) == set(sector_frames)
    if not found_complete:
        st.info("Search for transits before running the masked second-pass detrend.")
        return

    mask_preview_key = st.selectbox("Transit-mask sector preview", list(sector_frames.keys()), key="ttv_transit_mask_sector_select")
    found_preview = found_by_sector.get(mask_preview_key, pd.DataFrame())
    mask_preview_first_pass = _read_first_pass_cache(first_pass_by_sector.get(mask_preview_key))
    if mask_preview_first_pass.empty:
        st.warning("The cached first-pass data for this preview sector was not found. Re-run the first-pass heavy detrend.")
        return
    if isinstance(found_preview, pd.DataFrame) and not found_preview.empty:
        st.plotly_chart(
            _transit_search_preview_figure(mask_preview_first_pass, found_preview),
            use_container_width=True,
            config=photometry_import_module.PLOT_CONFIG,
            key=f"ttv_transit_search_preview_{mask_preview_key}",
        )
        st.dataframe(
            found_preview[["planet", "epoch", "expected_tmid", "tmid", "offset_days", "duration_hours", "mask_duration_multiplier", "points"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No transit windows were found in this preview sector.")

    second_pass_window_col, second_pass_cval_col = st.columns([2, 1])
    with second_pass_window_col:
        second_pass_window = st.slider(
            "2nd pass Wotan detrending window [days]",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get("ttv_second_pass_window", st.session_state.get("ttv_first_pass_window", 1.2))),
            step=0.1,
            key="ttv_second_pass_window_input",
        )
    with second_pass_cval_col:
        second_pass_cval = st.slider(
            "2nd pass Wotan robust cval",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get("ttv_second_pass_cval", st.session_state.get("ttv_first_pass_cval", 5.0))),
            step=0.5,
            key="ttv_second_pass_cval_input",
            help="Robustness tuning passed to Wotan for the masked second-pass detrend.",
        )
    if st.button("accept the transit masking for all sectors and do 2nd pass detrending", use_container_width=True, key="ttv_accept_masks_second_pass"):
        new_flattening: dict[str, dict] = {}
        with st.spinner("Running second-pass Wotan detrend from original data with searched transits masked..."):
            progress = st.progress(0.0)
            status_text = st.empty()
            sector_items = list(sector_frames.items())
            total_sectors = len(sector_items)
            for index, (key, frame) in enumerate(sector_items, start=1):
                sector_label = frame["sector"].iloc[0] if not frame.empty and "sector" in frame else key
                status_text.caption(f"Second-pass detrending sector {index}/{total_sectors}: {sector_label}")
                found = found_by_sector.get(key, pd.DataFrame())
                if not isinstance(found, pd.DataFrame):
                    found = pd.DataFrame(found)
                transit_mask = _mask_from_found_transits(frame, found)
                prepared = _flatten_sector_with_explicit_mask(
                    photometry_import_module,
                    frame,
                    sector_uncertainties[key],
                    window_length=float(second_pass_window),
                    method="biweight",
                    transit_mask=transit_mask,
                    cval=float(second_pass_cval),
                    sigma_clip=sigma_clip,
                )
                new_flattening[key] = {
                    "window_length": float(second_pass_window),
                    "method": "biweight",
                    "cval": float(second_pass_cval),
                    "mask_transits": True,
                    "mask_ephemerides": _ephemerides_from_planet_rows(planets),
                    "mask_planets": int(planets["planet"].nunique()) if "planet" in planets else len(planets),
                    "mask_width_durations": 1.5,
                    "masked_points": int(np.sum(transit_mask)),
                    "found_transits": int(len(found)),
                    "mask_found_transits": found.to_dict("records"),
                    "high_outliers": int(prepared["is_outlier"].sum()),
                }
                progress.progress(index / total_sectors)
            status_text.caption(f"Second-pass masked detrend complete for {total_sectors} sector(s).")
        st.session_state["sector_flattening"] = new_flattening
        st.session_state["ttv_second_pass_window"] = float(second_pass_window)
        st.session_state["ttv_second_pass_method"] = "biweight"
        st.session_state["ttv_second_pass_cval"] = float(second_pass_cval)
        st.success(f"Second-pass masked detrend accepted for {len(new_flattening)} sector(s).")
        st.rerun()


def patch_photometry_loading_workflow(photometry_import_module, photometry_fit_module=None) -> None:
    """Keep the imported photometry workflow visible before files are loaded."""
    if getattr(photometry_import_module, "_ttv_fitter_loading_workflow_patch", False):
        return

    original_loaded_file_prompt = getattr(photometry_import_module, "_render_loaded_file_prompt", None)
    original_downloaded_files = photometry_import_module._render_downloaded_files
    original_uploaded_files = getattr(photometry_import_module, "_render_uploaded_files", None)
    original_uncertainty = photometry_import_module._render_uncertainty_workflow
    original_flattening = photometry_import_module._render_flattening_workflow
    original_stitching = photometry_import_module._render_stitching_controls
    original_flatten_sector = photometry_import_module.flatten_sector
    original_flattening_status_table = photometry_import_module.flattening_status_table
    original_stitch_flattened_sectors = photometry_import_module.stitch_flattened_sectors

    photometry_import_module.flatten_sector = lambda frame, uncertainty, *, window_length, sigma_clip=5.0, cval=5.0: _flatten_sector_with_method(
        photometry_import_module,
        frame,
        uncertainty,
        window_length=window_length,
        method="biweight",
        cval=cval,
        sigma_clip=sigma_clip,
    )
    photometry_import_module.flattening_status_table = _flattening_status_table_with_method
    photometry_import_module.stitch_flattened_sectors = lambda sector_frames, sector_uncertainties, sector_flattening, *, sigma_clip=5.0: _stitch_flattened_sectors_with_method(
        photometry_import_module,
        sector_frames,
        sector_uncertainties,
        sector_flattening,
        sigma_clip=sigma_clip,
    )

    def render_sector_files() -> list[str]:
        rows = []
        for path in st.session_state.get("mast_downloaded_paths", []):
            file_path = Path(path)
            rows.append(
                {
                    "source": "MAST download",
                    "file": file_path.name,
                    "path": str(file_path),
                    "exists": file_path.exists(),
                }
            )
        for path in st.session_state.get("uploaded_photometry_paths", []):
            file_path = Path(path)
            rows.append(
                {
                    "source": "Upload",
                    "file": file_path.name,
                    "path": str(file_path),
                    "exists": file_path.exists(),
                }
            )

        existing_paths = list(dict.fromkeys(row["path"] for row in rows if row["exists"]))
        if not rows:
            return []

        st.subheader("Sector files")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.button("Reset preparation workflow", use_container_width=True, key="reset_photometry_preparation_workflow"):
            _reset_photometry_preparation_workflow()
            st.rerun()
        return existing_paths

    def render_uploaded_files(_paths: list[str]) -> list[str]:
        # Uploaded files are merged into the unified sector file list above.
        return []

    def render_loaded_file_prompt() -> None:
        _clear_photometry_preparation_state()
        st.subheader("Sector files")
        st.info(
            "Load/download sector files first. Download selected MAST light-curve products or upload "
            "FITS/CSV/TXT/DAT photometry files above to unlock the preparation workflow."
        )
        st.caption(
            "Uploaded photometry files are saved locally and passed through the same uncertainty, "
            "flattening, and stitching workflow as downloaded MAST files."
        )
        _staged_workflow_status()

    def render_uncertainty(paths: list[str], sigma_clip: float, stitch: bool) -> None:
        if not paths:
            _clear_photometry_preparation_state()
            st.subheader("1. Examine or determine sector uncertainties")
            st.info("Waiting for sector files from MAST downloads or uploaded photometry.")
            st.caption(
                "Once files are available, each sector can use supplied flux uncertainties or an "
                "uncertainty estimated from a selected quiet region."
            )
            return
        sector_frames = st.session_state.get("sector_frames", {})
        sector_uncertainties = st.session_state.get("sector_uncertainties", {})
        if _sector_frames_match_paths(paths) and _workflow_sets_complete(sector_uncertainties, sector_frames):
            summary = photometry_import_module.sector_uncertainty_table(sector_frames, sector_uncertainties)
            _render_completed_step(
                "1. Examine or determine sector uncertainties",
                "Uncertainties have been accepted for every loaded sector. Use Reset preparation workflow above to reopen this step.",
                summary,
            )
            return
        _render_uncertainty_workflow_with_bulk_accept(photometry_import_module, paths, sigma_clip, stitch)
        sector_uncertainties = st.session_state.get("sector_uncertainties", {})
        if _sector_frames_match_paths(paths) and _workflow_sets_complete(sector_uncertainties, sector_frames):
            st.rerun()

    def render_flattening(sigma_clip: float) -> None:
        if not st.session_state.get("sector_frames", {}):
            _render_waiting_step(
                "2. Detrend, flatten, and reject high outliers",
                "Available after sector files are loaded and uncertainties have been reviewed.",
            )
            return
        sector_frames = st.session_state.get("sector_frames", {})
        sector_uncertainties = st.session_state.get("sector_uncertainties", {})
        if not _workflow_sets_complete(sector_uncertainties, sector_frames):
            missing = len(set(sector_frames) - set(sector_uncertainties))
            _render_waiting_step(
                "2. Detrend, flatten, and reject high outliers",
                f"Complete uncertainty review for every sector first. Missing: {missing}.",
            )
            return
        sector_flattening = st.session_state.get("sector_flattening", {})
        if _workflow_sets_complete(sector_flattening, sector_frames):
            summary = photometry_import_module.flattening_status_table(sector_frames, sector_flattening)
            _render_completed_step(
                "2. Detrend, flatten, and reject high outliers",
                "Flattening has been accepted for every loaded sector. Use Reset preparation workflow above to reopen this step.",
                summary,
            )
            return
        _render_flattening_workflow_with_method(photometry_import_module, photometry_fit_module, sigma_clip)
        sector_flattening = st.session_state.get("sector_flattening", {})
        if _workflow_sets_complete(sector_flattening, sector_frames):
            st.rerun()

    def render_stitching(sigma_clip: float, stitch: bool) -> None:
        st.subheader("3. Stitch sectors (optional)")
        if not st.session_state.get("sector_frames", {}):
            st.info("Waiting")
            st.caption("Available after sector uncertainties and flattening are complete.")
            return
        sector_frames = st.session_state.get("sector_frames", {})
        sector_uncertainties = st.session_state.get("sector_uncertainties", {})
        sector_flattening = st.session_state.get("sector_flattening", {})
        if not _workflow_sets_complete(sector_uncertainties, sector_frames):
            st.info("Waiting")
            st.caption("Waiting for uncertainty review to finish.")
            return
        if not _workflow_sets_complete(sector_flattening, sector_frames):
            missing = len(set(sector_frames) - set(sector_flattening))
            st.info("Waiting")
            st.caption(f"Waiting for flattening review to finish. Missing: {missing}.")
            return

        if st.button("Download data as sectors", use_container_width=True, key="download_prepared_sector_files_button"):
            progress = st.progress(0.0)
            status_text = st.empty()

            def update_sector_progress(index: int, total: int, key: str, frame: pd.DataFrame) -> None:
                sector_label = frame["sector"].iloc[0] if not frame.empty and "sector" in frame else key
                status_text.caption(f"Preparing sector file {index}/{total}: {sector_label}")
                progress.progress(index / max(total, 1))

            with st.spinner("Preparing sector CSV files..."):
                sector_tables, sector_summary = _prepared_sector_tables_with_method(
                    photometry_import_module,
                    sector_frames,
                    sector_uncertainties,
                    sector_flattening,
                    sigma_clip=sigma_clip,
                    progress_callback=update_sector_progress,
                )
                target_part = photometry_import_module._target_filename_part()
                sector_dir = Path("data") / "prepared" / target_part / "sectors"
                sector_dir.mkdir(parents=True, exist_ok=True)
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for key, table in sector_tables.items():
                        sector_label = str(table["sector"].iloc[0]) if "sector" in table and not table.empty else str(key)
                        clean_sector = re.sub(r"[^A-Za-z0-9_.-]+", "_", sector_label).strip("_") or str(key)
                        filename = f"{target_part}_sector_{clean_sector}.csv"
                        export = table.loc[~table["is_outlier"], ["time", "flux", "flux_err", "source_file", "sector"]].copy()
                        local_path = sector_dir / filename
                        export.to_csv(local_path, index=False)
                        archive.writestr(f"sectors/{filename}", export.to_csv(index=False))
                    archive.writestr("sectors/sector_summary.csv", sector_summary.to_csv(index=False))
                zip_buffer.seek(0)
            st.session_state["prepared_sector_photometry"] = sector_tables
            st.session_state["prepared_sector_summary"] = sector_summary
            st.session_state["prepared_sector_directory"] = str(sector_dir)
            status_text.caption(f"Prepared {len(sector_tables)} sector file(s) in {sector_dir}.")
            st.download_button(
                "Download prepared sector CSV bundle",
                data=zip_buffer.getvalue(),
                file_name=f"{target_part}_sectors.zip",
                mime="application/zip",
                use_container_width=True,
                key="download_prepared_sector_zip",
            )
            st.success(f"Sector CSV files written to {sector_dir}.")

        had_prepared = st.session_state.get("prepared_photometry") is not None
        if st.button("Prepare stitched photometry", type="primary", disabled=not stitch):
            progress = st.progress(0.0)
            status_text = st.empty()

            def update_progress(index: int, total: int, key: str, frame: pd.DataFrame) -> None:
                sector_label = frame["sector"].iloc[0] if not frame.empty and "sector" in frame else key
                status_text.caption(f"Stitching sector {index}/{total}: {sector_label}")
                progress.progress(index / max(total, 1))

            with st.spinner("Preparing stitched photometry..."):
                stitched, summary = _stitch_flattened_sectors_with_method(
                    photometry_import_module,
                    sector_frames,
                    sector_uncertainties,
                    sector_flattening,
                    sigma_clip=sigma_clip,
                    progress_callback=update_progress,
                )
            status_text.caption(f"Stitched {len(summary)} sector file(s) into {len(stitched):,} photometry point(s).")
            st.session_state["prepared_photometry"] = stitched
            st.session_state["prepared_photometry_summary"] = summary
            st.success("Stitched photometry prepared.")
        has_prepared = st.session_state.get("prepared_photometry") is not None
        if has_prepared and not had_prepared:
            st.rerun()

    def render_prepared_photometry() -> None:
        stitched = st.session_state.get("prepared_photometry")
        summary = st.session_state.get("prepared_photometry_summary")
        if stitched is None or summary is None:
            return

        st.subheader("Prepared stitched photometry")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Files", f"{len(summary)}")
        metric_cols[1].metric("Points", f"{len(stitched):,}")
        metric_cols[2].metric("Outliers", f"{int(stitched['is_outlier'].sum()):,}")
        metric_cols[3].metric("Sectors", f"{stitched['sector'].nunique()}")

        if len(stitched) > 100_000:
            st.info(
                f"Skipping the stitched photometry plot because this dataset has {len(stitched):,} points. "
                "The full dataset is still available in the table summary and CSV download."
            )
        else:
            st.plotly_chart(
                photometry_import_module.prepared_photometry_preview(stitched),
                use_container_width=True,
                config=photometry_import_module.PLOT_CONFIG,
                key="prepared_photometry_preview",
            )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        export = stitched.loc[~stitched["is_outlier"], ["time", "flux", "flux_err", "source_file", "sector"]].copy()
        st.download_button(
            "Download prepared photometry CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name=f"prepared_photometry_{photometry_import_module._target_filename_part()}.csv",
            mime="text/csv",
            key="download_prepared_photometry",
        )

    photometry_import_module._render_downloaded_files = render_sector_files
    if original_uploaded_files is not None:
        photometry_import_module._render_uploaded_files = render_uploaded_files
    photometry_import_module._render_loaded_file_prompt = render_loaded_file_prompt
    photometry_import_module._render_uncertainty_workflow = render_uncertainty
    photometry_import_module._render_flattening_workflow = render_flattening
    photometry_import_module._render_stitching_controls = render_stitching
    photometry_import_module._render_prepared_photometry = render_prepared_photometry
    photometry_import_module._ttv_fitter_original_downloaded_files = original_downloaded_files
    photometry_import_module._ttv_fitter_original_uploaded_files = original_uploaded_files
    photometry_import_module._ttv_fitter_original_loaded_file_prompt = original_loaded_file_prompt
    photometry_import_module._ttv_fitter_original_uncertainty_workflow = original_uncertainty
    photometry_import_module._ttv_fitter_original_flattening_workflow = original_flattening
    photometry_import_module._ttv_fitter_original_stitching_controls = original_stitching
    photometry_import_module._ttv_fitter_original_flatten_sector = original_flatten_sector
    photometry_import_module._ttv_fitter_original_flattening_status_table = original_flattening_status_table
    photometry_import_module._ttv_fitter_original_stitch_flattened_sectors = original_stitch_flattened_sectors
    photometry_import_module._ttv_fitter_loading_workflow_patch = True


def patch_photometry_status_tables(photometry_import_module) -> None:
    """Remove workflow-completion status columns from the preparation tables."""
    if getattr(photometry_import_module, "_ttv_fitter_status_table_patch", False):
        return

    original_uncertainty_table = photometry_import_module.sector_uncertainty_table
    original_flattening_table = photometry_import_module.flattening_status_table
    original_style_status = photometry_import_module._style_status_table

    def sector_uncertainty_table_without_status(*args, **kwargs) -> pd.DataFrame:
        table = original_uncertainty_table(*args, **kwargs)
        return table.drop(columns=["status"], errors="ignore")

    def flattening_status_table_without_status(*args, **kwargs) -> pd.DataFrame:
        table = original_flattening_table(*args, **kwargs)
        return table.drop(columns=["status"], errors="ignore")

    def style_table_without_status(table: pd.DataFrame):
        if "status" not in table.columns:
            return table
        return original_style_status(table)

    photometry_import_module.sector_uncertainty_table = sector_uncertainty_table_without_status
    photometry_import_module.flattening_status_table = flattening_status_table_without_status
    photometry_import_module._style_status_table = style_table_without_status
    photometry_import_module._ttv_fitter_original_sector_uncertainty_table = original_uncertainty_table
    photometry_import_module._ttv_fitter_original_flattening_status_table = original_flattening_table
    photometry_import_module._ttv_fitter_original_style_status_table = original_style_status
    photometry_import_module._ttv_fitter_status_table_patch = True


def patch_fast_uncertainty_preview(plots_module, photometry_import_module) -> None:
    """Use a light WebGL-only uncertainty plot so zooming large sectors stays responsive."""
    if getattr(plots_module, "_ttv_fitter_fast_uncertainty_patch", False):
        photometry_import_module.sector_uncertainty_preview = plots_module.sector_uncertainty_preview
        return

    original_preview = plots_module.sector_uncertainty_preview

    def fast_sector_uncertainty_preview(frame: pd.DataFrame, trend: np.ndarray | None = None) -> go.Figure:
        if frame.empty:
            return original_preview(frame, trend)

        fig = go.Figure()
        marker_opacity = 0.32 if trend is not None else 0.55
        stride = _simple_plot_stride(len(frame))
        display_frame = frame.iloc[::stride].copy() if stride > 1 else frame
        display_trend = None
        if trend is not None and len(trend) == len(frame):
            trend_array = np.asarray(trend, dtype=float)
            display_trend = trend_array[::stride] if stride > 1 else trend_array
        fig.add_trace(
            go.Scattergl(
                x=display_frame["time"],
                y=display_frame["flux"],
                mode="markers",
                marker={"size": 4, "color": "#2563eb", "opacity": marker_opacity},
                name="Flux",
            )
        )
        if trend is None and "flux_err" in frame.columns:
            finite_err = pd.to_numeric(frame["flux_err"], errors="coerce")
            err_frame = frame.loc[np.isfinite(finite_err) & (finite_err > 0)].copy()
            if not err_frame.empty:
                err_frame = err_frame.sort_values("time") if "time" in err_frame.columns else err_frame
                stride = max(1, math.ceil(len(err_frame) / 1_200))
                err_frame = err_frame.iloc[::stride]
                fig.add_trace(
                    go.Scatter(
                        x=err_frame["time"],
                        y=err_frame["flux"],
                        error_y={
                            "type": "data",
                            "array": pd.to_numeric(err_frame["flux_err"], errors="coerce"),
                            "visible": True,
                            "thickness": 0.7,
                            "width": 1,
                        },
                        mode="markers",
                        marker={"size": 3, "color": "#0f766e", "opacity": 0.45},
                        name="Flux uncertainty sample",
                    )
                )
        if display_trend is not None:
            fig.add_trace(
                go.Scattergl(
                    x=display_frame["time"],
                    y=display_trend,
                    mode="lines",
                    line={"color": "#ef4444", "width": 3},
                    name="Wotan trend",
                )
            )
        fig = plots_module._base_layout(fig, "Sector uncertainty review", "Time - BTJD", "Relative flux")
        fig.update_layout(uirevision="sector_uncertainty_review")
        if stride > 1:
            note = f"Only every {_ordinal_word(stride)} data point is plotted to save memory ({len(display_frame):,} of {len(frame):,} points shown)."
            fig.update_layout(
                meta={
                    "ttv_sampled_plot_note": note
                }
            )
            fig.add_annotation(
                text=note,
                x=0.01,
                y=1.06,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font={"size": 12, "color": "#64748b"},
            )
        return fig

    plots_module._ttv_fitter_original_sector_uncertainty_preview = original_preview
    plots_module.sector_uncertainty_preview = fast_sector_uncertainty_preview
    photometry_import_module.sector_uncertainty_preview = fast_sector_uncertainty_preview
    plots_module._ttv_fitter_fast_uncertainty_patch = True


def _simple_plot_stride(point_count: int) -> int:
    if point_count > 75_000:
        return 4
    if point_count > 50_000:
        return 3
    if point_count > 25_000:
        return 2
    return 1


def _ordinal_word(value: int) -> str:
    return {2: "second", 3: "third", 4: "fourth"}.get(int(value), f"{int(value)}th")


def _sample_photometry_for_plot(frame: pd.DataFrame, max_points: int = 12_000) -> pd.DataFrame:
    if frame is None or frame.empty or len(frame) <= max_points:
        return frame
    table = frame.copy()
    if "time" in table.columns:
        table = table.sort_values("time")
    keep_indices: set[int] = set()
    uniform_target = max_points * 3 // 4
    stride = max(1, math.ceil(len(table) / uniform_target))
    keep_indices.update(table.index[::stride].tolist())

    if {"time", "flux"}.issubset(table.columns):
        time_values = pd.to_numeric(table["time"], errors="coerce")
        flux_values = pd.to_numeric(table["flux"], errors="coerce")
        finite = table.loc[np.isfinite(time_values) & np.isfinite(flux_values)].copy()
        if not finite.empty:
            finite_time = pd.to_numeric(finite["time"], errors="coerce")
            bins = min(max_points // 12, 900, max(50, len(finite) // 30))
            if bins > 1 and finite_time.max() > finite_time.min():
                bin_id = pd.cut(finite_time, bins=bins, labels=False, include_lowest=True)
                local_minima = finite.assign(_bin_id=bin_id).dropna(subset=["_bin_id"]).groupby("_bin_id", observed=True)["flux"].idxmin()
                keep_indices.update(local_minima.dropna().tolist())

    if len(keep_indices) > max_points:
        ordered = sorted(keep_indices, key=lambda idx: table.index.get_loc(idx))
        stride = max(1, math.ceil(len(ordered) / max_points))
        keep_indices = set(ordered[::stride])
    return table.loc[sorted(keep_indices, key=lambda idx: table.index.get_loc(idx))]


def _sample_model_series(time_values: np.ndarray, flux_values: np.ndarray, max_points: int = 6_000) -> tuple[np.ndarray, np.ndarray]:
    time_values = np.asarray(time_values, dtype=float)
    flux_values = np.asarray(flux_values, dtype=float)
    if len(time_values) <= max_points:
        return time_values, flux_values
    order = np.argsort(time_values)
    ordered_time = time_values[order]
    ordered_flux = flux_values[order]
    keep = set(range(0, len(order), max(1, math.ceil(len(order) / max_points))))
    finite_flux = ordered_flux[np.isfinite(ordered_flux)]
    if finite_flux.size:
        low_cut = float(np.nanquantile(finite_flux, 0.12))
        keep.update(np.flatnonzero(ordered_flux <= low_cut).tolist())
    if len(keep) > max_points:
        chosen = np.array(sorted(keep), dtype=int)
        stride = max(1, math.ceil(len(chosen) / max_points))
        chosen = chosen[::stride]
    else:
        chosen = np.array(sorted(keep), dtype=int)
    return ordered_time[chosen], ordered_flux[chosen]


def patch_lightweight_photometry_plots(plots_module, photometry_import_module, photometry_fit_module) -> None:
    """Reduce browser-side Plotly payloads while leaving underlying data untouched."""
    if getattr(plots_module, "_ttv_fitter_lightweight_plot_patch", False):
        photometry_import_module.prepared_photometry_preview = plots_module.prepared_photometry_preview
        photometry_fit_module.prepared_photometry_preview = plots_module.prepared_photometry_preview
        photometry_fit_module.photometry_model_overlay = plots_module.photometry_model_overlay
        patch_sampled_plot_notice(photometry_import_module)
        patch_sampled_plot_notice(photometry_fit_module)
        return

    original_prepared_preview = plots_module.prepared_photometry_preview
    original_model_overlay = plots_module.photometry_model_overlay

    def lightweight_prepared_photometry_preview(frame: pd.DataFrame) -> go.Figure:
        if frame is None or frame.empty:
            return original_prepared_preview(frame)
        stride = _simple_display_stride(len(frame), max_display_points=20_000)
        display = frame.iloc[::stride].copy() if stride > 1 else frame
        if "is_outlier" not in display.columns:
            display["is_outlier"] = False
        fig = original_prepared_preview(display)
        if stride > 1:
            note = _stride_note(len(frame), len(display), stride)
            fig.update_layout(
                meta={
                    "ttv_sampled_plot_note": note
                }
            )
            fig.add_annotation(
                text=note,
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font={"size": 11, "color": "#64748b"},
            )
        return fig

    def lightweight_photometry_model_overlay(
        frame: pd.DataFrame,
        model_flux: np.ndarray | None,
        *,
        title: str = "Initial transit model on data",
        model_name: str = "Initial model",
    ) -> go.Figure:
        fig = go.Figure()
        if frame is None or frame.empty:
            return plots_module._base_layout(fig, title, "Time - BTJD", "Relative flux")

        display = _sample_photometry_for_plot(frame, max_points=12_000)
        fig.add_trace(
            go.Scattergl(
                x=display["time"],
                y=display["flux"],
                mode="markers",
                marker={"size": 3, "color": "#2563eb", "opacity": 0.32},
                text=display["source_file"] if "source_file" in display.columns else None,
                name="Prepared data",
            )
        )
        if model_flux is not None and len(model_flux) == len(frame):
            model_time, model_display = _sample_model_series(frame["time"].to_numpy(dtype=float), np.asarray(model_flux, dtype=float), 6_000)
            fig.add_trace(
                go.Scattergl(
                    x=model_time,
                    y=model_display,
                    mode="lines",
                    line={"color": "#ef4444", "width": 4},
                    name=model_name,
                )
            )
        fig = plots_module._base_layout(fig, title, "Time - BTJD", "Relative flux")
        fig.update_layout(uirevision="photometry_model_overlay")
        if len(frame) > len(display):
            fig.update_layout(
                meta={
                    "ttv_sampled_plot_note": (
                        f"This plot shows a reduced-size display sample ({len(display):,} of {len(frame):,} points) "
                        "to keep the browser responsive. Fits and exports still use the full dataset."
                    )
                }
            )
            fig.add_annotation(
                text=f"displaying {len(display):,} of {len(frame):,} data points",
                x=0.01,
                y=0.98,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font={"size": 11, "color": "#64748b"},
            )
        return fig

    plots_module._ttv_fitter_original_prepared_photometry_preview = original_prepared_preview
    plots_module._ttv_fitter_original_photometry_model_overlay = original_model_overlay
    plots_module.prepared_photometry_preview = lightweight_prepared_photometry_preview
    plots_module.photometry_model_overlay = lightweight_photometry_model_overlay
    photometry_import_module.prepared_photometry_preview = lightweight_prepared_photometry_preview
    photometry_fit_module.prepared_photometry_preview = lightweight_prepared_photometry_preview
    photometry_fit_module.photometry_model_overlay = lightweight_photometry_model_overlay
    plots_module._ttv_fitter_lightweight_plot_patch = True
    patch_sampled_plot_notice(photometry_import_module)
    patch_sampled_plot_notice(photometry_fit_module)


def _sampled_plot_notice(figure: object) -> str:
    try:
        meta = getattr(getattr(figure, "layout", None), "meta", None)
    except Exception:
        return ""
    if isinstance(meta, dict):
        return str(meta.get("ttv_sampled_plot_note", ""))
    return ""


def patch_sampled_plot_notice(module) -> None:
    if getattr(module, "_ttv_fitter_sampled_plot_notice_patch", False):
        return

    original_render = module.render

    def render_with_sampled_plot_notice(*args, **kwargs):
        original_plotly_chart = st.plotly_chart

        def plotly_chart_with_notice(figure_or_data, *plot_args, **plot_kwargs):
            note = _sampled_plot_notice(figure_or_data)
            if note:
                st.info(note)
            return original_plotly_chart(figure_or_data, *plot_args, **plot_kwargs)

        st.plotly_chart = plotly_chart_with_notice
        try:
            return original_render(*args, **kwargs)
        finally:
            st.plotly_chart = original_plotly_chart

    module.render = render_with_sampled_plot_notice
    module._ttv_fitter_original_sampled_plot_notice_render = original_render
    module._ttv_fitter_sampled_plot_notice_patch = True


def _round_down_to_step(value: float, step: float) -> float:
    if not np.isfinite(value) or step <= 0:
        return float(value)
    return float(math.floor(float(value) / step) * step)


def _round_up_to_step(value: float, step: float) -> float:
    if not np.isfinite(value) or step <= 0:
        return float(value)
    return float(math.ceil(float(value) / step) * step)


def _coerce_float(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if np.isfinite(number) else float(fallback)


def _escape_adql_literal(value: str) -> str:
    return str(value).replace("'", "''")


def _extract_tic_from_identifiers(identifiers: pd.Series | list[object]) -> str:
    for value in identifiers:
        match = re.search(r"\bTIC\s+(\d+)\b", str(value), flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _find_tic_from_simbad_identifiers(target_name: str) -> str:
    """Resolve a target name to a TIC ID with a direct SIMBAD TAP sync query."""
    target = " ".join(str(target_name or "").strip().split())
    if not target:
        return ""
    query = f"""
SELECT TOP 200 i.id
FROM ident AS i
JOIN ident AS q ON i.oidref = q.oidref
WHERE q.id = '{_escape_adql_literal(target)}'
"""
    try:
        import requests

        response = requests.post(
            SIMBAD_TAP_SYNC_URL,
            data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
            timeout=20,
        )
        response.raise_for_status()
        table = pd.read_csv(StringIO(response.text))
    except Exception:
        return ""
    if table.empty or "id" not in table.columns:
        return ""
    return _extract_tic_from_identifiers(table["id"])


def _find_tic_from_local_toi_list(target_name: str) -> str:
    text = " ".join(str(target_name or "").strip().split()).upper()
    match = re.match(r"^TOI[-\s]*(\d+)(?:\.(\d+))?$", text)
    if not match or not LOCAL_EXOFOP_TOI_LIST.exists():
        return ""
    toi_number = int(match.group(1))
    toi_suffix = match.group(2)
    try:
        table = pd.read_csv(LOCAL_EXOFOP_TOI_LIST, usecols=["TIC ID", "TOI"])
    except Exception:
        return ""
    toi_values = pd.to_numeric(table["TOI"], errors="coerce")
    if toi_suffix is not None:
        target_toi = float(f"{toi_number}.{toi_suffix}")
        matches = table.loc[np.isclose(toi_values, target_toi, rtol=0, atol=1e-6)]
    else:
        matches = table.loc[np.floor(toi_values) == float(toi_number)]
    if matches.empty:
        return ""
    tic = pd.to_numeric(matches.iloc[0].get("TIC ID"), errors="coerce")
    return str(int(tic)) if np.isfinite(tic) else ""


def _stellar_values_from_prior_table(
    table: pd.DataFrame | None,
    stellar_mass: float = 1.0,
    stellar_radius: float = 1.0,
) -> tuple[float, float]:
    if not isinstance(table, pd.DataFrame) or table.empty or not {"planet", "parameter", "value"}.issubset(table.columns):
        return float(stellar_mass), float(stellar_radius)
    host_rows = table.loc[table["planet"].astype(str) == "host"]
    for parameter, current in [("stellar_mass", stellar_mass), ("stellar_radius", stellar_radius)]:
        match = host_rows.loc[host_rows["parameter"].astype(str) == parameter] if not host_rows.empty else pd.DataFrame()
        if match.empty:
            continue
        value = _coerce_float(match.iloc[0].get("value"), float(current))
        if parameter == "stellar_mass":
            stellar_mass = value
        else:
            stellar_radius = value
    return float(stellar_mass), float(stellar_radius)


def _repair_host_prior_parameter_names(table: pd.DataFrame | None) -> pd.DataFrame | None:
    """Restore host stellar parameter labels if a Streamlit editor returns blanks."""
    if not isinstance(table, pd.DataFrame) or table.empty or "planet" not in table.columns or "parameter" not in table.columns:
        return table
    repaired = table.copy()
    host_indices = list(repaired.index[repaired["planet"].astype(str).str.lower() == "host"])
    if not host_indices:
        return repaired

    parameters = repaired["parameter"].astype(str).str.strip().str.lower()
    missing = [name for name in ["stellar_mass", "stellar_radius"] if name not in set(parameters)]
    if not missing:
        return repaired

    blank_values = {"", "nan", "none", "<na>"}
    for idx, name in zip(host_indices, ["stellar_mass", "stellar_radius"]):
        value = str(repaired.at[idx, "parameter"]).strip().lower()
        if value in blank_values and name in missing:
            repaired.at[idx, "parameter"] = name
    return repaired


def patch_photometry_fit_prior_bounds(photometry_fit_module) -> None:
    """Round ExoFOP-seeded uniform bounds to easier-to-read intervals."""
    if getattr(photometry_fit_module, "_ttv_fitter_prior_bounds_patch", False):
        return

    original_default_planet_rows = photometry_fit_module._default_planet_rows

    def default_planet_rows_with_readable_bounds(matches: pd.DataFrame, planet_count: int, prior_type: str) -> pd.DataFrame:
        table = original_default_planet_rows(matches, planet_count, prior_type).copy()
        source = matches.iloc[0] if matches is not None and not matches.empty else pd.Series(dtype=object)
        multiplier = float(getattr(photometry_fit_module, "EXOFOP_UNIFORM_SIGMA_MULTIPLIER", 50))
        stellar_mass = _coerce_float(source.get("Stellar Mass (M_Sun)", np.nan), 1.0)
        stellar_mass_err = _coerce_float(source.get("Stellar Mass (M_Sun) err", np.nan), max(stellar_mass * 0.1, 0.05))
        stellar_radius = _coerce_float(source.get("Stellar Radius (R_Sun)", np.nan), 1.0)
        stellar_radius_err = _coerce_float(source.get("Stellar Radius (R_Sun) err", np.nan), max(stellar_radius * 0.1, 0.05))
        stellar_rows = pd.DataFrame(
            [
                {
                    "planet": "host",
                    "parameter": "stellar_mass",
                    "value": stellar_mass,
                    "prior_type": prior_type,
                    "sigma": stellar_mass_err,
                    "lower": max(stellar_mass - multiplier * stellar_mass_err, 0.01),
                    "upper": stellar_mass + multiplier * stellar_mass_err,
                    "fit": False,
                },
                {
                    "planet": "host",
                    "parameter": "stellar_radius",
                    "value": stellar_radius,
                    "prior_type": prior_type,
                    "sigma": stellar_radius_err,
                    "lower": max(stellar_radius - multiplier * stellar_radius_err, 0.01),
                    "upper": stellar_radius + multiplier * stellar_radius_err,
                    "fit": False,
                },
            ]
        )
        table = pd.concat([stellar_rows, table], ignore_index=True)
        if table.empty or "parameter" not in table.columns:
            return table

        for idx, row in table.iterrows():
            parameter = str(row.get("parameter"))
            lower = float(pd.to_numeric(pd.Series([row.get("lower")]), errors="coerce").iloc[0])
            upper = float(pd.to_numeric(pd.Series([row.get("upper")]), errors="coerce").iloc[0])
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            if parameter in {"T0", "period"}:
                rounded_lower = _round_down_to_step(lower, 0.1)
                rounded_upper = _round_up_to_step(upper, 0.1)
                if parameter == "period":
                    rounded_lower = max(rounded_lower, 1e-6)
                if rounded_upper <= rounded_lower:
                    rounded_upper = rounded_lower + 0.1
                table.loc[idx, "lower"] = rounded_lower
                table.loc[idx, "upper"] = rounded_upper
            elif parameter == "radius_ratio":
                rounded_lower = max(_round_down_to_step(lower, 0.1), 0.0)
                rounded_upper = min(max(_round_up_to_step(upper, 0.1), rounded_lower + 0.1), 1.0)
                table.loc[idx, "lower"] = rounded_lower
                table.loc[idx, "upper"] = rounded_upper
        return table

    photometry_fit_module._default_planet_rows = default_planet_rows_with_readable_bounds
    photometry_fit_module._ttv_fitter_original_default_planet_rows = original_default_planet_rows
    photometry_fit_module._ttv_fitter_prior_bounds_patch = True


def _impact_factor_note() -> str:
    return (
        "Geometry translation note: this interface calls the transit chord parameter `impact factor`. "
        "For circular-orbit geometry, `impact_factor = (a/Rstar) * cos(i)`, so "
        "`cos(i) = impact_factor / (a/Rstar)`. allesfitter samples native `cosi`, while this page "
        "lets you edit the more readable impact factor and converts it before writing `params.csv`. "
        "`rsuma = (Rstar + Rp) / a = (1 + radius_ratio) / (a/Rstar)`, with `a/Rstar` derived from "
        "period, stellar mass, and stellar radius using Kepler's law."
    )


def _rename_impact_factor_labels(params: pd.DataFrame) -> pd.DataFrame:
    table = params.copy()
    if "label" in table.columns:
        table["label"] = (
            table["label"]
            .astype(str)
            .str.replace("cos inclination from impact setup", "cos inclination from impact factor setup", regex=False)
            .str.replace("impact setup", "impact factor setup", regex=False)
        )
    return table


def patch_photometry_fit_impact_factor_language(photometry_fit_module) -> None:
    """Use impact-factor terminology and explain the conversion under generated params."""
    if getattr(photometry_fit_module, "_ttv_fitter_impact_factor_language_patch", False):
        return

    original_build_params = photometry_fit_module._build_allesfitter_params_from_setup
    original_run_least_squares = photometry_fit_module._run_least_squares_refinement
    original_setup = photometry_fit_module._render_transit_parameter_setup
    original_render = photometry_fit_module.render

    def build_params_with_impact_factor_labels(*args, **kwargs):
        if args and isinstance(args[0], pd.DataFrame):
            planet_table = _repair_host_prior_parameter_names(args[0])
            planet_table = planet_table.copy() if isinstance(planet_table, pd.DataFrame) else args[0].copy()
            stellar_mass = args[2] if len(args) > 2 else kwargs.get("stellar_mass", 1.0)
            stellar_radius = args[3] if len(args) > 3 else kwargs.get("stellar_radius", 1.0)
            stellar_mass, stellar_radius = _stellar_values_from_prior_table(planet_table, stellar_mass, stellar_radius)
            planet_table = planet_table.loc[planet_table["planet"].astype(str) != "host"].copy()
            args = (planet_table, *args[1:2], stellar_mass, stellar_radius, *args[4:])
        params, derived = original_build_params(*args, **kwargs)
        return _rename_impact_factor_labels(params), derived

    def run_least_squares_with_host_rows(
        data,
        planet_table,
        nuisance_table,
        stellar_mass,
        stellar_radius,
        *args,
        **kwargs,
    ):
        planet_table = _repair_host_prior_parameter_names(planet_table)
        stellar_mass, stellar_radius = _stellar_values_from_prior_table(planet_table, stellar_mass, stellar_radius)
        return original_run_least_squares(
            data,
            planet_table,
            nuisance_table,
            stellar_mass,
            stellar_radius,
            *args,
            **kwargs,
        )

    def patched_setup_context(render_call):
        original_data_editor = st.data_editor
        original_dataframe = st.dataframe
        original_number_input = st.number_input
        original_button = st.button
        original_plotly_chart = st.plotly_chart
        original_success = st.success
        delta_generator_cls = type(st.container())
        original_delta_number_input = delta_generator_cls.number_input
        original_batman_model = photometry_fit_module._batman_model_on_data

        def hidden_stellar_number_input(label, *args, original_call=None, **kwargs):
            label_text = str(label)
            if ("Stellar mass" in label_text and "Msun" in label_text) or ("Stellar radius" in label_text and "Rsun" in label_text):
                fallback = _coerce_float(kwargs.get("value"), 1.0)
                table = st.session_state.get("phot_fit_latest_edited_priors")
                if not isinstance(table, pd.DataFrame):
                    table = st.session_state.get("phot_fit_planet_priors_data")
                table = _repair_host_prior_parameter_names(table)
                mass, radius = _stellar_values_from_prior_table(table, fallback, fallback)
                return mass if label_text.startswith("Stellar mass") else radius
            return (original_call or original_number_input)(label, *args, **kwargs)

        def hidden_stellar_delta_number_input(self, label, *args, **kwargs):
            return hidden_stellar_number_input(
                label,
                *args,
                original_call=lambda inner_label, *inner_args, **inner_kwargs: original_delta_number_input(
                    self,
                    inner_label,
                    *inner_args,
                    **inner_kwargs,
                ),
                **kwargs,
            )

        def data_editor_with_impact_factor_language(data=None, *args, **kwargs):
            if isinstance(data, pd.DataFrame) and "parameter" in data.columns:
                data = _repair_host_prior_parameter_names(data)
                parameters = set(data["parameter"].astype(str))
                planet_parameter_names = {
                    "T0",
                    "period",
                    "radius_ratio",
                    "impact",
                    "impact_factor",
                    "stellar_mass",
                    "stellar_radius",
                }
                if not parameters.issubset(planet_parameter_names):
                    return original_data_editor(data, *args, **kwargs)
                data = data.copy()
                data["parameter"] = data["parameter"].replace({"impact": "impact_factor"})
                column_config = kwargs.get("column_config")
                if isinstance(column_config, dict) and "parameter" in column_config:
                    column_config = dict(column_config)
                    column_config["parameter"] = st.column_config.SelectboxColumn(
                        "parameter",
                        options=["stellar_mass", "stellar_radius", "T0", "period", "radius_ratio", "impact_factor"],
                        help=(
                            "User-facing transit parameter. impact_factor is converted to allesfitter's native "
                            "cosi using cos(i) = impact_factor / (a/Rstar)."
                        ),
                    )
                    kwargs["column_config"] = column_config
                edited = original_data_editor(data, *args, **kwargs)
                if isinstance(edited, pd.DataFrame) and "parameter" in edited.columns:
                    edited = edited.copy()
                    edited["parameter"] = edited["parameter"].replace({"impact_factor": "impact"})
                    edited = _repair_host_prior_parameter_names(edited)
                return edited
            return original_data_editor(data, *args, **kwargs)

        def dataframe_with_geometry_note(data=None, *args, **kwargs):
            result = original_dataframe(data, *args, **kwargs)
            generated = st.session_state.get("phot_fit_generated_params")
            if isinstance(data, pd.DataFrame) and isinstance(generated, pd.DataFrame) and data is generated:
                st.info(_impact_factor_note())
            return result

        def button_with_separate_param_generation(label, *args, **kwargs):
            label_text = str(label)
            if label_text != "Generate table and initial model from current priors":
                return original_button(label, *args, **kwargs)

            button_kwargs = dict(kwargs)
            button_kwargs.pop("key", None)
            cols = st.columns(2)
            with cols[0]:
                params_clicked = original_button(
                    "Generate params table from current priors",
                    *args,
                    key="phot_fit_generate_params_only",
                    **button_kwargs,
                )
            model_kwargs = dict(button_kwargs)
            model_kwargs.pop("type", None)
            with cols[1]:
                model_clicked = original_button(
                    "Generate initial model from generated table",
                    *args,
                    key="phot_fit_generate_initial_model_only",
                    **model_kwargs,
                )
            st.session_state["_ttv_generate_initial_model_clicked"] = bool(model_clicked)
            return bool(params_clicked or model_clicked)

        def batman_model_only_when_requested(*args, **kwargs):
            if st.session_state.get("_ttv_generate_initial_model_clicked"):
                return original_batman_model(*args, **kwargs)
            return None

        def plotly_chart_without_empty_initial_model(fig, *args, **kwargs):
            if kwargs.get("key") == "phot_fit_initial_model_overlay" and st.session_state.get("phot_fit_initial_model") is None:
                st.info("Generated the parameter table only. Use the separate initial-model button to evaluate and plot the model over loaded data.")
                return None
            return original_plotly_chart(fig, *args, **kwargs)

        def success_with_split_generate_message(body, *args, **kwargs):
            if (
                str(body) == "Generated allesfitter parameters and initial BATMAN model from the current prior table."
                and not st.session_state.get("_ttv_generate_initial_model_clicked")
            ):
                body = "Generated allesfitter parameters from the current prior table. Initial model generation was skipped."
            return original_success(body, *args, **kwargs)

        st.data_editor = data_editor_with_impact_factor_language
        st.dataframe = dataframe_with_geometry_note
        st.number_input = hidden_stellar_number_input
        st.button = button_with_separate_param_generation
        st.plotly_chart = plotly_chart_without_empty_initial_model
        st.success = success_with_split_generate_message
        delta_generator_cls.number_input = hidden_stellar_delta_number_input
        photometry_fit_module._batman_model_on_data = batman_model_only_when_requested
        try:
            return render_call()
        finally:
            st.data_editor = original_data_editor
            st.dataframe = original_dataframe
            st.number_input = original_number_input
            st.button = original_button
            st.plotly_chart = original_plotly_chart
            st.success = original_success
            delta_generator_cls.number_input = original_delta_number_input
            photometry_fit_module._batman_model_on_data = original_batman_model

    def render_setup_with_impact_factor_note() -> None:
        st.session_state.pop("_ttv_generate_initial_model_clicked", None)
        return patched_setup_context(original_setup)

    def render_with_impact_factor_note() -> None:
        return original_render()

    photometry_fit_module._build_allesfitter_params_from_setup = build_params_with_impact_factor_labels
    photometry_fit_module._run_least_squares_refinement = run_least_squares_with_host_rows
    photometry_fit_module._render_transit_parameter_setup = render_setup_with_impact_factor_note
    photometry_fit_module.render = render_with_impact_factor_note
    photometry_fit_module._ttv_fitter_original_build_params_for_impact_language = original_build_params
    photometry_fit_module._ttv_fitter_original_run_least_squares_for_host_rows = original_run_least_squares
    photometry_fit_module._ttv_fitter_original_transit_parameter_setup_for_host_rows = original_setup
    photometry_fit_module._ttv_fitter_impact_factor_language_patch = True


def _pid_state(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        result = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "stat="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _clear_sampler_state() -> None:
    for key in [
        "phot_fit_sampler_pid",
        "phot_fit_sampler_total_steps",
        "phot_fit_sampler_stopped",
    ]:
        st.session_state.pop(key, None)


def patch_photometry_fit_sampler_controls(photometry_fit_module) -> None:
    """Make sampler process state robust and steer expensive MCMC runs toward fast_fit windows."""
    if getattr(photometry_fit_module, "_ttv_fitter_sampler_controls_patch", False):
        return

    original_launch_sampler = photometry_fit_module._launch_sampler
    original_render = photometry_fit_module.render

    def process_is_running(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
        except OSError:
            return False
        state = _pid_state(pid)
        if state.startswith("Z"):
            return False
        return True

    def stop_sampler(pid: int | None) -> tuple[bool, str]:
        if not pid:
            _clear_sampler_state()
            return False, "No sampler process is recorded."
        pid_int = int(pid)
        if not process_is_running(pid_int):
            _clear_sampler_state()
            st.toast("That sampler process has already stopped.")
            st.rerun()
        current_group = os.getpgrp()
        try:
            process_group = os.getpgid(pid_int)
        except OSError:
            process_group = None
        try:
            if process_group and process_group != current_group:
                os.killpg(process_group, signal.SIGTERM)
                target = f"sampler process group {process_group}"
            else:
                os.kill(pid_int, signal.SIGTERM)
                target = f"sampler PID {pid_int}"
        except ProcessLookupError:
            _clear_sampler_state()
            st.toast("That sampler process has already stopped.")
            st.rerun()
        except OSError as exc:
            return False, f"Could not stop sampler process: {exc}"
        _clear_sampler_state()
        st.toast(f"Sent stop signal to {target}.")
        st.rerun()

    def sampler_is_active(pid: int | None, sampler: str, fit_dir: str | Path | None, total_steps: int, launched_total_steps: int | None = None) -> bool:
        if not process_is_running(pid):
            return False
        if sampler == "MCMC":
            iteration = photometry_fit_module._mcmc_iteration(fit_dir)
            target_steps = int(launched_total_steps or total_steps)
            if iteration is not None and iteration >= target_steps:
                return False
        return True

    def launch_sampler(fit_dir: Path, sampler: str, *, initial_guess_first: bool, fresh_start: bool):
        archive_dir = photometry_fit_module._archive_previous_sampler_outputs(fit_dir, sampler) if fresh_start else None
        call = "allesfitter.mcmc_fit(datadir)" if sampler == "MCMC" else "allesfitter.ns_fit(datadir)"
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
            start_new_session=True,
        )
        return process, archive_dir

    def render_with_fast_fit_default() -> None:
        original_checkbox = st.checkbox

        def checkbox_with_fast_fit_default(label, *args, **kwargs):
            if str(label) == "Use fast_fit transit windows":
                if "phot_fit_fast_fit" not in st.session_state:
                    kwargs["value"] = True
                kwargs.setdefault(
                    "help",
                    "Restricts each likelihood call to windows around predicted transits. "
                    "For TESS sectors this is usually much faster than fitting every out-of-transit point.",
                )
                result = original_checkbox(label, *args, **kwargs)
                st.caption(
                    "Fast-fit windows tell allesfitter to evaluate photometry only near predicted transits. "
                    "The generated settings use `fast_fit_width = 0.333333 d`, so each transit keeps about "
                    "8 hours total, centered on the current ephemeris. This is usually appropriate for transit-only "
                    "MCMC because distant out-of-transit baseline points contribute little to T0, period, radius ratio, "
                    "and impact-factor constraints but dominate runtime."
                )
                if not result:
                    st.warning(
                        "fast_fit is off, so allesfitter will evaluate the full photometry table on every MCMC step. "
                        "That can easily make 59k-point TESS fits run at only a few iterations per minute."
                    )
                return result
            return original_checkbox(label, *args, **kwargs)

        st.checkbox = checkbox_with_fast_fit_default
        try:
            return original_render()
        finally:
            st.checkbox = original_checkbox

    photometry_fit_module._ttv_fitter_original_launch_sampler = original_launch_sampler
    photometry_fit_module._launch_sampler = launch_sampler
    photometry_fit_module._process_is_running = process_is_running
    photometry_fit_module._stop_sampler = stop_sampler
    photometry_fit_module._sampler_is_active = sampler_is_active
    photometry_fit_module.render = render_with_fast_fit_default
    photometry_fit_module._ttv_fitter_sampler_controls_patch = True


def patch_photometry_fit_exofop_resolution(mast_module, photometry_fit_module) -> None:
    """Let the ExoFOP search field resolve names through the same SIMBAD TIC lookup as MAST."""
    if getattr(photometry_fit_module, "_ttv_fitter_exofop_resolution_patch", False):
        return

    original_extract_tic_id = photometry_fit_module.extract_tic_id

    @lru_cache(maxsize=128)
    def resolve_tic_from_text(raw_value: str) -> tuple[str, str]:
        text = " ".join(str(raw_value or "").strip().split())
        tic = original_extract_tic_id(text)
        if tic:
            return tic, ""
        if not text:
            return "", "Enter a target name or TIC ID before searching ExoFOP."
        first_error = ""
        try:
            resolved = mast_module._find_tic_from_simbad(mast_module.normalize_target_name(text))
        except Exception as exc:  # noqa: BLE001 - user-facing resolver/network issue
            first_error = str(exc)
            resolved = ""
        if not resolved:
            resolved = _find_tic_from_local_toi_list(text) or _find_tic_from_simbad_identifiers(text)
        if not resolved:
            if first_error:
                return (
                    "",
                    f"Could not resolve `{text}` to a TIC ID via SIMBAD. "
                    f"The primary resolver failed with: {first_error}",
                )
            return "", f"Could not resolve `{text}` to a TIC ID via SIMBAD. Try entering a TIC ID directly."
        return str(resolved), ""

    def resolved_extract_tic_id(value: str | int | None) -> str:
        tic = original_extract_tic_id(value)
        if tic:
            return tic
        resolved, _error = resolve_tic_from_text(str(value or ""))
        return resolved

    def current_tic_id() -> str:
        # Keep page render cheap: only use direct TIC-looking values here.
        # Free-form names are resolved when the user presses Search ExoFOP.
        for key in ["mast_resolved_target", "target_name", "photometry_target_input"]:
            tic = original_extract_tic_id(st.session_state.get(key))
            if tic:
                return tic
        return ""

    def render_exofop_retrieval() -> None:
        st.subheader("Retrieve parameters from ExoFOP")
        tic_id = current_tic_id()
        target_name = st.session_state.get("target_name") or st.session_state.get("photometry_target_input") or ""
        cache_info = photometry_fit_module.cached_toi_info()

        input_col, button_col = st.columns([0.7, 0.3], vertical_alignment="bottom")
        with input_col:
            tic_input = st.text_input(
                "Target name or TIC ID for ExoFOP search",
                value=tic_id or target_name,
                placeholder="TIC 243921117 or WASP-80",
                key="phot_fit_exofop_tic_input",
                help="A plain target name is resolved to a TIC ID with the same SIMBAD lookup used by the MAST photometry query.",
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
            search_tic, resolve_error = resolve_tic_from_text(tic_input)
            if not search_tic:
                st.error(resolve_error)
            else:
                action = "Searching saved ExoFOP TOI list" if use_cached and cache_info["exists"] else "Downloading ExoFOP TOI list"
                with st.spinner(f"{action} for TIC {search_tic}..."):
                    try:
                        toi_table, source_info = photometry_fit_module.load_toi_table(use_cached=use_cached)
                        matches = photometry_fit_module.find_toi_parameters(toi_table, search_tic)
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
            st.caption(f"Target context: {target_name or 'not set'}. ExoFOP lookup will list every TOI row for the resolved TIC.")
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
        display_columns = photometry_fit_module.preferred_toi_columns(matches)
        st.dataframe(matches[display_columns], use_container_width=True, hide_index=True)

        with st.expander("Show all ExoFOP columns"):
            st.dataframe(matches, use_container_width=True, hide_index=True)

    photometry_fit_module.extract_tic_id = resolved_extract_tic_id
    photometry_fit_module._current_tic_id = current_tic_id
    photometry_fit_module._render_exofop_retrieval = render_exofop_retrieval
    photometry_fit_module._ttv_fitter_original_extract_tic_id = original_extract_tic_id
    photometry_fit_module._ttv_fitter_resolve_tic_from_text = resolve_tic_from_text
    photometry_fit_module._ttv_fitter_exofop_resolution_patch = True


def filter_existing_mast_result(mast_module) -> None:
    result = st.session_state.get("mast_result")
    if result is None or getattr(result, "_ttv_fitter_filtered", False):
        return
    products = _best_light_curve_products(result)
    observations = result.observations
    if not products.empty and "obs_id" in products.columns and "obs_id" in observations.columns:
        observations = observations.loc[observations["obs_id"].isin(products["obs_id"])].copy()
    filtered = mast_module.MastQueryResult(
        target=result.target,
        resolved_target=result.resolved_target,
        observations=observations.reset_index(drop=True),
        products=products,
    )
    object.__setattr__(filtered, "_ttv_fitter_filtered", True)
    st.session_state["mast_result"] = filtered


def _download_products_from_original(mast_module, original_query, target: str, cadence: str, product_filenames: list[str], download_root: str | Path):
    if not product_filenames:
        return []
    from astroquery.mast import Observations

    result = original_query(target, cadence)
    if result.observations.empty:
        return []
    observation_table = mast_module._query_by_target_name(result.resolved_target)
    products = Observations.get_product_list(observation_table)
    product_df = products.to_pandas()
    selected_mask = product_df["productFilename"].isin(product_filenames)
    selected_products = products[selected_mask.to_numpy()]
    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    manifest = Observations.download_products(selected_products, download_dir=str(download_root))
    if manifest is None or len(manifest) == 0 or "Local Path" not in manifest.colnames:
        return []
    return [Path(path) for path in manifest["Local Path"] if path]


def _download_products_by_uri(products: pd.DataFrame, download_root: str | Path) -> list[Path]:
    from astroquery.mast import Observations

    download_root = Path(download_root)
    download_root.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for _, row in products.iterrows():
        data_uri = str(row.get("dataURI", "") or "").strip()
        if not data_uri:
            continue
        filename = str(row.get("productFilename", "") or "").strip() or _product_filename_from_mast_uri(data_uri)
        if not filename:
            continue
        local_path = download_root / Path(filename).name
        Observations.download_file(data_uri, local_path=str(local_path), cache=True, verbose=False)
        if local_path.exists():
            downloaded.append(local_path)
    return downloaded


def _selected_product_local_paths(products: pd.DataFrame, product_filenames: list[str], download_root: str | Path) -> pd.DataFrame:
    if products.empty or "productFilename" not in products.columns or not product_filenames:
        return pd.DataFrame()
    selected_names = {str(name) for name in product_filenames}
    selected = products.loc[products["productFilename"].astype(str).isin(selected_names)].copy()
    if selected.empty:
        return selected
    root = Path(download_root)
    selected["local_path"] = selected["productFilename"].astype(str).map(lambda name: str(root / Path(name).name))
    selected["already_downloaded"] = selected["local_path"].map(lambda path: Path(path).exists())
    return selected


def patch_mast_product_download_button(photometry_import_module) -> None:
    if getattr(photometry_import_module, "_ttv_fitter_download_button_patch", False):
        return

    def display_mast_result(result) -> None:
        st.success(
            f"Found {len(result.observations)} TESS/HLSP time-series observation(s) "
            f"for `{result.target}` using MAST target `{result.resolved_target}`."
        )
        if result.observations.empty:
            st.warning("No matching TESS time-series observations were returned for the selected cadence.")
            return

        preferred = [
            "target_name",
            "obs_collection",
            "provenance_name",
            "sequence_number",
            "t_exptime",
            "t_min",
            "t_max",
            "obs_id",
        ]
        obs_columns = [col for col in preferred if col in result.observations.columns]
        st.dataframe(result.observations[obs_columns], use_container_width=True, hide_index=True)

        if result.products.empty:
            st.warning("No light-curve/FITS products were found for these observations.")
            return

        st.caption(f"Light-curve/FITS product candidates: {len(result.products)}")
        product_columns = [
            col
            for col in [
                "obs_id",
                "productFilename",
                "description",
                "productType",
                "productSubGroupDescription",
                "size",
            ]
            if col in result.products.columns
        ]
        selectable = result.products[product_columns].copy()
        selectable.insert(0, "download", True)
        selected = st.data_editor(
            selectable,
            use_container_width=True,
            hide_index=True,
            key="mast_product_selection",
            column_config={"download": st.column_config.CheckboxColumn("download")},
        )
        selected_files = selected.loc[selected["download"], "productFilename"].dropna().astype(str).tolist()
        st.caption(f"Selected products: {len(selected_files)}")

        download_root = photometry_import_module._target_download_root(result)
        selected_status = _selected_product_local_paths(result.products, selected_files, download_root)
        existing_count = int(selected_status["already_downloaded"].sum()) if not selected_status.empty else 0
        missing_files = (
            selected_status.loc[~selected_status["already_downloaded"], "productFilename"].astype(str).tolist()
            if not selected_status.empty
            else []
        )
        if selected_files:
            st.caption(f"Already obtained in this folder: {existing_count}; missing from selection: {len(missing_files)}.")

        button_col, path_col = st.columns([0.28, 0.72], vertical_alignment="center")
        with button_col:
            if st.button("Download selected products if not already obtained", use_container_width=True, disabled=not selected_files):
                try:
                    if missing_files:
                        with st.spinner(f"Downloading {len(missing_files)} missing product(s) to {download_root}..."):
                            photometry_import_module.download_tess_products(
                                result.target,
                                st.session_state.get("mast_cadence_filter", "2 min"),
                                missing_files,
                                download_root,
                            )
                    refreshed = _selected_product_local_paths(result.products, selected_files, download_root)
                    paths = refreshed["local_path"].astype(str).tolist() if not refreshed.empty else []
                    st.session_state["mast_downloaded_paths"] = paths
                    st.session_state.pop("mast_download_error", None)
                    st.success(f"Loaded {len([path for path in paths if Path(path).exists()])} selected product file(s) into the sector workflow.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - user-facing network/file error
                    st.session_state["mast_download_error"] = str(exc)
        with path_col:
            st.text_input("Download folder", value=str(download_root), disabled=True)

        if st.session_state.get("mast_download_error"):
            st.error(st.session_state["mast_download_error"])

    photometry_import_module._display_mast_result = display_mast_result
    photometry_import_module._ttv_fitter_download_button_patch = True


def bridge_prepared_photometry() -> None:
    prepared = st.session_state.get("prepared_photometry")
    if isinstance(prepared, pd.DataFrame) and not prepared.empty:
        export = prepared.copy()
        if "is_outlier" in export.columns:
            export = export.loc[~export["is_outlier"]].copy()
        st.session_state["photometry"] = normalize_photometry(export)


def bridge_fit_planets() -> None:
    params = st.session_state.get("phot_fit_generated_params")
    if not isinstance(params, pd.DataFrame) or params.empty or "name" not in params.columns:
        return
    values: dict[str, float] = {}
    for _, row in params.iterrows():
        try:
            values[str(row["name"])] = float(row["value"])
        except (TypeError, ValueError):
            continue
    rows = []
    companions = sorted({name.split("_", 1)[0] for name in values if name.endswith("_period")})
    for companion in companions:
        period = values.get(f"{companion}_period", 1.0)
        t0 = values.get(f"{companion}_epoch", 0.0)
        rr = values.get(f"{companion}_rr", 0.08)
        rsuma = values.get(f"{companion}_rsuma", np.nan)
        cosi = values.get(f"{companion}_cosi", np.nan)
        a_over_rstar = (1.0 + rr) / rsuma if np.isfinite(rsuma) and rsuma > 0 else np.nan
        inclination = float(np.rad2deg(np.arccos(np.clip(cosi, 0.0, 1.0)))) if np.isfinite(cosi) else 87.0
        impact = a_over_rstar * cosi if np.isfinite(a_over_rstar) and np.isfinite(cosi) else 0.4
        rows.append(
            {
                "name": companion,
                "period": period,
                "t0": t0,
                "radius_ratio": rr,
                "impact": impact,
                "a_over_rstar": a_over_rstar,
                "inclination_deg": inclination,
            }
        )
    if rows:
        st.session_state["planets"] = coerce_planet_table(pd.DataFrame(rows))


def _clean_fit_sector_table(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    data = normalize_photometry(frame)
    if "source_file" not in data:
        data["source_file"] = source
    if "sector" not in data:
        data["sector"] = Path(source).stem
    if "is_outlier" not in data:
        data["is_outlier"] = False
    return data


def _sector_tables_from_directory(directory: str) -> dict[str, pd.DataFrame]:
    root = Path(str(directory)).expanduser()
    if not root.exists() or not root.is_dir():
        return {}
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(root.glob("*.csv")):
        if path.name.lower() == "sector_summary.csv":
            continue
        try:
            table = _clean_fit_sector_table(pd.read_csv(path), path.name)
        except Exception:
            continue
        if not table.empty:
            key = str(table["sector"].iloc[0]) if "sector" in table else path.stem
            tables[key] = table
    return tables


def _default_sector_directory() -> str:
    prepared = st.session_state.get("prepared_sector_directory")
    if prepared:
        return str(prepared)
    target = _target_name_for_exofop_lookup()
    clean_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target or "TOI-216")).strip("_") or "TOI-216"
    return str(Path("data") / "prepared" / clean_target / "sectors")


def _load_linear_fit_data_from_tables(tables: dict[str, pd.DataFrame], source: str) -> None:
    cleaned = {str(key): _clean_fit_sector_table(table, str(key)) for key, table in tables.items() if isinstance(table, pd.DataFrame) and not table.empty}
    if not cleaned:
        return
    combined = pd.concat(cleaned.values(), ignore_index=True).sort_values("time").reset_index(drop=True)
    st.session_state["photometry_fit_sector_data"] = cleaned
    st.session_state["photometry_fit_data_mode"] = "sectors"
    st.session_state["photometry_fit_data"] = combined
    st.session_state["photometry_fit_data_source"] = source


def _load_linear_fit_stitched_data(frame: pd.DataFrame, source: str) -> None:
    data = _clean_fit_sector_table(frame, source).sort_values("time").reset_index(drop=True)
    st.session_state["photometry_fit_data"] = data
    st.session_state["photometry_fit_data_source"] = source
    st.session_state["photometry_fit_data_mode"] = "stitched"
    st.session_state.pop("photometry_fit_sector_data", None)


def _render_linear_fit_data_loader(photometry_fit_module) -> None:
    st.subheader("Load photometry data")
    prepared = st.session_state.get("prepared_photometry")
    prepared_sectors = st.session_state.get("prepared_sector_photometry")

    import_col, sector_col, dir_col = st.columns(3, vertical_alignment="top")
    with import_col:
        st.caption("Use the stitched data from the TTV data preparation workflow.")
        if isinstance(prepared, pd.DataFrame) and not prepared.empty:
            if st.button("Import stitched data from data preparation", use_container_width=True, type="primary"):
                export = prepared.loc[~prepared["is_outlier"]].copy() if "is_outlier" in prepared else prepared.copy()
                _load_linear_fit_stitched_data(export, "TTV data preparation stitched data")
                st.rerun()
        else:
            st.button("Import stitched data from data preparation", use_container_width=True, disabled=True)
            st.info("No stitched prepared data is available yet.")

    with sector_col:
        st.caption("Use the unstitched sector tables prepared in the previous tab.")
        if isinstance(prepared_sectors, dict) and prepared_sectors:
            if st.button("Import sector data from data preparation", use_container_width=True):
                _load_linear_fit_data_from_tables(prepared_sectors, "TTV data preparation sector data")
                st.rerun()
        else:
            st.button("Import sector data from data preparation", use_container_width=True, disabled=True)
            st.info("Prepare sector data first, or load a sector directory.")

    with dir_col:
        st.caption("Load a local directory containing one prepared CSV per sector.")
        directory = st.text_input("Sector directory", value=_default_sector_directory(), key="photometry_fit_sector_directory")
        if st.button("Load sector directory", use_container_width=True):
            tables = _sector_tables_from_directory(directory)
            if not tables:
                st.warning("No usable sector CSV files were found in that directory.")
            else:
                _load_linear_fit_data_from_tables(tables, f"Sector directory: {directory}")
                st.success(f"Loaded {len(tables)} sector file(s).")
                st.rerun()

    uploaded = st.file_uploader(
        "Or load prepared photometry CSV",
        type=["csv"],
        key="photometry_fit_csv_upload",
        help="Expected columns: time, flux, flux_err. Optional columns such as sector and source_file are kept.",
    )
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")
        else:
            if st.button("Use uploaded CSV", use_container_width=True):
                _load_linear_fit_stitched_data(frame, uploaded.name)
                st.rerun()

    data = st.session_state.get("photometry_fit_data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return

    mode = st.session_state.get("photometry_fit_data_mode", "stitched")
    st.caption(f"Current fit data source: {st.session_state.get('photometry_fit_data_source', 'unknown')}")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Points", f"{len(data):,}")
    metric_cols[1].metric("Sectors", f"{data['sector'].nunique() if 'sector' in data else 0}")
    metric_cols[2].metric("Time min", f"{data['time'].min():.4f}")
    metric_cols[3].metric("Time max", f"{data['time'].max():.4f}")

    plot_data = data
    if mode == "sectors":
        sector_tables = st.session_state.get("photometry_fit_sector_data", {})
        if isinstance(sector_tables, dict) and sector_tables:
            sector_key = st.selectbox("Sector to plot", list(sector_tables.keys()), key="photometry_fit_sector_plot_select")
            plot_data = sector_tables[sector_key]
    plot_stride = _simple_display_stride(len(plot_data), max_display_points=20_000)
    if plot_stride > 1:
        displayed_points = len(plot_data.iloc[::plot_stride])
        st.info(f"{_stride_note(len(plot_data), displayed_points, plot_stride)} Fits still use all loaded rows.")

    st.plotly_chart(
        photometry_fit_module.prepared_photometry_preview(plot_data),
        use_container_width=True,
        config=photometry_fit_module.PLOT_CONFIG,
        key="photometry_fit_loaded_data_preview",
    )


def _single_transit_data_sources() -> dict[str, pd.DataFrame]:
    sources: dict[str, pd.DataFrame] = {}
    for label, key in [
        ("Current Linear fit photometry", "photometry_fit_data"),
        ("Prepared photometry", "prepared_photometry"),
        ("Bridge photometry", "photometry"),
    ]:
        data = st.session_state.get(key)
        if isinstance(data, pd.DataFrame) and not data.empty:
            sources[label] = normalize_photometry(data)
    sector_frames = st.session_state.get("sector_frames", {})
    if isinstance(sector_frames, dict):
        for sector, frame in sector_frames.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                sources[f"Loaded sector {sector}"] = normalize_photometry(frame)
    return sources


def _single_transit_subset_options(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    options = {"All selected data": data}
    for column, prefix in [("sector", "Sector"), ("source_file", "Dataset")]:
        if column not in data.columns:
            continue
        for value in sorted(data[column].dropna().astype(str).unique()):
            if value:
                subset = data.loc[data[column].astype(str) == value].copy()
                if not subset.empty:
                    options[f"{prefix}: {value}"] = subset
    return options


def _single_transit_start_values(planet_name: str, data: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    params = st.session_state.get("phot_fit_generated_params")
    if isinstance(params, pd.DataFrame) and not params.empty and {"name", "value"}.issubset(params.columns):
        raw: dict[str, float] = {}
        for _, row in params.iterrows():
            try:
                raw[str(row["name"])] = float(row["value"])
            except (TypeError, ValueError):
                continue
        rr = raw.get(f"{planet_name}_rr", raw.get(f"{planet_name}_radius_ratio", np.nan))
        rsuma = raw.get(f"{planet_name}_rsuma", np.nan)
        cosi = raw.get(f"{planet_name}_cosi", np.nan)
        a_over_rstar = raw.get(f"{planet_name}_a", raw.get(f"{planet_name}_a_over_rstar", np.nan))
        if np.isfinite(rr) and np.isfinite(rsuma) and rsuma > 0:
            a_over_rstar = (1.0 + rr) / rsuma
        impact = raw.get(f"{planet_name}_impact", np.nan)
        if not np.isfinite(impact) and np.isfinite(a_over_rstar) and np.isfinite(cosi):
            impact = float(a_over_rstar) * float(cosi)
        instrument = "TESS"
        for name in raw:
            if name.startswith("baseline_offset_flux_"):
                instrument = name.removeprefix("baseline_offset_flux_")
                break
        values = {
            "t0": raw.get(f"{planet_name}_epoch", np.nan),
            "period": raw.get(f"{planet_name}_period", np.nan),
            "radius_ratio": rr,
            "impact": impact,
            "a_over_rstar": a_over_rstar,
            "limb_darkening_u1": raw.get(f"host_ldc_u1_{instrument}", np.nan),
            "limb_darkening_u2": raw.get(f"host_ldc_u2_{instrument}", np.nan),
            "baseline_offset": raw.get(f"baseline_offset_flux_{instrument}", 0.0),
        }
        values = {key: value for key, value in values.items() if np.isfinite(float(value))}

    planets = coerce_planet_table(st.session_state.get("planets"))
    match = planets.loc[planets["name"].astype(str) == str(planet_name)]
    record = match.iloc[0].to_dict() if not match.empty else {}
    for key, fallback in [
        ("t0", record.get("t0", float(data["time"].median()))),
        ("period", record.get("period", 1.0)),
        ("radius_ratio", record.get("radius_ratio", 0.1)),
        ("impact", record.get("impact", 0.5)),
        ("a_over_rstar", record.get("a_over_rstar", 10.0)),
        ("duration_hours", record.get("duration_hours", 3.0)),
    ]:
        if key not in values or not np.isfinite(float(values[key])):
            values[key] = fallback
    defaults = {
        "t0": float(data["time"].median()),
        "period": 1.0,
        "radius_ratio": 0.1,
        "impact": 0.5,
        "duration_hours": 3.0,
        "a_over_rstar": 10.0,
        "limb_darkening_u1": 0.5,
        "limb_darkening_u2": 0.1,
        "baseline_offset": 0.0,
    }
    defaults.update({key: value for key, value in values.items() if pd.notna(value)})
    return {key: float(value) for key, value in defaults.items()}


def _single_transit_plot_frame(data: pd.DataFrame, max_display_points: int = 20_000) -> tuple[pd.DataFrame, int]:
    stride = _simple_display_stride(len(data), max_display_points=max_display_points)
    if stride <= 1:
        return data, 1
    return data.iloc[::stride].copy(), stride


def _single_transit_window_figure(
    data: pd.DataFrame,
    start: float | None,
    end: float | None,
    *,
    show_errors: bool = False,
    marker_opacity: float = 0.38,
) -> go.Figure:
    fig = go.Figure()
    err = data["flux_err"] if show_errors and "flux_err" in data.columns else None
    fig.add_trace(
        go.Scattergl(
            x=data["time"],
            y=data["flux"],
            error_y=dict(type="data", array=err, visible=err is not None),
            mode="markers",
            marker=dict(size=4, color="#2563eb", opacity=marker_opacity),
            name="photometry",
        )
    )
    if start is not None and end is not None and np.isfinite(start) and np.isfinite(end) and end > start:
        fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=0.14, line_width=0)
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=30, b=45),
        dragmode="select",
        xaxis_title="Time",
        yaxis_title="Flux",
        uirevision="single_transit_window",
    )
    return fig


def _single_transit_fit_figure(data: pd.DataFrame, fit: dict[str, float]) -> go.Figure:
    from .models import limb_darkened_transit_model

    plot_data, _stride = _single_transit_plot_frame(data)
    fig = _single_transit_window_figure(plot_data, None, None, show_errors=False, marker_opacity=0.24)
    time = np.linspace(float(data["time"].min()), float(data["time"].max()), 800)
    model = limb_darkened_transit_model(
        time,
        float(fit["period"]),
        float(fit["t0"]),
        float(fit["radius_ratio"]),
        float(fit["impact"]),
        float(fit["duration_hours"]),
        float(fit["limb_darkening_u1"]),
        float(fit["limb_darkening_u2"]),
        baseline_offset=float(fit.get("baseline_offset", 0.0)),
        a_over_rstar=float(fit.get("a_over_rstar", np.nan)),
    )
    fig.add_trace(
        go.Scattergl(
            x=time,
            y=model,
            mode="lines",
            line=dict(color="#ef4444", width=6),
            name="single-transit LS fit",
        )
    )
    return fig


def _exofop_single_transit_seed(planet_name: str, data: pd.DataFrame) -> dict[str, float]:
    matches = st.session_state.get("phot_fit_exofop_matches")
    if not isinstance(matches, pd.DataFrame) or matches.empty:
        return {}
    planet_index = max(ord(str(planet_name or "b")[0].lower()) - ord("b"), 0)
    source = matches.iloc[min(planet_index, len(matches) - 1)]
    period = _coerce_float(source.get("Period (days)", np.nan), np.nan)
    t0 = _coerce_float(source.get("Epoch (BJD)", np.nan), np.nan)
    if np.isfinite(t0) and t0 > 100000 and float(data["time"].max()) < 100000:
        t0 -= 2457000.0
    if np.isfinite(t0) and np.isfinite(period) and period > 0:
        median_time = float(data["time"].median())
        t0 = float(t0 + np.round((median_time - t0) / period) * period)

    stellar_radius = _coerce_float(source.get("Stellar Radius (R_Sun)", np.nan), np.nan)
    planet_radius = _coerce_float(source.get("Planet Radius (R_Earth)", np.nan), np.nan)
    radius_ratio = np.nan
    if np.isfinite(planet_radius) and np.isfinite(stellar_radius) and stellar_radius > 0:
        radius_ratio = planet_radius * 0.0091577 / stellar_radius
    depth_ppm = _coerce_float(source.get("Depth (ppm)", np.nan), np.nan)
    if not np.isfinite(radius_ratio) and np.isfinite(depth_ppm):
        radius_ratio = math.sqrt(max(depth_ppm, 0.0) / 1_000_000.0)

    duration_hours = np.nan
    for column in ["Duration (hours)", "Duration (hrs)", "Duration (days)", "Transit Duration (hours)", "Transit Duration (days)"]:
        duration_hours = _coerce_float(source.get(column, np.nan), np.nan)
        if np.isfinite(duration_hours):
            if "days" in column.lower():
                duration_hours *= 24.0
            break

    stellar_mass = _coerce_float(source.get("Stellar Mass (M_Sun)", np.nan), np.nan)
    a_over_rstar = derive_a_over_rstar(period, stellar_mass, stellar_radius) if np.isfinite(period) else np.nan
    seed = {
        "t0": t0,
        "period": period,
        "radius_ratio": radius_ratio,
        "impact": 0.5,
        "duration_hours": duration_hours,
        "a_over_rstar": a_over_rstar,
        "limb_darkening_u1": 0.5,
        "limb_darkening_u2": 0.1,
        "baseline_offset": 0.0,
    }
    return {key: float(value) for key, value in seed.items() if np.isfinite(float(value))}


def _render_single_transit_fit_subtab() -> None:
    st.subheader("Fit One Transit")
    st.caption("Use one clean transit to estimate the transit shape when large TTVs make a global linear-transit fit unreliable.")
    bridge_fit_planets()

    upload = st.file_uploader(
        "Upload photometry table for single-transit fitting",
        type=["csv", "tsv", "txt", "dat"],
        key="single_transit_photometry_upload",
        help="Optional. Expected columns are time and flux; flux_err, sector, and source_file are used if present.",
    )
    if upload is not None:
        try:
            st.session_state["single_transit_uploaded_photometry"] = normalize_photometry(read_table(upload))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read uploaded photometry: {exc}")

    sources = _single_transit_data_sources()
    uploaded = st.session_state.get("single_transit_uploaded_photometry")
    if isinstance(uploaded, pd.DataFrame) and not uploaded.empty:
        sources["Uploaded single-transit table"] = normalize_photometry(uploaded)
    if not sources:
        st.info("Load prepared photometry on the Data Import tab, load data in Linear fit, or upload a table here.")
        return

    source_label = st.selectbox("Photometry source", list(sources.keys()), key="single_transit_source")
    source_data = sources[source_label].copy()
    subsets = _single_transit_subset_options(source_data)
    subset_label = st.selectbox("Sector or dataset", list(subsets.keys()), key="single_transit_subset")
    data = subsets[subset_label].sort_values("time").reset_index(drop=True)
    if data.empty:
        st.info("The selected data set is empty.")
        return

    planets = coerce_planet_table(st.session_state.get("planets"))
    if planets.empty:
        st.info("Create or retrieve planet parameters first so each planet can be fit separately.")
        return
    planet_options = []
    planet_lookup = {}
    for _, row in planets.iterrows():
        name = str(row.get("name", ""))
        period = _coerce_float(row.get("period"), np.nan)
        label = f"{name} - period {period:.6g}d" if np.isfinite(period) else name
        planet_options.append(label)
        planet_lookup[label] = name
    planet_label = st.selectbox("Planet", planet_options, key="single_transit_planet")
    planet_name = planet_lookup[planet_label]
    defaults = _single_transit_start_values(planet_name, data)

    tmin = float(data["time"].min())
    tmax = float(data["time"].max())
    center = float(defaults.get("t0", data["time"].median()))
    if not tmin <= center <= tmax:
        center = float(data["time"].median())
    width = min(max(float(defaults.get("duration_hours", 3.0)) / 12.0, (tmax - tmin) * 0.05), max(tmax - tmin, 1e-6))

    pending_seed = st.session_state.pop("_single_transit_pending_exofop_seed", None)
    if isinstance(pending_seed, dict):
        key_map = {
            "t0": "single_transit_t0",
            "period": "single_transit_period",
            "radius_ratio": "single_transit_radius_ratio",
            "impact": "single_transit_impact",
            "duration_hours": "single_transit_duration",
            "a_over_rstar": "single_transit_a_over_rstar",
            "limb_darkening_u1": "single_transit_u1",
            "limb_darkening_u2": "single_transit_u2",
            "baseline_offset": "single_transit_baseline_offset",
        }
        for name, key in key_map.items():
            if name in pending_seed:
                st.session_state[key] = float(pending_seed[name])
        if "t0" in pending_seed and "duration_hours" in pending_seed:
            half_width = max(float(pending_seed["duration_hours"]) / 12.0, 0.05)
            st.session_state["single_transit_window_start"] = float(np.clip(float(pending_seed["t0"]) - half_width, tmin, tmax))
            st.session_state["single_transit_window_end"] = float(np.clip(float(pending_seed["t0"]) + half_width, tmin, tmax))
        st.success("Seeded single-transit start points from ExoFOP.")

    st.session_state.setdefault("single_transit_window_start", max(tmin, center - width))
    st.session_state.setdefault("single_transit_window_end", min(tmax, center + width))
    start = float(np.clip(st.session_state["single_transit_window_start"], tmin, tmax))
    end = float(np.clip(st.session_state["single_transit_window_end"], tmin, tmax))
    st.session_state["single_transit_window_start"] = start
    st.session_state["single_transit_window_end"] = end

    plot_data, plot_stride = _single_transit_plot_frame(data)
    if plot_stride > 1:
        st.info(f"{_stride_note(len(data), len(plot_data), plot_stride)} Fitting still uses every point in the selected window.")

    selection = st.plotly_chart(
        _single_transit_window_figure(plot_data, start, end, show_errors=False),
        width="stretch",
        key="single_transit_window_plot",
        on_select="rerun",
        selection_mode=("box", "lasso"),
        config={"displaylogo": False, "scrollZoom": True},
    )
    selected_points = getattr(selection, "selection", {}).get("points", []) if selection is not None else []
    selected_x = [point.get("x") for point in selected_points if point.get("x") is not None]
    if selected_x:
        new_start = float(np.nanmin(selected_x))
        new_end = float(np.nanmax(selected_x))
        if np.isfinite(new_start) and np.isfinite(new_end) and new_end > new_start:
            st.session_state["single_transit_window_start"] = new_start
            st.session_state["single_transit_window_end"] = new_end
            start, end = new_start, new_end

    range_cols = st.columns(2)
    start = range_cols[0].number_input("Window start", value=float(start), min_value=tmin, max_value=tmax, format="%.10f", key="single_transit_window_start")
    end = range_cols[1].number_input("Window end", value=float(end), min_value=tmin, max_value=tmax, format="%.10f", key="single_transit_window_end")
    window = data.loc[(data["time"] >= min(start, end)) & (data["time"] <= max(start, end))].copy()
    st.caption(f"Selected window contains {len(window):,} point(s).")

    if st.button("Seed single-transit start points from ExoFOP", use_container_width=True, key="single_transit_seed_exofop"):
        seed = _exofop_single_transit_seed(planet_name, data)
        if not seed:
            st.warning("No ExoFOP row is available yet. Use the ExoFOP retrieval on the Linear fit subtab first.")
        else:
            st.session_state["_single_transit_pending_exofop_seed"] = seed
            st.rerun()

    param_cols = st.columns(4)
    defaults["t0"] = param_cols[0].number_input("T0", value=float(defaults.get("t0", window["time"].median() if not window.empty else tmin)), format="%.10f", key="single_transit_t0")
    defaults["period"] = param_cols[1].number_input("Period [days]", value=max(float(defaults.get("period", 1.0)), 1e-8), min_value=1e-8, format="%.10f", key="single_transit_period")
    defaults["radius_ratio"] = param_cols[2].number_input("Radius ratio", value=float(defaults.get("radius_ratio", 0.1)), min_value=1e-6, max_value=1.0, format="%.8f", key="single_transit_radius_ratio")
    defaults["impact"] = param_cols[3].number_input("Impact parameter", value=float(defaults.get("impact", 0.5)), min_value=0.0, max_value=2.0, format="%.8f", key="single_transit_impact")
    shape_cols = st.columns(5)
    defaults["duration_hours"] = shape_cols[0].number_input("Duration [hours]", value=max(float(defaults.get("duration_hours", 3.0)), 0.01), min_value=0.001, format="%.6f", key="single_transit_duration")
    defaults["a_over_rstar"] = shape_cols[1].number_input("a/Rstar", value=max(float(defaults.get("a_over_rstar", 10.0)), 1.0001), min_value=1.0001, format="%.8f", key="single_transit_a_over_rstar")
    defaults["limb_darkening_u1"] = shape_cols[2].number_input("Limb darkening u1", value=float(defaults.get("limb_darkening_u1", 0.5)), min_value=-1.0, max_value=1.0, format="%.8f", key="single_transit_u1")
    defaults["limb_darkening_u2"] = shape_cols[3].number_input("Limb darkening u2", value=float(defaults.get("limb_darkening_u2", 0.1)), min_value=-1.0, max_value=1.0, format="%.8f", key="single_transit_u2")
    defaults["baseline_offset"] = shape_cols[4].number_input("Baseline offset flux", value=float(defaults.get("baseline_offset", 0.0)), min_value=-0.5, max_value=0.5, format="%.8f", key="single_transit_baseline_offset")

    fit_options = [
        "t0",
        "period",
        "radius_ratio",
        "impact",
        "duration_hours",
        "a_over_rstar",
        "limb_darkening_u1",
        "limb_darkening_u2",
        "baseline_offset",
    ]
    fit_parameters = st.multiselect(
        "Fit parameters",
        fit_options,
        default=[name for name in fit_options if name != "period"],
        key="single_transit_fit_parameters",
    )
    if "duration_hours" in fit_parameters and np.isfinite(float(defaults.get("a_over_rstar", np.nan))):
        st.caption("Duration is kept fixed when a/Rstar is available, because those two controls are degenerate for this single-transit model.")
    if st.button("Run single-transit least-squares fit", use_container_width=True, disabled=window.empty):
        from .fitting import fit_limb_darkened_single_transit

        result = fit_limb_darkened_single_transit(window, defaults, fit_parameters)
        if not result.success:
            st.warning(f"Single-transit fit stopped before formal convergence: {result.message}")
        fit = {
            "planet": planet_name,
            "source": source_label,
            "subset": subset_label,
            "start": float(min(start, end)),
            "end": float(max(start, end)),
            "points": int(len(window)),
            **result.params,
        }
        st.session_state["phot_fit_single_transit_ls_result"] = fit
        all_results = st.session_state.setdefault("phot_fit_single_transit_ls_results_by_planet", {})
        all_results[str(planet_name)] = fit
        st.session_state["phot_fit_single_transit_ls_results_by_planet"] = all_results
        st.success("Single-transit least-squares fit saved for the per-transit T0 page.")

    latest = st.session_state.get("phot_fit_single_transit_ls_result")
    if isinstance(latest, dict) and latest:
        fit_table = pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in latest.items() if isinstance(value, (int, float, np.floating))]
        )
        st.dataframe(fit_table, use_container_width=True, hide_index=True)
        fit_window = data.loc[(data["time"] >= float(latest["start"])) & (data["time"] <= float(latest["end"]))].copy()
        if not fit_window.empty:
            fit_plot_data, fit_stride = _single_transit_plot_frame(fit_window)
            if fit_stride > 1:
                st.info(f"{_stride_note(len(fit_window), len(fit_plot_data), fit_stride)} The single-transit fit used all {len(fit_window):,} selected points.")
            st.plotly_chart(
                _single_transit_fit_figure(fit_window, latest),
                width="stretch",
                key="single_transit_fit_overlay",
                config={"displaylogo": False, "scrollZoom": True},
            )

    all_results = st.session_state.get("phot_fit_single_transit_ls_results_by_planet", {})
    if isinstance(all_results, dict) and all_results:
        st.subheader("Single-transit LS fit outputs by planet")
        output_lines = []
        host_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", _target_name_for_exofop_lookup() or "target").strip("_") or "target"
        for planet, fit in sorted(all_results.items()):
            st.caption(f"Planet {planet}")
            fit_table = pd.DataFrame(
                [{"planet": planet, "parameter": key, "value": value} for key, value in fit.items() if isinstance(value, (int, float, np.floating))]
            )
            st.dataframe(fit_table, use_container_width=True, hide_index=True)
            source_data_for_fit = sources.get(str(fit.get("source", "")), data)
            fit_subsets = _single_transit_subset_options(source_data_for_fit)
            fit_data = fit_subsets.get(str(fit.get("subset", "")), source_data_for_fit)
            fit_window = fit_data.loc[(fit_data["time"] >= float(fit["start"])) & (fit_data["time"] <= float(fit["end"]))].copy()
            if not fit_window.empty:
                fit_plot_data, fit_stride = _single_transit_plot_frame(fit_window)
                if fit_stride > 1:
                    st.info(f"{_stride_note(len(fit_window), len(fit_plot_data), fit_stride)} The single-transit fit used all {len(fit_window):,} selected points.")
                st.plotly_chart(
                    _single_transit_fit_figure(fit_window, fit),
                    width="stretch",
                    key=f"single_transit_fit_overlay_{planet}",
                    config={"displaylogo": False, "scrollZoom": True},
                )
            output_lines.append(f"[planet {planet}]")
            for key, value in fit.items():
                output_lines.append(f"{key}: {value}")
            output_lines.append("")
        st.download_button(
            "Download single-transit planet parameters",
            data="\n".join(output_lines).encode("utf-8"),
            file_name=f"{host_name}_single_transit_planet_parameters.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_single_transit_planet_parameters",
        )


def render_import_workflows() -> None:
    photometry_import, _rv_import, _photometry_fit = import_allesfitter_pages()
    from app import mast  # type: ignore

    filter_existing_mast_result(mast)
    photometry_import.render()
    if st.session_state.get("mast_product_warning"):
        st.warning(st.session_state["mast_product_warning"])
    bridge_prepared_photometry()
    if not st.session_state.get("photometry", pd.DataFrame()).empty:
        st.success("Prepared photometry is available to the TTV and physical model tabs.")


def render_linear_transit_workflow() -> None:
    _photometry_import, _rv_import, photometry_fit = import_allesfitter_pages()
    st.subheader("Linear Transit Fit")
    _render_linear_fit_data_loader(photometry_fit)
    st.divider()
    photometry_fit._render_exofop_retrieval()
    st.divider()
    linear_tab, single_tab = st.tabs(["Linear fit", "Fit one transit"])
    with linear_tab:
        photometry_fit._render_transit_parameter_setup()
    with single_tab:
        _render_single_transit_fit_subtab()
    bridge_fit_planets()
    bridge_prepared_photometry()


def render_bridge_summary() -> None:
    planets = st.session_state.get("planets", pd.DataFrame())
    photometry = st.session_state.get("photometry", pd.DataFrame())
    rv = st.session_state.get("rv", pd.DataFrame())
    cols = st.columns(3)
    cols[0].metric("Bridge photometry rows", f"{len(photometry):,}")
    cols[1].metric("Bridge RV rows", f"{len(rv):,}")
    cols[2].metric("Bridge planets", f"{len(planets):,}")
