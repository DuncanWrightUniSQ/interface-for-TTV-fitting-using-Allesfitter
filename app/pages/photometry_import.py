"""Photometry import and preparation page."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from app.mast import MastQueryResult, download_tess_products, query_tess_photometry, safe_target_slug
from app.photometry import (
    estimate_uncertainty_for_region,
    flatten_sector,
    flattening_status_table,
    load_sector_frames,
    normalized_sector,
    sector_uncertainty_table,
    stitch_flattened_sectors,
    wotan_trend,
)
from app.plots import (
    PLOT_CONFIG,
    flattened_sector_preview,
    prepared_photometry_preview,
    sector_uncertainty_preview,
    uncertainty_residual_preview,
)
from app.ui import page_header


CACHE_VERSION = 3


@st.cache_data(ttl=3600, show_spinner=False)
def cached_query_tess_photometry(target: str, cadence: str, cache_version: int) -> MastQueryResult:
    return query_tess_photometry(target, cadence)


def _sync_photometry_target() -> None:
    st.session_state["target_name"] = st.session_state.get("photometry_target_input", "")


def _target_filename_part() -> str:
    target = st.session_state.get("target_name") or st.session_state.get("photometry_target_input") or "target"
    cleaned = "".join(str(target).split())
    return cleaned or "target"


def _target_download_root(result: MastQueryResult) -> Path:
    return Path("data") / "targets" / safe_target_slug(result.resolved_target) / "mast"


def _upload_root() -> Path:
    target = _target_filename_part()
    return Path("data") / "uploads" / safe_target_slug(target) / "photometry"


def _safe_upload_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name).strip("._")
    return cleaned or "uploaded_photometry.dat"


def _persist_uploaded_photometry(files) -> list[str]:
    if not files:
        st.session_state["uploaded_photometry_paths"] = []
        return []

    upload_root = _upload_root()
    upload_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, uploaded in enumerate(files):
        name = _safe_upload_name(uploaded.name)
        destination = upload_root / name
        if destination.exists() and str(destination) in paths:
            destination = upload_root / f"{destination.stem}_{index}{destination.suffix}"
        destination.write_bytes(uploaded.getbuffer())
        paths.append(str(destination))
    st.session_state["uploaded_photometry_paths"] = paths
    return paths


def _display_mast_result(result: MastQueryResult) -> None:
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

    button_col, path_col = st.columns([0.28, 0.72], vertical_alignment="center")
    with button_col:
        if st.button("Download selected products", use_container_width=True, disabled=not selected_files):
            download_root = _target_download_root(result)
            with st.spinner(f"Downloading {len(selected_files)} product(s) to {download_root}..."):
                try:
                    paths = download_tess_products(
                        result.target,
                        st.session_state.get("mast_cadence_filter", "2 min"),
                        selected_files,
                        download_root,
                    )
                except Exception as exc:  # noqa: BLE001 - user-facing network/file error
                    st.session_state["mast_download_error"] = str(exc)
                else:
                    st.session_state["mast_downloaded_paths"] = [str(path) for path in paths]
                    st.session_state.pop("mast_download_error", None)
    with path_col:
        st.text_input("Download folder", value=str(_target_download_root(result)), disabled=True)

    if st.session_state.get("mast_download_error"):
        st.error(st.session_state["mast_download_error"])


def _render_downloaded_files() -> list[str]:
    downloaded_paths = st.session_state.get("mast_downloaded_paths", [])
    if not downloaded_paths:
        return []

    st.subheader("Downloaded sector files")
    files = pd.DataFrame(
        {
            "file": [Path(path).name for path in downloaded_paths],
            "path": downloaded_paths,
            "exists": [Path(path).exists() for path in downloaded_paths],
        }
    )
    st.dataframe(files, use_container_width=True, hide_index=True)
    return [path for path in downloaded_paths if Path(path).exists()]


def _render_uploaded_files(paths: list[str]) -> list[str]:
    existing = [path for path in paths if Path(path).exists()]
    if not existing:
        return []

    st.subheader("Uploaded photometry files")
    files = pd.DataFrame(
        {
            "file": [Path(path).name for path in existing],
            "path": existing,
            "exists": [Path(path).exists() for path in existing],
        }
    )
    st.dataframe(files, use_container_width=True, hide_index=True)
    return existing


def _render_loaded_file_prompt() -> None:
    st.info(
        "Download TESS products from MAST or upload photometry files above to start the sector workflow: "
        "uncertainty review, Wotan flattening, high-side outlier rejection, and stitching."
    )


def _selection_x_range(selection_state) -> tuple[float, float] | None:
    if not selection_state:
        return None
    try:
        points = selection_state.selection.points
    except AttributeError:
        points = selection_state.get("selection", {}).get("points", [])
    xs = []
    for point in points:
        x_value = point.get("x") if isinstance(point, dict) else getattr(point, "x", None)
        if x_value is not None:
            xs.append(float(x_value))
    if len(xs) < 2:
        return None
    return min(xs), max(xs)


def _load_sector_frames(paths: list[str], quality_zero_only: bool) -> dict[str, pd.DataFrame]:
    state_key = (tuple(paths), quality_zero_only)
    if st.session_state.get("sector_frames_key") != state_key:
        sector_frames = load_sector_frames(paths, quality_zero_only=quality_zero_only)
        st.session_state["sector_frames"] = sector_frames
        st.session_state["sector_frames_key"] = state_key
        st.session_state["sector_uncertainties"] = {
            key: value for key, value in st.session_state.get("sector_uncertainties", {}).items() if key in sector_frames
        }
        st.session_state["sector_flattening"] = {
            key: value for key, value in st.session_state.get("sector_flattening", {}).items() if key in sector_frames
        }
        st.session_state.pop("prepared_photometry", None)
        st.session_state.pop("prepared_photometry_summary", None)
    return st.session_state.get("sector_frames", {})


def _style_status_table(table: pd.DataFrame):
    def color_status(value: str) -> str:
        if value == "complete":
            return "background-color: #dcfce7; color: #166534"
        return "background-color: #fee2e2; color: #991b1b"

    return table.style.map(color_status, subset=["status"])


def _render_uncertainty_workflow(paths: list[str], sigma_clip: float, stitch: bool) -> None:
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
    sector_frames = _load_sector_frames(paths, quality_zero_only)
    sector_uncertainties = st.session_state.setdefault("sector_uncertainties", {})
    status = sector_uncertainty_table(sector_frames, sector_uncertainties)
    if not status.empty:
        st.dataframe(_style_status_table(status), use_container_width=True, hide_index=True)

    if not sector_frames:
        return

    sector_key = st.selectbox("Sector to review", list(sector_frames.keys()), key="uncertainty_sector_select")
    frame = normalized_sector(sector_frames[sector_key])
    finite_err = pd.Series(frame["flux_err"]).replace([float("inf"), -float("inf")], pd.NA).dropna()
    finite_err = finite_err[finite_err > 0]
    has_uncertainties = not finite_err.empty

    mode_options = ["Determine uncertainty from a quiet region"]
    if has_uncertainties:
        mode_options.insert(0, "Examine uncertainties from data")
    mode = st.radio("Uncertainty action", mode_options, horizontal=True, key=f"uncertainty_mode_{sector_key}")

    if mode == "Examine uncertainties from data":
        st.plotly_chart(
            sector_uncertainty_preview(frame),
            use_container_width=True,
            config=PLOT_CONFIG,
            key=f"sector_uncertainty_data_{sector_key}",
        )
        median_uncertainty = float(finite_err.median())
        st.metric("Median uncertainty in data", f"{median_uncertainty:.6g}")
        if st.button("Accept data uncertainties for this sector", key=f"accept_data_unc_{sector_key}"):
            sector_uncertainties[sector_key] = median_uncertainty
            st.session_state["sector_uncertainties"] = sector_uncertainties
            st.success(f"Sector uncertainty set to {median_uncertainty:.6g}.")
        return

    window_length = st.slider(
        "Wotan detrending window [days]",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.05,
        key=f"wotan_window_{sector_key}",
    )
    trend = wotan_trend(sector_frames[sector_key], window_length)
    st.caption("Press the selection control in the Plotly toolbar, then drag over a quiet region with typical noise and no obvious structure.")
    selection_state = st.plotly_chart(
        sector_uncertainty_preview(frame, trend=trend),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=f"sector_uncertainty_select_{sector_key}",
        on_select="rerun",
        selection_mode=("box", "lasso"),
    )
    selected_range = _selection_x_range(selection_state)
    if selected_range is None:
        st.info("Select a quiet region of this sector to estimate the uncertainty.")
        return

    region, uncertainty = estimate_uncertainty_for_region(
        sector_frames[sector_key],
        selected_range[0],
        selected_range[1],
        window_length=window_length,
        sigma_clip=5.0,
    )
    st.plotly_chart(
        uncertainty_residual_preview(region),
        use_container_width=True,
        config=PLOT_CONFIG,
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


def _render_flattening_workflow(sigma_clip: float) -> None:
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
    status = flattening_status_table(sector_frames, sector_flattening)
    if not status.empty:
        st.dataframe(_style_status_table(status), use_container_width=True, hide_index=True)

    sector_key = st.selectbox("Sector to flatten", list(sector_frames.keys()), key="flattening_sector_select")
    window_length = st.slider(
        "Wotan flattening window [days]",
        min_value=0.1,
        max_value=2.0,
        value=float(sector_flattening.get(sector_key, {}).get("window_length", 0.8)),
        step=0.05,
        key=f"flatten_window_{sector_key}",
    )
    normalized = normalized_sector(sector_frames[sector_key])
    trend = wotan_trend(sector_frames[sector_key], window_length)
    flattened = flatten_sector(
        sector_frames[sector_key],
        sector_uncertainties[sector_key],
        window_length=window_length,
        sigma_clip=sigma_clip,
    )

    st.plotly_chart(
        sector_uncertainty_preview(normalized, trend=trend),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=f"flattening_trend_{sector_key}",
    )
    st.plotly_chart(
        flattened_sector_preview(flattened),
        use_container_width=True,
        config=PLOT_CONFIG,
        key=f"flattening_preview_{sector_key}",
    )

    high_outliers = int(flattened["is_outlier"].sum())
    kept = len(flattened) - high_outliers
    cols = st.columns(3)
    cols[0].metric("Points", f"{len(flattened):,}")
    cols[1].metric("Kept after flattening", f"{kept:,}")
    cols[2].metric("High outliers", f"{high_outliers:,}")
    if st.button("Accept flattening for this sector", key=f"accept_flattening_{sector_key}"):
        sector_flattening[sector_key] = {
            "window_length": window_length,
            "high_outliers": high_outliers,
        }
        st.session_state["sector_flattening"] = sector_flattening
        st.success(f"Sector flattening accepted with a {window_length:.2f} day Wotan window.")


def _render_stitching_controls(sigma_clip: float, stitch: bool) -> None:
    sector_frames = st.session_state.get("sector_frames", {})
    sector_uncertainties = st.session_state.get("sector_uncertainties", {})
    sector_flattening = st.session_state.get("sector_flattening", {})
    if not sector_frames:
        return

    uncertainties_complete = set(sector_uncertainties) == set(sector_frames)
    flattening_complete = set(sector_flattening) == set(sector_frames)
    st.subheader("3. Stitch sectors")
    if not uncertainties_complete:
        missing = [key for key in sector_frames if key not in sector_uncertainties]
        st.warning(f"Complete uncertainties for all sectors before stitching. Missing uncertainties: {len(missing)}")
        return
    if not flattening_complete:
        missing = [key for key in sector_frames if key not in sector_flattening]
        st.warning(f"Flatten and review all sectors before stitching. Missing flattening: {len(missing)}")
        return
    if st.button("Prepare stitched photometry", type="primary", disabled=not stitch):
        stitched, summary = stitch_flattened_sectors(
            sector_frames,
            sector_uncertainties,
            sector_flattening,
            sigma_clip=sigma_clip,
        )
        st.session_state["prepared_photometry"] = stitched
        st.session_state["prepared_photometry_summary"] = summary
        st.success("Stitched photometry prepared.")


def _render_prepared_photometry() -> None:
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

    st.plotly_chart(
        prepared_photometry_preview(stitched),
        use_container_width=True,
        config=PLOT_CONFIG,
        key="prepared_photometry_preview",
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    export = stitched.loc[~stitched["is_outlier"], ["time", "flux", "flux_err", "source_file", "sector"]].copy()
    st.download_button(
        "Download prepared photometry CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"prepared_photometry_{_target_filename_part()}.csv",
        mime="text/csv",
        key="download_prepared_photometry",
    )


def render() -> None:
    page_header(
        "Import and Prepare Photometry",
        "Collect TESS light curves, inspect quality, normalize flux, and export allesfitter-ready files.",
        actions=["Save Prepared Set"],
    )

    upload_col, settings_col = st.columns([0.58, 0.42], vertical_alignment="top")
    with upload_col:
        files = st.file_uploader(
            "Photometry files",
            type=["csv", "txt", "dat", "fits"],
            accept_multiple_files=True,
            help="Uploaded FITS/CSV/TXT/DAT files enter the same uncertainty, flattening, and stitching workflow as MAST downloads.",
        )
        st.session_state["photometry_files"] = files or []
        uploaded_paths = _persist_uploaded_photometry(files)

    with settings_col:
        st.subheader("MAST target query")
        st.session_state.setdefault("target_name", "")
        if st.session_state.get("photometry_target_input") != st.session_state["target_name"]:
            st.session_state["photometry_target_input"] = st.session_state["target_name"]

        target = st.text_input(
            "Target name or TIC ID",
            placeholder="TIC 288246496 or WASP-100",
            key="photometry_target_input",
            on_change=_sync_photometry_target,
        )
        cadence = st.selectbox(
            "TESS cadence",
            ["2 min", "20 s", "10 min", "30 min", "All available cadences"],
            key="mast_cadence_filter",
        )
        if st.button("Query MAST", use_container_width=True, type="primary"):
            query_target = target.strip()
            if not query_target:
                st.warning("Enter a target name or TIC ID first, for example `TIC 288246496`.")
            else:
                st.session_state["target_name"] = query_target
                with st.spinner(f"Querying MAST for {query_target}..."):
                    try:
                        result = cached_query_tess_photometry(query_target, cadence, CACHE_VERSION)
                    except Exception as exc:  # noqa: BLE001 - show user-facing query failures
                        st.session_state["mast_error"] = str(exc)
                        st.session_state.pop("mast_result", None)
                    else:
                        st.session_state["mast_result"] = result
                        st.session_state["mast_resolved_target"] = result.resolved_target
                        st.session_state.pop("mast_error", None)

        if st.session_state.get("mast_error"):
            st.error(st.session_state["mast_error"])

        st.divider()
        st.subheader("Preparation controls")
        st.selectbox("Time column", ["Auto detect", "time", "btjd", "bjd_tdb"])
        st.selectbox("Flux column", ["Auto detect", "flux", "pdcsap_flux", "sap_flux", "kspsap_flux"])
        sigma_clip = st.slider("Outlier sigma clipping", 2.0, 10.0, 5.0, 0.5)
        stitch = True

    result = st.session_state.get("mast_result")
    if result is not None:
        st.subheader("MAST query results")
        _display_mast_result(result)

    downloaded_paths = _render_downloaded_files()
    uploaded_paths = _render_uploaded_files(st.session_state.get("uploaded_photometry_paths", uploaded_paths))
    workflow_paths = list(dict.fromkeys(downloaded_paths + uploaded_paths))
    if not workflow_paths:
        _render_loaded_file_prompt()

    _render_uncertainty_workflow(workflow_paths, sigma_clip, stitch)
    _render_flattening_workflow(sigma_clip)
    _render_stitching_controls(sigma_clip, stitch)
    _render_prepared_photometry()
