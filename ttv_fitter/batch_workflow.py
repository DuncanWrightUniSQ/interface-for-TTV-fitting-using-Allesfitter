"""Automatic multi-target workflow for the simplified TTV fitter."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
import re
import shutil
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.mast import MastQueryResult, download_tess_products, safe_target_slug
from app.photometry import normalized_sector, read_photometry_file
from ttv_fitter.alles_workflows import (
    _available_wotan_method,
    _clean_exofop_planet_label,
    _duration_hours_from_exofop_row,
    _transit_mask_for_ephemerides,
    _wotan_trend_with_method,
)
from ttv_fitter.fitting import (
    build_cutouts,
    fit_cutout_t0,
    fit_limb_darkened_single_transit,
    run_cutout_t0_mcmc,
)
from ttv_fitter.models import derive_a_over_rstar
from ttv_fitter.plots import cutout_fit_figure, oc_figure, t0_histogram_figure
from ttv_fitter.ttv import oc_table


Progress = Callable[[str], None] | None


def parse_target_list(text: str) -> list[str]:
    """Parse one target per line, ignoring blank lines and comments."""
    targets: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").splitlines():
        target = raw.strip()
        if not target or target.startswith("#"):
            continue
        if target not in seen:
            targets.append(target)
            seen.add(target)
    return targets


def filter_product_filenames(products: pd.DataFrame) -> list[str]:
    """Return every product except joined Diamante light curves."""
    if not isinstance(products, pd.DataFrame) or products.empty:
        return []
    filename_column = next((column for column in ("productFilename", "product_filename", "filename") if column in products), None)
    if filename_column is None:
        return []
    names = products[filename_column].dropna().astype(str)
    return sorted({name for name in names if "diamante" not in name.lower()})


def _finite_error(frame: pd.DataFrame) -> bool:
    if "flux_err" not in frame:
        return False
    values = pd.to_numeric(frame["flux_err"], errors="coerce")
    return bool((np.isfinite(values) & (values > 0)).any())


def estimate_residual_uncertainty(
    frame: pd.DataFrame,
    duration_hours: float,
    *,
    cval: float = 3.5,
    sigma_clip: float = 4.0,
    transit_mask: np.ndarray | None = None,
) -> float:
    """Estimate one sector uncertainty from detrended residual scatter."""
    normalized = normalized_sector(frame)
    window_days = max(float(duration_hours) / 24.0, 1e-4)
    trend = _wotan_trend_with_method(
        type("PhotometryAdapter", (), {"normalized_sector": staticmethod(normalized_sector)})(),
        frame,
        window_days,
        _available_wotan_method("biweight")[0],
        mask=transit_mask,
        cval=float(cval),
    )
    residual = normalized["flux"].to_numpy(dtype=float) - np.asarray(trend, dtype=float)
    residual = residual[np.isfinite(residual)]
    if residual.size < 2:
        return float("nan")
    center = float(np.nanmedian(residual))
    scatter = float(np.nanstd(residual, ddof=1))
    if not np.isfinite(scatter) or scatter <= 0:
        return scatter
    # This clipped array exists only inside the scalar uncertainty estimate;
    # it is never returned or used as fitting photometry.  Reject only
    # high-side residuals, preserving downward transit-like excursions.
    kept = residual[(residual - center) <= float(sigma_clip) * scatter]
    return float(np.nanstd(kept, ddof=1)) if kept.size > 1 else scatter


def _trend_and_flatten(
    frame: pd.DataFrame,
    uncertainty: float,
    duration_hours: float,
    *,
    cval: float = 3.5,
    sigma_clip: float = 4.0,
    transit_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    normalized = normalized_sector(frame)
    window_days = max(float(duration_hours) / 24.0, 1e-4)
    adapter = type("PhotometryAdapter", (), {"normalized_sector": staticmethod(normalized_sector)})()
    trend = _wotan_trend_with_method(adapter, frame, window_days, "biweight", mask=transit_mask, cval=float(cval))
    trend = np.asarray(trend, dtype=float)
    safe_trend = np.where(np.isfinite(trend) & (trend != 0), trend, np.nan)
    prepared = normalized.copy()
    prepared["trend"] = trend
    prepared["flux_before_flatten"] = prepared["flux"]
    prepared["flux"] = prepared["flux_before_flatten"] / safe_trend
    original_err = pd.to_numeric(frame.get("flux_err", pd.Series(np.nan, index=frame.index)), errors="coerce")
    if _finite_error(frame):
        median_raw = float(np.nanmedian(pd.to_numeric(frame["flux"], errors="coerce")))
        pointwise = original_err.to_numpy(dtype=float) / median_raw if np.isfinite(median_raw) and median_raw != 0 else original_err.to_numpy(dtype=float)
        prepared["flux_err"] = pointwise / safe_trend
        adopted_uncertainty = float(np.nanmedian(prepared["flux_err"]))
    else:
        prepared["flux_err"] = float(uncertainty) / safe_trend
        adopted_uncertainty = float(uncertainty)
    residual = prepared["flux_before_flatten"].to_numpy(dtype=float) - safe_trend
    scatter = adopted_uncertainty if np.isfinite(adopted_uncertainty) and adopted_uncertainty > 0 else np.nanstd(residual, ddof=1)
    # Do not discard transit dips: they are intentionally negative residuals
    # and are the signal being fitted.  Match simplified-mode handling by
    # clipping only high-side excursions, and never points in the transit mask.
    if np.isfinite(scatter) and scatter > 0:
        high_side = residual - np.nanmedian(residual) > float(sigma_clip) * scatter
        prepared["is_outlier"] = high_side & (~np.asarray(transit_mask, dtype=bool) if transit_mask is not None else True)
    else:
        prepared["is_outlier"] = False
    prepared["wotan_method"] = "biweight"
    prepared["wotan_cval"] = float(cval)
    prepared["wotan_window_days"] = window_days
    prepared["adopted_uncertainty"] = adopted_uncertainty
    return prepared


def prepare_sector_frames(
    frames: dict[str, pd.DataFrame],
    duration_hours: float,
    *,
    cval: float = 3.5,
    sigma_clip: float = 4.0,
    ephemeris: dict[str, float] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prepared: dict[str, pd.DataFrame] = {}
    summary: list[dict[str, object]] = []
    for key, frame in frames.items():
        transit_mask = None
        if ephemeris:
            transit_mask = _transit_mask_for_ephemerides(frame, [ephemeris], width_durations=2.0)
        uncertainty = estimate_residual_uncertainty(frame, duration_hours, cval=cval, sigma_clip=sigma_clip, transit_mask=transit_mask)
        if _finite_error(frame):
            uncertainty = float(np.nanmedian(pd.to_numeric(frame["flux_err"], errors="coerce")))
        # The one-duration trend above is only for estimating missing noise.
        # Preserve the actual transit signal with the normal simplified-mode
        # detrend: eight transit durations.
        flatten_duration_hours = float(duration_hours) * 8.0
        output = _trend_and_flatten(
            frame,
            uncertainty,
            flatten_duration_hours,
            cval=cval,
            sigma_clip=sigma_clip,
            transit_mask=transit_mask,
        )
        output["uncertainty_window_days"] = max(float(duration_hours) / 24.0, 1e-4)
        output["final_detrend_window_days"] = max(float(flatten_duration_hours) / 24.0, 1e-4)
        prepared[key] = output.sort_values("time").reset_index(drop=True)
        summary.append({
            "sector_key": key,
            "sector": str(output["sector"].iloc[0]) if "sector" in output else key,
            "source_file": str(output["source_file"].iloc[0]) if "source_file" in output else "",
            "points": len(output),
            "kept_points": int((~output["is_outlier"]).sum()),
            "high_outliers": int(output["is_outlier"].sum()),
            "uncertainty_source": "data" if _finite_error(frame) else "detrended residual std",
            "adopted_uncertainty": float(np.nanmedian(output["flux_err"])),
            "uncertainty_window_days": max(float(duration_hours) / 24.0, 1e-4),
            "wotan_window_days": max(float(duration_hours) * 8.0 / 24.0, 1e-4),
            "final_detrend_window_days": max(float(duration_hours) * 8.0 / 24.0, 1e-4),
            "wotan_cval": float(cval),
            "sigma_clip": float(sigma_clip),
        })
    return prepared, pd.DataFrame(summary)


def _align_t0(t0: float, photometry: pd.DataFrame) -> float:
    median = float(pd.to_numeric(photometry["time"], errors="coerce").median())
    if median < 100000 and t0 > 2400000:
        return t0 - 2457000.0
    if median > 2400000 and t0 < 100000:
        return t0 + 2457000.0
    return t0


def normalize_time_to_btjd(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize common JD/BJD time columns to the TESS BTJD scale."""
    out = frame.copy()
    values = pd.to_numeric(out.get("time", pd.Series(dtype=float)), errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return out, "unchanged"
    median = float(finite.median())
    if median > 2_400_000:
        out["time"] = values - 2_457_000.0
        return out, "BJD/JD → BTJD"
    return out, "BTJD"


def anchor_epoch_to_data(t0: float, period: float, photometry: pd.DataFrame) -> float:
    """Fold a reference epoch by integer periods so it lies near the data."""
    times = pd.to_numeric(photometry.get("time", pd.Series(dtype=float)), errors="coerce")
    finite = times[np.isfinite(times)]
    if finite.empty or not np.isfinite(t0) or not np.isfinite(period) or period <= 0:
        return float(t0)
    midpoint = float(finite.median())
    cycles = int(np.round((midpoint - float(t0)) / float(period)))
    return float(t0 + cycles * period)


def planet_b_seed(matches: pd.DataFrame, photometry: pd.DataFrame) -> dict[str, float]:
    if not isinstance(matches, pd.DataFrame) or matches.empty:
        raise ValueError("No ExoFOP planet parameters were found.")
    row = None
    for _, candidate in matches.iterrows():
        if _clean_exofop_planet_label(candidate, 0).lower() == "b":
            row = candidate
            break
    if row is None:
        row = matches.iloc[0]
    period = float(pd.to_numeric(pd.Series([row.get("Period (days)")]), errors="coerce").iloc[0])
    t0 = float(pd.to_numeric(pd.Series([row.get("Epoch (BJD)")]), errors="coerce").iloc[0])
    duration_hours = float(_duration_hours_from_exofop_row(row))
    if not np.isfinite(period) or period <= 0 or not np.isfinite(t0) or not np.isfinite(duration_hours):
        raise ValueError("ExoFOP planet b parameters do not include a usable epoch, period, and duration.")
    stellar_mass = float(pd.to_numeric(pd.Series([row.get("Stellar Mass (M_Sun)")]), errors="coerce").iloc[0])
    stellar_radius = float(pd.to_numeric(pd.Series([row.get("Stellar Radius (R_Sun)")]), errors="coerce").iloc[0])
    radius = float(pd.to_numeric(pd.Series([row.get("Planet Radius (R_Earth)")]), errors="coerce").iloc[0])
    rr = radius * 0.0091577 / stellar_radius if np.isfinite(radius) and np.isfinite(stellar_radius) and stellar_radius > 0 else np.nan
    if not np.isfinite(rr):
        depth = float(pd.to_numeric(pd.Series([row.get("Depth (ppm)")]), errors="coerce").iloc[0])
        rr = math.sqrt(max(depth, 0.0) / 1_000_000.0) if np.isfinite(depth) else 0.05
    a_over_rstar = derive_a_over_rstar(period, stellar_mass, stellar_radius) if np.isfinite(stellar_mass) and np.isfinite(stellar_radius) else 10.0
    return {
        "t0": _align_t0(t0, photometry), "period": period, "radius_ratio": float(np.clip(rr, 1e-5, 1.0)),
        "impact": 0.5, "duration_hours": duration_hours, "a_over_rstar": float(a_over_rstar),
        "limb_darkening_u1": 0.5, "limb_darkening_u2": 0.1, "baseline_offset": 0.0,
    }


def select_reference_cutout(photometry: pd.DataFrame, seed: dict[str, float]) -> tuple[dict[str, float], pd.DataFrame]:
    half_width = max(float(seed["duration_hours"]) / 24.0 * 2.5, 0.15)
    cutouts = build_cutouts(photometry, seed["t0"], seed["period"], half_width)
    if cutouts.empty:
        raise ValueError("No predicted planet b transit cutouts were found in the prepared data.")
    maximum = int(cutouts["points"].max())
    candidates = cutouts.loc[cutouts["points"] >= 0.9 * maximum].sort_values("expected_tmid")
    if candidates.empty:
        raise ValueError("No transit cutout met the 10% of maximum point-count criterion.")
    selected = candidates.iloc[0].to_dict()
    mask = (photometry["time"] >= selected["start"]) & (photometry["time"] <= selected["end"])
    return {"epoch": int(selected["epoch"]), "expected_tmid": float(selected["expected_tmid"]), "start": float(selected["start"]), "end": float(selected["end"]), "points": int(selected["points"])}, photometry.loc[mask].copy()


def run_timing_fits(photometry: pd.DataFrame, seed: dict[str, float], reference: dict[str, float], progress: Progress = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    ref_t0 = float(seed["t0"])
    relative_epoch = int(reference["epoch"])
    timing_t0 = float(seed["t0"])
    cutouts = build_cutouts(photometry, timing_t0, seed["period"], max(float(seed["duration_hours"]) / 24.0 * 4.0, 0.125))
    suitable = cutouts.loc[cutouts["points"] >= 20]
    ls_details: list[dict[str, object]] = []
    for item in suitable.to_dict("records"):
        epoch = int(item["epoch"]) + relative_epoch
        if progress:
            progress(f"least-squares transit {epoch}")
        mask = (photometry["time"] >= item["start"]) & (photometry["time"] <= item["end"])
        result = fit_cutout_t0(photometry.loc[mask], seed["period"], float(item["expected_tmid"]), seed["radius_ratio"], seed["impact"], seed["a_over_rstar"], seed["duration_hours"], seed["limb_darkening_u1"], seed["limb_darkening_u2"], seed["baseline_offset"], max(float(seed["duration_hours"]) / 24.0, 0.05), use_t0_multistart=True)
        if result.success:
            ls_details.append({"epoch": epoch, "tmid": result.params["tmid"], "expected_tmid": item["expected_tmid"], "start": item["start"], "end": item["end"], "points": int(item["points"]), **seed})
    timing_rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    samples: dict[int, pd.DataFrame] = {}
    for fit in ls_details:
        epoch = int(fit["epoch"])
        if progress:
            progress(f"MCMC transit {epoch}")
        mask = (photometry["time"] >= fit["start"]) & (photometry["time"] <= fit["end"])
        result = run_cutout_t0_mcmc(photometry.loc[mask], seed["period"], fit["tmid"], seed["radius_ratio"], seed["impact"], seed["a_over_rstar"], seed["duration_hours"], seed["limb_darkening_u1"], seed["limb_darkening_u2"], seed["baseline_offset"], search_half_width_days=max(float(seed["duration_hours"]) / 24.0, 0.05), nwalkers=6, nsteps=1000, burn=500, trim_nonconverged_walkers=True)
        if result.success:
            detail = dict(fit)
            detail.update(result.params)
            details.append(detail)
            timing_rows.append({"planet": "b", "epoch": epoch, "tmid": result.params["tmid"], "tmid_err": result.params["tmid_err"], "tmid_err_minus": result.params["tmid_err_minus"], "tmid_err_plus": result.params["tmid_err_plus"], "expected_tmid": fit["expected_tmid"], "points": fit["points"]})
            samples[epoch] = result.samples
    return pd.DataFrame(timing_rows), pd.DataFrame(details), samples


def _png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _fit_png_bytes(photometry: pd.DataFrame, fit: dict[str, object]) -> bytes:
    window = photometry.loc[(photometry["time"] >= float(fit["start"])) & (photometry["time"] <= float(fit["end"]))]
    time = np.linspace(float(fit["start"]), float(fit["end"]), 800)
    from ttv_fitter.models import limb_darkened_transit_model

    model = limb_darkened_transit_model(time, float(fit["period"]), float(fit["tmid"]), float(fit["radius_ratio"]), float(fit["impact"]), float(fit["duration_hours"]), float(fit["limb_darkening_u1"]), float(fit["limb_darkening_u2"]), baseline_offset=float(fit.get("baseline_offset", 0.0)), a_over_rstar=float(fit.get("a_over_rstar", np.nan)))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.errorbar(window["time"], window["flux"], yerr=window.get("flux_err"), fmt=".", color="#334155", alpha=0.45, ms=3)
    ax.plot(time, model, color="#dc2626", lw=2.4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.set_title(f"Transit {int(fit.get('epoch', 0))} T0 fit")
    fig.tight_layout()
    return _png_bytes(fig)


def _hist_png_bytes(samples: pd.DataFrame, tmid: float, err_minus: float, err_plus: float) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    if not samples.empty and "tmid" in samples:
        ax.hist(samples["tmid"], bins=40, color="#2563eb", alpha=0.72)
    ax.axvline(tmid, color="#dc2626", lw=2.4, label="median")
    ax.axvspan(tmid - err_minus, tmid + err_plus, color="#dc2626", alpha=0.14)
    ax.set_xlabel("T0")
    ax.set_ylabel("Samples")
    ax.legend(loc="best")
    fig.tight_layout()
    return _png_bytes(fig)


def _oc_png_bytes(timings: pd.DataFrame, t0: float, period: float) -> bytes:
    table = oc_table(timings, t0, period)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    if not table.empty:
        yerr = table["tmid_err"] * 1440.0 if "tmid_err" in table else None
        ax.errorbar(table["epoch"], table["oc_minutes"], yerr=yerr, fmt="o", color="#1d4ed8")
    ax.axhline(0, color="#94a3b8", lw=1)
    ax.set_xlabel("Transit epoch")
    ax.set_ylabel("O-C [minutes]")
    fig.tight_layout()
    return _png_bytes(fig)


def save_target_outputs(target: str, prepared: dict[str, pd.DataFrame], summary: pd.DataFrame, seed: dict[str, float], reference: dict[str, float], reference_fit: dict[str, object], timings: pd.DataFrame, details: pd.DataFrame, samples: dict[int, pd.DataFrame]) -> tuple[Path, Path]:
    base = Path("data") / "prepared" / safe_target_slug(target)
    plots = base / "plots"
    results = base / "results"
    plots.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    (base / "sectors").mkdir(parents=True, exist_ok=True)
    summary.to_csv(results / "sector_summary.csv", index=False)
    for key, frame in prepared.items():
        sector = safe_target_slug(str(frame["sector"].iloc[0]) if "sector" in frame else str(key))
        frame.loc[~frame["is_outlier"].astype(bool)].to_csv(base / "sectors" / f"{safe_target_slug(target)}_sector_{sector}.csv", index=False)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=140, sharex=True)
        axes[0].plot(frame["time"], frame["flux_before_flatten"], ".", ms=1.5, alpha=0.3, color="#334155")
        axes[0].plot(frame["time"], frame["trend"], color="#dc2626", lw=1.2)
        axes[0].set_ylabel("Raw / trend")
        axes[1].plot(frame["time"], frame["flux"], ".", ms=1.5, alpha=0.3, color="#2563eb")
        axes[1].set_ylabel("Flattened flux")
        axes[1].set_xlabel("Time")
        fig.tight_layout()
        fig.savefig(plots / f"{safe_target_slug(target)}_sector_{sector}_flattened.png")
        plt.close(fig)
        trend_fig = go.Figure()
        trend_fig.add_scatter(x=frame["time"], y=frame["flux_before_flatten"], mode="markers", name="flux", marker={"size": 3, "opacity": 0.35})
        trend_fig.add_scatter(x=frame["time"], y=frame["trend"], mode="lines", name="Wotan trend", line={"color": "#dc2626"})
        trend_fig.write_html(plots / f"{safe_target_slug(target)}_sector_{sector}_wotan_trend.html", include_plotlyjs="cdn")
        flattened_fig = go.Figure()
        flattened_fig.add_scatter(x=frame["time"], y=frame["flux"], mode="markers", name="flattened flux", marker={"size": 3, "opacity": 0.35})
        flattened_fig.write_html(plots / f"{safe_target_slug(target)}_sector_{sector}_flattened.html", include_plotlyjs="cdn")
    combined = pd.concat(prepared.values(), ignore_index=True).sort_values("time").reset_index(drop=True)
    reference_fit = dict(reference_fit)
    reference_fit["start"] = reference["start"]
    reference_fit["end"] = reference["end"]
    reference_fit["epoch"] = reference["epoch"]
    fit_fig = cutout_fit_figure(combined, reference_fit)
    fit_fig.write_html(plots / f"{safe_target_slug(target)}_single_transit_fit.html", include_plotlyjs="cdn")
    (plots / f"{safe_target_slug(target)}_single_transit_fit.png").write_bytes(_fit_png_bytes(combined, reference_fit))
    pd.DataFrame([reference_fit]).to_csv(results / f"{safe_target_slug(target)}_single_transit_fit.csv", index=False)
    if not details.empty:
        for detail in details.to_dict("records"):
            epoch = int(detail["epoch"])
            stem = f"{safe_target_slug(target)}_b_Tr{epoch}"
            fit_fig = cutout_fit_figure(pd.concat(prepared.values(), ignore_index=True), detail)
            fit_fig.write_html(plots / f"{stem}_mcmc_fit.html", include_plotlyjs="cdn")
            (plots / f"{stem}_mcmc_fit.png").write_bytes(_fit_png_bytes(combined, detail))
            samples_frame = samples.get(epoch, pd.DataFrame())
            t0_histogram_figure(samples_frame, float(detail["tmid"]), float(detail["tmid_err_minus"]), float(detail["tmid_err_plus"])).write_html(plots / f"{stem}_t0_distribution.html", include_plotlyjs="cdn")
            (plots / f"{stem}_t0_distribution.png").write_bytes(_hist_png_bytes(samples_frame, float(detail["tmid"]), float(detail["tmid_err_minus"]), float(detail["tmid_err_plus"])))
        reference_t0 = float(seed["t0"]) - int(reference["epoch"]) * float(seed["period"])
        oc_figure(timings, reference_t0, float(seed["period"])).write_html(plots / f"{safe_target_slug(target)}_b_OC_MCMC.html", include_plotlyjs="cdn")
        (plots / f"{safe_target_slug(target)}_b_OC_MCMC.png").write_bytes(_oc_png_bytes(timings, reference_t0, float(seed["period"])))
        shutil.copy2(plots / f"{safe_target_slug(target)}_b_OC_MCMC.html", results / f"{safe_target_slug(target)}_b_OC_MCMC.html")
        shutil.copy2(plots / f"{safe_target_slug(target)}_b_OC_MCMC.png", results / f"{safe_target_slug(target)}_b_OC_MCMC.png")
        timings.to_csv(results / f"{safe_target_slug(target)}_b_mcmc_timings.csv", index=False)
        oc_table(timings, reference_t0, float(seed["period"])).to_csv(results / f"{safe_target_slug(target)}_b_oc_table.csv", index=False)
    return plots, results


def run_target_batch(target: str, photometry_import_module, photometry_fit_module, progress: Progress = None) -> dict[str, object]:
    """Download, prepare, fit, and save one target."""
    output_base = Path("data") / "prepared" / safe_target_slug(target)
    if output_base.exists():
        shutil.rmtree(output_base)
    if progress:
        progress("starting; previous prepared outputs will be replaced")
    result: MastQueryResult = photometry_import_module.cached_query_tess_photometry(
        target,
        "All available cadences",
        int(getattr(photometry_import_module, "CACHE_VERSION", 1)),
    )
    product_names = filter_product_filenames(result.products)
    if not product_names and hasattr(photometry_import_module, "query_tess_photometry"):
        if progress:
            progress("MAST cache returned no products; retrying the live product query")
        live_result = photometry_import_module.query_tess_photometry(target, "All available cadences")
        if live_result.products is not None and not live_result.products.empty:
            result = live_result
            product_names = filter_product_filenames(result.products)
    if not product_names:
        raise ValueError("MAST returned no usable non-Diamante products.")
    if progress:
        progress(f"MAST query complete; downloading {len(product_names)} products")
    root = Path("data") / "targets" / safe_target_slug(result.resolved_target) / "mast"
    root.mkdir(parents=True, exist_ok=True)
    existing = {path.name: path for path in root.iterdir() if path.is_file()}
    missing = [name for name in product_names if name not in existing]
    downloaded = download_tess_products(target, "All available cadences", missing, root) if missing else []
    paths = [existing[name] for name in product_names if name in existing]
    paths.extend(Path(path) for path in downloaded)
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        try:
            frame = read_photometry_file(path, quality_zero_only=True)
        except Exception:
            continue
        if not frame.empty:
            normalized, _scale = normalize_time_to_btjd(frame)
            frames[f"S{normalized['sector'].iloc[0]} | {path.name}"] = normalized
    if not frames:
        raise ValueError("Downloaded products contained no readable photometry tables.")
    if progress:
        progress(f"download complete; loaded {len(frames)} readable sector product(s)")
    tic = photometry_fit_module.extract_tic_id(result.resolved_target)
    if not tic:
        raise ValueError(f"Could not resolve `{target}` to a TIC ID for ExoFOP.")
    toi_table, _source = photometry_fit_module.load_toi_table(use_cached=True)
    matches = photometry_fit_module.find_toi_parameters(toi_table, tic)
    combined_raw = pd.concat(frames.values(), ignore_index=True).sort_values("time").reset_index(drop=True)
    seed = planet_b_seed(matches, combined_raw)
    seed["t0"] = anchor_epoch_to_data(seed["t0"], seed["period"], combined_raw)
    if progress:
        progress(f"ExoFOP planet b parameters loaded for TIC {tic}")
    prepared, summary = prepare_sector_frames(
        frames,
        seed["duration_hours"],
        cval=3.5,
        sigma_clip=4.0,
        ephemeris=seed,
    )
    if progress:
        progress(
            "uncertainties accepted or estimated with 1x-duration masked trend; "
            "final sectors detrended with 8x-duration window"
        )
    combined = pd.concat([frame.loc[~frame["is_outlier"].astype(bool)] for frame in prepared.values()], ignore_index=True).sort_values("time").reset_index(drop=True)
    reference, reference_cutout = select_reference_cutout(combined, seed)
    if progress:
        progress(f"selected reference transit {int(reference['epoch'])} ({int(reference['points'])} points)")
    fit_params = ["t0", "radius_ratio", "impact", "a_over_rstar", "limb_darkening_u1", "limb_darkening_u2", "baseline_offset"]
    # Match the simplified UI: initialize the one-transit refinement at the
    # selected cutout's predicted midpoint, rather than at the global epoch
    # anchor (which may be several periods away from this particular cutout).
    reference_seed = {**seed, "t0": float(reference["expected_tmid"])}
    fit = fit_limb_darkened_single_transit(reference_cutout, reference_seed, fit_params)
    if not fit.success:
        raise ValueError(f"Reference transit fit did not converge: {fit.message}")
    fitted_seed = {**seed, **fit.params}
    fitted_seed["t0"] = float(fit.params["t0"])
    if progress:
        progress("single-transit least-squares refinement complete; starting timing fits")
    timings, details, samples = run_timing_fits(combined, fitted_seed, reference, progress)
    reference_fit = {**fitted_seed, "tmid": float(fitted_seed["t0"])}
    plots, results = save_target_outputs(target, prepared, summary, fitted_seed, reference, reference_fit, timings, details, samples)
    return {"target": target, "resolved_target": result.resolved_target, "products": len(product_names), "sectors": len(prepared), "points": len(combined), "reference_epoch": reference["epoch"], "reference_points": reference["points"], "timings": len(timings), "plots": str(plots), "results": str(results)}
