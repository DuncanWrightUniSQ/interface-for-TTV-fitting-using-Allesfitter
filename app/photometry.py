"""Photometry loading and first-pass preparation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from wotan import flatten


TIME_COLUMNS = ["TIME", "BTJD", "BJD_TDB", "BJD", "JD"]
FLUX_COLUMNS = [
    "PDCSAP_FLUX",
    "KSPSAP_FLUX",
    "SAP_FLUX",
    "CORR_FLUX",
    "PCA_FLUX",
    "RAW_FLUX",
    "FLUX",
    "RELATIVE_FLUX",
]
ERR_COLUMNS = ["PDCSAP_FLUX_ERR", "KSPSAP_FLUX_ERR", "SAP_FLUX_ERR", "FLUX_ERR", "ERR", "ERROR", "UNCERTAINTY"]


def _first_existing(names: list[str], available: list[str]) -> str | None:
    upper_map = {name.upper(): name for name in available}
    for name in names:
        if name in upper_map:
            return upper_map[name]
    return None


def _robust_scatter(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return float("nan")
    median = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - median))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(values[finite]))


def read_tess_fits(path: str | Path, *, quality_zero_only: bool = True) -> pd.DataFrame:
    """Read a common TESS/SPOC/QLP light-curve FITS product into a DataFrame."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        data_hdu = None
        for hdu in hdul[1:]:
            if hasattr(hdu, "columns") and "TIME" in [name.upper() for name in hdu.columns.names]:
                data_hdu = hdu
                break
        if data_hdu is None:
            raise ValueError(f"No TIME table found in {path.name}")

        names = list(data_hdu.columns.names)
        time_col = _first_existing(["TIME"], names)
        flux_col = _first_existing(FLUX_COLUMNS, names)
        err_col = _first_existing(ERR_COLUMNS, names)
        quality_col = _first_existing(["QUALITY"], names)

        if time_col is None or flux_col is None:
            raise ValueError(f"Could not identify TIME/flux columns in {path.name}")

        table = data_hdu.data
        time = np.asarray(table[time_col], dtype=float)
        flux = np.asarray(table[flux_col], dtype=float)
        if err_col:
            flux_err = np.asarray(table[err_col], dtype=float)
        else:
            flux_err = np.full_like(flux, np.nan, dtype=float)

        mask = np.isfinite(time) & np.isfinite(flux)
        if quality_zero_only and quality_col:
            quality = np.asarray(table[quality_col])
            mask &= quality == 0

        header = data_hdu.header
        primary_header = hdul[0].header
        sector = header.get("SECTOR", primary_header.get("SECTOR", np.nan))
        camera = header.get("CAMERA", primary_header.get("CAMERA", np.nan))
        ccd = header.get("CCD", primary_header.get("CCD", np.nan))

    frame = pd.DataFrame(
        {
            "time": time[mask],
            "flux": flux[mask],
            "flux_err": flux_err[mask],
            "source_file": path.name,
            "source_path": str(path),
            "sector": sector,
            "camera": camera,
            "ccd": ccd,
            "flux_column": flux_col,
            "err_column": err_col or "",
        }
    )
    return frame


def read_photometry_table(path: str | Path, *, quality_zero_only: bool = True) -> pd.DataFrame:
    """Read a generic CSV/TXT/DAT photometry table into the same schema as TESS FITS files."""
    path = Path(path)
    try:
        table = pd.read_csv(path, sep=None, engine="python", comment="#")
    except Exception:
        table = pd.read_csv(path, delim_whitespace=True, comment="#")
    if table.empty:
        raise ValueError(f"No rows found in {path.name}")

    names = list(table.columns)
    time_col = _first_existing(TIME_COLUMNS, names)
    flux_col = _first_existing(FLUX_COLUMNS, names)
    err_col = _first_existing(ERR_COLUMNS, names)
    quality_col = _first_existing(["QUALITY"], names)
    sector_col = _first_existing(["SECTOR"], names)

    if time_col is None or flux_col is None:
        raise ValueError(f"Could not identify time/flux columns in {path.name}")

    time = pd.to_numeric(table[time_col], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(table[flux_col], errors="coerce").to_numpy(dtype=float)
    if err_col:
        flux_err = pd.to_numeric(table[err_col], errors="coerce").to_numpy(dtype=float)
    else:
        flux_err = np.full_like(flux, np.nan, dtype=float)

    mask = np.isfinite(time) & np.isfinite(flux)
    if quality_zero_only and quality_col:
        quality = pd.to_numeric(table[quality_col], errors="coerce").to_numpy()
        mask &= quality == 0

    if sector_col:
        sectors = table.loc[mask, sector_col].to_numpy()
        sector = sectors[0] if len(sectors) else path.stem
    else:
        sector = path.stem

    return pd.DataFrame(
        {
            "time": time[mask],
            "flux": flux[mask],
            "flux_err": flux_err[mask],
            "source_file": path.name,
            "source_path": str(path),
            "sector": sector,
            "camera": np.nan,
            "ccd": np.nan,
            "flux_column": flux_col,
            "err_column": err_col or "",
        }
    )


def read_photometry_file(path: str | Path, *, quality_zero_only: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".fits", ".fit", ".fts"}:
        return read_tess_fits(path, quality_zero_only=quality_zero_only)
    return read_photometry_table(path, quality_zero_only=quality_zero_only)


def sector_key(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unknown"
    sector = frame["sector"].iloc[0]
    source_file = frame["source_file"].iloc[0]
    return f"S{sector} | {source_file}"


def load_sector_frames(paths: list[str | Path], *, quality_zero_only: bool = True) -> dict[str, pd.DataFrame]:
    sectors: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = read_photometry_file(path, quality_zero_only=quality_zero_only)
        sectors[sector_key(frame)] = frame
    return sectors


def sector_uncertainty_table(
    sector_frames: dict[str, pd.DataFrame],
    sector_uncertainties: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for key, frame in sector_frames.items():
        finite_err = np.isfinite(frame["flux_err"]) & (frame["flux_err"] > 0)
        rows.append(
            {
                "status": "complete" if key in sector_uncertainties else "incomplete",
                "sector": frame["sector"].iloc[0] if not frame.empty else "",
                "source_file": frame["source_file"].iloc[0] if not frame.empty else "",
                "points": len(frame),
                "data_has_uncertainties": bool(finite_err.any()),
                "median_data_uncertainty": float(np.nanmedian(frame.loc[finite_err, "flux_err"])) if finite_err.any() else np.nan,
                "adopted_uncertainty": sector_uncertainties.get(key, np.nan),
                "flux_column": frame["flux_column"].iloc[0] if not frame.empty else "",
                "err_column": frame["err_column"].iloc[0] if not frame.empty else "",
            }
        )
    return pd.DataFrame(rows)


def normalized_sector(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    median_flux = np.nanmedian(out["flux"])
    if np.isfinite(median_flux) and median_flux != 0:
        out["flux_raw"] = out["flux"]
        out["flux"] = out["flux"] / median_flux
        out["flux_err"] = out["flux_err"] / median_flux
    return out


def wotan_trend(frame: pd.DataFrame, window_length: float) -> np.ndarray:
    normalized = normalized_sector(frame)
    time = normalized["time"].to_numpy(dtype=float)
    flux = normalized["flux"].to_numpy(dtype=float)
    _, trend = flatten(
        time,
        flux,
        window_length=window_length,
        method="biweight",
        return_trend=True,
    )
    return np.asarray(trend, dtype=float)


def estimate_uncertainty_for_region(
    frame: pd.DataFrame,
    x_min: float,
    x_max: float,
    *,
    window_length: float,
    sigma_clip: float = 5.0,
) -> tuple[pd.DataFrame, float]:
    normalized = normalized_sector(frame)
    trend = wotan_trend(frame, window_length)
    normalized["trend"] = trend
    normalized["residual"] = normalized["flux"] - normalized["trend"]

    region = normalized[(normalized["time"] >= x_min) & (normalized["time"] <= x_max)].copy()
    if region.empty:
        return region, float("nan")

    center = np.nanmedian(region["residual"])
    scatter = _robust_scatter(region["residual"].to_numpy())
    if np.isfinite(scatter) and scatter > 0:
        region["uncertainty_outlier"] = np.abs(region["residual"] - center) > sigma_clip * scatter
    else:
        region["uncertainty_outlier"] = False

    kept = region.loc[~region["uncertainty_outlier"], "residual"].to_numpy(dtype=float)
    uncertainty = float(np.nanstd(kept, ddof=1)) if np.isfinite(kept).sum() > 1 else float("nan")
    return region, uncertainty


def stitch_with_uncertainties(
    sector_frames: dict[str, pd.DataFrame],
    sector_uncertainties: dict[str, float],
    *,
    sigma_clip: float = 5.0,
    normalize_by_file: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summary_rows = []
    for key, frame in sector_frames.items():
        if key not in sector_uncertainties:
            continue
        prepared = normalized_sector(frame) if normalize_by_file else frame.copy()
        prepared["flux_err"] = sector_uncertainties[key]
        median_norm = np.nanmedian(prepared["flux"])
        scatter_norm = _robust_scatter(prepared["flux"].to_numpy())
        if np.isfinite(scatter_norm) and scatter_norm > 0:
            prepared["is_outlier"] = np.abs(prepared["flux"] - median_norm) > sigma_clip * scatter_norm
        else:
            prepared["is_outlier"] = False
        frames.append(prepared)
        summary_rows.append(
            {
                "source_file": prepared["source_file"].iloc[0],
                "sector": prepared["sector"].iloc[0],
                "points_after_quality": len(prepared),
                "outliers": int(prepared["is_outlier"].sum()),
                "adopted_uncertainty": sector_uncertainties[key],
                "robust_scatter": scatter_norm,
                "flux_column": prepared["flux_column"].iloc[0],
                "err_column": prepared["err_column"].iloc[0],
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    stitched = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    return stitched, summary


def flatten_sector(
    frame: pd.DataFrame,
    uncertainty: float,
    *,
    window_length: float,
    sigma_clip: float = 5.0,
) -> pd.DataFrame:
    """Flatten one sector and flag only high-side outliers above the trend."""
    prepared = normalized_sector(frame)
    trend = wotan_trend(frame, window_length)
    prepared["trend"] = trend
    prepared["flux_before_flatten"] = prepared["flux"]

    safe_trend = np.where(np.isfinite(trend) & (trend != 0), trend, np.nan)
    prepared["flux"] = prepared["flux_before_flatten"] / safe_trend
    prepared["flux_err"] = uncertainty / safe_trend

    residual = prepared["flux_before_flatten"].to_numpy(dtype=float) - safe_trend
    scatter = float(uncertainty) if np.isfinite(uncertainty) and uncertainty > 0 else _robust_scatter(residual)
    if np.isfinite(scatter) and scatter > 0:
        prepared["is_outlier"] = residual > sigma_clip * scatter
    else:
        prepared["is_outlier"] = False
    return prepared


def flattening_status_table(
    sector_frames: dict[str, pd.DataFrame],
    sector_flattening: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for key, frame in sector_frames.items():
        settings = sector_flattening.get(key, {})
        rows.append(
            {
                "status": "complete" if key in sector_flattening else "incomplete",
                "sector": frame["sector"].iloc[0] if not frame.empty else "",
                "source_file": frame["source_file"].iloc[0] if not frame.empty else "",
                "points": len(frame),
                "wotan_window_days": settings.get("window_length", np.nan),
                "high_outliers": settings.get("high_outliers", np.nan),
            }
        )
    return pd.DataFrame(rows)


def stitch_flattened_sectors(
    sector_frames: dict[str, pd.DataFrame],
    sector_uncertainties: dict[str, float],
    sector_flattening: dict[str, dict],
    *,
    sigma_clip: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    summary_rows = []
    for key, frame in sector_frames.items():
        if key not in sector_uncertainties or key not in sector_flattening:
            continue
        window_length = float(sector_flattening[key]["window_length"])
        prepared = flatten_sector(
            frame,
            sector_uncertainties[key],
            window_length=window_length,
            sigma_clip=sigma_clip,
        )
        frames.append(prepared)
        summary_rows.append(
            {
                "source_file": prepared["source_file"].iloc[0],
                "sector": prepared["sector"].iloc[0],
                "points_after_quality": len(prepared),
                "high_outliers": int(prepared["is_outlier"].sum()),
                "adopted_uncertainty": sector_uncertainties[key],
                "wotan_window_days": window_length,
                "flux_column": prepared["flux_column"].iloc[0],
                "err_column": prepared["err_column"].iloc[0],
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    stitched = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    return stitched, summary


def prepare_light_curves(
    paths: list[str | Path],
    *,
    sigma_clip: float = 5.0,
    quality_zero_only: bool = True,
    normalize_by_file: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, normalize, outlier-flag, estimate missing uncertainties, and stitch light curves."""
    frames = []
    summary_rows = []
    for path in paths:
        frame = read_tess_fits(path, quality_zero_only=quality_zero_only)
        raw_count = len(frame)
        median_flux = np.nanmedian(frame["flux"])
        if normalize_by_file and np.isfinite(median_flux) and median_flux != 0:
            frame["flux_raw"] = frame["flux"]
            frame["flux"] = frame["flux"] / median_flux
            frame["flux_err"] = frame["flux_err"] / median_flux

        scatter = _robust_scatter(frame["flux"].to_numpy())
        missing_err = ~np.isfinite(frame["flux_err"]) | (frame["flux_err"] <= 0)
        if missing_err.any() and np.isfinite(scatter):
            frame.loc[missing_err, "flux_err"] = scatter

        median_norm = np.nanmedian(frame["flux"])
        scatter_norm = _robust_scatter(frame["flux"].to_numpy())
        if np.isfinite(scatter_norm) and scatter_norm > 0:
            frame["is_outlier"] = np.abs(frame["flux"] - median_norm) > sigma_clip * scatter_norm
        else:
            frame["is_outlier"] = False

        frames.append(frame)
        summary_rows.append(
            {
                "source_file": Path(path).name,
                "sector": frame["sector"].iloc[0] if raw_count else "",
                "points_after_quality": raw_count,
                "outliers": int(frame["is_outlier"].sum()),
                "median_flux": median_flux,
                "robust_scatter": scatter_norm,
                "flux_column": frame["flux_column"].iloc[0] if raw_count else "",
                "err_column": frame["err_column"].iloc[0] if raw_count else "",
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    stitched = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows)
    return stitched, summary
