"""Data loading and allesfitter-style conversion helpers."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


PHOTOMETRY_ALIASES = {
    "time": ["time", "bjd", "btjd", "jd", "mjd"],
    "flux": ["flux", "rel_flux", "pdcsap_flux", "sap_flux", "kspsap_flux"],
    "flux_err": ["flux_err", "fluxerr", "err", "error", "flux_error", "pdcsap_flux_err"],
}

RV_ALIASES = {
    "time": ["time", "bjd", "jd", "mjd"],
    "rv": ["rv", "radial_velocity", "velocity"],
    "rv_err": ["rv_err", "rverr", "err", "error", "sigma_rv"],
}

TIMING_ALIASES = {
    "epoch": ["epoch", "transit", "transit_epoch", "n"],
    "tmid": ["tmid", "t0", "midpoint", "time", "bjd"],
    "tmid_err": ["tmid_err", "t0_err", "err", "error", "sigma"],
}


def _decode_upload(uploaded_file) -> StringIO:
    return StringIO(uploaded_file.getvalue().decode("utf-8"))


def read_table(uploaded_file) -> pd.DataFrame:
    """Read CSV/TSV/whitespace tables from a Streamlit upload."""
    name = uploaded_file.name.lower()
    buffer = _decode_upload(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(buffer, sep="\t")
    if name.endswith(".txt") or name.endswith(".dat"):
        return pd.read_csv(buffer, sep=r"\s+", comment="#")
    return pd.read_csv(buffer)


def _find_col(columns: list[str], aliases: list[str]) -> str | None:
    lower = {str(col).strip().lower(): col for col in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def normalize_columns(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    out = pd.DataFrame()
    for canonical, choices in aliases.items():
        col = _find_col(list(frame.columns), choices)
        if col is not None:
            out[canonical] = pd.to_numeric(frame[col], errors="coerce")
    for col in frame.columns:
        if col not in out.columns and str(col).lower() not in aliases:
            out[str(col)] = frame[col]
    return out.dropna(subset=[next(iter(aliases))]).reset_index(drop=True) if not out.empty else out


def normalize_photometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(frame, PHOTOMETRY_ALIASES)
    if "flux_err" not in out.columns and "flux" in out.columns:
        scatter = np.nanstd(out["flux"].to_numpy(dtype=float))
        out["flux_err"] = scatter if np.isfinite(scatter) and scatter > 0 else 1e-3
    return out


def normalize_rv(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(frame, RV_ALIASES)
    if "rv_err" not in out.columns and "rv" in out.columns:
        scatter = np.nanstd(out["rv"].to_numpy(dtype=float))
        out["rv_err"] = scatter if np.isfinite(scatter) and scatter > 0 else 1.0
    return out


def normalize_timings(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(frame, TIMING_ALIASES)
    if "epoch" not in out.columns and "tmid" in out.columns:
        out["epoch"] = np.arange(len(out), dtype=int)
    if "tmid_err" not in out.columns and "tmid" in out.columns:
        out["tmid_err"] = np.nanstd(out["tmid"].to_numpy(dtype=float)) or 0.01
    return out


def parse_allesfitter_params(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert common allesfitter params.csv rows into an editable planet table."""
    if frame.empty or "name" not in frame.columns:
        return pd.DataFrame()
    values = dict(zip(frame["name"].astype(str), frame.get("value", pd.Series(dtype=object))))
    planets: dict[str, dict[str, object]] = {}
    for name, value in values.items():
        if "_" not in name:
            continue
        planet, param = name.split("_", 1)
        if not planet or planet in {"host", "baseline", "ln"}:
            continue
        row = planets.setdefault(planet, {"name": planet})
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if param in {"epoch", "tmid_0000"}:
            row["t0"] = number
        elif param == "period":
            row["period"] = number
        elif param == "rr":
            row["radius_ratio"] = number
        elif param == "K":
            row["rv_k"] = number
        elif param == "cosi":
            row["inclination_deg"] = float(np.rad2deg(np.arccos(np.clip(number, 0, 1))))
        elif param == "f_c":
            row["_fc"] = number
        elif param == "f_s":
            row["_fs"] = number
    rows = []
    for row in planets.values():
        fc = row.pop("_fc", 0.0)
        fs = row.pop("_fs", 0.0)
        row["ecc"] = min(fc**2 + fs**2, 0.95)
        row["omega_deg"] = float(np.rad2deg(np.arctan2(fs, fc)) % 360) if (fc or fs) else 90.0
        rows.append(row)
    return pd.DataFrame(rows)


def output_path(name: str) -> Path:
    path = Path("data") / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path / name

