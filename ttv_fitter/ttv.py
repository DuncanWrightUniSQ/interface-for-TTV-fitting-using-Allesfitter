"""Transit timing variation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def linear_ephemeris(epoch: np.ndarray, t0: float, period: float) -> np.ndarray:
    return float(t0) + np.asarray(epoch, dtype=float) * float(period)


def oc_table(timings: pd.DataFrame, t0: float, period: float) -> pd.DataFrame:
    if timings.empty or not {"epoch", "tmid"}.issubset(timings.columns):
        return pd.DataFrame()
    out = timings.copy()
    out["calculated"] = linear_ephemeris(out["epoch"].to_numpy(dtype=float), t0, period)
    out["oc_days"] = out["tmid"] - out["calculated"]
    out["oc_minutes"] = out["oc_days"] * 24.0 * 60.0
    return out


def fit_linear_ephemeris(timings: pd.DataFrame) -> dict[str, float]:
    if timings.empty or not {"epoch", "tmid"}.issubset(timings.columns):
        return {}
    epoch = timings["epoch"].to_numpy(dtype=float)
    tmid = timings["tmid"].to_numpy(dtype=float)
    err = timings.get("tmid_err", pd.Series(np.full(len(timings), 0.01))).to_numpy(dtype=float)
    err = np.where(np.isfinite(err) & (err > 0), err, 0.01)
    A = np.vstack([np.ones_like(epoch), epoch]).T
    W = np.diag(1.0 / err**2)
    beta = np.linalg.inv(A.T @ W @ A) @ (A.T @ W @ tmid)
    return {"t0": float(beta[0]), "period": float(beta[1])}


def sinusoid_oc(epoch: np.ndarray, amplitude_minutes: float, super_period_epochs: float, phase: float) -> np.ndarray:
    super_period_epochs = max(float(super_period_epochs), 1e-6)
    return float(amplitude_minutes) * np.sin(2 * np.pi * np.asarray(epoch, dtype=float) / super_period_epochs + phase)


def fit_sinusoidal_ttv(timings: pd.DataFrame, t0: float, period: float) -> tuple[dict[str, float], pd.DataFrame]:
    table = oc_table(timings, t0, period)
    if table.empty:
        return {}, table
    epoch = table["epoch"].to_numpy(dtype=float)
    oc_min = table["oc_minutes"].to_numpy(dtype=float)
    err_min = table.get("tmid_err", pd.Series(np.full(len(table), 0.01))).to_numpy(dtype=float) * 1440.0
    err_min = np.where(np.isfinite(err_min) & (err_min > 0), err_min, np.nanstd(oc_min) or 1.0)
    amp0 = max(float(np.nanstd(oc_min)), 1.0)
    span = max(float(np.nanmax(epoch) - np.nanmin(epoch)), 2.0)
    x0 = np.array([amp0, span, 0.0, 0.0])

    def residual(theta: np.ndarray) -> np.ndarray:
        amp, super_period, phase, offset = theta
        model = sinusoid_oc(epoch, amp, super_period, phase) + offset
        return (oc_min - model) / err_min

    result = least_squares(
        residual,
        x0,
        bounds=([-1e5, 1.0, -2 * np.pi, -1e5], [1e5, 1e6, 2 * np.pi, 1e5]),
        max_nfev=4000,
    )
    amp, super_period, phase, offset = result.x
    table["ttv_model_minutes"] = sinusoid_oc(epoch, amp, super_period, phase) + offset
    table["ttv_residual_minutes"] = table["oc_minutes"] - table["ttv_model_minutes"]
    return {
        "amplitude_minutes": float(amp),
        "super_period_epochs": float(super_period),
        "phase_rad": float(phase),
        "offset_minutes": float(offset),
    }, table


def allesfitter_ttv_rows(timings: pd.DataFrame, companion: str = "b") -> pd.DataFrame:
    rows = []
    for idx, row in timings.sort_values("epoch").reset_index(drop=True).iterrows():
        tmid = float(row["tmid"])
        err = float(row.get("tmid_err", 0.02))
        rows.append(
            {
                "name": f"{companion}_tmid_{idx:04d}",
                "value": f"{tmid:.10g}",
                "fit": 1,
                "bounds": f"normal {tmid:.10g} {max(err, 1e-8):.10g}",
                "label": f"{companion} transit {int(row['epoch'])} midpoint",
                "unit": "d",
                "coupled_with": "",
                "truth": "",
                "init_err": f"{max(err * 1e-3, 1e-10):.10g}",
            }
        )
    return pd.DataFrame(rows)

