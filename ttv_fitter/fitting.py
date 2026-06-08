"""Least-squares and MCMC fitting routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .models import PlanetParams, phase_fold, rv_model, trapezoid_transit_model


@dataclass
class FitResult:
    params: dict[str, float]
    success: bool
    message: str
    samples: pd.DataFrame | None = None


def _sigma(values: pd.Series | np.ndarray, fallback: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, fallback)
    return arr


def fit_single_transit_shape(phot: pd.DataFrame, guess: PlanetParams) -> FitResult:
    required = {"time", "flux"}
    if phot.empty or not required.issubset(phot.columns):
        return FitResult({}, False, "Photometry needs time and flux columns.")
    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)

    x0 = np.array([guess.t0, guess.period, guess.radius_ratio, guess.duration_hours, np.nanmedian(flux)])
    lower = np.array([np.nanmin(time) - guess.period, 1e-6, 1e-4, 0.05, 0.5])
    upper = np.array([np.nanmax(time) + guess.period, 1e4, 1.0, 48.0, 1.5])

    def residual(theta: np.ndarray) -> np.ndarray:
        model = trapezoid_transit_model(time, theta[1], theta[0], theta[2], theta[3], baseline=theta[4])
        return (flux - model) / err

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=4000)
    keys = ["t0", "period", "radius_ratio", "duration_hours", "baseline"]
    return FitResult(dict(zip(keys, result.x)), bool(result.success), result.message)


def fit_rv_curve(rv: pd.DataFrame, guess: PlanetParams) -> FitResult:
    if rv.empty or not {"time", "rv"}.issubset(rv.columns):
        return FitResult({}, False, "RV table needs time and rv columns.")
    time = rv["time"].to_numpy(dtype=float)
    velocity = rv["rv"].to_numpy(dtype=float)
    err = _sigma(rv.get("rv_err", pd.Series(np.full(len(rv), 1.0))), 1.0)
    x0 = np.array([guess.period, guess.t0, guess.ecc, guess.omega_deg, guess.rv_k, np.nanmedian(velocity)])
    lower = np.array([1e-6, np.nanmin(time) - guess.period, 0.0, 0.0, -1e5, -1e6])
    upper = np.array([1e5, np.nanmax(time) + guess.period, 0.95, 360.0, 1e5, 1e6])

    def residual(theta: np.ndarray) -> np.ndarray:
        model = rv_model(time, theta[0], theta[1], theta[2], theta[3], theta[4], theta[5])
        return (velocity - model) / err

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=4000)
    keys = ["period", "t0", "ecc", "omega_deg", "rv_k", "gamma"]
    return FitResult(dict(zip(keys, result.x)), bool(result.success), result.message)


def fit_cutout_t0(
    phot: pd.DataFrame,
    period: float,
    expected_t0: float,
    radius_ratio: float,
    duration_hours: float,
    search_half_width_days: float,
) -> FitResult:
    if phot.empty or not {"time", "flux"}.issubset(phot.columns):
        return FitResult({}, False, "Cutout needs time and flux columns.")
    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)
    x0 = np.array([expected_t0, np.nanmedian(flux)])
    lower = np.array([expected_t0 - search_half_width_days, 0.5])
    upper = np.array([expected_t0 + search_half_width_days, 1.5])

    def residual(theta: np.ndarray) -> np.ndarray:
        model = trapezoid_transit_model(
            time,
            period,
            theta[0],
            radius_ratio,
            duration_hours,
            baseline=theta[1],
        )
        return (flux - model) / err

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=1200)
    residuals = residual(result.x)
    dof = max(len(residuals) - len(result.x), 1)
    scatter = float(np.sqrt(np.sum(residuals**2) / dof))
    # Estimate timing uncertainty from local curvature using the Jacobian.
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * scatter**2
        t0_err = float(np.sqrt(max(cov[0, 0], 0)))
    except np.linalg.LinAlgError:
        t0_err = float("nan")
    return FitResult(
        {"tmid": float(result.x[0]), "baseline": float(result.x[1]), "tmid_err": t0_err},
        bool(result.success),
        result.message,
    )


def run_emcee_for_transit(phot: pd.DataFrame, start: dict[str, float], nsteps: int = 800) -> FitResult:
    try:
        import emcee
    except Exception as exc:  # noqa: BLE001
        return FitResult({}, False, f"emcee is not installed: {exc}")
    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)
    keys = ["t0", "period", "radius_ratio", "duration_hours", "baseline"]
    center = np.array([start[k] for k in keys], dtype=float)
    scale = np.maximum(np.abs(center) * 1e-4, [1e-4, 1e-5, 1e-4, 1e-3, 1e-5])

    def log_prob(theta: np.ndarray) -> float:
        t0, period, rr, duration, baseline = theta
        if period <= 0 or not (0 < rr < 1) or duration <= 0 or not (0.5 < baseline < 1.5):
            return -np.inf
        model = trapezoid_transit_model(time, period, t0, rr, duration, baseline=baseline)
        chi = (flux - model) / err
        return -0.5 * float(np.sum(chi**2))

    ndim = len(keys)
    nwalkers = max(2 * ndim + 2, 16)
    p0 = center + scale * np.random.default_rng(42).normal(size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, int(nsteps), progress=False)
    flat = sampler.get_chain(discard=max(nsteps // 2, 1), thin=5, flat=True)
    samples = pd.DataFrame(flat, columns=keys)
    return FitResult(samples.median().to_dict(), True, "MCMC complete.", samples=samples)


def build_cutouts(phot: pd.DataFrame, t0: float, period: float, half_width_days: float) -> pd.DataFrame:
    if phot.empty or "time" not in phot.columns or period <= 0:
        return pd.DataFrame()
    tmin = float(phot["time"].min())
    tmax = float(phot["time"].max())
    first = int(np.floor((tmin - t0) / period)) - 1
    last = int(np.ceil((tmax - t0) / period)) + 1
    rows = []
    for epoch in range(first, last + 1):
        expected = t0 + epoch * period
        mask = (phot["time"] >= expected - half_width_days) & (phot["time"] <= expected + half_width_days)
        npoints = int(mask.sum())
        if npoints:
            rows.append(
                {
                    "fit": True,
                    "epoch": epoch,
                    "expected_tmid": expected,
                    "start": expected - half_width_days,
                    "end": expected + half_width_days,
                    "points": npoints,
                }
            )
    return pd.DataFrame(rows)

