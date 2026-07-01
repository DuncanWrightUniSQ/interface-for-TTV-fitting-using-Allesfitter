"""Least-squares and MCMC fitting routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .models import PlanetParams, limb_darkened_transit_model, phase_fold, rv_model, trapezoid_transit_model


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


def fit_limb_darkened_single_transit(
    phot: pd.DataFrame,
    start: dict[str, float],
    fit_parameters: list[str] | tuple[str, ...] | set[str],
) -> FitResult:
    required = {"time", "flux"}
    if phot.empty or not required.issubset(phot.columns):
        return FitResult({}, False, "Photometry needs time and flux columns.")

    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)
    free = [name for name in fit_parameters if name in start]
    if "t0" not in free:
        free.insert(0, "t0")

    defaults = {
        "t0": float(np.nanmedian(time)),
        "period": 1.0,
        "radius_ratio": 0.1,
        "impact": 0.5,
        "duration_hours": 3.0,
        "a_over_rstar": 10.0,
        "limb_darkening_u1": 0.5,
        "limb_darkening_u2": 0.1,
        "baseline_offset": 0.0,
    }
    values = {key: float(start.get(key, default)) for key, default in defaults.items()}
    span = max(float(np.nanmax(time) - np.nanmin(time)), 1e-5)
    lower_map = {
        "t0": float(np.nanmin(time) - 0.2 * span),
        "period": 1e-6,
        "radius_ratio": 1e-5,
        "impact": 0.0,
        "duration_hours": 0.02,
        "a_over_rstar": 1.0001,
        "limb_darkening_u1": -1.0,
        "limb_darkening_u2": -1.0,
        "baseline_offset": -0.5,
    }
    upper_map = {
        "t0": float(np.nanmax(time) + 0.2 * span),
        "period": 1e5,
        "radius_ratio": 1.0,
        "impact": 2.0,
        "duration_hours": max(48.0, span * 24.0),
        "a_over_rstar": 1e4,
        "limb_darkening_u1": 1.0,
        "limb_darkening_u2": 1.0,
        "baseline_offset": 0.5,
    }
    x0 = np.array([np.clip(values[name], lower_map[name], upper_map[name]) for name in free], dtype=float)
    lower = np.array([lower_map[name] for name in free], dtype=float)
    upper = np.array([upper_map[name] for name in free], dtype=float)

    def merged(theta: np.ndarray) -> dict[str, float]:
        current = values.copy()
        current.update({name: float(value) for name, value in zip(free, theta)})
        current["period"] = max(current["period"], 1e-8)
        current["radius_ratio"] = float(np.clip(current["radius_ratio"], 1e-6, 1.0))
        current["impact"] = max(current["impact"], 0.0)
        current["duration_hours"] = max(current["duration_hours"], 1e-4)
        current["a_over_rstar"] = max(current["a_over_rstar"], 1.0001)
        return current

    def residual(theta: np.ndarray) -> np.ndarray:
        current = merged(theta)
        model = limb_darkened_transit_model(
            time,
            current["period"],
            current["t0"],
            current["radius_ratio"],
            current["impact"],
            current["duration_hours"],
            current["limb_darkening_u1"],
            current["limb_darkening_u2"],
            baseline_offset=current["baseline_offset"],
            a_over_rstar=current["a_over_rstar"],
        )
        return (flux - model) / err

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=6000)
    fitted = merged(result.x)
    fitted["baseline"] = 1.0 + fitted["baseline_offset"]
    fitted["cost"] = float(result.cost)
    fitted["nfev"] = float(result.nfev)
    return FitResult(fitted, bool(result.success), result.message)


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
    impact: float,
    a_over_rstar: float,
    duration_hours: float,
    limb_darkening_u1: float,
    limb_darkening_u2: float,
    baseline_offset: float,
    search_half_width_days: float,
) -> FitResult:
    if phot.empty or not {"time", "flux"}.issubset(phot.columns):
        return FitResult({}, False, "Cutout needs time and flux columns.")
    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)
    x0 = np.array([expected_t0, baseline_offset])
    lower = np.array([expected_t0 - search_half_width_days, -0.5])
    upper = np.array([expected_t0 + search_half_width_days, 0.5])

    def residual(theta: np.ndarray) -> np.ndarray:
        model = limb_darkened_transit_model(
            time,
            period,
            theta[0],
            radius_ratio,
            impact,
            duration_hours,
            limb_darkening_u1,
            limb_darkening_u2,
            baseline_offset=theta[1],
            a_over_rstar=a_over_rstar,
        )
        return (flux - model) / err

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=1200)
    return FitResult(
        {"tmid": float(result.x[0]), "baseline_offset": float(result.x[1]), "baseline": float(1.0 + result.x[1])},
        bool(result.success),
        result.message,
    )


def run_cutout_t0_mcmc(
    phot: pd.DataFrame,
    period: float,
    start_t0: float,
    radius_ratio: float,
    impact: float,
    a_over_rstar: float,
    duration_hours: float,
    limb_darkening_u1: float,
    limb_darkening_u2: float,
    baseline_offset: float,
    search_half_width_days: float = 1.5 / 24.0,
    nwalkers: int = 6,
    nsteps: int = 1000,
    burn: int = 500,
) -> FitResult:
    try:
        import emcee
    except Exception as exc:  # noqa: BLE001
        return FitResult({}, False, f"emcee is not installed: {exc}")
    if phot.empty or not {"time", "flux"}.issubset(phot.columns):
        return FitResult({}, False, "Cutout needs time and flux columns.")

    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)
    lower = float(start_t0) - float(search_half_width_days)
    upper = float(start_t0) + float(search_half_width_days)

    def log_prob(theta: np.ndarray) -> float:
        t0 = float(theta[0])
        if not lower <= t0 <= upper:
            return -np.inf
        model = limb_darkened_transit_model(
            time,
            period,
            t0,
            radius_ratio,
            impact,
            duration_hours,
            limb_darkening_u1,
            limb_darkening_u2,
            baseline_offset=baseline_offset,
            a_over_rstar=a_over_rstar,
        )
        chi = (flux - model) / err
        return -0.5 * float(np.sum(chi**2))

    rng = np.random.default_rng(42)
    nwalkers = max(int(nwalkers), 2)
    nsteps = max(int(nsteps), 2)
    burn = min(max(int(burn), 0), nsteps - 1)
    p0 = rng.uniform(lower, upper, size=(nwalkers, 1))
    sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    flat = sampler.get_chain(discard=burn, flat=True)[:, 0]
    p16, median, p84 = np.nanpercentile(flat, [16, 50, 84])
    samples = pd.DataFrame({"tmid": flat})
    return FitResult(
        {
            "tmid": float(median),
            "tmid_err_minus": float(median - p16),
            "tmid_err_plus": float(p84 - median),
            "tmid_err": float(0.5 * (p84 - p16)),
            "prior_lower": lower,
            "prior_upper": upper,
            "nwalkers": float(nwalkers),
            "nsteps": float(nsteps),
            "burn": float(burn),
        },
        True,
        "T0-only MCMC complete.",
        samples=samples,
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
