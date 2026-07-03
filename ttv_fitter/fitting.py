"""Least-squares and MCMC fitting routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .models import (
    PlanetParams,
    limb_darkened_transit_model,
    phase_fold,
    rv_model,
    trapezoid_transit_model,
)


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
    if "duration_hours" in free and np.isfinite(float(start.get("a_over_rstar", np.nan))):
        free.remove("duration_hours")

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
    duration_seed_days = max(values["duration_hours"] / 24.0, 1e-5)
    t0_half_width = min(0.45 * span, max(3.0 * duration_seed_days, 0.15))
    t0_center = float(np.clip(values["t0"], np.nanmin(time), np.nanmax(time)))
    radius_seed = float(np.clip(values["radius_ratio"], 1e-5, 1.0))
    duration_upper_hours = max(0.05, min(span * 24.0 * 0.6, max(values["duration_hours"] * 2.0, values["duration_hours"] + 1.0)))
    lower_map = {
        "t0": max(float(np.nanmin(time)), t0_center - t0_half_width),
        "period": 1e-6,
        "radius_ratio": max(1e-4, radius_seed * 0.25),
        "impact": 0.0,
        "duration_hours": max(0.02, values["duration_hours"] * 0.4),
        "a_over_rstar": max(1.0001, values["a_over_rstar"] * 0.25),
        "limb_darkening_u1": max(-1.0, values["limb_darkening_u1"] - 0.5),
        "limb_darkening_u2": max(-1.0, values["limb_darkening_u2"] - 0.5),
        "baseline_offset": -0.5,
    }
    upper_map = {
        "t0": min(float(np.nanmax(time)), t0_center + t0_half_width),
        "period": 1e5,
        "radius_ratio": min(1.0, max(radius_seed * 3.0, radius_seed + 0.05)),
        "impact": min(2.0, 1.0 + min(1.0, max(radius_seed * 3.0, radius_seed + 0.05))),
        "duration_hours": duration_upper_hours,
        "a_over_rstar": max(values["a_over_rstar"] * 3.0, values["a_over_rstar"] + 5.0, 1.0002),
        "limb_darkening_u1": min(1.0, values["limb_darkening_u1"] + 0.5),
        "limb_darkening_u2": min(1.0, values["limb_darkening_u2"] + 0.5),
        "baseline_offset": 0.5,
    }
    if upper_map["t0"] <= lower_map["t0"]:
        lower_map["t0"] = float(np.nanmin(time))
        upper_map["t0"] = float(np.nanmax(time))
    if upper_map["duration_hours"] <= lower_map["duration_hours"]:
        upper_map["duration_hours"] = lower_map["duration_hours"] + 0.05
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
    use_t0_multistart: bool = False,
) -> FitResult:
    if phot.empty or not {"time", "flux"}.issubset(phot.columns):
        return FitResult({}, False, "Cutout needs time and flux columns.")
    time = phot["time"].to_numpy(dtype=float)
    flux = phot["flux"].to_numpy(dtype=float)
    err = _sigma(phot.get("flux_err", pd.Series(np.full(len(phot), 1e-3))), 1e-3)
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

    if use_t0_multistart:
        duration_days = max(float(duration_hours) / 24.0, 1e-8)
        total_window_days = max(float(search_half_width_days) * 2.0, 0.0)
        n_starts = int(np.clip(np.ceil((total_window_days / duration_days) - 1e-9), 1, 10))
        if n_starts > 1:
            start_lower = lower[0] + 0.5 * duration_days
            start_upper = upper[0] - 0.5 * duration_days
            if start_upper > start_lower:
                t0_starts = np.linspace(start_lower, start_upper, n_starts)
            else:
                t0_starts = np.linspace(lower[0], upper[0], n_starts)
        else:
            t0_starts = np.array([expected_t0], dtype=float)
    else:
        t0_starts = np.array([expected_t0], dtype=float)

    best_result = None
    best_score = np.inf
    for t0_start in t0_starts:
        x0 = np.array([float(np.clip(t0_start, lower[0], upper[0])), baseline_offset], dtype=float)
        result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=1200)
        score = float(np.sum(residual(result.x) ** 2))
        if np.isfinite(score) and score < best_score:
            best_result = result
            best_score = score

    if best_result is None:
        return FitResult({}, False, "No finite least-squares solution was found.")
    result = best_result
    return FitResult(
        {
            "tmid": float(result.x[0]),
            "baseline_offset": float(result.x[1]),
            "baseline": float(1.0 + result.x[1]),
            "residual_sum_squares": float(best_score),
            "n_t0_startpoints": float(len(t0_starts)),
        },
        bool(result.success),
        result.message,
    )


def _post_burn_t0_chain_by_walker(chain: np.ndarray, nwalkers: int) -> np.ndarray:
    arr = np.asarray(chain, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=float)
    if arr.shape[1] == nwalkers:
        return arr
    if arr.shape[0] == nwalkers:
        return arr.T
    return arr


def _trim_nonconverged_t0_walkers(
    chain_by_step_walker: np.ndarray,
    duration_hours: float,
    enabled: bool,
) -> tuple[np.ndarray, int, int, bool, float]:
    chain = np.asarray(chain_by_step_walker, dtype=float)
    if chain.ndim != 2 or chain.size == 0:
        return np.array([], dtype=float), 0, 0, False, np.nan

    finite_by_walker = np.any(np.isfinite(chain), axis=0)
    if not np.any(finite_by_walker):
        return np.array([], dtype=float), 0, 0, False, np.nan

    active_chain = chain[:, finite_by_walker]
    full_flat = active_chain.reshape(-1)
    full_flat = full_flat[np.isfinite(full_flat)]
    total_walkers = int(active_chain.shape[1])
    if not enabled or total_walkers < 3 or full_flat.size == 0:
        return full_flat, total_walkers, total_walkers, False, np.nan

    walker_medians = np.nanmedian(active_chain, axis=0)
    finite_medians = np.isfinite(walker_medians)
    if int(np.sum(finite_medians)) < 3:
        return full_flat, total_walkers, total_walkers, False, np.nan

    center = float(np.nanmedian(full_flat))
    mad = float(np.nanmedian(np.abs(full_flat - center)))
    robust_sigma = 1.4826 * mad if np.isfinite(mad) else 0.0
    duration_days = max(float(duration_hours) / 24.0, 1e-8)
    threshold = max(5.0 * robust_sigma, 0.05 * duration_days, 1e-8)
    keep = finite_medians & (np.abs(walker_medians - center) <= threshold)

    min_keep = max(2, int(np.ceil(0.5 * total_walkers)))
    if int(np.sum(keep)) < min_keep:
        return full_flat, total_walkers, total_walkers, False, threshold

    trimmed = active_chain[:, keep].reshape(-1)
    trimmed = trimmed[np.isfinite(trimmed)]
    if trimmed.size == 0:
        return full_flat, total_walkers, total_walkers, False, threshold
    used_walkers = int(np.sum(keep))
    return trimmed, total_walkers, used_walkers, used_walkers < total_walkers, threshold


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
    trim_nonconverged_walkers: bool = True,
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

    nwalkers = max(int(nwalkers), 2)
    nsteps = max(int(nsteps), 2)
    burn = min(max(int(burn), 0), nsteps - 1)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return FitResult({}, False, "MCMC T0 search range is invalid.")

    rng = np.random.default_rng(42)
    span = upper - lower
    centers = np.linspace(lower, upper, nwalkers + 2, dtype=float)[1:-1]
    jitter_scale = 0.2 * span / max(nwalkers + 1, 1)
    p0_values = centers + rng.uniform(-jitter_scale, jitter_scale, size=nwalkers)
    p0 = np.clip(p0_values, lower + span * 1e-9, upper - span * 1e-9).reshape(nwalkers, 1)
    sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=False)
    chain = _post_burn_t0_chain_by_walker(sampler.get_chain(discard=burn, flat=False), nwalkers)
    flat, total_walkers, used_walkers, trim_applied, trim_threshold = _trim_nonconverged_t0_walkers(
        chain,
        duration_hours,
        trim_nonconverged_walkers,
    )
    if flat.size == 0:
        raw_flat = np.asarray(sampler.get_chain(discard=burn, flat=True), dtype=float)
        flat = raw_flat[:, 0] if raw_flat.ndim == 2 and raw_flat.shape[1] else raw_flat.reshape(-1)
        flat = flat[np.isfinite(flat)]
        total_walkers = nwalkers
        used_walkers = nwalkers
        trim_applied = False
        trim_threshold = np.nan
    if flat.size == 0:
        return FitResult({}, False, "MCMC did not produce finite T0 samples.")
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
            "nwalkers_total": float(total_walkers),
            "nwalkers_used": float(used_walkers),
            "nwalkers_trimmed": float(max(total_walkers - used_walkers, 0)),
            "nsteps": float(nsteps),
            "burn": float(burn),
            "trim_nonconverged_walkers": float(bool(trim_nonconverged_walkers)),
            "walker_trim_applied": float(bool(trim_applied)),
            "walker_trim_threshold_days": float(trim_threshold) if np.isfinite(trim_threshold) else np.nan,
            "initial_t0_min": float(np.min(p0[:, 0])),
            "initial_t0_max": float(np.max(p0[:, 0])),
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
