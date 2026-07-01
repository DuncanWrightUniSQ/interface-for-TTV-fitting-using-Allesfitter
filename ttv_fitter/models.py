"""Core transit, RV, and orbital geometry models."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np
import pandas as pd


G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30
M_JUP_KG = 1.89813e27
R_SUN_M = 6.957e8
DAY_S = 86400.0
AU_OVER_RSUN = 215.032
M_JUP_OVER_M_SUN = M_JUP_KG / M_SUN_KG
R_SUN_AU = 1.0 / AU_OVER_RSUN
AU_M = 1.495978707e11


@dataclass
class StarParams:
    name: str = "host"
    mass_solar: float = 1.0
    radius_solar: float = 1.0


@dataclass
class PlanetParams:
    name: str = "b"
    mass_jupiter: float = 0.003
    period: float = 3.0
    t0: float = 0.0
    radius_ratio: float = 0.08
    impact: float = 0.4
    duration_hours: float = 3.0
    a_over_rstar: float = 8.0
    inclination_deg: float = 87.0
    ecc: float = 0.0
    omega_deg: float = 90.0
    mean_anomaly_deg: float = 0.0
    rv_k: float = 5.0
    color: str = "#2563eb"

    def to_row(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_PLANET_COLUMNS = list(PlanetParams().to_row().keys())


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan")
    med = np.nanmedian(finite)
    mad = np.nanmedian(np.abs(finite - med))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(finite))


def coerce_planet_table(table: pd.DataFrame | None) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame([PlanetParams().to_row()], columns=DEFAULT_PLANET_COLUMNS)
    out = table.copy()
    for col, default in PlanetParams().to_row().items():
        if col not in out.columns:
            out[col] = default
    for col in DEFAULT_PLANET_COLUMNS:
        if col not in {"name", "color"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(PlanetParams().to_row()[col])
    out["name"] = out["name"].fillna("").astype(str).replace("", "b")
    out["color"] = out["color"].fillna("#2563eb").astype(str)
    return out[DEFAULT_PLANET_COLUMNS]


def phase_fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    return ((np.asarray(time, dtype=float) - t0 + 0.5 * period) % period) - 0.5 * period


def trapezoid_transit_model(
    time: np.ndarray,
    period: float,
    t0: float,
    radius_ratio: float,
    duration_hours: float,
    ingress_fraction: float = 0.18,
    baseline: float = 1.0,
) -> np.ndarray:
    """Return a fast trapezoid transit approximation for UI fitting."""
    time = np.asarray(time, dtype=float)
    duration = max(float(duration_hours) / 24.0, 1e-8)
    ingress = np.clip(float(ingress_fraction), 0.02, 0.49) * duration
    depth = np.clip(float(radius_ratio), 0.0, 1.0) ** 2
    dt = np.abs(phase_fold(time, max(period, 1e-8), t0))

    flux = np.full_like(time, float(baseline), dtype=float)
    flat_half = max(0.0, 0.5 * duration - ingress)
    full_half = 0.5 * duration
    in_flat = dt <= flat_half
    in_ingress = (dt > flat_half) & (dt <= full_half)
    flux[in_flat] -= depth
    if np.any(in_ingress):
        ramp = (full_half - dt[in_ingress]) / max(ingress, 1e-8)
        flux[in_ingress] -= depth * np.clip(ramp, 0, 1)
    return flux


def _derive_a_over_rstar_from_duration(
    period: float,
    radius_ratio: float,
    impact: float,
    duration_hours: float,
) -> float:
    duration_days = max(float(duration_hours) / 24.0, 1e-8)
    period = max(float(period), 1e-8)
    radius_ratio = float(np.clip(radius_ratio, 1e-6, 1.0))
    impact = float(np.clip(impact, 0.0, 1.0 + radius_ratio - 1e-6))
    sine_term = np.sin(np.pi * min(duration_days / period, 0.95))
    if not np.isfinite(sine_term) or sine_term <= 0:
        return 8.0
    chord_sq = max((1.0 + radius_ratio) ** 2 - impact**2, 1e-8)
    a_sq = (chord_sq + (sine_term * impact) ** 2) / max(sine_term**2, 1e-8)
    return float(max(np.sqrt(a_sq), 1.0 + radius_ratio + 1e-6))


def limb_darkened_transit_model(
    time: np.ndarray,
    period: float,
    t0: float,
    radius_ratio: float,
    impact: float,
    duration_hours: float,
    u1: float,
    u2: float,
    baseline_offset: float = 0.0,
    a_over_rstar: float | None = None,
) -> np.ndarray:
    """Return a quadratic limb-darkened transit model for per-transit inspection."""
    time = np.asarray(time, dtype=float)
    try:
        import batman

        if a_over_rstar is None or not np.isfinite(float(a_over_rstar)) or float(a_over_rstar) <= 0:
            a_over_rstar = _derive_a_over_rstar_from_duration(period, radius_ratio, impact, duration_hours)
        a_over_rstar = float(a_over_rstar)
        inclination = float(np.rad2deg(np.arccos(np.clip(float(impact) / a_over_rstar, 0.0, 1.0))))
        params = batman.TransitParams()
        params.t0 = float(t0)
        params.per = max(float(period), 1e-8)
        params.rp = float(np.clip(radius_ratio, 1e-6, 1.0))
        params.a = a_over_rstar
        params.inc = inclination
        params.ecc = 0.0
        params.w = 90.0
        params.u = [float(np.clip(u1, -1.0, 1.0)), float(np.clip(u2, -1.0, 1.0))]
        params.limb_dark = "quadratic"
        model = batman.TransitModel(params, time)
        return model.light_curve(params) + float(baseline_offset)
    except Exception:
        return trapezoid_transit_model(
            time,
            period,
            t0,
            radius_ratio,
            duration_hours,
            baseline=1.0 + float(baseline_offset),
        )


def multi_transit_model(time: np.ndarray, planets: pd.DataFrame, baseline: float = 1.0) -> np.ndarray:
    flux = np.full_like(np.asarray(time, dtype=float), baseline, dtype=float)
    for row in coerce_planet_table(planets).to_dict("records"):
        model = trapezoid_transit_model(
            time,
            row["period"],
            row["t0"],
            row["radius_ratio"],
            row["duration_hours"],
            baseline=1.0,
        )
        flux += model - 1.0
    return flux


def kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    eccentricity = float(np.clip(eccentricity, 0.0, 0.95))
    E = mean_anomaly + np.sign(np.sin(mean_anomaly)) * 0.85 * eccentricity
    for _ in range(80):
        delta = (E - eccentricity * np.sin(E) - mean_anomaly) / (
            1 - eccentricity * np.cos(E)
        )
        E -= delta
        if np.nanmax(np.abs(delta)) < 1e-12:
            break
    return E


def true_anomaly(time: np.ndarray, t0: float, period: float, eccentricity: float) -> np.ndarray:
    M = 2 * np.pi * (((np.asarray(time, dtype=float) - t0) / max(period, 1e-8)) % 1.0)
    E = kepler(M, eccentricity)
    return 2 * np.arctan2(
        np.sqrt(1 + eccentricity) * np.sin(E / 2),
        np.sqrt(1 - eccentricity) * np.cos(E / 2),
    )


def rv_model(
    time: np.ndarray,
    period: float,
    t0: float,
    ecc: float,
    omega_deg: float,
    rv_k: float,
    gamma: float = 0.0,
) -> np.ndarray:
    ecc = float(np.clip(ecc, 0.0, 0.95))
    omega = np.deg2rad(omega_deg)
    nu = true_anomaly(time, t0, period, ecc)
    return gamma + rv_k * (np.cos(nu + omega) + ecc * np.cos(omega))


def derive_a_over_rstar(period_days: float, star_mass_solar: float, star_radius_solar: float) -> float:
    if period_days <= 0 or star_mass_solar <= 0 or star_radius_solar <= 0:
        return float("nan")
    period_years = period_days / 365.25
    a_au = (star_mass_solar * period_years**2) ** (1.0 / 3.0)
    return float(a_au * AU_OVER_RSUN / star_radius_solar)


def derive_inclination_deg(a_over_rstar: float, impact: float, ecc: float = 0.0, omega_deg: float = 90.0) -> float:
    if a_over_rstar <= 0:
        return float("nan")
    omega = np.deg2rad(omega_deg)
    correction = (1 + ecc * np.sin(omega)) / max(1 - ecc**2, 1e-6)
    cosi = np.clip(impact / max(a_over_rstar * correction, 1e-8), 0.0, 1.0)
    return float(np.rad2deg(np.arccos(cosi)))


def orbital_position(
    phase: np.ndarray,
    a_over_rstar: float,
    inc_deg: float,
    ecc: float = 0.0,
    omega_deg: float = 90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=float)
    nu = 2 * np.pi * phase
    ecc = float(np.clip(ecc, 0.0, 0.95))
    radius = a_over_rstar * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    angle = nu + np.deg2rad(omega_deg)
    inc = np.deg2rad(inc_deg)
    x = radius * np.cos(angle)
    y_orb = radius * np.sin(angle)
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)
    return x, y, z


def planet_from_row(row: pd.Series | dict[str, object]) -> PlanetParams:
    data = PlanetParams().to_row()
    for key in data:
        if key in row:
            data[key] = row[key]
    return PlanetParams(**data)
