"""REBOUND-based physical multi-planet TTV utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .models import (
    AU_M,
    DAY_S,
    M_JUP_OVER_M_SUN,
    R_SUN_AU,
    coerce_planet_table,
    derive_a_over_rstar,
    trapezoid_transit_model,
)


G_AU3_MSUN_DAY2 = 4.0 * np.pi**2 / 365.25**2
AU_PER_DAY_TO_M_PER_S = AU_M / DAY_S


@dataclass(frozen=True)
class DynamicsResult:
    model_timings: pd.DataFrame
    rv_curve: pd.DataFrame
    orbit_trace: pd.DataFrame
    transit_flux: pd.DataFrame


def rebound_available() -> tuple[bool, str]:
    try:
        import rebound  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, ""


def true_to_mean_anomaly(true_anomaly_rad: float, eccentricity: float) -> float:
    eccentricity = float(np.clip(eccentricity, 0.0, 0.95))
    if eccentricity == 0:
        return true_anomaly_rad % (2 * np.pi)
    E = 2 * np.arctan2(
        np.sqrt(1 - eccentricity) * np.sin(true_anomaly_rad / 2),
        np.sqrt(1 + eccentricity) * np.cos(true_anomaly_rad / 2),
    )
    return (E - eccentricity * np.sin(E)) % (2 * np.pi)


def transit_mean_anomaly_deg(eccentricity: float, omega_deg: float) -> float:
    """Approximate inferior-conjunction mean anomaly for the app's sky convention."""
    true_anomaly = np.deg2rad(90.0 - float(omega_deg))
    return float(np.rad2deg(true_to_mean_anomaly(true_anomaly, eccentricity)) % 360.0)


def _align_time_value_to_reference(value: float, reference_time: float) -> float:
    if float(reference_time) < 100000 and float(value) > 2400000:
        return float(value) - 2457000.0
    if float(reference_time) > 2400000 and float(value) < 100000:
        return float(value) + 2457000.0
    return float(value)


def phase_from_t0(row: pd.Series, reference_time: float) -> float:
    mean_transit = transit_mean_anomaly_deg(float(row["ecc"]), float(row["omega_deg"]))
    n_deg_per_day = 360.0 / max(float(row["period"]), 1e-12)
    aligned_t0 = _align_time_value_to_reference(float(row["t0"]), float(reference_time))
    return float((mean_transit - n_deg_per_day * (aligned_t0 - reference_time)) % 360.0)


def prepare_dynamics_table(
    planets: pd.DataFrame,
    star_mass_solar: float,
    star_radius_solar: float,
    reference_time: float,
    *,
    initialize_from_t0: bool = True,
) -> pd.DataFrame:
    table = coerce_planet_table(planets).copy()
    for idx, row in table.iterrows():
        table.loc[idx, "t0"] = _align_time_value_to_reference(float(row["t0"]), float(reference_time))
        table.loc[idx, "a_over_rstar"] = derive_a_over_rstar(
            float(row["period"]),
            float(star_mass_solar),
            float(star_radius_solar),
        )
        if initialize_from_t0:
            table.loc[idx, "mean_anomaly_deg"] = phase_from_t0(row, reference_time)
    return table


def build_simulation(
    planets: pd.DataFrame,
    star_mass_solar: float,
    star_radius_solar: float,
    reference_time: float,
) -> object:
    import rebound

    table = coerce_planet_table(planets)
    sim = rebound.Simulation()
    sim.G = G_AU3_MSUN_DAY2
    sim.t = float(reference_time)
    sim.add(m=float(star_mass_solar))
    for row in table.to_dict("records"):
        a_au = float(row["a_over_rstar"]) * float(star_radius_solar) * R_SUN_AU
        if not np.isfinite(a_au) or a_au <= 0:
            a_au = derive_a_over_rstar(float(row["period"]), star_mass_solar, star_radius_solar) * star_radius_solar * R_SUN_AU
        sim.add(
            m=max(float(row["mass_jupiter"]), 0.0) * M_JUP_OVER_M_SUN,
            a=max(a_au, 1e-8),
            e=float(np.clip(row["ecc"], 0.0, 0.95)),
            inc=np.deg2rad(float(row["inclination_deg"])),
            Omega=0.0,
            omega=np.deg2rad(float(row["omega_deg"])),
            M=np.deg2rad(float(row["mean_anomaly_deg"])),
        )
    sim.move_to_com()
    return sim


def _relative_state(sim: object, planet_index: int) -> tuple[float, float, float]:
    star = sim.particles[0]
    planet = sim.particles[planet_index]
    return planet.x - star.x, planet.y - star.y, planet.z - star.z


def _sample_states(
    sim: object,
    planets: pd.DataFrame,
    times: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orbit_rows: list[dict[str, float | str]] = []
    rv_rows: list[dict[str, float]] = []
    for time in times:
        sim.integrate(float(time), exact_finish_time=0)
        star = sim.particles[0]
        rv_rows.append({"time": float(time), "rv_model": float(star.vz * AU_PER_DAY_TO_M_PER_S)})
        for planet_index, row in enumerate(planets.to_dict("records"), start=1):
            x, y, z = _relative_state(sim, planet_index)
            orbit_rows.append(
                {
                    "time": float(time),
                    "planet": row["name"],
                    "x_au": float(x),
                    "y_au": float(y),
                    "z_au": float(z),
                    "sky_sep_au": float(np.hypot(x, y)),
                    "color": row["color"],
                }
            )
    rv = pd.DataFrame(rv_rows)
    if not rv.empty:
        rv["rv_model"] -= float(rv["rv_model"].mean())
    return pd.DataFrame(orbit_rows), rv


def _crossing_time(
    sim: object,
    planet_index: int,
    left_time: float,
    right_time: float,
    iterations: int = 28,
) -> tuple[float, float, float]:
    left_sim = sim.copy()
    left_sim.integrate(left_time)
    left_x, _, _ = _relative_state(left_sim, planet_index)
    lo = float(left_time)
    hi = float(right_time)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        mid_sim = sim.copy()
        mid_sim.integrate(mid)
        mid_x, _, _ = _relative_state(mid_sim, planet_index)
        if np.sign(mid_x) == np.sign(left_x):
            lo = mid
            left_x = mid_x
        else:
            hi = mid
    tmid = 0.5 * (lo + hi)
    root_sim = sim.copy()
    root_sim.integrate(tmid)
    x, y, z = _relative_state(root_sim, planet_index)
    return tmid, float(np.hypot(x, y)), float(z)


def detect_transit_times(
    base_sim: object,
    planets: pd.DataFrame,
    times: np.ndarray,
    star_radius_solar: float,
    grazing_scale: float = 1.25,
) -> pd.DataFrame:
    sim = base_sim.copy()
    previous: dict[int, tuple[float, float, float, float]] = {}
    rows: list[dict[str, float | int | str]] = []
    star_radius_au = float(star_radius_solar) * R_SUN_AU
    planet_rows = planets.to_dict("records")
    for time in times:
        sim.integrate(float(time), exact_finish_time=1)
        for planet_index, row in enumerate(planet_rows, start=1):
            x, y, z = _relative_state(sim, planet_index)
            prior = previous.get(planet_index)
            if prior is not None:
                prior_time, prior_x, _prior_y, _prior_z = prior
                if prior_x == 0 or np.sign(prior_x) != np.sign(x):
                    tmid, sky_sep, z_at_mid = _crossing_time(base_sim, planet_index, prior_time, float(time))
                    radius_limit = star_radius_au * (1.0 + float(row["radius_ratio"])) * grazing_scale
                    if z_at_mid > 0 and sky_sep <= radius_limit:
                        epoch = int(np.round((tmid - float(row["t0"])) / max(float(row["period"]), 1e-12)))
                        rows.append(
                            {
                                "planet": row["name"],
                                "epoch": epoch,
                                "tmid_model": float(tmid),
                                "sky_sep_rstar": float(sky_sep / max(star_radius_au, 1e-12)),
                                "z_au": float(z_at_mid),
                            }
                        )
            previous[planet_index] = (float(time), float(x), float(y), float(z))
    if not rows:
        return pd.DataFrame(columns=["planet", "epoch", "tmid_model", "sky_sep_rstar", "z_au"])
    return pd.DataFrame(rows).drop_duplicates(["planet", "epoch"], keep="first").sort_values(["planet", "epoch"])


def transit_flux_from_timings(
    times: np.ndarray,
    model_timings: pd.DataFrame,
    planets: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame({"time": np.asarray(times, dtype=float)})
    flux = np.ones(len(out), dtype=float)
    table = coerce_planet_table(planets)
    for row in table.to_dict("records"):
        planet_timings = model_timings.loc[model_timings["planet"] == row["name"]]
        for tmid in planet_timings["tmid_model"].to_numpy(dtype=float):
            local = trapezoid_transit_model(
                out["time"].to_numpy(dtype=float),
                period=1e9,
                t0=tmid,
                radius_ratio=float(row["radius_ratio"]),
                duration_hours=float(row["duration_hours"]),
                baseline=1.0,
            )
            flux += local - 1.0
    out["flux_model"] = flux
    return out


def run_physical_ttv_model(
    planets: pd.DataFrame,
    star_mass_solar: float,
    star_radius_solar: float,
    reference_time: float,
    start_time: float,
    end_time: float,
    *,
    sample_step_days: float,
    initialize_from_t0: bool = True,
    integrator: str = "ias15",
    grazing_scale: float = 1.25,
) -> DynamicsResult:
    if end_time <= start_time:
        raise ValueError("End time must be greater than start time.")
    table = prepare_dynamics_table(
        planets,
        star_mass_solar,
        star_radius_solar,
        reference_time,
        initialize_from_t0=initialize_from_t0,
    )
    sim = build_simulation(table, star_mass_solar, star_radius_solar, reference_time)
    sim.integrator = integrator
    sim.dt = max(sample_step_days, 1e-4)
    times = np.arange(float(start_time), float(end_time) + sample_step_days, max(sample_step_days, 1e-4))
    if len(times) < 3:
        times = np.linspace(float(start_time), float(end_time), 3)
    orbit_trace, rv_curve = _sample_states(sim.copy(), table, times)
    model_timings = detect_transit_times(
        sim.copy(),
        table,
        times,
        star_radius_solar,
        grazing_scale=grazing_scale,
    )
    transit_flux = transit_flux_from_timings(times, model_timings, table)
    return DynamicsResult(model_timings, rv_curve, orbit_trace, transit_flux)


def _align_reference_t0_to_timescale(t0: float, times: pd.Series | np.ndarray) -> float:
    values = pd.to_numeric(pd.Series(times), errors="coerce").dropna()
    if values.empty:
        return float(t0)
    median_time = float(values.median())
    if median_time < 100000 and float(t0) > 2400000:
        return float(t0) - 2457000.0
    if median_time > 2400000 and float(t0) < 100000:
        return float(t0) + 2457000.0
    return float(t0)


def compare_model_to_timings(
    observed: pd.DataFrame,
    model_timings: pd.DataFrame,
    planet: str,
    t0: float,
    period: float,
) -> pd.DataFrame:
    model = model_timings.loc[model_timings["planet"] == planet].copy()
    if model.empty:
        return pd.DataFrame()
    reference_times = [model["tmid_model"]]
    if not observed.empty and "tmid" in observed.columns:
        reference_times.append(observed["tmid"])
    aligned_t0 = _align_reference_t0_to_timescale(t0, pd.concat(reference_times, ignore_index=True))
    model["epoch"] = np.round((model["tmid_model"] - aligned_t0) / max(float(period), 1e-12)).astype(int)
    model["linear_tmid"] = float(aligned_t0) + model["epoch"] * float(period)
    model["ttv_model_minutes"] = (model["tmid_model"] - model["linear_tmid"]) * 1440.0
    if observed.empty or not {"epoch", "tmid"}.issubset(observed.columns):
        model["tmid"] = np.nan
        model["tmid_err"] = np.nan
        model["oc_minutes"] = np.nan
        model["ttv_residual_minutes"] = np.nan
        return model
    obs = observed.copy()
    if "tmid_err" not in obs.columns:
        obs["tmid_err"] = np.nan
    obs["epoch"] = pd.to_numeric(obs["epoch"], errors="coerce").round().astype("Int64")
    obs["oc_minutes"] = (obs["tmid"] - (float(aligned_t0) + obs["epoch"].astype(float) * float(period))) * 1440.0
    merged = model.merge(obs[["epoch", "tmid", "tmid_err", "oc_minutes"]], on="epoch", how="left")
    merged["ttv_residual_minutes"] = merged["oc_minutes"] - merged["ttv_model_minutes"]
    return merged
