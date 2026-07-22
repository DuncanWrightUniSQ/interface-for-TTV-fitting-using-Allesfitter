"""Plotly visualizations for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .models import (
    coerce_planet_table,
    limb_darkened_transit_model,
    multi_transit_model,
    orbital_position,
    rv_model,
)
from .ttv import oc_table


PLOT_CONFIG = {"displaylogo": False, "responsive": True}


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=30))
    return fig


def photometry_figure(phot: pd.DataFrame, planets: pd.DataFrame | None = None) -> go.Figure:
    if phot.empty or not {"time", "flux"}.issubset(phot.columns):
        return empty_figure("Load photometry with time and flux columns.")
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=phot["time"],
            y=phot["flux"],
            mode="markers",
            marker=dict(size=3, color="#334155", opacity=0.55),
            name="data",
        )
    )
    if planets is not None and not planets.empty:
        time = np.linspace(float(phot["time"].min()), float(phot["time"].max()), 2000)
        fig.add_trace(
            go.Scatter(
                x=time,
                y=multi_transit_model(time, planets),
                mode="lines",
                line=dict(color="#dc2626", width=2),
                name="model",
            )
        )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=45), xaxis_title="Time", yaxis_title="Flux")
    return fig


def rv_figure(rv: pd.DataFrame, planets: pd.DataFrame | None = None) -> go.Figure:
    if rv.empty or not {"time", "rv"}.issubset(rv.columns):
        return empty_figure("Load RV data with time and rv columns.")
    fig = go.Figure()
    err = rv["rv_err"] if "rv_err" in rv.columns else None
    fig.add_trace(
        go.Scatter(
            x=rv["time"],
            y=rv["rv"],
            error_y=dict(type="data", array=err, visible=err is not None),
            mode="markers",
            marker=dict(size=7, color="#0f766e"),
            name="RV",
        )
    )
    if planets is not None and not planets.empty:
        row = coerce_planet_table(planets).iloc[0]
        time = np.linspace(float(rv["time"].min()), float(rv["time"].max()), 1000)
        y = rv_model(
            time,
            float(row["period"]),
            float(row["t0"]),
            float(row["ecc"]),
            float(row["omega_deg"]),
            float(row["rv_k"]),
            0.0,
        )
        fig.add_trace(go.Scatter(x=time, y=y, mode="lines", line=dict(color="#b45309"), name="model"))
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=45), xaxis_title="Time", yaxis_title="RV [m/s]")
    return fig


def oc_figure(timings: pd.DataFrame, t0: float, period: float, model_table: pd.DataFrame | None = None) -> go.Figure:
    table = model_table if model_table is not None and not model_table.empty else oc_table(timings, t0, period)
    if table.empty:
        return empty_figure("Load or fit transit timings to see an O-C diagram.")
    fig = go.Figure()
    err = table["tmid_err"] * 1440.0 if "tmid_err" in table.columns else None
    fig.add_trace(
        go.Scatter(
            x=table["epoch"],
            y=table["oc_minutes"],
            error_y=dict(type="data", array=err, visible=err is not None),
            mode="markers",
            marker=dict(size=8, color="#1d4ed8"),
            name="O-C",
        )
    )
    if "ttv_model_minutes" in table.columns:
        sorted_table = table.sort_values("epoch")
        fig.add_trace(
            go.Scatter(
                x=sorted_table["epoch"],
                y=sorted_table["ttv_model_minutes"],
                mode="lines",
                line=dict(color="#dc2626", width=2),
                name="TTV model",
            )
        )
    fig.update_layout(
        height=420,
        # Leave room for the rotated O-C label and tick values in exported
        # standalone HTML figures (which do not get Streamlit's auto-margin).
        margin=dict(l=90, r=25, t=30, b=55),
        xaxis_title="Transit epoch",
        yaxis_title="O-C [minutes]",
        yaxis=dict(automargin=True, title_standoff=14),
    )
    return fig


def cutout_fit_figure(phot: pd.DataFrame, fit: dict[str, float]) -> go.Figure:
    required = {"time", "flux"}
    if phot.empty or not required.issubset(phot.columns):
        return empty_figure("Load photometry to inspect a fitted transit.")
    start = float(fit["start"])
    end = float(fit["end"])
    window = phot.loc[(phot["time"] >= start) & (phot["time"] <= end)].copy()
    if window.empty:
        return empty_figure("No photometry points are available for the selected transit.")

    fig = go.Figure()
    err = window["flux_err"] if "flux_err" in window.columns else None
    fig.add_trace(
        go.Scatter(
            x=window["time"],
            y=window["flux"],
            error_y=dict(type="data", array=err, visible=err is not None),
            mode="markers",
            marker=dict(size=5, color="#334155", opacity=0.55),
            name="photometry",
        )
    )
    time = np.linspace(start, end, 600)
    model = limb_darkened_transit_model(
        time,
        float(fit["period"]),
        float(fit["tmid"]),
        float(fit["radius_ratio"]),
        float(fit["impact"]),
        float(fit["duration_hours"]),
        float(fit["limb_darkening_u1"]),
        float(fit["limb_darkening_u2"]),
        baseline_offset=float(fit.get("baseline_offset", 0.0)),
        a_over_rstar=float(fit.get("a_over_rstar", np.nan)),
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=model,
            mode="lines",
            line=dict(color="#dc2626", width=4),
            name="fit",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=30, b=45),
        xaxis_title="Time",
        yaxis_title="Flux",
    )
    return fig


def t0_histogram_figure(samples: pd.DataFrame, tmid: float, err_minus: float, err_plus: float) -> go.Figure:
    if samples.empty or "tmid" not in samples.columns:
        return empty_figure("Run T0 MCMC to see the timing posterior.")
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=samples["tmid"],
            nbinsx=40,
            marker=dict(color="#2563eb", opacity=0.72),
            name="trimmed samples",
        )
    )
    fig.add_vline(x=tmid, line=dict(color="#dc2626", width=3), annotation_text="median")
    fig.add_vrect(
        x0=tmid - err_minus,
        x1=tmid + err_plus,
        fillcolor="#dc2626",
        opacity=0.16,
        line_width=0,
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=45),
        xaxis_title="T0",
        yaxis_title="Samples",
    )
    return fig


def system_3d_figure(planets: pd.DataFrame, phase: float = 0.0) -> go.Figure:
    table = coerce_planet_table(planets)
    fig = go.Figure()
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, "#f8fafc"], [1, "#facc15"]],
            showscale=False,
            opacity=0.96,
            name="star",
        )
    )
    samples = np.linspace(0, 1, 360)
    max_a = 1.0
    for row in table.to_dict("records"):
        ox, oy, oz = orbital_position(samples, row["a_over_rstar"], row["inclination_deg"], row["ecc"], row["omega_deg"])
        max_a = max(max_a, float(np.nanmax(np.abs([ox, oy, oz]))))
        fig.add_trace(
            go.Scatter3d(
                x=ox,
                y=oy,
                z=oz,
                mode="lines",
                line=dict(color=row["color"], width=4),
                name=f"{row['name']} orbit",
            )
        )
        px, py, pz = orbital_position(np.array([phase % 1.0]), row["a_over_rstar"], row["inclination_deg"], row["ecc"], row["omega_deg"])
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="markers+text",
                marker=dict(size=max(4, row["radius_ratio"] * 42), color=row["color"]),
                text=[row["name"]],
                textposition="top center",
                name=row["name"],
            )
        )
    observer_z0 = max_a * 1.22
    observer_z1 = max_a * 1.55
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[observer_z0, observer_z1],
            mode="lines+markers+text",
            line=dict(color="#111827", width=6),
            marker=dict(size=[3, 8], color="#111827", symbol="diamond"),
            text=["", "Observer"],
            textposition="top center",
            name="observer direction",
            hovertemplate="Direction to observer<br>+z line of sight<extra></extra>",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[0],
            y=[0],
            z=[observer_z1],
            u=[0],
            v=[0],
            w=[max_a * 0.22],
            sizemode="absolute",
            sizeref=max_a * 0.16,
            anchor="tail",
            colorscale=[[0, "#111827"], [1, "#111827"]],
            showscale=False,
            name="observer arrow",
            hovertemplate="Direction to observer<br>+z line of sight<extra></extra>",
        )
    )
    axis = dict(range=[-max_a * 1.75, max_a * 1.75], showbackground=False, zeroline=False, title="")
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube"),
        showlegend=True,
    )
    return fig


def physical_oc_figure(comparison: pd.DataFrame) -> go.Figure:
    if comparison.empty:
        return empty_figure("Run a REBOUND model with transit crossings to see physical TTVs.")
    fig = go.Figure()
    if "oc_minutes" in comparison.columns and comparison["oc_minutes"].notna().any():
        err = comparison["tmid_err"] * 1440.0 if "tmid_err" in comparison.columns else None
        fig.add_trace(
            go.Scatter(
                x=comparison["epoch"],
                y=comparison["oc_minutes"],
                error_y=dict(type="data", array=err, visible=err is not None),
                mode="markers",
                marker=dict(size=8, color="#1d4ed8"),
                name="observed O-C",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=comparison["epoch"],
            y=comparison["ttv_model_minutes"],
            mode="markers+lines",
            marker=dict(size=7, color="#dc2626"),
            line=dict(color="#dc2626", width=2),
            name="REBOUND model",
        )
    )
    if "ttv_residual_minutes" in comparison.columns and comparison["ttv_residual_minutes"].notna().any():
        fig.add_trace(
            go.Bar(
                x=comparison["epoch"],
                y=comparison["ttv_residual_minutes"],
                marker_color="#94a3b8",
                opacity=0.45,
                name="data - model",
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(title="Residual [min]", overlaying="y", side="right", showgrid=False)
        )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=30, b=45),
        xaxis_title="Transit epoch",
        yaxis_title="TTV [minutes]",
    )
    return fig


def physical_rv_figure(rv: pd.DataFrame, rv_curve: pd.DataFrame) -> go.Figure:
    if rv_curve.empty:
        return empty_figure("Run a REBOUND model to see the host-star RV curve.")
    fig = go.Figure()
    if not rv.empty and {"time", "rv"}.issubset(rv.columns):
        err = rv["rv_err"] if "rv_err" in rv.columns else None
        fig.add_trace(
            go.Scatter(
                x=rv["time"],
                y=rv["rv"],
                error_y=dict(type="data", array=err, visible=err is not None),
                mode="markers",
                marker=dict(size=7, color="#0f766e"),
                name="RV data",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=rv_curve["time"],
            y=rv_curve["rv_model"],
            mode="lines",
            line=dict(color="#b45309", width=2),
            name="REBOUND host RV",
        )
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=45), xaxis_title="Time", yaxis_title="RV [m/s]")
    return fig


def physical_transit_figure(phot: pd.DataFrame, transit_flux: pd.DataFrame) -> go.Figure:
    if transit_flux.empty:
        return empty_figure("Run a REBOUND model to see transit predictions.")
    fig = go.Figure()
    if not phot.empty and {"time", "flux"}.issubset(phot.columns):
        fig.add_trace(
            go.Scattergl(
                x=phot["time"],
                y=phot["flux"],
                mode="markers",
                marker=dict(size=3, color="#334155", opacity=0.45),
                name="photometry",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=transit_flux["time"],
            y=transit_flux["flux_model"],
            mode="lines",
            line=dict(color="#dc2626", width=2),
            name="REBOUND TTV transit model",
        )
    )
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=30, b=45), xaxis_title="Time", yaxis_title="Flux")
    return fig


def rebound_3d_figure(orbit_trace: pd.DataFrame, phase_index: int | None = None) -> go.Figure:
    if orbit_trace.empty:
        return empty_figure("Run a REBOUND model to see integrated 3D orbits.")
    fig = go.Figure()
    max_a = 1e-6
    for planet, group in orbit_trace.groupby("planet"):
        color = str(group["color"].iloc[0]) if "color" in group else "#2563eb"
        max_a = max(max_a, float(np.nanmax(np.abs(group[["x_au", "y_au", "z_au"]].to_numpy()))))
        fig.add_trace(
            go.Scatter3d(
                x=group["x_au"],
                y=group["y_au"],
                z=group["z_au"],
                mode="lines",
                line=dict(color=color, width=4),
                name=f"{planet} integrated orbit",
            )
        )
        if phase_index is None:
            row = group.iloc[-1]
        else:
            row = group.iloc[int(np.clip(phase_index, 0, len(group) - 1))]
        fig.add_trace(
            go.Scatter3d(
                x=[row["x_au"]],
                y=[row["y_au"]],
                z=[row["z_au"]],
                mode="markers+text",
                marker=dict(size=6, color=color),
                text=[planet],
                textposition="top center",
                name=planet,
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(size=10, color="#facc15"),
            name="host",
        )
    )
    observer_z0 = max_a * 1.22
    observer_z1 = max_a * 1.55
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[observer_z0, observer_z1],
            mode="lines+markers+text",
            line=dict(color="#111827", width=6),
            marker=dict(size=[3, 8], color="#111827", symbol="diamond"),
            text=["", "Observer"],
            textposition="top center",
            name="observer direction",
            hovertemplate="Direction to observer<br>+z line of sight<extra></extra>",
        )
    )
    fig.add_trace(
        go.Cone(
            x=[0],
            y=[0],
            z=[observer_z1],
            u=[0],
            v=[0],
            w=[max_a * 0.22],
            sizemode="absolute",
            sizeref=max_a * 0.16,
            anchor="tail",
            colorscale=[[0, "#111827"], [1, "#111827"]],
            showscale=False,
            name="observer arrow",
            hovertemplate="Direction to observer<br>+z line of sight<extra></extra>",
        )
    )
    axis = dict(range=[-max_a * 1.75, max_a * 1.75], showbackground=False, zeroline=False, title="")
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube"),
    )
    return fig
