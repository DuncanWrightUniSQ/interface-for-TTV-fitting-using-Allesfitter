"""Plotly visualizations for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .models import coerce_planet_table, multi_transit_model, orbital_position, rv_model
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
        margin=dict(l=20, r=20, t=30, b=45),
        xaxis_title="Transit epoch",
        yaxis_title="O-C [minutes]",
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
    axis = dict(range=[-max_a * 1.15, max_a * 1.15], showbackground=False, zeroline=False, title="")
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube"),
        showlegend=True,
    )
    return fig
