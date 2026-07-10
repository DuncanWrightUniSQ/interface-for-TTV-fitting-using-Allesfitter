"""Interactive placeholder plots for early GUI development."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": ["drawline", "eraseshape"],
}


def _base_layout(fig: go.Figure, title: str, xaxis: str, yaxis: str) -> go.Figure:
    fig.update_layout(
        title=title,
        height=430,
        margin={"l": 48, "r": 24, "t": 56, "b": 48},
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(title_text=xaxis, showgrid=True, zeroline=False)
    fig.update_yaxes(title_text=yaxis, showgrid=True, zeroline=False)
    return fig


def photometry_preview() -> go.Figure:
    time = np.linspace(0, 27.4, 1500)
    phase = ((time - 2.1) % 4.2) / 4.2
    transit = 0.012 * np.exp(-0.5 * ((phase - 0.5) / 0.028) ** 2)
    trend = 0.0015 * np.sin(2 * np.pi * time / 13.7)
    noise = 0.0007 * np.sin(2 * np.pi * time * 3.1)
    flux = 1.0 + trend - transit + noise

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=time,
            y=flux,
            mode="markers",
            marker={"size": 3, "color": "#2563eb", "opacity": 0.55},
            name="Flux",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=1.0 + trend,
            mode="lines",
            line={"color": "#111827", "width": 2},
            name="Baseline",
        )
    )
    return _base_layout(fig, "Photometry preview", "Time - BTJD", "Relative flux")


def prepared_photometry_preview(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return _base_layout(fig, "Prepared photometry", "Time - BTJD", "Relative flux")

    clean = frame[~frame.get("is_outlier", False)]
    outliers = frame[frame.get("is_outlier", False)]
    fig.add_trace(
        go.Scattergl(
            x=clean["time"],
            y=clean["flux"],
            mode="markers",
            marker={"size": 3, "color": "#2563eb", "opacity": 0.5},
            text=clean["source_file"],
            name="Prepared flux",
        )
    )
    if not outliers.empty:
        fig.add_trace(
            go.Scattergl(
                x=outliers["time"],
                y=outliers["flux"],
                mode="markers",
                marker={"size": 6, "color": "#dc2626", "symbol": "x"},
                text=outliers["source_file"],
                name="Outliers",
            )
        )
    return _base_layout(fig, "Prepared stitched photometry", "Time - BTJD", "Relative flux")


def photometry_model_overlay(
    frame: pd.DataFrame,
    model_flux: np.ndarray | None,
    *,
    title: str = "Initial transit model on data",
    model_name: str = "Initial model",
) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return _base_layout(fig, title, "Time - BTJD", "Relative flux")

    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["flux"],
            mode="markers",
            marker={"size": 4, "color": "#2563eb", "opacity": 0.28},
            name="Prepared data",
        )
    )
    if model_flux is not None and len(model_flux) == len(frame):
        order = np.argsort(frame["time"].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter(
                x=frame["time"].to_numpy(dtype=float)[order],
                y=np.asarray(model_flux, dtype=float)[order],
                mode="lines",
                line={"color": "#ef4444", "width": 5},
                name=model_name,
            )
        )
    return _base_layout(fig, title, "Time - BTJD", "Relative flux")


def sector_uncertainty_preview(frame: pd.DataFrame, trend: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return _base_layout(fig, "Sector uncertainty review", "Time - BTJD", "Relative flux")

    show_errors = trend is None
    has_err = show_errors and "flux_err" in frame and np.isfinite(frame["flux_err"]).any()
    marker_opacity = 0.32 if trend is not None else 0.55
    point_trace = go.Scatter if trend is not None else go.Scattergl
    fig.add_trace(
        point_trace(
            x=frame["time"],
            y=frame["flux"],
            error_y={"type": "data", "array": frame["flux_err"], "visible": bool(has_err)},
            mode="markers",
            marker={"size": 4, "color": "#2563eb", "opacity": marker_opacity},
            name="Flux",
        )
    )
    if trend is not None and len(trend) == len(frame):
        fig.add_trace(
            go.Scatter(
                x=frame["time"],
                y=trend,
                mode="lines",
                line={"color": "#ef4444", "width": 5},
                name="Wotan trend",
            )
        )
    return _base_layout(fig, "Sector uncertainty review", "Time - BTJD", "Relative flux")


def uncertainty_residual_preview(region: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if region.empty:
        return _base_layout(fig, "Selected noise residuals", "Time - BTJD", "Detrended flux")

    clean = region[~region["uncertainty_outlier"]]
    outliers = region[region["uncertainty_outlier"]]
    fig.add_trace(
        go.Scattergl(
            x=clean["time"],
            y=clean["residual"],
            mode="markers",
            marker={"size": 5, "color": "#0f766e", "opacity": 0.65},
            name="Used for uncertainty",
        )
    )
    if not outliers.empty:
        fig.add_trace(
            go.Scattergl(
                x=outliers["time"],
                y=outliers["residual"],
                mode="markers",
                marker={"size": 8, "color": "#dc2626", "symbol": "x"},
                name="Rejected > 5 sigma",
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    return _base_layout(fig, "Selected noise residuals", "Time - BTJD", "Detrended flux")


def flattened_sector_preview(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return _base_layout(fig, "Flattened sector", "Time - BTJD", "Flattened flux")

    clean = frame[~frame.get("is_outlier", False)]
    outliers = frame[frame.get("is_outlier", False)]
    fig.add_trace(
        go.Scattergl(
            x=clean["time"],
            y=clean["flux"],
            mode="markers",
            marker={"size": 4, "color": "#2563eb", "opacity": 0.55},
            name="Kept points",
        )
    )
    if not outliers.empty:
        fig.add_trace(
            go.Scattergl(
                x=outliers["time"],
                y=outliers["flux"],
                mode="markers",
                marker={"size": 8, "color": "#dc2626", "symbol": "x"},
                name="High outliers",
            )
        )
    fig.add_hline(y=1, line_dash="dash", line_color="#64748b")
    return _base_layout(fig, "Flattened sector with high-side outliers", "Time - BTJD", "Flattened flux")


def rv_preview() -> go.Figure:
    phase = np.linspace(0, 1, 38)
    rv = 18.0 * np.sin(2 * np.pi * phase + 0.5) + 2.2 * np.sin(6 * np.pi * phase)
    err = np.full_like(phase, 2.4)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=phase,
            y=rv,
            error_y={"type": "data", "array": err, "visible": True},
            mode="markers",
            marker={"size": 8, "color": "#be123c"},
            name="RV",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 200),
            y=18.0 * np.sin(2 * np.pi * np.linspace(0, 1, 200) + 0.5),
            mode="lines",
            line={"color": "#334155", "width": 2},
            name="Model",
        )
    )
    return _base_layout(fig, "RV preview", "Orbital phase", "RV [m/s]")


def ttv_oc_preview() -> go.Figure:
    epochs = np.arange(-12, 22)
    oc = 2.4 * np.sin(2 * np.pi * (epochs - 1) / 17.0) + 0.35 * np.cos(epochs)
    err = 0.55 + 0.15 * (np.sin(epochs) ** 2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=oc,
            error_y={"type": "data", "array": err, "visible": True},
            mode="markers",
            marker={"size": 8, "color": "#0f766e"},
            name="Observed - calculated",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    return _base_layout(fig, "Transit timing residuals", "Epoch", "O-C [minutes]")


def folded_transit_preview() -> go.Figure:
    phase = np.linspace(-0.08, 0.08, 900)
    model = 1.0 - 0.014 * np.exp(-0.5 * (phase / 0.018) ** 2)
    flux = model + 0.0008 * np.sin(phase * 420)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=phase,
            y=flux,
            mode="markers",
            marker={"size": 4, "color": "#7c3aed", "opacity": 0.42},
            name="Folded data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=phase,
            y=model,
            mode="lines",
            line={"color": "#111827", "width": 3},
            name="Transit model",
        )
    )
    return _base_layout(fig, "Phase folded transit", "Phase [days]", "Relative flux")


def corner_placeholder() -> go.Figure:
    x = np.linspace(-3.0, 3.0, 80)
    y = np.linspace(-3.0, 3.0, 80)
    xx, yy = np.meshgrid(x, y)
    zz = np.exp(-0.5 * (xx**2 + (yy - 0.4 * xx) ** 2 / 0.55))

    fig = go.Figure(data=go.Contour(x=x, y=y, z=zz, colorscale="Viridis", contours_coloring="heatmap"))
    return _base_layout(fig, "Posterior surface placeholder", "Parameter A", "Parameter B")


def mcmc_chain_plot(chain: np.ndarray, labels: list[str], *, title: str = "MCMC chains after burn-in trim") -> go.Figure:
    if chain.size == 0:
        return _base_layout(go.Figure(), "MCMC chains", "Step", "Parameter value")

    steps, walkers, ndim = chain.shape
    shown_walkers = min(walkers, 32)
    stride = max(1, steps // 2500)
    x = np.arange(steps)[::stride]
    fig = make_subplots(
        rows=ndim,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=[labels[i] if i < len(labels) else f"parameter {i + 1}" for i in range(ndim)],
    )
    for dim in range(ndim):
        for walker in range(shown_walkers):
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=chain[::stride, walker, dim],
                    mode="lines",
                    line={"width": 1, "color": "#2563eb"},
                    opacity=0.22,
                    showlegend=False,
                    hovertemplate="step=%{x}<br>value=%{y:.10g}<extra></extra>",
                ),
                row=dim + 1,
                col=1,
            )
        fig.update_yaxes(title_text=labels[dim] if dim < len(labels) else f"p{dim + 1}", row=dim + 1, col=1)

    fig.update_layout(
        title=title,
        height=max(360, min(1200, 170 * ndim)),
        margin={"l": 72, "r": 24, "t": 56, "b": 48},
        hovermode="closest",
    )
    fig.update_xaxes(title_text="Step", row=ndim, col=1)
    return fig


def mcmc_corner_plot(samples: np.ndarray, labels: list[str]) -> go.Figure:
    if samples.size == 0:
        return _base_layout(go.Figure(), "Posterior corner plot", "Parameter", "Parameter")

    ndim = samples.shape[1]
    if ndim < 2:
        fig = go.Figure(
            go.Histogram(
                x=samples[:, 0],
                marker={"color": "#7c3aed", "opacity": 0.7},
                name=labels[0] if labels else "parameter 1",
            )
        )
        return _base_layout(fig, "Interactive posterior distribution", labels[0] if labels else "Parameter", "Count")

    max_points = 5000
    if len(samples) > max_points:
        rng = np.random.default_rng(42)
        samples = samples[rng.choice(len(samples), size=max_points, replace=False)]

    axis_labels = [labels[dim] if dim < len(labels) else f"parameter {dim + 1}" for dim in range(ndim)]
    fig = make_subplots(
        rows=ndim - 1,
        cols=ndim - 1,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.015,
        vertical_spacing=0.015,
    )
    for y_dim in range(1, ndim):
        for x_dim in range(y_dim):
            row = y_dim
            col = x_dim + 1
            fig.add_trace(
                go.Scattergl(
                    x=samples[:, x_dim],
                    y=samples[:, y_dim],
                    mode="markers",
                    marker={"size": 3, "color": "#7c3aed", "opacity": 0.35},
                    showlegend=False,
                    hovertemplate=f"{axis_labels[x_dim]}=%{{x:.10g}}<br>{axis_labels[y_dim]}=%{{y:.10g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
            if row == ndim - 1:
                fig.update_xaxes(title_text=axis_labels[x_dim], row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text=axis_labels[y_dim], row=row, col=col)

    fig.update_layout(
        title="Interactive posterior corner plot",
        height=max(520, min(1200, 190 * (ndim - 1))),
        margin={"l": 56, "r": 24, "t": 56, "b": 56},
        dragmode="select",
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


def isochrone_preview() -> go.Figure:
    teff = np.linspace(3200, 7200, 240)
    luminosity = 10 ** ((teff - 5772) / 1800)
    sample = pd.DataFrame(
        {
            "teff": [5700, 5900, 5600, 6050],
            "luminosity": [0.92, 1.08, 0.98, 1.22],
            "label": ["median", "sample A", "sample B", "sample C"],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=teff, y=luminosity, mode="lines", line={"color": "#475569"}, name="Isochrone")
    )
    fig.add_trace(
        go.Scatter(
            x=sample["teff"],
            y=sample["luminosity"],
            text=sample["label"],
            mode="markers+text",
            marker={"size": 10, "color": "#f97316"},
            textposition="top center",
            name="Stellar samples",
        )
    )
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(type="log")
    return _base_layout(fig, "Isochrone/model context", "Effective temperature [K]", "Luminosity [solar]")
