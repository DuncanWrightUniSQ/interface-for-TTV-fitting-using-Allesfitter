# TTV Fitter

An initial Streamlit workbench for transit timing variation analysis and
multi-planet 3D system rendering.

The v0.1 app has five workflow tabs:

- **Data Import**: Load photometry, RV, timing, and allesfitter-style parameter
  CSV files.
- **Linear Transit/RV Fit**: Run non-TTV least-squares fits and optional MCMC
  uncertainty sampling.
- **Per-Transit T0 Fit**: Build expected transit cutouts from a linear ephemeris
  and fit one midpoint per cutout while keeping the transit shape fixed.
- **TTV Model**: Edit star/planet/TTV parameters and fit a simple sinusoidal
  O-C timing model.
- **3D System Model**: Render a multi-planet orbital model using the fitted or
  edited parameters.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

The app is designed to keep working even if optional dependencies such as
`emcee` are unavailable; those controls will simply disable the optional MCMC
actions.

