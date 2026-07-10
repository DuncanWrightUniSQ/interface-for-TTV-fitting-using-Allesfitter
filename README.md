# TTV Fitter

Version 1.0.0 Streamlit workbench for transit timing variation analysis,
TESS photometry preparation, per-transit timing fits, and multi-planet system
rendering.

This project provides an interface for TTV fitting using Allesfitter. It
builds on and interoperates with Allesfitter workflows originally created by
Maximilian Guenther <maximilian.guenther@esa.int> and Tansu Daylan
<tansu@wustl.edu>. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
third-party attribution.

The v1.0 app has five workflow tabs:

- **TTV data preparation workflow**: Query/load TESS photometry, review sector
  uncertainties, detrend high-cadence sector data, mask transits, and export
  either stitched photometry or one prepared CSV per sector.
- **Linear Transit Fit**: Load stitched or sector photometry, retrieve ExoFOP
  planet/star parameters, run global linear transit fits, or fit one clean
  transit per planet to seed later timing work.
- **Per-Transit T0 Fit**: Build expected transit cutouts from a linear ephemeris
  or single-transit seed, run robust multi-start least-squares midpoint fits,
  and optionally estimate T0 uncertainties with one-parameter MCMC.
- **TTV Model**: Edit star/planet/TTV parameters and fit a simple sinusoidal
  O-C timing model or inspect a physical REBOUND model.
- **3D System Model**: Render a multi-planet orbital model using the fitted or
  edited parameters.

Generated target data, downloaded MAST files, and prepared sector products are
written under `data/` and are intentionally ignored by git.

## Setup

Clone the repository first:

```bash
git clone https://github.com/DuncanWrightUniSQ/interface-for-TTV-fitting-using-Allesfitter.git
cd interface-for-TTV-fitting-using-Allesfitter
```

Then create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you downloaded the repository as a ZIP file instead of using git, unzip it
first, open a terminal in the unzipped folder
`interface-for-TTV-fitting-using-Allesfitter-main`, and then run the same
virtual-environment and install commands above.

## Run

```bash
streamlit run streamlit_app.py
```

The app is designed to keep working even if optional dependencies such as
`emcee` are unavailable; those controls will simply disable the optional MCMC
actions.

## Folder Structure

The repository is self-contained for normal app use:

- `streamlit_app.py` is the Streamlit entry point.
- `ttv_fitter/` contains the TTV-specific workflow code.
- `app/` contains the small Allesfitter-style workflow helpers used by the
  interface. This folder is intentionally included in git so the app no longer
  depends on a local-only `Allesfitter_work` directory.
- `data/` is created locally for downloaded MAST products and prepared target
  files; it is ignored by git.

The optional full Allesfitter sampler launcher uses a Python executable that
can import `allesfitter`. By default it tries the current virtual environment.
Advanced users can point it at a separate runtime by setting:

```bash
export TTV_FITTER_ALLESFITTER_PYTHON=/path/to/python-with-allesfitter
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).

Allesfitter is also distributed under the MIT License by its original creators,
Maximilian Guenther and Tansu Daylan; this project includes attribution in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
