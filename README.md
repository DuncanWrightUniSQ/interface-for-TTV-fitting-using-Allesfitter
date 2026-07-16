# Simplified TTV Fitter

This branch extends the completed simplified TTV fitter into a single automatic
multi-target workflow. Upload one target per line and the app queries MAST,
prepares each target, retrieves planet b parameters, fits a representative
transit, measures every suitable transit time with least-squares and MCMC, and
saves the plots and results before moving to the next target.

This project provides an interface for TTV fitting using Allesfitter. It
builds on and interoperates with Allesfitter workflows originally created by
Maximilian Guenther <maximilian.guenther@esa.int> and Tansu Daylan
<tansu@wustl.edu>. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
third-party attribution.

The batch app has one visible workflow:

- **Automatic multi-target TTV workflow**: Upload a UTF-8 text file containing
  one target name or TIC ID per line. All available TESS cadences are queried;
  joined Diamante products are excluded, while other readable products are
  downloaded and prepared. Data uncertainties are accepted when supplied, or
  estimated from one-duration Wōtan residuals with cval 3.5 and 4-sigma
  clipping. Planet b parameters are retrieved from ExoFOP, the first transit
  within 10% of the maximum cutout point count is refined automatically, and
  all suitable transit timing MCMCs are run automatically. Each target gets
  its own `data/prepared/<target>/plots` and `results` directories.

The earlier single-target fitting, TTV model, and 3D system capabilities remain
available in the source code for later development, but are hidden from this
batch app entry point.

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

Third-party components retain their own licences. Allesfitter and several
direct dependencies use the MIT License, while BATMAN and REBOUND use GPLv3.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and links
to the applicable upstream licences.
