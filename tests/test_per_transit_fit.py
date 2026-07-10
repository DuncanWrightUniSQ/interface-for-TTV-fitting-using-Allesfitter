"""Regression tests for the per-transit T0 fit workflow."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
import tempfile
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch
import re

import numpy as np
import pandas as pd

from ttv_fitter.alles_workflows import (
    PROJECT_ROOT,
    _available_wotan_method,
    _clean_fit_sector_table,
    _default_sector_directory,
    _extract_tic_from_identifiers,
    _exofop_ephemerides_from_matches,
    _exofop_planet_rows_from_ephemerides,
    _exofop_single_transit_seed,
    _data_uncertainty_medians_by_sector,
    _flattening_status_table_with_method,
    _find_tic_from_local_toi_list,
    _find_tic_from_simbad_identifiers,
    _mask_from_found_transits,
    _products_from_observation_data_urls,
    _refine_transits_in_sector,
    _repair_host_prior_parameter_names,
    _selected_product_local_paths,
    _simple_display_stride,
    _simple_plot_stride,
    _single_transit_fit_figure,
    _single_transit_plot_frame,
    _statsmodels_available,
    _stitch_flattened_sectors_with_method,
    _transit_mask_for_ephemerides,
    _wotan_trend_with_method,
)
from ttv_fitter.alles_workflows import ensure_allesfitter_workflows_available, patch_mast_light_curve_filter
from ttv_fitter.dynamics import compare_model_to_timings, rebound_available, run_physical_ttv_model
from ttv_fitter.fitting import fit_cutout_t0, fit_limb_darkened_single_transit, run_cutout_t0_mcmc
from ttv_fitter.models import coerce_planet_table, limb_darkened_transit_model
from ttv_fitter.plots import cutout_fit_figure, oc_figure
from ttv_fitter.ttv import fit_sinusoidal_ttv


class PerTransitFitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.period = 2.0
        self.expected_tmid = 10.0
        self.true_tmid = 10.012
        self.radius_ratio = 0.08
        self.impact = 0.35
        self.a_over_rstar = 12.0
        self.duration_hours = 2.0
        self.u1 = 0.4
        self.u2 = 0.2
        self.baseline_offset = 0.003
        time = np.linspace(9.85, 10.15, 80)
        flux = limb_darkened_transit_model(
            time,
            self.period,
            self.true_tmid,
            self.radius_ratio,
            self.impact,
            self.duration_hours,
            self.u1,
            self.u2,
            baseline_offset=self.baseline_offset,
            a_over_rstar=self.a_over_rstar,
        )
        self.photometry = pd.DataFrame(
            {
                "time": time,
                "flux": flux,
                "flux_err": np.full(time.size, 0.001),
            }
        )

    def test_app_imports_cutout_plot_helper(self) -> None:
        import streamlit_app

        self.assertTrue(callable(streamlit_app.cutout_fit_figure))

    def test_vendored_allesfitter_helpers_import_from_repo(self) -> None:
        ensure_allesfitter_workflows_available()

        import app
        from app import mast, photometry, plots

        self.assertEqual(Path(app.__file__).resolve().parents[1], PROJECT_ROOT)
        self.assertTrue(callable(mast.query_tess_photometry))
        self.assertTrue(callable(photometry.read_photometry_file))
        self.assertTrue(callable(plots.prepared_photometry_preview))
        self.assertNotIn("Allesfitter_work", str(PROJECT_ROOT))

    def test_least_squares_cutout_fit_does_not_emit_uncertainty(self) -> None:
        result = fit_cutout_t0(
            self.photometry,
            self.period,
            self.expected_tmid,
            self.radius_ratio,
            self.impact,
            self.a_over_rstar,
            self.duration_hours,
            self.u1,
            self.u2,
            self.baseline_offset,
            search_half_width_days=0.05,
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.params["tmid"], self.true_tmid, places=6)
        self.assertAlmostEqual(result.params["baseline_offset"], self.baseline_offset, places=6)
        self.assertNotIn("tmid_err", result.params)

    def test_least_squares_multistart_tiles_t0_range_by_duration(self) -> None:
        starts: list[float] = []

        def fake_least_squares(residual, x0, bounds, max_nfev):
            starts.append(float(x0[0]))
            return SimpleNamespace(x=np.asarray(x0, dtype=float), success=True, message="ok")

        with patch("ttv_fitter.fitting.least_squares", side_effect=fake_least_squares):
            result = fit_cutout_t0(
                self.photometry,
                self.period,
                self.expected_tmid,
                self.radius_ratio,
                self.impact,
                self.a_over_rstar,
                duration_hours=2.0,
                limb_darkening_u1=self.u1,
                limb_darkening_u2=self.u2,
                baseline_offset=self.baseline_offset,
                search_half_width_days=5.0 / 24.0,
                use_t0_multistart=True,
            )

        expected_offsets_hours = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        observed_offsets_hours = (np.asarray(starts) - self.expected_tmid) * 24.0
        self.assertTrue(result.success)
        self.assertEqual(len(starts), 5)
        np.testing.assert_allclose(observed_offsets_hours, expected_offsets_hours, atol=1e-10)
        self.assertEqual(result.params["n_t0_startpoints"], 5.0)

    def test_selected_cutout_plot_and_oc_plot_render_without_uncertainties(self) -> None:
        result = fit_cutout_t0(
            self.photometry,
            self.period,
            self.expected_tmid,
            self.radius_ratio,
            self.impact,
            self.a_over_rstar,
            self.duration_hours,
            self.u1,
            self.u2,
            self.baseline_offset,
            search_half_width_days=0.05,
        )
        fit = {
            "start": float(self.photometry["time"].min()),
            "end": float(self.photometry["time"].max()),
            "period": self.period,
            "tmid": result.params["tmid"],
            "radius_ratio": self.radius_ratio,
            "impact": self.impact,
            "a_over_rstar": self.a_over_rstar,
            "duration_hours": self.duration_hours,
            "limb_darkening_u1": self.u1,
            "limb_darkening_u2": self.u2,
            "baseline_offset": result.params["baseline_offset"],
        }

        cutout_fig = cutout_fit_figure(self.photometry, fit)
        timings = pd.DataFrame([{"planet": "b", "epoch": 0, "tmid": result.params["tmid"]}])
        oc_fig = oc_figure(timings, self.expected_tmid, self.period)

        self.assertEqual(len(cutout_fig.data), 2)
        self.assertEqual(len(oc_fig.data), 1)
        self.assertFalse(oc_fig.data[0].error_y.visible)

    def test_t0_only_mcmc_returns_trimmed_uncertainties(self) -> None:
        result = run_cutout_t0_mcmc(
            self.photometry,
            self.period,
            self.expected_tmid,
            self.radius_ratio,
            self.impact,
            self.a_over_rstar,
            self.duration_hours,
            self.u1,
            self.u2,
            self.baseline_offset,
            search_half_width_days=0.05,
            nwalkers=6,
            nsteps=80,
            burn=40,
        )

        self.assertTrue(result.success, result.message)
        self.assertIn("tmid_err", result.params)
        self.assertGreater(result.params["tmid_err"], 0)
        self.assertEqual(len(result.samples), 6 * 40)

    def test_t0_only_mcmc_initializes_walkers_across_search_range(self) -> None:
        calls: dict[str, np.ndarray | int] = {}

        class FakeSampler:
            def __init__(self, nwalkers, ndim, log_prob):
                self.nwalkers = nwalkers
                self.ndim = ndim
                self.log_prob = log_prob

            def run_mcmc(self, p0, nsteps, progress=False):
                calls["p0"] = np.asarray(p0, dtype=float)
                calls["nsteps"] = int(nsteps)
                self.log_prob(calls["p0"][0])

            def get_chain(self, discard=0, flat=False):
                p0 = np.asarray(calls["p0"], dtype=float)
                chain = np.repeat(p0[None, :, :], 10, axis=0)
                if flat:
                    return chain.reshape(-1, p0.shape[1])
                return chain

        emcee_module = types.ModuleType("emcee")
        emcee_module.EnsembleSampler = FakeSampler

        with patch.dict(sys.modules, {"emcee": emcee_module}):
            result = run_cutout_t0_mcmc(
                self.photometry,
                self.period,
                self.expected_tmid,
                self.radius_ratio,
                self.impact,
                self.a_over_rstar,
                self.duration_hours,
                self.u1,
                self.u2,
                self.baseline_offset,
                search_half_width_days=0.5,
                nwalkers=8,
                nsteps=20,
                burn=5,
            )

        p0 = np.asarray(calls["p0"], dtype=float)[:, 0]
        self.assertTrue(result.success, result.message)
        self.assertGreater(np.ptp(p0), 0.75)
        self.assertGreaterEqual(np.min(p0), self.expected_tmid - 0.5)
        self.assertLessEqual(np.max(p0), self.expected_tmid + 0.5)
        self.assertAlmostEqual(result.params["prior_lower"], self.expected_tmid - 0.5)
        self.assertAlmostEqual(result.params["prior_upper"], self.expected_tmid + 0.5)

    def test_t0_only_mcmc_trims_nonconverged_walker_samples(self) -> None:
        class FakeSampler:
            def __init__(self, nwalkers, ndim, log_prob):
                self.nwalkers = nwalkers
                self.ndim = ndim
                self.log_prob = log_prob

            def run_mcmc(self, p0, nsteps, progress=False):
                self.log_prob(np.asarray(p0, dtype=float)[0])

            def get_chain(self, discard=0, flat=False):
                steps = np.linspace(-1e-4, 1e-4, 20)
                good_offsets = np.array([0.0, 1e-5, -1e-5, 2e-5, -2e-5])
                good = self_t0 + steps[:, None] + good_offsets[None, :]
                bad = np.full((steps.size, 1), self_t0 - 0.45)
                chain = np.hstack([good, bad])[:, :, None]
                if flat:
                    return chain.reshape(-1, 1)
                return chain

        self_t0 = self.expected_tmid
        emcee_module = types.ModuleType("emcee")
        emcee_module.EnsembleSampler = FakeSampler

        with patch.dict(sys.modules, {"emcee": emcee_module}):
            result = run_cutout_t0_mcmc(
                self.photometry,
                self.period,
                self.expected_tmid,
                self.radius_ratio,
                self.impact,
                self.a_over_rstar,
                self.duration_hours,
                self.u1,
                self.u2,
                self.baseline_offset,
                search_half_width_days=0.5,
                nwalkers=6,
                nsteps=30,
                burn=5,
                trim_nonconverged_walkers=True,
            )

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.params["nwalkers_total"], 6.0)
        self.assertEqual(result.params["nwalkers_used"], 5.0)
        self.assertEqual(result.params["nwalkers_trimmed"], 1.0)
        self.assertEqual(result.params["walker_trim_applied"], 1.0)
        self.assertEqual(len(result.samples), 5 * 20)
        self.assertAlmostEqual(result.params["tmid"], self.expected_tmid, places=4)
        self.assertLess(result.params["tmid_err"], 0.001)

    def test_ttv_tab_prefers_latest_mcmc_timings(self) -> None:
        import streamlit as st
        import streamlit_app

        st.session_state.cutout_mcmc_timings = pd.DataFrame(
            [{"planet": "b", "epoch": 0, "tmid": 12.0, "tmid_err": 0.0002}]
        )
        st.session_state.timings = pd.DataFrame(
            [{"planet": "b", "epoch": 0, "tmid": 10.0}]
        )

        timings, source = streamlit_app._latest_timing_table_for_ttv()

        self.assertEqual(source, "latest per-transit MCMC T0 fits")
        self.assertAlmostEqual(float(timings.iloc[0]["tmid"]), 12.0)
        self.assertIn("tmid_err", timings.columns)

    def test_linear_ls_parameters_populate_ttv_planet_table(self) -> None:
        import streamlit as st
        import streamlit_app

        st.session_state.photometry = pd.DataFrame({"time": [2770.0, 2771.0]})
        st.session_state.planets = pd.DataFrame()
        st.session_state.ttv_host_mass = 1.0
        st.session_state.ttv_host_radius = 1.0
        st.session_state.phot_fit_latest_edited_priors = pd.DataFrame(
            [
                {"planet": "host", "parameter": "stellar_mass", "value": 0.62},
                {"planet": "host", "parameter": "stellar_radius", "value": 0.64},
            ]
        )
        st.session_state.phot_fit_generated_params = pd.DataFrame(
            [
                {"name": "b_epoch", "value": 2459770.026},
                {"name": "b_period", "value": 3.067897046},
                {"name": "b_rr", "value": 0.1681392515},
                {"name": "b_rsuma", "value": 0.083},
                {"name": "b_cosi", "value": 0.0194},
            ]
        )

        planets, mass, radius = streamlit_app._linear_fit_planet_table_from_ls()

        self.assertFalse(planets.empty)
        self.assertAlmostEqual(mass, 0.62)
        self.assertAlmostEqual(radius, 0.64)
        self.assertAlmostEqual(float(planets.iloc[0]["period"]), 3.067897046)
        self.assertAlmostEqual(float(planets.iloc[0]["t0"]), 2770.026)
        self.assertGreater(float(planets.iloc[0]["a_over_rstar"]), 1.0)

    def test_sinusoidal_ttv_uses_timing_uncertainties(self) -> None:
        timings = pd.DataFrame(
            [
                {"epoch": 0, "tmid": 10.0, "tmid_err": 0.0001},
                {"epoch": 1, "tmid": 12.0, "tmid_err": 0.0001},
                {"epoch": 2, "tmid": 14.1, "tmid_err": 0.2},
                {"epoch": 3, "tmid": 16.0, "tmid_err": 0.0001},
            ]
        )

        params, model = fit_sinusoidal_ttv(timings, 10.0, 2.0)

        self.assertIn("amplitude_minutes", params)
        self.assertIn("ttv_model_minutes", model.columns)
        self.assertTrue(np.isfinite(model["ttv_model_minutes"]).all())

    def test_simbad_identifier_fallback_extracts_tic_id(self) -> None:
        identifiers = pd.Series(["Gaia DR3 123", "TIC 243921117", "WASP-80"])

        self.assertEqual(_extract_tic_from_identifiers(identifiers), "243921117")

    def test_direct_simbad_identifier_query_parses_csv(self) -> None:
        class Response:
            text = 'id\n"TIC 243921117"\n"WASP-80"\n'

            def raise_for_status(self) -> None:
                return None

        with patch("requests.post", return_value=Response()) as post:
            tic = _find_tic_from_simbad_identifiers("WASP-80")

        self.assertEqual(tic, "243921117")
        query = post.call_args.kwargs["data"]["QUERY"]
        self.assertIn("JOIN ident AS q", query)
        self.assertIn("WASP-80", query)

    def test_mast_query_wrapper_uses_direct_simbad_fallback_tic(self) -> None:
        @dataclass(frozen=True)
        class Result:
            target: str
            resolved_target: str
            observations: pd.DataFrame
            products: pd.DataFrame

        calls = []

        def original_query(target: str, cadence: str = "2 min") -> Result:
            calls.append(target)
            if target != "272213425":
                raise RuntimeError("old SIMBAD capabilities failure")
            return Result(target, target, pd.DataFrame(), pd.DataFrame())

        observation_table = pd.DataFrame([{"obs_id": "obs1", "obs_collection": "TESS", "dataproduct_type": "timeseries"}])
        mast_module = SimpleNamespace(
            query_tess_photometry=original_query,
            download_tess_products=lambda *args, **kwargs: [],
            normalize_target_name=lambda value: " ".join(str(value).strip().split()),
            TIC_RE=re.compile(r"^(?:TIC\s*)?(\d+)$", re.IGNORECASE),
            MastQueryResult=Result,
            Observations=SimpleNamespace(_portal_api_connection=SimpleNamespace(TIMEOUT=10)),
            _query_by_target_name=lambda target: observation_table if target == "272213425" else pd.DataFrame(),
            _to_dataframe=lambda table: table.copy(),
            _filter_observations=lambda observations, cadence: observations.assign(_mast_row=observations.index),
            _product_summary=lambda table: pd.DataFrame(
                [{"obs_id": "obs1", "productFilename": "file-lc.fits", "description": "Light curves", "productSubGroupDescription": "LC"}]
            ),
        )
        photometry_import_module = SimpleNamespace()

        with patch("ttv_fitter.alles_workflows._find_tic_from_local_toi_list", return_value="272213425"), patch(
            "ttv_fitter.alles_workflows._find_tic_from_simbad_identifiers", return_value=""
        ), patch(
            "ttv_fitter.alles_workflows._product_summary_with_timeout",
            return_value=pd.DataFrame(
                [{"obs_id": "obs1", "productFilename": "file-lc.fits", "description": "Light curves", "productSubGroupDescription": "LC"}]
            ),
        ):
            patch_mast_light_curve_filter(mast_module, photometry_import_module)
            result = mast_module.query_tess_photometry("TOI-1904", "All available cadences")

        self.assertEqual(calls, [])
        self.assertEqual(result.target, "TOI-1904")
        self.assertEqual(result.resolved_target, "272213425")

    def test_local_toi_list_resolves_toi_name(self) -> None:
        self.assertEqual(_find_tic_from_local_toi_list("TOI-1904"), "272213425")

    def test_mast_products_can_be_built_from_observation_data_urls(self) -> None:
        observations = pd.DataFrame(
            [
                {
                    "obs_id": "tess2021091135823-s0037-0000000272213425-0208-a_fast",
                    "dataURL": "mast:TESS/product/tess2021091135823-s0037-0000000272213425-0208-a_fast-lc.fits",
                },
                {
                    "obs_id": "hlsp_qlp_tess_ffi_s0037-0000000272213425_tess_v01_llc",
                    "dataURL": "mast:HLSP/qlp/s0037/0000/0002/7221/3425/hlsp_qlp_tess_ffi_s0037-0000000272213425_tess_v01_llc.fits",
                },
                {
                    "obs_id": "tess2021091135823-s0037-0000000272213425-0208-dvt",
                    "dataURL": "mast:TESS/product/tess2021091135823-s0037-0000000272213425-0208-dvt.fits",
                },
            ]
        )

        products = _products_from_observation_data_urls(observations)

        self.assertEqual(len(products), 2)
        self.assertIn("tess2021091135823-s0037-0000000272213425-0208-a_fast-lc.fits", set(products["productFilename"]))
        self.assertIn("hlsp_qlp_tess_ffi_s0037-0000000272213425_tess_v01_llc.fits", set(products["productFilename"]))
        self.assertEqual(products.loc[0, "dataURI"], observations.loc[0, "dataURL"])
        self.assertNotIn("dvt", " ".join(products["productFilename"].astype(str)).lower())

    def test_selected_product_local_paths_marks_existing_files(self) -> None:
        products = pd.DataFrame(
            [
                {"productFilename": "first-lc.fits"},
                {"productFilename": "second-lc.fits"},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "first-lc.fits").write_text("cached", encoding="utf-8")

            status = _selected_product_local_paths(products, ["first-lc.fits", "second-lc.fits"], tmp)

        self.assertEqual(len(status), 2)
        self.assertEqual(int(status["already_downloaded"].sum()), 1)
        self.assertTrue(status.loc[status["productFilename"] == "first-lc.fits", "already_downloaded"].iloc[0])
        self.assertFalse(status.loc[status["productFilename"] == "second-lc.fits", "already_downloaded"].iloc[0])

    def test_data_uncertainty_medians_skip_sectors_without_valid_errors(self) -> None:
        sector_frames = {
            "s001": pd.DataFrame({"flux_err": [0.001, 0.003, np.nan]}),
            "s002": pd.DataFrame({"flux_err": [np.nan, -1.0, 0.0]}),
            "s003": pd.DataFrame({"flux": [1.0, 1.01]}),
        }

        medians = _data_uncertainty_medians_by_sector(sector_frames, lambda frame: frame.copy())

        self.assertEqual(set(medians), {"s001"})
        self.assertAlmostEqual(medians["s001"], 0.002)

    def test_flattening_status_table_records_wotan_method_per_sector(self) -> None:
        sector_frames = {
            "s001": pd.DataFrame({"sector": ["S1"], "source_file": ["one.fits"]}),
            "s002": pd.DataFrame({"sector": ["S2"], "source_file": ["two.fits"]}),
        }
        sector_flattening = {
            "s001": {"window_length": 0.8, "method": "biweight", "cval": 4.0, "high_outliers": 1},
            "s002": {"window_length": 1.2, "method": "huber", "cval": 2.5, "high_outliers": 2},
        }

        status = _flattening_status_table_with_method(sector_frames, sector_flattening)

        self.assertEqual(status.loc[status["sector"] == "S1", "wotan_method"].iloc[0], "biweight")
        self.assertEqual(status.loc[status["sector"] == "S2", "wotan_method"].iloc[0], "huber")
        self.assertAlmostEqual(status.loc[status["sector"] == "S2", "wotan_window_days"].iloc[0], 1.2)
        self.assertAlmostEqual(status.loc[status["sector"] == "S2", "wotan_cval"].iloc[0], 2.5)

    def test_wotan_trend_passes_robust_cval_to_flatten(self) -> None:
        calls: dict[str, float] = {}

        def fake_flatten(time, flux, **kwargs):
            calls.update(kwargs)
            return flux, np.ones_like(flux)

        wotan_module = types.ModuleType("wotan")
        wotan_module.flatten = fake_flatten
        frame = pd.DataFrame({"time": np.arange(5.0), "flux": np.ones(5)})
        photometry_import = SimpleNamespace(normalized_sector=lambda data: data.copy())

        with patch.dict(sys.modules, {"wotan": wotan_module}):
            trend = _wotan_trend_with_method(photometry_import, frame, 0.5, "biweight", cval=2.0)

        self.assertEqual(len(trend), len(frame))
        self.assertEqual(calls["method"], "biweight")
        self.assertAlmostEqual(calls["cval"], 2.0)

    def test_large_plot_stride_uses_simple_thinning(self) -> None:
        self.assertEqual(_simple_plot_stride(20_000), 1)
        self.assertEqual(_simple_plot_stride(30_000), 2)
        self.assertEqual(_simple_plot_stride(60_000), 3)
        self.assertEqual(_simple_plot_stride(90_000), 4)
        self.assertEqual(_simple_display_stride(20_000), 1)
        self.assertEqual(_simple_display_stride(90_000), 5)

    def test_single_transit_plot_thins_data_and_draws_fit_last(self) -> None:
        frame = pd.DataFrame(
            {
                "time": np.linspace(0.0, 1.0, 90_000),
                "flux": np.ones(90_000),
                "flux_err": np.full(90_000, 0.001),
            }
        )
        display, stride = _single_transit_plot_frame(frame)
        fit = {
            "period": 2.0,
            "t0": 0.5,
            "radius_ratio": 0.1,
            "impact": 0.4,
            "duration_hours": 2.0,
            "limb_darkening_u1": 0.5,
            "limb_darkening_u2": 0.1,
            "baseline_offset": 0.0,
            "a_over_rstar": 12.0,
        }

        fig = _single_transit_fit_figure(frame, fit)

        self.assertEqual(stride, 5)
        self.assertEqual(len(display), 18_000)
        self.assertEqual(fig.data[-1].name, "single-transit LS fit")

    def test_linear_fit_sector_loader_adds_missing_outlier_column(self) -> None:
        table = _clean_fit_sector_table(
            pd.DataFrame({"time": [1.0, 2.0], "flux": [1.0, 0.99], "flux_err": [0.001, 0.001]}),
            "sector_1.csv",
        )

        self.assertIn("is_outlier", table.columns)
        self.assertFalse(table["is_outlier"].any())

    def test_default_sector_directory_has_example_when_none_prepared(self) -> None:
        import streamlit as st

        st.session_state.pop("prepared_sector_directory", None)
        st.session_state["exofop_target_query"] = ""

        self.assertEqual(_default_sector_directory(), "data/prepared/TOI-216/sectors")

    def test_cutout_sector_loader_merges_sector_tables_without_outliers(self) -> None:
        import streamlit as st
        import streamlit_app

        tables = {
            "S1": pd.DataFrame(
                {
                    "time": [1.0, 2.0],
                    "flux": [1.0, 0.99],
                    "flux_err": [0.001, 0.001],
                    "is_outlier": [False, True],
                    "sector": ["S1", "S1"],
                }
            ),
            "S2": pd.DataFrame(
                {
                    "time": [3.0],
                    "flux": [1.01],
                    "flux_err": [0.001],
                    "sector": ["S2"],
                }
            ),
        }

        self.assertTrue(streamlit_app._load_cutout_sector_tables(tables, "test sectors"))
        self.assertEqual(len(st.session_state.photometry), 2)
        self.assertEqual(st.session_state["cutout_photometry_source"], "test sectors")
        self.assertEqual(st.session_state.photometry["time"].tolist(), [1.0, 3.0])

    def test_planet_row_cutout_values_include_timing_and_shape(self) -> None:
        import streamlit_app

        values = streamlit_app._planet_row_cutout_values(
            {
                "t0": 1331.25,
                "period": 34.5,
                "radius_ratio": 0.12,
                "impact": 0.4,
                "a_over_rstar": 32.0,
                "duration_hours": 3.8,
            }
        )

        self.assertAlmostEqual(values["t0"], 1331.25)
        self.assertAlmostEqual(values["period"], 34.5)
        self.assertAlmostEqual(values["radius_ratio"], 0.12)
        self.assertAlmostEqual(values["limb_darkening_u1"], 0.5)

    def test_single_transit_parameter_file_parser_reads_download_format(self) -> None:
        import streamlit_app

        text = """
[planet b]
planet: b
t0: 1331.2852
period: 34.5059
radius_ratio: 0.121
impact: 0.000001
duration_hours: 3.8207
a_over_rstar: 53.5949
limb_darkening_u1: 0.5
source: Current Linear fit photometry

[planet c]
t0: 1335.25
period: 17.389
radius_ratio: 0.074
"""

        parsed = streamlit_app._parse_single_transit_parameter_text(text)

        self.assertEqual(set(parsed), {"b", "c"})
        self.assertAlmostEqual(parsed["b"]["t0"], 1331.2852)
        self.assertAlmostEqual(parsed["b"]["limb_darkening_u1"], 0.5)
        self.assertNotIn("source", parsed["b"])
        self.assertAlmostEqual(parsed["c"]["period"], 17.389)

    def test_single_transit_parameter_file_updates_planet_table(self) -> None:
        import streamlit as st
        import streamlit_app

        st.session_state["planets"] = coerce_planet_table(pd.DataFrame([{"name": "b", "period": 3.0, "t0": 0.0}]))

        streamlit_app._update_planets_from_single_transit_file(
            {
                "b": {"t0": 1331.2852, "period": 34.5059, "radius_ratio": 0.121},
                "c": {"t0": 1335.25, "period": 17.389, "radius_ratio": 0.074},
            }
        )

        planets = coerce_planet_table(st.session_state["planets"])
        self.assertEqual(planets["name"].astype(str).tolist(), ["b", "c"])
        self.assertAlmostEqual(float(planets.loc[planets["name"] == "b", "period"].iloc[0]), 34.5059)
        self.assertAlmostEqual(float(planets.loc[planets["name"] == "c", "radius_ratio"].iloc[0]), 0.074)
        self.assertIn("c", st.session_state["phot_fit_single_transit_ls_results_by_planet"])

    def test_host_prior_parameter_names_are_repaired(self) -> None:
        table = pd.DataFrame(
            [
                {"planet": "host", "parameter": np.nan, "value": 0.84},
                {"planet": "host", "parameter": "", "value": 0.80},
                {"planet": "b", "parameter": "T0", "value": 2458331.489},
            ]
        )

        repaired = _repair_host_prior_parameter_names(table)

        self.assertEqual(repaired.loc[0, "parameter"], "stellar_mass")
        self.assertEqual(repaired.loc[1, "parameter"], "stellar_radius")
        self.assertEqual(repaired.loc[2, "parameter"], "T0")

    def test_stitching_reuses_cached_first_pass_without_reflattening(self) -> None:
        cached = pd.DataFrame(
            {
                "time": [2.0, 1.0],
                "flux": [1.0, 0.99],
                "flux_err": [0.001, 0.001],
                "source_file": ["s1.fits", "s1.fits"],
                "sector": ["S1", "S1"],
                "trend": [1.0, 1.0],
                "is_outlier": [False, False],
                "wotan_transit_mask": [False, False],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "first_pass.pkl"
            cached.to_pickle(cache_path)
            photometry_import = SimpleNamespace(normalized_sector=lambda _frame: (_ for _ in ()).throw(AssertionError("should not reflatten")))

            stitched, summary = _stitch_flattened_sectors_with_method(
                photometry_import,
                {"s001": pd.DataFrame({"time": [1.0, 2.0], "flux": [0.99, 1.0], "sector": ["S1", "S1"]})},
                {"s001": 0.001},
                {
                    "s001": {
                        "window_length": 1.2,
                        "method": "biweight",
                        "cval": 3.0,
                        "use_cached_first_pass": True,
                        "first_pass_cache_path": str(cache_path),
                    }
                },
            )

        self.assertEqual(len(stitched), 2)
        self.assertEqual(stitched["time"].tolist(), [1.0, 2.0])
        self.assertAlmostEqual(summary["wotan_cval"].iloc[0], 3.0)

    def test_huber_wotan_method_falls_back_when_statsmodels_is_missing(self) -> None:
        method, warning = _available_wotan_method("huber")
        if _statsmodels_available():
            self.assertEqual(method, "huber")
            self.assertEqual(warning, "")
        else:
            self.assertEqual(method, "biweight")
            self.assertIn("statsmodels", warning)

        time = np.linspace(0.0, 1.0, 80)
        frame = pd.DataFrame(
            {
                "time": time,
                "flux": 1.0 + 0.001 * np.sin(2 * np.pi * time),
            }
        )
        photometry_import = SimpleNamespace(normalized_sector=lambda data: data.copy())

        trend = _wotan_trend_with_method(photometry_import, frame, 0.25, "huber")

        self.assertEqual(len(trend), len(frame))
        self.assertTrue(np.isfinite(trend).any())

    def test_transit_mask_flags_predicted_midpoints_for_wotan(self) -> None:
        frame = pd.DataFrame({"time": np.linspace(9.0, 15.0, 601), "flux": np.ones(601)})
        mask = _transit_mask_for_ephemerides(
            frame,
            [{"t0": 10.0, "period": 2.0, "duration_hours": 2.4}],
            width_durations=2.0,
        )

        self.assertTrue(mask[np.argmin(np.abs(frame["time"].to_numpy() - 10.0))])
        self.assertTrue(mask[np.argmin(np.abs(frame["time"].to_numpy() - 12.0))])
        self.assertFalse(mask[np.argmin(np.abs(frame["time"].to_numpy() - 11.0))])
        self.assertGreater(int(mask.sum()), 0)

    def test_ttv_transit_search_refines_shifted_transits_and_builds_mask(self) -> None:
        time = np.linspace(9.7, 14.3, 1200)
        flux = np.ones_like(time)
        for tmid in [10.03, 12.04, 14.02]:
            flux += (
                limb_darkened_transit_model(
                    time,
                    2.0,
                    tmid,
                    0.08,
                    0.35,
                    2.4,
                    0.5,
                    0.1,
                )
                - 1.0
            )
        first_pass = pd.DataFrame({"time": time, "flux": flux})
        planets = _exofop_planet_rows_from_ephemerides(
            [
                {
                    "planet": "b",
                    "t0": 10.0,
                    "period": 2.0,
                    "duration_hours": 2.4,
                    "radius_ratio": 0.08,
                    "mask_duration_multiplier": 3.0,
                }
            ]
        )

        found = _refine_transits_in_sector(first_pass, planets, search_half_width_days=0.08, grid_step_days=0.01)
        mask = _mask_from_found_transits(first_pass, found)

        self.assertGreaterEqual(len(found), 3)
        first_three = found.sort_values("expected_tmid").head(3)["tmid"].to_numpy()
        np.testing.assert_allclose(first_three, [10.03, 12.04, 14.02], atol=0.012)
        self.assertTrue(mask[np.argmin(np.abs(time - 12.04))])
        self.assertAlmostEqual(float(found["mask_duration_multiplier"].iloc[0]), 3.0)
        self.assertAlmostEqual(float(found["mask_half_width_days"].iloc[0]), 0.15)
        self.assertFalse(mask[np.argmin(np.abs(time - 11.0))])

    def test_exofop_ephemerides_include_all_usable_planet_rows(self) -> None:
        matches = pd.DataFrame(
            [
                {"Planet Name": np.nan, "TOI": "1.01", "Epoch (BJD)": 2457010.0, "Period (days)": 2.0, "Duration (hours)": 2.4},
                {"Planet Name": np.nan, "TOI": "1.02", "Epoch (BJD)": 2457011.0, "Period (days)": 3.0, "Duration (days)": 0.2},
                {"TOI": "bad", "Epoch (BJD)": np.nan, "Period (days)": 4.0, "Duration (hours)": 1.0},
            ]
        )

        ephemerides = _exofop_ephemerides_from_matches(matches)

        self.assertEqual(len(ephemerides), 2)
        self.assertEqual([row["planet"] for row in ephemerides], ["b", "c"])
        self.assertAlmostEqual(ephemerides[0]["period"], 2.0)
        self.assertAlmostEqual(ephemerides[1]["duration_hours"], 4.8)

    def test_limb_darkened_single_transit_fit_recovers_shape_parameters(self) -> None:
        time = np.linspace(9.88, 10.15, 120)
        flux = limb_darkened_transit_model(time, 2.0, 10.012, 0.09, 0.32, 2.6, 0.45, 0.15, baseline_offset=0.002, a_over_rstar=12.0)
        phot = pd.DataFrame({"time": time, "flux": flux, "flux_err": np.full_like(time, 8e-4)})
        start = {
            "t0": 10.0,
            "period": 2.0,
            "radius_ratio": 0.08,
            "impact": 0.4,
            "duration_hours": 2.4,
            "a_over_rstar": 11.0,
            "limb_darkening_u1": 0.5,
            "limb_darkening_u2": 0.1,
            "baseline_offset": 0.0,
        }

        result = fit_limb_darkened_single_transit(
            phot,
            start,
            ["t0", "radius_ratio", "impact", "duration_hours", "a_over_rstar", "limb_darkening_u1", "limb_darkening_u2", "baseline_offset"],
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.params["period"], 2.0, places=8)
        self.assertAlmostEqual(result.params["t0"], 10.012, places=3)
        self.assertAlmostEqual(result.params["radius_ratio"], 0.09, places=2)
        self.assertIn("a_over_rstar", result.params)

    def test_single_transit_fit_with_btjd_window_stays_on_visible_transit(self) -> None:
        rng = np.random.default_rng(42)
        time = np.linspace(1330.0, 1333.0, 800)
        flux = limb_darkened_transit_model(
            time,
            34.5059,
            1331.25,
            0.12,
            0.45,
            3.8,
            0.5,
            0.1,
            a_over_rstar=34.0,
        )
        flux = flux + rng.normal(0.0, 5e-4, len(time))
        phot = pd.DataFrame({"time": time, "flux": flux, "flux_err": np.full_like(time, 8e-4)})
        start = {
            "t0": 1331.4886,
            "period": 34.5059,
            "radius_ratio": 0.12,
            "impact": 0.5,
            "duration_hours": 3.8,
            "a_over_rstar": 34.0,
            "limb_darkening_u1": 0.5,
            "limb_darkening_u2": 0.1,
            "baseline_offset": 0.0,
        }

        result = fit_limb_darkened_single_transit(
            phot,
            start,
            ["t0", "radius_ratio", "impact", "duration_hours", "a_over_rstar", "limb_darkening_u1", "limb_darkening_u2", "baseline_offset"],
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.params["t0"], 1331.25, places=2)
        self.assertGreater(result.params["radius_ratio"], 0.05)
        self.assertLess(result.params["duration_hours"], 10.0)

    def test_exofop_single_transit_seed_aligns_epoch_to_selected_data(self) -> None:
        import streamlit as st

        st.session_state["phot_fit_exofop_matches"] = pd.DataFrame(
            [
                {
                    "Epoch (BJD)": 2459770.026,
                    "Period (days)": 3.067897,
                    "Planet Radius (R_Earth)": 1.17,
                    "Stellar Radius (R_Sun)": 0.62,
                    "Stellar Mass (M_Sun)": 0.62,
                    "Duration (hours)": 2.5,
                }
            ]
        )
        data = pd.DataFrame({"time": np.linspace(2769.8, 2796.2, 50), "flux": np.ones(50)})

        seed = _exofop_single_transit_seed("b", data)

        self.assertLess(seed["t0"], 3000.0)
        self.assertGreater(seed["t0"], 2760.0)
        self.assertAlmostEqual(seed["period"], 3.067897)
        self.assertAlmostEqual(seed["duration_hours"], 2.5)
        self.assertGreater(seed["radius_ratio"], 0.0)

    def test_physical_comparison_reindexes_bjd_t0_to_btjd_model_times(self) -> None:
        model = pd.DataFrame(
            [
                {"planet": "b", "epoch": -800875, "tmid_model": 2770.026},
                {"planet": "b", "epoch": -800874, "tmid_model": 2773.094},
            ]
        )
        observed = pd.DataFrame(
            [
                {"epoch": 0, "tmid": 2770.026, "tmid_err": 0.001},
                {"epoch": 1, "tmid": 2773.094, "tmid_err": 0.001},
            ]
        )

        comparison = compare_model_to_timings(observed, model, "b", 2459770.026, 3.068)

        self.assertEqual(comparison["epoch"].tolist(), [0, 1])
        self.assertTrue(comparison["oc_minutes"].notna().all())

    def test_physical_model_aligns_bjd_t0_before_phase_initialization(self) -> None:
        ok, _message = rebound_available()
        if not ok:
            self.skipTest("REBOUND is not installed")
        planet = pd.DataFrame(
            [
                {
                    "name": "b",
                    "mass_jupiter": 0.0,
                    "period": 3.067897046,
                    "t0": 2459770.026,
                    "radius_ratio": 0.168,
                    "impact": 0.27,
                    "duration_hours": 3.0,
                    "a_over_rstar": 11.0,
                    "inclination_deg": 89.0,
                    "ecc": 0.0,
                    "omega_deg": 90.0,
                    "mean_anomaly_deg": 0.0,
                    "rv_k": 0.0,
                    "color": "#2563eb",
                }
            ]
        )

        result = run_physical_ttv_model(
            planet,
            star_mass_solar=0.62,
            star_radius_solar=0.64,
            reference_time=2769.9,
            start_time=2769.9,
            end_time=2774.0,
            sample_step_days=0.02,
            initialize_from_t0=True,
        )
        comparison = compare_model_to_timings(
            pd.DataFrame([{"epoch": 0, "tmid": 2770.026, "tmid_err": 0.001}]),
            result.model_timings,
            "b",
            2459770.026,
            3.067897046,
        )

        self.assertLess(abs(float(comparison.iloc[0]["ttv_model_minutes"])), 1.0)

    def test_sinusoid_seed_sets_indicative_perturber_phase(self) -> None:
        import streamlit as st
        import streamlit_app

        st.session_state.ttv_params = {
            "amplitude_minutes": 3.0,
            "super_period_epochs": 4.0,
            "phase_rad": 0.0,
        }
        base = pd.DataFrame(
            [
                {
                    "name": "b",
                    "mass_jupiter": 0.003,
                    "period": 2.0,
                    "t0": 10.0,
                    "radius_ratio": 0.1,
                    "impact": 0.5,
                    "duration_hours": 2.0,
                    "a_over_rstar": 10.0,
                    "inclination_deg": 87.0,
                    "ecc": 0.0,
                    "omega_deg": 90.0,
                    "mean_anomaly_deg": 0.0,
                    "rv_k": 0.0,
                    "color": "#2563eb",
                }
            ]
        )

        seeded = streamlit_app._seed_nontransiting_planet_from_sinusoid(base, 1.0, 1.0)

        self.assertAlmostEqual(float(seeded.iloc[0]["t0"]), 12.0)
        self.assertGreater(float(seeded.iloc[0]["period"]), 2.0)


if __name__ == "__main__":
    unittest.main()
