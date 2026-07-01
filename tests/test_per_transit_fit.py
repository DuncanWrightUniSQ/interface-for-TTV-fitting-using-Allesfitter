"""Regression tests for the per-transit T0 fit workflow."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import re

import numpy as np
import pandas as pd

from ttv_fitter.alles_workflows import (
    _available_wotan_method,
    _extract_tic_from_identifiers,
    _exofop_ephemerides_from_matches,
    _exofop_single_transit_seed,
    _data_uncertainty_medians_by_sector,
    _flattening_status_table_with_method,
    _find_tic_from_local_toi_list,
    _find_tic_from_simbad_identifiers,
    _products_from_observation_data_urls,
    _selected_product_local_paths,
    _statsmodels_available,
    _transit_mask_for_ephemerides,
    _wotan_trend_with_method,
)
from ttv_fitter.alles_workflows import patch_mast_light_curve_filter
from ttv_fitter.dynamics import compare_model_to_timings, rebound_available, run_physical_ttv_model
from ttv_fitter.fitting import fit_cutout_t0, fit_limb_darkened_single_transit, run_cutout_t0_mcmc
from ttv_fitter.models import limb_darkened_transit_model
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
            "s001": {"window_length": 0.8, "method": "biweight", "high_outliers": 1},
            "s002": {"window_length": 1.2, "method": "huber", "high_outliers": 2},
        }

        status = _flattening_status_table_with_method(sector_frames, sector_flattening)

        self.assertEqual(status.loc[status["sector"] == "S1", "wotan_method"].iloc[0], "biweight")
        self.assertEqual(status.loc[status["sector"] == "S2", "wotan_method"].iloc[0], "huber")
        self.assertAlmostEqual(status.loc[status["sector"] == "S2", "wotan_window_days"].iloc[0], 1.2)

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

    def test_exofop_ephemerides_include_all_usable_planet_rows(self) -> None:
        matches = pd.DataFrame(
            [
                {"TOI": "1.01", "Epoch (BJD)": 2457010.0, "Period (days)": 2.0, "Duration (hours)": 2.4},
                {"TOI": "1.02", "Epoch (BJD)": 2457011.0, "Period (days)": 3.0, "Duration (days)": 0.2},
                {"TOI": "bad", "Epoch (BJD)": np.nan, "Period (days)": 4.0, "Duration (hours)": 1.0},
            ]
        )

        ephemerides = _exofop_ephemerides_from_matches(matches)

        self.assertEqual(len(ephemerides), 2)
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
