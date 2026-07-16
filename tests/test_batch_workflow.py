"""Tests for the automatic multi-target simplified workflow."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ttv_fitter.batch_workflow import (
    anchor_epoch_to_data,
    filter_product_filenames,
    normalize_time_to_btjd,
    parse_target_list,
    planet_b_seed,
    select_reference_cutout,
)


class BatchWorkflowTests(unittest.TestCase):
    def test_target_list_parser_ignores_comments_blanks_and_duplicates(self) -> None:
        self.assertEqual(parse_target_list(" WASP-85\n\n# ignore\nTIC 123\nWASP-85\n"), ["WASP-85", "TIC 123"])

    def test_product_filter_excludes_diamante_case_insensitively(self) -> None:
        products = pd.DataFrame({"productFilename": ["a.fits", "hlsp_diamante_tess_lightcurve.fits", "A.fits", None]})
        self.assertEqual(filter_product_filenames(products), ["A.fits", "a.fits"])

    def test_planet_b_seed_aligns_bjd_to_btjd_data(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "Planet Name": "b",
                    "Epoch (BJD)": 2459000.25,
                    "Period (days)": 2.5,
                    "Duration (hours)": 2.0,
                    "Depth (ppm)": 10000,
                }
            ]
        )
        photometry = pd.DataFrame({"time": [1999.0, 2000.0], "flux": [1.0, 1.0]})
        seed = planet_b_seed(matches, photometry)
        self.assertAlmostEqual(seed["t0"], 2000.25)
        self.assertAlmostEqual(seed["period"], 2.5)
        self.assertAlmostEqual(seed["duration_hours"], 2.0)

    def test_reference_cutout_is_first_within_ten_percent_of_maximum(self) -> None:
        # The first event has 8 points; the next has 10, so the next event is
        # the first one meeting the 90% point-count rule.
        times = np.array([9.82, 9.87, 9.92, 9.97, 10.02, 10.07, 10.12, 10.17, 12.31, 12.35, 12.39, 12.43, 12.47, 12.51, 12.55, 12.59, 12.63, 12.67])
        photometry = pd.DataFrame({"time": times, "flux": np.ones(times.size), "flux_err": np.full(times.size, 0.001)})
        seed = {"t0": 10.0, "period": 2.5, "duration_hours": 2.0}
        reference, cutout = select_reference_cutout(photometry, seed)
        self.assertEqual(reference["epoch"], 1)
        self.assertEqual(reference["points"], len(cutout))

    def test_epoch_is_folded_into_observed_data_span(self) -> None:
        photometry = pd.DataFrame({"time": [2539.0, 2541.0]})
        self.assertAlmostEqual(anchor_epoch_to_data(3262.6699, 2.6557, photometry), 2540.3195, places=3)

    def test_jd_times_are_converted_to_btjd(self) -> None:
        frame = pd.DataFrame({"time": [2459540.0, 2459541.0], "flux": [1.0, 1.0]})
        converted, scale = normalize_time_to_btjd(frame)
        self.assertEqual(scale, "BJD/JD → BTJD")
        self.assertTrue(np.allclose(converted["time"], [2540.0, 2541.0]))


if __name__ == "__main__":
    unittest.main()
