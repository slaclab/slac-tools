import unittest
import tempfile
import os
import numpy as np
import h5py

from slac_tools.wires.ws_analysis_results import (
    WireMeasurementAnalysisResult,
    DetectorFit,
    ProfileMeasurement,
    DetectorProfileMeasurement,
    FitResult,
    load_from_h5,
)
from slac_tools.wires.ws_collection_results import (
    WireMeasurementCollectionResult,
    MeasurementMetadata,
)


def _make_dummy_collection():
    from datetime import datetime

    metadata = MeasurementMetadata(
        wire_name="WIRE",
        area="AREA",
        beampath="BP",
        detectors=["DET1"],
        default_detector="DET1",
        scan_ranges={"x": (0, 1)},
        timestamp=datetime(2025, 1, 1),
        active_profiles=["x"],
        install_angle=0.0,
    )
    raw_data = {"WIRE": np.array([0.0, 0.5, 1.0]), "DET1": np.array([1, 2, 3])}

    return WireMeasurementCollectionResult(
        raw_data=raw_data, metadata=metadata
    )


def _make_dummy_analysis():
    coll = _make_dummy_collection()
    # simple one detector fit
    det_fit = DetectorFit(
        mean=0.5,
        sigma=0.1,
        amplitude=10.0,
        offset=0.0,
        curve=np.array([0.0, 1.0, 0.0]),
        positions=np.array([0.0, 0.5, 1.0]),
    )
    fit_result = {"x": FitResult(detectors={"DET1": det_fit})}
    profiles = {
        "x": ProfileMeasurement(
            positions=np.array([0.0, 0.5, 1.0]),
            profile_indices=np.array([0, 1, 2]),
            detectors={
                "DET1": DetectorProfileMeasurement(
                    values=np.array([1, 2, 3]), units="counts", label="DET1"
                )
            },
        )
    }
    return WireMeasurementAnalysisResult(
        fit_result=fit_result,
        collection_result=coll,
        profiles=profiles,
        rms_sizes=(0.1, 0.2),
        centroids=np.array([0.5]),
        total_intensities=np.array([6]),
        signal_to_noise_ratios=np.array([5]),
        metadata=coll.metadata,
    )


class TestWireMeasurementAnalysisResults(unittest.TestCase):
    def test_save_and_load(self):
        result = _make_dummy_analysis()
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()
        try:
            result.save_to_h5(tmp.name)
            # file should exist
            self.assertTrue(os.path.exists(tmp.name))

            # load back
            loaded = load_from_h5(tmp.name)
            # compare dictionary representations
            self.assertEqual(
                result.fit_result.keys(), loaded.fit_result.keys()
            )
            # verify nested parameters
            orig = result.fit_result["x"].detectors["DET1"]
            new = loaded.fit_result["x"].detectors["DET1"]
            self.assertAlmostEqual(orig.mean, new.mean)
            self.assertAlmostEqual(orig.sigma, new.sigma)
            np.testing.assert_array_equal(orig.curve, new.curve)
            np.testing.assert_array_equal(orig.positions, new.positions)

            # profiles
            np.testing.assert_array_equal(
                result.profiles["x"].positions, loaded.profiles["x"].positions
            )
            np.testing.assert_array_equal(
                result.profiles["x"].detectors["DET1"].values,
                loaded.profiles["x"].detectors["DET1"].values,
            )

            # metadata should round-trip
            self.assertEqual(
                result.metadata.wire_name, loaded.metadata.wire_name
            )

        finally:
            os.remove(tmp.name)

    def test_file_structure(self):
        """Verify the groups/datasets are created as expected."""
        result = _make_dummy_analysis()
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()
        try:
            result.save_to_h5(tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("collection_result", f)
                self.assertIn("analysis", f)
                self.assertIn("fit_result", f["analysis"])
                self.assertIn("profiles", f["analysis"])
        finally:
            os.remove(tmp.name)
