import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

# Patch missing modules before importing analysis/collection modules
sys.modules["meme"] = MagicMock()
sys.modules["meme.names"] = MagicMock()
sys.modules["edef"] = MagicMock()
sys.modules["slac_devices"] = MagicMock()
sys.modules["slac_devices.reader"] = MagicMock()
sys.modules["slac_devices.wire"] = MagicMock()
sys.modules["slac_devices.device"] = MagicMock()

from slac_tools.wires.ws_analysis import WireMeasurementAnalysis
from slac_tools.wires.ws_analysis_results import (
    ProfileMeasurement,
    DetectorProfileMeasurement as DetectorMeasurement,
)
from slac_tools.wires.ws_collection import WireMeasurementCollection
from slac_tools.wires.ws_collection_results import (
    WireMeasurementCollectionResult,
    MeasurementMetadata,
)

class MockWire:
    def __init__(self, name="TEST_WIRE", area="TEST_AREA"):
        self.name = name
        self.area = area
        self.x_range = (100, 200)
        self.y_range = (150, 250)
        self.u_range = (200, 300)
        self.use_x_wire = True
        self.use_y_wire = True
        self.use_u_wire = True
        self.metadata = MagicMock()
        self.metadata.detectors = ["LBLM:TEST_AREA", "PMT:TEST_AREA"]


class TestWireMeasurementAnalysisMethods(unittest.TestCase):
    def setUp(self):
        self.analysis = MagicMock(spec=WireMeasurementAnalysis)
        self.mock_wire = MockWire()

        self.analysis.my_wire = self.mock_wire
        self.analysis.beam_profile_device = self.mock_wire
        self.analysis.collection_result = None

        self.analysis._get_profile_range = WireMeasurementAnalysis._get_profile_range.__get__(
            self.analysis, WireMeasurementAnalysis
        )
        self.analysis._mono_array = WireMeasurementAnalysis._mono_array.__get__(
            self.analysis, WireMeasurementAnalysis
        )
        self.analysis._get_indices_in_range = (
            WireMeasurementAnalysis._get_indices_in_range.__get__(
                self.analysis, WireMeasurementAnalysis
            )
        )
        self.analysis._check_range_in_position = (
            WireMeasurementAnalysis._check_range_in_position.__get__(
                self.analysis, WireMeasurementAnalysis
            )
        )
        self.analysis._create_detector_measurement = (
            WireMeasurementAnalysis._create_detector_measurement.__get__(
                self.analysis, WireMeasurementAnalysis
            )
        )
        self.analysis._create_profile_measurement = (
            WireMeasurementAnalysis._create_profile_measurement.__get__(
                self.analysis, WireMeasurementAnalysis
            )
        )
        self.analysis._get_monotonic_indices = (
            WireMeasurementAnalysis._get_monotonic_indices.__get__(
                self.analysis, WireMeasurementAnalysis
            )
        )

    def test_mono_array_monotonic_increasing(self):
        pos = np.array([1, 2, 3, 4, 5])
        result = self.analysis._mono_array(pos)
        self.assertTrue(np.all(result))

    def test_mono_array_with_reversal(self):
        pos = np.array([1, 2, 3, 2, 4, 5])
        result = self.analysis._mono_array(pos)
        expected = np.array([True, True, True, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_mono_array_flat(self):
        pos = np.array([1, 1, 1, 1, 1])
        result = self.analysis._mono_array(pos)
        self.assertTrue(np.all(result))

    def test_get_profile_range(self):
        

        self.analysis.collection_result = WireMeasurementCollectionResult(
            raw_data={},
            metadata=MeasurementMetadata(
                wire_name=self.mock_wire.name,
                area=self.mock_wire.area,
                beampath="CU_HXR",
                detectors=[],
                default_detector="",
                scan_ranges={
                    "x": self.mock_wire.x_range,
                    "y": self.mock_wire.y_range,
                    "u": self.mock_wire.u_range,
                },
                timestamp=datetime.now(),
                active_profiles=["x", "y", "u"],
                install_angle=0.0,
                notes=None,
            ),
        )

        self.assertEqual(self.analysis._get_profile_range("x"), self.mock_wire.x_range)
        self.assertEqual(self.analysis._get_profile_range("y"), self.mock_wire.y_range)
        self.assertEqual(self.analysis._get_profile_range("u"), self.mock_wire.u_range)

    def test_get_indices_in_range(self):
        position_data = np.array([100, 150, 200, 250, 300, 350])
        indices = self.analysis._get_indices_in_range(position_data, 150, 250)
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(indices, expected)

    def test_get_indices_in_range_empty(self):
        position_data = np.array([100, 150, 200])
        indices = self.analysis._get_indices_in_range(position_data, 300, 400)
        self.assertEqual(len(indices), 0)

    def test_check_range_in_position_valid(self):
        position_data = np.array([100, 150, 200, 250, 300])
        self.analysis._check_range_in_position(position_data, "x", (100, 250))

    def test_check_range_in_position_insufficient(self):
        position_data = np.array([100, 150, 200])
        with self.assertRaises(RuntimeError):
            self.analysis._check_range_in_position(position_data, "x", (250, 400))

    def test_create_detector_measurement(self):
        self.analysis._get_units_for_device = MagicMock(return_value="counts")

        data = np.array([100, 200, 300])
        measurement = self.analysis._create_detector_measurement("LBLM", data)

        self.assertIsInstance(measurement, DetectorMeasurement)
        np.testing.assert_array_equal(measurement.values, data)
        self.assertEqual(measurement.units, "counts")
        self.assertEqual(measurement.label, "LBLM")

    def test_create_profile_measurement(self):
        positions = np.array([100, 150, 200])
        detectors = {
            "LBLM": DetectorMeasurement(values=np.array([1, 2, 3]), units="counts"),
            "PMT": DetectorMeasurement(values=np.array([4, 5, 6]), units="counts"),
        }
        indices = np.array([0, 1, 2])

        measurement = self.analysis._create_profile_measurement(
            positions, detectors, indices
        )

        self.assertIsInstance(measurement, ProfileMeasurement)
        np.testing.assert_array_equal(measurement.positions, positions)
        self.assertEqual(len(measurement.detectors), 2)
        np.testing.assert_array_equal(measurement.profile_indices, indices)

    def test_get_monotonic_indices(self):
        position_data = np.array([100, 110, 120, 115, 130, 140])
        indices = np.array([0, 1, 2, 3, 4, 5])
        mono_indices = self.analysis._get_monotonic_indices(position_data, indices)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(mono_indices, expected)


class TestWireMeasurementAnalysisIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_wire = MockWire()
        self.mock_wire.metadata.detectors = ["LBLM:TEST_AREA"]

        self.collection = MagicMock(spec=WireMeasurementCollection)
        self.collection.beam_profile_device = self.mock_wire
        self.collection.my_wire = self.mock_wire
        self.collection.beampath = "CU_HXR"
        self.collection.devices = {self.mock_wire.name: self.mock_wire}
        self.collection.detectors = [
            d.split(":")[0] for d in self.mock_wire.metadata.detectors
        ]
        self.collection.data = None
        self.collection.profiles = None

        self.collection.get_profile_range_indices = (
            WireMeasurementAnalysis.get_profile_range_indices.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._active_profiles = (
            WireMeasurementCollection._active_profiles.__get__(
                self.collection, WireMeasurementCollection
            )
        )
        self.collection._get_profile_range = (
            WireMeasurementAnalysis._get_profile_range.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._validate_position_data = (
            WireMeasurementCollection._validate_position_data.__get__(
                self.collection, WireMeasurementCollection
            )
        )
        self.collection._check_range_in_position = (
            WireMeasurementAnalysis._check_range_in_position.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._get_indices_in_range = (
            WireMeasurementAnalysis._get_indices_in_range.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._get_monotonic_indices = (
            WireMeasurementAnalysis._get_monotonic_indices.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._mono_array = WireMeasurementAnalysis._mono_array.__get__(
            self.collection, WireMeasurementAnalysis
        )

    def test_get_profile_range_indices_workflow(self):
        position_data = np.linspace(100, 350, 500)
        self.collection.data = {self.mock_wire.name: position_data}

        self.collection.collection_result = WireMeasurementCollectionResult(
            raw_data={self.mock_wire.name: position_data},
            metadata=MeasurementMetadata(
                wire_name=self.mock_wire.name,
                area=self.mock_wire.area,
                beampath="CU_HXR",
                detectors=[],
                default_detector="",
                scan_ranges={"x": (100, 200), "y": (200, 300), "u": (300, 400)},
                timestamp=datetime.now(),
                active_profiles=["x", "y", "u"],
                install_angle=0.0,
                notes=None,
            ),
        )

        profile_indices = self.collection.get_profile_range_indices()

        self.assertIn("x", profile_indices)
        self.assertIn("y", profile_indices)
        self.assertIn("u", profile_indices)

        for profile in ["x", "y", "u"]:
            self.assertGreater(len(profile_indices[profile]), 0)

    def test_organize_data_by_profile_integration(self):
        self.collection.organize_data_by_profile = (
            WireMeasurementAnalysis.organize_data_by_profile.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._create_detector_measurement = (
            WireMeasurementAnalysis._create_detector_measurement.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._create_profile_measurement = (
            WireMeasurementAnalysis._create_profile_measurement.__get__(
                self.collection, WireMeasurementAnalysis
            )
        )
        self.collection._get_units_for_device = (
            WireMeasurementCollection._get_units_for_device.__get__(
                self.collection, WireMeasurementCollection
            )
        )

        mock_detector = MagicMock()
        mock_detector.measure = MagicMock(return_value=np.arange(3))
        self.collection.devices = {
            self.mock_wire.name: self.mock_wire,
            "LBLM": mock_detector,
        }

        self.collection.data = {
            self.mock_wire.name: np.array([100, 150, 200]),
            "LBLM": np.array([1, 2, 3]),
        }

        self.collection.collection_result = WireMeasurementCollectionResult(
            raw_data={
                self.mock_wire.name: np.array([100, 150, 200]),
                "LBLM": np.array([1, 2, 3]),
            },
            metadata=MeasurementMetadata(
                wire_name=self.mock_wire.name,
                area=self.mock_wire.area,
                beampath="CU_HXR",
                detectors=["LBLM"],
                default_detector="LBLM",
                scan_ranges={"x": (100, 200)},
                timestamp=datetime.now(),
                active_profiles=["x"],
                install_angle=0.0,
                notes=None,
            ),
        )

        profile_indices = {"x": np.array([0, 1, 2])}

        profiles = self.collection.organize_data_by_profile(profile_indices)

        self.assertIn("x", profiles)
        self.assertIsInstance(profiles["x"], ProfileMeasurement)
        self.assertIn("LBLM", profiles["x"].detectors)
