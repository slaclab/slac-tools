import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock

import h5py
import numpy as np

sys.modules["slac_devices"] = MagicMock()
sys.modules["slac_devices.device"] = MagicMock()

from slac_tools.wires.ws_collection_results import (
    MeasurementMetadata,
    MultiWireMeasurementCollectionResult,
    WireMeasurementCollectionResult,
    load_from_h5,
    load_multi_wire_from_h5,
)


def _make_metadata(wire_name: str) -> MeasurementMetadata:
    return MeasurementMetadata(
        wire_name=wire_name,
        area="TEST_AREA",
        beampath="CU_HXR",
        detectors=["LBLM", "PMT"],
        default_detector="LBLM",
        scan_ranges={"x": (100, 200), "y": (150, 250)},
        timestamp=datetime(2026, 3, 10, 12, 0, 0),
        active_profiles=["x", "y"],
        install_angle=0.0,
        notes="test note",
    )


def _make_single_result(wire_name: str) -> WireMeasurementCollectionResult:
    return WireMeasurementCollectionResult(
        raw_data={
            wire_name: np.array([100.0, 150.0, 200.0]),
            "LBLM": np.array([10, 20, 30]),
            "PMT": np.array([5, 6, 7]),
        },
        metadata=_make_metadata(wire_name),
    )


class TestWireMeasurementCollectionResults(unittest.TestCase):
    def test_save_and_load_single_wire_round_trip(self):
        result = _make_single_result("WIRE:01")
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()

        try:
            result.save_to_h5(tmp.name)
            loaded = load_from_h5(tmp.name)

            self.assertEqual(loaded.metadata.wire_name, result.metadata.wire_name)
            self.assertEqual(loaded.metadata.detectors, result.metadata.detectors)
            self.assertEqual(loaded.metadata.scan_ranges, result.metadata.scan_ranges)
            self.assertEqual(loaded.metadata.notes, result.metadata.notes)

            np.testing.assert_array_equal(
                loaded.raw_data[result.metadata.wire_name],
                result.raw_data[result.metadata.wire_name],
            )
            np.testing.assert_array_equal(loaded.raw_data["LBLM"], result.raw_data["LBLM"])
            np.testing.assert_array_equal(loaded.raw_data["PMT"], result.raw_data["PMT"])
        finally:
            os.remove(tmp.name)

    def test_save_raw_data_fallback_for_unsupported_type(self):
        result = _make_single_result("WIRE:02")
        result.raw_data["UNSUPPORTED"] = {"a": 1}

        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()

        try:
            result.save_to_h5(tmp.name)
            with h5py.File(tmp.name, "r") as h5f:
                self.assertIn("raw_data", h5f)
                self.assertIn("UNSUPPORTED_unsupported", h5f["raw_data"].attrs)
        finally:
            os.remove(tmp.name)

    def test_save_and_load_multi_wire_round_trip(self):
        wire_a = _make_single_result("WIRE:A")
        wire_b = _make_single_result("WIRE:B")

        multi = MultiWireMeasurementCollectionResult(
            wire_results={"WIRE:A": wire_a, "WIRE:B": wire_b},
            timestamp=datetime(2026, 3, 10, 13, 0, 0),
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        tmp.close()

        try:
            multi.save_to_h5(tmp.name)
            loaded = load_multi_wire_from_h5(tmp.name)

            self.assertEqual(set(loaded.wire_results.keys()), {"WIRE:A", "WIRE:B"})
            self.assertEqual(loaded.timestamp, multi.timestamp)

            np.testing.assert_array_equal(
                loaded.wire_results["WIRE:A"].raw_data["LBLM"],
                wire_a.raw_data["LBLM"],
            )
            np.testing.assert_array_equal(
                loaded.wire_results["WIRE:B"].raw_data["PMT"],
                wire_b.raw_data["PMT"],
            )

            self.assertEqual(
                loaded.wire_results["WIRE:A"].metadata.scan_ranges,
                wire_a.metadata.scan_ranges,
            )
        finally:
            os.remove(tmp.name)
