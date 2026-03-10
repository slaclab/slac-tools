import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import numpy as np
import logging

# Patch missing modules before importing ws_collection
sys.modules["meme"] = MagicMock()
sys.modules["meme.names"] = MagicMock()
sys.modules["edef"] = MagicMock()
sys.modules["slac_devices"] = MagicMock()
sys.modules["slac_devices.reader"] = MagicMock()
sys.modules["slac_devices.wire"] = MagicMock()
sys.modules["slac_devices.device"] = MagicMock()

from slac_tools.wires.ws_collection import WireMeasurementCollection
from slac_tools.wires.ws_collection_results import (
    MeasurementMetadata,
)


class MockWire:
    """Mock Wire device for testing."""

    def __init__(self, name="TEST_WIRE", area="TEST_AREA"):
        self.name = name
        self.area = area
        self.x_range = (100, 200)
        self.y_range = (150, 250)
        self.u_range = (200, 300)
        self.x_wire_inner = 100
        self.x_wire_outer = 200
        self.y_wire_inner = 150
        self.y_wire_outer = 250
        self.u_wire_inner = 200
        self.u_wire_outer = 300
        self.install_angle = 0.0
        self.use_x_wire = True
        self.use_y_wire = True
        self.use_u_wire = True
        self.beam_rate = 120
        self.scan_pulses = 350
        self.motor_rbv = 150
        self.motor = 150
        self.speed = 100
        self.speed_max = 500
        self.speed_min = 10
        self.enabled = False
        self.metadata = MagicMock()
        self.metadata.detectors = ["LBLM:TEST_AREA", "PMT:TEST_AREA"]

    def start_scan(self):
        """Simulate starting a scan."""
        self.enabled = True

    def initialize(self):
        """Simulate enabling wire for step scan."""
        self.enabled = True

    def measure(self):
        """Return mock measurement data."""
        return np.array([100, 150, 200, 250, 300])


class MockBuffer:
    """Mock BSA buffer for testing."""

    def __init__(self, number=1):
        self.number = number
        self._acquisition_complete = False

    def start(self):
        """Start the buffer."""
        self._acquisition_complete = True

    def is_acquisition_complete(self):
        """Check if acquisition is complete."""
        return self._acquisition_complete

    def release(self):
        """Release the buffer."""
        pass


class TestWireMeasurementCollectionMethods(unittest.TestCase):
    """Tests for WireMeasurementCollection individual methods using mocked instances."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch external dependencies
        self.patch_custom_logger = patch(
            "slac_tools.wires.ws_collection.custom_logger"
        )
        self.mock_logger = self.patch_custom_logger.start()
        self.mock_logger.return_value = logging.getLogger("test")

        self.patch_reserve_buffer = patch(
            "slac_tools.wires.ws_collection.reserve_buffer"
        )
        self.mock_reserve_buffer = self.patch_reserve_buffer.start()
        self.mock_reserve_buffer.return_value = MockBuffer(number=1)

        self.patch_create_lblm = patch(
            "slac_tools.wires.ws_collection.create_lblm"
        )
        self.mock_create_lblm = self.patch_create_lblm.start()
        self.mock_create_lblm.return_value = MagicMock(name="LBLM_TEST")

        self.patch_create_pmt = patch(
            "slac_tools.wires.ws_collection.create_pmt"
        )
        self.mock_create_pmt = self.patch_create_pmt.start()
        self.mock_create_pmt.return_value = MagicMock(name="PMT_TEST")

        self.patch_tmit_loss = patch(
            "slac_tools.wires.ws_collection.TMITLoss"
        )
        self.mock_tmit_loss = self.patch_tmit_loss.start()

        # Create a collection instance with mocked dependencies
        self.collection = self._create_instance()

    def tearDown(self):
        """Clean up patches."""
        self.patch_custom_logger.stop()
        self.patch_reserve_buffer.stop()
        self.patch_create_lblm.stop()
        self.patch_create_pmt.stop()
        self.patch_tmit_loss.stop()

    def _create_instance(self):
        """Create a mock collection instance."""
        collection = MagicMock(spec=WireMeasurementCollection)
        mock_wire = MockWire()
        mock_wire.metadata.detectors = ["LBLM:TEST_AREA", "PMT:TEST_AREA"]

        collection.beam_profile_device = mock_wire
        collection.my_wire = mock_wire
        collection.beampath = "CU_HXR"
        collection.my_buffer = MockBuffer(number=1)
        collection.devices = {mock_wire.name: mock_wire}
        collection.detectors = [d.split(":")[0] for d in mock_wire.metadata.detectors]
        collection.data = None
        collection.profiles = None
        collection.logger = logging.getLogger("test")

        # Bind the real methods to the mock
        collection.create_metadata = WireMeasurementCollection.create_metadata.__get__(
            collection, WireMeasurementCollection
        )
        collection._active_profiles = (
            WireMeasurementCollection._active_profiles.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._validate_position_data = (
            WireMeasurementCollection._validate_position_data.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._get_units_for_device = (
            WireMeasurementCollection._get_units_for_device.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._get_buffer_collection_method = (
            WireMeasurementCollection._get_buffer_collection_method.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._wait_until = WireMeasurementCollection._wait_until.__get__(
            collection, WireMeasurementCollection
        )
        collection._calc_buffer_points = (
            WireMeasurementCollection._calc_buffer_points.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._get_default_detector = (
            WireMeasurementCollection._get_default_detector.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._reserve_buffer = WireMeasurementCollection._reserve_buffer.__get__(
            collection, WireMeasurementCollection
        )
        collection._load_yaml_config = MagicMock(
            return_value=None
        )  # Mock yaml loading to return None
        collection._get_step_positions = (
            WireMeasurementCollection._get_step_positions.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._calculate_step_speed = (
            WireMeasurementCollection._calculate_step_speed.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._move_to_step_position = (
            WireMeasurementCollection._move_to_step_position.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._perform_step_scan = (
            WireMeasurementCollection._perform_step_scan.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection.scan_with_wire = WireMeasurementCollection.scan_with_wire.__get__(
            collection, WireMeasurementCollection
        )
        collection._start_scan_with_retry = (
            WireMeasurementCollection._start_scan_with_retry.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._initialize_wire_for_step_scan = (
            WireMeasurementCollection._initialize_wire_for_step_scan.__get__(
                collection, WireMeasurementCollection
            )
        )
        collection._initialize_wire_with_retry = (
            WireMeasurementCollection._initialize_wire_with_retry.__get__(
                collection, WireMeasurementCollection
            )
        )

        return collection

    def test_create_metadata(self):
        """Test metadata creation."""
        metadata = self.collection.create_metadata()

        self.assertIsInstance(metadata, MeasurementMetadata)
        self.assertEqual(metadata.wire_name, self.collection.my_wire.name)
        self.assertEqual(metadata.area, self.collection.my_wire.area)
        self.assertEqual(metadata.beampath, "CU_HXR")
        self.assertIsInstance(metadata.timestamp, datetime)

    def test_active_profiles_all_enabled(self):
        """Test _active_profiles when all are enabled."""
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = True
        self.collection.my_wire.use_u_wire = True

        active = self.collection._active_profiles()
        self.assertEqual(active, ["x", "y", "u"])

    def test_active_profiles_selective(self):
        """Test _active_profiles with selective enablement."""
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = False
        self.collection.my_wire.use_u_wire = True

        active = self.collection._active_profiles()
        self.assertEqual(active, ["x", "u"])

    def test_active_profiles_only_x(self):
        """Test _active_profiles with only X enabled."""
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = False
        self.collection.my_wire.use_u_wire = False

        active = self.collection._active_profiles()
        self.assertEqual(active, ["x"])

    def test_validate_position_data_valid(self):
        """Test _validate_position_data with valid data."""
        position_data = np.array([100, 150, 200, 250])

        # Should not raise
        self.collection._validate_position_data(position_data)

    def test_validate_position_data_invariant(self):
        """Test _validate_position_data with invariant data."""
        position_data = np.array([150, 150, 150, 150])

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.collection._validate_position_data(position_data)

    def test_get_units_for_device_tmitloss(self):
        """Test _get_units_for_device for TMITLOSS."""
        units = self.collection._get_units_for_device("TMITLOSS")
        self.assertEqual(units, "%% beam loss")

    def test_get_units_for_device_other(self):
        """Test _get_units_for_device for other devices."""
        units = self.collection._get_units_for_device("LBLM")
        self.assertEqual(units, "counts")

        units = self.collection._get_units_for_device("PMT")
        self.assertEqual(units, "counts")

    def test_get_buffer_collection_method_wire(self):
        """Test _get_buffer_collection_method for wire device."""
        method = self.collection._get_buffer_collection_method(
            self.collection.my_wire.name
        )
        self.assertEqual(method, "position_buffer")

    def test_get_buffer_collection_method_lblm(self):
        """Test _get_buffer_collection_method for LBLM detector."""
        method = self.collection._get_buffer_collection_method("LBLM")
        self.assertEqual(method, "fast_buffer")

    def test_get_buffer_collection_method_pmt(self):
        """Test _get_buffer_collection_method for PMT detector."""
        method = self.collection._get_buffer_collection_method("PMT")
        self.assertEqual(method, "qdcraw_buffer")

    def test_get_buffer_collection_method_tmitloss(self):
        """Test _get_buffer_collection_method for TMITLOSS."""
        method = self.collection._get_buffer_collection_method("TMITLOSS")
        self.assertIsNone(method)

    def test_wait_until_condition_true(self):
        """Test _wait_until when condition becomes true."""
        call_count = [0]

        def condition():
            call_count[0] += 1
            return call_count[0] >= 3  # Return True on 3rd call

        result = self.collection._wait_until(condition, timeout=2, period=0.01)

        self.assertTrue(result)

    def test_wait_until_timeout(self):
        """Test _wait_until when timeout expires."""

        def condition():
            return False  # Never True

        result = self.collection._wait_until(condition, timeout=0.1, period=0.05)

        self.assertFalse(result)

    def test_calc_buffer_points_default_rate(self):
        """Test _calc_buffer_points with default rate."""
        self.collection.my_wire.beam_rate = 120
        self.collection.my_wire.scan_pulses = 350

        points = self.collection._calc_buffer_points()

        # Should return reasonable buffer size
        self.assertGreater(points, 0)
        self.assertLess(points, 20000)  # System limit

    def test_calc_buffer_points_high_rate(self):
        """Test _calc_buffer_points with high beam rate."""
        self.collection.my_wire.beam_rate = 16000
        self.collection.my_wire.scan_pulses = 5000

        points = self.collection._calc_buffer_points()

        # Should return larger buffer for high rate
        self.assertGreater(points, 0)
        self.assertLess(points, 20000)  # System limit

    def test_calc_buffer_points_invalid_rate(self):
        """Test _calc_buffer_points with invalid rate."""
        self.collection.my_wire.beam_rate = None
        self.collection.my_wire.scan_pulses = 350

        # Should default to 120 Hz
        points = self.collection._calc_buffer_points()

        self.assertGreater(points, 0)

    def test_reserve_buffer(self):
        """Test _reserve_buffer method."""
        buffer = self.collection._reserve_buffer()

        self.assertIsNotNone(buffer)
        self.mock_reserve_buffer.assert_called()

    def test_get_step_positions_all_profiles(self):
        """Test _get_step_positions with all profiles enabled."""
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = True
        self.collection.my_wire.use_u_wire = True

        positions = self.collection._get_step_positions()

        # Should have 6 positions (inner/outer for each of x, y, u)
        self.assertEqual(len(positions), 6)
        # Should be sorted
        self.assertEqual(positions, sorted(positions))

    def test_get_step_positions_selective_profiles(self):
        """Test _get_step_positions with only X and U enabled."""
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = False
        self.collection.my_wire.use_u_wire = True

        positions = self.collection._get_step_positions()

        # Should have 4 positions (inner/outer for x and u)
        self.assertEqual(len(positions), 4)
        self.assertEqual(positions, sorted(positions))

    def test_calculate_step_speed_inner_position(self):
        """Test _calculate_step_speed for inner positions (even indices)."""
        positions = [10, 20, 30, 40]
        self.collection.my_wire.speed_max = 100

        # Index 0 (even) = inner position
        speed = self.collection._calculate_step_speed(0, positions)
        self.assertEqual(speed, 100)

        # Index 2 (even) = inner position
        speed = self.collection._calculate_step_speed(2, positions)
        self.assertEqual(speed, 100)

    def test_calculate_step_speed_outer_position(self):
        """Test _calculate_step_speed for outer positions (odd indices)."""
        self.collection.my_wire.scan_pulses = 350
        self.collection.my_wire.beam_rate = 120
        positions = [10, 20, 30, 40]

        # Index 1 (odd) = outer position
        # distance = 20 - 10 = 10
        # speed = (10 / 350) * 120 ≈ 3
        speed = self.collection._calculate_step_speed(1, positions)
        expected = int((10 / 350) * 120)
        self.assertEqual(speed, expected)

    def test_move_to_step_position_success(self):
        """Test _move_to_step_position when motor reaches target."""
        target_pos = 50
        self.collection.my_wire.motor_rbv = 0  # Set initial position
        self.collection.my_wire.x_wire_inner = 20
        self.collection.my_wire.x_wire_outer = 50
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = False
        self.collection.my_wire.use_u_wire = False

        # Mock _wait_until to simulate motor reaching target
        def mock_wait(condition, timeout=5, period=0.1):
            # Simulate motor reaching target by updating motor_rbv
            self.collection.my_wire.motor_rbv = target_pos
            return condition()

        self.collection._wait_until = mock_wait

        # Should not raise
        self.collection._move_to_step_position(target_pos, 1, 2)

        # Verify motor was set
        self.assertEqual(self.collection.my_wire.motor, target_pos)

    def test_move_to_step_position_timeout(self):
        """Test _move_to_step_position when motor fails to reach target."""
        target_pos = 50
        self.collection.my_wire.motor_rbv = 0
        self.collection.my_wire.x_wire_inner = 20
        self.collection.my_wire.x_wire_outer = 50
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = False
        self.collection.my_wire.use_u_wire = False

        # Mock _wait_until to always return False (timeout)
        self.collection._wait_until = MagicMock(return_value=False)

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.collection._move_to_step_position(target_pos, 1, 2)

    def test_perform_step_scan(self):
        """Test _perform_step_scan orchestration."""
        # Setup wire for step scan
        self.collection.my_wire.x_wire_inner = 10
        self.collection.my_wire.x_wire_outer = 20
        self.collection.my_wire.use_x_wire = True
        self.collection.my_wire.use_y_wire = False
        self.collection.my_wire.use_u_wire = False
        self.collection.my_wire.motor_rbv = 0
        self.collection.my_wire.speed_max = 100

        buf = MockBuffer()
        buf.start = MagicMock()
        buf.is_acquisition_complete = MagicMock(side_effect=[False, True])
        self.collection.my_buffer = buf

        # Mock _initialize_wire_for_step_scan to simulate wire initialization
        self.collection._initialize_wire_for_step_scan = MagicMock()

        # Mock _wait_until to avoid actual waiting
        def mock_wait(condition, timeout=5, period=0.1):
            return True

        self.collection._wait_until = mock_wait

        # Perform step scan
        self.collection._perform_step_scan()

        # Verify wire initialization was attempted (without start_scan)
        self.collection._initialize_wire_for_step_scan.assert_called_once()

        # Verify buffer was started
        buf.start.assert_called_once()

        # Verify wire was retracted to position 100
        self.assertEqual(self.collection.my_wire.motor, 100)

    def test_initialize_wire_for_step_scan_success(self):
        """Test _initialize_wire_for_step_scan when wire enables on first attempt."""
        self.collection.my_wire.enabled = False

        def mock_wait(condition, timeout=5, period=0.1):
            # Simulate wire becoming enabled
            self.collection.my_wire.enabled = True
            return condition()

        self.collection._wait_until = mock_wait

        # Should not raise
        self.collection._initialize_wire_for_step_scan()

        # Verify initialize was called
        # Note: my_wire is a real MockWire, check its behavior
        self.assertTrue(self.collection.my_wire.enabled)

    def test_initialize_wire_for_step_scan_timeout(self):
        """Test _initialize_wire_for_step_scan when wire fails to enable."""
        self.collection.my_wire.enabled = False

        # Mock _wait_until to always return False (timeout)
        self.collection._wait_until = MagicMock(return_value=False)

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.collection._initialize_wire_for_step_scan()

    def test_initialize_wire_with_retry_start_scan(self):
        """Test _initialize_wire_with_retry with start_scan action."""
        self.collection.my_wire.enabled = False
        self.collection.my_wire._start_scan_called = False

        original_start_scan = self.collection.my_wire.start_scan

        def track_start_scan():
            self.collection.my_wire._start_scan_called = True
            original_start_scan()

        self.collection.my_wire.start_scan = track_start_scan

        def mock_wait(condition, timeout=5, period=0.1):
            self.collection.my_wire.enabled = True
            return condition()

        self.collection._wait_until = mock_wait

        # Call with start_scan action
        self.collection._initialize_wire_with_retry(wire_action="start_scan")

        # Verify start_scan was called
        self.assertTrue(self.collection.my_wire._start_scan_called)

    def test_initialize_wire_with_retry_initialize(self):
        """Test _initialize_wire_with_retry with initialize action."""
        self.collection.my_wire.enabled = False
        self.collection.my_wire._initialize_called = False

        original_initialize = self.collection.my_wire.initialize

        def track_initialize():
            self.collection.my_wire._initialize_called = True
            original_initialize()

        self.collection.my_wire.initialize = track_initialize

        def mock_wait(condition, timeout=5, period=0.1):
            self.collection.my_wire.enabled = True
            return condition()

        self.collection._wait_until = mock_wait

        # Call with initialize action
        self.collection._initialize_wire_with_retry(wire_action="initialize")

        # Verify initialize was called
        self.assertTrue(self.collection.my_wire._initialize_called)

    def test_initialize_wire_with_retry_invalid_action(self):
        """Test _initialize_wire_with_retry with invalid action."""
        # Should raise ValueError for unknown action
        with self.assertRaises(ValueError):
            self.collection._initialize_wire_with_retry(wire_action="invalid_action")

    def test_initialize_wire_with_retry_timeout(self):
        """Test _initialize_wire_with_retry when wire fails to enable."""
        self.collection.my_wire.enabled = False

        # Mock _wait_until to always return False (timeout)
        self.collection._wait_until = MagicMock(return_value=False)

        # Should raise RuntimeError
        with self.assertRaises(RuntimeError):
            self.collection._initialize_wire_with_retry(wire_action="initialize")

    def test_scan_with_wire_default_uses_retry(self):
        """Default scan type should call _start_scan_with_retry."""
        # prepare environment
        self.collection.my_buffer = MockBuffer()

        # Mock _start_scan_with_retry
        self.collection._start_scan_with_retry = MagicMock()

        # call without specifying scan_type
        self.collection.scan_with_wire()
        self.collection._start_scan_with_retry.assert_called_once()

    def test_scan_with_wire_step_calls_perform_step_scan(self):
        """Step scan should call _perform_step_scan."""
        buf = MockBuffer()
        self.collection.my_buffer = buf

        # Mock _perform_step_scan
        self.collection._perform_step_scan = MagicMock()

        # call with step scan_type
        self.collection.scan_with_wire(scan_type="step")
        self.collection._perform_step_scan.assert_called_once()

    def test_measure_calls_correct_methods_for_scan_types(self):
        """Ensure measure dispatches scan_type correctly and handles buffer start."""
        # patch methods to observe calls
        self.collection.scan_with_wire = MagicMock()
        self.collection.start_timing_buffer = MagicMock()
        self.collection.get_data_from_buffer = MagicMock(
            return_value={"TEST_WIRE": np.array([1])}
        )

        # default (on_the_fly)
        WireMeasurementCollection.measure(self.collection)
        self.collection.scan_with_wire.assert_called_once_with(scan_type="on_the_fly")
        self.collection.start_timing_buffer.assert_called_once()
        self.collection.scan_with_wire.reset_mock()
        self.collection.start_timing_buffer.reset_mock()

        # explicit step
        WireMeasurementCollection.measure(self.collection, scan_type="step")
        self.collection.scan_with_wire.assert_called_once_with(scan_type="step")
        self.assertFalse(self.collection.start_timing_buffer.called)



