from collections import deque
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from pydantic import model_validator
from typing_extensions import Self

from slac_devices.wire import Wire
from slac_tools.beam_profile import BeamProfileMeasurement
from slac_tools.ws_collection import (
    WireMeasurementCollection,
    _logger_config,
)
from slac_tools.ws_collection_results import (
    MultiWireMeasurementCollectionResult,
)


class MultiWireMeasurementCollection(BeamProfileMeasurement):
    """
    Collects wire scan measurements from multiple wires simultaneously.

    Initializes all wires concurrently, then executes measurements in quick
    succession to capture beam profile data from 3-4 wire devices.

    Design note:
        This class intentionally uses composition (it manages multiple
        WireMeasurementCollection instances) instead of inheriting from
        WireMeasurementCollection, because multi-wire orchestration is not a
        single-wire measurement.

    Attributes:
        wires (list[Wire]): List of 3-4 Wire devices to measure.
        beampath (str): Beamline identifier for buffer and device selection.
        wire_collections (dict): WireMeasurementCollection objects
            by wire name.
        logger (logging.Logger): File-based measurement logger.
    """

    name: str = "Multi-Wire Beam Profile Measurement"
    wires: list[Wire]
    beampath: str
    beam_profile_device: Optional[Wire] = None  # Override parent requirement

    # Extra fields to be set after validation
    wire_collections: Optional[dict] = None
    logger: Optional[logging.Logger] = None

    @model_validator(mode="after")
    def run_setup(self) -> Self:
        # Validate number of wires
        if not (3 <= len(self.wires) <= 4):
            raise ValueError(
                "MultiWireMeasurementCollection requires 3 or 4 wires"
            )

        # Check for unique wire names
        wire_names = [wire.name for wire in self.wires]
        if len(wire_names) != len(set(wire_names)):
            raise ValueError("All wire names must be unique")

        # Setup logger using standard configuration
        self.logger = _logger_config()

        # Create WireMeasurementCollection for each wire
        self.logger.info(
            f"Initializing MultiWireMeasurementCollection with "
            f"{len(self.wires)} wires"
        )
        self.wire_collections = {}
        for wire in self.wires:
            self.logger.info(
                f"Creating WireMeasurementCollection for {wire.name}"
            )
            self.wire_collections[wire.name] = WireMeasurementCollection(
                beam_profile_device=wire,
                beampath=self.beampath,
            )

        return self

    def measure(
        self, scan_type: str = "step", initialize_timeout: int = 10
    ) -> MultiWireMeasurementCollectionResult:
        """
        Execute a multi-wire scan as wires become ready.

        This method:
        1. Simultaneously initializes all wire devices
        2. Starts scanning the first wire that becomes enabled
        3. Queues additional wires in the order they become enabled
        4. Runs only one scan at a time until all queued wires are measured
        5. Collates all results into MultiWireMeasurementCollectionResult

        Parameters
        ----------
        scan_type : str, optional
            ``"on_the_fly"`` or ``"step"``. Passed to each wire's measure().
        initialize_timeout : int, optional
            Timeout (seconds) for each initialization attempt.
            Multi-wire initialization retries up to 3 attempts.

        Returns
        -------
        MultiWireMeasurementCollectionResult
            Combined results from all wire measurements.

        Raises
        ------
        RuntimeError
            If any wire fails to initialize after all retry attempts.
        """
        self.logger.info("Starting multi-wire measurement sequence")
        measurement_timestamp = datetime.now()

        wire_results = self._measure_wires_in_enable_order(
            scan_type=scan_type,
            timeout=initialize_timeout,
        )

        # Step 5: Collate results
        self.logger.info("All measurements complete. Creating result object.")
        return MultiWireMeasurementCollectionResult(
            wire_results=wire_results,
            timestamp=measurement_timestamp,
        )

    def _measure_wires_in_enable_order(
        self,
        scan_type: str,
        max_attempts: int = 3,
        timeout: int = 10,
    ) -> dict:
        """
        Initialize all wires and scan them in the order they become enabled.

        Initialization continues in the background with retry logic while
        completed scans proceed strictly one at a time.

        Parameters
        ----------
        scan_type : str
            Scan mode passed through to each wire collection.
        max_attempts : int
            Maximum initialize attempts per wire.
        timeout : int
            Maximum time (seconds) to wait before retrying initialize.

        Returns
        -------
        dict
            Mapping of wire name to `WireMeasurementCollectionResult`.

        Raises
        ------
        RuntimeError
            If any wire fails to initialize after all retry attempts.
        """
        overall_start_time = time.time()
        check_period = 0.2
        queued_wires = deque()
        queued_wire_names = set()
        completed_wires = set()
        failed_wires = set()
        wire_results = {}
        attempts = {name: 0 for name in self.wire_collections}
        last_initialize_time = {
            name: None for name in self.wire_collections
        }
        state_lock = threading.Lock()
        stop_event = threading.Event()

        def monitor_wire_enablement() -> None:
            last_log_time = overall_start_time

            while not stop_event.is_set():
                current_time = time.time()
                enabled_status = {}
                initialize_now = []
                exhausted_now = []

                with state_lock:
                    for wire_name, collection in self.wire_collections.items():
                        if (
                            wire_name in completed_wires
                            or wire_name in failed_wires
                        ):
                            continue

                        is_enabled = collection.my_wire.enabled
                        enabled_status[wire_name] = is_enabled

                        if is_enabled:
                            if wire_name not in queued_wire_names:
                                queued_wires.append(wire_name)
                                queued_wire_names.add(wire_name)
                                elapsed = current_time - overall_start_time
                                self.logger.info(
                                    (
                                        f"{wire_name} enabled after "
                                        f"{elapsed:.1f}s; queued for scan"
                                    )
                                )
                            continue

                        last_attempt = last_initialize_time[wire_name]
                        attempt_count = attempts[wire_name]

                        if attempt_count == 0:
                            attempts[wire_name] += 1
                            last_initialize_time[wire_name] = current_time
                            initialize_now.append(wire_name)
                        elif (
                            last_attempt is not None
                            and current_time - last_attempt >= timeout
                        ):
                            if attempt_count < max_attempts:
                                attempts[wire_name] += 1
                                last_initialize_time[wire_name] = current_time
                                initialize_now.append(wire_name)
                            else:
                                exhausted_now.append(wire_name)

                for wire_name in initialize_now:
                    attempt_number = attempts[wire_name]
                    self.logger.info(
                        f"Sending initialize command to {wire_name} "
                        f"(Attempt {attempt_number}/{max_attempts})"
                    )
                    try:
                        self.wire_collections[wire_name].my_wire.initialize()
                    except Exception as exc:
                        self.logger.error(
                            f"Failed to initialize {wire_name}: {exc}"
                        )
                        with state_lock:
                            failed_wires.add(wire_name)

                if exhausted_now:
                    with state_lock:
                        failed_wires.update(exhausted_now)
                    self.logger.warning(
                        "Some wires did not enable after all retries: "
                        f"{sorted(exhausted_now)}"
                    )

                if current_time - last_log_time >= 10.0:
                    self.logger.info(f"Wire enable status: {enabled_status}")
                    last_log_time = current_time

                with state_lock:
                    all_done = (
                        len(completed_wires) + len(failed_wires)
                        == len(self.wire_collections)
                    )

                if all_done:
                    stop_event.set()
                    return

                time.sleep(check_period)

        monitor_thread = threading.Thread(
            target=monitor_wire_enablement,
            name="wire-enable-monitor",
            daemon=True,
        )
        monitor_thread.start()

        try:
            while len(wire_results) < len(self.wire_collections):
                next_wire_name = None

                with state_lock:
                    if queued_wires:
                        next_wire_name = queued_wires.popleft()
                        queued_wire_names.remove(next_wire_name)

                    all_done = (
                        len(completed_wires) + len(failed_wires)
                        == len(self.wire_collections)
                    )

                if next_wire_name is None:
                    if all_done:
                        break
                    time.sleep(check_period)
                    continue

                self.logger.info(f"Starting measurement for {next_wire_name}")
                try:
                    result = self.wire_collections[next_wire_name].measure(
                        scan_type=scan_type
                    )
                except Exception as exc:
                    self.logger.error(
                        f"Failed to measure {next_wire_name}: {exc}"
                    )
                    raise

                wire_results[next_wire_name] = result
                with state_lock:
                    completed_wires.add(next_wire_name)

                self.logger.info(
                    f"Successfully completed measurement for {next_wire_name}"
                )
        finally:
            stop_event.set()
            monitor_thread.join(timeout=1.0)

        if len(wire_results) != len(self.wire_collections):
            failed_wire_names = sorted(failed_wires)
            if not failed_wire_names:
                failed_wire_names = sorted(
                    set(self.wire_collections) - set(wire_results)
                )

            raise RuntimeError(
                "Failed to initialize all wires after "
                f"{max_attempts} attempts. Failed wires: "
                f"{', '.join(failed_wire_names)}"
            )

        elapsed = time.time() - overall_start_time
        self.logger.info(
            "All wires successfully initialized and scanned "
            f"after {elapsed:.1f}s"
        )
        return wire_results
