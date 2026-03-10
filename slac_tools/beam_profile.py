from abc import abstractmethod
from typing import Any, Dict

from slac_devices.device import Device
from slac_tools.measurement import Measurement
from pydantic import (
    ConfigDict,
    SerializeAsAny,
)
from typing import Optional

from slac-tools.utils import NDArrayAnnotatedType
import lcls_tools


class BeamProfileMeasurementResult(slac_tools.BaseModel):
    """
    Class that contains the results of a beam profile measurement
    (for any set of axes)

    Attributes
    ----------
    rms_sizes : ndarray
        Numpy array of rms sizes of the beam in microns.
    centroids : ndarray
        Numpy array of centroids of the beam in microns.
    total_intensities : ndarray
        Numpy array of total intensities of the beam.
    metadata : Any
        Metadata information related to the measurement.

    """

    rms_sizes: Optional[NDArrayAnnotatedType] = None
    centroids: Optional[NDArrayAnnotatedType] = None
    total_intensities: Optional[NDArrayAnnotatedType] = None
    signal_to_noise_ratios: Optional[NDArrayAnnotatedType] = None
    metadata: SerializeAsAny[Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BeamProfileCollectionResult(BeamProfileMeasurementResult):
    """
    Class that contains the results of a beam profile measurement
    collection (for any set of axes)

    Attributes
    ----------
    raw_data : Dict[str, Any]
        Dictionary of device data as np.ndarrays.
        Keys are device names.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw_data: Dict[str, Any]


class BeamProfileMeasurement(Measurement):
    """
    Class that allows for beam profile measurements and fitting
    (for any set of axes)
    ------------------------
    Arguments:
    name: str (name of measurement default is beam_profile),
    device: Device (device that will be performing the measurement),
    ------------------------
    Methods:
    measure: measures beam profile
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "beam_profile"
    beam_profile_device: Device

    @abstractmethod
    def measure(self) -> BeamProfileMeasurementResult:
        """
        Measure the beam profile and return a BeamProfileMeasurementResult
        """
        pass


class BeamProfileAnalysis(slac_tools.BaseModel):
    """
    Abstract base class for post-processing analysis of beam profile measurements.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    collection_result: BeamProfileCollectionResult

    @abstractmethod
    def analyze(self):
        """
        Perform analysis on the measurement result.
        """
        pass
