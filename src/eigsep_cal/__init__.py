from importlib.metadata import PackageNotFoundError, version

__author__ = "Christian Hellum Bye"

try:
    __version__ = version("eigsep_cal")
except PackageNotFoundError:
    __version__ = "unknown"

from . import conventions
from .data import Observation, Reflection, SkyTemperature
from .forward import Source, power, radiometer_noise
from .network import SParams, embed
from .receiver import ReceiverModel, noise_wave_map
