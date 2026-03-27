from importlib.metadata import PackageNotFoundError, version

__author__ = "Christian Hellum Bye"

try:
    __version__ = version("eigsep_cal")
except PackageNotFoundError:
    __version__ = "unknown"
