"""Construction-time validation shared by eigsep_cal objects."""

import numpy as np

from .conventions import STATES


def _finish(name, arr, ndim):
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}, got {arr.ndim}")
    arr.flags.writeable = False
    return arr


def float_array(name, value, ndim=None):
    """A read-only float64 copy of *value*; complex input is an error."""
    arr = np.asarray(value)
    if np.iscomplexobj(arr):
        raise TypeError(f"{name} must be real, got complex")
    return _finish(name, np.array(arr, dtype=np.float64), ndim)


def complex_array(name, value, ndim=None):
    """A read-only complex128 copy of *value*."""
    return _finish(name, np.array(value, dtype=np.complex128), ndim)


def freqs_mhz(value):
    """A validated frequency axis: 1-D, non-empty, strictly increasing."""
    arr = float_array("freqs_mhz", value, ndim=1)
    if arr.size == 0 or np.any(np.diff(arr) <= 0):
        raise ValueError("freqs_mhz must be non-empty and strictly increasing")
    return arr


def same_grid(name, freqs, reference):
    """Require *freqs* to equal *reference* exactly."""
    if not np.array_equal(freqs, reference):
        raise ValueError(
            f"{name} is on a different frequency grid; "
            "eigsep_cal never interpolates in frequency"
        )


def check_shape(name, arr, *allowed):
    """Require *arr* to match one allowed shape (None matches any size)."""
    for shape in allowed:
        if len(shape) == arr.ndim and all(
            want is None or want == got for want, got in zip(shape, arr.shape)
        ):
            return
    raise ValueError(
        f"{name} has shape {arr.shape}; allowed {allowed} (None = any)"
    )


def states(name, value):
    """A read-only array of switch-state names, all from STATES."""
    arr = np.array(value, dtype=str)
    unknown = sorted(set(arr.ravel()) - set(STATES))
    if unknown:
        raise ValueError(
            f"{name} has unknown states {unknown}; allowed {STATES}"
        )
    arr.flags.writeable = False
    return arr
