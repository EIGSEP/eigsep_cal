"""Data objects shared by adapters, the generator and stages (spec § 5)."""

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from . import _validate as v
from .conventions import ANTENNAS, STATES, d5_freqs_mhz

REFLECTION_SOURCES = STATES + ("receiver",)

# Tolerance of the gamma_cov checks, relative to the largest entry of
# each 2x2 block. Roundoff in J C J^T is a few eps (~1e-16), and a real
# asymmetry is of order one; np.random.multivariate_normal checks
# covariances to the same 1e-8.
_COV_RTOL = 1e-8


def _set(obj, **values):
    for key, value in values.items():
        object.__setattr__(obj, key, value)


def _series(name, value, n):
    arr = v.float_array(name, value, ndim=1)
    if arr.shape != (n,):
        raise ValueError(f"{name} needs {n} values, got shape {arr.shape}")
    return arr


def _times(value, n):
    """``times_unix`` as a finite, strictly increasing (n,) series."""
    arr = _series("times_unix", value, n)
    if not np.all(np.isfinite(arr)) or not np.all(np.diff(arr) > 0):
        raise ValueError("times_unix must be finite and strictly increasing")
    return arr


def _antenna(value):
    if value not in ANTENNAS:
        raise ValueError(f"antenna must be one of {ANTENNAS}, got {value!r}")
    return value


def _readonly(arr):
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, kw_only=True)
class SkyTemperature:
    """Free-space antenna temperature from eigsim (spec § 5.1)."""

    t_ant_k: np.ndarray
    freqs_mhz: np.ndarray
    times_unix: np.ndarray
    antenna: str
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        freqs = v.freqs_mhz(self.freqs_mhz)
        t_ant = v.float_array("t_ant_k", self.t_ant_k, ndim=2)
        v.check_shape("t_ant_k", t_ant, (None, freqs.size))
        n_time = t_ant.shape[0]
        _set(
            self,
            freqs_mhz=freqs,
            t_ant_k=t_ant,
            times_unix=_times(self.times_unix, n_time),
            elevation_deg=_series("elevation_deg", self.elevation_deg, n_time),
            azimuth_deg=_series("azimuth_deg", self.azimuth_deg, n_time),
            antenna=_antenna(self.antenna),
            meta=MappingProxyType(dict(self.meta)),
        )


@dataclass(frozen=True, kw_only=True)
class Reflection:
    """Calibrated reflection coefficients at P, from stage 2 (spec § 5.3)."""

    gamma: np.ndarray
    gamma_cov: np.ndarray
    times_unix: np.ndarray
    source: str
    freqs_mhz: np.ndarray

    def __post_init__(self):
        freqs = v.freqs_mhz(self.freqs_mhz)
        gamma = v.complex_array("gamma", self.gamma, ndim=2)
        v.check_shape("gamma", gamma, (None, freqs.size))
        n_meas = gamma.shape[0]
        cov = v.float_array("gamma_cov", self.gamma_cov, ndim=4)
        v.check_shape("gamma_cov", cov, (n_meas, freqs.size, 2, 2))
        if not np.all(np.isfinite(cov)):
            raise ValueError("gamma_cov must be finite")
        scale = np.max(np.abs(cov), axis=(-2, -1), keepdims=True)
        unit = cov / np.where(scale > 0, scale, 1.0)
        if np.any(np.abs(unit[..., 0, 1] - unit[..., 1, 0]) > _COV_RTOL):
            raise ValueError("gamma_cov must be symmetric")
        if np.any(cov[..., 0, 0] < 0) or np.any(cov[..., 1, 1] < 0):
            raise ValueError("gamma_cov variances must be >= 0")
        off = 0.5 * (unit[..., 0, 1] + unit[..., 1, 0])
        if np.any(unit[..., 0, 0] * unit[..., 1, 1] - off**2 < -_COV_RTOL):
            raise ValueError("gamma_cov must be positive semi-definite")
        if self.source not in REFLECTION_SOURCES:
            raise ValueError(
                f"source must be one of {REFLECTION_SOURCES}, "
                f"got {self.source!r}"
            )
        _set(
            self,
            freqs_mhz=freqs,
            gamma=gamma,
            gamma_cov=cov,
            times_unix=_times(self.times_unix, n_meas),
        )

    def save(self, path):
        """Write to an npz file (spec § 5.6)."""
        from . import io

        io.save(self, path)

    @classmethod
    def load(cls, path):
        """Read an npz file written by :meth:`save`."""
        from . import io

        return io.load(path, cls)


@dataclass(frozen=True, kw_only=True)
class Observation:
    """The data vector for one antenna (spec § 5.4)."""

    _ARRAY_DICTS = ("covariates",)

    power: np.ndarray
    state: np.ndarray
    times_unix: np.ndarray
    tau_s: np.ndarray
    n_int: np.ndarray
    flags: np.ndarray
    freqs_mhz: np.ndarray
    antenna: str
    chan: np.ndarray | None = None
    covariates: dict = field(default_factory=dict)
    changepoints_unix: np.ndarray = field(default_factory=lambda: np.zeros(0))
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        freqs = v.freqs_mhz(self.freqs_mhz)
        power = v.float_array("power", self.power, ndim=2)
        v.check_shape("power", power, (None, freqs.size))
        n_time = power.shape[0]

        state = v.states("state", self.state)
        v.check_shape("state", state, (n_time,))
        tau = _series("tau_s", self.tau_s, n_time)
        if not np.all(tau > 0):
            raise ValueError("tau_s must be > 0")
        n_int = np.array(self.n_int)
        if (
            not np.issubdtype(n_int.dtype, np.integer)
            or n_int.shape != (n_time,)
            or np.any(n_int < 1)
        ):
            raise ValueError("n_int must be (n_time,) integers >= 1")

        flags = np.array(self.flags)
        if flags.dtype != np.bool_:
            raise TypeError("flags must be bool (True = bad)")
        v.check_shape("flags", flags, (n_time, freqs.size))

        chan = None
        if self.chan is not None:
            chan = np.array(self.chan)
            v.check_shape("chan", chan, (freqs.size,))
            if not np.array_equal(d5_freqs_mhz(chan), freqs):
                raise ValueError(
                    "freqs_mhz must be the D5 frequencies of chan"
                )
            chan = _readonly(chan.astype(np.int64))

        covariates = {
            name: _series(f"covariates[{name}]", value, n_time)
            for name, value in self.covariates.items()
        }
        changepoints = v.float_array(
            "changepoints_unix", self.changepoints_unix, ndim=1
        )
        if not np.all(np.isfinite(changepoints)) or not np.all(
            np.diff(changepoints) > 0
        ):
            raise ValueError("changepoints_unix must be strictly increasing")

        _set(
            self,
            freqs_mhz=freqs,
            power=power,
            state=state,
            times_unix=_times(self.times_unix, n_time),
            tau_s=tau,
            n_int=_readonly(n_int.astype(np.int64)),
            flags=_readonly(flags),
            antenna=_antenna(self.antenna),
            chan=chan,
            covariates=MappingProxyType(covariates),
            changepoints_unix=changepoints,
            provenance=MappingProxyType(dict(self.provenance)),
        )

    @property
    def n_time(self):
        return self.power.shape[0]

    @property
    def n_freq(self):
        return self.power.shape[1]

    def save(self, path):
        """Write to an npz file (spec § 5.6).

        ``provenance`` is stored as JSON: tuples reload as lists and
        non-string keys as strings (see :func:`eigsep_cal.io.save`).
        """
        from . import io

        io.save(self, path)

    @classmethod
    def load(cls, path):
        """Read an npz file written by :meth:`save`."""
        from . import io

        return io.load(path, cls)
