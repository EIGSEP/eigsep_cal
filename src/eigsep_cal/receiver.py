"""Receiver noise parameters (spec § 4.2; M003 eqs. Ps and map)."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import _validate as v

PARAMETERISATIONS = ("rogers_bowman", "intrinsic")
_RB = ("t_unc_k", "t_cos_k", "t_sin_k", "t0_k")
_INTRINSIC = ("t_r_k", "t_l_k", "c_k")


def noise_wave_map(gamma_rec, t_r_k, t_l_k, c_k):
    """Rogers & Bowman noise parameters from intrinsic ones (M003 eq. map).

    T_0 = (1 - |G|^2) T_R
    T_unc = T_L + |G|^2 T_R + 2 Re(G c*)
    T_cos - i T_sin = 2 sqrt(1 - |G|^2) (c + G T_R)

    with G the receiver reflection coefficient and c = <A_R* A_L>.
    Bucher+2026 print T_unc without the factor 2; that is wrong.

    Returns
    -------
    t0_k, t_unc_k, t_cos_k, t_sin_k : np.ndarray
    """
    g = np.asarray(gamma_rec, dtype=np.complex128)
    t_r = np.asarray(t_r_k, dtype=np.float64)
    t_l = np.asarray(t_l_k, dtype=np.float64)
    c = np.asarray(c_k, dtype=np.complex128)
    matched = 1 - np.abs(g) ** 2
    t0 = matched * t_r
    t_unc = t_l + np.abs(g) ** 2 * t_r + 2 * np.real(g * np.conj(c))
    z = 2 * np.sqrt(matched) * (c + g * t_r)
    return t0, t_unc, z.real, -z.imag


@dataclass(frozen=True, kw_only=True)
class ReceiverModel:
    """Receiver parameters for the forward model (spec § 4.2).

    Give either the Rogers & Bowman set (t_unc_k, t_cos_k, t_sin_k,
    t0_k) or the intrinsic set (t_r_k, t_l_k, c_k), selected by
    ``parameterisation``.
    """

    gamma_rec: np.ndarray
    gain: np.ndarray
    parameterisation: str = "rogers_bowman"
    t_unc_k: np.ndarray | None = None
    t_cos_k: np.ndarray | None = None
    t_sin_k: np.ndarray | None = None
    t0_k: np.ndarray | None = None
    t_r_k: np.ndarray | None = None
    t_l_k: np.ndarray | None = None
    c_k: np.ndarray | None = None
    path_gain_ratio: dict | None = None

    def __post_init__(self):
        if self.parameterisation not in PARAMETERISATIONS:
            raise ValueError(
                f"parameterisation must be one of {PARAMETERISATIONS}"
            )
        gain = v.float_array("gain", self.gain, ndim=2)
        if not np.all(gain > 0):
            raise ValueError("gain must be > 0")
        object.__setattr__(self, "gain", gain)
        n_time, n_freq = gain.shape
        allowed = ((n_freq,), (n_time, n_freq))

        gamma = v.complex_array("gamma_rec", self.gamma_rec)
        v.check_shape("gamma_rec", gamma, *allowed)
        if not np.all(np.abs(gamma) < 1):
            raise ValueError(
                "gamma_rec must satisfy |gamma_rec| < 1; the model takes "
                "sqrt(1 - |gamma_rec|^2)"
            )
        object.__setattr__(self, "gamma_rec", gamma)

        wanted, unwanted = (
            (_RB, _INTRINSIC)
            if self.parameterisation == "rogers_bowman"
            else (_INTRINSIC, _RB)
        )
        for key in unwanted:
            if getattr(self, key) is not None:
                raise ValueError(
                    f"{key} belongs to the other parameterisation "
                    f"(this one is {self.parameterisation!r}; "
                    f"intrinsic is {_INTRINSIC})"
                )
        for key in wanted:
            value = getattr(self, key)
            if value is None:
                raise ValueError(
                    f"{key} is required for {self.parameterisation}"
                )
            conv = v.complex_array if key == "c_k" else v.float_array
            arr = conv(key, value)
            v.check_shape(key, arr, *allowed)
            object.__setattr__(self, key, arr)

        ratios = {}
        for state, value in (self.path_gain_ratio or {}).items():
            v.states("path_gain_ratio", [state])
            arr = v.float_array(f"path_gain_ratio[{state}]", value)
            v.check_shape(f"path_gain_ratio[{state}]", arr, (n_freq,))
            if not np.all(arr > 0):
                raise ValueError(f"path_gain_ratio[{state}] must be > 0")
            ratios[state] = arr
        object.__setattr__(self, "path_gain_ratio", MappingProxyType(ratios))

    @property
    def n_time(self):
        return self.gain.shape[0]

    @property
    def n_freq(self):
        return self.gain.shape[1]

    def rogers_bowman(self):
        """(t_unc_k, t_cos_k, t_sin_k, t0_k), mapped if intrinsic."""
        if self.parameterisation == "rogers_bowman":
            return self.t_unc_k, self.t_cos_k, self.t_sin_k, self.t0_k
        t0, t_unc, t_cos, t_sin = noise_wave_map(
            self.gamma_rec, self.t_r_k, self.t_l_k, self.c_k
        )
        return t_unc, t_cos, t_sin, t0

    def path_gain(self, state):
        """Path-gain ratio r_s for *state*, ones when not given."""
        if state in self.path_gain_ratio:
            return self.path_gain_ratio[state]
        return v.float_array("path_gain", np.ones(self.n_freq))
