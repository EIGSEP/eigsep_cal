"""Two-port networks: S-parameters and embedding (spec § 4.3, § 5.2)."""

from dataclasses import dataclass

import numpy as np

from . import _validate as v


@dataclass(frozen=True)
class SParams:
    """S-parameters of a switch path or cable.

    Port convention (spec § 5.2, cmt_vna ``calkit``): port 1 (``s11``)
    faces the reference side and port 2 (``s22``) faces the
    termination. Only the product ``s12s21`` is stored.
    """

    s11: np.ndarray
    s12s21: np.ndarray
    s22: np.ndarray
    name: str

    def __post_init__(self):
        for key in ("s11", "s12s21", "s22"):
            arr = v.complex_array(key, getattr(self, key), ndim=1)
            object.__setattr__(self, key, arr)
        if not self.s11.shape == self.s12s21.shape == self.s22.shape:
            raise ValueError("s11, s12s21 and s22 must share one shape")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")


def embed(gamma_term, t_term_k, sparams, t_path_k):
    """The source seen at the reference side of a two-port.

    Gamma_s = S11 + S12S21 Gamma_t / (1 - S22 Gamma_t), and
    T_s = G T_t + (1 - G) T_p with the available gain
    G = |S12S21| (1 - |Gamma_t|^2) / (|1 - S22 Gamma_t|^2 (1 - |Gamma_s|^2))
    (M003 eq. availgain, with the termination on port 2). |S21|^2 is
    taken as |S12S21|, which assumes a reciprocal, passive path.

    Parameters
    ----------
    gamma_term : array_like
        Termination reflection coefficient, ``(..., n_freq)``.
    t_term_k : array_like
        Termination available noise temperature in K, broadcastable.
    sparams : SParams
        The path, on the same frequency grid.
    t_path_k : array_like
        Physical temperature of the path in K, broadcastable.

    Returns
    -------
    gamma_s : np.ndarray
        Reflection coefficient at the reference side, complex128.
    t_s_k : np.ndarray
        Available noise temperature at the reference side in K.
    """
    gamma_t = v.complex_array("gamma_term", gamma_term)
    t_t = v.float_array("t_term_k", t_term_k)
    t_p = v.float_array("t_path_k", t_path_k)
    denom = 1 - sparams.s22 * gamma_t
    gamma_s = sparams.s11 + sparams.s12s21 * gamma_t / denom
    gain = (
        np.abs(sparams.s12s21)
        * (1 - np.abs(gamma_t) ** 2)
        / (np.abs(denom) ** 2 * (1 - np.abs(gamma_s) ** 2))
    )
    return gamma_s, gain * t_t + (1 - gain) * t_p
