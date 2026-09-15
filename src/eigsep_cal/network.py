"""Two-port networks: S-parameters and embedding (spec § 4.3, § 5.2)."""

from dataclasses import dataclass

import numpy as np

from . import _validate as v


@dataclass(frozen=True)
class SParams:
    """S-parameters of a switch path or cable.

    The direction fixes the port labels (spec § 5.2). A signal crosses
    the path from port 2 (``s22``), where the termination sits, to
    port 1 (``s11``), the reference side it is observed from: P for
    ``RF*`` paths, the VNA for ``VNA*`` paths. These are the labels of
    cmt_vna ``calkit`` and of Monsalve et al. (2024). Only the product
    ``s12s21`` is stored.
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

    Embeds, and only embeds (spec § 5.2): it carries a termination on
    port 2 out through the path to the reference side on port 1::

        Gamma_s = S11 + S12S21 Gamma_t / (1 - S22 Gamma_t)   (eq. embed)
        T_s = G T_t + (1 - G) T_p,
        G = |S12S21| (1 - |Gamma_t|^2)
            / (|1 - S22 Gamma_t|^2 (1 - |Gamma_s|^2))   (eq. availgain)

    G is the available gain from port 2 to port 1 (M003 eqs. embed and
    availgain; spec § 4.3). These are Monsalve et al. (2024) eqs. 16
    and 17, the balun efficiency, in the same port labels, and
    T_s = G T_t + (1 - G) T_p is Monsalve et al. (2017) eq. 8.
    Monsalve et al. (2017) eq. 9 has S11 in the denominator because
    their port 1 carries the termination. |S12|^2 is taken as |S12S21|,
    which assumes a reciprocal path; 0 <= G <= 1 needs it passive.

    Through a lossy path T_s lies strictly between T_t and T_p. The
    inverse, T_t = (T_s - (1 - G) T_p) / G, is de-embedding, which
    eigsep_cal never does.

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
