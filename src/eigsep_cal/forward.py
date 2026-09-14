"""Instrument forward model: power per switch state (spec § 4)."""

from dataclasses import dataclass

import numpy as np

from . import _validate as v


@dataclass(frozen=True)
class Source:
    """A source at reference plane P: reflection and noise temperature.

    Both arrays have shape ``(1 or n_time, n_freq)`` (spec § 6).
    """

    gamma_s: np.ndarray
    t_s_k: np.ndarray

    def __post_init__(self):
        gamma = v.complex_array("gamma_s", self.gamma_s, ndim=2)
        t_s = v.float_array("t_s_k", self.t_s_k, ndim=2)
        if gamma.shape != t_s.shape:
            raise ValueError("gamma_s and t_s_k must share one shape")
        object.__setattr__(self, "gamma_s", gamma)
        object.__setattr__(self, "t_s_k", t_s)


def _rows(name, arr, rows, n_time, n_freq):
    """Select *rows* from a (1 or n_time, n_freq) or (n_freq,) array."""
    v.check_shape(name, arr, (n_freq,), (1, n_freq), (n_time, n_freq))
    if arr.ndim == 1 or arr.shape[0] == 1:
        return arr.reshape(1, n_freq)
    return arr[rows]


def power(sources, receiver, state, times_unix, additive=None):
    """Noiseless power per sample (spec § 4.1, M003 eq. Ps).

    P_s = g r_s [M_s T_s + |G_s F_s|^2 T_unc + Re(G_s F_s) T_cos
                 + Im(G_s F_s) T_sin + T_0] + A_s

    with F_s = sqrt(1 - |G_rec|^2) / (1 - G_s G_rec) and
    M_s = (1 - |G_s|^2) |F_s|^2.

    Parameters
    ----------
    sources : dict[str, Source]
        Source per switch state.
    receiver : ReceiverModel
        Receiver parameters evaluated on the samples.
    state : array_like of str
        Switch state per sample, ``(n_time,)``.
    times_unix : array_like
        Sample times, ``(n_time,)``; checked for alignment only.
    additive : dict[str, array_like] or None
        Additive post-switch term A_s per state, ``(n_freq,)`` or
        ``(n_time, n_freq)``. Absent states get zero.

    Returns
    -------
    power : np.ndarray
        Shape ``(n_time, n_freq)``.
    """
    n_time, n_freq = receiver.n_time, receiver.n_freq
    state = v.states("state", state)
    times = v.float_array("times_unix", times_unix, ndim=1)
    if state.shape != (n_time,) or times.shape != (n_time,):
        raise ValueError(
            f"state and times_unix need n_time={n_time} samples, got "
            f"{state.shape} and {times.shape}"
        )
    additive = additive or {}
    v.states("additive", list(additive))
    t_unc, t_cos, t_sin, t0 = receiver.rogers_bowman()

    out = np.empty((n_time, n_freq))
    for s in np.unique(state):
        if s not in sources:
            raise ValueError(f"no source given for state {s}")
        rows = np.flatnonzero(state == s)
        src = sources[s]
        g_s = _rows("gamma_s", src.gamma_s, rows, n_time, n_freq)
        t_s = _rows("t_s_k", src.t_s_k, rows, n_time, n_freq)
        g_r = _rows("gamma_rec", receiver.gamma_rec, rows, n_time, n_freq)

        f = np.sqrt(1 - np.abs(g_r) ** 2) / (1 - g_s * g_r)
        gf = g_s * f
        mismatch = (1 - np.abs(g_s) ** 2) * np.abs(f) ** 2
        bracket = (
            mismatch * t_s
            + np.abs(gf) ** 2 * _rows("t_unc_k", t_unc, rows, n_time, n_freq)
            + gf.real * _rows("t_cos_k", t_cos, rows, n_time, n_freq)
            + gf.imag * _rows("t_sin_k", t_sin, rows, n_time, n_freq)
            + _rows("t0_k", t0, rows, n_time, n_freq)
        )
        out[rows] = receiver.gain[rows] * receiver.path_gain(s) * bracket
        if s in additive:
            a_s = v.float_array(f"additive[{s}]", additive[s])
            out[rows] += _rows(f"additive[{s}]", a_s, rows, n_time, n_freq)
    return out


def radiometer_noise(power, enbw_hz, tau_s, n_int, rng):
    """Gaussian radiometer noise (spec § 4.4).

    sigma^2 = P^2 / (B_eff tau n_int) per sample, channels independent.

    Parameters
    ----------
    power : array_like
        Noiseless power, ``(n_time, n_freq)``.
    enbw_hz : float
        Equivalent noise bandwidth; ``conventions.D5_ENBW_HZ`` for D5.
    tau_s : array_like
        Time per integration in s, ``(n_time,)``.
    n_int : array_like of int
        Integrations averaged into each sample, ``(n_time,)``.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    noise : np.ndarray
        Same shape as *power*.
    """
    p = v.float_array("power", power, ndim=2)
    tau = v.float_array("tau_s", tau_s, ndim=1)
    n = np.asarray(n_int)
    if not np.issubdtype(n.dtype, np.integer):
        raise ValueError("n_int must be integers")
    if tau.shape != (p.shape[0],) or n.shape != (p.shape[0],):
        raise ValueError("tau_s and n_int need one value per time sample")
    if not (enbw_hz > 0) or not np.all(tau > 0) or np.any(n < 1):
        raise ValueError("enbw_hz and tau_s must be > 0, n_int >= 1")
    sigma = p / np.sqrt(enbw_hz * tau[:, None] * n[:, None])
    return rng.normal(0.0, sigma)
