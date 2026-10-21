"""The noise source's temperature from a hot/reference-load measurement, and
what a receiver takes of it.

The model is the minimal tier of ``dicke`` (memo M003 section 7.1): each
switch state's power is ``P_s = g (M_s T_s + T_r)``, with ``M_s`` from
:func:`eigsep_cal.dicke.mismatch` at the reference plane P, the same gain
``g`` and receiver noise ``T_r`` in every state (no noise waves), and no loss
in the switch paths.

A bench measurement switches a hot load (temperature ``T_hot``), a reference
load (``T_ref``) and the noise source on, and forms the switched ratio
``Q = (P_hot - P_ref) / (P_NON - P_ref)``
(:func:`eigsep_cal.dicke.calc_Tant_star` with ``T_ns = 1`` and
``T_load = 0``). With the noise-source path's reflection at P known, the
model gives::

    M_hot T_hot - M_ref T_ref = Q (M_NS T_NON - M_ref T_ref)

so ``y = M_hot T_hot - (1 - Q) M_ref T_ref`` is ``T_NON`` times
``x = Q M_NS``. ``T_NON`` is the noise-source path's temperature at P when
on: a property of the source, independent of the receiver. A receiver then
takes ``T_NS = M_NS T_NON - M_load T_load`` above its load
(:func:`tns_as_received`), the quantity ``dicke.tstar_coefficients`` uses.
"""

import numpy as np

from .dicke import mismatch

#: Reference temperature of the ENR definition, K.
T0_K = 290.0


def excess_from_enr(enr_db, pad_db=0.0):
    """Excess noise temperature (K) of a source of ENR ``enr_db`` (dB) seen
    through a matched attenuator of ``pad_db`` dB:
    290 K x 10**((ENR - pad)/10)."""
    return T0_K * 10 ** ((np.asarray(enr_db, float) - pad_db) / 10)


def enr_from_excess(excess_k, pad_db=0.0):
    """Inverse of :func:`excess_from_enr`: the ENR (dB) whose excess behind
    ``pad_db`` dB is ``excess_k``."""
    return pad_db + 10 * np.log10(np.asarray(excess_k, float) / T0_K)


def source_temperature_terms(q, m_hot, m_ref, m_ns, t_hot_k, t_ref_k):
    """Abscissa and ordinate whose slope is the noise source's ``T_NON``.

    ``x = Q M_NS`` and ``y = M_hot T_hot - (1 - Q) M_ref T_ref`` (module
    docstring). ``q`` and the mismatch factors are (..., nchan); the load
    temperatures broadcast against them (e.g. (ncycle, 1) for one value per
    switch cycle). Returns ``(x, y)``."""
    q = np.asarray(q, float)
    y = np.real(m_hot) * t_hot_k - (1.0 - q) * np.real(m_ref) * t_ref_k
    return q * np.real(m_ns), y


def fit_source_temperature(x, y, use=None, min_points=4):
    """Straight line ``y = a + b x`` per channel by least squares
    (:func:`numpy.polyfit` channel by channel, since ``x`` differs between
    channels).

    ``x`` and ``y`` are (ncycle, nchan); ``use`` (ncycle,) selects cycles;
    non-finite points are left out per channel, and a channel with fewer than
    ``min_points`` gets NaN. Returns a dict: intercept ``a``, slope ``b``
    (``T_NON`` for :func:`source_temperature_terms`), the slope's standard
    error ``sb`` from the residuals, the number of points ``n`` and the
    residuals ``res`` (NaN where not fitted)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must both be (ncycle, nchan)")
    use = np.ones(x.shape[0], bool) if use is None else np.asarray(use, bool)
    nchan = x.shape[1]
    a, b, sb = (np.full(nchan, np.nan) for _ in range(3))
    n = np.zeros(nchan, int)
    res = np.full(x.shape, np.nan)
    for k in range(nchan):
        ok = use & np.isfinite(x[:, k]) & np.isfinite(y[:, k])
        n[k] = ok.sum()
        if n[k] < min_points:
            continue
        (b[k], a[k]), cov = np.polyfit(x[ok, k], y[ok, k], 1, cov=True)
        sb[k] = np.sqrt(cov[0, 0])
        res[ok, k] = y[ok, k] - a[k] - b[k] * x[ok, k]
    return {"a": a, "b": b, "sb": sb, "n": n, "res": res}


def tns_as_received(
    t_ex_k, gamma_ns, gamma_rec, gamma_load, t_load_k, t_off_k=None
):
    """``T_NS`` as a receiver takes it above its load:
    ``P_NS - P_load = g T_NS`` with
    ``T_NS = M_NS (T_off + T_ex) - M_load T_load``.

    ``t_ex_k`` is the noise source's excess at P over its off state, whose
    temperature ``t_off_k`` defaults to ``t_load_k`` (the noise path and the
    load sit at nearly the same physical temperature). The reflections are at
    P; the mismatch factors are taken against ``gamma_rec``. All arguments
    broadcast."""
    t_off_k = t_load_k if t_off_k is None else t_off_k
    m_ns = np.real(mismatch(gamma_ns, gamma_rec))
    m_load = np.real(mismatch(gamma_load, gamma_rec))
    return m_ns * (t_off_k + np.asarray(t_ex_k, float)) - m_load * t_load_k


def rescale_tns(t_star, t_cal, m_ant, t_load_k, t_ns_old, t_ns_new):
    """A calibration made with ``t_ns_old`` redone with ``t_ns_new``.

    ``T* = T_load + T_NS Q`` is linear in ``T_NS``, and
    ``T = (T* - (1 - M_load) T_load) / M_ant``
    (:func:`eigsep_cal.dicke.receiver_s11_coefficients`), so
    ``T*_new = T_load + (T_NS,new / T_NS,old) (T*_old - T_load)`` and
    ``T_new = T_old + (T*_new - T*_old) / M_ant``.
    Returns ``(T*_new, T_new)``."""
    t_star_new = t_load_k + (np.asarray(t_ns_new, float) / t_ns_old) * (
        t_star - t_load_k
    )
    return t_star_new, t_cal + (t_star_new - t_star) / m_ant


def switched_ratio_noise(
    p_hot, p_ref, p_non, n_hot, n_ref, n_non, enbw_hz, tau_s
):
    """Radiometer noise of ``Q = (P_hot - P_ref) / (P_NON - P_ref)``, each
    power a mean of ``n_*`` integrations of ``tau_s`` s in a channel of
    ``enbw_hz`` Hz (sigma_P = P / sqrt(enbw tau n)), propagated to first
    order with the three powers independent."""

    def sig(p, n):
        return np.asarray(p, float) / np.sqrt(
            enbw_hz * tau_s * np.asarray(n, float)
        )

    d = np.asarray(p_non, float) - p_ref
    q = (np.asarray(p_hot, float) - p_ref) / d
    var = (
        sig(p_hot, n_hot) ** 2
        + ((q - 1) * sig(p_ref, n_ref)) ** 2
        + (q * sig(p_non, n_non)) ** 2
    )
    return np.sqrt(var) / np.abs(d)
