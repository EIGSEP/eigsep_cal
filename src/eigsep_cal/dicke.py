"""
Tools for Dicke switching calibration.
"""

import numpy as np


def calc_Tant_star(P_ant, P_ns, P_load, T_ns, T_load):
    """
    Calculate the *uncalibrated* antenna temperature, Tant*, using
    data from Dicke switching measurements. This is Equation 1 in
    Monsalve+2017.

    Here noise source is really (load + noise source) and load is
    really (load + noise source off). So the difference is the noise
    source contribution.

    Parameters
    ----------
    P_ant : array-like
        Power measured from the antenna.
    P_ns : array-like
        Power measured from the noise source.
    P_load : array-like
        Power measured from the load.
    T_ns : float
        Realistic assumption of the excess temperature of the noise
        source above the load.
    T_load : float
        Realistic assumption of temperature of the load.

    Returns
    -------
    Tant_star : array-like
        Uncalibrated antenna temperature.
    """
    Tant_star = (P_ant - P_load) / (P_ns - P_load) * T_ns + T_load
    return Tant_star


def mismatch(gamma, gamma_rec):
    """
    Fraction of a source's available power the receiver takes, relative
    to a matched source: ``(1 - |G|^2) / |1 - G G_rec|^2``.

    The receiver's own ``1 - |G_rec|^2`` is common to every source and
    is left in the gain.

    Parameters
    ----------
    gamma : array-like, complex
        Source reflection coefficient at the receiver reference plane.
    gamma_rec : array-like, complex
        Receiver reflection coefficient at the same plane.

    Returns
    -------
    M : array-like
        Mismatch factor. Can exceed 1 where ``|1 - G G_rec| < 1``.
    """
    gamma = np.asarray(gamma)
    return (1 - np.abs(gamma) ** 2) / np.abs(1 - gamma * gamma_rec) ** 2


def tstar_coefficients(P_ns, P_load, T_ns, T_load):
    """
    ``calc_Tant_star`` as a straight line in the measured power:
    ``Tant* = scale * P + offset``.

    Every quantity but the sky power is fixed by the calibration
    measurements, so the scale can be built once per calibration and
    applied to any integration, of any switch state, afterwards.

    Parameters
    ----------
    P_ns, P_load : array-like
        Powers in the noise-source and load states.
    T_ns : float or array-like
        Excess temperature of the noise-source state over the load.
    T_load : float or array-like
        Physical temperature of the load.

    Returns
    -------
    scale, offset : array-like
        In K per unit power and K. Where ``P_ns == P_load`` the scale is
        inf or NaN, as in ``calc_Tant_star``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = T_ns / (np.asarray(P_ns) - P_load)
    return scale, T_load - scale * P_load


def receiver_s11_coefficients(
    scale_star, offset_star, T_load, gamma_ant, gamma_load, gamma_rec
):
    """
    Antenna, ambient-load and receiver reflection corrections applied to
    the ``Tant*`` line of :func:`tstar_coefficients`, giving
    ``T_ant = scale * P_ant + offset``.

    The powers are taken as ``P_ant = g (M_ant T_ant + T_r)``,
    ``P_load = g (M_load T_load + T_r)`` and ``P_ns - P_load = g T_ns``,
    with ``M`` from :func:`mismatch`, so that
    ``Tant* = M_ant T_ant + (1 - M_load) T_load`` and
    ``T_ant = [Tant* - (1 - M_load) T_load] / M_ant``.

    This is the minimal tier of the D5 calibration: equal gain on every
    switch path, no receiver noise waves, and ``T_ns`` the excess the
    receiver actually takes. ``T_ant`` is the source temperature at the
    reference plane: it includes everything between the antenna and
    that plane (balun, coax, switch path), none of it removed.

    Returns
    -------
    scale, offset : array-like
        Valid for antenna-state integrations only; ``M_ant`` belongs to
        the antenna.
    """
    m_ant = mismatch(gamma_ant, gamma_rec)
    m_load = mismatch(gamma_load, gamma_rec)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            scale_star / m_ant,
            (offset_star - (1 - m_load) * T_load) / m_ant,
        )
