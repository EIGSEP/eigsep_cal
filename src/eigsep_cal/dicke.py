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
        Realistic assumption of temperature of the noise source.
    T_load : float
        Realistic assumption of temperature of the load.

    Returns
    -------
    Tant_star : array-like
        Uncalibrated antenna temperature.
    """
    Tant_star = (P_ant - P_load) / (P_ns - P_load) * T_ns + T_load
    return Tant_star
