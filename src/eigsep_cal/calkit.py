"""
Tools for calibrating S11 measurements.

The calibration formalism is described in the MIST instrument paper (Monsalve
et al. 2024; M24). Equations related to OSL calibration of reflection
coefficient measurements are described in Monsalve et al. 2016 (M16).
"""

import numpy as np


def impedance_to_gamma(Z, Z0):
    """
    Convert impedance to reflection coefficient.

    Parameters
    ----------
    Z : complex
        Impedance.
    Z0 : complex
        Characteristic impedance.

    Returns
    -------
    complex
        Reflection coefficient.

    """
    gamma = np.atleast_1d((Z - Z0) / (Z + Z0))
    gamma[np.isinf(Z)] = 1
    if np.isscalar(Z):
        return gamma[0]
    return gamma


def calc_Z_off(Z0, delta_1ghz, freq_Hz):
    """
    Calculate the impedance of the offset for the calibration standard. See
    M16, Eq. 20.

    Parameters
    ----------
    Z0 : float
        Characteristic impedance.
    delta_1ghz : float
        One-way loss at 1 GHz.
    freq_Hz : float
        Frequency in Hz.

    Returns
    -------
    complex
        Impedance of the offset.


    """
    x = delta_1ghz / (4 * np.pi * freq_Hz) * np.sqrt(freq_Hz / 1e9)
    return Z0 + (1 - 1j) * x


def calc_l_x_gamma(Z0, delta_1ghz, delay, freq_Hz):
    """
    Calculate the product of the length and propagation constant of the offset
    for the calibration standard. See M16, Eq. 21.

    Parameters
    ----------
    Z0 : float
        Characteristic impedance.
    delta_1ghz : float
        One-way loss at 1 GHz.
    delay : float
        One-way delay in seconds.
    freq_Hz : float
        Frequency in Hz.

    Returns
    -------
    float
        Product of length and propagation constant.

    """
    x = delay * delta_1ghz / (2 * Z0) * np.sqrt(freq_Hz / 1e9)
    return 2j * np.pi * freq_Hz * delay + (1 + 1j) * x


def network_sparams(gamma_true, gamma_meas):
    """
    Get the S-parameters of a network by comparing the measured reflection
    coefficients of the open, short, and match standards to a model of their
    reflection coefficients (the "true" values). See M16, Eq. 3.

    Parameters
    ----------
    gamma_true : array-like
        True reflection coefficients for the open, short, and match standards.
        These are the unprimed quantities in Eq. 3. First dimension should
        have length 3 (open, short, match).
    gamma_meas : array-like
        Measured reflection coefficients for the open, short, and match. These
        are the primed quantities in Eq. 3. First dimension should have length
        3 (open, short, match).

    Returns
    -------
    sparams : ndarray
        S-parameters in the form [S11, S12 * S21, S22]. We only care about
        the product of S12 and S21, not their individual values.

    """
    gamma_true = np.array(gamma_true, dtype=complex)
    gamma_meas = np.array(gamma_meas, dtype=complex)

    _orig_shape = gamma_true.shape

    # Reshape to (N, 3)
    gamma_true = gamma_true.T  # shape (N, 3)
    gamma_meas = gamma_meas.T  # shape (N, 3)

    # Construct (N, 3, 3) matrix
    mat = np.stack(
        [np.ones_like(gamma_true), gamma_true, gamma_true * gamma_meas],
        axis=-1,
    )  # shape (N, 3, 3)

    gamma_meas = gamma_meas[..., np.newaxis]  # or gamma_meas.reshape(N, 3, 1)

    sparams = np.linalg.solve(mat, gamma_meas)  # shape (N, 3, 1)

    sparams = np.squeeze(sparams, axis=-1)  # shape (N, 3)

    # Reshape back to original
    sparams = sparams.T.reshape(_orig_shape)

    # Compute S12 * S21
    sparams[1] += sparams[0] * sparams[2]

    return np.squeeze(sparams)


def embed_sparams(sparams, gamma):
    """
    Embed S-parameters into a network with reflection coefficient gamma. See
    M16, Eq. 1.

    Parameters
    ----------
    sparams : array-like
        S-parameters in the form [S11, S12 * S21, S22]. We only care about
        the product of S12 and S21, not their individual values.
    gamma : complex
        Intrinsic reflection coefficient.

    Returns
    -------
    gamma_prime : complex
        Embedded reflection coefficient, measured at reference plane.

    """
    s11, s12s21, s22 = sparams
    gamma_prime = s11 + s12s21 * gamma / (1 - s22 * gamma)
    return gamma_prime


def de_embed_sparams(sparams, gamma_prime):
    """
    De-embed S-parameters from a network with measured reflection coefficient
    gamma_prime. See M16, Eq. 2.

    Parameters
    ----------
    sparams : array-like
        S-parameters in the form [S11, S12 * S21, S22]. We only care about
        the product of S12 and S21, not their individual values.
    gamma_prime : complex
        Measured reflection coefficient.

    Returns
    -------
    gamma : complex
        Intrinsic reflection coefficient.

    """
    s11, s12s21, s22 = sparams
    d = gamma_prime - s11
    gamma = d / (s12s21 + s22 * d)
    return gamma


def calibrate(gammas, sprms_dict):
    """
    Calibrate all gammas in gammas dict with respect to all sparams in
    sprms dict. Applicable to both gammas and standards.

    IN
    kit: CalKit object.
    gammas : array that contains all gamma values to be processed.
    sprms_dict : sparams dict to de-embed from the gammas.
    OUT
    returns the calibrated gammas as an array.
    """
    for key, sprm in sprms_dict.items():
        gammas = np.array(
            [de_embed_sparams(sparams=sprm, gamma_prime=g) for g in gammas]
        )
    return gammas


class CalStandard:
    def __init__(self, Z_ter, Z_off, l_x_gamma, Z0=50):
        """
        Useful measured and derived values for a calibration standard. Note
        that all quantities are in SI units.

        Parameters
        ----------
        freq_Hz : float
            Frequency in Hz.
        Z_ter : complex
            Impedance of termination.
        Z_off : complex
            Impedance of offset.
        l_x_gamma : float
           Product of length and propagation constant of offset, denoted
           by l and lowercase gamma in M16. Only the product shows up in
           the equations, its value can be calcuclated with Eq 21 in M16.
        Z0 : float
            Characteristic impedance.

        """
        self.Z0 = Z0
        self.Z_ter = Z_ter
        self.Z_off = Z_off
        self.l_x_gamma = l_x_gamma

    @property
    def gamma_ter(self):
        """
        Reflection coefficient for the termination.

        Returns
        -------
        complex
            Reflection coefficient.

        """
        return impedance_to_gamma(self.Z_ter, self.Z0)

    @property
    def gamma_off(self):
        """
        Reflection coefficient for the offset.

        Returns
        -------
        complex
            Reflection coefficient.

        """
        return impedance_to_gamma(self.Z_off, self.Z0)

    @property
    def gamma(self):
        """
        Combined reflection coefficient for impedance of termination and lossy
        characteristic impedance of offset. See M16, Eq. 18.

        Returns
        -------
        complex
            Reflection coefficient.

        """

        exp = np.exp(-2 * self.l_x_gamma)
        # numerator
        num1 = self.gamma_off * (1 - exp) + self.gamma_ter * exp
        num2 = self.gamma_off**2 * self.gamma_ter
        num = num1 - num2
        # denominator
        den_bracket = self.gamma_off * exp + self.gamma_ter * (1 - exp)
        den = 1 - self.gamma_off * den_bracket
        return num / den


class CalKit:
    def __init__(self, freq_Hz, Z0=50):
        """
        Useful measured and derived values for a calibration kit. Note that all
        quantities are in SI units.

        Parameters
        ----------
        freq_Hz : float
            Frequency in Hz.
        Z0 : float
            Characteristic impedance.

        """
        self.Z0 = Z0
        self.freq_Hz = freq_Hz

    @property
    def omega(self):
        return 2 * np.pi * self.freq_Hz

    def _add_standard(self, Z_ter, delta_1ghz, delay):
        """
        Add a calibration standard to the calibration kit. Users should use
        the add_open and add_short methods instead.

        Parameters
        ----------
        Z_ter : float
            Impedance of termination in ohms.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        Returns
        -------
        CalStandard

        """
        Z_off = calc_Z_off(self.Z0, delta_1ghz, self.freq_Hz)
        lxg = calc_l_x_gamma(self.Z0, delta_1ghz, delay, self.freq_Hz)
        return CalStandard(Z_ter, Z_off, lxg, Z0=self.Z0)

    def add_open(self, C_open, delta_1ghz, delay):
        """
        Add an open standard to the calibration kit.

        Parameters
        ----------
        C_open : float
            Capacitance of open in farads.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        """
        Z_ter = -1j / (self.omega * C_open)
        self.open = self._add_standard(Z_ter, delta_1ghz, delay)

    def add_short(self, L_short, delta_1ghz, delay):
        """
        Add a short standard to the calibration kit.

        Parameters
        ----------
        L_short : float
            Inductance of short in henries.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        """
        Z_ter = 1j * self.omega * L_short
        self.short = self._add_standard(Z_ter, delta_1ghz, delay)

    def add_match(self, Z_match, delta_1ghz, delay):
        """
        Add a match standard to the calibration kit. This is usually assumed
        to be 50 ohms with no loss or delay; however, the actual values may
        differ from this assumption. See M16 for more information.

        Parameters
        ----------
        Z_match : float
            Impedance of match in ohms. This is usally assumed to be 50 ohms;
            however, MIST measures this following M16.
        delta_1ghz : float
            One-way loss at 1 GHz.
        delay : float
            One-way delay of offset in seconds.

        """
        self.match = self._add_standard(Z_match, delta_1ghz, delay)


class BasicLoadStandard:
    """
    Temporary class to hold a load standard for calibration. Will be
    updated with improved models in the future.

    Parameters
    impedance : the impedance of your load. generally is 50Ohm.
    Z0 : the characteristic impedance of your system. default is 50Ohm.

    Properties
    gamma : calls impedance_to_gamma on the impedance of the load and Z0.
    """

    def __init__(self, impedance, Z0=50):
        self.impedance = impedance
        self.Z0 = Z0

    @property
    def gamma(self):
        return impedance_to_gamma(self.impedance, self.Z0)


class S911T(CalKit):
    def __init__(self, freq_Hz, match_resistance=50):
        """
        Values for EIGSEP calibration kit.

        Parameters:
            freq_Hz (np.array of floats) : Frequency Range in Hz.
            match_resistance (float) : Resistance of match standard in ohms.
        """

        Z0 = 50

        # open standard
        c_coefs = (6.18e-45, -226e-36, 2470e-27, -7.425e-15)
        c_open = np.polyval(c_coefs, freq_Hz)
        open_delay = 30.821e-12  # s
        open_loss = 2e9  # Ohm/s

        # short standard
        l_coefs = (-6.13e-42, 303.8e-33, -5010e-24, 27.98e-12)
        l_short = np.polyval(l_coefs, freq_Hz)
        short_delay = 30.688e-12  # s
        short_loss = 2e9  # Ohm/s

        # load standard
        load_Z = 50  # Ohm
        load = np.ones(len(freq_Hz)) * load_Z

        super().__init__(freq_Hz, Z0=Z0)
        self.add_open(c_open, open_loss, open_delay)
        self.add_short(l_short, short_loss, short_delay)
        self.load = BasicLoadStandard(load)

    @property
    def std_gamma(self):
        open_gamma = self.open.gamma
        shor_gamma = self.short.gamma
        load_gamma = self.load.gamma
        gamma = np.vstack([open_gamma, shor_gamma, load_gamma])
        return gamma

    def sparams(self, stds_meas, model=None):
        """
        Return a scattering matrix based on measured and model standards.
        IN
        stds_meas : np.array (3, N)
            The standards reflection coefficients measured at the
            desired reference plane.
        model : np.array (3,N) or None
            If none, the model reflection coefficient given by Copper
            Mountain is used.
        OUT
        np.array (3,N)
            (S11, S12*S21, S22) s matrix.
        """
        if model is None:
            model = self.std_gamma  # get model standards if none are provided
        sparams = network_sparams(model, stds_meas)
        return sparams
