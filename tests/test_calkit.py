import numpy as np

from eigsep_cal import calkit as cal


def test_impedance_to_gamma():
    # perfect match gives 0 reflection
    Z = 50
    Z0 = 50
    gamma = cal.impedance_to_gamma(Z, Z0)
    assert np.isclose(gamma, 0)

    # open circuit, reflection is 1 with phase 0
    Z = np.inf
    assert np.isclose(cal.impedance_to_gamma(Z, Z0), 1)

    # short circuit, reflection is 1 with phase 180
    Z = 0
    assert np.isclose(cal.impedance_to_gamma(Z, Z0), -1)


def test_calc_Z_off():
    Z0 = 50
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz

    # no loss gives Zoff = Z0
    assert np.allclose(cal.calc_Z_off(Z0, 0, f_Hz), Z0)

    # lossy line
    delta = 2e9  # loss
    Zoff = cal.calc_Z_off(Z0, delta, f_Hz)
    dZ = Zoff - Z0
    # real and imag parts are equal in magnitude, opposite in sign
    assert np.allclose(dZ.real, -dZ.imag)
    # dZ scale with 1/sqrt(f)
    assert np.allclose(dZ, dZ[0] * np.sqrt(f_Hz[0] / f_Hz))


def test_calc_l_x_gamma():
    Z0 = 50
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    omega = 2 * np.pi * f_Hz

    # no loss means l*gamma is just due to delay
    delay = 30e-12  # 30 ps
    assert np.allclose(
        cal.calc_l_x_gamma(Z0, 0, delay, f_Hz), 1j * omega * delay
    )

    # lgamma is 0 if there's no delay
    delta = 2e9  # loss
    assert np.allclose(cal.calc_l_x_gamma(Z0, delta, 0, f_Hz), 0)

    # generic case with loss and delay
    lxg_dly = 1j * omega * delay  # l*gamma due to delay
    d = cal.calc_l_x_gamma(Z0, delta, delay, f_Hz) - lxg_dly
    assert np.allclose(d.real, d.imag)  # remainder has equal real/imag
    # and scale with sqrt(f)
    assert np.allclose(d, d[0] * np.sqrt(f_Hz / f_Hz[0]))


def test_gamma():
    Z0 = 50
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    Z_ters = [0, Z0, np.inf, 125]  # short, match, open, random
    # no loss means gamma = gamma_termination
    Z_off = Z0
    for Z_ter in Z_ters:
        cal_std = cal.CalStandard(Z_ter, Z_off, 0, Z0=Z0)
        assert np.allclose(cal_std.gamma, cal_std.gamma_ter)

    # infinite loss gives gamma = gamma_offset
    Z_off = 1e12 / np.sqrt(f_Hz) * (1 - 1j)  # very high loss
    for Z_ter in Z_ters:
        cal_std = cal.CalStandard(Z_ter, Z_off, np.inf, Z0=Z0)
        assert np.allclose(cal_std.gamma, cal_std.gamma_off)

    # Z_ter = Z0
    delta = 2e9  # loss
    delay = 30e-12  # delay
    Z_off = cal.calc_Z_off(Z0, delta, f_Hz)
    lxg = cal.calc_l_x_gamma(Z0, delta, delay, f_Hz)
    cal_std = cal.CalStandard(Z0, Z_off, lxg, Z0=Z0)
    exp = np.exp(-2 * lxg)
    gamma_off = cal_std.gamma_off
    gamma_expected = gamma_off * (1 - exp) / (1 - gamma_off**2 * exp)
    assert np.allclose(cal_std.gamma, gamma_expected)

    # Z_off = Z0
    for Z_ter in Z_ters:
        cal_std = cal.CalStandard(Z_ter, Z0, lxg, Z0=Z0)
        assert np.allclose(cal_std.gamma, cal_std.gamma_ter * exp)


def test_calkit():
    # nominal values
    delta = 2e9  # loss
    delay = 30e-12  # delay
    Z0 = 50  # characteristic impedance
    f_Hz = np.arange(1, 126) * 1e6  # 1 MHz to 125 MHz
    calkit = cal.CalKit(f_Hz, Z0=Z0)
    assert np.allclose(calkit.Z0, Z0)
    assert np.allclose(calkit.omega, 2 * np.pi * f_Hz)
    # eq 20 in Monsalve et al 2016
    Z_off = Z0 + (1 - 1j) * delta / (4 * np.pi * f_Hz) * np.sqrt(f_Hz / 1e9)
    gamma_off = cal.impedance_to_gamma(Z_off, Z0)

    # ideal case for open: C_open = 0 -> gamma_open = 1
    calkit.add_open(0, 0, 0)
    assert np.all(calkit.open.gamma == 1)
    assert np.all(calkit.open.gamma_off == 0)
    assert np.all(calkit.open.gamma_ter == 1)

    # ideal case for short: L_short = 0 -> gamma_short = -1
    calkit.add_short(0, 0, 0)
    assert np.all(calkit.short.gamma == -1)
    assert np.all(calkit.short.gamma_off == 0)
    assert np.all(calkit.short.gamma_ter == -1)
    calkit.add_match(Z0, 0, 0)
    assert np.all(calkit.match.gamma == 0)
    assert np.all(calkit.match.gamma_off == 0)
    assert np.all(calkit.match.gamma_ter == 0)

    # frequency dependent C_open
    C_open = 1e-15 + 1e-27 * f_Hz + 1e-36 * f_Hz**2 + 1e-45 * f_Hz**3
    calkit.add_open(C_open, delta, delay)
    # eq 22 in Monsalve et al 2016
    Z_open = -1j / (2 * np.pi * f_Hz * C_open)
    gamma_ter_open = cal.impedance_to_gamma(Z_open, Z0)
    assert np.allclose(calkit.open.gamma_ter, gamma_ter_open)
    assert np.allclose(calkit.open.gamma_off, gamma_off)

    # frequency dependent L_short
    L_short = 1e-12 + 1e-24 * f_Hz + 1e-33 * f_Hz**2 + 1e-42 * f_Hz**3
    calkit.add_short(L_short, delta, delay)
    # eq 23 in Monsalve et al 2016
    Z_short = 1j * 2 * np.pi * f_Hz * L_short
    gamma_ter_short = cal.impedance_to_gamma(Z_short, Z0)
    assert np.allclose(calkit.short.gamma_ter, gamma_ter_short)
    assert np.allclose(calkit.short.gamma_off, gamma_off)

    # match with loss and delay and Z != Z0
    Z_match = Z0 * 1.05  # 5% mismatch
    calkit.add_match(Z_match, delta, delay)
    gamma_ter_match = cal.impedance_to_gamma(Z_match, Z0)
    assert np.allclose(calkit.match.gamma_ter, gamma_ter_match)
    assert np.allclose(calkit.match.gamma_off, gamma_off)


def test_network_sparams():
    Nfreq = 125
    # ideal values for open, short, match
    gamma_true = np.empty((3, Nfreq), dtype=complex)
    gamma_true[0] = 1  # open
    gamma_true[1] = -1  # short
    gamma_true[2] = 0  # match
    # case with ideal values, network sparams no influence
    gamma_meas = np.empty((3, Nfreq), dtype=complex)
    gamma_meas[0] = 1  # open
    gamma_meas[1] = -1  # short
    gamma_meas[2] = 0  # match
    sparams = cal.network_sparams(gamma_true, gamma_meas)
    # in this case s11=s22=0, s12=s21=1
    assert np.all(sparams[0] == 0)
    assert np.all(sparams[1] == 1)
    assert np.all(sparams[2] == 0)


def test_S911T():
    # recalculate standards, compare to the ones in S911T

    Z0 = 50
    fake_freqs = np.linspace(50e6, 250e6, 1001)
    calkit = cal.S911T(freq_Hz=fake_freqs)

    # open standard
    c_open = (
        -7.425e-15
        + 2470e-27 * fake_freqs
        - 226e-36 * fake_freqs**2
        + 6.18e-45 * fake_freqs**3
    )
    c_delay = 30.821e-12  # s
    c_loss = 2e9  # Ohm/s

    Z_off_open = Z0 + (1 - 1j) * c_loss / (4 * np.pi * fake_freqs) * np.sqrt(
        fake_freqs / 1e9
    )
    gamma_off_open = cal.impedance_to_gamma(Z_off_open, Z0)
    Z_ter_open = -1j / (2 * np.pi * fake_freqs * c_open)
    gamma_ter_open = cal.impedance_to_gamma(Z_ter_open, Z0)
    g_l_open = 1j * 2 * np.pi * fake_freqs * c_delay + (
        1 + 1j
    ) * c_delay * c_loss / (2 * Z0) * np.sqrt(fake_freqs / 1e9)
    gamma_open = gamma_off_open * (
        1 - np.exp(-2 * g_l_open) - gamma_off_open * gamma_ter_open
    ) + gamma_ter_open * np.exp(-2 * g_l_open)
    gamma_open /= 1 - gamma_off_open * (
        np.exp(-2 * g_l_open) * gamma_off_open
        + gamma_ter_open * (1 - np.exp(-2 * g_l_open))
    )

    assert np.allclose(calkit.open.gamma_ter, gamma_ter_open)
    assert np.allclose(calkit.open.gamma_off, gamma_off_open)
    assert np.allclose(calkit.open.gamma, gamma_open)

    # short standard
    l_short = (
        27.98e-12
        - 5010e-24 * fake_freqs
        + 303.8e-33 * fake_freqs**2
        - 6.13e-42 * fake_freqs**3
    )
    l_delay = 30.688e-12  # s
    l_loss = 2e9  # Ohm/s

    Z_off_short = Z0 + (1 - 1j) * l_loss / (4 * np.pi * fake_freqs) * np.sqrt(
        fake_freqs / 1e9
    )
    gamma_off_short = cal.impedance_to_gamma(Z_off_short, Z0)
    Z_ter_short = 1j * 2 * np.pi * fake_freqs * l_short
    gamma_ter_short = cal.impedance_to_gamma(Z_ter_short, Z0)

    g_l_short = 1j * 2 * np.pi * fake_freqs * l_delay + (
        1 + 1j
    ) * l_delay * l_loss / (2 * Z0) * np.sqrt(fake_freqs / 1e9)

    gamma_short = gamma_off_short * (
        1 - np.exp(-2 * g_l_short) - gamma_off_short * gamma_ter_short
    ) + gamma_ter_short * np.exp(-2 * g_l_short)
    gamma_short /= 1 - gamma_off_short * (
        np.exp(-2 * g_l_short) * gamma_off_short
        + gamma_ter_short * (1 - np.exp(-2 * g_l_short))
    )

    assert np.allclose(calkit.short.gamma_ter, gamma_ter_short)
    assert np.allclose(calkit.short.gamma_off, gamma_off_short)
    assert np.allclose(calkit.short.gamma, gamma_short)


def test_sparams():
    fake_freqs = np.linspace(50e6, 250e6, 1001)
    calkit = cal.S911T(freq_Hz=fake_freqs)
    gamma = calkit.std_gamma

    # gamma equal to self.gamma, yielding a perfect through Smatrix
    sprms = calkit.sparams(gamma)
    assert np.allclose(sprms[0], 0)  # s11
    assert np.allclose(sprms[1], 1)  # s12s21
    assert np.allclose(sprms[2], 0)  # s22


def test_sparams_with_explicit_model():
    """Passing model as a numpy array should not raise ValueError."""
    fake_freqs = np.linspace(50e6, 250e6, 1001)
    calkit = cal.S911T(freq_Hz=fake_freqs)
    gamma = calkit.std_gamma

    sprms = calkit.sparams(gamma, model=gamma)
    assert np.allclose(sprms[0], 0)  # s11
    assert np.allclose(sprms[1], 1)  # s12s21
    assert np.allclose(sprms[2], 0)  # s22
