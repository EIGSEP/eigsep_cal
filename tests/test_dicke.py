import numpy as np

from eigsep_cal.dicke import (
    calc_Tant_star,
    mismatch,
    receiver_s11_coefficients,
    tstar_coefficients,
)


def test_calc_Tant_star_recovers_linear_receiver():
    rng = np.random.default_rng(0)
    gain = rng.uniform(0.5, 2.0, size=64)
    T_rx = rng.uniform(20.0, 80.0, size=64)
    T_load, T_ns, T_ant = 300.0, 400.0, rng.uniform(150, 5000, 64)

    def power(T):
        return gain * (T + T_rx)

    got = calc_Tant_star(
        power(T_ant), power(T_load + T_ns), power(T_load), T_ns, T_load
    )
    np.testing.assert_allclose(got, T_ant, rtol=1e-12)


def test_calc_Tant_star_limits():
    P_load, P_ns = 3.0, 7.0
    assert calc_Tant_star(P_load, P_ns, P_load, 400.0, 300.0) == 300.0
    assert calc_Tant_star(P_ns, P_ns, P_load, 400.0, 300.0) == 700.0


def _gammas(rng, n, rmax):
    r = rng.uniform(0, rmax, n)
    return r * np.exp(1j * rng.uniform(0, 2 * np.pi, n))


def test_tstar_coefficients_are_calc_Tant_star():
    rng = np.random.default_rng(1)
    P_ant, P_ns, P_load = rng.uniform(1, 2, (3, 32)) * [[1], [3], [0.5]]
    T_ns, T_load = 917.0, rng.uniform(290, 305, 32)
    scale, offset = tstar_coefficients(P_ns, P_load, T_ns, T_load)
    np.testing.assert_allclose(
        scale * P_ant + offset,
        calc_Tant_star(P_ant, P_ns, P_load, T_ns, T_load),
        rtol=1e-12,
    )


def test_mismatch_is_one_for_a_matched_pair_and_bounded_by_the_source():
    rng = np.random.default_rng(2)
    g = _gammas(rng, 64, 0.9)
    np.testing.assert_allclose(mismatch(np.zeros(4), _gammas(rng, 4, 0.9)), 1)
    np.testing.assert_allclose(mismatch(g, 0), 1 - np.abs(g) ** 2)


def test_receiver_s11_coefficients_invert_their_own_model():
    # Data built exactly as the docstring states the model; the
    # correction has to give back the antenna temperature put in.
    rng = np.random.default_rng(3)
    n = 64
    g, T_r = rng.uniform(0.5, 2, n), rng.uniform(50, 500, n)
    T_ant, T_load, T_ns = rng.uniform(100, 8000, n), 299.0, 917.0
    G_ant, G_load, G_rec = (_gammas(rng, n, r) for r in (0.8, 0.1, 0.7))
    m_ant, m_load = mismatch(G_ant, G_rec), mismatch(G_load, G_rec)
    P_ant = g * (m_ant * T_ant + T_r)
    P_load = g * (m_load * T_load + T_r)
    P_ns = P_load + g * T_ns
    a, b = tstar_coefficients(P_ns, P_load, T_ns, T_load)
    scale, offset = receiver_s11_coefficients(
        a, b, T_load, G_ant, G_load, G_rec
    )
    np.testing.assert_allclose(scale * P_ant + offset, T_ant, rtol=1e-10)


def test_receiver_s11_coefficients_reduce_to_tstar_when_matched():
    a, b = np.array([2.0, 3.0]), np.array([-10.0, 5.0])
    z = np.zeros(2)
    scale, offset = receiver_s11_coefficients(a, b, 300.0, z, z, z)
    np.testing.assert_allclose(scale, a)
    np.testing.assert_allclose(offset, b)
