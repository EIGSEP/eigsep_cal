import numpy as np

from eigsep_cal.dicke import calc_Tant_star


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
