import numpy as np
import pytest

from eigsep_cal.dicke import (
    calc_Tant_star,
    mismatch,
    receiver_s11_coefficients,
    tstar_coefficients,
)
from eigsep_cal.noise_source import (
    enr_from_excess,
    excess_from_enr,
    fit_source_temperature,
    rescale_tns,
    source_temperature_terms,
    switched_ratio_noise,
    tns_as_received,
)

NCHAN = 32


def _gamma(rng, amp, n=NCHAN):
    return amp * rng.uniform(0.2, 1.0, n) * np.exp(2j * np.pi * rng.uniform(size=n))


def _power(gain, t_r, gamma, gamma_rec, t_source):
    """The model of the module: P = g (M T + T_r)."""
    return gain * (mismatch(gamma, gamma_rec) * t_source + t_r)


def test_enr_round_trip_and_nameplate():
    assert excess_from_enr(35.0, pad_db=30.0) == pytest.approx(917.06, abs=0.01)
    assert excess_from_enr(30.0, pad_db=30.0) == pytest.approx(290.0)
    enr = np.array([30.0, 32.3, 35.0])
    np.testing.assert_allclose(enr_from_excess(excess_from_enr(enr, 30.0), 30.0), enr)


def test_fit_recovers_source_temperature_with_mismatch():
    """Synthetic bench: random reflections at P, a hot load stepped over a
    range, a reference load drifting by 1 K. The noise source's temperature at
    P is recovered exactly, with zero intercept."""
    rng = np.random.default_rng(1)
    g_rec = _gamma(rng, 0.4)
    g_hot, g_ref, g_ns = _gamma(rng, 0.1), _gamma(rng, 0.1), _gamma(rng, 0.08)
    gain = rng.uniform(1e3, 3e3, NCHAN)
    t_r = rng.uniform(50, 150, NCHAN)
    t_non = rng.uniform(700, 900, NCHAN)
    ncyc = 40
    t_hot = np.linspace(295, 360, ncyc)[:, None]
    t_ref = 304 + np.linspace(0, 1, ncyc)[:, None]
    p_hot = _power(gain, t_r, g_hot, g_rec, t_hot)
    p_ref = _power(gain, t_r, g_ref, g_rec, t_ref)
    p_non = _power(gain, t_r, g_ns, g_rec, t_non * np.ones((ncyc, 1)))
    q = calc_Tant_star(p_hot, p_non, p_ref, 1.0, 0.0)
    x, y = source_temperature_terms(q, mismatch(g_hot, g_rec), mismatch(g_ref, g_rec),
                                    mismatch(g_ns, g_rec), t_hot, t_ref)
    fit = fit_source_temperature(x, y)
    np.testing.assert_allclose(fit["b"], t_non, rtol=1e-9)
    np.testing.assert_allclose(fit["a"], 0.0, atol=1e-6)
    assert np.all(fit["n"] == ncyc)


def test_fit_masks_cycles_and_short_channels():
    x = np.tile(np.arange(10.0)[:, None], (1, 3))
    y = 2.0 + 5.0 * x
    y[3, 0] = np.nan                      # a missing point in channel 0
    y[:, 2] = np.nan
    y[:3, 2] = 1.0                        # channel 2: too few points
    use = np.ones(10, bool)
    use[-1] = False
    fit = fit_source_temperature(x, y, use)
    np.testing.assert_allclose(fit["b"][:2], 5.0)
    np.testing.assert_allclose(fit["a"][:2], 2.0)
    assert list(fit["n"]) == [8, 9, 3]
    assert np.isnan(fit["b"][2])
    assert np.isnan(fit["res"][-1]).all()


def test_tns_as_received_matches_the_power_difference():
    """T_NS from tns_as_received is exactly (P_NS - P_load) / g in the model,
    and equals the excess when everything is matched."""
    rng = np.random.default_rng(2)
    g_rec, g_ns, g_load = _gamma(rng, 0.7), _gamma(rng, 0.09), _gamma(rng, 0.1)
    gain, t_r = 2.0e3, 80.0
    t_load, t_ex = 300.0, rng.uniform(400, 600, NCHAN)
    p_ns = _power(gain, t_r, g_ns, g_rec, t_load + t_ex)
    p_load = _power(gain, t_r, g_load, g_rec, t_load)
    np.testing.assert_allclose(tns_as_received(t_ex, g_ns, g_rec, g_load, t_load),
                               (p_ns - p_load) / gain, rtol=1e-12)
    zero = np.zeros(NCHAN)
    np.testing.assert_allclose(tns_as_received(t_ex, zero, g_rec, zero, t_load), t_ex)


def test_rescale_tns_matches_rebuilding_the_coefficients():
    rng = np.random.default_rng(3)
    g_ant, g_load, g_rec = _gamma(rng, 0.7), _gamma(rng, 0.1), _gamma(rng, 0.5)
    p_ns, p_load = rng.uniform(5, 6, NCHAN), rng.uniform(3, 4, NCHAN)
    p_ant = rng.uniform(3, 9, NCHAN)
    t_load = 300.0
    t_ns_new = rng.uniform(400, 600, NCHAN)

    def calibrate(t_ns):
        a, b = tstar_coefficients(p_ns, p_load, t_ns, t_load)
        s, o = receiver_s11_coefficients(a, b, t_load, g_ant, g_load, g_rec)
        return a * p_ant + b, np.real(s * p_ant + o)

    ts_old, t_old = calibrate(917.0)
    ts_new, t_new = calibrate(t_ns_new)
    m_ant = np.real(mismatch(g_ant, g_rec))
    got_ts, got_t = rescale_tns(ts_old, t_old, m_ant, t_load, 917.0, t_ns_new)
    np.testing.assert_allclose(got_ts, ts_new, rtol=1e-12)
    np.testing.assert_allclose(got_t, t_new, rtol=1e-10)


def test_switched_ratio_noise_matches_monte_carlo():
    rng = np.random.default_rng(4)
    enbw, tau = 244140.625, 1.07
    p = {"hot": 1.2, "ref": 1.0, "non": 1.8}
    n = {"hot": 8, "ref": 7, "non": 8}
    sig = {k: p[k] / np.sqrt(enbw * tau * n[k]) for k in p}
    draws = {k: p[k] + sig[k] * rng.standard_normal(200000) for k in p}
    q = (draws["hot"] - draws["ref"]) / (draws["non"] - draws["ref"])
    pred = switched_ratio_noise(p["hot"], p["ref"], p["non"], n["hot"], n["ref"], n["non"], enbw, tau)
    assert np.std(q) == pytest.approx(pred, rel=0.02)
