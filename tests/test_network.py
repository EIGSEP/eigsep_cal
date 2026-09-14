import numpy as np
import pytest

from eigsep_cal.network import SParams, embed

N = 5


def _sp(s11, s12s21, s22, name="test"):
    ones = np.ones(N)
    return SParams(s11 * ones, s12s21 * ones, s22 * ones, name)


def _lossless_reciprocal(rng, n):
    """Random symmetric unitary S = O diag(d1, d2) O^T, well conditioned."""
    theta = rng.uniform(0.3, 1.2, n)
    c, s = np.cos(theta), np.sin(theta)
    phi1 = rng.uniform(0, 2 * np.pi, n)
    phi2 = phi1 + rng.uniform(1.0, 5.0, n)
    d1, d2 = np.exp(1j * phi1), np.exp(1j * phi2)
    s11 = c * c * d1 + s * s * d2
    s22 = s * s * d1 + c * c * d2
    s12 = c * s * (d1 - d2)
    return SParams(s11, s12**2, s22, "lossless")


class TestSParams:
    def test_arrays_are_complex_and_readonly(self):
        sp = SParams([0.1, 0.2], [0.9, 0.8], [0.0, 0.1], "RFANT")
        assert sp.s11.dtype == np.complex128
        with pytest.raises(ValueError):
            sp.s22[0] = 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="share one shape"):
            SParams([0.1, 0.2], [0.9], [0.0, 0.1], "RFANT")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            SParams([0.1], [0.9], [0.0], "")


class TestEmbed:
    def test_matched_lossless_line_rotates_gamma(self):
        phase = np.exp(-2j * np.pi * np.linspace(0, 1, N))
        sp = SParams(np.zeros(N), phase, np.zeros(N), "line")
        gamma_t = 0.3 * np.exp(0.4j) * np.ones(N)

        gamma_s, t_s = embed(gamma_t, 500.0, sp, 290.0)

        np.testing.assert_allclose(gamma_s, gamma_t * phase, atol=1e-15)
        np.testing.assert_allclose(t_s, 500.0, rtol=1e-14)

    def test_matched_pad_is_the_spec_formula(self):
        """Spec § 4.3: a matched pad gives T_t / L + (1 - 1/L) T_p."""
        loss = 10 ** (30 / 10)
        gamma_s, t_s = embed(np.zeros(N), 1e6, _sp(0, 1 / loss, 0), 300.0)
        np.testing.assert_allclose(
            t_s, 1e6 / loss + (1 - 1 / loss) * 300.0, rtol=1e-14
        )
        np.testing.assert_array_equal(gamma_s, 0)

    def test_lossless_network_passes_temperature_unchanged(self):
        """Available gain is 1 for any lossless reciprocal two-port."""
        rng = np.random.default_rng(1)
        n = 200
        sp = _lossless_reciprocal(rng, n)
        gamma_t = (
            0.9
            * np.sqrt(rng.uniform(0, 1, n))
            * np.exp(2j * np.pi * rng.uniform(0, 1, n))
        )

        gamma_s, t_s = embed(gamma_t, 1000.0, sp, 0.0)

        np.testing.assert_allclose(t_s, 1000.0, rtol=1e-9)
        # The same formula with S11 in the denominator is not unitary here.
        wrong = (
            np.abs(sp.s12s21)
            * (1 - np.abs(gamma_t) ** 2)
            / (np.abs(1 - sp.s11 * gamma_t) ** 2 * (1 - np.abs(gamma_s) ** 2))
        )
        assert np.max(np.abs(wrong - 1)) > 1e-2

    def test_equal_temperatures_are_preserved(self):
        rng = np.random.default_rng(2)
        lossless = _lossless_reciprocal(rng, N)
        lossy = SParams(
            lossless.s11, 0.5 * lossless.s12s21, lossless.s22, "lossy"
        )
        _, t_s = embed(0.2 * np.ones(N), 296.0, lossy, 296.0)
        np.testing.assert_allclose(t_s, 296.0, rtol=1e-12)

    def test_two_pads_cascade_to_one(self):
        l1, l2, t_p, t_t = 2.0, 5.0, 300.0, 4000.0
        g1, t1 = embed(np.zeros(N), t_t, _sp(0, 1 / l1, 0), t_p)
        _, t2 = embed(g1, t1, _sp(0, 1 / l2, 0), t_p)
        _, t12 = embed(np.zeros(N), t_t, _sp(0, 1 / (l1 * l2), 0), t_p)
        np.testing.assert_allclose(t2, t12, rtol=1e-13)
