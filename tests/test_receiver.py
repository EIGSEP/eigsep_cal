import numpy as np
import pytest

from eigsep_cal.receiver import ReceiverModel, noise_wave_map

N_TIME, N_FREQ = 3, 4


def _rb_kwargs():
    ones = np.ones(N_FREQ)
    return dict(
        t_unc_k=100 * ones,
        t_cos_k=10 * ones,
        t_sin_k=-5 * ones,
        t0_k=50 * ones,
    )


def _gain():
    return np.full((N_TIME, N_FREQ), 2.0)


class TestNoiseWaveMap:
    def test_matched_receiver_limit(self):
        c = 12.0 - 7.0j
        t0, t_unc, t_cos, t_sin = noise_wave_map(0.0, 80.0, 60.0, c)
        assert t0 == 80.0 and t_unc == 60.0
        assert t_cos == 2 * c.real and t_sin == -2 * c.imag

    def test_matches_notebook_003_formulas(self):
        """Port of notebook 003 rb_params (factor 2 in T_unc)."""
        rng = np.random.default_rng(3)
        g = 0.4 * np.exp(2j * np.pi * rng.uniform(size=50))
        tr, tl = rng.uniform(10, 500, (2, 50))
        c = rng.uniform(-50, 50, 50) + 1j * rng.uniform(-50, 50, 50)
        tc, tsn = c.real, -c.imag
        s = np.sqrt(1 - np.abs(g) ** 2)
        want = (
            (1 - np.abs(g) ** 2) * tr,
            tl + np.abs(g) ** 2 * tr + 2 * (tc * g.real - tsn * g.imag),
            2 * s * (tc + tr * g.real),
            2 * s * (tsn - tr * g.imag),
        )
        for got, ref in zip(noise_wave_map(g, tr, tl, c), want):
            np.testing.assert_allclose(got, ref, rtol=1e-12)


class TestReceiverModel:
    def test_rogers_bowman_roundtrip(self):
        rx = ReceiverModel(
            gamma_rec=0.1 * np.ones(N_FREQ), gain=_gain(), **_rb_kwargs()
        )
        assert (rx.n_time, rx.n_freq) == (N_TIME, N_FREQ)
        t_unc, t_cos, t_sin, t0 = rx.rogers_bowman()
        np.testing.assert_array_equal(t_unc, 100.0)
        np.testing.assert_array_equal(t0, 50.0)

    def test_intrinsic_is_mapped(self):
        g = 0.2j * np.ones(N_FREQ)
        rx = ReceiverModel(
            gamma_rec=g,
            gain=_gain(),
            parameterisation="intrinsic",
            t_r_k=80 * np.ones(N_FREQ),
            t_l_k=60 * np.ones(N_FREQ),
            c_k=(5 + 3j) * np.ones(N_FREQ),
        )
        t0, t_unc, t_cos, t_sin = noise_wave_map(g, 80.0, 60.0, 5 + 3j)
        got = rx.rogers_bowman()
        for a, b in zip(got, (t_unc, t_cos, t_sin, t0)):
            np.testing.assert_allclose(a, b, rtol=1e-14)

    def test_mixed_parameterisations_raise(self):
        with pytest.raises(ValueError, match="intrinsic"):
            ReceiverModel(
                gamma_rec=np.zeros(N_FREQ),
                gain=_gain(),
                t_r_k=np.ones(N_FREQ),
                **_rb_kwargs(),
            )

    def test_missing_field_raises(self):
        kwargs = _rb_kwargs()
        del kwargs["t0_k"]
        with pytest.raises(ValueError, match="t0_k"):
            ReceiverModel(gamma_rec=np.zeros(N_FREQ), gain=_gain(), **kwargs)

    def test_gain_must_be_positive(self):
        gain = _gain()
        gain[0, 0] = 0.0
        with pytest.raises(ValueError, match="gain"):
            ReceiverModel(
                gamma_rec=np.zeros(N_FREQ), gain=gain, **_rb_kwargs()
            )

    def test_shapes_checked(self):
        kwargs = _rb_kwargs()
        kwargs["t_cos_k"] = np.ones(N_FREQ + 1)
        with pytest.raises(ValueError, match="t_cos_k"):
            ReceiverModel(gamma_rec=np.zeros(N_FREQ), gain=_gain(), **kwargs)

    def test_path_gain_ratio(self):
        rx = ReceiverModel(
            gamma_rec=np.zeros(N_FREQ),
            gain=_gain(),
            path_gain_ratio={"RFNON": 1.1 * np.ones(N_FREQ)},
            **_rb_kwargs(),
        )
        np.testing.assert_array_equal(rx.path_gain("RFNON"), 1.1)
        np.testing.assert_array_equal(rx.path_gain("RFANT"), 1.0)
        with pytest.raises(ValueError, match="VNAANT"):
            ReceiverModel(
                gamma_rec=np.zeros(N_FREQ),
                gain=_gain(),
                path_gain_ratio={"VNAANT": np.ones(N_FREQ)},
                **_rb_kwargs(),
            )

    def test_path_gain_ratio_is_read_only(self):
        rx = ReceiverModel(
            gamma_rec=np.zeros(N_FREQ),
            gain=_gain(),
            path_gain_ratio={"RFNON": 1.1 * np.ones(N_FREQ)},
            **_rb_kwargs(),
        )
        with pytest.raises(TypeError):
            rx.path_gain_ratio["RFANT"] = -np.ones(N_FREQ)
        with pytest.raises(TypeError):
            rx.path_gain_ratio["RFNON"] = np.ones(N_FREQ)
        with pytest.raises(ValueError):
            rx.path_gain("RFANT")[0] = 5.0
