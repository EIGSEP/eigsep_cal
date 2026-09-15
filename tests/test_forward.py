import numpy as np
import pytest

from eigsep_cal.conventions import D5_ENBW_HZ
from eigsep_cal.forward import Source, power, radiometer_noise
from eigsep_cal.receiver import ReceiverModel, noise_wave_map

N_TIME, N_FREQ = 3, 4


def _absorbed_exact(gs, ts, gin, cn):
    """Direct solution of the four wave equations (notebook 003 § A).

    Unknowns [r1, l1, r2, l2]: r1/l1 between source and noise block,
    r2/l2 between noise block and amplifier. Inputs u = [A_R, A_L, A_S],
    with cn[i, j] = <u_i* u_j> for i, j in (R, L).
    """
    m = np.array(
        [[1, -gs, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1], [0, 0, -gin, 1]],
        dtype=complex,
    )
    b = np.array(
        [
            [0, 0, np.sqrt(1 - abs(gs) ** 2)],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=complex,
    )
    w = np.linalg.solve(m, b)[2]  # r2 = w @ u
    cov = np.zeros((3, 3), complex)
    cov[:2, :2] = cn
    cov[2, 2] = ts
    return (1 - abs(gin) ** 2) * np.real(np.conj(w) @ cov @ w)


def _random_networks(n, seed=42):
    rng = np.random.default_rng(seed)

    def gamma(rmax):
        return np.sqrt(rng.uniform(0, rmax**2, n)) * np.exp(
            2j * np.pi * rng.uniform(size=n)
        )

    gin, gs = gamma(0.6), gamma(0.95)
    ts = rng.uniform(0, 5000, n)
    tr, tl = rng.uniform(10, 500, (2, n))
    c = (
        np.sqrt(rng.uniform(size=n))
        * np.exp(2j * np.pi * rng.uniform(size=n))
        * np.sqrt(tr * tl)
    )  # <A_R* A_L>, |correlation| <= 1
    return gin, gs, ts, tr, tl, c


def _one_sample(gin, gs, ts, **noise):
    """Each network is one frequency channel of a single time sample."""
    n = gin.size
    receiver = ReceiverModel(gamma_rec=gin, gain=np.ones((1, n)), **noise)
    sources = {"RFANT": Source(gs[None], ts[None])}
    return power(sources, receiver, ["RFANT"], [0.0])[0]


class TestWaveEquation:
    def test_power_equals_wave_equation_solution(self):
        """Spec § 10: eq. Ps with eq. map, to 1e-11 K (notebook 003 § A)."""
        gin, gs, ts, tr, tl, c = _random_networks(2000)
        exact = np.array(
            [
                _absorbed_exact(
                    gs[i],
                    ts[i],
                    gin[i],
                    np.array([[tr[i], c[i]], [np.conj(c[i]), tl[i]]]),
                )
                for i in range(gin.size)
            ]
        )

        got = _one_sample(
            gin,
            gs,
            ts,
            parameterisation="intrinsic",
            t_r_k=tr,
            t_l_k=tl,
            c_k=c,
        )

        np.testing.assert_allclose(got, exact, rtol=0, atol=1e-11)

    def test_map_without_factor_two_is_wrong(self):
        """Spec § 10: the map as printed by Bucher+2026 fails the check."""
        gin, gs, ts, tr, tl, c = _random_networks(2000)
        t0, t_unc, t_cos, t_sin = noise_wave_map(gin, tr, tl, c)
        printed_t_unc = t_unc - np.real(gin * np.conj(c))  # factor 1

        right = _one_sample(
            gin,
            gs,
            ts,
            t_unc_k=t_unc,
            t_cos_k=t_cos,
            t_sin_k=t_sin,
            t0_k=t0,
        )
        wrong = _one_sample(
            gin,
            gs,
            ts,
            t_unc_k=printed_t_unc,
            t_cos_k=t_cos,
            t_sin_k=t_sin,
            t0_k=t0,
        )

        assert np.max(np.abs(wrong - right)) > 1.0


def _receiver(**overrides):
    kwargs = dict(
        gamma_rec=np.zeros(N_FREQ),
        gain=np.ones((N_TIME, N_FREQ)),
        t_unc_k=np.full(N_FREQ, 100.0),
        t_cos_k=np.full(N_FREQ, 10.0),
        t_sin_k=np.full(N_FREQ, -5.0),
        t0_k=np.full(N_FREQ, 50.0),
    )
    kwargs.update(overrides)
    return ReceiverModel(**kwargs)


def _sources():
    return {
        "RFANT": Source(
            np.full((1, N_FREQ), 0.3j), np.full((1, N_FREQ), 800.0)
        ),
        "RFAMB": Source(np.zeros((1, N_FREQ)), np.full((1, N_FREQ), 296.0)),
    }


STATE = ["RFANT", "RFAMB", "RFANT"]
TIMES = [0.0, 60.0, 120.0]


class TestPower:
    def test_matched_source_and_receiver(self):
        got = power(_sources(), _receiver(), STATE, TIMES)
        np.testing.assert_allclose(got[1], 296.0 + 50.0, rtol=1e-14)

    def test_gain_and_path_gain_scale(self):
        base = power(_sources(), _receiver(), STATE, TIMES)
        scaled = power(
            _sources(),
            _receiver(
                gain=np.full((N_TIME, N_FREQ), 3.0),
                path_gain_ratio={"RFANT": np.full(N_FREQ, 1.5)},
            ),
            STATE,
            TIMES,
        )
        np.testing.assert_allclose(scaled[[0, 2]], 4.5 * base[[0, 2]])
        np.testing.assert_allclose(scaled[1], 3.0 * base[1])

    def test_additive_term_only_on_its_state(self):
        base = power(_sources(), _receiver(), STATE, TIMES)
        comb = np.linspace(1.0, 2.0, N_FREQ)
        got = power(
            _sources(), _receiver(), STATE, TIMES, additive={"RFANT": comb}
        )
        np.testing.assert_allclose(got[[0, 2]], base[[0, 2]] + comb)
        np.testing.assert_array_equal(got[1], base[1])

    def test_additive_rejects_unknown_state(self):
        comb = np.linspace(1.0, 2.0, N_FREQ)
        with pytest.raises(ValueError, match="RFant"):
            power(
                _sources(),
                _receiver(),
                STATE,
                TIMES,
                additive={"RFant": comb},
            )

    def test_per_time_source_equals_broadcast(self):
        src = _sources()
        tiled = {
            k: Source(
                np.repeat(s.gamma_s, N_TIME, axis=0),
                np.repeat(s.t_s_k, N_TIME, axis=0),
            )
            for k, s in src.items()
        }
        np.testing.assert_array_equal(
            power(src, _receiver(), STATE, TIMES),
            power(tiled, _receiver(), STATE, TIMES),
        )

    def test_time_varying_inputs_match_per_sample_calls(self):
        """A routing bug like arr[:rows.size] would still pass the other
        power tests, since they use n_time=1 or tiled/constant rows."""
        rng = np.random.default_rng(7)
        n_time, n_freq = 6, 5
        state = ["RFANT", "RFNON", "RFAMB", "RFANT", "RFNON", "RFANT"]
        times = np.arange(n_time, dtype=float)

        def draw_complex(shape, max_mag):
            mag = max_mag * rng.uniform(size=shape)
            phase = 2 * np.pi * rng.uniform(size=shape)
            return mag * np.exp(1j * phase)

        gamma_rec = draw_complex((n_time, n_freq), 0.3)
        gain = rng.uniform(0.5, 2.0, (n_time, n_freq))
        t_unc_k = rng.uniform(0.0, 100.0, (n_time, n_freq))
        t_cos_k = rng.uniform(-50.0, 50.0, (n_time, n_freq))
        t_sin_k = rng.uniform(-50.0, 50.0, (n_time, n_freq))
        t0_k = rng.uniform(0.0, 100.0, (n_time, n_freq))
        path_gain_ratio = {"RFNON": rng.uniform(0.8, 1.2, n_freq)}

        receiver = ReceiverModel(
            gamma_rec=gamma_rec,
            gain=gain,
            t_unc_k=t_unc_k,
            t_cos_k=t_cos_k,
            t_sin_k=t_sin_k,
            t0_k=t0_k,
            path_gain_ratio=path_gain_ratio,
        )

        rfant_gamma = draw_complex((n_time, n_freq), 0.5)
        rfant_t = rng.uniform(0.0, 1000.0, (n_time, n_freq))
        rfnon_gamma = draw_complex((n_time, n_freq), 0.5)
        rfnon_t = rng.uniform(0.0, 1000.0, (n_time, n_freq))
        rfamb_gamma = draw_complex((1, n_freq), 0.5)
        rfamb_t = rng.uniform(0.0, 1000.0, (1, n_freq))
        additive_rfant = rng.uniform(-1.0, 1.0, (n_time, n_freq))

        sources = {
            "RFANT": Source(rfant_gamma, rfant_t),
            "RFNON": Source(rfnon_gamma, rfnon_t),
            "RFAMB": Source(rfamb_gamma, rfamb_t),
        }
        vectorised = power(
            sources,
            receiver,
            state,
            times,
            additive={"RFANT": additive_rfant},
        )

        looped = []
        for i in range(n_time):
            row_receiver = ReceiverModel(
                gamma_rec=gamma_rec[i],
                gain=gain[i : i + 1],
                t_unc_k=t_unc_k[i],
                t_cos_k=t_cos_k[i],
                t_sin_k=t_sin_k[i],
                t0_k=t0_k[i],
                path_gain_ratio=path_gain_ratio,
            )
            row_sources = {
                "RFANT": Source(rfant_gamma[i : i + 1], rfant_t[i : i + 1]),
                "RFNON": Source(rfnon_gamma[i : i + 1], rfnon_t[i : i + 1]),
                "RFAMB": Source(rfamb_gamma, rfamb_t),
            }
            row_power = power(
                row_sources,
                row_receiver,
                state[i : i + 1],
                times[i : i + 1],
                additive={"RFANT": additive_rfant[i]},
            )
            looped.append(row_power[0])

        np.testing.assert_allclose(
            vectorised, np.stack(looped), rtol=1e-13, atol=0
        )

    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="RFNON"):
            power(_sources(), _receiver(), ["RFANT", "RFNON", "RFANT"], TIMES)

    def test_sample_count_checked(self):
        with pytest.raises(ValueError, match="n_time"):
            power(_sources(), _receiver(), STATE[:2], TIMES[:2])

    def test_source_shapes_checked(self):
        with pytest.raises(ValueError, match="share one shape"):
            Source(np.zeros((1, N_FREQ)), np.zeros((1, N_FREQ + 1)))

    @pytest.mark.parametrize("bad", [1.0 + 1e-9, -1.2, 0.8 + 0.8j, np.nan])
    def test_source_reflection_must_be_passive(self, bad):
        gamma = np.zeros((1, N_FREQ), dtype=complex)
        gamma[0, 1] = bad
        with pytest.raises(ValueError, match="gamma_s"):
            Source(gamma, np.full((1, N_FREQ), 300.0))

    def test_lossless_source_accepted(self):
        """|Gamma_s| = 1 (ideal short or open) is finite in eq. Ps, and
        M_s = 0 there, so T_s drops out. Roundoff just above 1 passes."""
        gamma = np.array([[-1.0, 1j, np.nextafter(1.0, 2.0), np.exp(0.3j)]])
        receiver = _receiver(gamma_rec=np.full(N_FREQ, 0.2 - 0.1j))
        got = []
        for t_s in (800.0, 0.0):
            sources = _sources()
            sources["RFANT"] = Source(gamma, np.full((1, N_FREQ), t_s))
            got.append(power(sources, receiver, STATE, TIMES))
        hot, cold = got
        assert np.all(np.isfinite(hot))
        np.testing.assert_allclose(hot, cold, rtol=1e-12)


class TestRadiometerNoise:
    def _draw(self, n_int, seed=0, n_time=20000):
        p = np.full((n_time, 2), 100.0)
        tau = np.full(n_time, 0.5)
        n = np.full(n_time, n_int)
        return radiometer_noise(
            p, D5_ENBW_HZ, tau, n, np.random.default_rng(seed)
        )

    def test_std_matches_radiometer_equation(self):
        noise = self._draw(n_int=4)
        sigma = 100.0 / np.sqrt(D5_ENBW_HZ * 0.5 * 4)
        np.testing.assert_allclose(noise.std(axis=0), sigma, rtol=0.02)
        np.testing.assert_allclose(
            noise.mean(axis=0), 0.0, atol=5 * sigma / 141
        )

    def test_averaging_reduces_noise(self):
        ratio = self._draw(n_int=1).std() / self._draw(n_int=16).std()
        np.testing.assert_allclose(ratio, 4.0, rtol=0.03)

    def test_reproducible(self):
        np.testing.assert_array_equal(
            self._draw(n_int=2, seed=7), self._draw(n_int=2, seed=7)
        )

    @pytest.mark.parametrize(
        "tau, n_int",
        [
            ([0.5, 0.0], [1, 1]),
            ([0.5, 0.5], [1, 0]),
            ([0.5], [1, 1]),
            ([0.5, np.nan], [1, 1]),
        ],
    )
    def test_invalid_inputs(self, tau, n_int):
        with pytest.raises(ValueError):
            radiometer_noise(
                np.ones((2, 3)),
                D5_ENBW_HZ,
                tau,
                n_int,
                np.random.default_rng(0),
            )

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_power_must_be_finite_and_non_negative(self, bad):
        """Negative power raised numpy's "scale < 0"; NaN returned NaN."""
        p = np.ones((2, 3))
        p[1, 2] = bad
        with pytest.raises(ValueError, match="power"):
            radiometer_noise(
                p, D5_ENBW_HZ, [0.5, 0.5], [1, 1], np.random.default_rng(0)
            )

    def test_n_int_must_be_integer(self):
        with pytest.raises(ValueError, match="n_int"):
            radiometer_noise(
                np.ones((2, 3)),
                D5_ENBW_HZ,
                [0.5, 0.5],
                [1.5, 1.0],
                np.random.default_rng(0),
            )
