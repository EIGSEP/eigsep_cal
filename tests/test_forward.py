import numpy as np
import pytest

from eigsep_cal.forward import Source, power
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

    def test_missing_source_raises(self):
        with pytest.raises(ValueError, match="RFNON"):
            power(_sources(), _receiver(), ["RFANT", "RFNON", "RFANT"], TIMES)

    def test_sample_count_checked(self):
        with pytest.raises(ValueError, match="n_time"):
            power(_sources(), _receiver(), STATE[:2], TIMES[:2])

    def test_source_shapes_checked(self):
        with pytest.raises(ValueError, match="share one shape"):
            Source(np.zeros((1, N_FREQ)), np.zeros((1, N_FREQ + 1)))
