import dataclasses

import numpy as np
import pytest

from eigsep_cal.conventions import d5_freqs_mhz
from eigsep_cal.data import Observation, Reflection, SkyTemperature

CHAN = np.array([192, 208, 528])
FREQS = d5_freqs_mhz(CHAN)
N_TIME = 4


def observation(**overrides):
    kwargs = dict(
        power=np.ones((N_TIME, FREQS.size)),
        state=["RFANT", "RFANT", "RFNON", "RFAMB"],
        times_unix=1.7843e9 + 60.0 * np.arange(N_TIME),
        tau_s=np.full(N_TIME, 0.537),
        n_int=np.full(N_TIME, 100),
        flags=np.zeros((N_TIME, FREQS.size), dtype=bool),
        freqs_mhz=FREQS,
        antenna="box-air",
        chan=CHAN,
        covariates={"t_amb_k": [296.0, 296.1, np.nan, 296.3]},
        changepoints_unix=[1.7843e9 + 90.0],
        provenance={"files": ["a.h5"], "selection": {"night": "16/17"}},
    )
    kwargs.update(overrides)
    return Observation(**kwargs)


def reflection(**overrides):
    cov = np.zeros((2, FREQS.size, 2, 2))
    cov[..., 0, 0] = cov[..., 1, 1] = 1e-6
    kwargs = dict(
        gamma=np.full((2, FREQS.size), 0.1 + 0.2j),
        gamma_cov=cov,
        times_unix=[1.7843e9, 1.7843e9 + 3600.0],
        source="RFANT",
        freqs_mhz=FREQS,
    )
    kwargs.update(overrides)
    return Reflection(**kwargs)


class TestSkyTemperature:
    def _make(self, **overrides):
        kwargs = dict(
            t_ant_k=np.full((2, FREQS.size), 3000.0),
            freqs_mhz=FREQS,
            times_unix=[1.7843e9, 1.7843e9 + 60.0],
            antenna="box-air",
            elevation_deg=[0.0, 5.0],
            azimuth_deg=[0.0, 0.0],
            meta={"eigsim": "0.1.0"},
        )
        kwargs.update(overrides)
        return SkyTemperature(**kwargs)

    def test_valid(self):
        sky = self._make()
        assert sky.t_ant_k.dtype == np.float64
        assert sky.meta["eigsim"] == "0.1.0"

    def test_antenna_checked(self):
        with pytest.raises(ValueError, match="antenna"):
            self._make(antenna="vivaldi")

    def test_series_length_checked(self):
        with pytest.raises(ValueError, match="elevation_deg"):
            self._make(elevation_deg=[0.0])


class TestReflection:
    def test_valid_and_readonly(self):
        refl = reflection()
        assert refl.gamma.dtype == np.complex128
        with pytest.raises(ValueError):
            refl.gamma[0, 0] = 0.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            refl.source = "RFAMB"

    def test_receiver_source_allowed(self):
        assert reflection(source="receiver").source == "receiver"

    def test_vna_path_rejected(self):
        with pytest.raises(ValueError, match="source"):
            reflection(source="VNAANT")

    def test_covariance_must_be_symmetric(self):
        cov = reflection().gamma_cov.copy()
        cov[0, 0, 0, 1] = 1e-7
        with pytest.raises(ValueError, match="symmetric"):
            reflection(gamma_cov=cov)

    def test_covariance_symmetric_to_roundoff_accepted(self):
        """Stage 2 builds gamma_cov as J C J^T, which is symmetric only
        to roundoff; the nextafter step makes that asymmetry certain."""
        rng = np.random.default_rng(5)
        jac = rng.normal(size=(2, FREQS.size, 2, 6))
        root = rng.normal(size=(6, 6))
        cov = jac @ (1e-6 * root @ root.T) @ np.swapaxes(jac, -1, -2)
        cov[0, 0, 1, 0] = np.nextafter(cov[0, 0, 0, 1], np.inf)
        reflection(gamma_cov=cov)

    def test_covariance_asymmetry_beyond_roundoff_rejected(self):
        cov = reflection().gamma_cov.copy()
        cov[..., 0, 1] = cov[..., 1, 0] = 5e-7
        cov[0, 0, 1, 0] *= 1 + 1e-6
        with pytest.raises(ValueError, match="symmetric"):
            reflection(gamma_cov=cov)

    @pytest.mark.parametrize(
        "index, bad",
        [
            ((0, 0, 0, 1), np.nan),
            ((0, 0, 0, 0), np.nan),
            ((1, 2, 1, 1), np.inf),
        ],
    )
    def test_covariance_must_be_finite(self, index, bad):
        """NaN off the diagonal used to report "symmetric", and NaN or
        inf on it passed."""
        cov = reflection().gamma_cov.copy()
        cov[index] = bad
        with pytest.raises(ValueError, match="finite"):
            reflection(gamma_cov=cov)

    @pytest.mark.parametrize(
        "variance, off_diagonal",
        [(1e-12, 1.0), (1e-6, 1e-6 * (1 + 1e-6))],
    )
    def test_covariance_must_be_positive_semidefinite(
        self, variance, off_diagonal
    ):
        """Correlation 1e12, and correlation 1 + 1e-6 (beyond roundoff)."""
        cov = reflection().gamma_cov.copy()
        cov[0, 1, 0, 0] = cov[0, 1, 1, 1] = variance
        cov[0, 1, 0, 1] = cov[0, 1, 1, 0] = off_diagonal
        with pytest.raises(ValueError, match="positive semi-definite"):
            reflection(gamma_cov=cov)

    def test_rank_one_covariance_accepted(self):
        """sigma^2 u u^T is PSD, but roundoff leaves some determinants
        slightly negative."""
        rng = np.random.default_rng(1)
        u = rng.normal(size=(2, FREQS.size, 2))
        var = rng.uniform(1e-8, 1e-4, (2, FREQS.size))
        cov = var[..., None, None] * u[..., :, None] * u[..., None, :]
        det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] ** 2
        assert np.any(det < 0)
        reflection(gamma_cov=cov)

    def test_grid_shape_checked(self):
        with pytest.raises(ValueError, match="gamma"):
            reflection(gamma=np.zeros((2, FREQS.size + 1)))


class TestObservation:
    def test_valid(self):
        obs = observation()
        assert (obs.n_time, obs.n_freq) == (N_TIME, FREQS.size)
        assert np.isnan(obs.covariates["t_amb_k"][2])
        with pytest.raises(TypeError):
            obs.covariates["t_switch_k"] = np.zeros(N_TIME)

    def test_flags_must_be_bool(self):
        with pytest.raises(TypeError, match="flags"):
            observation(flags=np.zeros((N_TIME, FREQS.size), dtype=int))

    def test_unknown_state(self):
        with pytest.raises(ValueError, match="MISSING"):
            observation(state=["RFANT", "MISSING", "RFNON", "RFAMB"])

    def test_chan_must_match_frequencies(self):
        with pytest.raises(ValueError, match="chan"):
            observation(chan=CHAN + 1)

    def test_chan_optional(self):
        assert observation(chan=None).chan is None

    @pytest.mark.parametrize(
        "field, value",
        [
            ("tau_s", np.zeros(N_TIME)),
            ("tau_s", np.array([0.5, np.nan, 0.5, 0.5])),
            ("n_int", np.full(N_TIME, 1.5)),
            ("covariates", {"t_amb_k": [296.0]}),
            ("changepoints_unix", [2.0, 1.0]),
            ("changepoints_unix", [1.0, np.nan]),
        ],
    )
    def test_invalid_fields(self, field, value):
        with pytest.raises(ValueError):
            observation(**{field: value})
