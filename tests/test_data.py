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

    def test_grid_shape_checked(self):
        with pytest.raises(ValueError, match="gamma"):
            reflection(gamma=np.zeros((2, FREQS.size + 1)))

    def test_sys_modes_default_none(self):
        assert reflection().gamma_sys_modes is None

    def test_sys_modes_valid_and_readonly(self):
        modes = np.full((2, 3, FREQS.size, 2), 1e-4)
        refl = reflection(gamma_sys_modes=modes)
        assert refl.gamma_sys_modes.dtype == np.float64
        assert refl.gamma_sys_modes.shape == (2, 3, FREQS.size, 2)
        with pytest.raises(ValueError):
            refl.gamma_sys_modes[0, 0, 0, 0] = 0.0

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 3, FREQS.size, 2),
            (2, 3, FREQS.size + 1, 2),
            (2, 3, FREQS.size, 3),
            (2, 0, FREQS.size, 2),
            (2, FREQS.size, 2),
        ],
    )
    def test_sys_modes_shape_checked(self, shape):
        with pytest.raises(ValueError, match="gamma_sys_modes"):
            reflection(gamma_sys_modes=np.zeros(shape))

    def test_sys_modes_must_be_finite(self):
        modes = np.full((2, 3, FREQS.size, 2), 1e-4)
        modes[0, 0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="gamma_sys_modes"):
            reflection(gamma_sys_modes=modes)


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
