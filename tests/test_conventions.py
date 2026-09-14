import numpy as np
import pytest

from eigsep_cal import conventions


def test_d5_channel_grid():
    got = conventions.d5_freqs_mhz([0, 192, 1008, 1023])
    np.testing.assert_array_equal(got, [0.0, 46.875, 246.09375, 249.755859375])


def test_default_enbw_is_one_channel():
    assert conventions.D5_ENBW_HZ == 244140.625


@pytest.mark.parametrize("chan", [[1024], [-1], [1.5]])
def test_d5_channel_range_and_type(chan):
    with pytest.raises(ValueError):
        conventions.d5_freqs_mhz(chan)


def test_states_match_spec():
    assert conventions.STATES == (
        "RFANT",
        "RFAMB",
        "RFNON",
        "RFNOFF",
        "RFSP1_SHORT",
        "RFSP1_OPEN",
    )
